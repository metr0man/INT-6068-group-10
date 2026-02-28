#!/usr/bin/env python3
"""大规模单无人机环境 - 每个环境一架无人机，通过大量并行环境提高GPU利用率"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scene_creation import DroneScene
from reward_system import RewardSystem
import pybullet as p
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import List, Optional, Tuple, Union
import multiprocessing as mp

class SingleDronePathPlanningEnv(gym.Env):
    """单无人机路径规划RL环境 - 专为大规模并行训练设计"""
    
    def __init__(self, ray_length=1000.0, env_id=0):
        super().__init__()
        self.env_id = env_id  # 环境ID，用于区分不同环境
        
        # 创建场景
        self.scene = DroneScene(gui=(self.env_id==0))  # 只让第0号环境开GUI
        self.scene.create_obstacles()
        # 为大规模训练创建更多目标点
        self.scene.create_targets(num_targets=128, area=8000)
        
        # 创建单个无人机
        self.drone_id = self.scene.create_drone()
        
        # 动作空间：3个动作 [dx, dy, dz]，范围[-1, 1]米
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # 状态空间：35维状态
        # 3(位置) + 3(目标) + 26(射线距离) + 3(速度)
        self.observation_space = spaces.Box(
            low=-10000, high=10000, 
            shape=(35,), 
            dtype=np.float32
        )
        
        self.max_steps = 50000  # 每回合最大步数
        self.current_step = 0
        self.max_energy = 15000.0
        self.energy = self.max_energy
        self.reward_system = RewardSystem()
        self.previous_position = None
        self.ray_length = ray_length
        
        # 定义26个方向的射线
        self.ray_directions = self._generate_ray_directions()
        
        # 为每个环境分配不同的目标点
        self.target = self._assign_target_by_env_id()
        
        # 记录统计信息
        self.stats = {
            'episodes': 0,
            'total_reward': 0.0,  # 改为float
            'targets_reached': 0,
            'collisions': 0,
            'avg_episode_length': 0.0  # 改为float
        }
        self.episode_reward = 0  # 新增：累计本回合奖励

    def _generate_ray_directions(self):
        """生成26个方向的射线向量"""
        directions = []
        
        # 6个主要方向：±X, ±Y, ±Z
        main_dirs = [
            [1, 0, 0], [-1, 0, 0],  # +X, -X
            [0, 1, 0], [0, -1, 0],  # +Y, -Y  
            [0, 0, 1], [0, 0, -1]   # +Z, -Z
        ]
        directions.extend(main_dirs)
        
        # 12个边方向：每个坐标轴的正负组合
        edge_dirs = [
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],  # XY平面
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],  # XZ平面
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]   # YZ平面
        ]
        directions.extend(edge_dirs)
        
        # 8个角方向：三个坐标轴的正负组合
        corner_dirs = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]
        directions.extend(corner_dirs)
        
        # 归一化所有方向向量
        normalized_dirs = []
        for direction in directions:
            norm = np.linalg.norm(direction)
            normalized_dirs.append([d/norm for d in direction])
        
        return normalized_dirs

    def _assign_target_by_env_id(self):
        """根据环境ID分配目标点，确保不同环境有不同的目标"""
        if not self.scene.targets:
            return [0, 0, 2.5]  # 默认目标
        
        # 使用环境ID来选择目标点
        target_index = self.env_id % len(self.scene.targets)
        return self.scene.targets[target_index]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置场景
        self.scene.reset_scene(drone_ids=[self.drone_id])
        
        # 重置状态
        self.current_step = 0
        self.energy = self.max_energy
        self.reward_system.reset_episode()
        self.previous_position = None
        
        # 重新分配目标点（增加随机性）
        self.target = self._assign_target_by_env_id()
        
        # 更新统计
        self.stats['episodes'] += 1
        self.episode_reward = 0  # 新增：回合开始时清零
        
        obs = self._get_obs()
        info = {
            'env_id': self.env_id,
            'target': self.target,
            'energy': self.energy
        }
        return obs, info

    def step(self, action):
        # 限制动作幅度
        action = np.clip(action, -1, 1)
        
        # 获取当前位置
        pos = np.array(self.scene.get_drone_position(self.drone_id))
        
        # 计算新位置
        new_pos = pos + action
        # 限制最大飞行高度
        new_pos[2] = np.clip(new_pos[2], 0.5, 100)
        move_dist = np.linalg.norm(action)
        
        # 消耗能量
        self.energy -= move_dist
        
        # 检查与障碍物的碰撞
        collision_occurred = False
        if self.scene.check_collision(new_pos.tolist()):
            self.energy = 0  # 碰撞障碍物，失去电量
            collision_occurred = True
            reward = -500.0  # 碰撞惩罚
            reached_target = False  # 修复：保证后续引用不会报错
        else:
            # 移动无人机
            self.scene.move_drone_to(new_pos.tolist(), self.drone_id)
            
            # 计算奖励
            reward, reached_target = self.reward_system.get_step_reward(
                self.scene, new_pos.tolist(), [self.target], 
                previous_pos=self.previous_position
            )
            
            # 更新统计
            if reached_target:
                self.stats['targets_reached'] += 1
        
        # 更新统计
        if collision_occurred:
            self.stats['collisions'] += 1
        
        self.stats['total_reward'] += reward
        self.episode_reward += reward  # 新增：累计本回合奖励
        self.previous_position = pos.tolist()
        
        self.current_step += 1
        
        # 检查终止条件
        terminated = False
        truncated = False
        reached_target_flag = reached_target is not None and reached_target is not False
        if self.energy <= 0:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True
        # 新增：到达目标点立即终止
        if reached_target_flag:
            terminated = True
        # 更新平均回合长度和打印到达目标点信息
        if terminated or truncated:
            self.stats['avg_episode_length'] = (
                (self.stats['avg_episode_length'] * (self.stats['episodes'] - 1) + self.current_step) 
                / self.stats['episodes']
            )
            if reached_target_flag and terminated:
                print(f"[环境 {self.env_id}] 到达目标点，回合结束，用时: {self.current_step} 步，总奖励: {self.episode_reward:.2f}，目标点: {self.target}")
            else:
                print(f"[环境 {self.env_id}] 回合结束，总奖励: {self.episode_reward:.2f}，步数: {self.current_step}，目标点: {self.target}")  # 中文输出
        
        info = {
            'env_id': self.env_id,
            'energy': self.energy,
            'steps_taken': self.current_step,
            'collision_occurred': collision_occurred,
            'target_reached': reached_target if 'reached_target' in locals() else False,
            'stats': self.stats.copy()
        }
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """获取观察状态"""
        # 获取无人机位置
        pos = self.scene.get_drone_position(self.drone_id)
        if pos is None:
            pos = [0, 0, 2.5]
        
        # 获取速度（简化：使用位置变化）
        velocity = [0, 0, 0]
        if self.previous_position is not None:
            velocity = [
                pos[0] - self.previous_position[0],
                pos[1] - self.previous_position[1], 
                pos[2] - self.previous_position[2]
            ]
        
        # 射线检测
        ray_distances = self._ray_cast_all_directions(pos)
        
        # 构建观察向量
        obs = np.concatenate([
            pos,                    # 3维：位置
            self.target,            # 3维：目标位置
            ray_distances,          # 26维：射线距离
            velocity               # 3维：速度
        ], dtype=np.float32)
        
        return obs

    def _ray_cast_all_directions(self, drone_pos):
        """在所有26个方向进行射线检测"""
        distances = []
        
        for direction in self.ray_directions:
            # 计算射线终点
            end_pos = [
                drone_pos[0] + direction[0] * self.ray_length,
                drone_pos[1] + direction[1] * self.ray_length,
                drone_pos[2] + direction[2] * self.ray_length
            ]
            
            # 进行射线检测
            result = p.rayTest(drone_pos, end_pos)[0]
            
            if result[0] == -1:  # 没有碰撞
                distance = self.ray_length
            else:
                # 计算碰撞距离
                hit_pos = result[3]
                distance = np.linalg.norm(np.array(hit_pos) - np.array(drone_pos))
            
            distances.append(distance)
        
        return np.array(distances, dtype=np.float32)

    def render(self):
        """渲染环境（在无头模式下不执行）"""
        pass

def make_single_drone_env(env_id):
    """创建单个无人机环境的工厂函数"""
    def _init():
        return SingleDronePathPlanningEnv(env_id=env_id)
    return _init

def create_massive_single_drone_envs(num_envs=32, num_cpu=8):
    """创建大规模单无人机环境"""
    print(f"🚀 创建 {num_envs} 个单无人机环境...")
    print(f"💻 使用 {num_cpu} 个CPU进程")
    
    # 创建环境列表
    env_fns = [make_single_drone_env(i) for i in range(num_envs)]
    
    # 创建向量化环境 - SubprocVecEnv会自动使用多进程
    env = SubprocVecEnv(env_fns, start_method='fork')
    
    print(f"✅ 成功创建 {num_envs} 个并行环境")
    print(f"📊 观察空间维度: {env.observation_space.shape}")
    print(f"🎯 动作空间维度: {env.action_space.shape}")
    
    return env

if __name__ == "__main__":
    # 测试单个环境
    print("🧪 测试单个无人机环境...")
    env = SingleDronePathPlanningEnv(env_id=0)
    
    # 测试重置
    obs, info = env.reset()
    print(f"✅ 环境重置成功，观察维度: {obs.shape}")
    print(f"🎯 目标位置: {info['target']}")
    
    # 测试动作
    action = np.random.uniform(-1, 1, 3)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ 动作执行成功，奖励: {reward:.2f}")
    
    print("🎉 单个环境测试完成！")
    
    # 测试大规模环境
    print("\n🚀 测试大规模并行环境...")
    try:
        massive_env = create_massive_single_drone_envs(num_envs=8, num_cpu=4)
        
        # 测试批量重置
        obs = massive_env.reset()
        print(f"✅ 批量重置成功，观察形状: {obs.shape}")
        
        # 测试批量动作
        actions = np.random.uniform(-1, 1, (8, 3))
        # 修复：SubprocVecEnv.step 返回4个值（obs, rewards, dones, infos）
        obs, rewards, dones, infos = massive_env.step(actions)
        print(f"✅ 批量动作执行成功，奖励形状: {rewards.shape}")
        
        print("🎉 大规模环境测试完成！")
        
    except Exception as e:
        print(f"❌ 大规模环境测试失败: {e}")
        import traceback
        traceback.print_exc() 