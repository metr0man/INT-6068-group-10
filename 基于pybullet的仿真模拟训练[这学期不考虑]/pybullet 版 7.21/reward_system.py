import numpy as np
from typing import Dict, Any

class RewardSystem:
    def __init__(self):
        # 奖励权重
        self.weights = {
            'target_reached': 500.0,    # 到达目标点奖励（增大）
            'collision_penalty': -25.0,  # 碰撞惩罚（保持不变）
            'distance_reward': -0.5,     # 距离奖励（增加权重，负值表示距离越近奖励越高）
            'efficiency_reward': 1.0,    # 效率奖励
            'exploration_reward': 5.0,   # 探索奖励（减少，避免过度探索）
            'safety_reward': 1.0,        # 安全奖励（减少，避免过度保守）
            'time_penalty': -0.05,       # 时间惩罚（减少，给更多时间）
            'energy_penalty': -0.02      # 能耗惩罚（减少，避免过度节能）
        }
        
        # 状态记录
        self.episode_data = {
            'total_reward': 0,
            'targets_reached': 0,
            'collisions': 0,
            'total_distance': 0,
            'total_time': 0,
            'energy_consumed': 0,
            'visited_positions': set()
        }
        
    def reset_episode(self):
        """重置回合数据"""
        self.episode_data = {
            'total_reward': 0,
            'targets_reached': 0,
            'collisions': 0,
            'total_distance': 0,
            'total_time': 0,
            'energy_consumed': 0,
            'visited_positions': set()
        }
    
    def calculate_target_reward(self, drone_pos, targets, threshold=1.0):
        """计算目标达成奖励"""
        reward = 0
        reached_target = None
        
        for i, target in enumerate(targets):
            # 检查目标是否已经被访问过
            # NOTE: 这里假设 target 可以挂载属性 visited（target.visited）。
            # 但在当前项目的其他代码里 targets 往往是 [x, y, z] 形式的 list/ndarray，
            # 这类对象不能像自定义类一样随意新增属性，执行 target.visited = True 会直接报错。
            # 如果要记录“目标是否到达”，更稳妥的方式是：
            # - 用 visited_target_indices: set[int] 记录已达成的目标索引；或
            # - 把 target 统一封装成 dict/对象（包含 position/visited 字段）。
            if not hasattr(target, 'visited') or not target.visited:
                target_pos = target  # targets 是 [x, y, z] 列表
                distance = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
                
                if distance < threshold:
                    reward += self.weights['target_reached']
                    # 标记目标为已访问
                    target.visited = True
                    reached_target = target
                    self.episode_data['targets_reached'] += 1
        return reward, reached_target
    
    def calculate_collision_penalty(self, scene, drone_pos):
        """计算碰撞惩罚"""
        if scene.check_collision(drone_pos):
            self.episode_data['collisions'] += 1
            return self.weights['collision_penalty']
        return 0.0
    
    def calculate_distance_reward(self, current_pos, target_pos, previous_pos=None):
        """距离奖励（距离越近奖励越高，避免来回移动刷分）"""
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if previous_pos is not None:
            # 计算距离变化
            previous_distance = np.linalg.norm(np.array(previous_pos) - np.array(target_pos))
            distance_change = previous_distance - distance
            # 如果距离减少，给予奖励；如果距离增加，给予惩罚
            if distance_change > 0:
                # 增加向目标移动的奖励
                return self.weights['distance_reward'] * distance_change * 2.0  # 加倍奖励
            else:
                # 增加远离目标的惩罚
                return self.weights['distance_reward'] * distance_change * 1.5  # 增加惩罚
        else:
            # 如果没有前一个位置，使用原来的距离奖励
            return self.weights['distance_reward'] * distance
    
    def calculate_efficiency_reward(self, path_length, straight_line_distance):
        """计算效率奖励（路径越短越好）"""
        if straight_line_distance > 0:
            efficiency = straight_line_distance / path_length
            return self.weights['efficiency_reward'] * efficiency
        return 0.0
    
    def calculate_exploration_reward(self, current_pos, visited_positions):
        """计算探索奖励（鼓励访问新位置）"""
        pos_key = tuple(np.round(current_pos, 1))  # 四舍五入到0.1精度
        
        if pos_key not in visited_positions:
            visited_positions.add(pos_key)
            return self.weights['exploration_reward']
        return 0.0
    
    def calculate_safety_reward(self, drone_pos, obstacles, safe_distance=2.0):
        """计算安全奖励（远离障碍物）"""
        min_distance = float('inf')
        
        for obstacle in obstacles:
            obs_pos = obstacle['position']
            distance = np.linalg.norm(np.array(drone_pos) - np.array(obs_pos))
            min_distance = min(min_distance, distance)
        
        if min_distance < safe_distance:
            # 距离障碍物越近，安全奖励越低
            safety_factor = min_distance / safe_distance
            return self.weights['safety_reward'] * safety_factor
        else:
            return self.weights['safety_reward']
    
    def calculate_energy_penalty(self, velocity, acceleration):
        """计算能耗惩罚"""
        # 简化的能耗模型：与速度和加速度的平方成正比
        energy = np.linalg.norm(velocity) ** 2 + np.linalg.norm(acceleration) ** 2
        self.episode_data['energy_consumed'] += energy
        return self.weights['energy_penalty'] * energy
    
    def calculate_time_penalty(self):
        """计算时间惩罚"""
        self.episode_data['total_time'] += 1
        return self.weights['time_penalty']
    
    def get_step_reward(self, scene, drone_pos, targets, velocity=None, acceleration=None, previous_pos=None):
        """计算单步奖励"""
        total_reward = 0.0
        
        # 1. 目标达成奖励
        target_reward, reached_target = self.calculate_target_reward(drone_pos, targets)
        total_reward += target_reward
        
        # 2. 碰撞惩罚
        collision_penalty = self.calculate_collision_penalty(scene, drone_pos)
        total_reward += collision_penalty
        
        # 3. 距离奖励（改进版，避免来回移动刷分）
        if len(targets) > 0 and (not hasattr(targets[0], 'visited') or not targets[0].visited):
            target_pos = targets[0]  # targets[0] 是 [x, y, z] 列表
            distance_reward = self.calculate_distance_reward(drone_pos, target_pos, previous_pos)
            total_reward += distance_reward
        
        # 4. 探索奖励
        exploration_reward = self.calculate_exploration_reward(
            drone_pos, self.episode_data['visited_positions']
        )
        total_reward += exploration_reward
        
        # 5. 安全奖励
        safety_reward = self.calculate_safety_reward(drone_pos, scene.obstacles)
        total_reward += safety_reward
        
        # 6. 时间惩罚
        time_penalty = self.calculate_time_penalty()
        total_reward += time_penalty
        
        # 7. 能耗惩罚（如果有速度和加速度信息）
        if velocity is not None and acceleration is not None:
            energy_penalty = self.calculate_energy_penalty(velocity, acceleration)
            total_reward += energy_penalty
        
        # 更新总奖励
        self.episode_data['total_reward'] += total_reward
        
        return total_reward, reached_target
    
    def get_episode_reward(self):
        """获取回合总奖励"""
        return self.episode_data['total_reward']
    
    def get_episode_stats(self):
        """获取回合统计信息"""
        return {
            'total_reward': self.episode_data['total_reward'],
            'targets_reached': self.episode_data['targets_reached'],
            'collisions': self.episode_data['collisions'],
            'total_time': self.episode_data['total_time'],
            'energy_consumed': self.episode_data['energy_consumed'],
            'visited_positions_count': len(self.episode_data['visited_positions'])
        }
    
    def is_episode_done(self, targets, max_steps=1000, max_collisions=5):
        """判断回合是否结束"""
        # 所有目标都到达
        all_targets_reached = all(hasattr(target, 'visited') and target.visited for target in targets)
        
        # 超过最大步数
        time_limit_reached = self.episode_data['total_time'] >= max_steps
        
        # 碰撞次数过多
        too_many_collisions = self.episode_data['collisions'] >= max_collisions
        
        return all_targets_reached or time_limit_reached or too_many_collisions
    
    def get_success_rate(self, targets):
        """计算成功率"""
        total_targets = len(targets)
        reached_targets = sum(1 for target in targets if hasattr(target, 'visited') and target.visited)
        return reached_targets / total_targets if total_targets > 0 else 0.0
    
    def print_episode_summary(self, targets):
        """打印回合总结"""
        stats = self.get_episode_stats()
        success_rate = self.get_success_rate(targets)
        
        print("=== 回合总结 ===")
        print(f"总奖励: {stats['total_reward']:.2f}")
        print(f"到达目标: {stats['targets_reached']}/{len(targets)}")
        print(f"成功率: {success_rate:.2%}")
        print(f"碰撞次数: {stats['collisions']}")
        print(f"总步数: {stats['total_time']}")
        print(f"能耗: {stats['energy_consumed']:.2f}")
        print(f"访问位置数: {stats['visited_positions_count']}")
        print("================")

if __name__ == "__main__":
    # 测试奖惩系统
    reward_system = RewardSystem()
    
    # 模拟场景数据
    class MockScene:
        def __init__(self):
            self.obstacles = [
                {'position': [2, 2, 2], 'size': 1.0},
                {'position': [5, 5, 3], 'size': 1.5}
            ]
        
        def check_collision(self, pos):
            for obs in self.obstacles:
                distance = np.linalg.norm(np.array(pos) - np.array(obs['position']))
                if distance < obs['size']:
                    return True
            return False
    
    scene = MockScene()
    targets = [
        {'position': [3, 3, 3], 'visited': False},
        {'position': [7, 7, 4], 'visited': False}
    ]
    
    # 测试单步奖励
    drone_pos = [1, 1, 1]
    reward, reached = reward_system.get_step_reward(scene, drone_pos, targets)
    print(f"单步奖励: {reward:.2f}")
    
    # 测试多个步骤
    for i in range(10):
        drone_pos = [i, i, 1]
        reward, reached = reward_system.get_step_reward(scene, drone_pos, targets)
        print(f"步骤 {i+1}: 位置 {drone_pos}, 奖励 {reward:.2f}")
    
    # 打印总结
    reward_system.print_episode_summary(targets) 