import pybullet as p
import numpy as np
import time
from scene_creation import DroneScene
from path_planning import PathPlanning
from reward_system import RewardSystem

class DronePathPlanningSimulation:
    def __init__(self, gui=True):
        # 初始化各个组件
        self.scene = DroneScene(gui=gui)
        self.planner = PathPlanning(self.scene)
        self.reward_system = RewardSystem()
        
        # 仿真参数
        self.max_steps = 1000
        self.step_delay = 0.01
        
    def setup_scene(self):
        """设置场景"""
        print("正在创建场景...")
        self.scene.create_obstacles()
        self.scene.create_targets()
        self.scene.create_drone()
        print(f"场景创建完成！障碍物: {len(self.scene.obstacles)}, 目标点: {len(self.scene.targets)}")
    
    def run_simulation(self):
        """运行仿真"""
        print("开始无人机路径规划仿真...")
        
        # 重置奖惩系统
        self.reward_system.reset_episode()
        
        step_count = 0
        current_path = []
        path_index = 0
        
        # 获取所有目标点的路径
        start_pos = self.scene.get_drone_position()
        all_paths = self.planner.plan_to_targets(start_pos, self.scene.targets)
        
        if not all_paths:
            print("无法找到有效路径！")
            return
        
        print(f"规划了 {len(all_paths)} 条路径")
        
        # 执行路径
        for path_idx, path in enumerate(all_paths):
            print(f"执行路径 {path_idx + 1}/{len(all_paths)}")
            
            for i, target_pos in enumerate(path):
                if step_count >= self.max_steps:
                    print("达到最大步数限制")
                    break
                
                # 移动无人机到目标位置
                self.scene.move_drone_to(target_pos)
                
                # 获取当前状态
                current_pos = self.scene.get_drone_position()
                
                # 计算奖励
                reward, reached_target = self.reward_system.get_step_reward(
                    self.scene, current_pos, self.scene.targets
                )
                
                # 检查是否到达目标
                if reached_target:
                    print(f"到达目标点 {path_idx + 1}!")
                
                # 检查碰撞
                if self.scene.check_collision(current_pos):
                    print("发生碰撞！")
                
                # 渲染场景
                self.scene.render()
                time.sleep(self.step_delay)
                
                step_count += 1
                
                # 检查回合是否结束
                if self.reward_system.is_episode_done(self.scene.targets):
                    print("回合结束")
                    break
        
        # 打印仿真结果
        self.reward_system.print_episode_summary(self.scene.targets)
    
    def run_multiple_episodes(self, num_episodes=5):
        """运行多个回合"""
        print(f"运行 {num_episodes} 个回合的仿真...")
        
        episode_rewards = []
        success_rates = []
        
        for episode in range(num_episodes):
            print(f"\n=== 回合 {episode + 1}/{num_episodes} ===")
            
            # 重置场景
            self.scene.reset_scene()
            self.reward_system.reset_episode()
            
            # 运行仿真
            self.run_simulation()
            
            # 记录结果
            episode_rewards.append(self.reward_system.get_episode_reward())
            success_rates.append(self.reward_system.get_success_rate(self.scene.targets))
        
        # 打印统计结果
        print("\n=== 多回合统计 ===")
        print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"平均成功率: {np.mean(success_rates):.2%} ± {np.std(success_rates):.2%}")
        print(f"最佳奖励: {max(episode_rewards):.2f}")
        print(f"最佳成功率: {max(success_rates):.2%}")
    
    def test_path_planning_algorithms(self):
        """测试路径规划算法"""
        print("测试路径规划算法...")
        
        start_pos = (0, 0, 1)
        goal_pos = (5, 5, 3)
        
        # 测试A*算法
        print("\n1. 测试A*算法:")
        a_star_path = self.planner.a_star(start_pos, goal_pos)
        if a_star_path:
            print(f"A*路径长度: {len(a_star_path)}")
            print(f"A*路径: {a_star_path[:3]}...{a_star_path[-3:]}")
        else:
            print("A*算法未找到路径")
        
        # 测试RRT算法
        print("\n2. 测试RRT算法:")
        rrt_path = self.planner.rrt_planning(start_pos, goal_pos)
        if rrt_path:
            print(f"RRT路径长度: {len(rrt_path)}")
            print(f"RRT路径: {rrt_path[:3]}...{rrt_path[-3:]}")
        else:
            print("RRT算法未找到路径")
        
        # 路径平滑测试
        if a_star_path:
            print("\n3. 测试路径平滑:")
            smoothed_path = self.planner.smooth_path(a_star_path)
            print(f"原始路径长度: {len(a_star_path)}")
            print(f"平滑后路径长度: {len(smoothed_path)}")
            print(f"路径缩短: {((len(a_star_path) - len(smoothed_path)) / len(a_star_path) * 100):.1f}%")

def main():
    """主函数"""
    print("无人机路径规划仿真系统")
    print("=" * 50)
    
    # 创建仿真对象
    simulation = DronePathPlanningSimulation(gui=True)
    
    # 设置场景
    simulation.setup_scene()
    
    # 测试路径规划算法
    simulation.test_path_planning_algorithms()
    
    # 运行单次仿真
    print("\n" + "=" * 50)
    simulation.run_simulation()
    
    # 运行多回合仿真
    print("\n" + "=" * 50)
    simulation.run_multiple_episodes(num_episodes=3)

if __name__ == "__main__":
    main() 