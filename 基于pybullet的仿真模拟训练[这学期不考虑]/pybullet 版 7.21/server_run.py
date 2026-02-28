import pybullet as p
import numpy as np
import time
import json
from datetime import datetime
from scene_creation import DroneScene
from path_planning import PathPlanning
from reward_system import RewardSystem

class ServerDroneSimulation:
    def __init__(self):
        # 服务器模式：无GUI
        self.scene = DroneScene(gui=False)
        self.planner = PathPlanning(self.scene)
        self.reward_system = RewardSystem()
        
        # 仿真参数
        self.max_steps = 1000
        self.results = []
        
    def setup_scene(self):
        """设置场景"""
        print("正在创建场景...")
        self.scene.create_obstacles()
        self.scene.create_targets()
        self.scene.create_drone()
        print(f"场景创建完成！障碍物: {len(self.scene.obstacles)}, 目标点: {len(self.scene.targets)}")
    
    def run_single_episode(self, episode_id):
        """运行单个回合"""
        print(f"开始回合 {episode_id}...")
        
        # 重置场景和奖惩系统
        self.scene.reset_scene()
        self.reward_system.reset_episode()
        
        start_time = time.time()
        step_count = 0
        
        # 获取所有目标点的路径
        start_pos = self.scene.get_drone_position()
        all_paths = self.planner.plan_to_targets(start_pos, self.scene.targets)
        
        if not all_paths:
            print(f"回合 {episode_id}: 无法找到有效路径！")
            return None
        
        # 执行路径
        for path_idx, path in enumerate(all_paths):
            for target_pos in path:
                if step_count >= self.max_steps:
                    break
                
                # 移动无人机到目标位置
                self.scene.move_drone_to(target_pos)
                
                # 获取当前状态
                current_pos = self.scene.get_drone_position()
                
                # 计算奖励
                reward, reached_target = self.reward_system.get_step_reward(
                    self.scene, current_pos, self.scene.targets
                )
                
                step_count += 1
                
                # 检查回合是否结束
                if self.reward_system.is_episode_done(self.scene.targets):
                    break
        
        end_time = time.time()
        episode_time = end_time - start_time
        
        # 收集结果
        stats = self.reward_system.get_episode_stats()
        success_rate = self.reward_system.get_success_rate(self.scene.targets)
        
        episode_result = {
            'episode_id': episode_id,
            'total_reward': stats['total_reward'],
            'targets_reached': stats['targets_reached'],
            'total_targets': len(self.scene.targets),
            'success_rate': success_rate,
            'collisions': stats['collisions'],
            'total_steps': stats['total_time'],
            'energy_consumed': stats['energy_consumed'],
            'visited_positions': stats['visited_positions_count'],
            'episode_time': episode_time,
            'paths_found': len(all_paths),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"回合 {episode_id} 完成:")
        print(f"  总奖励: {stats['total_reward']:.2f}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  运行时间: {episode_time:.2f}秒")
        
        return episode_result
    
    def run_batch_simulation(self, num_episodes=10):
        """运行批量仿真"""
        print(f"开始批量仿真，共 {num_episodes} 个回合...")
        
        for episode in range(num_episodes):
            result = self.run_single_episode(episode + 1)
            if result:
                self.results.append(result)
            
            # 每5个回合保存一次结果
            if (episode + 1) % 5 == 0:
                self.save_results()
        
        # 最终保存
        self.save_results()
        
        # 打印统计结果
        self.print_batch_summary()
    
    def save_results(self):
        """保存结果到文件"""
        filename = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {filename}")
    
    def print_batch_summary(self):
        """打印批量仿真总结"""
        if not self.results:
            print("没有可用的结果数据")
            return
        
        rewards = [r['total_reward'] for r in self.results]
        success_rates = [r['success_rate'] for r in self.results]
        collisions = [r['collisions'] for r in self.results]
        episode_times = [r['episode_time'] for r in self.results]
        
        print("\n" + "="*50)
        print("批量仿真总结")
        print("="*50)
        print(f"总回合数: {len(self.results)}")
        print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"平均成功率: {np.mean(success_rates):.2%} ± {np.std(success_rates):.2%}")
        print(f"平均碰撞次数: {np.mean(collisions):.2f} ± {np.std(collisions):.2f}")
        print(f"平均运行时间: {np.mean(episode_times):.2f} ± {np.std(episode_times):.2f}秒")
        print(f"最佳奖励: {max(rewards):.2f}")
        print(f"最佳成功率: {max(success_rates):.2%}")
        print("="*50)
    
    def test_algorithms(self):
        """测试路径规划算法性能"""
        print("测试路径规划算法...")
        
        test_cases = [
            ((0, 0, 1), (5, 5, 3)),
            ((0, 0, 1), (8, 8, 5)),
            ((0, 0, 1), (-5, -5, 4))
        ]
        
        algorithm_results = {
            'a_star': {'success': 0, 'total_time': 0, 'path_lengths': []},
            'rrt': {'success': 0, 'total_time': 0, 'path_lengths': []}
        }
        
        for start_pos, goal_pos in test_cases:
            print(f"\n测试路径: {start_pos} -> {goal_pos}")
            
            # 测试A*算法
            start_time = time.time()
            a_star_path = self.planner.a_star(start_pos, goal_pos)
            a_star_time = time.time() - start_time
            
            if a_star_path:
                algorithm_results['a_star']['success'] += 1
                algorithm_results['a_star']['total_time'] += a_star_time
                algorithm_results['a_star']['path_lengths'].append(len(a_star_path))
                print(f"A*: 成功, 时间: {a_star_time:.3f}s, 路径长度: {len(a_star_path)}")
            else:
                print("A*: 失败")
            
            # 测试RRT算法
            start_time = time.time()
            rrt_path = self.planner.rrt_planning(start_pos, goal_pos)
            rrt_time = time.time() - start_time
            
            if rrt_path:
                algorithm_results['rrt']['success'] += 1
                algorithm_results['rrt']['total_time'] += rrt_time
                algorithm_results['rrt']['path_lengths'].append(len(rrt_path))
                print(f"RRT: 成功, 时间: {rrt_time:.3f}s, 路径长度: {len(rrt_path)}")
            else:
                print("RRT: 失败")
        
        # 打印算法比较结果
        print("\n" + "="*50)
        print("算法性能比较")
        print("="*50)
        
        for algo, results in algorithm_results.items():
            if results['success'] > 0:
                avg_time = results['total_time'] / results['success']
                avg_length = np.mean(results['path_lengths'])
                success_rate = results['success'] / len(test_cases)
                print(f"{algo.upper()}:")
                print(f"  成功率: {success_rate:.2%}")
                print(f"  平均时间: {avg_time:.3f}s")
                print(f"  平均路径长度: {avg_length:.1f}")
            else:
                print(f"{algo.upper()}: 全部失败")
        
        print("="*50)

def main():
    print("服务器端无人机路径规划仿真系统")
    print("="*50)
    
    # 创建仿真对象
    simulation = ServerDroneSimulation()
    
    # 设置场景
    simulation.setup_scene()
    
    # 测试算法性能
    simulation.test_algorithms()
    
    # 运行批量仿真
    print("\n" + "="*50)
    simulation.run_batch_simulation(num_episodes=20)
    
    print("\n仿真完成！")

if __name__ == "__main__":
    main() 