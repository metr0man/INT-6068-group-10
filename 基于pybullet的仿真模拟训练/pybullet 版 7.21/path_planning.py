import numpy as np
import heapq
from typing import List, Tuple, Optional

class PathPlanning:
    def __init__(self, scene):
        self.scene = scene
        self.grid_size = 0.5  # 网格大小
        self.grid_bounds = {
            'x': (-10, 10),
            'y': (-10, 10),
            'z': (0, 8)
        }
        
    def world_to_grid(self, world_pos):
        """转换为网格坐标"""
        x = int((world_pos[0] - self.grid_bounds['x'][0]) / self.grid_size)
        y = int((world_pos[1] - self.grid_bounds['y'][0]) / self.grid_size)
        z = int((world_pos[2] - self.grid_bounds['z'][0]) / self.grid_size)
        return (x, y, z)
    
    def grid_to_world(self, grid_pos):
        """转换为世界坐标"""
        x = grid_pos[0] * self.grid_size + self.grid_bounds['x'][0]
        y = grid_pos[1] * self.grid_size + self.grid_bounds['y'][0]
        z = grid_pos[2] * self.grid_size + self.grid_bounds['z'][0]
        return (x, y, z)
    
    def is_valid_position(self, world_pos):
        """检查位置是否有效（不碰撞）"""
        return not self.scene.check_collision(world_pos)
    
    def get_neighbors(self, grid_pos):
        """获取相邻网格"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    new_pos = (grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz)
                    world_pos = self.grid_to_world(new_pos)
                    
                    # 检查边界
                    if (self.grid_bounds['x'][0] <= world_pos[0] <= self.grid_bounds['x'][1] and
                        self.grid_bounds['y'][0] <= world_pos[1] <= self.grid_bounds['y'][1] and
                        self.grid_bounds['z'][0] <= world_pos[2] <= self.grid_bounds['z'][1]):
                        
                        if self.is_valid_position(world_pos):
                            neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, pos1, pos2):
        """启发函数（曼哈顿距离）"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
    
    def a_star(self, start_pos, goal_pos):
        """A*路径规划算法"""
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)
        
        # 优先队列
        open_set = [(0, start_grid)]
        heapq.heapify(open_set)
        
        # 记录路径
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current))
                    current = came_from[current]
                path.append(self.grid_to_world(start_grid))
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 没有找到路径
    
    def rrt_planning(self, start_pos, goal_pos, max_iterations=100):
        """RRT路径规划算法"""
        # 确保位置是元组类型
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        
        tree = {start_pos: None}  # 节点: 父节点
        step_size = 1.0
        
        for i in range(max_iterations):
            # 随机采样
            if np.random.random() < 0.1:  # 10%概率直接采样目标点
                random_point = goal_pos
            else:
                random_point = (
                    np.random.uniform(self.grid_bounds['x'][0], self.grid_bounds['x'][1]),
                    np.random.uniform(self.grid_bounds['y'][0], self.grid_bounds['y'][1]),
                    np.random.uniform(self.grid_bounds['z'][0], self.grid_bounds['z'][1])
                )
            
            # 找到最近的节点
            nearest_node = min(tree.keys(), 
                             key=lambda x: np.linalg.norm(np.array(x) - np.array(random_point)))
            
            # 向随机点方向扩展
            direction = np.array(random_point) - np.array(nearest_node)
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance * step_size
                new_point = tuple(np.array(nearest_node) + direction)
                
                # 检查路径是否有效
                if self.is_valid_position(new_point):
                    tree[new_point] = nearest_node
                    
                    # 检查是否到达目标
                    if np.linalg.norm(np.array(new_point) - np.array(goal_pos)) < step_size:
                        tree[goal_pos] = new_point
                        
                        # 重建路径
                        path = []
                        current = goal_pos
                        while current is not None:
                            path.append(current)
                            current = tree[current]
                        path.reverse()
                        return path
        
        return None  # 没有找到路径
    
    def smooth_path(self, path, max_iterations=10):
        """路径平滑"""
        if not path or len(path) < 3:
            return path
        
        smoothed_path = path.copy()
        
        for _ in range(max_iterations):
            changed = False
            
            for i in range(1, len(smoothed_path) - 1):
                # 尝试连接前后两个点
                prev_point = smoothed_path[i-1]
                next_point = smoothed_path[i+1]
                
                # 检查直接连接是否可行
                if self.is_valid_position(prev_point) and self.is_valid_position(next_point):
                    # 简单的线性插值检查
                    mid_point = tuple((np.array(prev_point) + np.array(next_point)) / 2)
                    if self.is_valid_position(mid_point):
                        # 移除中间点
                        smoothed_path.pop(i)
                        changed = True
                        break
            
            if not changed:
                break
        
        return smoothed_path
    
    def plan_to_targets(self, start_pos, targets):
        """为多个目标点规划路径"""
        all_paths = []
        current_pos = start_pos
        
        for target in targets:
            if not target['visited']:
                # 使用A*算法
                path = self.a_star(current_pos, target['position'])
                
                if path is None:
                    # 如果A*失败，尝试RRT
                    path = self.rrt_planning(current_pos, target['position'])
                
                if path:
                    # 平滑路径
                    smoothed_path = self.smooth_path(path)
                    all_paths.append(smoothed_path)
                    current_pos = target['position']
                else:
                    print(f"无法找到到目标 {target['position']} 的路径")
        
        return all_paths

if __name__ == "__main__":
    # 测试路径规划
    from scene_creation import DroneScene
    
    scene = DroneScene(gui=False)
    scene.create_obstacles()
    scene.create_targets()
    
    planner = PathPlanning(scene)
    
    start_pos = (0, 0, 1)
    goal_pos = (5, 5, 3)
    
    print("测试A*算法...")
    path = planner.a_star(start_pos, goal_pos)
    if path:
        print(f"找到路径，长度: {len(path)}")
        print(f"路径: {path}")
    else:
        print("未找到路径")
    
    print("\n测试RRT算法...")
    path = planner.rrt_planning(start_pos, goal_pos)
    if path:
        print(f"找到路径，长度: {len(path)}")
        print(f"路径: {path}")
    else:
        print("未找到路径") 