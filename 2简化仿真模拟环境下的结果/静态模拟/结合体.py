import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
from typing import Optional, Tuple, List, Dict
import warnings
from scipy.spatial import KDTree

# 设置中文字体并忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning)


class MultiAgentDroneDeliveryEnv(Env):
    def __init__(self, render_mode: Optional[str] = None, num_obstacles: int = 12, num_drones: int = 3,
                 city_size: int = 1000):
        super().__init__()

        self.num_drones = num_drones
        self.city_size = city_size

        # 动作空间 - 每个无人机有3维动作
        self.action_space = [spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                             for _ in range(num_drones)]

        # 观察空间 - 为每个无人机定义独立的观察空间
        self.observation_space = [spaces.Dict({
            'position': spaces.Box(low=0, high=city_size, shape=(3,), dtype=np.float32),
            'velocity': spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'goal': spaces.Box(low=0, high=city_size, shape=(3,), dtype=np.float32),
            'obstacles': spaces.Box(low=0, high=city_size, shape=(num_obstacles, 5), dtype=np.float32),
            'other_drones': spaces.Box(low=-city_size, high=city_size, shape=(num_drones - 1, 3), dtype=np.float32),
            'battery': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        }) for _ in range(num_drones)]

        # 生成城市建筑作为障碍物
        self._generate_city_buildings(num_obstacles)

        # 环境参数
        self.world_bounds = {'x': (0, city_size), 'y': (0, city_size), 'z': (5, 300)}

        # 随机选择起点和终点
        self._select_start_positions_and_goals()

        self.max_steps = 600
        self.position_scale = 2.0  # 增加移动速度以适应更大的城市
        self.velocity_scale = 1.2

        # 无人机参数
        self.battery_capacity = 100.0
        self.energy_consumption_rate = 0.05  # 降低能耗以适应更大空间
        self.max_speed = 10.0  # 提高最大速度
        self.min_distance_between_drones = 10.0
        self.dynamic_safety_distance = True

        # 精度参数
        self.goal_radius = 5.0  # 增大目标半径
        self.height_tolerance = 2.0
        self.terminal_phase_threshold = 50.0

        # 避障参数
        self.safety_distance = 20.0  # 增大安全距离
        self.closest_obstacle_distances = [float('inf')] * num_drones

        # 步数控制参数
        self.optimal_steps = [200, 220, 240][:num_drones]  # 增加步数以适应更大空间
        self.step_penalty_factor = 0.05
        self.path_efficiency_window = 20

        # 渲染与状态跟踪
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.trajectories = []
        self.episode_count = 0
        self.drones_reached_goal = [False] * num_drones
        self.collision_count = 0
        self.path_efficiency = [1.0] * num_drones
        self.last_positions = []

        self.reset()

    def _generate_city_buildings(self, num_buildings):
        """生成三维城市建筑作为障碍物"""
        avg_building_size = 40
        min_height = 20
        max_height = 150

        self.buildings = []
        for _ in range(num_buildings):
            width = np.random.normal(avg_building_size, avg_building_size / 3)
            depth = np.random.normal(avg_building_size, avg_building_size / 3)
            x = np.random.uniform(0, self.city_size - width)
            y = np.random.uniform(0, self.city_size - depth)

            if np.random.random() < 0.02:
                height = np.random.uniform(max_height * 0.7, max_height)
            else:
                height = np.random.uniform(min_height, max_height * 0.3)

            self.buildings.append((x, y, width, depth, height))

        # 转换为障碍物格式 (x, y, r, h)
        self.obstacles = []
        for b in self.buildings:
            x, y, w, d, h = b
            # 近似为圆形障碍物，半径取宽度和深度的平均值
            r = (w + d) / 4
            self.obstacles.append((x + w / 2, y + d / 2, r, h))

    def _select_start_positions_and_goals(self):
        """随机选择起点和终点位置"""
        # 选择不同的建筑作为起点和终点
        start_indices = np.random.choice(len(self.buildings), self.num_drones, replace=False)
        end_indices = np.random.choice(len(self.buildings), self.num_drones, replace=False)

        self.start_positions = []
        self.goals = []

        for i in range(self.num_drones):
            # 起点
            start_b = self.buildings[start_indices[i]]
            start_x = start_b[0] + start_b[2] / 2
            start_y = start_b[1] + start_b[3] / 2
            start_z = start_b[4] + 5  # 在建筑顶部上方5米

            # 终点
            end_b = self.buildings[end_indices[i]]
            end_x = end_b[0] + end_b[2] / 2
            end_y = end_b[1] + end_b[3] / 2
            end_z = end_b[4] + 5  # 在建筑顶部上方5米

            self.start_positions.append(np.array([start_x, start_y, start_z], dtype=np.float32))
            self.goals.append(np.array([end_x, end_y, end_z], dtype=np.float32))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[dict], dict]:
        super().reset(seed=seed)
        self.positions = [pos.copy() for pos in self.start_positions]
        self.velocities = [np.zeros(3, dtype=np.float32) for _ in range(self.num_drones)]
        self.batteries = [self.battery_capacity for _ in range(self.num_drones)]
        self.current_step = 0
        self.trajectories = [[pos.copy()] for pos in self.start_positions]
        self.best_distances = [np.linalg.norm(pos - goal) for pos, goal in zip(self.positions, self.goals)]
        self.closest_obstacle_distances = [float('inf')] * self.num_drones
        self.drones_reached_goal = [False] * self.num_drones
        self.collision_count = 0
        self.path_efficiency = [1.0] * self.num_drones
        self.last_positions = [pos.copy() for pos in self.start_positions]

        if hasattr(self, 'dynamic_obstacles') and self.dynamic_obstacles:
            self.obstacle_velocities = [
                np.random.uniform(-0.5, 0.5, size=2) for _ in range(len(self.obstacles))
            ]

        observations = [self._get_obs(i) for i in range(self.num_drones)]
        return observations, {}

    def step(self, actions: List[np.ndarray]) -> Tuple[List[dict], List[float], bool, bool, dict]:
        actions = [np.clip(action, -1, 1) for action in actions]
        prev_positions = [pos.copy() for pos in self.positions]
        prev_closest_distances = self.closest_obstacle_distances.copy()

        # 更新无人机状态（带加速度限制）
        max_acceleration = 0.5
        for i in range(self.num_drones):
            acceleration = actions[i] - self.velocities[i]
            if np.linalg.norm(acceleration) > max_acceleration:
                acceleration = acceleration / np.linalg.norm(acceleration) * max_acceleration
            self.velocities[i] += acceleration

            speed = np.linalg.norm(self.velocities[i])
            if speed > self.max_speed:
                self.velocities[i] = self.velocities[i] / speed * self.max_speed

            self.positions[i] += self.velocities[i] * self.position_scale
            self.positions[i] = np.clip(
                self.positions[i],
                [self.world_bounds['x'][0], self.world_bounds['y'][0], self.world_bounds['z'][0]],
                [self.world_bounds['x'][1], self.world_bounds['y'][1], self.world_bounds['z'][1]]
            )

            # 更新电池与轨迹
            self.batteries[i] -= self.energy_consumption_rate * (1 + speed / self.max_speed)
            self.batteries[i] = max(0, self.batteries[i])
            self.trajectories[i].append(self.positions[i].copy())
            current_distance = np.linalg.norm(self.positions[i] - self.goals[i])
            self.best_distances[i] = min(self.best_distances[i], current_distance)

        # 更新动态障碍物与最近障碍物距离
        if hasattr(self, 'dynamic_obstacles') and self.dynamic_obstacles:
            self._update_dynamic_obstacles()
        self._update_closest_obstacle_distances()

        # 计算路径效率
        for i in range(self.num_drones):
            actual_move = np.linalg.norm(self.positions[i] - self.last_positions[i])
            optimal_move = np.linalg.norm(self.velocities[i] * self.position_scale)
            if optimal_move > 0:
                efficiency = actual_move / optimal_move
                self.path_efficiency[i] = 0.9 * self.path_efficiency[i] + 0.1 * efficiency
            self.last_positions[i] = self.positions[i].copy()

        # 计算奖励与终止条件
        rewards = []
        terminated = False
        all_terminated = True

        for i in range(self.num_drones):
            current_distance = np.linalg.norm(self.positions[i] - self.goals[i])

            # 步数控制奖励
            step_reward = 0
            if self.current_step > self.optimal_steps[i]:
                step_penalty = -self.step_penalty_factor * (self.current_step - self.optimal_steps[i])
                step_reward += step_penalty

            # 路径效率奖励
            path_efficiency_reward = 2.0 * (self.path_efficiency[i] - 0.7)

            reward, drone_terminated = self._calculate_reward(
                i, prev_positions[i], current_distance, prev_closest_distances[i])

            reward += step_reward + path_efficiency_reward
            rewards.append(reward)
            if not drone_terminated:
                all_terminated = False

        # 团队奖励
        num_reached = sum(self.drones_reached_goal)
        if num_reached > 0:
            team_reward = 80 * num_reached
            for i in range(self.num_drones):
                if not self.drones_reached_goal[i]:
                    rewards[i] += team_reward

        if all(self.drones_reached_goal):
            for i in range(self.num_drones):
                rewards[i] += 600
            print("所有无人机均到达目标！")

        terminated = all_terminated
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        observations = [self._get_obs(i) for i in range(self.num_drones)]
        info = {
            "collision_count": self.collision_count,
            "path_efficiency": np.mean(self.path_efficiency)
        }
        return observations, rewards, terminated, truncated, info

    def _update_closest_obstacle_distances(self):
        for i in range(self.num_drones):
            pos = self.positions[i]
            min_dist = float('inf')
            for (x, y, r, h) in self.obstacles:
                horizontal_dist = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                vertical_overlap = pos[2] < h
                if vertical_overlap:
                    effective_dist = horizontal_dist - r
                    if effective_dist < min_dist:
                        min_dist = effective_dist
            self.closest_obstacle_distances[i] = min_dist

    def _calculate_reward(self, drone_idx: int, prev_position: np.ndarray,
                          current_distance: float, prev_closest_obstacle_dist: float) -> Tuple[float, bool]:
        terminated = False
        reward = 0.0
        current_closest = self.closest_obstacle_distances[drone_idx]

        # 目标导向奖励
        prev_distance = np.linalg.norm(prev_position - self.goals[drone_idx])
        distance_reduction = prev_distance - current_distance
        reward += 3.0 * distance_reduction * self.path_efficiency[drone_idx]

        # 目标达成奖励
        if current_distance < self.goal_radius:
            height_error = abs(self.positions[drone_idx][2] - self.goals[drone_idx][2])
            if height_error < self.height_tolerance:
                reward += 300
            else:
                reward += 150
            terminated = True
            self.drones_reached_goal[drone_idx] = True
            print(f"无人机 {drone_idx} 在 {self.current_step} 步到达目标!")

        # 避障奖励
        if current_closest < self.safety_distance:
            reward -= 10 * (self.safety_distance - current_closest)
        else:
            reward += 4.0

        # 无人机间避碰惩罚
        min_drone_dist = min(np.linalg.norm(self.positions[i] - self.positions[drone_idx])
                             for i in range(self.num_drones) if i != drone_idx)
        if min_drone_dist < 2 * self.min_distance_between_drones:
            reward -= 15 * (2 * self.min_distance_between_drones - min_drone_dist)

        # 碰撞惩罚
        if self._check_obstacle_collision(drone_idx):
            reward -= 300
            terminated = True
            self.collision_count += 1
            print(f"无人机 {drone_idx} 障碍物碰撞!")

        if self._check_drone_collision(drone_idx):
            reward -= 400
            terminated = True
            self.collision_count += 1
            print(f"无人机 {drone_idx} 互撞!")

        # 边界违规惩罚
        if (self.positions[drone_idx][0] <= self.world_bounds['x'][0] or
                self.positions[drone_idx][0] >= self.world_bounds['x'][1] or
                self.positions[drone_idx][1] <= self.world_bounds['y'][0] or
                self.positions[drone_idx][1] >= self.world_bounds['y'][1] or
                self.positions[drone_idx][2] <= self.world_bounds['z'][0] or
                self.positions[drone_idx][2] >= self.world_bounds['z'][1]):
            reward -= 100
            terminated = True
            print(f"无人机 {drone_idx} 边界违规!")

        # 电池耗尽惩罚
        if self.batteries[drone_idx] <= 0:
            reward -= 100
            terminated = True
            print(f"无人机 {drone_idx} 电池耗尽!")

        # 平稳飞行奖励
        velocity_change = np.linalg.norm(self.velocities[drone_idx] -
                                         (np.zeros(3) if len(self.trajectories[drone_idx]) < 2
                                          else (self.positions[drone_idx] - self.trajectories[drone_idx][
                                             -2]) / self.position_scale))
        smoothness_reward = max(0, 3.0 - velocity_change)
        reward += smoothness_reward

        return reward, terminated

    def _get_obs(self, drone_idx: int) -> dict:
        obstacle_info = np.zeros((len(self.obstacles), 5), dtype=np.float32)
        for i, (x, y, r, h) in enumerate(self.obstacles):
            dx = x - self.positions[drone_idx][0]
            dy = y - self.positions[drone_idx][1]
            dist = max(np.sqrt(dx ** 2 + dy ** 2) - r, 0.1)
            obstacle_info[i] = [dx, dy, r, h - self.positions[drone_idx][2], 1.0 / dist]

        other_drones = []
        for i in range(self.num_drones):
            if i != drone_idx:
                other_drones.append(self.positions[i] - self.positions[drone_idx])
        other_drones = np.array(other_drones, dtype=np.float32)

        return {
            'position': self.positions[drone_idx].copy(),
            'velocity': self.velocities[drone_idx].copy(),
            'goal': self.goals[drone_idx].copy() - self.positions[drone_idx].copy(),
            'obstacles': obstacle_info,
            'other_drones': other_drones,
            'battery': np.array([self.batteries[drone_idx]], dtype=np.float32)
        }

    def _update_dynamic_obstacles(self) -> None:
        new_obstacles = []
        for i, ((x, y, r, h), vel) in enumerate(zip(self.obstacles, self.obstacle_velocities)):
            new_x = x + vel[0]
            new_y = y + vel[1]

            if new_x - r < self.world_bounds['x'][0]:
                new_x = self.world_bounds['x'][0] + r
                self.obstacle_velocities[i][0] *= -1
            elif new_x + r > self.world_bounds['x'][1]:
                new_x = self.world_bounds['x'][1] - r
                self.obstacle_velocities[i][0] *= -1

            if new_y - r < self.world_bounds['y'][0]:
                new_y = self.world_bounds['y'][0] + r
                self.obstacle_velocities[i][1] *= -1
            elif new_y + r > self.world_bounds['y'][1]:
                new_y = self.world_bounds['y'][1] - r
                self.obstacle_velocities[i][1] *= -1

            new_obstacles.append((new_x, new_y, r, h))

        self.obstacles = new_obstacles

    def _check_obstacle_collision(self, drone_idx: int) -> bool:
        pos = self.positions[drone_idx]
        for x, y, r, h in self.obstacles:
            horizontal_dist = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            vertical_clearance = pos[2] < h
            if horizontal_dist < r and vertical_clearance:
                return True
        return False

    def _check_drone_collision(self, drone_idx: int) -> bool:
        pos = self.positions[drone_idx]
        for i in range(self.num_drones):
            if i != drone_idx:
                if self.dynamic_safety_distance:
                    speed_factor = (np.linalg.norm(self.velocities[drone_idx]) +
                                    np.linalg.norm(self.velocities[i])) / (2 * self.max_speed)
                    current_min_distance = self.min_distance_between_drones * (1 + speed_factor)
                else:
                    current_min_distance = self.min_distance_between_drones

                distance = np.linalg.norm(pos - self.positions[i])
                if distance < current_min_distance:
                    return True

                next_pos = pos + self.velocities[drone_idx] * self.position_scale
                next_other_pos = self.positions[i] + self.velocities[i] * self.position_scale
                if np.linalg.norm(next_pos - next_other_pos) < current_min_distance:
                    return True
        return False

    def render(self) -> None:
        if self.render_mode is None:
            return

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(self.world_bounds['x'])
            self.ax.set_ylim(self.world_bounds['y'])
            self.ax.set_zlim(self.world_bounds['z'])
            self.ax.set_xlabel('X (米)')
            self.ax.set_ylabel('Y (米)')
            self.ax.set_zlabel('高度 (米)')
            self.ax.set_title('多无人机城市配送环境', pad=20)
            self.ax.grid(True)

        self.ax.clear()

        # 重新设置坐标轴和标题
        self.ax.set_xlim(self.world_bounds['x'])
        self.ax.set_ylim(self.world_bounds['y'])
        self.ax.set_zlim(self.world_bounds['z'])
        self.ax.set_xlabel('X (米)')
        self.ax.set_ylabel('Y (米)')
        self.ax.set_zlabel('高度 (米)')
        self.ax.set_title(f'多无人机城市配送 - 步骤: {self.current_step}/{self.max_steps}', pad=20)
        self.ax.grid(True)

        # 绘制城市建筑
        max_height = max(b[4] for b in self.buildings) if self.buildings else 1
        for building in self.buildings:
            x, y, width, depth, height = building

            # 根据建筑高度计算颜色
            height_ratio = height / max_height
            color = (
                0.3 + height_ratio * 0.7,
                0.7 - height_ratio * 0.4,
                0.7 - height_ratio * 0.4,
                0.8
            )

            # 绘制建筑
            vertices = [
                [x, y, 0], [x + width, y, 0], [x + width, y + depth, 0], [x, y + depth, 0],
                [x, y, height], [x + width, y, height], [x + width, y + depth, height], [x, y + depth, height]
            ]
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[3], vertices[0], vertices[4], vertices[7]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[2], vertices[3]]
            ]
            self.ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k'))

        # 绘制无人机轨迹
        colors = ['blue', 'green', 'purple'][:self.num_drones]
        for i in range(self.num_drones):
            if len(self.trajectories[i]) > 1:
                traj = np.array(self.trajectories[i])
                self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                             color=colors[i], linewidth=2, alpha=0.8,
                             label=f'无人机 {i} 轨迹')

        # 绘制无人机
        drone_size = 5
        for i in range(self.num_drones):
            x, y, z = self.positions[i]
            vertices = [
                [x, y, z], [x + drone_size, y, z], [x + drone_size, y + drone_size, z], [x, y + drone_size, z],
                [x, y, z + drone_size], [x + drone_size, y, z + drone_size],
                [x + drone_size, y + drone_size, z + drone_size], [x, y + drone_size, z + drone_size]
            ]
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[3], vertices[0], vertices[4], vertices[7]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[2], vertices[3]]
            ]
            self.ax.add_collection3d(Poly3DCollection(faces, facecolors=colors[i], linewidths=1, edgecolors='k'))

            # 添加朝向指示器
            if np.linalg.norm(self.velocities[i]) > 0.1:
                direction = self.velocities[i] / np.linalg.norm(self.velocities[i]) * 10
                self.ax.quiver(
                    x, y, z,
                    direction[0], direction[1], direction[2],
                    color='black', arrow_length_ratio=0.3, linewidth=1)

        # 绘制目标
        for i in range(self.num_drones):
            x, y, z = self.goals[i]
            self.ax.scatter(x, y, z, c='gold', marker='*', s=300, label=f'目标 {i}',
                            edgecolors='red', linewidths=1)

            # 添加目标区域指示
            theta = np.linspace(0, 2 * np.pi, 50)
            x_circle = x + self.goal_radius * np.cos(theta)
            y_circle = y + self.goal_radius * np.sin(theta)
            self.ax.plot(x_circle, y_circle, np.ones_like(x_circle) * z,
                         'r--', alpha=0.5)

        # 信息显示
        min_drone_dist = min(np.linalg.norm(self.positions[i] - self.positions[j])
                             for i in range(self.num_drones)
                             for j in range(i + 1, self.num_drones))

        info_text = (f"环境状态:\n"
                     f"当前步数: {self.current_step}/{self.max_steps}\n"
                     f"碰撞次数: {self.collision_count}\n"
                     f"无人机最小间距: {min_drone_dist:.1f}m\n"
                     f"平均路径效率: {np.mean(self.path_efficiency):.2f}\n\n"
                     f"无人机状态:")

        for i in range(self.num_drones):
            distance = np.linalg.norm(self.positions[i] - self.goals[i])
            status = "已到达" if self.drones_reached_goal[i] else "飞行中"
            info_text += (f"\n无人机 {i}: {status} | "
                          f"电池: {self.batteries[i]:.1f}% | "
                          f"距离目标: {distance:.1f}m | "
                          f"速度: {np.linalg.norm(self.velocities[i]):.1f}m/s")

        self.ax.text2D(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
            fontsize=9)

        # 添加图例
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

        # 调整视角
        self.ax.view_init(elev=45, azim=-60)

        plt.draw()
        plt.pause(0.05)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class MultiAgentDronePolicy:
    def __init__(self, env: MultiAgentDroneDeliveryEnv, drone_idx: int):
        self.env = env
        self.drone_idx = drone_idx
        self.last_action = np.zeros(3)
        self.action_history = np.zeros((5, 3))
        self.step_count_since_last_progress = 0
        self.last_goal_distance = float('inf')

    def predict(self, obs: dict) -> np.ndarray:
        position = obs['position']
        velocity = obs['velocity']
        goal_direction = obs['goal']
        obstacles = obs['obstacles']
        other_drones = obs['other_drones']
        battery = obs['battery'][0]

        # 跟踪目标距离变化
        current_goal_distance = np.linalg.norm(goal_direction)
        if current_goal_distance >= self.last_goal_distance - 0.1:
            self.step_count_since_last_progress += 1
        else:
            self.step_count_since_last_progress = 0
        self.last_goal_distance = current_goal_distance

        # 目标导向
        goal_dist = np.linalg.norm(goal_direction)
        if goal_dist > 0:
            goal_dir = goal_direction / goal_dist
        else:
            goal_dir = np.zeros(3)

        # 当长时间未接近目标时，增强目标导向
        urgency_factor = 1.0 + min(1.0, self.step_count_since_last_progress / 20)
        action = goal_dir * 1.0 * urgency_factor

        # 速度控制优化
        optimal_speed = min(6.0, goal_dist / 10)  # 调整速度以适应更大空间
        if battery < 30:
            optimal_speed *= 0.7
        current_speed = np.linalg.norm(velocity)
        speed_diff = optimal_speed - current_speed
        action *= 1 + 0.2 * np.tanh(speed_diff)

        # 障碍物规避
        obstacle_avoid_action = np.zeros(3)
        for i, (rel_x, rel_y, radius, rel_h, repulsion_strength) in enumerate(obstacles):
            obstacle_dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
            if obstacle_dist < 50:  # 增大障碍物检测范围
                repulsion = min(repulsion_strength * 0.5, 3.0)
                avoid_dir = np.array([-rel_x, -rel_y, 0])
                if np.linalg.norm(avoid_dir) > 0:
                    avoid_dir = avoid_dir / np.linalg.norm(avoid_dir)
                if rel_h < 10:  # 调整高度避障阈值
                    avoid_dir[2] += 0.5
                obstacle_avoid_action += repulsion * avoid_dir
        action += obstacle_avoid_action * 0.8

        # 无人机避碰
        drone_avoid_action = np.zeros(3)
        for drone_rel_pos in other_drones:
            drone_dist = np.linalg.norm(drone_rel_pos)
            if drone_dist < 30:  # 增大无人机避碰检测范围
                repulsion = 2.0 / (drone_dist + 1e-6) ** 2
                repulsion = min(repulsion, 3.0)
                avoid_dir = -drone_rel_pos / (drone_dist + 1e-6)
                drone_avoid_action += repulsion * avoid_dir * 0.7
        action += drone_avoid_action

        # 终端阶段优化
        if goal_dist < self.env.terminal_phase_threshold:
            action *= 0.5
            height_error = self.env.goals[self.drone_idx][2] - position[2]
            action[2] += 0.3 * np.tanh(height_error / 2)
            if goal_dist < 20:  # 增大终端阶段阈值
                action += goal_dir * 0.3 * goal_dist / 20

        # 动作平滑
        smoothed_action = 0.6 * action + 0.4 * np.mean(self.action_history, axis=0)
        self.action_history = np.roll(self.action_history, shift=-1, axis=0)
        self.action_history[-1] = smoothed_action

        return np.clip(smoothed_action, -1, 1)


def main():
    num_drones = 3  # 使用3架无人机
    env = MultiAgentDroneDeliveryEnv(render_mode='human', num_obstacles=50, num_drones=num_drones)
    policies = [MultiAgentDronePolicy(env, i) for i in range(num_drones)]

    try:
        for episode in range(5):
            observations, _ = env.reset()
            total_rewards = [0.0 for _ in range(num_drones)]
            done = False

            while not done:
                actions = [policies[i].predict(observations[i]) for i in range(num_drones)]
                observations, rewards, terminated, truncated, info = env.step(actions)
                for i in range(num_drones):
                    total_rewards[i] += rewards[i]
                done = terminated or truncated
                if env.render_mode == 'human':
                    env.render()

            print(f"Episode {episode + 1} 总奖励: {total_rewards}")
            print(f"碰撞次数: {info['collision_count']}")
            print(f"平均路径效率: {info['path_efficiency']:.2f}")

    except KeyboardInterrupt:
        print("模拟被用户中断")
    finally:
        env.close()


if __name__ == "__main__":
    main()