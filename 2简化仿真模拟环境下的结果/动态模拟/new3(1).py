import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from typing import Optional, Tuple, List, Dict


class MultiAgentDroneDeliveryEnv(Env):
    def __init__(self, render_mode: Optional[str] = None, num_obstacles: int = 8, num_drones: int = 2):
        super().__init__()

        self.num_drones = num_drones

        # 动作空间 - 每个无人机有3维动作
        self.action_space = [spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                             for _ in range(num_drones)]

        # 观察空间 - 为每个无人机定义独立的观察空间
        self.observation_space = [spaces.Dict({
            'position': spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32),
            'velocity': spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'goal': spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32),
            'obstacles': spaces.Box(low=0, high=200, shape=(num_obstacles, 4), dtype=np.float32),
            'other_drones': spaces.Box(low=-200, high=200, shape=(num_drones - 1, 3), dtype=np.float32),  # 其他无人机的相对位置
            'battery': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        }) for _ in range(num_drones)]

        # 环境参数
        self.world_bounds = {'x': (0, 200), 'y': (0, 200), 'z': (5, 30)}
        self.goals = np.array([[180, 180, 15], [160, 190, 15]], dtype=np.float32)[:num_drones]
        self.start_positions = np.array([[20, 20, 10], [30, 25, 10]], dtype=np.float32)[:num_drones]
        self.max_steps = 600  # 延长最大步数
        self.position_scale = 0.5
        self.velocity_scale = 1.5

        # 障碍物参数
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.min_obstacle_radius = 5.0
        self.max_obstacle_radius = 12.0
        self.obstacle_height_range = (10, 25)
        self.dynamic_obstacles = True

        # 无人机参数（优化电池续航）
        self.battery_capacity = 100.0
        self.energy_consumption_rate = 0.08  # 降低能耗，延长续航
        self.max_speed = 8.0
        self.min_distance_between_drones = 3.0

        # 精度参数
        self.goal_radius = 2.0
        self.height_tolerance = 0.5
        self.terminal_phase_threshold = 25.0

        # 新增：避障奖励相关参数
        self.safety_distance = self.min_obstacle_radius + 5.0  # 安全距离=障碍物半径+5m
        self.closest_obstacle_distances = [float('inf')] * num_drones  # 记录最近障碍物距离

        # 渲染与状态跟踪
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.trajectories = []
        self.episode_count = 0
        self.drones_reached_goal = [False] * num_drones  # 跟踪已到达的无人机

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[dict], dict]:
        super().reset(seed=seed)
        self.positions = [pos.copy() for pos in self.start_positions]
        self.velocities = [np.zeros(3, dtype=np.float32) for _ in range(self.num_drones)]
        self.batteries = [self.battery_capacity for _ in range(self.num_drones)]
        self.current_step = 0
        self.trajectories = [[pos.copy()] for pos in self.start_positions]
        self.best_distances = [np.linalg.norm(pos - goal) for pos, goal in zip(self.positions, self.goals)]
        self._generate_obstacles()
        self.closest_obstacle_distances = [float('inf')] * self.num_drones
        self.drones_reached_goal = [False] * self.num_drones  # 重置到达状态

        if self.dynamic_obstacles:
            self.obstacle_velocities = [
                np.random.uniform(-0.5, 0.5, size=2) for _ in range(self.num_obstacles)
            ]

        observations = [self._get_obs(i) for i in range(self.num_drones)]
        return observations, {}

    def step(self, actions: List[np.ndarray]) -> Tuple[List[dict], List[float], bool, bool, dict]:
        actions = [np.clip(action, -1, 1) for action in actions]
        prev_positions = [pos.copy() for pos in self.positions]
        prev_closest_distances = self.closest_obstacle_distances.copy()  # 保存上一步的最近距离

        # 更新无人机状态
        for i in range(self.num_drones):
            # 更新速度与位置（保持原有逻辑）
            self.velocities[i] = self.velocities[i] * 0.7 + actions[i] * self.velocity_scale
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
            speed = np.linalg.norm(self.velocities[i])
            self.batteries[i] -= self.energy_consumption_rate * (1 + speed / self.max_speed)
            self.batteries[i] = max(0, self.batteries[i])
            self.trajectories[i].append(self.positions[i].copy())
            current_distance = np.linalg.norm(self.positions[i] - self.goals[i])
            self.best_distances[i] = min(self.best_distances[i], current_distance)

        # 更新动态障碍物与最近障碍物距离
        if self.dynamic_obstacles:
            self._update_dynamic_obstacles()
        self._update_closest_obstacle_distances()  # 刷新最近障碍物距离

        # 计算奖励与终止条件
        rewards = []
        terminated = False
        all_terminated = True

        for i in range(self.num_drones):
            current_distance = np.linalg.norm(self.positions[i] - self.goals[i])
            # 传入上一步的最近障碍物距离用于计算避障奖励
            reward, drone_terminated = self._calculate_reward(
                i, prev_positions[i], current_distance, prev_closest_distances[i])
            rewards.append(reward)
            if not drone_terminated:
                all_terminated = False

        # 团队奖励：部分到达时给予激励
        num_reached = sum(self.drones_reached_goal)
        if num_reached > 0:
            team_reward = 80 * num_reached  # 每有1个到达，未到达的获得80奖励
            for i in range(self.num_drones):
                if not self.drones_reached_goal[i]:
                    rewards[i] += team_reward

        # 团队奖励：全部到达时给予高额奖励
        if all(self.drones_reached_goal):
            for i in range(self.num_drones):
                rewards[i] += 600  # 全到达奖励
            print("所有无人机均到达目标！")

        terminated = all_terminated
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        observations = [self._get_obs(i) for i in range(self.num_drones)]
        return observations, rewards, terminated, truncated, {}

    def _update_closest_obstacle_distances(self):
        """更新每个无人机与最近障碍物的距离"""
        for i in range(self.num_drones):
            pos = self.positions[i]
            min_dist = float('inf')
            for (x, y, r, h) in self.obstacles:
                # 计算水平距离（忽略高度，垂直方向在碰撞检测中处理）
                horizontal_dist = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                # 垂直方向是否有重叠（无人机高度 < 障碍物高度）
                vertical_overlap = pos[2] < h
                if vertical_overlap:
                    # 有效距离 = 水平距离 - 障碍物半径（负值表示已进入障碍物）
                    effective_dist = horizontal_dist - r
                    if effective_dist < min_dist:
                        min_dist = effective_dist
            self.closest_obstacle_distances[i] = min_dist

    def _calculate_reward(self, drone_idx: int, prev_position: np.ndarray, current_distance: float,
                          prev_closest_obstacle_dist: float) -> Tuple[float, bool]:
        terminated = False
        reward = 0.0
        current_closest = self.closest_obstacle_distances[drone_idx]

        # 1. 增强目标导向奖励（提高权重）
        prev_distance = np.linalg.norm(prev_position - self.goals[drone_idx])
        distance_reduction = prev_distance - current_distance
        reward += 3.0 * distance_reduction  # 提高目标吸引力

        # 2. 目标达成奖励（区分部分到达与完全到达）
        if current_distance < self.goal_radius:
            height_error = abs(self.positions[drone_idx][2] - self.goals[drone_idx][2])
            if height_error < self.height_tolerance:
                reward += 300  # 完美到达奖励
            else:
                reward += 150  # 基础到达奖励
            terminated = True
            self.drones_reached_goal[drone_idx] = True  # 标记为已到达
            print(f"无人机 {drone_idx} 在 {self.current_step} 步到达目标!")

        # 3. 新增：避障奖励
        # 3.1 保持安全距离奖励
        if current_closest > self.safety_distance:
            reward += 4.0  # 持续保持安全距离的奖励

        # 3.2 远离障碍物奖励（从危险区域逃离时）
        if prev_closest_obstacle_dist < self.safety_distance * 1.2:  # 仅对之前较近的情况奖励
            distance_increase = current_closest - prev_closest_obstacle_dist
            if distance_increase > 0:  # 成功远离
                reward += 3.0 * distance_increase  # 远离越多，奖励越多

        # 4. 障碍物碰撞惩罚（保持不变）
        if self._check_obstacle_collision(drone_idx):
            reward -= 150
            terminated = True
            print(f"无人机 {drone_idx} 障碍物碰撞!")

        # 5. 无人机间碰撞惩罚（保持不变）
        if self._check_drone_collision(drone_idx):
            reward -= 200
            terminated = True
            print(f"无人机 {drone_idx} 互撞!")

        # 6. 边界违规惩罚（保持不变）
        if (self.positions[drone_idx][0] <= self.world_bounds['x'][0] or
                self.positions[drone_idx][0] >= self.world_bounds['x'][1] or
                self.positions[drone_idx][1] <= self.world_bounds['y'][0] or
                self.positions[drone_idx][1] >= self.world_bounds['y'][1] or
                self.positions[drone_idx][2] <= self.world_bounds['z'][0] or
                self.positions[drone_idx][2] >= self.world_bounds['z'][1]):
            reward -= 100
            terminated = True
            print(f"无人机 {drone_idx} 边界违规!")

        # 7. 电池耗尽惩罚（保持不变）
        if self.batteries[drone_idx] <= 0:
            reward -= 100
            terminated = True
            print(f"无人机 {drone_idx} 电池耗尽!")

        # 8. 平稳飞行与能效奖励（微调）
        velocity_change = np.linalg.norm(self.velocities[drone_idx] -
                                         (np.zeros(3) if len(self.trajectories[drone_idx]) < 2
                                          else (self.positions[drone_idx] - self.trajectories[drone_idx][
                                             -2]) / self.position_scale))
        smoothness_reward = max(0, 3.0 - velocity_change)
        reward += smoothness_reward

        speed = np.linalg.norm(self.velocities[drone_idx])
        efficiency_reward = max(0, 2.0 - (speed / self.max_speed))
        reward += efficiency_reward

        return reward, terminated

    # 以下方法保持不变，但策略部分有优化
    def _get_obs(self, drone_idx: int) -> dict:
        obstacle_info = np.zeros((self.num_obstacles, 4), dtype=np.float32)
        for i, (x, y, r, h) in enumerate(self.obstacles):
            obstacle_info[i] = [
                x - self.positions[drone_idx][0],
                y - self.positions[drone_idx][1],
                r,
                h - self.positions[drone_idx][2]
            ]

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

    def _generate_obstacles(self) -> None:
        self.obstacles = []
        for _ in range(self.num_obstacles):
            valid = False
            while not valid:
                x = np.random.uniform(40, 160)
                y = np.random.uniform(40, 160)
                r = np.random.uniform(self.min_obstacle_radius, self.max_obstacle_radius)
                h = np.random.uniform(*self.obstacle_height_range)

                valid = True
                for pos in self.start_positions:
                    start_dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                    if start_dist < 30:
                        valid = False
                        break

                if valid:
                    for goal in self.goals:
                        goal_dist = np.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)
                        if goal_dist < 30:
                            valid = False
                            break

                if valid:
                    for (ox, oy, orad, _) in self.obstacles:
                        if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < (r + orad + 10):
                            valid = False
                            break

                if valid:
                    self.obstacles.append((x, y, r, h))

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
                distance = np.linalg.norm(pos - self.positions[i])
                if distance < self.min_distance_between_drones:
                    return True
        return False

    def render(self) -> None:
        if self.render_mode is None:
            return

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(self.world_bounds['x'])
            self.ax.set_ylim(self.world_bounds['y'])
            self.ax.set_zlim(self.world_bounds['z'])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('多无人机配送环境')

        self.ax.clear()

        # 绘制障碍物
        for x, y, r, h in self.obstacles:
            z = np.linspace(0, h, 20)
            theta = np.linspace(0, 2 * np.pi, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = r * np.cos(theta_grid) + x
            y_grid = r * np.sin(theta_grid) + y
            self.ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.5)
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = r * np.cos(theta) + x
            y_circle = r * np.sin(theta) + y
            self.ax.plot(x_circle, y_circle, h, color='darkred')

        # 绘制轨迹
        colors = ['blue', 'purple'][:self.num_drones]
        for i in range(self.num_drones):
            if len(self.trajectories[i]) > 1:
                traj = np.array(self.trajectories[i])
                self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                             color=colors[i], linewidth=1, alpha=0.5)

        # 绘制无人机
        for i in range(self.num_drones):
            self.ax.scatter(
                self.positions[i][0], self.positions[i][1], self.positions[i][2],
                c=colors[i], marker='o', s=50, label=f'无人机 {i}')

        # 绘制目标
        for i in range(self.num_drones):
            self.ax.scatter(
                self.goals[i][0], self.goals[i][1], self.goals[i][2],
                c='gold', marker='*', s=200, label=f'目标 {i}')

        # 信息显示
        self.ax.legend()
        info_text = f"步骤: {self.current_step}/{self.max_steps}\n"
        for i in range(self.num_drones):
            distance = np.linalg.norm(self.positions[i] - self.goals[i])
            info_text += f"无人机 {i}: 电池 {self.batteries[i]:.1f}% 距离 {distance:.1f}m\n"
        self.ax.text2D(0.02, 0.95, info_text, transform=self.ax.transAxes)

        plt.draw()
        plt.pause(0.01)

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
        self.obstacle_memory = {}

    def predict(self, obs: dict) -> np.ndarray:
        position = obs['position']
        velocity = obs['velocity']
        goal_direction = obs['goal']
        obstacles = obs['obstacles']
        other_drones = obs['other_drones']
        battery = obs['battery'][0]

        # 增强目标导向（提高基础权重）
        goal_dist = np.linalg.norm(goal_direction)
        if goal_dist > 0:
            goal_dir = goal_direction / goal_dist
        else:
            goal_dir = np.zeros(3)
        action = goal_dir * 1.0  # 提高目标吸引力

        # 速度控制优化
        optimal_speed = min(4.0, goal_dist / 5)
        if battery < 30:
            optimal_speed *= 0.7
        current_speed = np.linalg.norm(velocity)
        speed_diff = optimal_speed - current_speed
        action *= 1 + 0.2 * np.tanh(speed_diff)

        # 障碍物规避（限制幅度，避免过度避障）
        obstacle_avoid_action = np.zeros(3)
        for i, (rel_x, rel_y, radius, rel_h) in enumerate(obstacles):
            obstacle_dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
            if obstacle_dist < 50 and rel_h < 10:  # 仅处理近距障碍物
                repulsion = 1.0 / (obstacle_dist - radius + 1e-6) ** 2
                repulsion = min(repulsion, 2.0)
                avoid_dir = np.array([-rel_x, -rel_y, 0])
                if np.linalg.norm(avoid_dir) > 0:
                    avoid_dir = avoid_dir / np.linalg.norm(avoid_dir)
                if rel_h < 5:
                    avoid_dir[2] = 0.5
                obstacle_avoid_action += repulsion * avoid_dir * 0.5
        obstacle_avoid_action = np.clip(obstacle_avoid_action, -0.8, 0.8)  # 限制避障幅度
        action += obstacle_avoid_action

        # 无人机避碰（限制幅度）
        drone_avoid_action = np.zeros(3)
        for drone_rel_pos in other_drones:
            drone_dist = np.linalg.norm(drone_rel_pos)
            if drone_dist < 15:
                repulsion = 2.0 / (drone_dist + 1e-6) ** 2
                repulsion = min(repulsion, 3.0)
                avoid_dir = -drone_rel_pos / (drone_dist + 1e-6)
                drone_avoid_action += repulsion * avoid_dir * 0.7
        drone_avoid_action = np.clip(drone_avoid_action, -0.9, 0.9)  # 限制避碰幅度
        action += drone_avoid_action

        # 终端阶段优化（增强目标导向）
        if goal_dist < self.env.terminal_phase_threshold:
            action *= 0.5  # 减速
            height_error = self.env.goals[self.drone_idx][2] - position[2]
            action[2] += 0.3 * np.tanh(height_error / 2)
            if goal_dist < 10:
                action += goal_dir * 0.3 * goal_dist / 10  # 更强的精细定位

        # 平滑动作变化
        self.last_action = self.last_action * 0.4 + action * 0.6
        return np.clip(self.last_action, -1, 1)


def main():
    # 添加以下两行设置中文字体
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体

    num_drones = 2
    env = MultiAgentDroneDeliveryEnv(render_mode='human', num_obstacles=8, num_drones=num_drones)



    num_drones = 2
    env = MultiAgentDroneDeliveryEnv(render_mode='human', num_obstacles=8, num_drones=num_drones)
    policies = [MultiAgentDronePolicy(env, i) for i in range(num_drones)]

    try:
        for episode in range(5):
            observations, _ = env.reset()
            total_rewards = [0.0 for _ in range(num_drones)]
            done = False

            while not done:
                actions = [policies[i].predict(observations[i]) for i in range(num_drones)]
                observations, rewards, terminated, truncated, _ = env.step(actions)
                for i in range(num_drones):
                    total_rewards[i] += rewards[i]
                done = terminated or truncated
                if env.render_mode == 'human':
                    env.render()

            print(f"Episode {episode + 1} 总奖励: {total_rewards}")

    except KeyboardInterrupt:
        print("模拟被用户中断")
    finally:
        env.close()


if __name__ == "__main__":
    main()

#减少碰撞次数