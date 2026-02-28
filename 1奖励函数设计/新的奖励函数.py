import numpy as np

def _calculate_reward(self, prev_position, current_dist):
    """增强的奖励函数"""
    terminated = False
    reward = 0.0

    # 1. 目标导向奖励 - 更平滑的距离奖励
    current_goal = self.goals[self.current_goal_idx]
    prev_dist = np.linalg.norm(prev_position - current_goal)
    dist_reduction = prev_dist - current_dist

    # 距离奖励 - 距离越近奖励密度越高
    norm_dist = current_dist / (self.city_size * 1.5)  # 归一化距离
    distance_reward = 15 * (1 - norm_dist ** 0.5)  # 使用平方根使近距离奖励更高
    reward += distance_reward

    # 移动方向奖励 - 朝向目标的速度分量
    goal_direction = (current_goal - prev_position) / (prev_dist + 1e-8)
    forward_velocity = np.dot(self.velocity, goal_direction)
    reward += 2 * max(0, forward_velocity)  # 仅奖励朝向目标的速度

    # 2. 目标达成奖励 - 分阶段奖励
    if self.reached_goal:
        completion_bonus = 400  # 基础奖励
        # 额外奖励：剩余电池越多、用时越短奖励越高
        time_bonus = 100 * (1 - self.current_step / self.max_steps)
        battery_bonus = 50 * (self.battery / self.battery_capacity)
        reward += completion_bonus + time_bonus + battery_bonus

        # 额外奖励：连续成功完成目标
        if len(self.completed_goals) > 1:
            reward += 50 * (len(self.completed_goals) - 1)

    # 3. 避障奖励 - 更敏感的安全距离奖励
    safety_ratio = min(1.0, self.closest_obstacle_dist / self.safety_distance)
    safety_reward = 10 * safety_ratio ** 2  # 使用平方使安全距离奖励更敏感
    reward += safety_reward

    # 危险警告惩罚 - 接近障碍物时给予惩罚
    if self.closest_obstacle_dist < self.safety_distance * 0.7:
        reward -= 20 * (1 - safety_ratio) ** 2

    # 4. 碰撞惩罚 - 严重惩罚但不终止
    collision_penalty = 0
    if self._check_collision():
        if self.position[2] <= self.world_bounds['z'][0] + 1:  # 地面碰撞
            collision_penalty = -500
        elif any(np.sqrt((self.position[0] - o[0]) ** 2 + (self.position[1] - o[1]) ** 2) < o[2]
                 for o in self.obstacles + self.trees):  # 障碍物碰撞
            collision_penalty = -600
        else:  # 边界碰撞
            collision_penalty = -400
        terminated = True

    # 5. 能量效率奖励 - 鼓励高效飞行
    energy_efficiency = 15 * np.exp(-1.5 * np.linalg.norm(self.velocity) / self.max_speed)
    reward += energy_efficiency

    # 6. 风力补偿 - 逆风飞行惩罚，顺风飞行奖励
    wind_effect = np.dot(self.velocity, self.wind_direction) * self.wind_strength * 0.05
    reward -= wind_effect  # 逆风惩罚，顺风奖励

    # 7. 高度优化奖励 - 鼓励在安全高度飞行
    optimal_height = 80
    height_diff = abs(self.position[2] - optimal_height) / optimal_height
    height_reward = 8 * (1 - height_diff ** 2)
    reward += height_reward

    # 8. 稳定性奖励 - 鼓励平稳飞行
    jerk = np.linalg.norm(self.velocity - (self.position - prev_position))
    stability_reward = 5 * np.exp(-3 * jerk)
    reward += stability_reward

    return reward + collision_penalty, terminated