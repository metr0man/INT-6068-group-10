import numpy as np
from collections import deque

class SafetyModule:
    """增强型控制屏障函数(CBF)安全模块"""

    def __init__(self, safety_distance=20.0, max_acceleration=0.7, detection_range=200):
        self.safety_distance = safety_distance  # 安全距离阈值
        self.max_acceleration = max_acceleration  # 最大加速度限制
        self.detection_range = detection_range  # 检测范围

        # 安全参数
        self.alpha = 0.8  # CBF稳定性参数
        self.emergency_threshold = 8.0  # 紧急情况阈值
        self.emergency_deceleration = 0.9  # 紧急减速系数

        # 状态跟踪
        self.closest_obstacle_dist = float('inf')
        self.closest_obstacle_dir = np.zeros(3)
        self.emergency_mode = False
        self.emergency_counter = 0

        # 避障历史记录
        self.avoidance_history = deque(maxlen=10)

    def compute_safe_action(self, position, velocity, obstacles, action):
        """应用CBF约束确保动作安全"""
        safe_action = action.copy()

        # 更新最近障碍物信息
        self._update_closest_obstacle(position, obstacles)

        # 如果距离过近，应用CBF约束
        if self.closest_obstacle_dist < self.safety_distance:
            # 计算安全控制量
            h = self.closest_obstacle_dist - self.safety_distance
            Lfh = np.dot(velocity[:2], self.closest_obstacle_dir[:2])
            Lgh = -self.closest_obstacle_dir[:2]

            # 求解QP问题(简化版)
            if np.dot(Lgh, Lgh) > 0:
                cbf_adjustment = max(0, (Lfh + self.alpha * h) / np.dot(Lgh, Lgh)) * Lgh
                safe_action[:2] = action[:2] - cbf_adjustment

        # 限制最大加速度
        safe_action = np.clip(safe_action, -self.max_acceleration, self.max_acceleration)

        # 紧急情况处理
        if self.emergency_mode:
            recovery_action = self._emergency_avoidance(position, velocity, obstacles)
            if recovery_action is not None:
                safe_action = recovery_action
                self.emergency_counter += 1
                # 持续紧急状态一段时间后退出
                if self.emergency_counter > 50:
                    self.emergency_mode = False
                    self.emergency_counter = 0

        # 记录避障动作
        if np.linalg.norm(safe_action - action) > 0.1:
            self.avoidance_history.append((position, safe_action))

        return safe_action

    def emergency_recovery(self, position, velocity, wind_speed, obstacles):
        """紧急恢复策略"""
        # 检查是否需要进入紧急模式
        self._update_closest_obstacle(position, obstacles)

        # 风速过大或接近障碍物时触发紧急模式
        if wind_speed > 7.5 or self.closest_obstacle_dist < self.emergency_threshold:
            self.emergency_mode = True
            self.emergency_counter = 0

        # 如果处于紧急模式，执行紧急恢复
        if self.emergency_mode:
            return self._emergency_avoidance(position, velocity, obstacles)

        return None

    def _emergency_avoidance(self, position, velocity, obstacles):
        """紧急避障策略"""
        # 优先考虑向上飞行以避免碰撞
        avoidance_dir = self.closest_obstacle_dir

        # 智能高度调整
        if self.closest_obstacle_dist < self.emergency_threshold * 0.5:
            # 接近障碍物时增加向上分量
            avoidance_dir[2] = 0.8
        else:
            avoidance_dir[2] = 0.4

        # 检查历史避障方向，避免振荡
        if len(self.avoidance_history) > 3:
            recent_dirs = np.array([a[1][:2] for a in self.avoidance_history])
            avg_dir = np.mean(recent_dirs, axis=0)
            if np.linalg.norm(avg_dir) > 0.2:
                # 如果最近避障方向一致，尝试引入随机性
                random_perturbation = np.random.normal(0, 0.1, 2)
                avoidance_dir[:2] += random_perturbation
                avoidance_dir[:2] /= np.linalg.norm(avoidance_dir[:2]) + 1e-6

        # 如果速度过快，先减速
        if np.linalg.norm(velocity) > 6.0:
            recovery_action = -velocity * self.emergency_deceleration
            return np.clip(recovery_action, -self.max_acceleration, self.max_acceleration)

        # 否则，执行避障动作
        recovery_action = avoidance_dir * self.max_acceleration * 0.8
        return np.clip(recovery_action, -self.max_acceleration, self.max_acceleration)

    def _update_closest_obstacle(self, position, obstacles):
        """更新最近障碍物距离和方向"""
        min_dist = float('inf')
        closest_dir = np.zeros(3)

        for obs in obstacles:
            if len(obs) == 4:  # 圆柱体障碍物(x,y,r,h)
                dx = position[0] - obs[0]
                dy = position[1] - obs[1]
                dist = np.sqrt(dx ** 2 + dy ** 2) - obs[2]
                if dist < min_dist and position[2] < obs[3]:
                    min_dist = dist
                    closest_dir = np.array([dx, dy, 0])
            elif len(obs) == 6:  # 长方体障碍物(x1,y1,z1,x2,y2,z2)
                dx = max(obs[0] - position[0], position[0] - obs[3], 0)
                dy = max(obs[1] - position[1], position[1] - obs[4], 0)
                dz = max(obs[2] - position[2], position[2] - obs[5], 0)
                dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if dist < min_dist:
                    min_dist = dist
                    # 计算避开方向
                    if dx > 0:
                        dir_x = position[0] - ((obs[0] + obs[3]) / 2)
                    else:
                        dir_x = 0
                    if dy > 0:
                        dir_y = position[1] - ((obs[1] + obs[4]) / 2)
                    else:
                        dir_y = 0
                    if dz > 0:
                        dir_z = position[2] - ((obs[2] + obs[5]) / 2)
                    else:
                        dir_z = 0
                    closest_dir = np.array([dir_x, dir_y, dir_z])

        self.closest_obstacle_dist = max(min_dist, 0.1)
        self.closest_obstacle_dir = closest_dir / (np.linalg.norm(closest_dir) + 1e-6) if min_dist < float(
            'inf') else np.zeros(3)

    def get_safety_metrics(self):
        """获取安全指标"""
        return {
            'closest_obstacle_dist': self.closest_obstacle_dist,
            'safety_distance': self.safety_distance,
            'emergency_mode': self.emergency_mode,
            'avoidance_frequency': len(self.avoidance_history)
        }