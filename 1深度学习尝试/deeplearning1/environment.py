import numpy as np

class DroneEnv:
    """
    无人机路径规划仿真环境
    状态空间：6维 [x,y,z, dx,dy,dz] (当前位置 + 目标方向向量)
    动作空间：3维 [vx,vy,vz] (三维速度向量)
    """
    
    def __init__(self):
        # 环境维度参数
        self.action_dim = 3  # 动作空间维度（三维速度）
        self.state_dim = 6   # 状态空间维度（3D位置 + 3D目标方向）
        self.step_count = 0  # 当前步数计数器
        self.max_step = 200  # 单回合最大步数

    def reset(self):
        """
        重置环境到初始状态
        返回：初始状态向量 (numpy数组)
        """
        self.step_count = 0
        self.position = np.zeros(3)  # 初始位置设为原点
        self.target = np.array([10, 10, 5])  # 固定目标位置
        return self._get_state()

    def _get_state(self):
        """
        生成当前状态向量
        返回：拼接后的状态向量 [x,y,z, dx,dy,dz]
        """
        # NOTE: 这里的 direction 未做归一化/裁剪，数值尺度会随“离目标远近”变化，
        # 可能导致网络输入分布漂移、训练不稳定。常见做法：direction 归一化 + 单独提供距离标量。
        direction = self.target - self.position  # 计算目标方向向量
        return np.concatenate([self.position, direction])

    def step(self, action):
        """
        执行动作并返回环境反馈
        参数：
            action (numpy数组): 三维速度向量 [-1,1] 范围
        返回：
            state: 新状态
            reward: 即时奖励
            done: 是否终止
            info: 调试信息
        """
        # 物理模拟（简化版）
        # NOTE: 环境内部未对 action 做 clip。当前 TD3.select_action() 会 clip 到 [-1, 1]，
        # 但若未来更换策略/调试时传入越界动作，可能导致状态更新与越界惩罚逻辑异常。
        self.position += action * 0.1  # 按动作更新位置
        self.step_count += 1
        
        # 奖励计算
        distance = np.linalg.norm(self.target - self.position)
        reward = -distance * 0.1  # 基础奖励（距离越近奖励越高）
        
        # 终止条件判断
        done = False
        # 边界碰撞检测（位置超出±15米）
        if np.any(np.abs(self.position) > 15):
            reward -= 5  # 碰撞惩罚
            done = True
        else:
            # 步数超限或到达目标（暂未实现到达检测）
            done = self.step_count >= self.max_step
        
        return self._get_state(), reward, done, {}