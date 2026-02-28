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