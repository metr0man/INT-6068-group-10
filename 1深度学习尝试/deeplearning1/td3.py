import copy          # 深拷贝：用于创建和主网络完全独立的目标网络
import numpy as np   # 数值计算：处理状态/动作数组、随机采样/加噪声
import torch         # PyTorch核心：构建神经网络、张量运算、梯度计算
import torch.nn.functional as F  # 函数式接口：这里用MSE损失计算,在datamining和deep learning中都有讲
import torch.optim as optim       # 优化器：Adam更新网络参数（Adam 是强化学习最常用的优化器）

from model import Actor, Critic  # 导入Actor（策略网络）、Critic（价值网络）

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        # state_dim：状态空间维度（比如机器人的位置 + 速度，维度为 6）；
        # action_dim：动作空间维度（比如机械臂 3 个关节，维度为 3）；
        # max_action：动作的最大值（连续动作需裁剪到 [-max_action, max_action]，保证合法性）
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        # self.actor：主 Actor 网络（实时更新，输出当前状态的最优动作）；
        # self.actor_target：目标 Actor 网络（深拷贝主 Actor，参数更新滞后，用于计算目标 Q 值，避免训练震荡）。
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        # self.critic：主 Critic 网络（实时更新，评估当前状态-动作对的价值）；
        # self.critic_target：目标 Critic 网络（深拷贝主 Critic，参数更新滞后，用于计算目标 Q 值，避免训练震荡）。
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        # self.actor_optimizer：Actor 网络的优化器（Adam 优化器，学习率 3e-4）；
        # self.critic_optimizer：Critic 网络的优化器（Adam 优化器，学习率 3e-4）。
        
        self.discount = 0.99        # 折扣因子γ：越接近1越重视远期奖励
        self.tau = 0.005            # 软更新系数：目标网络参数缓慢更新
        self.policy_noise = 0.2     # 目标策略平滑的噪声标准差
        self.noise_clip = 0.5       # 噪声裁剪阈值：避免动作超出范围
        self.policy_freq = 2        # 策略更新延迟步数：Actor每2步更1次
        self.total_it = 0           # 迭代计数器：判断是否更新Actor
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))  # 状态转Tensor，适配网络输入（batch_size=1）
        action = self.actor(state).data.numpy().flatten()  # Actor输出动作，转numpy并展平
        noise = np.random.normal(0, self.policy_noise, size=action.shape)  # 生成高斯探索噪声
        return np.clip(action + noise, -self.max_action, self.max_action)  # 动作+噪声后裁剪

    def update(self, replay_buffer, batch_size=100):
        self.total_it += 1  # 每调用一次update，迭代计数器加1
        
        # 从经验回放中采样，返回的是 5 个 Tensor：当前状态、执行的动作、获得的奖励、下一个状态、是否终止（done=1 表示 episode 结束）。
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        # NOTE: 这里的 rewards/dones 形状当前是 [batch]（见 ReplayBuffer.sample），
        # 而 critic/target_q 很常见是 [batch, 1]。PyTorch 会发生广播，通常不报错但可能造成
        # 隐蔽的训练异常。更稳妥做法是把 rewards/dones reshape 成 [batch, 1] 再参与计算。
        
        # Critic网络更新
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.discount * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # 修复: 正确获取Q值
            q1, _ = self.critic(states, self.actor(states))
            actor_loss = -q1.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ReplayBuffer 保持不变

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in indices]
        
        # 将列表转换为numpy数组后再转为Tensor
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)