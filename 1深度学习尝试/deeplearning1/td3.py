import copy
import numpy as np
import torch
import torch.nn.functional as F  # 添加导入
import torch.optim as optim

from model import Actor, Critic

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0  # 初始化迭代计数器

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).data.numpy().flatten()
        noise = np.random.normal(0, self.policy_noise, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, replay_buffer, batch_size=100):
        self.total_it += 1
        
        # 从经验回放中采样
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