# 导入必要的模块
from environment import DroneEnv  # 无人机仿真环境
from td3 import TD3, ReplayBuffer  # 强化学习算法及经验回放池
import numpy as np
import torch  # 深度学习框架
import csv  # 新增导入

# 初始化无人机训练环境
env = DroneEnv()
state_dim = env.state_dim  # 状态空间维度（位置+目标方向）
action_dim = env.action_dim  # 动作空间维度（三维速度）
max_action = 1.0  # 动作取值范围[-1,1]

# 创建经验回放池（存储转移样本）
replay_buffer = ReplayBuffer(capacity=100000)

# 初始化TD3算法代理
agent = TD3(state_dim, action_dim, max_action)
max_episodes = 1000  # 最大训练回合数

# 开始训练循环
# 在训练循环前初始化记录文件
with open('training_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'reward', 'steps'])

for episode in range(max_episodes):
    # 重置环境获取初始状态
    state = env.reset()
    episode_reward = 0  # 本回合累计奖励
    
    # 单回合最大步长控制
    for t in range(200):
        # 使用策略网络选择动作（含探索噪声）
        action = agent.select_action(state)
        
        # 执行动作并获取环境反馈
        next_state, reward, done, _ = env.step(action)
        
        # 存储转移样本到经验池
        replay_buffer.add(state, action, reward, next_state, done)
        
        # 当经验池足够时更新网络参数
        if len(replay_buffer) > 1000:
            agent.update(replay_buffer)
        
        # 状态转移并累计奖励
        state = next_state
        episode_reward += reward
        
        # 提前终止条件检查
        if done:
            break
    
    # 输出训练进度
    print(f"Episode {episode} | Reward: {episode_reward:.2f}")
    
    # 在回合结束后记录数据
    with open('training_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, episode_reward, t+1])
    
    # 每100回合保存一次中间结果
    if episode % 100 == 0:
        torch.save(agent.actor.state_dict(), f"td3_actor_{episode}.pth")
        torch.save(agent.critic.state_dict(), f"td3_critic_{episode}.pth")
# 训练完成后保存模型参数
# Actor网络：策略网络
# Critic网络：价值评估网络
torch.save(agent.actor.state_dict(), "td3_actor.pth")
torch.save(agent.critic.state_dict(), "td3_critic.pth")