import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
log_df = pd.read_csv('training_log.csv')

# 绘制奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(log_df['episode'], log_df['reward'], label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.legend()
plt.grid(True)
plt.savefig('reward_curve.png')
plt.show()

# 输出基础统计信息
print('训练结果统计：')
print(log_df[['reward', 'steps']].describe())