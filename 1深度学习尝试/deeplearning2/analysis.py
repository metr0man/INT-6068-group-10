import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import seaborn as sns

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--runs_dir', required=True, help='Directory containing all run_xxx folders')
parser.add_argument('--output_dir', required=True, help='Directory to save analysis results')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 批量读取所有运行日志
all_data = []
run_dirs = [d for d in os.listdir(args.runs_dir) if d.startswith('run_') and os.path.isdir(os.path.join(args.runs_dir, d))]

for run_dir in run_dirs:
    run_path = os.path.join(args.runs_dir, run_dir)
    log_path = os.path.join(run_path, 'training_log.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df['run_id'] = run_dir
        all_data.append(df)

# 合并所有运行数据
combined_df = pd.concat(all_data, ignore_index=True)

# 计算基本统计量
mean_reward = combined_df.groupby('episode')['reward'].mean().reset_index()
std_reward = combined_df.groupby('episode')['reward'].std().reset_index()

# 数据有效性检查
print(f"调试信息: 唯一episode数量: {len(combined_df['episode'].unique())}")
print(f"调试信息: 输出目录: {args.output_dir}")
if len(combined_df['episode'].unique()) < 2:
    print("警告: 检测到数据中仅包含单个episode，无法生成有效曲线")
    # 生成单数据点占位图像
    plt.figure(figsize=(12, 8))
    episode_value = combined_df['episode'].unique()[0]
    reward_mean = combined_df['reward'].mean()
    print(f"调试信息: 绘制单episode散点图 - episode: {episode_value}, reward均值: {reward_mean}")
    plt.scatter(episode_value, reward_mean, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Insufficient Data: Only Single Episode Available')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'all_runs_reward_curve.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(mean_reward['episode'].iloc[0], mean_reward['reward'].iloc[0], color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Insufficient Data: Only Single Episode Available')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'average_reward_curve.png'))
    plt.close()
else:
    # 绘制所有运行的奖励曲线
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=combined_df, x='episode', y='reward', hue='run_id', alpha=0.3, legend=False)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curves for All Training Runs')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'all_runs_reward_curve.png'))
    plt.close()
    
    # 绘制平均奖励曲线
    plt.figure(figsize=(12, 8))
    mean_episode = mean_reward['episode'].iloc[0]
    mean_value = mean_reward['reward'].iloc[0]
    print(f"调试信息: 绘制平均奖励散点图 - episode: {mean_episode}, 均值: {mean_value}")
    plt.scatter(mean_episode, mean_value, color='blue')
    plt.plot(mean_reward['episode'], mean_reward['reward'], label='Mean Reward')
    plt.fill_between(std_reward['episode'], mean_reward['reward']-std_reward['reward'], mean_reward['reward']+std_reward['reward'], alpha=0.2, label='Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average Reward with Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'average_reward_curve.png'))
    plt.close()
# 绘制奖励分布箱线图
plt.figure(figsize=(12, 8))
sns.boxplot(data=combined_df, x='episode', y='reward')
plt.xlabel('Episode')
plt.ylabel('Reward Distribution')
plt.title('Reward Distribution Across Episodes')
plt.grid(True)
plt.savefig(os.path.join(args.output_dir, 'reward_boxplot.png'))
plt.close()

# 输出总体统计信息
print('所有运行的总体统计信息：')
print(combined_df[['reward', 'steps']].describe())

# 保存合并后的数据
combined_df.to_csv(os.path.join(args.output_dir, 'combined_training_log.csv'), index=False)