import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建可视化结果目录
output_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件
csv_path = 'training_log_20250723_103839.csv'
df = pd.read_csv(csv_path)

# 生成奖励值折线图
plt.figure(figsize=(12, 6))
plt.plot(df['episode'], df['reward'], marker='o', markersize=3, linestyle='-', color='b')
plt.title('奖励值随轮次变化趋势')
plt.xlabel('轮次 (episode)')
plt.ylabel('奖励值 (reward)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
reward_chart_path = os.path.join(output_dir, 'reward_trend.png')
plt.savefig(reward_chart_path, dpi=300)
plt.close()

# 生成步数折线图
plt.figure(figsize=(12, 6))
plt.plot(df['episode'], df['steps'], marker='s', markersize=3, linestyle='-', color='g')
plt.title('步数随轮次变化趋势')
plt.xlabel('轮次 (episode)')
plt.ylabel('步数 (steps)')
steps_min = df['steps'].min()
steps_max = df['steps'].max()
plt.ylim(steps_min - 5, steps_max + 5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
steps_chart_path = os.path.join(output_dir, 'steps_trend.png')
plt.savefig(steps_chart_path, dpi=300)
plt.close()

# 生成奖励值与步数对比图
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(df['episode'], df['reward'], marker='o', markersize=3, linestyle='-', color='b', label='奖励值')
ax2.plot(df['episode'], df['steps'], marker='s', markersize=3, linestyle='-', color='g', label='步数')

ax1.set_title('奖励值与步数随轮次变化对比')
ax1.set_xlabel('轮次 (episode)')
ax1.set_ylabel('奖励值 (reward)', color='b')
ax2.set_ylabel('步数 (steps)', color='g')
steps_min = df['steps'].min()
steps_max = df['steps'].max()
ax2.set_ylim(steps_min - 5, steps_max + 5)
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='g')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.legend()
comparison_chart_path = os.path.join(output_dir, 'reward_vs_steps.png')
plt.savefig(comparison_chart_path, dpi=300)
plt.close()

print(f"折线统计图生成完成！\n")
print(f"奖励值趋势图: {reward_chart_path}")
print(f"步数趋势图: {steps_chart_path}")
print(f"奖励值与步数对比图: {comparison_chart_path}")