import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 基本配置
# =========================================================
ROOT_DIR = "ablation_results_full"   # 你的实验输出目录
MAX_EPISODES = 2500                  # 只分析前 2500 个 episode
SMOOTH_WINDOW = 50                   # 平滑窗口
SEEDS = [0, 1, 2]

# 只分析这 3 组无 curriculum 的噪声实验
EXPERIMENTS = {
    "baseline_no_curriculum_decay": "decay",
    "no_curriculum_small": "small",
    "no_curriculum_large": "large noise",
}

# 画图顺序
PLOT_ORDER = ["large noise", "decay", "small"]


# =========================================================
# 1. 读取训练日志
# =========================================================
all_logs = []

for exp_name, label in EXPERIMENTS.items():
    for seed in SEEDS:
        log_path = os.path.join(ROOT_DIR, f"{exp_name}_seed{seed}", "training_log.csv")
        if not os.path.exists(log_path):
            print(f"[WARNING] 文件不存在: {log_path}")
            continue

        df = pd.read_csv(log_path).copy()

        # 兼容 episode 从 0 或 1 开始的情况
        # 这里只保留前 2500 个 episode
        df = df[df["episode"] < MAX_EPISODES].copy()

        # 补充标识列
        df["experiment_name"] = exp_name
        df["exp_label"] = label
        df["seed"] = seed

        # 按 episode 排序，避免 rolling 出问题
        df = df.sort_values("episode").reset_index(drop=True)

        all_logs.append(df)

if len(all_logs) == 0:
    raise ValueError("没有读到任何 training_log.csv，请检查 ROOT_DIR 路径和文件结构。")

train_df = pd.concat(all_logs, ignore_index=True)

print("读取完成。")
print(train_df.head())
print("\n实验统计：")
print(train_df.groupby(["exp_label", "seed"]).size())


# =========================================================
# 2. 定义指标计算函数
#    对应第二张图中的三个参数
# =========================================================
def compute_metrics_one_seed(df_one_seed, smooth_window=50, late_ratio=0.2):
    """
    输入: 单个实验、单个 seed 的训练日志
    输出:
        Mean Absolute Reward Change
        Reward Residual Std.
        Late-stage Reward Std.
    """
    df_one_seed = df_one_seed.sort_values("episode").reset_index(drop=True)

    reward = df_one_seed["reward"].to_numpy(dtype=float)

    # 平滑 reward
    reward_smooth = (
        pd.Series(reward)
        .rolling(window=smooth_window, min_periods=1)
        .mean()
        .to_numpy()
    )

    # 1) Mean Absolute Reward Change
    # 用“相邻 episode 的 reward 变化绝对值均值”
    if len(reward) >= 2:
        mean_abs_reward_change = np.mean(np.abs(np.diff(reward)))
    else:
        mean_abs_reward_change = np.nan

    # 2) Reward Residual Std.
    # 原始 reward 相对于平滑 reward 的残差标准差
    reward_residual_std = np.std(reward - reward_smooth)

    # 3) Late-stage Reward Std.
    # 后 20% episode 的 reward 标准差（2500 ep -> 后 500 ep）
    late_n = max(1, int(len(reward) * late_ratio))
    late_stage_reward_std = np.std(reward[-late_n:])

    return {
        "Mean Absolute Reward Change": mean_abs_reward_change,
        "Reward Residual Std.": reward_residual_std,
        "Late-stage Reward Std.": late_stage_reward_std,
    }


# =========================================================
# 3. 计算表格指标（先按 seed 算，再对 seed 取平均）
# =========================================================
metric_rows = []

for (exp_label, seed), g in train_df.groupby(["exp_label", "seed"]):
    metrics = compute_metrics_one_seed(
        g,
        smooth_window=SMOOTH_WINDOW,
        late_ratio=0.2
    )
    metric_rows.append({
        "Experiment": exp_label,
        "Seed": seed,
        **metrics
    })

metric_seed_df = pd.DataFrame(metric_rows)

metric_table = (
    metric_seed_df
    .groupby("Experiment", as_index=False)[
        ["Mean Absolute Reward Change", "Reward Residual Std.", "Late-stage Reward Std."]
    ]
    .mean()
)

# 按指定顺序排序
metric_table["Experiment"] = pd.Categorical(
    metric_table["Experiment"],
    categories=PLOT_ORDER,
    ordered=True
)
metric_table = metric_table.sort_values("Experiment").reset_index(drop=True)

# 保留 3 位小数
metric_table_rounded = metric_table.copy()
for c in ["Mean Absolute Reward Change", "Reward Residual Std.", "Late-stage Reward Std."]:
    metric_table_rounded[c] = metric_table_rounded[c].round(3)

print("\n=== 指标表 ===")
print(metric_table_rounded)

metric_table_rounded.to_csv("noise_outcome_metrics_2500ep.csv", index=False)


# =========================================================
# 4. 生成用于画曲线的数据
# =========================================================
def build_smoothed_curve(df, value_col, smooth_window=50):
    """
    先对每个 seed 单独 rolling smooth，再对不同 seed 取 mean/std
    """
    smoothed_list = []

    for (exp_label, seed), g in df.groupby(["exp_label", "seed"]):
        g = g.sort_values("episode").copy()
        g["smoothed"] = g[value_col].rolling(window=smooth_window, min_periods=1).mean()
        smoothed_list.append(g[["episode", "exp_label", "seed", "smoothed"]])

    smoothed_df = pd.concat(smoothed_list, ignore_index=True)

    curve_df = (
        smoothed_df
        .groupby(["exp_label", "episode"])["smoothed"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "y_mean", "std": "y_std"})
    )

    curve_df["y_std"] = curve_df["y_std"].fillna(0.0)
    return curve_df


reward_curve_df = build_smoothed_curve(train_df, "reward", smooth_window=SMOOTH_WINDOW)
success_curve_df = build_smoothed_curve(train_df, "train_success", smooth_window=SMOOTH_WINDOW)


# =========================================================
# 5. 颜色映射
#    尽量接近你示例图：
#    large noise = 红色
#    decay = 蓝色
#    small = 金黄色
# =========================================================
COLOR_MAP = {
    "large noise": "red",
    "decay": "blue",
    "small": "#E6C200"
}


# =========================================================
# 6. 画 Smoothed Reward Curves
# =========================================================
plt.figure(figsize=(10, 6))

for exp_label in PLOT_ORDER:
    g = reward_curve_df[reward_curve_df["exp_label"] == exp_label].sort_values("episode")
    if len(g) == 0:
        continue

    x = g["episode"].to_numpy()
    y = g["y_mean"].to_numpy()
    s = g["y_std"].to_numpy()
    color = COLOR_MAP[exp_label]

    plt.plot(x, y, label=exp_label, color=color, linewidth=2)
    plt.fill_between(x, y - s, y + s, color=color, alpha=0.15)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"Smoothed Reward Curves (First {MAX_EPISODES} Episodes, window={SMOOTH_WINDOW})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("smoothed_reward_curves_2500ep.png", dpi=200)
plt.show()


# =========================================================
# 7. 画 Smoothed Success Rate Curves
# =========================================================
plt.figure(figsize=(10, 6))

for exp_label in PLOT_ORDER:
    g = success_curve_df[success_curve_df["exp_label"] == exp_label].sort_values("episode")
    if len(g) == 0:
        continue

    x = g["episode"].to_numpy()
    y = g["y_mean"].to_numpy()
    s = g["y_std"].to_numpy()
    color = COLOR_MAP[exp_label]

    plt.plot(x, y, label=exp_label, color=color, linewidth=2)
    plt.fill_between(x, y - s, y + s, color=color, alpha=0.15)

plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title(f"Smoothed Success Rate Curves (First {MAX_EPISODES} Episodes, window={SMOOTH_WINDOW})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("smoothed_success_curves_2500ep.png", dpi=200)
plt.show()


# =========================================================
# 8. 画表格
# =========================================================
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

table_data = metric_table_rounded.values.tolist()
col_labels = metric_table_rounded.columns.tolist()

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 2.0)

# 表头加粗
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#F2F2F2")

plt.title("Outcomes Metrics Table (First 2500 Episodes)", fontsize=16, weight="bold", pad=18)
plt.tight_layout()
plt.savefig("outcomes_metrics_table_2500ep.png", dpi=200, bbox_inches="tight")
plt.show()


# =========================================================
# 9. 导出一个汇总图（可选）
# =========================================================
fig = plt.figure(figsize=(16, 10))

# 左上：表格
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.axis("off")
tbl = ax1.table(
    cellText=metric_table_rounded.values.tolist(),
    colLabels=metric_table_rounded.columns.tolist(),
    loc="center",
    cellLoc="center",
    colLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 2.0)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#F2F2F2")
ax1.set_title("Outcomes", fontsize=24, weight="bold", pad=10)

# 右上：reward curve
ax2 = plt.subplot2grid((2, 2), (0, 1))
for exp_label in PLOT_ORDER:
    g = reward_curve_df[reward_curve_df["exp_label"] == exp_label].sort_values("episode")
    if len(g) == 0:
        continue
    x = g["episode"].to_numpy()
    y = g["y_mean"].to_numpy()
    s = g["y_std"].to_numpy()
    color = COLOR_MAP[exp_label]
    ax2.plot(x, y, label=exp_label, color=color, linewidth=2)
    ax2.fill_between(x, y - s, y + s, color=color, alpha=0.15)
ax2.set_title(f"Smoothed Reward Curves (First {MAX_EPISODES} Episodes, window={SMOOTH_WINDOW})")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Reward")
ax2.grid(True, alpha=0.3)
ax2.legend()

# 右下：success curve
ax3 = plt.subplot2grid((2, 2), (1, 1))
for exp_label in PLOT_ORDER:
    g = success_curve_df[success_curve_df["exp_label"] == exp_label].sort_values("episode")
    if len(g) == 0:
        continue
    x = g["episode"].to_numpy()
    y = g["y_mean"].to_numpy()
    s = g["y_std"].to_numpy()
    color = COLOR_MAP[exp_label]
    ax3.plot(x, y, label=exp_label, color=color, linewidth=2)
    ax3.fill_between(x, y - s, y + s, color=color, alpha=0.15)
ax3.set_title(f"Smoothed Success Rate Curves (First {MAX_EPISODES} Episodes, window={SMOOTH_WINDOW})")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Success Rate")
ax3.grid(True, alpha=0.3)
ax3.legend()

# 左下留白，可自行加 bullet points
ax4 = plt.subplot2grid((2, 2), (1, 0))
ax4.axis("off")

summary_text = (
    "• All three noise settings eventually improve performance.\n"
    "• Small noise tends to learn more slowly at the beginning.\n"
    "• Large noise and decaying noise usually converge faster early on.\n"
    "• Decaying noise is often more stable in the middle and later stages.\n"
    "• Smaller oscillation metrics indicate better stability."
)
ax4.text(0.02, 0.95, summary_text, va="top", fontsize=14, weight="bold", linespacing=1.8)

plt.tight_layout()
plt.savefig("outcomes_summary_2500ep.png", dpi=220, bbox_inches="tight")
plt.show()

print("\n已保存文件：")
print("- noise_outcome_metrics_2500ep.csv")
print("- smoothed_reward_curves_2500ep.png")
print("- smoothed_success_curves_2500ep.png")
print("- outcomes_metrics_table_2500ep.png")
print("- outcomes_summary_2500ep.png")