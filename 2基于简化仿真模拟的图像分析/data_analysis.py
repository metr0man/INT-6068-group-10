import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_training_data(file_path):
    """加载训练日志数据"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    return pd.read_csv(file_path)

def plot_line_chart(data, x_col, y_col, title, save_path=None):
    """绘制折线图"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x_col, y=y_col)
    plt.title(title, fontsize=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiple_lines(data, x_col, y_cols, title, save_path=None):
    """绘制多折线图"""
    plt.figure(figsize=(12, 6))
    for y_col in y_cols:
        sns.lineplot(data=data, x=x_col, y=y_col, label=y_col)
    plt.title(title, fontsize=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel("数值", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(data, x_col, y_col, title, save_path=None):
    """绘制柱状图"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x=x_col, y=y_col)
    plt.title(title, fontsize=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(alpha=0.3, axis='y')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_chart(data, x_col, y_col, title, save_path=None):
    """绘制散点图"""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col)
    plt.title(title, fontsize=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap(data, title, save_path=None):
    """绘制热力图"""
    plt.figure(figsize=(10, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_training_logs(file_path, output_dir="visualizations"):
    """分析训练日志并生成多种可视化图表"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df = load_training_data(file_path)
    if df is None:
        return
    
    # 获取文件名作为前缀
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    
    # 假设CSV文件包含常见的训练指标列名
    common_columns = {"epoch": ["epoch", "Epoch", "轮次"],
                     "loss": ["loss", " Loss", "损失"],
                     "accuracy": ["acc", "accuracy", "准确率"],
                     "val_loss": ["val_loss", "validation_loss", "验证损失"],
                     "val_acc": ["val_acc", "validation_accuracy", "验证准确率"]}
    
    # 映射列名到标准名称
    column_mapping = {}
    for std_col, possible_names in common_columns.items():
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                column_mapping[std_col] = col
                break
    
    # 生成折线图 - 损失
    if "epoch" in column_mapping and "loss" in column_mapping:
        plot_line_chart(
            df, 
            column_mapping["epoch"], 
            column_mapping["loss"], 
            f"{file_prefix} - 训练损失随轮次变化",
            os.path.join(output_dir, f"{file_prefix}_loss_line.png")
        )
    
    # 生成多折线图 - 训练与验证损失
    if "epoch" in column_mapping and "loss" in column_mapping and "val_loss" in column_mapping:
        plot_multiple_lines(
            df, 
            column_mapping["epoch"], 
            [column_mapping["loss"], column_mapping["val_loss"]], 
            f"{file_prefix} - 训练与验证损失对比",
            os.path.join(output_dir, f"{file_prefix}_loss_comparison.png")
        )
    
    # 生成折线图 - 准确率
    if "epoch" in column_mapping and "accuracy" in column_mapping:
        plot_line_chart(
            df, 
            column_mapping["epoch"], 
            column_mapping["accuracy"], 
            f"{file_prefix} - 准确率随轮次变化",
            os.path.join(output_dir, f"{file_prefix}_accuracy_line.png")
        )
    
    # 生成多折线图 - 训练与验证准确率
    if "epoch" in column_mapping and "accuracy" in column_mapping and "val_acc" in column_mapping:
        plot_multiple_lines(
            df, 
            column_mapping["epoch"], 
            [column_mapping["accuracy"], column_mapping["val_acc"]], 
            f"{file_prefix} - 训练与验证准确率对比",
            os.path.join(output_dir, f"{file_prefix}_accuracy_comparison.png")
        )
    
    # 生成柱状图 - 准确率
    if "epoch" in column_mapping and "accuracy" in column_mapping:
        # 只取前20个轮次的数据，避免图表过于拥挤
        plot_data = df[[column_mapping["epoch"], column_mapping["accuracy"]]].head(20)
        plot_bar_chart(
            plot_data, 
            column_mapping["epoch"], 
            column_mapping["accuracy"], 
            f"{file_prefix} - 前20轮准确率柱状图",
            os.path.join(output_dir, f"{file_prefix}_accuracy_bar.png")
        )
    
    # 生成散点图 - 损失与准确率关系
    if "loss" in column_mapping and "accuracy" in column_mapping:
        plot_scatter_chart(
            df, 
            column_mapping["loss"], 
            column_mapping["accuracy"], 
            f"{file_prefix} - 损失与准确率关系散点图",
            os.path.join(output_dir, f"{file_prefix}_loss_vs_accuracy_scatter.png")
        )
    
    # 生成热力图 - 特征相关性
    plot_heatmap(
        df, 
        f"{file_prefix} - 特征相关性热力图",
        os.path.join(output_dir, f"{file_prefix}_correlation_heatmap.png")
    )
    
    # 生成折线图 - 奖励值
    if "epoch" in column_mapping and "reward" in column_mapping:
        plot_line_chart(
            df, 
            column_mapping["epoch"], 
            column_mapping["reward"], 
            f"{file_prefix} - 奖励值随轮次变化",
            os.path.join(output_dir, f"{file_prefix}_reward_line.png")
        )
    
    # 生成折线图 - 步数
    if "epoch" in column_mapping and "steps" in column_mapping:
        plot_line_chart(
            df, 
            column_mapping["epoch"], 
            column_mapping["steps"], 
            f"{file_prefix} - 步数随轮次变化",
            os.path.join(output_dir, f"{file_prefix}_steps_line.png")
        )
    
    # 生成多折线图 - 奖励与步数对比
    if "epoch" in column_mapping and "reward" in column_mapping and "steps" in column_mapping:
        plot_multiple_lines(
            df, 
            column_mapping["epoch"], 
            [column_mapping["reward"], column_mapping["steps"]], 
            f"{file_prefix} - 奖励与步数对比",
            os.path.join(output_dir, f"{file_prefix}_reward_steps_comparison.png")
        )
    
    print(f"可视化完成！图表已保存至 {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 分析目录下所有CSV文件
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
    
    if not csv_files:
        print("未找到CSV文件")
    else:
        for csv_file in csv_files:
            print(f"正在分析: {csv_file}")
            analyze_training_logs(csv_file)
        print("所有文件分析完成！")