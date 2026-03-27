# deeplearning2 - TD3 增强版

**⚠️ 实验性质** - TD3 增强版本，侧重多次训练对比实验

> 📌 推荐使用【🔥最新】简单环境+简单避障+简单三维仿真 中的最新优化版本

---

## 文件结构

```
deeplearning2/
├── environment.py      # 无人机环境
├── td3.py             # TD3 算法
├── model.py           # 网络架构
├── train.py           # 训练入口
├── new.py             # 增强版训练脚本
├── analysis.py        # 结果分析
├── run_001 ~ run_622  # 多次训练运行记录
└── pst/               # 训练结果
```

---

## 特点

- 包含 **600+ 次独立训练运行**（run_001 ~ run_622）用于对比分析
- 详细的训练日志与结果存档
- 增强版训练脚本 `new.py`

---

## 与 deeplearning1 对比

| 特性 | deeplearning1 | deeplearning2 |
|------|--------------|---------------|
| 训练次数 | 1次 | 600+次 |
| 目的 | 基线验证 | 统计对比 |
| 状态 | 完整保存 | 历史存档 |

---

## 快速开始

```bash
cd 1深度学习尝试/deeplearning2/
python train.py
```
