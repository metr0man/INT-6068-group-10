# 1深度学习尝试

**⚠️ 实验性质文件夹** - TD3 深度学习基线实验代码

本目录包含早期 TD3 深度学习实验代码，供参考和对比研究。

---

## 目录结构

```
1深度学习尝试/
├── deeplearning1/      # TD3 baseline 完整版本
├── deeplearning2/       # TD3 增强版（开发中）
├── 1奖励函数设计/       # 奖励函数设计实验
└── 介绍.txt
```

---

## 子模块说明

### deeplearning1/
**TD3 Baseline** - 完整的 TD3 实现

核心文件：
- `environment.py` - 无人机环境（6维状态，3维动作）
- `td3.py` - TD3 算法实现
- `model.py` - Actor/Critic 网络架构
- `train.py` - 训练脚本
- `analysis.py` - 结果分析

**启动方式：**
```bash
cd 1深度学习尝试/deeplearning1/
python train.py
```

### deeplearning2/
**TD3 增强版** - 扩展实验版本

包含多次训练运行记录（run_001 ~ run_622+）和更详细的分析工具。

### 1奖励函数设计/
奖励函数设计实验代码与文档。

---

## 与最新版本的区别

| 特性 | 本目录 (deeplearning1/2) | 【🔥最新】简单环境+简单避障+简单三维仿真 |
|------|------------------------|----------------------------------------|
| 3D 可视化 | 无 | Plotly HTML 交互式 |
| 收敛曲线 | 基础 | 平滑噪声修正，更稳定 |
| 代码形式 | Python 脚本 | Jupyter Notebook |
| 维护状态 | ❌ 不再维护 | ✅ 当前重点开发 |

---

## 技术栈

- PyTorch 2.0.1
- Gymnasium
- NumPy / Pandas / Matplotlib
