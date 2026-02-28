
# 🤖 Drone Reinforcement Learning Tutorial

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

一个面向强化学习新手的无人机路径规划项目，从基础算法到物理仿真，循序渐进地掌握深度强化学习技术。

## 📖 项目简介

本项目包含四个递进阶段的无人机强化学习实验环境：
- **阶段1**: 简化环境 + TD3算法（最适合入门）
- **阶段2**: 扩展训练 + 超参数优化
- **阶段3**: PyBullet物理仿真 + PPO算法
- **阶段4**: 多智能体协调（多无人机协同）
### 💡 为什么要在无人机自主路径规划中使用强化学习？

对于刚接触路径规划的同学来说，你可能会问：__我们已经有了 A_、RRT、动态规划等经典的传统算法，为什么还要费力使用强化学习（RL）来让无人机“自主学习”飞行呢？_*

这是因为在真实的复杂三维世界中，传统算法面临着许多难以逾越的瓶颈，而强化学习恰好为突破这些瓶颈提供了全新的思路：

#### 1. 传统算法的局限性（行业痛点）

- **难以应对动态环境**：现有的传统 A* / RRT 等算法大多只适用于静态环境，当面对复杂的动态障碍物时，它们的适应性很差 。
    
- **极度依赖已知地图**：传统方法往往需要预先知道全局地图信息，无法直接处理来自传感器的实时感知数据，这大大降低了系统在未知环境中的灵活性 。
    
- **动作空间的局限与高能耗**：传统算法常将无人机连续可变的动作（如滚转、俯仰、偏航和推力）粗暴地量化为有限的离散指令 。这种做法不仅会引入 22° 的航向量化误差、导致锯齿状的飞行轨迹，还会产生额外的加减速，使得实测能耗增加约 7%，到达时间延长 18% 。这很难满足无人机在密集障碍物中平滑、精准且低能耗的连续飞控需求 。
    
- **缺乏多机协同与泛化能力**：在多无人机交互、灾难救援等复杂场景下，传统算法泛化能力不足 。此外，在室内高密度障碍物环境中，传统算法极易陷入局部死区，导致算法死锁，无法有效规划出航迹 。
    

#### 2. 强化学习的破局优势（技术革新）

- **实时感知驱动与动态避障**：强化学习模型不需要精准建图，而是可以直接处理传感器数据 。例如，无人机可以在三维空间中向周边（如 26 个方向）发射射线来感知障碍物距离，从而在密集的动态障碍物场景中具备极强的实时自适应和避障能力 。
    
- **多目标协同优化**：强化学习提供了一个高度可扩展的奖励函数框架 。我们可以设计多维度的奖励（如目标导向奖励、主动避障奖励、碰撞惩罚和边界违规惩罚等），引导无人机在一次训练中学会平衡和处理多个复杂的飞行任务 。
    
- **支持连续与平滑控制**：结合三维物理仿真，强化学习突破了离散动作的限制，可以直接输出连续控制指令，更加符合无人机真实的物理动力学特性，实现平稳高效的飞行 。
    
- **强泛化能力与多智能体协同**：强化学习能很好地支持多智能体（MARL）协同路径规划，满足群体作业的需求，在复杂多变的场景中具有传统算法不可替代的优势 

希望这段介绍能帮助你理解本项目的设计初衷！如果你也对赋予无人机真正的“自主决策”能力感兴趣，请继续往下阅读我们的代码结构和快速开始指南。
## 🚀 快速开始

### 📋 环境要求
```bash
# 基础依赖
pip install numpy torch matplotlib

# 阶段3需要额外安装
pip install pybullet gymnasium stable-baselines3
```

### 📁 项目结构
```
drone-rl-tutorial/
├── deeplearning1/          # 阶段1：基础TD3算法
│   ├── environment.py      # 简化环境
│   ├── td3.py             # TD3算法实现
│   ├── train.py           # 训练脚本
│   └── model.py           # 神经网络模型
├── deeplearning2/          # 阶段2：扩展训练
├── pybullet-simulation/    # 阶段3：物理仿真
└── multi-agent-simulation/ # 阶段4：多智能体
```

# 🤖 Drone Reinforcement Learning Tutorial

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个面向强化学习新手的无人机路径规划项目，从基础算法到物理仿真，循序渐进地掌握深度强化学习技术。

## 📖 项目简介

本项目包含四个递进阶段的无人机强化学习实验环境：
- **阶段1**: 简化环境 + TD3算法（最适合入门）
- **阶段2**: 扩展训练 + 超参数优化
- **阶段3**: PyBullet物理仿真 + PPO算法
- **阶段4**: 多智能体协调（多无人机协同）

## 🚀 快速开始

### 📋 环境要求
```bash
# 基础依赖
pip install numpy torch matplotlib

# 阶段3需要额外安装
pip install pybullet gymnasium stable-baselines3
```

### 📁 项目结构
```
drone-rl-tutorial/
├── deeplearning1/          # 阶段1：基础TD3算法
│   ├── environment.py      # 简化环境
│   ├── td3.py             # TD3算法实现
│   ├── train.py           # 训练脚本
│   └── model.py           # 神经网络模型
├── deeplearning2/          # 阶段2：扩展训练
├── pybullet-simulation/    # 阶段3：物理仿真
└── multi-agent-simulation/ # 阶段4：多智能体
```

## 🎯 项目演进逻辑与新手入门指南

作为强化学习新手，我来为你梳理这个项目的完整演进路径和启动顺序：

### 📋 **项目演进时间线**

```
阶段1: 深度学习实验 (基础算法研究)
    ↓
阶段2: PyBullet物理仿真 (单智能体真实环境)  
    ↓
阶段3: 简化多智能体仿真 (多无人机协调)
    ↓
阶段4: 奖励函数优化 (当前状态)
```

### 🚀 **新手启动顺序（推荐）**

#### **第1步：从深度学习实验开始** ⭐ **最适合新手**
**原因**：环境简单，算法清晰，易于理解和调试

deeplearning1是针对静态环境下的简化环境的训练尝试
deeplearning2是进一步的修改，增大训练量等，还未完成
##### [点击链接跳转Colab，逐步运行即可预览deeplearning1，省去环境部署（但是跑完要很久）](https://colab.research.google.com/drive/1do7k5KFhJ-CpMvLQHdmNUIqYWpbaNtfp#scrollTo=ZqEK_3Anj6BC)
本地运行，在终端输入：
```bash
# 进入最简单的实验环境
cd drone-rl-tutorial/deeplearning1

# 运行基础训练
python train.py
```

**关键参数调优位置**：
- **训练回合数**：`max_episodes = 1000`（train.py第18行）
- **每回合步数**：`for t in range(200)`（train.py第25行）
- **经验池大小**：`ReplayBuffer(capacity=100000)`（train.py第16行）

#### **第2步：进阶到deeplearning2（这个我没完成，模型表现较差）**
**改进**：支持多轮实验和更系统的超参数调优，支持动态环境（马路，电动车道这种才算动态环境，建筑不算动态环境）

```bash
cd drone-rl-tutorial/deeplearning2

# 运行带输出目录的实验
python train.py --output_dir pst/run_001
```

#### **第3步：PyBullet物理仿真**
**进阶**：真实物理环境，更复杂的状态空间

```bash
cd drone-rl-tutorial/pybullet-simulation

# 路径规划训练
python train_rl.py --env path --timesteps 100000

# 动力学控制训练  
python train_rl.py --env dynamics --timesteps 100000
```

#### ~~**第4步：多智能体仿真**~~
~~**高级**：多无人机协调，最复杂的场景~~
放在future development
```bash
cd drone-rl-tutorial/multi-agent-simulation

# 运行多智能体环境
python multi_agent_env.py
```

### 🔧 **关键调参位置速查表**

| 参数类型 | 文件位置 | 推荐新手值 | 说明 |
|---------|---------|------------|------|
| **学习率** | td3.py第18-19行 | `lr=3e-4` | Adam优化器学习率 |
| **折扣因子** | td3.py第24行 | `discount=0.99` | 未来奖励衰减系数 |
| **噪声参数** | td3.py第25-26行 | `policy_noise=0.2` | 探索噪声强度 |
| **网络更新频率** | td3.py第28行 | `policy_freq=2` | 策略网络更新间隔 |
| **奖励缩放** | environment.py第54行 | `reward = -distance * 0.1` | 距离奖励系数 |
| **碰撞惩罚** | environment.py第58行 | `reward -= 5` | 边界碰撞惩罚 |

### 📊 **训练监控指标**

1. **实时指标**：每回合奖励（控制台输出）
2. **训练日志**：`training_log.csv`（回合奖励、步数）
3. **模型保存**：每100回合保存中间模型（`td3_actor_XXX.pth`）

### 🎯 **新手建议的实验顺序**

```python
# 实验1：基础训练（默认参数）
python train.py

# 实验2：减少训练回合（快速验证）
# 修改 max_episodes = 200
python train.py

# 实验3：调整探索噪声（观察收敛速度）
# 修改 policy_noise = 0.1 或 0.3
python train.py

# 实验4：调整奖励缩放（观察策略变化）
# 修改 reward = -distance * 0.05 或 0.2
python train.py
```

### ⚠️ **注意事项**

1. **从简单开始**：先掌握deeplearning1，再逐步进阶
2. **小步试错**：每次只改一个参数，观察效果
3. **记录实验**：保存不同参数组合的训练结果
4. **监控收敛**：关注奖励曲线的变化趋势

这样的渐进式学习路径能让你逐步理解强化学习的核心概念，同时避免被复杂的环境和算法淹没。
