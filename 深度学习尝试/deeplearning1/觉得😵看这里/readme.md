          
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

#### **第2步：进阶到deeplearning2**
**改进**：支持多轮实验和更系统的超参数调优

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

#### **第4步：多智能体仿真**
**高级**：多无人机协调，最复杂的场景

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

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 如何贡献
1. Fork本项目
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个Pull Request

### 贡献内容建议
- 🐛 修复bug
- 📚 改进文档
- ✨ 添加新特性
- 🧪 分享实验结果
- 💡 提出改进建议

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢PyTorch团队提供的优秀深度学习框架
- 感谢Stable Baselines3团队提供的强化学习算法实现
- 感谢PyBullet团队提供的物理仿真环境

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 [GitHub Issue](https://github.com/your-username/drone-rl-tutorial/issues)
- 发送邮件至：your-email@example.com

## ⭐ Star历史

如果这个项目对你有帮助，请给它一个Star！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/drone-rl-tutorial&type=Date)](https://star-history.com/#your-username/drone-rl-tutorial&Date)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 如何贡献
1. Fork本项目
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个Pull Request

### 贡献内容建议
- 🐛 修复bug
- 📚 改进文档
- ✨ 添加新特性
- 🧪 分享实验结果
- 💡 提出改进建议

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢PyTorch团队提供的优秀深度学习框架
- 感谢Stable Baselines3团队提供的强化学习算法实现
- 感谢PyBullet团队提供的物理仿真环境

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 [GitHub Issue](https://github.com/your-username/drone-rl-tutorial/issues)
- 发送邮件至：your-email@example.com

## ⭐ Star历史

如果这个项目对你有帮助，请给它一个Star！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/drone-rl-tutorial&type=Date)](https://star-history.com/#your-username/drone-rl-tutorial&Date)