# deeplearning1 - TD3 Baseline

**⚠️ 历史版本** - 完整 TD3 实现基线代码

> 📌 推荐使用【🔥最新】简单环境+简单避障+简单三维仿真 中的最新优化版本

---

## 文件结构

```
deeplearning1/
├── environment.py      # 无人机环境 (DroneEnv)
├── td3.py             # TD3 算法核心
├── model.py           # Actor/Critic 网络
├── train.py           # 训练入口
├── analysis.py        # 结果分析
├── training_log.csv   # 训练日志
├── reward_curve.png   # 奖励曲线
├── td3_actor.pth      # 训练好的 Actor 网络
├── td3_critic.pth     # 训练好的 Critic 网络
└── td3_actor_*.pth    # 每 100 epoch 保存的检查点
```

---

## 快速开始

```bash
cd 1深度学习尝试/deeplearning1/
python train.py
```

---

## 算法配置

| 参数 | 值 |
|------|-----|
| 状态维度 | 6 (位置3 + 目标方向3) |
| 动作维度 | 3 (速度指令 xyz) |
| 学习率 | 3e-4 |
| 折扣因子 | 0.99 |
| 目标平滑噪声 | 0.2 |
| 策略更新频率 | 2 |
| 经验回放容量 | 100,000 |

---

## 与最新版本对比

本目录为早期实验版本。如需查看收敛效果更好、包含 3D 可视化的最新代码，请使用：

```
【🔥最新】简单环境+简单避障+简单三维仿真/
└── 【史诗级修复】优化+3D仿真新+曲线收敛+平滑噪声修正/
```
