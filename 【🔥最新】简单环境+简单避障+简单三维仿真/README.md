# 【🔥最新】简单环境+简单避障+简单三维仿真

**当前开发重点** - TD3 强化学习训练与 3D 可视化

本目录包含最新的 TD3 深度学习实验代码，支持 3D 无人机轨迹可视化与收敛曲线分析。

---

## 目录结构

```
【🔥最新】简单环境+简单避障+简单三维仿真/
├── 【史诗级修复】优化+3D仿真新+曲线收敛+平滑噪声修正/   ⭐ 推荐使用
│   ├── 【史诗级重制】优化+3D仿真新+曲线收敛+平滑噪声修正.ipynb
│   ├── drone_3d_flight_Plz_Converge.html         # 3D 轨迹可视化
│   ├── analysis_results_Plz_Converge/           # 训练结果分析
│   └── results_drone_Plz_Converge/              # 训练日志
│
├── 优化+3D仿真新+TD3目标平滑噪声修正/             # TD3 目标平滑优化版本
│   ├── 优化+3D仿真新+TD3目标平滑噪声修正+画图修正.ipynb
│   ├── drone_3d_flight_Fix_noise.html
│   ├── analysis_results_Fix_noise/
│   └── results_drone_Fix_noise/
│
└── 优化+3D仿真新（Q2有问题）/                     # Q2 问题待修复
    ├── 优化+3D仿真新.ipynb
    ├── drone_3d_flight_new.html
    ├── analysis_results/
    └── results_drone/
```

---

## 快速开始

### 1. 运行最新优化版本（推荐）

```bash
cd 【史诗级修复】优化+3D仿真新+曲线收敛+平滑噪声修正/
jupyter notebook 【史诗级重制】优化+3D仿真新+曲线收敛+平滑噪声修正.ipynb
```

### 2. 查看 3D 轨迹可视化

直接用浏览器打开 `drone_3d_flight_Plz_Converge.html` 即可查看交互式 3D 轨迹。

### 3. 查看训练结果分析

```bash
cd analysis_results_Plz_Converge/
# 查看收敛曲线等分析图表
```

---

## 核心功能

- **TD3 算法**：Twin Delayed DDPG，连续动作空间最优算法
- **3D 轨迹可视化**：Plotly 交互式 HTML 可视化
- **收敛曲线分析**：奖励曲线、损失曲线
- **目标平滑噪声修正**：提升训练稳定性
- **平滑噪声处理**：减少震荡，提高收敛效果

---

## 环境依赖

```
torch
numpy
pandas
matplotlib
seaborn
plotly
gymnasium
scipy
```

安装：
```bash
pip install -r requirement.txt
```
