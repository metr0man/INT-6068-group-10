# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic project (INT6068 - Neural Networks & Deep Learning) implementing **3D drone path planning via reinforcement learning**. The project evolved from PyBullet physics simulation → simplified multi-agent simulation → custom TD3 deep learning implementations.

## Current Focus

### Primary Development: `【🔥最新】简单环境+简单避障+简单三维仿真/`

TD3 training with 3D visualization — **this is the main active development folder**.

**Entry point (Jupyter Notebook):**
```bash
cd 【🔥最新】简单环境+简单避障+简单三维仿真/【史诗级修复】优化+3D仿真新+曲线收敛+平滑噪声修正/
jupyter notebook 【史诗级重制】优化+3D仿真新+曲线收敛+平滑噪声修正.ipynb
```

**Key variants:**
| Folder | Status | Description |
|--------|--------|-------------|
| 【史诗级修复】优化+3D仿真新+曲线收敛+平滑噪声修正/ | ✅ Recommended | Convergence fixed, noise smoothed |
| 优化+3D仿真新+TD3目标平滑噪声修正/ | ✅ Available | TD3 target smoothing fix |
| 优化+3D仿真新（Q2有问题）/ | ❌ Broken | Q2 has issues, avoid |

**3D Visualization:** Open `drone_3d_flight_Plz_Converge.html` in a browser for interactive Plotly 3D trajectories.

### Presentation: `【🔥最新】Pre内容/`

Group presentation materials organized by section:
- 小组汇报大纲.md
- 第一部分：项目背景与 MDP 建模
- 第二部分：核心算法与神经网络细节
- 第三部分：3D仿真环境与训练细节
- 第四部分新：实验结果与 3D 轨迹展示
- 第五部分：消融实验

## Legacy Folders (Not Maintained)

| Folder | Status | Description |
|--------|--------|-------------|
| `1深度学习尝试/` | ⚠️ Legacy | Early TD3 experiments (deeplearning1/, deeplearning2/) |
| `2简化仿真模拟环境下的结果/` | ⚠️ Legacy | Simplified simulation — heuristic-based, no RL training |
| `基于pybullet的仿真模拟训练/` | ❌ Removed | PyBullet physics sim (early iteration, code may not exist) |

## Architecture

### TD3 Algorithm (current implementation)

**State space (6-dim):** position (3) + goal direction vector (3)
**Action space (3-dim):** velocity commands [vx, vy, vz] in [-1, 1]
**Algorithm:** TD3 with twin Q-networks, policy delay, target action smoothing

### Tech Stack

- **PyTorch 2.0.1** — Neural networks
- **Gymnasium** — RL environment interface
- **Plotly** — 3D HTML trajectory visualization
- **Matplotlib/Seaborn** — Training curves
- **SciPy KDTree** — Spatial collision queries (simplified sim only)

## Project Evolution

1. **PyBullet simulation** — physics-based (deprecated/removed)
2. **Simplified simulation** — lightweight multi-agent, heuristic-based
3. **Deep learning (TD3)** — custom PyTorch implementation with 3D visualization

## Entry Points

```bash
# Run latest TD3 training
cd 【🔥最新】简单环境+简单避障+简单三维仿真/【史诗级修复】优化+3D仿真新+曲线收敛+平滑噪声修正/
jupyter notebook 【史诗级重制】优化+3D仿真新+曲线收敛+平滑噪声修正.ipynb

# Legacy (not recommended)
cd 1深度学习尝试/deeplearning1/
python train.py
```
