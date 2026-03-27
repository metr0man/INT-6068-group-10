# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic project (INT6068 - Neural Networks & Deep Learning) implementing **3D drone path planning via reinforcement learning**. The project evolved from a PyBullet physics simulation to focus on simplified lightweight simulations and custom TD3 deep learning implementations.

**Note:** The PyBullet sub-project (`基于pybullet的仿真模拟训练/`) was an earlier iteration and is no longer maintained — the code referenced in older docs may not exist.

## Active Sub-Projects & Entry Points

### 1. Deep Learning Experiments (`1深度学习尝试/`)

Custom TD3 (Twin Delayed DDPG) implementation with PyTorch — the current focus of development.

**`deeplearning1/` — Complete TD3 baseline:**
```bash
cd 1深度学习尝试/deeplearning1/
python train.py
```

**`deeplearning2/` — Enhanced TD3 (in development):**
```bash
cd 1深度学习尝试/deeplearning2/
python train.py
```

### 2. Simplified Simulation (`2简化仿真模拟环境下的结果/`)

Lightweight multi-agent drone simulation (no physics engine). Entry point `结合体.py` in each variant:

```bash
# Static environment
cd 2简化仿真模拟环境下的结果/静态模拟/
python 结合体.py

# Dynamic environment
cd 2简化仿真模拟环境下的结果/动态模拟/
python 结合体.py

# Safety module
cd 2简化仿真模拟环境下的结果/安全模块/
python 结合体.py
```

## Architecture

### Deep Learning (`deeplearning1/`)

| File | Responsibility |
|------|----------------|
| `environment.py` | `DroneEnv` — Gymnasium Env: 6-dim state (pos + goal dir), 3-dim action (velocity) |
| `td3.py` | TD3 class — Actor-Critic with twin Q-networks, target policy smoothing |
| `model.py` | `Actor` and `Critic` network architectures |
| `train.py` | Training loop with ReplayBuffer (100k capacity), CSV logging |
| `analysis.py` | Training metrics analysis |

**State space (6-dim):** position (3) + goal direction vector (3)
**Action space (3-dim):** velocity commands [vx, vy, vz] in [-1, 1]
**Algorithm:** TD3 with dual Critics, policy delay, target action smoothing

### Simplified Simulation (`结合体.py`)

`MultiAgentDroneDeliveryEnv` — multi-drone coordination environment:
- **Gymnasium standard** — compatible with `gymnasium.make()`
- **KDTree** — scipy spatial queries for city building collision detection
- **Battery/energy** constraints per drone
- **Wind effect** simulation
- **Matplotlib 3D** trajectory visualization

State/Observation space per drone: position (3), velocity (3), goal (3), obstacles (12×5), other drones ((n-1)×3), battery (1)

## Tech Stack

- **PyTorch 2.0.1** — Neural networks (custom TD3)
- **Gymnasium** — RL environment interface
- **NumPy / Pandas / Matplotlib / Seaborn** — Data and visualization
- **SciPy KDTree** — Spatial collision queries (simplified sim)
- **PyBullet** — (deprecated/removed — earlier iteration only)

## Reward System

The reward function is distributed across implementations:
- **Deep learning:** `environment.py` — distance-based reward, boundary collision penalty
- **Simplified sim:** `结合体.py` — multi-component reward (distance, goal completion, collision, energy, wind compensation)

## Project Evolution

The project progressed through iterations:
1. **PyBullet simulation** — physics-based, modular design (deprecated, code may not exist)
2. **Simplified simulation** — lightweight multi-agent,放弃了物理真实性，专注于多智能体强化学习算法研究
3. **Deep learning (TD3)** — custom PyTorch implementation, static environment training
