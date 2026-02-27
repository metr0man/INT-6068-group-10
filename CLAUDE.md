# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic project (INT6068 - Neural Networks & Deep Learning) implementing **3D drone path planning via reinforcement learning**. It contains three independent sub-projects that evolved iteratively.

## Sub-Projects & Entry Points

### 1. PyBullet Simulation (Main — `基于pybullet的仿真模拟训练/pybullet 版 7.21/`)
The primary, most complete implementation using physics-based simulation.

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation/visualization
python main.py

# Train with PPO (Stable-Baselines3)
python train_rl.py

# Large-scale parallel training (128 targets, SubprocVecEnv)
python train_massive_single_drone.py

# Server-mode execution
python server_run.py

# Analyze training results
python data_analysis.py
python line_chart_generator.py
```

### 2. Simplified Simulation Results (`简化仿真模拟环境下的结果/`)
Three variants: static (`静态模拟/`), dynamic (`动态模拟/`), and safety module (`安全模块/`).
Each has a `结合体.py` as the main entry point.

### 3. Deep Learning Experiments (`深度学习尝试/`)
```bash
# TD3 training on static environment (complete)
cd deeplearning1 && python train.py

# Enhanced TD3 training (in progress)
cd deeplearning2 && python train.py
```

## Architecture

### PyBullet Version (Modular Design)

| File | Class | Responsibility |
|------|-------|----------------|
| `scene_creation.py` | `DroneScene` | PyBullet 3D environment, obstacles, targets, drone physics |
| `path_planning.py` | `PathPlanning` | A* and RRT path finding with path smoothing |
| `reward_system.py` | `RewardSystem` | Multi-component reward calculation |
| `massive_single_drone_env.py` | `SingleDronePathPlanningEnv` | Gymnasium-compatible RL environment |
| `train_rl.py` | — | PPO training loop with SubprocVecEnv parallelization |
| `main.py` | — | Entry point: loads trained model, runs visual demo |

**State space (35-dim):** position (3) + goal direction (3) + ray distances (26) + velocity (3)
**Action space (3-dim):** velocity commands (x, y, z)
**Ray casting:** 26-directional obstacle detection for perception
**Algorithm:** PPO via Stable-Baselines3; trained for ~6M steps

### Simplified Simulation (`结合体.py`)

`MultiAgentDroneDeliveryEnv` handles multi-drone coordination with:
- KDTree-based city building collision detection
- Battery/energy constraints
- Wind effect simulation
- Matplotlib 3D trajectory visualization

### Deep Learning (`deeplearning1/`)

Custom TD3 implementation with separate files:
- `td3.py` — Actor-Critic networks and TD3 update logic
- `model.py` — Network architectures
- `environment.py` — `DroneEnv` custom Gym environment
- `analysis.py` — Training metrics analysis

## Reward Function Design (`奖励函数设计/`)

The reward system uses multiple components:
- Distance reward: smooth non-linear function toward goal
- Target completion bonus: +400 pts
- Collision penalty: −400 to −600 pts
- Energy efficiency reward
- Wind compensation reward
- Obstacle avoidance safety reward

The final integrated reward functions are in `reward_system.py` (PyBullet version) and embedded in `结合体.py` (simplified version).

## Tech Stack

- **PyBullet 3.2.5** — Physics simulation
- **Gymnasium 0.28.1** — RL environment interface
- **Stable-Baselines3 2.0.0** — PPO algorithm
- **PyTorch 2.0.1** — Neural networks (TD3 custom implementation)
- **NumPy / Pandas / Matplotlib / Seaborn** — Data and visualization
- **SciPy KDTree** — Spatial collision queries

## Trained Models

Pre-trained model checkpoints are stored in `基于pybullet的仿真模拟训练/model/`. `main.py` loads these for demo visualization.
