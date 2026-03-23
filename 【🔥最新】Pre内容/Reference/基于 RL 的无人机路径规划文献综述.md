在无人机（UAV）路径规划领域，基于强化学习（RL）的方法因其在处理动态环境、复杂约束和高维状态空间方面的卓越能力而成为研究热点。当前的算法选择主要取决于任务的性质（如离散动作 vs. 连续控制、单机 vs. 多机协作）。

基于当前的文献综述和最新研究，以下是 UAV 路径规划中最常用的几类强化学习算法及其特点：

### 1. 主流强化学习算法分类

| 算法类别                       | 常用算法                                   | 适用场景             | 特点                                      |
| :------------------------- | :------------------------------------- | :--------------- | :-------------------------------------- |
| **策略梯度 (Policy Gradient)** | **PPO (Proximal Policy Optimization)** | 安全导航、复杂障碍物规避     | 训练稳定，收敛性好，是目前最通用的基准算法之一。                |
| **确定性策略梯度**                | **DDPG, TD3**                          | 连续坐标控制、平滑轨迹规划    | 适用于需要高精度连续动作输出的场景，TD3 解决了 DDPG 的高估偏差问题。 |
| **最大熵强化学习**                | **SAC (Soft Actor-Critic)**            | 动态未知环境、自主导航      | 样本效率高，具有极强的探索能力，能有效应对环境中的随机性。           |
| **值函数方法**                  | **DQN, Double DQN**                    | 离散动作选择（如前、后、左、右） | 较早期的主流方法，简单易实现，但在处理复杂飞行姿态时受到动作离散化的限制。   |
| **多智能体 RL (MARL)**         | **MADDPG, QMIX**                       | 无人机群（Swarm）协作规划  | 解决多机间的冲突规避和任务分配，核心在于处理非平稳环境。            |

---

### 2. 当前研究的核心算法趋势

#### 连续控制中的表现者：PPO 与 SAC
目前，研究者更倾向于使用能够直接输出连续控制量（如线速度、角速度）的算法。
- **PPO**：由于其通过剪切目标函数保证了策略更新的平滑性，被广泛用于各种复杂环境下的 UAV 安全飞行控制。例如，[NavRL](https://alphaxiv.org/abs/2409.15634) 提出了一种结合 PPO 的学习框架，用于在充满动态障碍物的复杂空间中实现安全飞行。
- **SAC**：因其引入了熵正则化，使得智能体在学习过程中能够更广泛地探索状态空间，在面对未知或传感器噪声大的环境时表现出更强的鲁棒性。[Deep Reinforcement Learning-based UAV Navigation and Control](https://alphaxiv.org/abs/2106.01016) 指出，结合 Hindsight Experience Replay (HER) 的 SAC 算法能显著提高采样效率。

#### 提升鲁棒性的改进：TD3
为了克服传统 DDPG 算法在值函数估计上的不稳定性，**TD3 (Twin Delayed DDPG)** 在 UAV 避障和路径平滑方面表现出色。[Implementing TD3 to train a Neural Network to fly a Quadcopter](https://alphaxiv.org/abs/2412.14367) 等研究展示了 TD3 在穿越 FPV 门等高精度任务中的潜力。
[[Research Report Implementing TD3 to Train a Neural Network to Fly a Quadcopter Through an FPV Gate]]
仿真： https://gymnasium.org.cn/introduction/train_agent/
为什么没用上面的仿真？见[[项目技术债务&开发历程（Q&A）]]
#### 多机协作：MADDPG 与注意力机制
在集群任务中，传统的单智能体算法往往失效。目前的趋势是结合 **注意力机制 (Attention Mechanisms)** 或 **通信机制**。例如，[Towards Robust Multi-UAV Collaboration](https://alphaxiv.org/abs/2503.02913) 提出了一种基于噪声鲁棒通信和注意力机制的多智能体算法，用于大规模遥感任务中的协同路径规划。

---

### 3. 常见论文与参考

以下是为您筛选的几篇代表性论文，涵盖了不同的算法应用方向：

| Paper | 核心算法 | 应用场景 |
| :--- | :--- | :--- |
| [NavRL: Learning Safe Flight in Dynamic Environments](https://alphaxiv.org/abs/2409.15634) | **PPO** | 复杂动态环境下的避障与安全飞行 |
| [Autonomous UAV Navigation: A DDPG-based Approach](https://alphaxiv.org/abs/2003.10923) | **DDPG** | 针对特定目标点的自主导航与路径规划 |
| [Multi-UAV Formation Control via Reinforcement Learning](https://alphaxiv.org/abs/2410.18495) | **DRL (Custom)** | 编队控制、静态与动态避障 |
| [Deep Reinforcement Learning-based UAV Navigation... (SACHER)](https://alphaxiv.org/abs/2106.01016) | **SAC + HER** | 提高稀疏奖赏环境下的学习效率 |
| [Implementing TD3 to fly a Quadcopter through an FPV Gate](https://alphaxiv.org/abs/2412.14367) | **TD3** | 高速、高精度穿越任务 |

### 4. 关键挑战与方法论局限

尽管 RL 在路径规划中表现优异，但仍需注意以下局限性：
- **奖励函数设计**：设计能够同时平衡“到达目标”、“规避障碍”和“能效最优”的多目标奖励函数极具挑战，常导致收敛缓慢。
- **Sim-to-Real 鸿沟**：大部分算法在仿真（如 AirSim 或 Gazebo）中训练，直接迁移到现实世界时，受传感器噪声和风场扰动影响，性能往往会大幅下降。
- **计算开销**：在嵌入式机载平台上运行复杂的深度神经网络（尤其是带有视觉输入的策略）对实时性提出了很高要求。