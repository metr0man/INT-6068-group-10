## pybullet版本: 使用PyBullet内置的GUI渲染，提供真实的3D物理可视化:
# 建议别碰
模块化设计，分为多个文件：scene_creation.py、path_planning.py、reward_system.py
主类DronePathPlanningSimulation协调各个模块
注意：**不是标准的RL环境，更像是一个路径规划测试平台，如果要看强化学习看静态路径/结合体.py**



pybullet版7.21是一个比较完整的版本，但我们后面发现奖励函数仍有问题，于是进一步修改
pybullet版7.21最新版本.zip是最终版本，我们基于此进行了600万次的训练
pybullet版7.21可视化尝试.zip是可视化尝试

model文件夹中包含了pybullet版7.21训练出的一个模型
和最终版本训练出的模型

注：非苹果打开会有 MACOSX文件夹

好久之前的项目，没有项目管理的意识。最后多次修改，可能找错版本，深感抱歉，可以联系我


