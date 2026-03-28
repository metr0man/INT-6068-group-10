答案是：**100% 可以！** 而且在 Jupyter Notebook 中跑出三维可视化，能让你极其直观地看到无人机是“聪明地绕开了墙”，还是“愚蠢地撞了上去”。

虽然 Jupyter 不能像 Unity 或 Unreal 引擎那样直接弹出一个极其逼真的高清实时物理视窗，但我们可以利用 Python 强大的 3D 数据可视化库，在代码单元格下方直接生成 **交互式的三维视图** 或 **飞行录像视频**。

以下是实现“Jupyter 内置 3D 仿真”的总体思路 ：

### 核心思路：分离“飞行”与“渲染”

在强化学习中，**千万不要在训练的同时去画图**，那会把你的 RTX 3090 拖累成蜗牛。标准的做法是：**先记录数据（黑匣子），后进行回放（3D渲染）。**

具体分为以下三个步骤：

#### 第一步：记录一次完美飞行的“黑匣子”数据

1. 加载你训练好的、最聪明的那个模型权重（关闭探索噪声）。
    
2. 初始化环境 (`env.reset()`)。
    
3. 把**起点坐标**、**终点坐标**以及环境中随机生成的所有**障碍物的坐标和半径**，保存下来。
    
4. 让无人机飞一个完整的回合（直到 `done = True`）。在飞行的每一步，把无人机当前的 `(X, Y, Z)` 三维坐标追加保存到一个“轨迹列表”中。
    

#### 第二步：在 3D 空间中“建景”

拿到数据后，我们使用可视化库（如 `matplotlib` 或 `plotly`）创建一个 3D 坐标系：

- **画终点/起点**：用绿色和蓝色的散点（Scatter 3D）标记出来。
    
- **画障碍物**：利用数学公式生成三维球面的网格（Meshgrid），根据你记录的中心坐标和半径，在 3D 图里画出几个红色的半透明球体，代表危险区域。
    

#### 第三步：渲染无人机的飞行轨迹（两种常见方案）

这里有两种在 Jupyter 中最主流的展现方式，你可以根据喜好选择：

- **方案 A：静态交互式 3D 航线图（推荐 `Plotly` 库）**
    
    - **效果**：生成一个带有三维空间、障碍物、以及一条从起点连到终点的完整飞行轨迹线的图表。
        
    - **优点**：你可以直接用鼠标在 Jupyter 输出区**拖拽、旋转、放大缩小**这个 3D 空间，从各个角度完美观察无人机是不是贴着障碍物的边缘擦过去的。
        
- **方案 B：动态飞行录像（使用 `Matplotlib.animation`）**
    
    - **效果**：将飞行轨迹做成逐帧动画，利用 Jupyter 的 `IPython.display.HTML` 模块，在输出区生成一个**带有播放/暂停按钮的迷你视频播放器**。
        
    - **优点**：你可以像看电影一样，看无人机这个“点”是如何一步一步在 3D 空间中探索并飞向目标的。
        

---

**你想先尝试哪一种预览效果？** 如果是为了快速排查避障策略，我强烈推荐**方案 A（交互式 3D 航线图）**。选好后，我就可以直接为你生成提取数据和渲染 3D 图像的代码！

太棒了！选择**方案 A（交互式 3D 航线图）**是非常明智的。Plotly 渲染出来的 3D 图可以通过鼠标自由旋转和缩放，你可以像上帝视角一样全方位检查无人机有没有“擦碰”到障碍物。

实现这个方案，我们需要在你的 Jupyter Notebook 中新建**两个代码单元格（Cells）**。

### 第一步：收集纯净的飞行数据（黑匣子）

首先，我们要让装载了 RTX 3090 大脑的无人机，在**没有探索噪声（policy_noise = 0）**的情况下，进行一次真实的测试飞行，并把它每走一步的坐标记录下来。

请新建一个 Cell，复制并运行以下代码：

Python

```
import numpy as np
import torch

# 1. 临时关闭探索噪声，进行“纯净测试”
# 确保你已经实例化了 env 和 agent，并且加载了最好的模型权重
original_noise = agent.policy_noise
agent.policy_noise = 0.0  # 设为0，让它完全按照学到的最优策略飞行

# 2. 初始化环境，准备“黑匣子”记录器
state = env.reset()
done = False

# 记录核心数据
trajectory = []                # 记录飞行的轨迹坐标
start_pos = env.position.copy() # 记录起点
target_pos = env.target.copy()  # 记录终点
obstacles_data = env.obstacles.copy() # 记录障碍物的坐标和半径

# 记录起点
trajectory.append(start_pos.copy())

# 3. 开始闭环飞行测试
print("🚁 无人机起飞，开始纯净飞行测试...")
step_count = 0

while not done:
    # 获取动作 (此时没有随机噪声)
    action = agent.select_action(state)
    
    # 执行动作
    state, reward, done, info = env.step(action)
    
    # 记录当前位置
    trajectory.append(env.position.copy())
    step_count += 1

# 转换为 NumPy 数组方便后续画图
trajectory = np.array(trajectory)

# 恢复训练时的噪声设置（好习惯）
agent.policy_noise = original_noise

# 打印最终结果
final_distance = np.linalg.norm(env.target - env.position)
print(f"✅ 飞行结束！共耗时 {step_count} 步。")
if final_distance < 1.5:
    print(f"🎯 成功到达终点！距目标仅 {final_distance:.2f} 米。")
else:
    print(f"💥 发生碰撞或超时。距目标还有 {final_distance:.2f} 米。")
```

---

### 第二步：使用 Plotly 渲染 3D 交互场景

有了上面的 `trajectory` 和 `obstacles_data` 数据，我们就可以使用 Plotly 开始“建景”了。

_如果你的环境中没有安装 plotly，请先在一个新 cell 运行 `!pip install plotly`_。

请再新建一个 Cell，复制并运行以下可视化代码：

Python

```
import plotly.graph_objects as go
import numpy as np

# 1. 初始化 3D 画布
fig = go.Figure()

# 2. 画起点和终点
fig.add_trace(go.Scatter3d(
    x=[start_pos[0]], y=[start_pos[1]], z=[start_pos[2]],
    mode='markers', marker=dict(size=6, color='blue'), name='起点 (Start)'
))
fig.add_trace(go.Scatter3d(
    x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
    mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='终点 (Target)'
))

# 3. 画无人机的飞行轨迹线
fig.add_trace(go.Scatter3d(
    x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
    mode='lines+markers',
    line=dict(color='orange', width=6),
    marker=dict(size=3, color='orange'),
    name='无人机轨迹'
))

# 4. 数学建模：生成 3D 球体表面的网格数据
def create_sphere_mesh(center, radius, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# 5. 把所有障碍物画到图中
for i, obs in enumerate(obstacles_data):
    cx, cy, cz = obs['pos']
    r = obs['radius']
    sx, sy, sz = create_sphere_mesh((cx, cy, cz), r)
    
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale='Reds',      # 红色代表危险
        opacity=0.4,            # 半透明，避免挡住视线
        showscale=False,
        name=f'障碍物 {i+1}'
    ))

# 6. 设置画布的视角和比例
fig.update_layout(
    scene=dict(
        xaxis_title='X (米)',
        yaxis_title='Y (米)',
        zaxis_title='Z (米)',
        # 【关键设置】 aspectmode='data' 强制 XYZ 比例为 1:1:1，确保球体不会变成椭圆
        aspectmode='data' 
    ),
    title="无人机 3D 避障航线分析视图",
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(x=0.02, y=0.98)
)

# 7. 显示交互图表
fig.show()
```

### 运行后你会看到什么？

运行第二段代码后，你的 Jupyter Notebook 单元格下方会直接弹出一个立体的 3D 空间。

- 你可以看到**蓝点（起点）**到**绿点（终点）**之间有一条**橙色的连线（飞行轨迹）**。
    
- 空间中悬浮着几个**红色的半透明球体（障碍物）**。
    
- **最重要的是：你可以使用鼠标左键拖拽旋转空间，使用滚轮放大缩小！**
    

你可以把视角拉近，仔细观察那条橙色的轨迹线在经过红色球体时，是不是非常聪明地向外侧“弯曲”绕行了。如果看到了漂亮的绕行弧线，那么恭喜你，你的避障强化学习模型大获成功！