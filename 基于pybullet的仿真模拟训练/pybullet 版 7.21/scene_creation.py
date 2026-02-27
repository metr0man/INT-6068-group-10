import pybullet as p
import pybullet_data
import numpy as np
import time

class DroneScene:
    def __init__(self, gui=False):
        """初始化无人机场景"""
        self.gui = gui
        # 强制使用 DIRECT 模式，避免 X11 问题
        p.connect(p.DIRECT)
        
        # 设置物理引擎参数
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / 240)
        
        # 加载地面
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        
        # 场景参数
        self.scene_size = [200, 200, 10]  # 场景大小扩大
        self.obstacles = []
        self.targets = []
        self.drone_id = None
        # 初始摄像头设置（后续会自动调整）
        p.resetDebugVisualizerCamera(
            cameraDistance=120,  # 距离更远
            cameraYaw=45,
            cameraPitch=-60,
            cameraTargetPosition=[0, 0, 100]  # 目标高度大幅提升
        )
        # 可视化起点（绿色大圆）
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=15,
            length=1,
            rgbaColor=[0, 1, 0, 0.5],
            visualFramePosition=[0, 0, 0]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, 0.5]
        )
        
    def auto_set_camera(self):
        """自动根据场景内容设置摄像头"""
        positions = []
        if self.obstacles:
            positions += [obs['position'] for obs in self.obstacles]
        if self.targets:
            positions += self.targets
        if self.drone_id is not None:
            pos, orn = p.getBasePositionAndOrientation(self.drone_id)
            positions.append(pos)
        if not positions:
            positions = [[0, 0, 10]]
        positions = np.array(positions)
        center = positions.mean(axis=0)
        max_dist = np.linalg.norm(positions - center, axis=1).max()
        camera_distance = max(20, max_dist * 2)
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=45,
            cameraPitch=-60,
            cameraTargetPosition=center.tolist()
        )
    
    def create_obstacles(self, num_obstacles=150, area=150, min_height=1.0, max_height=100.0, min_dist_between=60.0, max_dist_between=200.0):
        """创建障碍物，障碍物为方形，长宽10~30米，高度不超过100米，避开无人机初始点和所有目标点，障碍物之间距离至少60米"""
        self.obstacles = []
        drone_pos = [0, 0]  # 无人机初始xy
        min_dist = 10.0      # 与无人机xy距离大于10m
        placed_positions = []
        # 获取所有目标点xy
        target_positions = []
        if hasattr(self, 'targets') and self.targets:
            target_positions = [[t[0], t[1]] for t in self.targets]
        for i in range(num_obstacles):
            for _ in range(200):  # 最多尝试200次
                x = np.random.uniform(-area, area)
                y = np.random.uniform(-area, area)
                # 距离无人机起点
                if np.linalg.norm([x-drone_pos[0], y-drone_pos[1]]) <= min_dist:
                    continue
                # 距离所有目标点
                if target_positions:
                    if min([np.linalg.norm([x-tx, y-ty]) for tx, ty in target_positions]) < 60.0:
                        continue
                # 距离已放置障碍物
                if placed_positions:
                    dists = [np.linalg.norm([x-px, y-py]) for px, py in placed_positions]
                    if min(dists) < min_dist_between:
                        continue
                length = np.random.uniform(10, 30)
                width = np.random.uniform(10, 30)
                height = np.random.uniform(min_height, max_height)
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[length/2, width/2, height/2],
                    rgbaColor=[1, 0, 0, 0.8],
                    visualFramePosition=[0, 0, height/2]
                )
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[length/2, width/2, height/2],
                    collisionFramePosition=[0, 0, height/2]
                )
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=[x, y, 0]
                )
                self.obstacles.append({
                    'id': obstacle_id,
                    'position': [x, y, height/2],
                    'length': length,
                    'width': width,
                    'height': height
                })
                placed_positions.append([x, y])
                break
        # self.auto_set_camera()  # 固定摄像头，不再自动调整
    
    def create_targets(self, num_targets=2, area=8000):
        """创建目标点，目标点xy距离起点(0,0)至少6000米，且不在障碍物内"""
        self.targets = []
        min_dist = 6000.0
        safety_margin = 2.0  # 目标点距离障碍物边缘的最小安全距离
        for i in range(num_targets):
            for _ in range(20000):  # 最多尝试2万次
                x = np.random.uniform(-area, area)
                y = np.random.uniform(-area, area)
                if np.linalg.norm([x, y]) < min_dist:
                    continue
                # 检查是否在障碍物内（方形判定）
                in_obstacle = False
                for obs in self.obstacles:
                    ox, oy, _ = obs['position']
                    l = obs['length'] / 2 + safety_margin
                    w = obs['width'] / 2 + safety_margin
                    if (abs(x - ox) < l) and (abs(y - oy) < w):
                        in_obstacle = True
                        break
                if in_obstacle:
                    continue
                break
            else:
                print(f"警告：第{i+1}个目标点生成失败，未找到合适位置")
                continue
            z = np.random.uniform(1, 10)  # 目标点高度范围改为1~10米
            self.targets.append([x, y, z])
            # 可视化终点（绿色大圆）
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=15,
                length=1,
                rgbaColor=[0, 1, 0, 0.5],
                visualFramePosition=[0, 0, 0]
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[x, y, 0.5]
            )
        # self.auto_set_camera()  # 固定摄像头，不再自动调整
    
    def create_drone(self, position=[0,0,2.5], size=5, drone_id=None):
        """创建无人机"""
        # 创建一个可见的球体作为无人机
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size,
            rgbaColor=[0, 0, 1, 1],
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=size
        )
        drone_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        if self.drone_id is None:
            self.drone_id = drone_id  # 保持向后兼容
        # self.auto_set_camera()  # 固定摄像头，不再自动调整
        return drone_id
    
    def get_drone_position(self, drone_id=None):
        """获取无人机当前位置"""
        if drone_id is None:
            drone_id = self.drone_id
        if drone_id is not None:
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            return pos
        return None
    
    def move_drone_to(self, target_position, drone_id=None, max_force=100):
        """移动无人机到指定位置"""
        if drone_id is None:
            drone_id = self.drone_id
        if drone_id is not None:
            # 使用位置控制
            p.resetBasePositionAndOrientation(
                drone_id,
                target_position,
                [0, 0, 0, 1]
            )
    
    def check_collision(self, position, radius=0.5):
        """检查指定位置是否与障碍物碰撞（方形障碍物）"""
        x, y, z = position
        for obs in self.obstacles:
            ox, oy, _ = obs['position']
            l = obs['length'] / 2 + radius
            w = obs['width'] / 2 + radius
            if (abs(x - ox) < l) and (abs(y - oy) < w):
                return True
        return False
    
    def check_target_reached(self, position, threshold=1.0):
        """检查是否到达目标点"""
        for target in self.targets:
            if not target['visited']:
                target_pos = target['position']
                distance = np.linalg.norm(np.array(position) - np.array(target_pos))
                
                if distance < threshold:
                    target['visited'] = True
                    return True, target
        return False, None
    
    def reset_scene(self, drone_ids=None):
        """重置场景"""
        # 重置无人机位置
        if drone_ids is not None:
            # 多无人机重置
            for i, drone_id in enumerate(drone_ids):
                if drone_id is not None:
                    # 为每个无人机设置稍微不同的起始位置，避免重叠
                    offset_x = i * 10  # 每个无人机间隔10米
                    p.resetBasePositionAndOrientation(
                        drone_id,
                        [offset_x, 0, 2.5],
                        [0, 0, 0, 1]
                    )
        elif self.drone_id is not None:
            # 单无人机重置（向后兼容）
            p.resetBasePositionAndOrientation(
                self.drone_id,
                [0, 0, 1],
                [0, 0, 0, 1]
            )
        
        # 目标点为 [x, y, z] 列表，无需设置 visited 标记
    
    def set_camera_near_drone(self, distance=3, yaw=30, pitch=-20):
        """将摄像机放到无人机旁边（跟随视角）"""
        if self.drone_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.drone_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=pos
            )
    
    def render(self):
        """渲染场景"""
        self.set_camera_near_drone()  # 每帧跟随无人机
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    # 测试场景创建
    scene = DroneScene(gui=True)
    scene.create_obstacles()
    scene.create_targets()
    drone_id = scene.create_drone()
    
    print("场景创建完成！")
    print(f"障碍物数量: {len(scene.obstacles)}")
    print(f"目标点数量: {len(scene.targets)}")
    
    # 运行仿真
    for i in range(1000):
        scene.render() 