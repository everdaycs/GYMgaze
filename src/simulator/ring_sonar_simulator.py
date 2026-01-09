#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring Sonar Simulator - 环形超声波雷达模拟器

用12个均匀分布的超声波传感器替代主动摄像头
- 传感器布局：半径15cm圆盘边缘，均匀分布12个传感器
- 传感器参数：65° FoV，最大探测距离12.5m
- 2D俯视图环境
"""

import numpy as np
import os
import sys
import random
import math
import cv2
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 导入工具函数
from src.utils.geometry import (
    clamp, angnorm_deg, angdiff_deg,
    add_global_feature
)
from src.utils.fisher import SonarFisherCalculator

# 导入配置
from configs import (
    SimulationConfig, RobotPhysicsConfig, SensorConfig, WorldConfig,
    DEFAULT_CONFIG, DEMO_CONFIG, print_config
)

# 导入传感器和触发器
from src.simulator.sensors import SonarSensor
from src.simulator.trigger import (
    TriggerManager, TriggerMode, TriggerConfig,
    DEFAULT_TRIGGER_CONFIG, DATA_COLLECTION_TRIGGER_CONFIG
)
from src.simulator.trigger_strategies import TriggerContext

# 导入多样化地图生成器
from src.simulator.map_generator import DiverseMapGenerator

# 导入训练好的全局地图预测模型
try:
    from src.models.global_map import GlobalMapPredictor, ConvBlock
    MODEL_AVAILABLE = True
except ImportError:
    try:
        # 兼容旧路径
        from train_global_map_model import GlobalMapPredictor, ConvBlock
        MODEL_AVAILABLE = True
    except ImportError:
        MODEL_AVAILABLE = False
        print("Warning: Could not import GlobalMapPredictor model")


# ------------------------------- 核心模拟器 -------------------------------- #

class RingSonarCore:

    def __init__(self,
                 world_width: float = None,
                 world_height: float = None,
                 pixel_per_meter: int = 20,
                 robot_size: float = None,
                 sensor_ring_radius: float = None,
                 num_sensors: int = None,
                 sensor_fov: float = None,
                 sensor_max_range: float = None,
                 feature_map_size: int = 100,
                 feature_map_resolution: float = 0.25,
                 control_frequency: float = 5.0,
                 trigger_mode: str = "sequential",  # 触发模式
                 randomize_trigger: bool = False,   # 是否随机选择触发模式
                 dt: float = None,
                 config: SimulationConfig = None):  # 新增：使用配置对象
        
        # 使用配置对象（如果提供）或使用默认配置
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config
        
        # 世界参数（配置优先，然后是参数，最后是默认值）
        self.world_width = float(world_width if world_width is not None else config.world.world_width)
        self.world_height = float(world_height if world_height is not None else config.world.world_height)
        self.pixel_per_meter = int(pixel_per_meter)
        self.robot_size = float(robot_size if robot_size is not None else config.robot.robot_size)
        
        # 传感器参数
        self.sensor_ring_radius = float(sensor_ring_radius if sensor_ring_radius is not None else config.sensor.ring_radius)
        self.num_sensors = int(num_sensors if num_sensors is not None else config.sensor.num_sensors)
        self.sensor_fov = float(sensor_fov if sensor_fov is not None else config.sensor.fov_angle)
        self.sensor_max_range = float(sensor_max_range if sensor_max_range is not None else config.sensor.max_range)
        
        # 初始化传感器阵列
        self.sensors: List[SonarSensor] = []
        self._init_sensors()
        
        # Fisher地图 (2D，机器人中心的局部视图)
        self.feature_map_size = int(feature_map_size)
        self.feature_map_resolution = float(feature_map_resolution)
        self.feature_map = np.zeros((self.feature_map_size, self.feature_map_size), dtype=np.float32)
        self.global_feature_map_size = int(max(self.world_width, self.world_height) * 2 / self.feature_map_resolution)
        self.global_feature_map = np.zeros((self.global_feature_map_size, self.global_feature_map_size), dtype=np.float32)
        
        # 时间与控制（使用配置中的dt）
        self.dt = float(dt if dt is not None else config.robot.dt)
        self.sim_time = 0.0
        
        # 机器人状态
        self.robot_pos = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float64)
        self.robot_angle = 0.0           # deg, 机器人朝向
        self._robot_angle_rad_cache = 0.0  # 缓存弧度值，避免重复转换
        self.velocity = 0.0              # m/s
        self.angular_velocity = 0.0      # rad/s
        self.max_linear_velocity = config.robot.max_linear_velocity
        self.max_angular_velocity = config.robot.max_angular_velocity
        
        # 传感器触发控制 - 控制何时实际扫描传感器
        self.sensor_trigger_interval = max(1, int(config.robot.sensor_trigger_interval))
        self.next_allowed_trigger_time = 0.0
        
        # 传感器就绪时间 (每个传感器下一次可以触发的仿真时间)
        self.sensor_ready_times = np.zeros(self.num_sensors, dtype=np.float64)
        
        # 传感器读数 (每个传感器的距离测量)
        self.sonar_readings = np.full(self.num_sensors, self.sensor_max_range, dtype=np.float32)
        
        # 声学仿真参数 (用于跨帧串扰)
        self.speed_of_sound = 343.0  # 声速 (m/s)
        self.max_echo_time = 2.0 * self.sensor_max_range / self.speed_of_sound  # 最大往返回波时间
        self.pending_reflections = []  # 存储在场中飞行的反射事件
        self.raise_on_crosstalk = False  # 调试阶段可以手动设为 True
        
        # 统计信息
        self.crosstalk_count = 0
        self.total_sensor_firings = 0
        
        # 触发管理器（使用新的TriggerManager）
        # 注意：必须在sonar_readings初始化之后调用，因为greedy模式会使用它
        self.randomize_trigger = randomize_trigger
        self._init_trigger_manager(trigger_mode, randomize_trigger)
        
        # 跟踪本帧哪些传感器被扫描过（用于occupancy grid更新）
        self.active_sensors_this_frame = set()
        
        # 障碍物
        self.obstacles = []
        self._have_map = False
        
        # 诊断信息
        self.step_counter = 0
        self._collision_occurred = False
        self._stuck_counter = 0
        
        # 缓存的转换
        self.width = int(self.world_width * self.pixel_per_meter)
        self.height = int(self.world_height * self.pixel_per_meter)
        
        # Fisher计算器（超声波雷达专用）
        self.fisher_calc = SonarFisherCalculator(
            num_sensors=self.num_sensors,
            sensor_spacing=360.0 / self.num_sensors,
            sensor_fov=self.sensor_fov,
            max_range=self.sensor_max_range
        )
        
        # 多样化地图生成器
        self.map_generator = DiverseMapGenerator(
            world_width=self.world_width,
            world_height=self.world_height
        )
        self._map_seed = 0  # 当前地图种子
        self._current_scene_type = None  # 当前场景类型
    
    # -------- 触发管理器 -------- #
    
    def _init_trigger_manager(self, trigger_mode: str, randomize: bool):
        """初始化触发管理器"""
        # 选择触发配置
        if randomize:
            trigger_config = DATA_COLLECTION_TRIGGER_CONFIG
        else:
            trigger_config = DEFAULT_TRIGGER_CONFIG
        
        # 创建触发管理器
        initial_mode = TriggerMode.from_string(trigger_mode) if not randomize else None
        self.trigger_manager = TriggerManager(
            num_sensors=self.num_sensors if hasattr(self, 'num_sensors') else 12,
            config=trigger_config,
            initial_mode=initial_mode
        )
        
        # 如果需要随机化，立即随机选择模式
        if randomize:
            self.trigger_manager.set_random_mode()
        
        # 保持向后兼容的trigger_mode属性
        self.trigger_mode = str(self.trigger_manager.mode)
        
        # 打印触发信息
        info = self.trigger_manager.get_mode_info()
        print(f"触发模式: {info['mode']} ({info['description']})")
        if randomize:
            print("  (随机选择模式已启用)")
    
    @property
    def current_trigger_mode(self) -> TriggerMode:
        """当前触发模式"""
        return self.trigger_manager.mode
    
    def set_trigger_mode(self, mode: str) -> None:
        """设置触发模式"""
        self.trigger_manager.set_mode_from_string(mode)
        self.trigger_mode = str(self.trigger_manager.mode)
    
    def randomize_trigger_mode(self) -> str:
        """随机选择触发模式"""
        mode = self.trigger_manager.set_random_mode()
        self.trigger_mode = str(mode)
        return self.trigger_mode
    
    # -------- 传感器初始化 -------- #
    
    def _init_sensors(self):
        """初始化12个均匀分布的传感器"""
        self.sensors.clear()
        angle_step = 360.0 / self.num_sensors
        
        for i in range(self.num_sensors):
            angle = i * angle_step  # 传感器相对机器人的角度
            angle_rad = math.radians(angle)
            
            # 计算传感器在机器人坐标系中的偏移
            offset_x = self.sensor_ring_radius * math.cos(angle_rad)
            offset_y = self.sensor_ring_radius * math.sin(angle_rad)
            
            sensor = SonarSensor(
                id=i,
                angle=angle,
                offset_x=offset_x,
                offset_y=offset_y,
                fov_angle=self.sensor_fov,
                max_range=self.sensor_max_range
            )
            self.sensors.append(sensor)
        
        print(f"Initialized {self.num_sensors} sonar sensors in a ring (radius={self.sensor_ring_radius}m)")
    
    # -------- 公共API -------- #
    
    def reset(self, regenerate_map: bool = True, seed: Optional[int] = None, 
              scene_type: Optional[str] = None) -> None:
        """
        重置环境
        
        参数:
            regenerate_map: 是否重新生成地图
            seed: 随机种子（None则使用随机种子）
            scene_type: 指定场景类型（None则随机选择）
                可选值: 'sparse', 'dense', 'clustered', 'corridor', 
                       'rooms', 'mixed', 'maze_like', 'open_center'
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._map_seed = seed
        else:
            self._map_seed = random.randint(0, 1000000)
        
        if regenerate_map or not self._have_map:
            self._gen_obstacles(scene_type=scene_type)
            self._have_map = True
        
        self.robot_pos = self._find_safe_start()
        self.robot_angle = float(random.randint(0, 360))
        self._robot_angle_rad_cache = math.radians(self.robot_angle)
        
        self.velocity = 0.0
        self.angular_velocity = 0.0
        
        self.sim_time = 0.0
        
        self.feature_map.fill(0.0)
        self.global_feature_map.fill(0.0)
        self.sonar_readings.fill(self.sensor_max_range)
        self.sensor_ready_times.fill(0.0)
        self.next_allowed_trigger_time = 0.0
        
        self.crosstalk_count = 0
        self.total_sensor_firings = 0
        
        # 重置触发管理器
        if hasattr(self, 'trigger_manager'):
            self.trigger_manager.reset()
        
        self._collision_occurred = False
        self._stuck_counter = 0
        self.step_counter = 0
    
    def set_velocity(self, linear_vel: float, angular_vel: float) -> None:
        """设置机器人速度"""
        self.velocity = clamp(float(linear_vel), -5.0, 5.0)
        self.angular_velocity = clamp(float(angular_vel), -1.5, 1.5)
    
    def step(self) -> None:
        """执行一步仿真"""
        prev_pos = self.robot_pos.copy()
        self._collision_occurred = False
        
        # 更新时间
        self.sim_time += self.dt
        
        # 更新位姿
        self._update_robot()
        
        # 检测碰撞（通过位移判断）
        moved = np.linalg.norm(self.robot_pos - prev_pos)
        if moved < 0.01:
            self._collision_occurred = True
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        
        # 清空本帧活跃传感器记录（在开始时清空）
        self.active_sensors_this_frame.clear()
        
        # 尝试触发传感器 (基于就绪状态)
        self._process_sensor_trigger()
        
        self.step_counter += 1
    
    def update_maps(self) -> None:
        """更新Fisher信息地图"""
        self._apply_feature_decay()
        self._detect_and_add_features_to_global_map()
        self._extract_local_feature_map()
    
    def state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'position': self.robot_pos.copy(),
            'angle': float(self.robot_angle),
            'linear_velocity': float(self.velocity),
            'angular_velocity': float(self.angular_velocity),
            'sonar_readings': self.sonar_readings.copy(),
            'step_counter': int(self.step_counter),
            'collision_occurred': bool(self._collision_occurred),
            'stuck_counter': int(self._stuck_counter),
            'sim_time': float(self.sim_time)
        }
    
    def fisher_map_stats(self) -> Dict[str, float]:
        """Fisher地图统计信息"""
        flat = self.feature_map.ravel()
        nz = flat[flat > 0]
        if nz.size == 0:
            return {'mean_fisher': 0.0, 'total_features': 0.0, 'density': 0.0}
        return {
            'mean_fisher': float(nz.mean()),
            'total_features': float(nz.size),
            'density': float(nz.size) / float(flat.size)
        }
    
    # -------- 内部实现 -------- #
    
    def _inside_world(self, x: float, y: float) -> bool:
        """检查点是否在世界范围内"""
        return 0.0 <= x < self.world_width and 0.0 <= y < self.world_height
    
    def _gen_obstacles(self, scene_type: Optional[str] = None):
        """
        生成障碍物（使用多样化地图生成器）
        
        参数:
            scene_type: 场景类型，None则随机选择
        """
        self.obstacles.clear()
        
        # 使用 DiverseMapGenerator 生成障碍物
        self.obstacles = self.map_generator.generate_obstacles(
            seed=self._map_seed,
            scene_type=scene_type
        )
        
        # 记录当前场景类型
        self._current_scene_type = self.map_generator.get_scene_type_from_seed(self._map_seed) if scene_type is None else scene_type
        
        print(f"🗺️  生成地图: {self._current_scene_type} (seed={self._map_seed}, 障碍物数={len(self.obstacles)})")
    
    def _find_safe_start(self) -> np.ndarray:
        """寻找安全的起始位置"""
        margin = self.robot_size + 0.5
        for _ in range(100):
            p = np.array([
                random.uniform(margin, self.world_width - margin),
                random.uniform(margin, self.world_height - margin)
            ], dtype=np.float64)
            if self._position_safe(p):
                return p
        
        # 备用：中心位置
        c = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float64)
        return c
    
    def _position_safe(self, pos: np.ndarray) -> bool:
        """检查位置是否安全"""
        if (pos[0] < self.robot_size or pos[0] > self.world_width - self.robot_size or
            pos[1] < self.robot_size or pos[1] > self.world_height - self.robot_size):
            return False
        return not self._collide_at(pos)
    
    def _point_in_obstacle(self, x: float, y: float) -> bool:
        """检查点是否在障碍物内"""
        for kind, data in self.obstacles:
            if kind == 'rect':
                ox, oy, w, h = data
                if ox <= x <= ox + w and oy <= y <= oy + h:
                    return True
        return False
    
    def _collide_at(self, target_pos: np.ndarray) -> bool:
        """检查位置是否碰撞"""
        rx, ry = target_pos[0], target_pos[1]
        rs = self.robot_size
        for kind, data in self.obstacles:
            if kind == 'rect':
                x, y, w, h = data
                cx = clamp(rx, x, x + w)
                cy = clamp(ry, y, y + h)
                if math.hypot(rx - cx, ry - cy) < rs:
                    return True
        return False
    
    def _update_robot(self):
        """更新机器人位姿"""
        # 使用缓存的弧度值计算位置增量
        delta_x = math.cos(self._robot_angle_rad_cache) * self.velocity * self.dt
        delta_y = math.sin(self._robot_angle_rad_cache) * self.velocity * self.dt
        
        # 更新位置
        new_pos = self.robot_pos.copy()
        new_pos[0] = clamp(new_pos[0] + delta_x, self.robot_size, self.world_width - self.robot_size)
        new_pos[1] = clamp(new_pos[1] + delta_y, self.robot_size, self.world_height - self.robot_size)
        
        if self._collide_at(new_pos):
            self._handle_collision()
        else:
            self.robot_pos = new_pos
            # 更新角度和缓存
            self.robot_angle = (self.robot_angle + math.degrees(self.angular_velocity * self.dt)) % 360.0
            self._robot_angle_rad_cache = math.radians(self.robot_angle)
    
    def _handle_collision(self):
        """处理碰撞"""
        self.velocity = clamp(-self.velocity * random.uniform(0.5, 1.0) + random.uniform(-0.5, 0.5),
                              -self.max_linear_velocity, self.max_linear_velocity)
        if abs(self.angular_velocity) < 0.1:
            self.angular_velocity = random.choice([-1, 1]) * random.uniform(0.3, 0.8)
        else:
            self.angular_velocity = clamp(-self.angular_velocity + random.uniform(-0.2, 0.2),
                                          -self.max_angular_velocity, self.max_angular_velocity)
    
    # -------- 传感器扫描 -------- #
    
    def _get_active_sensor_ids(self) -> List[int]:
        """获取当前帧应激活的传感器ID列表"""
        return self.trigger_manager.get_active_sensors()
    
    def _process_sensor_trigger(self, force_ids: List[int] = None):
        """处理传感器触发逻辑 (基于就绪状态)"""
        if force_ids is not None:
            candidate_ids = force_ids
        else:
            # 创建触发上下文
            context = TriggerContext(
                step_count=self.step_counter,
                sim_time=self.sim_time,
                sonar_readings=self.sonar_readings,
                robot_pos=self.robot_pos,
                robot_angle=self.robot_angle,
                sensors=self.sensors,
                sensor_ready_times=self.sensor_ready_times,
                global_feature_map=self.global_feature_map,
                feature_map_resolution=self.feature_map_resolution,
                world_dims=(self.world_width, self.world_height),
                sensor_max_range=self.sensor_max_range
            )
            
            # 获取策略建议激活的传感器ID
            candidate_ids = self.trigger_manager.get_active_sensors(context)
        
        if not candidate_ids:
            return
            
        # 检查哪些候选传感器已就绪
        active_ids = [sid for sid in candidate_ids if self.sim_time >= self.sensor_ready_times[sid]]
        
        if not active_ids:
            return
            
        # 检查全局触发间隔 (控制扫描频率上限)
        # 注意：对于 Greedy 和 RL 模式，我们希望尽可能快地触发（由物理 flight_time 驱动）
        # 因此在这两种模式下，我们忽略 sensor_trigger_interval 的限制
        current_mode = self.trigger_manager.mode
        is_async_mode = (str(current_mode) in ['greedy', 'rl'])
        
        if not is_async_mode and self.sim_time < self.next_allowed_trigger_time:
            return # 等待全局间隔
            
        # 执行扫描
        # 更新全局下一次允许触发的时间
        # 间隔 = 步数 * dt
        # 仅在非异步模式下更新此限制
        if not is_async_mode:
            self.next_allowed_trigger_time = self.sim_time + (self.sensor_trigger_interval * self.dt)
        
        # 扫描激活的传感器
        for sensor_id in active_ids:
            self.total_sensor_firings += 1
            sensor = self.sensors[sensor_id]
            distance = self._scan_single_sensor(sensor)
            self.sonar_readings[sensor_id] = distance
            self.active_sensors_this_frame.add(sensor_id)
            
            # 更新该传感器的下一次就绪时间
            # 就绪时间 = 当前时间 + 飞行时间 (2 * distance / c)
            # 增加一个极小的缓冲时间 (e.g. 10ms) 模拟硬件处理延迟
            flight_time = max(0.01, 2.0 * distance / self.speed_of_sound)
            self.sensor_ready_times[sensor_id] = self.sim_time + flight_time
        
        # 新增：先登记反射事件（记录发射时间和反射点）
        self._register_reflection_events(active_ids)

        # 修改后的串扰逻辑会使用 pending_reflections + 时间轴
        self._simulate_reflection_interference(active_ids)
        
        # 前进到下一个触发状态
        self.trigger_manager.advance()
    
    def _scan_single_sensor(self, sensor: SonarSensor) -> float:
        """扫描单个传感器，返回最近障碍物距离"""
        # 获取传感器世界位置和朝向
        sensor_pos = sensor.get_world_position(self.robot_pos, self.robot_angle)
        sensor_angle = sensor.get_world_angle(self.robot_angle)
        
        # 扫描FoV范围内的射线
        min_distance = sensor.max_range
        half_fov = sensor.fov_angle / 2.0
        
        # 使用多条射线扫描FoV
        num_rays = 9  # 每个传感器9条射线
        for i in range(num_rays):
            # 计算射线角度
            if num_rays == 1:
                ray_angle = sensor_angle
            else:
                offset = -half_fov + (i / (num_rays - 1)) * sensor.fov_angle
                ray_angle = (sensor_angle + offset) % 360.0
            
            # 沿射线前进，检测障碍物
            ray_angle_rad = math.radians(ray_angle)
            step = 0.1  # 步进距离
            distance = 0.0
            
            while distance < sensor.max_range:
                distance += step
                wx = sensor_pos[0] + math.cos(ray_angle_rad) * distance
                wy = sensor_pos[1] + math.sin(ray_angle_rad) * distance
                
                # 检查是否出界或碰到障碍物
                if not self._inside_world(wx, wy) or self._point_in_obstacle(wx, wy):
                    min_distance = min(min_distance, distance)
                    break
        
        return min_distance
    
    # -------- Fisher地图更新 -------- #
    
    def _apply_feature_decay(self):
        """应用特征衰减"""
        self.global_feature_map *= (1.0 - 5e-6)
        self.global_feature_map[self.global_feature_map < 0.1] = 0.0
    
    def _detect_and_add_features_to_global_map(self):
        """从传感器读数中检测并添加特征到全局地图 - 仅处理本帧扫描过的传感器"""
        # 只处理在本帧实际被扫描过的传感器，避免使用旧数据
        for sensor_id in self.active_sensors_this_frame:
            sensor = self.sensors[sensor_id]
            distance = self.sonar_readings[sensor.id]
            
            # 如果检测到障碍物（距离在有效范围内且未受串扰污染）
            if 0 < distance < sensor.max_range:
                # 获取传感器世界位置和朝向
                sensor_pos = sensor.get_world_position(self.robot_pos, self.robot_angle)
                sensor_angle = sensor.get_world_angle(self.robot_angle)
                
                # 计算障碍物位置
                angle_rad = math.radians(sensor_angle)
                wx = sensor_pos[0] + math.cos(angle_rad) * distance
                wy = sensor_pos[1] + math.sin(angle_rad) * distance
                
                # 计算Fisher信息
                fisher = self._fisher_at(wx, wy, distance, sensor_angle)
                
                # 添加到全局地图
                self._add_global_feature(wx, wy, fisher)
    
    def _fisher_at(self, wx: float, wy: float, distance: float, angle_deg: float) -> float:
        """
        计算特定位置的Fisher信息值
        
        超声波雷达版本：
        - 考虑距离因子（近距离更可靠）
        - 考虑传感器覆盖度（重叠区域更可信）
        - 不考虑FOV质量（超声波只有距离信息）
        """
        # 计算相对机器人的角度（归一化到0-360）
        relative_angle = (angle_deg - self.robot_angle) % 360.0
        
        # 使用超声波专用Fisher计算器
        return self.fisher_calc.compute(
            distance=distance,
            angle_deg=relative_angle
        )
    
    def _add_global_feature(self, wx: float, wy: float, val: float):
        """将特征添加到全局地图"""
        add_global_feature(
            global_map=self.global_feature_map,
            wx=wx, wy=wy,
            fisher_value=val,
            map_size=self.global_feature_map_size,
            resolution=self.feature_map_resolution,
            world_width=self.world_width,
            world_height=self.world_height,
            spread_neighbors=True
        )
    
    def _extract_local_feature_map(self):
        """提取以机器人为中心的局部特征地图"""
        m = self.global_feature_map
        size = self.global_feature_map_size
        res = self.feature_map_resolution
        half = self.feature_map_size // 2
        
        rx = int(self.robot_pos[0] / res + size // 2 - self.world_width // (2 * res))
        ry = int(self.robot_pos[1] / res + size // 2 - self.world_height // (2 * res))
        
        gx0, gy0 = rx - half, ry - half
        gx1, gy1 = gx0 + self.feature_map_size, gy0 + self.feature_map_size
        
        sx0 = max(0, -gx0)
        sy0 = max(0, -gy0)
        sx1 = self.feature_map_size - max(0, gx1 - size)
        sy1 = self.feature_map_size - max(0, gy1 - size)
        
        Gx0 = max(0, gx0)
        Gy0 = max(0, gy0)
        Gx1 = min(size, gx1)
        Gy1 = min(size, gy1)
        
        self.feature_map.fill(0.0)
        if sx0 < sx1 and sy0 < sy1:
            self.feature_map[sy0:sy1, sx0:sx1] = m[Gy0:Gy1, Gx0:Gx1]
        
        # 轻微模糊
        self.feature_map = cv2.GaussianBlur(self.feature_map, (3, 3), 0.5)
            
    def _simulate_reflection_interference(self, active_ids: List[int]) -> None:
        """
        带声速和时间轴的反射串扰模型：
        - pending_reflections 保存过去若干帧发射产生的“反射事件”
        - 当前帧 active 的传感器，如果在自己的监听时间窗内遇到这些回波，就可能被串扰
        """
        if not active_ids or not self.pending_reflections:
            return

        cur_t = self.sim_time

        # 1. 丢弃已经超出最大回波时间的事件
        self.pending_reflections = [
            e for e in self.pending_reflections
            if cur_t - e["emit_time"] <= self.max_echo_time
        ]
        if not self.pending_reflections:
            return

        # 2. 遍历当前帧的“受害传感器”
        for victim_id in active_ids:
            victim_sensor = self.sensors[victim_id]
            victim_pos = victim_sensor.get_world_position(self.robot_pos, self.robot_angle)
            victim_angle = victim_sensor.get_world_angle(self.robot_angle)

            for event in self.pending_reflections:
                # 如需忽略自身反射，可以跳过同一 ID
                if event["source_id"] == victim_id:
                    continue

                hit_x = event["hit_x"]
                hit_y = event["hit_y"]
                dist_src_hit = event["dist_src_hit"]
                emit_time = event["emit_time"]

                # 源传感器 -> 障碍点到达时间
                hit_time = emit_time + dist_src_hit / self.speed_of_sound

                # 障碍点 -> 受害传感器距离与时间
                dist_hit_to_victim = math.hypot(hit_x - victim_pos[0], hit_y - victim_pos[1])
                arrive_time = hit_time + dist_hit_to_victim / self.speed_of_sound

                # 监听窗口检查：
                # 传感器在 cur_t 发射，在 cur_t + flight_time 收到真实回波。
                # 如果干扰波在 (cur_t, cur_t + flight_time) 之间到达，则可能被误判为真实回波。
                # 注意：flight_time 是基于当前真实探测距离计算的。
                
                current_true_dist = self.sonar_readings[victim_id]
                true_echo_time = cur_t + 2.0 * current_true_dist / self.speed_of_sound
                
                # 干扰必须在发射之后，且在真实回波之前到达
                if not (cur_t < arrive_time < true_echo_time):
                    continue

                # FOV 检查：反射点是否在受害者视场以内
                angle_to_hit = math.degrees(math.atan2(hit_y - victim_pos[1], hit_x - victim_pos[0]))
                angle_diff = abs(angdiff_deg(angle_to_hit, victim_angle))
                if angle_diff > victim_sensor.fov_angle / 2.0:
                    continue

                # 总路径长度（A->hit + hit->B）
                total_path_length = dist_src_hit + dist_hit_to_victim

                # 超出量程或比当前读数更远则跳过
                if total_path_length >= self.sensor_max_range:
                    continue
                if total_path_length >= self.sonar_readings[victim_id]:
                    continue

                # 概率模型：距离越远串扰概率越低（可以沿用原先的平方衰减）
                prob = 1.0 - (total_path_length / self.sensor_max_range) ** 2
                if random.random() < prob:
                    self.crosstalk_count += 1
                    msg = (
                        f"[CROSSTALK ERROR] source={event['source_id']} -> victim={victim_id}, "
                        f"path={total_path_length:.3f} m, arrive_time={arrive_time:.6f} s"
                    )

                    # 打印错误信息到 stderr
                    # print(msg, file=sys.stderr) # 暂时注释掉，避免刷屏

                    # 如开启严格模式，则直接抛异常
                    if getattr(self, "raise_on_crosstalk", False):
                        raise RuntimeError(msg)

                    # 将读数标记为无效值 (-1.0)，表示数据因串扰而损坏
                    self.sonar_readings[victim_id] = -1.0

    def _register_reflection_events(self, active_ids: List[int]) -> None:
        """为本帧所有激活的传感器登记反射事件（只记录几何+时间，不立刻做串扰）"""
        emit_time = self.sim_time  # 约定当前仿真时间是发射时间

        for source_id in active_ids:
            source_dist = float(self.sonar_readings[source_id])
            # 没有打到障碍物（接近最大量程），则不产生反射事件
            if source_dist >= self.sensor_max_range * 0.99:
                continue

            source_sensor = self.sensors[source_id]
            source_pos = source_sensor.get_world_position(self.robot_pos, self.robot_angle)
            source_angle = source_sensor.get_world_angle(self.robot_angle)
            angle_rad = math.radians(source_angle)

            # 沿波束中心线计算障碍点位置
            hit_x = source_pos[0] + math.cos(angle_rad) * source_dist
            hit_y = source_pos[1] + math.sin(angle_rad) * source_dist

            self.pending_reflections.append({
                "source_id": source_id,
                "hit_x": hit_x,
                "hit_y": hit_y,
                "dist_src_hit": source_dist,
                "emit_time": emit_time
            })


# ------------------------------- 渲染器 -------------------------------- #

class RingSonarRenderer:
    """
    环形超声波雷达渲染器 - 2D俯视图
    显示：机器人、传感器布局、FoV扇区、障碍物、Fisher地图、栅格占用图
    """
    
    def __init__(self, core: RingSonarCore, render_mode: Optional[str] = "human", enable_prediction: bool = True):
        self.core = core
        self.render_mode = render_mode
        self.enable_prediction = enable_prediction  # 是否启用障碍物预测
        self.world_img = np.ones((core.height, core.width, 3), dtype=np.uint8) * 255
        
        # 全局栅格占用图 (Global Occupancy Grid Map)
        # 分辨率：每个栅格代表0.1米
        self.grid_resolution = 0.1  # 米/栅格
        self.grid_width = int(core.world_width / self.grid_resolution)
        self.grid_height = int(core.world_height / self.grid_resolution)
        
        # 占用栅格地图 (0=障碍物, 127=未知, 255=无障碍)
        self.occupancy_grid = np.ones((self.grid_height, self.grid_width), dtype=np.uint8) * 127
        # 访问计数：记录每个栅格被扫描的次数
        self.visit_count = np.zeros((self.grid_height, self.grid_width), dtype=np.uint16)
        
        # 障碍物预测地图 (0-255: 0=确定无障碍, 255=确定有障碍)
        self.obstacle_prediction = np.ones((self.grid_height, self.grid_width), dtype=np.uint8) * 127
        # 预测置信度 (0-100: 置信度百分比)
        self.prediction_confidence = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # 全局地图预测模型相关
        self.sequence_length = 5  # 与训练时保持一致
        self.model = None
        self.frame_buffer = []  # 存储历史帧用于时间序列
        self.device = torch.device('cpu')  # 默认设备
        
        # 只有在启用预测时才加载模型
        if self.enable_prediction:
            # 全局地图预测模型配置
            self.in_channels = 8  # 5帧局部观测 + 全局累积 + 访问计数 + 已知掩码
            
            # 智能设备选择：优先GPU，如果GPU内存不足则回退到CPU
            if torch.cuda.is_available():
                try:
                    print("测试GPU内存是否足够进行全局地图预测推理...")
                    torch.cuda.empty_cache()
                    
                    # 创建模型并尝试加载到GPU
                    test_model = GlobalMapPredictor(
                        in_channels=self.in_channels, 
                        base_channels=32
                    ).cuda()
                    
                    # 加载权重
                    checkpoint = torch.load('./checkpoints/global_map_model.pth', map_location='cuda')
                    if 'model_state_dict' in checkpoint:
                        test_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        test_model.load_state_dict(checkpoint)
                    
                    # 尝试一次完整的推理
                    test_input = torch.randn(1, self.in_channels, 400, 400).cuda()
                    test_model.eval()
                    with torch.no_grad():
                        test_output = test_model(test_input)
                    
                    # 清理测试资源
                    del test_model, test_input, test_output, checkpoint
                    torch.cuda.empty_cache()
                    
                    self.device = torch.device('cuda')
                    print("✅ GPU内存充足，使用GPU进行全局地图预测推理")
                except RuntimeError as e:
                    print(f"❌ GPU内存不足: {e}，回退到CPU推理")
                    torch.cuda.empty_cache()
                    self.device = torch.device('cpu')
                except FileNotFoundError as e:
                    print(f"⚠️ 模型文件未找到: {e}")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
                print("未检测到GPU，使用CPU推理")
            
            # 尝试加载训练好的模型
            self._load_model()
        else:
            print("📊 数据收集")
    
    def _load_model(self):
        """加载训练好的全局地图预测模型"""
        if not MODEL_AVAILABLE:
            print("全局地图预测模型不可用，将使用传统扩散预测")
            return
        
        model_path = './checkpoints/global_map_model.pth'
        if not os.path.exists(model_path):
            print(f"⚠️ 模型文件不存在: {model_path}")
            print("请先运行 train_global_map_model.py 训练模型")
            print("将使用传统扩散预测方法")
            self.model = None
            return
            
        try:
            # 创建模型实例 - 使用与训练时相同的参数
            self.model = GlobalMapPredictor(
                in_channels=self.in_channels, 
                base_channels=32  # 匹配训练时的默认参数
            ).to(self.device)
            
            # 加载训练好的权重
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"✅ 成功加载全局地图预测模型: {model_path}")
            print(f"   推理设备: {self.device}")
            print(f"   模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("将使用传统扩散预测方法")
            self.model = None
    
    def render(self):
        if self.render_mode is None:
            return None
        
        self.world_img.fill(255)
        self._draw_obstacles()
        self._draw_sensor_fov()
        self._draw_robot()
        self._draw_sensor_readings()
        self._draw_trigger_info()  # 显示触发模式信息
        
        # 仅在有新扫描数据时更新占用栅格
        if len(self.core.active_sensors_this_frame) > 0:
            self._update_occupancy_grid()
        
        if self.render_mode == "human":
            self._show_windows()
        elif self.render_mode == "rgb_array":
            return self.world_img.copy()
    
    def _w2p(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界坐标转像素坐标"""
        return int(wx * self.core.pixel_per_meter), int(wy * self.core.pixel_per_meter)
    
    def _draw_direction_arrow(self, img: np.ndarray, cx: int, cy: int, 
                             angle_rad: float, velocity: float, scale: float = 1.0):
        """绘制移动方向箭头（统一方法）"""
        if abs(velocity) > 0.01:  # 移动中
            speed_factor = min(abs(velocity) / 2.0, 1.0)
            arrow_len = int(18 * scale * (0.5 + 0.5 * speed_factor))
            
            if velocity > 0:  # 前进
                color = (0, 0, 255)  # 红色
                thickness = 3
            else:  # 后退
                color = (0, 165, 255)  # 橙色
                thickness = 3
                angle_rad += math.pi
                
            ex = int(cx + math.cos(angle_rad) * arrow_len)
            ey = int(cy + math.sin(angle_rad) * arrow_len)
            cv2.arrowedLine(img, (cx, cy), (ex, ey), color, thickness)
        else:  # 静止
            arrow_len = int(18 * scale * 0.6)
            ex = int(cx + math.cos(angle_rad) * arrow_len)
            ey = int(cy + math.sin(angle_rad) * arrow_len)
            cv2.arrowedLine(img, (cx, cy), (ex, ey), (255, 200, 100), 2)
    
    def _draw_obstacles(self):
        """绘制障碍物"""
        ppm = self.core.pixel_per_meter
        for kind, data in self.core.obstacles:
            if kind == 'rect':
                x, y, w, h = data
                px, py = int(x * ppm), int(y * ppm)
                pw, ph = int(w * ppm), int(h * ppm)
                cv2.rectangle(self.world_img, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
    
    def _draw_robot(self):
        """绘制机器人和传感器布局"""
        ppm = self.core.pixel_per_meter
        cx, cy = self._w2p(self.core.robot_pos[0], self.core.robot_pos[1])
        
        # 绘制机器人主体
        r = int(self.core.robot_size * ppm)
        cv2.circle(self.world_img, (cx, cy), r, (0, 255, 0), -1)
        
        # 绘制移动方向箭头
        angle_rad = math.radians(self.core.robot_angle)
        self._draw_direction_arrow(self.world_img, cx, cy, angle_rad, 
                                   self.core.velocity, ppm / 20)
        
        # 绘制传感器环
        ring_r = int(self.core.sensor_ring_radius * ppm)
        cv2.circle(self.world_img, (cx, cy), ring_r, (150, 150, 150), 1)
        
        # 绘制每个传感器
        for sensor in self.core.sensors:
            sx, sy = sensor.get_world_position(self.core.robot_pos, self.core.robot_angle)
            sx_pix, sy_pix = self._w2p(sx, sy)
            cv2.circle(self.world_img, (sx_pix, sy_pix), 3, (255, 0, 255), -1)
    
    def _draw_sensor_fov(self):
        """绘制传感器的FoV扇区（根据触发模式）"""
        ppm = self.core.pixel_per_meter
        overlay = np.zeros_like(self.world_img)
        
        # 使用上一帧激活的传感器（从core的记录中获取）
        active_sensor_ids = list(self.core.active_sensors_this_frame)
        
        for sensor in self.core.sensors:
            # 只绘制激活的传感器
            if sensor.id not in active_sensor_ids:
                continue
            # 获取传感器世界位置和朝向
            sx, sy = sensor.get_world_position(self.core.robot_pos, self.core.robot_angle)
            sensor_angle = sensor.get_world_angle(self.core.robot_angle)
            
            sx_pix, sy_pix = self._w2p(sx, sy)
            
            # 创建扇形的点列表
            fov_pts = [(sx_pix, sy_pix)]
            half_fov = sensor.fov_angle / 2.0
            
            # 使用当前传感器读数作为绘制范围
            display_range = min(self.core.sonar_readings[sensor.id], sensor.max_range)
            range_pix = int(display_range * ppm)
            
            # 绘制扇形边缘
            for angle_offset in np.linspace(-half_fov, half_fov, 20):
                angle = math.radians(sensor_angle + angle_offset)
                ex = int(sx_pix + math.cos(angle) * range_pix)
                ey = int(sy_pix + math.sin(angle) * range_pix)
                ex = clamp(ex, 0, self.core.width - 1)
                ey = clamp(ey, 0, self.core.height - 1)
                fov_pts.append((ex, ey))
            
            # 填充扇形
            if len(fov_pts) > 2:
                pts = np.array(fov_pts, dtype=np.int32)
                # 根据距离选择颜色（近距离红色，远距离蓝色）
                if display_range < sensor.max_range * 0.5:
                    color = (100, 100, 255)  # 红色偏向
                else:
                    color = (255, 150, 100)  # 蓝色偏向
                cv2.fillPoly(overlay, [pts], color)
        
        # 混合overlay
        cv2.addWeighted(self.world_img, 0.7, overlay, 0.3, 0, self.world_img)
    
    def _draw_sensor_readings(self):
        """在图像上绘制传感器读数文本 - 仅显示本帧实际扫描的传感器"""
        ppm = self.core.pixel_per_meter
        
        # 只显示本帧实际扫描过的传感器读数
        for sensor_id in self.core.active_sensors_this_frame:
            sensor = self.core.sensors[sensor_id]
            sx, sy = sensor.get_world_position(self.core.robot_pos, self.core.robot_angle)
            sx_pix, sy_pix = self._w2p(sx, sy)
            
            # 显示距离读数
            distance = self.core.sonar_readings[sensor.id]
            text = f"{distance:.1f}"
            
            # 文本位置稍微偏移
            text_x = sx_pix + 5
            text_y = sy_pix - 5
            
            cv2.putText(self.world_img, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    def _draw_trigger_info(self):
        """显示触发模式信息"""
        active_ids = list(self.core.active_sensors_this_frame)
        
        # 根据触发模式显示不同信息
        mode_info = {
            "sector": ("Sector Polling (3 sensors)", "Warning: May have interference", (380, 85)),
            "sequential": ("Sequential (1 sensor)", "Status: No interference!", (350, 85)),
            "interleaved": ("Interleaved (1 sensor)", "Status: 60deg spacing, minimal interference", (420, 85)),
            "all": ("All Sensors (12 sensors)", "Warning: High interference in real world", (420, 60))
        }
        
        mode, status, box_size = mode_info.get(self.core.trigger_mode, 
                                               ("Unknown Mode", "Unknown status", (420, 85)))
        
        # 绘制信息框
        cv2.rectangle(self.world_img, (5, 5), box_size, (255, 255, 255), -1)
        cv2.rectangle(self.world_img, (5, 5), box_size, (0, 0, 0), 2)
        
        # 显示触发模式
        cv2.putText(self.world_img, f"Trigger: {mode}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        
        # 显示活跃传感器（如果数量少于4个）
        if len(active_ids) <= 4:
            angles = [f"{sid * 30}deg" for sid in active_ids]
            sensor_text = f"Active: {active_ids} ({', '.join(angles)})"
            cv2.putText(self.world_img, sensor_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
        
        # 显示状态
        color = (0, 200, 0) if "No interference" in status else (0, 100, 200)
        y_pos = 75 if len(active_ids) <= 4 else 50
        cv2.putText(self.world_img, status, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _update_occupancy_grid(self):
        """更新全局栅格占用图 - 优化版本，使用NumPy数组操作"""
        # 创建临时标记数组（0=未处理, 1=无障碍, 2=障碍物）
        temp_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # 只使用本帧实际扫描过的传感器
        for sensor in self.core.sensors:
            if sensor.id not in self.core.active_sensors_this_frame:
                continue
                
            # 获取传感器世界位置和朝向
            sx, sy = sensor.get_world_position(self.core.robot_pos, self.core.robot_angle)
            sensor_angle = sensor.get_world_angle(self.core.robot_angle)
            detected_distance = self.core.sonar_readings[sensor.id]
            
            # 圆锥形FoV扫描
            half_fov = sensor.fov_angle / 2.0
            num_angle_steps = max(3, int(sensor.fov_angle / 2))
            
            for angle_offset in np.linspace(-half_fov, half_fov, num_angle_steps):
                ray_angle = math.radians(sensor_angle + angle_offset)
                cos_ray = math.cos(ray_angle)
                sin_ray = math.sin(ray_angle)
                
                # 沿射线标记无障碍区域
                num_steps = max(2, int(detected_distance / (self.grid_resolution * 0.5)))
                distances = np.linspace(0, detected_distance, num_steps)
                
                wx = sx + cos_ray * distances
                wy = sy + sin_ray * distances
                
                gx = (wx / self.grid_resolution).astype(int)
                gy = (wy / self.grid_resolution).astype(int)
                
                # 过滤边界内的点
                valid = (gx >= 0) & (gx < self.grid_width) & (gy >= 0) & (gy < self.grid_height)
                temp_grid[gy[valid], gx[valid]] = 1  # 标记为无障碍
                
                # 标记障碍物（如果未达到最大距离）
                if detected_distance < sensor.max_range * 0.95:
                    wx_obs = sx + cos_ray * detected_distance
                    wy_obs = sy + sin_ray * detected_distance
                    gx_obs = int(wx_obs / self.grid_resolution)
                    gy_obs = int(wy_obs / self.grid_resolution)
                    
                    if 0 <= gx_obs < self.grid_width and 0 <= gy_obs < self.grid_height:
                        if temp_grid[gy_obs, gx_obs] == 0:  # 只在未标记为无障碍时设置
                            temp_grid[gy_obs, gx_obs] = 2  # 标记为障碍物
        
        # 批量更新占用栅格
        free_mask = (temp_grid == 1)
        obstacle_mask = (temp_grid == 2)
        
        self.occupancy_grid[free_mask] = 255
        self.occupancy_grid[obstacle_mask] = 50
        
        # 更新访问计数（防止溢出）
        visited_mask = (temp_grid > 0)
        self.visit_count[visited_mask] = np.minimum(65535, self.visit_count[visited_mask] + 1)
        
        # 管理帧缓冲区（用于时空推理）
        self._update_frame_buffer()
        
        # 更新障碍物预测
        self._predict_obstacles()
    
    def _update_frame_buffer(self):
        """更新帧缓冲区，用于时空推理"""
        # 创建当前帧的数据
        current_frame = {
            'occupancy': self.occupancy_grid.copy(),
            'visit_count': self.visit_count.copy(),
            'robot_pos': self.core.robot_pos.copy(),
            'step': self.core.step_counter
        }
        
        # 添加到缓冲区
        self.frame_buffer.append(current_frame)
        
        # 保持缓冲区大小为sequence_length
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
    
    def _predict_obstacles(self):
        """
        基于全局地图预测模型预测障碍物位置
        
        如果模型可用，使用深度学习预测；否则使用传统扩散方法
        """
        # 如果禁用了预测，直接返回
        if not self.enable_prediction:
            return
            
        if self.model is not None and len(self.frame_buffer) >= self.sequence_length:
            # 使用训练好的全局地图预测模型
            self._predict_with_model()
        else:
            # 使用传统扩散预测方法
            self._predict_with_diffusion()
    
    def _predict_with_model(self):
        """使用训练好的全局地图预测模型进行预测"""
        try:
            with torch.no_grad():
                # 准备输入序列
                sequence_frames = self.frame_buffer[-self.sequence_length:]
                
                # 获取最后一帧的全局累积信息
                last_frame = sequence_frames[-1]
                
                # 构建时间序列输入 (T, H, W) - 局部观测序列
                local_seq = np.stack([
                    f['occupancy'].astype(np.float32) / 255.0 
                    for f in sequence_frames
                ], axis=0)  # (T, H, W)
                
                # 全局累积地图（使用当前的占用栅格作为全局累积）
                global_acc = self.occupancy_grid.astype(np.float32) / 255.0  # (H, W)
                
                # 访问计数归一化
                global_visit = np.clip(self.visit_count.astype(np.float32) / 100.0, 0, 1)  # (H, W)
                
                # 创建已知区域掩码 (H, W) - 非127的区域为已知
                known_mask = (self.occupancy_grid != 127).astype(np.float32)
                
                # 组合输入 (T+3, H, W)
                # - T帧局部观测
                # - 1帧全局累积
                # - 1帧访问计数
                # - 1帧已知掩码
                input_tensor = np.concatenate([
                    local_seq,                          # (T, H, W)
                    global_acc[np.newaxis, :, :],       # (1, H, W)
                    global_visit[np.newaxis, :, :],     # (1, H, W)
                    known_mask[np.newaxis, :, :]        # (1, H, W)
                ], axis=0)  # (T+3, H, W)
                
                # 转换为PyTorch张量并添加批次维度 (1, C, H, W)
                input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0).to(self.device)
                
                # 模型推理
                output = self.model(input_tensor)
                
                # 处理输出 (B, H, W) -> (H, W)
                prediction = output.squeeze(0).cpu().numpy()
                
                # 转换为0-255范围（模型输出0=空闲，1=障碍物）
                # 转换为占用栅格格式：0=障碍物，255=空闲，127=未知
                prediction_occupancy = np.zeros_like(prediction, dtype=np.uint8)
                prediction_occupancy[prediction < 0.3] = 255  # 高可信度空闲
                prediction_occupancy[prediction > 0.7] = 0    # 高可信度障碍物
                prediction_occupancy[(prediction >= 0.3) & (prediction <= 0.7)] = 127  # 不确定区域
                
                # 更新预测地图
                self.obstacle_prediction = prediction_occupancy
                
                # 计算置信度（离0.5越远置信度越高）
                confidence = np.abs(prediction - 0.5) * 200  # 0-100
                self.prediction_confidence = confidence.astype(np.uint8)
                
        except Exception as e:
            print(f"模型预测失败: {e}，切换到传统方法")
            import traceback
            traceback.print_exc()
            self._predict_with_diffusion()
    
    def _predict_with_diffusion(self):
        """
        基于扩散模型预测障碍物位置
        
        核心思想：
        1. 空闲空间的边界很可能是障碍物
        2. 使用形态学操作检测边界
        3. 基于周围空闲空间密度计算置信度
        """
        # 创建二值化地图：已知空闲区域
        free_space = (self.occupancy_grid > 200).astype(np.uint8)
        known_obstacles = (self.occupancy_grid < 80).astype(np.uint8)
        
        # 方法1: 边界检测 - 空闲空间边缘扩散
        # 膨胀空闲区域，找到边界
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 膨胀空闲区域（扩散1-2个栅格）
        dilated_free = cv2.dilate(free_space, kernel, iterations=2)
        
        # 边界 = 膨胀区域 - 原始空闲区域
        boundary = cv2.subtract(dilated_free, free_space)
        
        # 排除已知的空闲区域和已知的障碍物
        unknown_mask = (self.occupancy_grid > 100) & (self.occupancy_grid < 200)
        potential_obstacles = boundary & unknown_mask.astype(np.uint8)
        
        # 方法2: 基于邻域密度的概率扩散
        # 计算每个栅格周围的空闲空间密度
        kernel_large = np.ones((5, 5), dtype=np.float32) / 25.0
        free_density = cv2.filter2D(free_space.astype(np.float32), -1, kernel_large)
        
        # 高密度空闲空间边缘 -> 高概率障碍物
        # 使用梯度检测密度变化
        gradient_x = cv2.Sobel(free_density, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(free_density, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 归一化梯度到 0-255
        if gradient_magnitude.max() > 0:
            gradient_norm = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
        else:
            gradient_norm = np.zeros_like(gradient_magnitude, dtype=np.uint8)
        
        # 方法3: 方向性扩散 - 考虑传感器视线方向
        # 从机器人位置向外扩散，未访问区域在已知空闲区域后方可能是障碍物
        robot_gx = int(self.core.robot_pos[0] / self.grid_resolution)
        robot_gy = int(self.core.robot_pos[1] / self.grid_resolution)
        
        # 创建距离地图
        y_coords, x_coords = np.ogrid[:self.grid_height, :self.grid_width]
        distance_from_robot = np.sqrt((x_coords - robot_gx)**2 + (y_coords - robot_gy)**2)
        
        # 综合预测：组合多种方法
        # 权重：边界检测(40%) + 梯度检测(40%) + 距离衰减(20%)
        prediction = np.zeros_like(self.obstacle_prediction, dtype=np.float32)
        
        # 边界贡献：边界区域标记为可能的障碍物
        prediction += potential_obstacles.astype(np.float32) * 200.0 * 0.4
        
        # 梯度贡献：梯度大的区域可能是障碍物
        prediction += gradient_norm.astype(np.float32) * 0.4
        
        # 距离衰减：离机器人远且未访问的区域，降低预测置信度
        distance_factor = np.clip(1.0 - distance_from_robot / (self.grid_width * 0.3), 0, 1)
        unvisited_mask = (self.visit_count == 0).astype(np.float32)
        prediction += gradient_norm.astype(np.float32) * distance_factor * unvisited_mask * 0.2
        
        # 裁剪到 0-255
        prediction = np.clip(prediction, 0, 255).astype(np.uint8)
        
        # 已知区域保持不变
        prediction[free_space > 0] = 0  # 已知空闲 -> 预测为无障碍
        prediction[known_obstacles > 0] = 255  # 已知障碍 -> 预测为障碍
        
        # 计算置信度：基于周围已知信息的数量
        kernel_conf = np.ones((5, 5), dtype=np.float32)
        known_mask = ((self.occupancy_grid < 80) | (self.occupancy_grid > 200)).astype(np.float32)
        confidence = cv2.filter2D(known_mask, -1, kernel_conf) / 25.0 * 100
        confidence = np.clip(confidence, 0, 100).astype(np.uint8)
        
        # 更新预测地图
        self.obstacle_prediction = prediction
        self.prediction_confidence = confidence
    
    def reset_grid(self):
        """重置栅格地图"""
        self.visit_count.fill(0)
        self.occupancy_grid.fill(127)
        self.obstacle_prediction.fill(127)
        self.prediction_confidence.fill(0)
    
    def _show_windows(self):
        """显示窗口"""
        # 世界视图
        cv2.imshow("Ring Sonar Simulation", self.world_img)
        
        # Fisher地图视图
        fmap = self.core.feature_map
        vmax = float(np.max(fmap))
        if vmax > 0:
            norm = (fmap / vmax * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(fmap, dtype=np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        
        # 标记机器人中心
        c = self.core.feature_map_size // 2
        cv2.circle(heat, (c, c), 3, (255, 255, 255), -1)
        
        # 绘制机器人朝向
        L = 10
        ex = int(c + math.cos(math.radians(self.core.robot_angle)) * L)
        ey = int(c + math.sin(math.radians(self.core.robot_angle)) * L)
        cv2.arrowedLine(heat, (c, c), (ex, ey), (255, 255, 255), 2)
        
        view = cv2.resize(heat, (600, 600), interpolation=cv2.INTER_NEAREST)
        stats = self.core.fisher_map_stats()
        cv2.putText(view, "Fisher Information Map", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, f"Features: {int(stats['total_features'])}", (10, 560),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(view, f"Avg Fisher: {stats['mean_fisher']:.2f}", (150, 560),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("Feature Map", view)
        
        # 栅格占用图视图 (SLAM-like)
        # 0=未知(灰), 127=可能无障碍(浅灰), 255=确认无障碍(白)
        # 转换为可视化：反转颜色使障碍物为黑色
        grid_vis = self.occupancy_grid.copy()
        
        # 在栅格图上标记机器人位置
        robot_gx = int(self.core.robot_pos[0] / self.grid_resolution)
        robot_gy = int(self.core.robot_pos[1] / self.grid_resolution)
        
        # 转换为彩色图以绘制机器人
        grid_color = cv2.cvtColor(grid_vis, cv2.COLOR_GRAY2BGR)
        
        # 绘制机器人和方向箭头
        if 0 <= robot_gx < self.grid_width and 0 <= robot_gy < self.grid_height:
            robot_r = max(2, int(self.core.robot_size / self.grid_resolution))
            cv2.circle(grid_color, (robot_gx, robot_gy), robot_r, (0, 0, 255), -1)
            
            # 使用统一的箭头绘制方法
            angle_rad = math.radians(self.core.robot_angle)
            self._draw_direction_arrow(grid_color, robot_gx, robot_gy, angle_rad, 
                                      self.core.velocity, 1.0)
        
        # 缩放显示
        scale_factor = max(1, 600 // max(self.grid_width, self.grid_height))
        grid_display = cv2.resize(grid_color, 
                                  (self.grid_width * scale_factor, self.grid_height * scale_factor),
                                  interpolation=cv2.INTER_NEAREST)
        
        # 计算地图统计信息
        explored_cells = np.sum(self.visit_count > 0)
        total_cells = self.grid_width * self.grid_height
        coverage = explored_cells / total_cells * 100
        
        obstacle_cells = np.sum(self.occupancy_grid < 50)
        free_cells = np.sum(self.occupancy_grid > 200)
        
        # 添加标题和统计信息
        cv2.putText(grid_display, "Global Occupancy Grid Map", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(grid_display, f"Resolution: {self.grid_resolution}m/cell", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(grid_display, f"Explored: {coverage:.1f}%", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(grid_display, f"Free: {free_cells} | Obstacle: {obstacle_cells}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imshow("Occupancy Grid", grid_display)
        
        # 障碍物预测地图可视化
        self._show_obstacle_prediction()
    
    def _show_obstacle_prediction(self):
        """显示障碍物预测地图
        
        显示说明：
        - 绿色区域：已探索的空闲区域（机器人已确认可通过）
        - 红色区域：已探索的障碍物区域（机器人已确认有障碍）
        - 热力图颜色（蓝→绿→红）：未知区域的预测
          * 蓝色 = 预测为空闲的可能性高
          * 红色 = 预测为有障碍物的可能性高
          * 绿色 = 预测不确定
        """
        # ============ 修复：正确的颜色映射 ============
        # obstacle_prediction 的值：
        #   0 = 障碍物（模型输出高值，然后被反转）
        #   127 = 未知
        #   255 = 空闲
        # 
        # 我们希望：红色=障碍物，蓝色=空闲
        # 所以需要反转值再应用热力图
        prediction_inverted = 255 - self.obstacle_prediction  # 反转：障碍物变高值(红)，空闲变低值(蓝)
        prediction_colored = cv2.applyColorMap(prediction_inverted, cv2.COLORMAP_JET)
        
        # 叠加置信度（透明度）
        # 高置信度区域更不透明
        confidence_alpha = (self.prediction_confidence / 100.0 * 0.8 + 0.2)  # 0.2-1.0
        
        # 创建已知/未知掩码
        # 已知区域 = occupancy_grid 有明确值（<80 或 >200）
        # 未知区域 = occupancy_grid 在 80-200 之间（没有确定信息）
        known_free_mask = (self.occupancy_grid > 200)      # 已知空闲区域
        known_obstacle_mask = (self.occupancy_grid < 80)   # 已知障碍区域
        unknown_mask = (~known_free_mask) & (~known_obstacle_mask)  # 未知区域
        
        # ============ 关键修改：清晰区分已知和未知区域 ============
        # 1. 已知区域显示实际的occupancy_grid信息（用户已经知道）
        #    这些区域不显示模型的预测，因为已经有真实信息
        
        # 已知空闲：绿色（明确的通过区域）
        prediction_colored[known_free_mask] = [0, 255, 0]
        
        # 已知障碍：红色（明确的障碍物）
        prediction_colored[known_obstacle_mask] = [0, 0, 255]
        
        # 2. 未知区域保持热力图颜色
        #    这样可以看到模型的预测：
        #    - 蓝色热力 = 预测为空闲
        #    - 红色热力 = 预测为障碍
        #    热力图在未知区域已经正确保留，无需额外处理
        
        # 3. 添加未知区域的边界线（更清晰地区分）
        # 创建未知区域的轮廓
        unknown_uint8 = unknown_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(unknown_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在未知区域边界绘制白色边框（强调区分）
        if len(contours) > 0:
            cv2.drawContours(prediction_colored, contours, -1, (255, 255, 255), 1)
        
        # 标记机器人位置
        robot_gx = int(self.core.robot_pos[0] / self.grid_resolution)
        robot_gy = int(self.core.robot_pos[1] / self.grid_resolution)
        
        if 0 <= robot_gx < self.grid_width and 0 <= robot_gy < self.grid_height:
            robot_r = max(2, int(self.core.robot_size / self.grid_resolution))
            cv2.circle(prediction_colored, (robot_gx, robot_gy), robot_r, (255, 255, 255), -1)
            
            # 绘制方向箭头
            angle_rad = math.radians(self.core.robot_angle)
            self._draw_direction_arrow(prediction_colored, robot_gx, robot_gy, angle_rad, 
                                      self.core.velocity, 1.0)
        
        # 缩放显示
        scale_factor = max(1, 600 // max(self.grid_width, self.grid_height))
        pred_display = cv2.resize(prediction_colored, 
                                  (self.grid_width * scale_factor, self.grid_height * scale_factor),
                                  interpolation=cv2.INTER_NEAREST)
        
        # 计算统计信息
        # 已知区域统计
        known_free_count = np.sum(known_free_mask)
        known_obstacle_count = np.sum(known_obstacle_mask)
        unknown_count = np.sum(unknown_mask)
        
        # 预测统计（仅在未知区域）
        predicted_obstacles_in_unknown = np.sum((self.obstacle_prediction > 180) & unknown_mask)
        predicted_free_in_unknown = np.sum((self.obstacle_prediction < 80) & unknown_mask)
        
        # 平均置信度（仅在有预测的区域）
        valid_predictions = self.prediction_confidence[self.prediction_confidence > 0]
        avg_confidence = np.mean(valid_predictions) if len(valid_predictions) > 0 else 0
        high_conf_predictions = np.sum(self.prediction_confidence > 70)
        
        # 添加标题和统计信息
        model_type = "Global Map Predictor" if self.model is not None and len(self.frame_buffer) >= self.sequence_length else "Diffusion Model"
        cv2.putText(pred_display, f"Obstacle Prediction ({model_type})", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示已知/未知统计
        y_offset = 50
        cv2.putText(pred_display, f"[KNOWN] Free: {known_free_count} | Obstacle: {known_obstacle_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(pred_display, f"[UNKNOWN] Total: {unknown_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # 显示帧缓冲状态
        if self.model is not None:
            buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
            cv2.putText(pred_display, buffer_status, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20
        
        # 显示预测统计（仅在未知区域）
        cv2.putText(pred_display, f"[PRED in UNKNOWN] Obstacle: {predicted_obstacles_in_unknown} | Free: {predicted_free_in_unknown}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        y_offset += 20
        cv2.putText(pred_display, f"Avg Confidence: {avg_confidence:.1f}% | High Conf: {high_conf_predictions}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # 图例
        legend_y = pred_display.shape[0] - 80
        cv2.putText(pred_display, "Legend:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        
        # 已知区域说明
        cv2.rectangle(pred_display, (10, legend_y + 15), (30, legend_y + 25), (0, 255, 0), -1)
        cv2.putText(pred_display, "= Known Free (Explored)", (35, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        cv2.rectangle(pred_display, (10, legend_y + 30), (30, legend_y + 40), (0, 0, 255), -1)
        cv2.putText(pred_display, "= Known Obstacle (Explored)", (35, legend_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # 预测颜色说明
        cv2.putText(pred_display, "Heatmap (in Unknown Area):", (10, legend_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(pred_display, "Blue = Predicted Free  |  Red = Predicted Obstacle", (10, legend_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        cv2.imshow("Obstacle Prediction", pred_display)


# ------------------------------- 主程序 -------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ring Sonar Simulator')
    parser.add_argument('--headless', action='store_true', help='无可视化模式')
    parser.add_argument('--realtime', action='store_true', help='实时速度运行')
    parser.add_argument('--steps', type=int, default=10000000, help='仿真步数')
    parser.add_argument('--world-size', type=float, default=40.0, help='世界大小(米)')
    parser.add_argument('--speed', type=float, default=1.0, help='速度倍率 (0.5=慢一倍, 2.0=快一倍)')
    parser.add_argument('--trigger-mode', type=str, default='sequential', 
                       choices=['sequential', 'interleaved', 'sector', 'all', 'greedy'], 
                       help='传感器触发模式:\n'
                            '  sequential=顺序扫描(推荐,无干扰)\n'
                            '  interleaved=交错扫描(60°间隔,低干扰)\n'
                            '  sector=扇区轮询(可能有干扰)\n'
                            '  all=全部触发(高干扰)\n'
                            '  greedy=贪心策略(基于信息增益)')
    parser.add_argument('--random-trigger', action='store_true',
                       help='随机选择触发模式（用于数据收集）')
    parser.add_argument('--demo-mode', action='store_true', 
                       help='演示模式：使用较慢的速度便于观察')
    parser.add_argument('--use-default-config', action='store_true',
                       help='使用与训练数据相同的默认配置（推荐用于推理）')
    args = parser.parse_args()
    
    # 选择配置
    if args.demo_mode:
        config = DEMO_CONFIG
        config_name = "演示配置 (DEMO_CONFIG)"
    elif args.use_default_config:
        config = DEFAULT_CONFIG
        config_name = "默认配置 (DEFAULT_CONFIG)"
    else:
        # 创建自定义配置 - 从DEFAULT_CONFIG复制基础配置，然后根据命令行参数调整
        config = SimulationConfig(
            robot=RobotPhysicsConfig(
                dt=0.05 / args.speed,  # 基于速度倍率调整
                sensor_trigger_interval=DEFAULT_CONFIG.robot.sensor_trigger_interval  # 使用配置文件的值
            ),
            world=WorldConfig(
                world_width=args.world_size,
                world_height=args.world_size
            )
        )
        config_name = "自定义配置"
    
    print("启动环形超声波雷达模拟器...")
    print_config(config, config_name)
    print(f"  - 无界面模式: {args.headless}")
    print(f"  - 实时模式: {args.realtime}")
    print(f"  - 速度倍率: {args.speed}x")
    print(f"  - 触发模式: {args.trigger_mode}" + (" (随机)" if args.random_trigger else ""))
    print(f"  - 仿真步数: {args.steps}")
    
    if args.demo_mode:
        print("\n" + "="*60)
        print("🚀 全局地图预测模型演示")
        print("="*60)
        print("该演示将展示全局地图预测模型如何利用时间序列")
        print("信息进行SLAM风格的全局地图重建和障碍物预测")
        print("按 'q' 或 ESC 退出演示")
        print("="*60)
    
    # 创建核心模拟器（使用配置和触发管理器）
    core = RingSonarCore(
        world_width=args.world_size, 
        world_height=args.world_size, 
        trigger_mode=args.trigger_mode,
        randomize_trigger=args.random_trigger,
        config=config
    )
    core.reset(regenerate_map=True)
    
    if not args.headless:
        renderer = RingSonarRenderer(core, render_mode="human")
    else:
        renderer = None
    
    init = core.state()
    print(f"机器人初始位置: [{init['position'][0]:.2f}, {init['position'][1]:.2f}] m")
    print(f"传感器数量: {core.num_sensors}, 环半径: {core.sensor_ring_radius}m")
    print(f"时间步长: {core.dt}s, 传感器触发间隔: {core.sensor_trigger_interval}步")
    
    start_real = time.time()
    expected_sim_t = 0.0    
    # 速度变化计数器
    velocity_change_counter = 0
    
    try:
        for step in range(args.steps):
            # 使用配置中的速度变化间隔和随机速度
            velocity_change_counter += 1
            if velocity_change_counter >= config.robot.velocity_change_interval:
                velocity_change_counter = 0
                linear_vel, angular_vel = config.robot.get_random_velocity()
                core.set_velocity(linear_vel, angular_vel)
            
            core.step()
            core.update_maps()
            
            if step % 50 == 0:
                st = core.state()
                f_stats = core.fisher_map_stats()
                print(f"Step {step:4d} (t={st['sim_time']:6.1f}s): "
                      f"Pos=[{st['position'][0]:6.2f},{st['position'][1]:6.2f}]m, "
                      f"Vel={st['linear_velocity']:5.2f}m/s, "
                      f"Fisher={f_stats['total_features']:4.0f}, "
                      f"Sonar={st['sonar_readings'][:4]}")  # 显示前4个传感器
            
            if renderer:
                renderer.render()
                cv2.waitKey(1)
            
            if args.realtime:
                expected_sim_t += core.dt
                now = time.time() - start_real
                sleep_t = expected_sim_t - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
            else:
                if renderer:
                    time.sleep(0.01)
            
            if renderer:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("用户退出")
                    break
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        st = core.state()
        f_stats = core.fisher_map_stats()
        print("\n最终结果:")
        print(f"  仿真时间: {st['sim_time']:.1f} 秒")
        print(f"  实际运行时间: {time.time() - start_real:.1f} 秒")
        print(f"  最终位置: [{st['position'][0]:.2f}, {st['position'][1]:.2f}] m")
        print(f"  总步数: {st['step_counter']}")
        print(f"  发现特征: {f_stats['total_features']:.0f}")
        print(f"  平均Fisher值: {f_stats['mean_fisher']:.3f}")
        print(f"  传感器读数:", st['sonar_readings'])
        if not args.headless:
            cv2.destroyAllWindows()
        print("仿真完成！")
