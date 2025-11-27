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
import random
import math
import cv2
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from fisher_utils import (
    clamp, angnorm_deg, angdiff_deg,
    add_global_feature
)
from sonar_fisher_calculator import SonarFisherCalculator


# ------------------------------- 传感器定义 -------------------------------- #

@dataclass
class SonarSensor:
    """单个超声波传感器的定义"""
    id: int                      # 传感器编号 (0-11)
    angle: float                 # 传感器相对机器人中心的角度 (degrees)
    offset_x: float              # 相对机器人中心的x偏移 (meters)
    offset_y: float              # 相对机器人中心的y偏移 (meters)
    fov_angle: float             # 视野角度 (degrees)
    max_range: float             # 最大探测距离 (meters)
    
    def get_world_position(self, robot_pos: np.ndarray, robot_angle: float) -> Tuple[float, float]:
        """获取传感器在世界坐标系中的位置"""
        # 考虑机器人旋转
        angle_rad = math.radians(robot_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 旋转偏移量
        wx = robot_pos[0] + cos_a * self.offset_x - sin_a * self.offset_y
        wy = robot_pos[1] + sin_a * self.offset_x + cos_a * self.offset_y
        
        return (wx, wy)
    
    def get_world_angle(self, robot_angle: float) -> float:
        """获取传感器在世界坐标系中的朝向"""
        return (robot_angle + self.angle) % 360.0


# ------------------------------- 核心模拟器 -------------------------------- #

class RingSonarCore:
    """
    环形超声波雷达核心模拟器
    - 12个超声波传感器均匀分布在15cm半径圆盘上
    - 每个传感器独立扫描65° FoV
    - 2D世界，俯视图
    """
    
    def __init__(self,
                 world_width: float = 40.0,
                 world_height: float = 40.0,
                 pixel_per_meter: int = 20,
                 robot_size: float = 0.5,
                 sensor_ring_radius: float = 0.15,  # 15cm
                 num_sensors: int = 12,
                 sensor_fov: float = 65.0,
                 sensor_max_range: float = 12.5,
                 feature_map_size: int = 100,
                 feature_map_resolution: float = 0.25,
                 control_frequency: float = 5.0,
                 trigger_mode: str = "sector",  # 触发模式
                 dt: float = 0.1):  # 新增：仿真时间步长（秒）
        
        # 世界参数
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.pixel_per_meter = int(pixel_per_meter)
        self.robot_size = float(robot_size)
        
        # 传感器参数
        self.sensor_ring_radius = float(sensor_ring_radius)
        self.num_sensors = int(num_sensors)
        self.sensor_fov = float(sensor_fov)
        self.sensor_max_range = float(sensor_max_range)
        
        # 触发模式配置
        self.trigger_mode = trigger_mode
        self._init_trigger_config()
        
        # 初始化传感器阵列
        self.sensors: List[SonarSensor] = []
        self._init_sensors()
        
        # Fisher地图 (2D，机器人中心的局部视图)
        self.feature_map_size = int(feature_map_size)
        self.feature_map_resolution = float(feature_map_resolution)
        self.feature_map = np.zeros((self.feature_map_size, self.feature_map_size), dtype=np.float32)
        self.global_feature_map_size = int(max(self.world_width, self.world_height) * 2 / self.feature_map_resolution)
        self.global_feature_map = np.zeros((self.global_feature_map_size, self.global_feature_map_size), dtype=np.float32)
        
        # 时间与控制
        self.dt = float(dt)  # 仿真时间步长（可配置）
        self.sim_time = 0.0
        
        # 机器人状态
        self.robot_pos = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float64)
        self.robot_angle = 0.0           # deg, 机器人朝向
        self._robot_angle_rad_cache = 0.0  # 缓存弧度值，避免重复转换
        self.velocity = 0.0              # m/s
        self.angular_velocity = 0.0      # rad/s
        self.max_linear_velocity = 3.0
        self.max_angular_velocity = 1.0
        
        # 传感器读数 (每个传感器的距离测量)
        self.sonar_readings = np.full(self.num_sensors, self.sensor_max_range, dtype=np.float32)
        
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
    
    # -------- 触发模式配置 -------- #
    
    def _init_trigger_config(self):
        """初始化触发配置"""
        if self.trigger_mode == "sector":
            # 扇区轮询模式：分4个扇区，每个扇区3个传感器
            # 传感器ID: 0(0°), 1(30°), 2(60°), 3(90°), ..., 11(330°)
            self.sectors = {
                "front": [11, 0, 1],      # 330°, 0°, 30° (机器人前方±30°)
                "right": [2, 3, 4],       # 60°, 90°, 120° (右侧)
                "back": [5, 6, 7],        # 150°, 180°, 210° (后方)
                "left": [8, 9, 10]        # 240°, 270°, 300° (左侧)
            }
            self.sector_sequence = ["front", "right", "back", "left"]
            self.current_sector_index = 0
            
            print(f"触发模式: {self.trigger_mode}")
            print(f"扇区配置: {self.sectors}")
            print(f"轮询顺序: {self.sector_sequence}")
            
        elif self.trigger_mode == "sequential":
            # 顺序扫描模式：每次只触发1个传感器，完全避免干扰
            self.current_sensor_index = 0
            print(f"触发模式: {self.trigger_mode} (顺序扫描，完全避免干扰)")
            print(f"扫描顺序: 0 → 1 → 2 → ... → 11 → 0")
            
        elif self.trigger_mode == "interleaved":
            # 交错扫描模式：传感器间隔足够大，避免干扰
            # 奇数轮次：0, 2, 4, 6, 8, 10 (间隔60°)
            # 偶数轮次：1, 3, 5, 7, 9, 11 (间隔60°)
            self.interleaved_groups = [
                [0, 2, 4, 6, 8, 10],  # 偶数ID，间隔60°
                [1, 3, 5, 7, 9, 11]   # 奇数ID，间隔60°
            ]
            self.current_group_index = 0
            self.current_sensor_in_group = 0
            print(f"触发模式: {self.trigger_mode} (交错扫描)")
            print(f"组1: {self.interleaved_groups[0]} (间隔60°)")
            print(f"组2: {self.interleaved_groups[1]} (间隔60°)")
        else:
            # all 或其他模式
            print(f"触发模式: {self.trigger_mode}")
    
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
    
    def reset(self, regenerate_map: bool = True, seed: Optional[int] = None) -> None:
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if regenerate_map or not self._have_map:
            self._gen_obstacles(num_obstacles=20)
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
        
        # 扫描所有传感器
        self._scan_all_sensors()
        
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
    
    def _gen_obstacles(self, num_obstacles: int = 20):
        """生成障碍物"""
        self.obstacles.clear()
        wall = 0.5
        
        # 边界墙
        self.obstacles += [
            ('rect', (0.0, 0.0, self.world_width, wall)),
            ('rect', (0.0, self.world_height - wall, self.world_width, wall)),
            ('rect', (0.0, 0.0, wall, self.world_height)),
            ('rect', (self.world_width - wall, 0.0, wall, self.world_height))
        ]
        
        # 随机障碍物
        for _ in range(num_obstacles):
            x = random.uniform(2.5, self.world_width - 2.5)
            y = random.uniform(2.5, self.world_height - 2.5)
            w = random.uniform(2.0, 5.0)
            h = random.uniform(2.0, 5.0)
            self.obstacles.append(('rect', (x, y, w, h)))
    
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
        if self.trigger_mode == "sector":
            current_sector_name = self.sector_sequence[self.current_sector_index]
            return self.sectors[current_sector_name]
        elif self.trigger_mode == "sequential":
            return [self.current_sensor_index]
        elif self.trigger_mode == "interleaved":
            current_group = self.interleaved_groups[self.current_group_index]
            return [current_group[self.current_sensor_in_group]]
        else:  # "all"
            return list(range(self.num_sensors))
    
    def _scan_all_sensors(self):
        """扫描传感器（根据触发模式）"""
        # 清空本帧活跃传感器记录
        self.active_sensors_this_frame.clear()
        
        # 获取需要激活的传感器ID
        active_ids = self._get_active_sensor_ids()
        
        # 扫描激活的传感器
        for sensor_id in active_ids:
            sensor = self.sensors[sensor_id]
            distance = self._scan_single_sensor(sensor)
            self.sonar_readings[sensor_id] = distance
            self.active_sensors_this_frame.add(sensor_id)
        
        # 更新触发模式的索引
        if self.trigger_mode == "sector":
            self.current_sector_index = (self.current_sector_index + 1) % len(self.sector_sequence)
        elif self.trigger_mode == "sequential":
            self.current_sensor_index = (self.current_sensor_index + 1) % self.num_sensors
        elif self.trigger_mode == "interleaved":
            self.current_sensor_in_group += 1
            if self.current_sensor_in_group >= len(self.interleaved_groups[self.current_group_index]):
                self.current_sensor_in_group = 0
                self.current_group_index = (self.current_group_index + 1) % len(self.interleaved_groups)
    
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
        """从传感器读数中检测并添加特征到全局地图"""
        for sensor in self.sensors:
            distance = self.sonar_readings[sensor.id]
            
            # 如果检测到障碍物（距离小于最大范围）
            if distance < sensor.max_range:
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


# ------------------------------- 渲染器 -------------------------------- #

class RingSonarRenderer:
    """
    环形超声波雷达渲染器 - 2D俯视图
    显示：机器人、传感器布局、FoV扇区、障碍物、Fisher地图、栅格占用图
    """
    
    def __init__(self, core: RingSonarCore, render_mode: Optional[str] = "human"):
        self.core = core
        self.render_mode = render_mode
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
    
    def render(self):
        if self.render_mode is None:
            return None
        
        self.world_img.fill(255)
        self._draw_obstacles()
        self._draw_sensor_fov()
        self._draw_robot()
        self._draw_sensor_readings()
        self._draw_trigger_info()  # 显示触发模式信息
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
        """在图像上绘制传感器读数文本"""
        ppm = self.core.pixel_per_meter
        
        for i, sensor in enumerate(self.core.sensors):
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
        
        # 更新障碍物预测
        self._predict_obstacles()
    
    def _predict_obstacles(self):
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
        """显示障碍物预测地图"""
        # 创建热力图：0=无障碍(蓝), 127=未知(绿), 255=障碍(红)
        prediction_colored = cv2.applyColorMap(self.obstacle_prediction, cv2.COLORMAP_JET)
        
        # 叠加置信度（透明度）
        # 高置信度区域更不透明
        confidence_alpha = (self.prediction_confidence / 100.0 * 0.8 + 0.2)  # 0.2-1.0
        
        # 在预测图上标记已知信息
        # 已知空闲：绿色边框
        free_mask = (self.occupancy_grid > 200)
        prediction_colored[free_mask] = [0, 255, 0]  # 绿色
        
        # 已知障碍：红色边框
        obstacle_mask = (self.occupancy_grid < 80)
        prediction_colored[obstacle_mask] = [0, 0, 255]  # 红色
        
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
        predicted_obstacles = np.sum((self.obstacle_prediction > 180) & 
                                     (self.occupancy_grid > 100) & 
                                     (self.occupancy_grid < 200))
        avg_confidence = np.mean(self.prediction_confidence[self.prediction_confidence > 0])
        high_conf_predictions = np.sum(self.prediction_confidence > 70)
        
        # 添加标题和统计信息
        cv2.putText(pred_display, "Obstacle Prediction (Diffusion Model)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pred_display, f"Predicted Obstacles: {predicted_obstacles}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(pred_display, f"Avg Confidence: {avg_confidence:.1f}%", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(pred_display, f"High Conf Cells: {high_conf_predictions}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 图例
        legend_y = pred_display.shape[0] - 60
        cv2.putText(pred_display, "Legend:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(pred_display, (10, legend_y + 5), (30, legend_y + 15), (0, 255, 0), -1)
        cv2.putText(pred_display, "= Known Free", (35, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(pred_display, (10, legend_y + 20), (30, legend_y + 30), (0, 0, 255), -1)
        cv2.putText(pred_display, "= Known Obstacle", (35, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(pred_display, "Blue->Red = Predicted Probability", (10, legend_y + 45),
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
                       choices=['sequential', 'interleaved', 'sector', 'all'], 
                       help='传感器触发模式:\n'
                            '  sequential=顺序扫描(推荐,无干扰)\n'
                            '  interleaved=交错扫描(60°间隔,低干扰)\n'
                            '  sector=扇区轮询(可能有干扰)\n'
                            '  all=全部触发(高干扰)')
    args = parser.parse_args()
    
    print("启动环形超声波雷达模拟器...")
    print(f"  - 无界面模式: {args.headless}")
    print(f"  - 实时模式: {args.realtime}")
    print(f"  - 速度倍率: {args.speed}x")
    print(f"  - 触发模式: {args.trigger_mode}")
    print(f"  - 仿真步数: {args.steps}")
    print(f"  - 世界大小: {args.world_size}m x {args.world_size}m")
    
    # 根据速度倍率调整时间步长
    dt = 0.1 / args.speed
    core = RingSonarCore(world_width=args.world_size, world_height=args.world_size, 
                         dt=dt, trigger_mode=args.trigger_mode)
    core.reset(regenerate_map=True)
    
    if not args.headless:
        renderer = RingSonarRenderer(core, render_mode="human")
    else:
        renderer = None
    
    init = core.state()
    print(f"机器人初始位置: [{init['position'][0]:.2f}, {init['position'][1]:.2f}] m")
    print(f"传感器数量: {core.num_sensors}, 环半径: {core.sensor_ring_radius}m")
    
    start_real = time.time()
    expected_sim_t = 0.0
    
    try:
        for step in range(args.steps):
            # 简单的随机移动策略
            if step % 50 == 0:
                core.set_velocity(
                    float(np.random.uniform(-2.0, 3.0)),
                    float(np.random.uniform(-0.8, 0.8))
                )
            
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
