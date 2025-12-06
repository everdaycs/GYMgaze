#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器定义模块
"""

import math
import numpy as np
from typing import Tuple
from dataclasses import dataclass


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
