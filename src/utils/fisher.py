#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sonar-specific Fisher Information Calculator
超声波雷达专用Fisher信息计算器

与视觉系统的区别：
1. 超声波只返回距离信息（不含角度精度信息）
2. 传感器重叠区域可信度更高（多传感器验证）
3. FOV中心/边缘质量差异不明显（超声波回波特性）
4. 距离精度：近距离更可靠
"""

import numpy as np
import math
from typing import List, Tuple, Set

# ============================= 常量定义 ============================= #

DISTANCE_SCALE = 12.5           # 最大探测距离（米）
MIN_DISTANCE_FACTOR = 0.1       # 最小距离因子
DISTANCE_DECAY_POWER = 1.5      # 距离衰减指数

MAX_COVERAGE_BONUS = 3.0        # 最大覆盖度加成
COVERAGE_SIGMA = 45.0           # 覆盖度计算的sigma

MIN_FISHER_VALUE = 0.1
MAX_FISHER_VALUE = 10.0


def clamp(value: float, lo: float, hi: float) -> float:
    """将值限制在[lo, hi]范围内"""
    return max(lo, min(hi, value))


def angdiff_deg(a: float, b: float) -> float:
    """计算两个角度之间的最小差值（0-180度）"""
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


class SonarFisherCalculator:
    """
    超声波雷达专用Fisher信息计算器
    
    考虑因素：
    1. 距离因子：近距离测量更准确
    2. 覆盖度因子：多传感器重叠区域更可靠
    """
    
    def __init__(self,
                 num_sensors: int = 12,
                 sensor_spacing: float = 30.0,
                 sensor_fov: float = 65.0,
                 max_range: float = 12.5):
        self.num_sensors = num_sensors
        self.sensor_spacing = sensor_spacing
        self.sensor_fov = sensor_fov
        self.max_range = max_range
        
        self.sensor_angles = [i * sensor_spacing for i in range(num_sensors)]
        self.max_overlap_count = self._estimate_max_overlap()
    
    def _estimate_max_overlap(self) -> int:
        """估计最大传感器重叠数量"""
        half_fov = self.sensor_fov / 2.0
        max_count = 0
        
        for test_angle in range(0, 360, 5):
            count = 0
            for sensor_angle in self.sensor_angles:
                if angdiff_deg(test_angle, sensor_angle) <= half_fov:
                    count += 1
            max_count = max(max_count, count)
        
        return max(1, max_count)
    
    def compute(self, distance: float, angle_deg: float) -> float:
        """
        计算超声波观测的Fisher信息值
        
        Args:
            distance: 观测距离（米）
            angle_deg: 观测角度（相对机器人，0-360度）
        
        Returns:
            Fisher信息值 [MIN_FISHER_VALUE, MAX_FISHER_VALUE]
        """
        normalized_dist = distance / self.max_range
        dist_factor = (1.0 - normalized_dist) ** DISTANCE_DECAY_POWER
        dist_factor = max(dist_factor, MIN_DISTANCE_FACTOR)
        
        coverage_count = self._compute_coverage(angle_deg)
        coverage_factor = 1.0 + (MAX_COVERAGE_BONUS - 1.0) * (coverage_count - 1) / max(1, self.max_overlap_count - 1)
        
        fisher = dist_factor * coverage_factor
        return clamp(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE)
    
    def _compute_coverage(self, angle_deg: float) -> int:
        """计算有多少传感器的FOV覆盖指定角度"""
        half_fov = self.sensor_fov / 2.0
        count = 0
        
        for sensor_angle in self.sensor_angles:
            angle_diff = angdiff_deg(angle_deg, sensor_angle)
            if angle_diff <= half_fov:
                count += 1
        
        return count
    
    def compute_with_active_sensors(self,
                                     distance: float,
                                     angle_deg: float,
                                     active_sensor_ids: Set[int]) -> float:
        """
        仅考虑激活传感器的Fisher计算
        """
        normalized_dist = distance / self.max_range
        dist_factor = (1.0 - normalized_dist) ** DISTANCE_DECAY_POWER
        dist_factor = max(dist_factor, MIN_DISTANCE_FACTOR)
        
        half_fov = self.sensor_fov / 2.0
        coverage_count = 0
        
        for sensor_id in active_sensor_ids:
            sensor_angle = sensor_id * self.sensor_spacing
            if angdiff_deg(angle_deg, sensor_angle) <= half_fov:
                coverage_count += 1
        
        if coverage_count == 0:
            return MIN_FISHER_VALUE
        
        coverage_factor = 1.0 + (coverage_count - 1) * 0.5
        fisher = dist_factor * coverage_factor
        
        return clamp(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE)
    
    def get_coverage_map(self, resolution: int = 360) -> np.ndarray:
        """生成覆盖度地图用于可视化"""
        angles = np.linspace(0, 360, resolution, endpoint=False)
        coverage = np.array([self._compute_coverage(angle) for angle in angles])
        return coverage
