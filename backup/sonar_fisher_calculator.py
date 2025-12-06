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
from typing import List, Tuple

# ============================= 常量定义 ============================= #

# 距离相关
DISTANCE_SCALE = 12.5           # 最大探测距离（米）
MIN_DISTANCE_FACTOR = 0.1       # 最小距离因子
DISTANCE_DECAY_POWER = 1.5      # 距离衰减指数（超声波近距离更可靠）

# 传感器覆盖相关
MAX_COVERAGE_BONUS = 3.0        # 最大覆盖度加成（多传感器重叠）
COVERAGE_SIGMA = 45.0           # 覆盖度计算的sigma（度）

# 最终值范围
MIN_FISHER_VALUE = 0.1
MAX_FISHER_VALUE = 10.0


# ============================= 工具函数 ============================= #

def clamp(value: float, lo: float, hi: float) -> float:
    """将值限制在[lo, hi]范围内"""
    return max(lo, min(hi, value))


def angdiff_deg(a: float, b: float) -> float:
    """计算两个角度之间的最小差值（0-180度）"""
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


# ============================= Sonar Fisher计算器 ============================= #

class SonarFisherCalculator:
    """
    超声波雷达专用Fisher信息计算器
    
    考虑因素：
    1. 距离因子：近距离测量更准确
    2. 覆盖度因子：多传感器重叠区域更可靠
    3. 不考虑FOV中心质量（超声波特性）
    """
    
    def __init__(self,
                 num_sensors: int = 12,
                 sensor_spacing: float = 30.0,  # 传感器间隔（度）
                 sensor_fov: float = 65.0,
                 max_range: float = 12.5):
        """
        初始化Sonar Fisher计算器
        
        Args:
            num_sensors: 传感器数量
            sensor_spacing: 传感器角度间隔（度）
            sensor_fov: 单个传感器FOV（度）
            max_range: 最大探测距离（米）
        """
        self.num_sensors = num_sensors
        self.sensor_spacing = sensor_spacing
        self.sensor_fov = sensor_fov
        self.max_range = max_range
        
        # 预计算传感器角度
        self.sensor_angles = [i * sensor_spacing for i in range(num_sensors)]
        
        # 计算理论最大覆盖度（用于归一化）
        # 在传感器正对方向，可能有相邻传感器的FOV重叠
        self.max_overlap_count = self._estimate_max_overlap()
    
    def _estimate_max_overlap(self) -> int:
        """估计最大传感器重叠数量"""
        # 在某个方向上，计算有多少传感器的FOV覆盖它
        half_fov = self.sensor_fov / 2.0
        max_count = 0
        
        # 测试0-360度
        for test_angle in range(0, 360, 5):
            count = 0
            for sensor_angle in self.sensor_angles:
                if angdiff_deg(test_angle, sensor_angle) <= half_fov:
                    count += 1
            max_count = max(max_count, count)
        
        return max(1, max_count)
    
    def compute(self,
                distance: float,
                angle_deg: float) -> float:
        """
        计算超声波观测的Fisher信息值
        
        Args:
            distance: 观测距离（米）
            angle_deg: 观测角度（相对机器人，0-360度）
        
        Returns:
            Fisher信息值 [MIN_FISHER_VALUE, MAX_FISHER_VALUE]
        """
        # 1. 距离因子：近距离更可靠
        # 使用非线性衰减：近距离优势更明显
        normalized_dist = distance / self.max_range
        dist_factor = (1.0 - normalized_dist) ** DISTANCE_DECAY_POWER
        dist_factor = max(dist_factor, MIN_DISTANCE_FACTOR)
        
        # 2. 覆盖度因子：计算有多少传感器的FOV覆盖这个角度
        coverage_count = self._compute_coverage(angle_deg)
        
        # 归一化到[1.0, MAX_COVERAGE_BONUS]
        # coverage_count=1 -> factor=1.0
        # coverage_count=max -> factor=MAX_COVERAGE_BONUS
        coverage_factor = 1.0 + (MAX_COVERAGE_BONUS - 1.0) * (coverage_count - 1) / max(1, self.max_overlap_count - 1)
        
        # 3. 综合
        fisher = dist_factor * coverage_factor
        
        return clamp(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE)
    
    def _compute_coverage(self, angle_deg: float) -> int:
        """
        计算有多少传感器的FOV覆盖指定角度
        
        Args:
            angle_deg: 目标角度（度）
        
        Returns:
            覆盖的传感器数量
        """
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
                                     active_sensor_ids: set) -> float:
        """
        仅考虑激活传感器的Fisher计算（用于sequential/interleaved模式）
        
        Args:
            distance: 观测距离（米）
            angle_deg: 观测角度（度）
            active_sensor_ids: 本帧激活的传感器ID集合
        
        Returns:
            Fisher信息值
        """
        # 距离因子
        normalized_dist = distance / self.max_range
        dist_factor = (1.0 - normalized_dist) ** DISTANCE_DECAY_POWER
        dist_factor = max(dist_factor, MIN_DISTANCE_FACTOR)
        
        # 只计算激活传感器的覆盖度
        half_fov = self.sensor_fov / 2.0
        coverage_count = 0
        
        for sensor_id in active_sensor_ids:
            sensor_angle = sensor_id * self.sensor_spacing
            if angdiff_deg(angle_deg, sensor_angle) <= half_fov:
                coverage_count += 1
        
        # 如果没有激活传感器覆盖，返回最小值
        if coverage_count == 0:
            return MIN_FISHER_VALUE
        
        # 根据激活传感器数量调整覆盖因子
        # 注意：这里不应该按照全部传感器的max_overlap来归一化
        # 而是按照实际可能的覆盖数量
        coverage_factor = 1.0 + (coverage_count - 1) * 0.5  # 每多一个传感器增加0.5
        
        fisher = dist_factor * coverage_factor
        
        return clamp(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE)
    
    def get_coverage_map(self, resolution: int = 360) -> np.ndarray:
        """
        生成覆盖度地图用于可视化
        
        Args:
            resolution: 角度分辨率（采样点数）
        
        Returns:
            覆盖度数组
        """
        angles = np.linspace(0, 360, resolution, endpoint=False)
        coverage = np.array([self._compute_coverage(angle) for angle in angles])
        return coverage
    
    def visualize_coverage(self):
        """可视化传感器覆盖度"""
        import matplotlib.pyplot as plt
        
        angles = np.linspace(0, 360, 360)
        coverage = self.get_coverage_map(360)
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
        ax.plot(np.radians(angles), coverage, 'b-', linewidth=2)
        ax.fill(np.radians(angles), coverage, alpha=0.3)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'Sensor Coverage Map ({self.num_sensors} sensors, {self.sensor_fov}° FOV)', 
                     pad=20, fontsize=14)
        ax.set_xlabel('Coverage Count', fontsize=12)
        
        # 标记传感器位置
        for i, sensor_angle in enumerate(self.sensor_angles):
            ax.plot([np.radians(sensor_angle)], [self.max_overlap_count + 0.5], 
                   'ro', markersize=10)
            ax.text(np.radians(sensor_angle), self.max_overlap_count + 1.0, 
                   f'S{i}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('sonar_coverage_map.png', dpi=150)
        print("Coverage map saved to 'sonar_coverage_map.png'")
        plt.show()


# ============================= 示例与测试 ============================= #

if __name__ == "__main__":
    print("=" * 70)
    print("Sonar Fisher Information Calculator - 示例")
    print("=" * 70)
    print()
    
    # 创建计算器（12传感器，30°间隔，65° FOV）
    calc = SonarFisherCalculator(
        num_sensors=12,
        sensor_spacing=30.0,
        sensor_fov=65.0,
        max_range=12.5
    )
    
    print(f"配置：")
    print(f"  传感器数量: {calc.num_sensors}")
    print(f"  传感器间隔: {calc.sensor_spacing}°")
    print(f"  传感器FOV: {calc.sensor_fov}°")
    print(f"  最大探测距离: {calc.max_range}m")
    print(f"  最大覆盖度: {calc.max_overlap_count} 个传感器")
    print()
    
    # 测试不同位置的Fisher值
    print("Fisher值计算示例：")
    print("-" * 70)
    
    test_cases = [
        # (distance, angle, description)
        (2.0, 0.0, "近距离, 正前方 (Sensor 0中心)"),
        (2.0, 15.0, "近距离, Sensor 0和1之间"),
        (2.0, 30.0, "近距离, Sensor 1中心"),
        (5.0, 0.0, "中距离, 正前方"),
        (10.0, 0.0, "远距离, 正前方"),
        (2.0, 45.0, "近距离, Sensor 1和2之间"),
        (5.0, 90.0, "中距离, 正右侧 (Sensor 3中心)"),
    ]
    
    for distance, angle, desc in test_cases:
        fisher = calc.compute(distance, angle)
        coverage = calc._compute_coverage(angle)
        print(f"  {desc}")
        print(f"    距离={distance}m, 角度={angle}° → Fisher={fisher:.3f}, 覆盖度={coverage}")
        print()
    
    # 覆盖度分析
    print("覆盖度分析：")
    print("-" * 70)
    for angle in [0, 15, 30, 45, 60, 90, 180]:
        coverage = calc._compute_coverage(angle)
        print(f"  角度 {angle:3d}° : {coverage} 个传感器覆盖")
    print()
    
    # 可视化
    print("生成覆盖度地图...")
    calc.visualize_coverage()
