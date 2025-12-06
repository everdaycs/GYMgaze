#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisher Information Calculation Utilities

统一的Fisher信息计算模块，消除代码重复。
"""

import math
import numpy as np
import numba as nb
from typing import Union

# ============================= 常量定义 ============================= #

# Fisher计算参数
DISTANCE_SCALE = 50.0           # 距离缩放因子 (meters)
MIN_DISTANCE_FACTOR = 0.1       # 最小距离因子
MAX_FISHER_VALUE = 10.0         # Fisher信息上限
MIN_FISHER_VALUE = 0.1          # Fisher信息下限

# 角度因子
MIN_ANGLE_FACTOR = 0.1          # 最小角度因子
PRINCIPAL_AXES = (0.0, 90.0, 180.0, 270.0)  # 主轴角度

# FOV因子
MIN_FOV_FACTOR = 0.2            # 最小FOV因子
FOV_SIGMA_DIVISOR = 4.0         # FOV衰减因子 (fov_angle / 4)

# 邻域扩散
NEIGHBOR_FISHER_RATIO = 0.4     # 8-邻域扩散比例


# ============================= 工具函数 ============================= #

def clamp(value: float, lo: float, hi: float) -> float:
    """将值限制在[lo, hi]范围内"""
    return max(lo, min(hi, value))


def angnorm_deg(angle: float) -> float:
    """归一化角度到[-180, 180)范围"""
    return (angle + 180.0) % 360.0 - 180.0


def angdiff_deg(a: float, b: float) -> float:
    """计算两个角度之间的最小差值（0-180度）"""
    return abs(angnorm_deg(a - b))


# ============================= Fisher计算 ============================= #

class FisherCalculator:
    """统一的Fisher信息计算器"""
    
    def __init__(self, 
                 distance_scale: float = DISTANCE_SCALE,
                 fov_sigma_divisor: float = FOV_SIGMA_DIVISOR):
        """
        初始化Fisher计算器
        
        Args:
            distance_scale: 距离缩放因子，用于归一化距离
            fov_sigma_divisor: FOV衰减因子，用于计算exp中的sigma
        """
        self.distance_scale = distance_scale
        self.fov_sigma_divisor = fov_sigma_divisor
    
    def compute(self, 
                distance: float, 
                angle_deg: float, 
                gaze_angle_deg: float, 
                fov_angle_deg: float) -> float:
        """
        计算Fisher信息值
        
        Args:
            distance: 到特征点的距离 (meters)
            angle_deg: 特征点的绝对角度 (degrees, 0-360)
            gaze_angle_deg: 视线方向 (degrees, 0-360)
            fov_angle_deg: 视野角度 (degrees)
        
        Returns:
            Fisher信息值，范围 [MIN_FISHER_VALUE, MAX_FISHER_VALUE]
        """
        # 1. 距离因子：近处特征更可靠
        dist_factor = 1.0 / max(distance / self.distance_scale, MIN_DISTANCE_FACTOR)
        dist_factor = min(dist_factor, MAX_FISHER_VALUE)
        
        # 2. 角度因子：与主轴对齐的特征更有价值
        min_dev = min(angdiff_deg(angle_deg, axis) for axis in PRINCIPAL_AXES)
        ang_factor = max(math.cos(math.radians(min_dev)) ** 2, MIN_ANGLE_FACTOR)
        
        # 3. FOV中心因子：视野中心的观测质量更高
        dev_from_gaze = angdiff_deg(angle_deg, gaze_angle_deg)
        sigma = fov_angle_deg / self.fov_sigma_divisor
        fov_factor = max(math.exp(-dev_from_gaze / sigma), MIN_FOV_FACTOR)
        
        # 组合并限制范围
        fisher = dist_factor * ang_factor * fov_factor
        return float(clamp(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE))
    
    def compute_batch(self,
                      distances: np.ndarray,
                      angles_deg: np.ndarray,
                      gaze_angle_deg: float,
                      fov_angle_deg: float) -> np.ndarray:
        """
        批量计算Fisher信息值（向量化）
        
        Args:
            distances: 距离数组 (meters)
            angles_deg: 角度数组 (degrees, 0-360)
            gaze_angle_deg: 视线方向 (degrees)
            fov_angle_deg: 视野角度 (degrees)
        
        Returns:
            Fisher信息数组
        """
        # 距离因子
        dist_factors = 1.0 / np.maximum(distances / self.distance_scale, MIN_DISTANCE_FACTOR)
        dist_factors = np.minimum(dist_factors, MAX_FISHER_VALUE)
        
        # 角度因子
        min_devs = np.min([angdiff_deg(angles_deg, axis) for axis in PRINCIPAL_AXES], axis=0)
        ang_factors = np.maximum(np.cos(np.radians(min_devs)) ** 2, MIN_ANGLE_FACTOR)
        
        # FOV因子
        devs_from_gaze = np.abs(angnorm_deg(angles_deg - gaze_angle_deg))
        sigma = fov_angle_deg / self.fov_sigma_divisor
        fov_factors = np.maximum(np.exp(-devs_from_gaze / sigma), MIN_FOV_FACTOR)
        
        # 组合
        fisher = dist_factors * ang_factors * fov_factors
        return np.clip(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE).astype(np.float32)


# ======================= Numba加速版本（3D环境） ======================= #

@nb.njit(fastmath=True)
def clamp_nb(value: float, lo: float, hi: float) -> float:
    """Numba版本的clamp"""
    return min(max(value, lo), hi)


@nb.njit(fastmath=True)
def angnorm_deg_nb(angle: float) -> float:
    """Numba版本的角度归一化"""
    return (angle + 180.0) % 360.0 - 180.0


@nb.njit(fastmath=True)
def angdiff_deg_nb(a: float, b: float) -> float:
    """Numba版本的角度差"""
    return abs(angnorm_deg_nb(a - b))


@nb.njit(parallel=False, nogil=True, fastmath=True)
def compute_fisher_nb(distance: float,
                      angle_rad: float,
                      gaze_angle_deg: float,
                      fov_angle_deg: float,
                      distance_scale: float = DISTANCE_SCALE,
                      fov_sigma_divisor: float = FOV_SIGMA_DIVISOR) -> float:
    """
    Numba加速的Fisher信息计算（用于3D环境的ray marching）
    
    Args:
        distance: 距离 (meters)
        angle_rad: 角度 (radians)
        gaze_angle_deg: 视线角度 (degrees)
        fov_angle_deg: FOV角度 (degrees)
        distance_scale: 距离缩放因子
        fov_sigma_divisor: FOV衰减因子
    
    Returns:
        Fisher信息值
    """
    # 距离因子
    dist_factor = min(1.0 / max(distance / distance_scale, MIN_DISTANCE_FACTOR), MAX_FISHER_VALUE)
    
    # 角度因子
    angle_deg = (np.rad2deg(angle_rad) % 360.0)
    min_dev = 1e8
    for d in (0.0, 90.0, 180.0, 270.0):
        current_dev = angdiff_deg_nb(angle_deg, d)
        if current_dev < min_dev:
            min_dev = current_dev
    ang_factor = max(np.cos(np.radians(min_dev)) ** 2, MIN_ANGLE_FACTOR)
    
    # FOV因子
    dev = angdiff_deg_nb(angle_deg, gaze_angle_deg % 360.0)
    sigma = fov_angle_deg / fov_sigma_divisor
    fov_factor = max(np.exp(-dev / sigma), MIN_FOV_FACTOR)
    
    return clamp_nb(dist_factor * ang_factor * fov_factor, MIN_FISHER_VALUE, MAX_FISHER_VALUE)


# ========================= 全局地图更新 ========================= #

def add_global_feature(global_map: np.ndarray,
                       wx: float, wy: float,
                       fisher_value: float,
                       map_size: int,
                       resolution: float,
                       world_width: float,
                       world_height: float,
                       spread_neighbors: bool = True):
    """
    将Fisher特征添加到全局地图（2D版本）
    
    Args:
        global_map: 全局特征地图
        wx, wy: 世界坐标 (meters)
        fisher_value: Fisher信息值
        map_size: 地图尺寸
        resolution: 分辨率 (meters/cell)
        world_width, world_height: 世界尺寸
        spread_neighbors: 是否扩散到邻域
    """
    # 世界坐标 -> 地图索引
    gx = int(wx / resolution + map_size // 2 - world_width // (2 * resolution))
    gy = int(wy / resolution + map_size // 2 - world_height // (2 * resolution))
    
    if 0 <= gx < map_size and 0 <= gy < map_size:
        # 更新中心点（取最大值）
        if global_map[gy, gx] < fisher_value:
            global_map[gy, gx] = fisher_value
        
        # 扩散到8邻域
        if spread_neighbors:
            neighbor_val = fisher_value * NEIGHBOR_FISHER_RATIO
            x0, x1 = max(0, gx - 1), min(map_size, gx + 2)
            y0, y1 = max(0, gy - 1), min(map_size, gy + 2)
            patch = global_map[y0:y1, x0:x1]
            global_map[y0:y1, x0:x1] = np.maximum(patch, neighbor_val)


@nb.njit(parallel=False, nogil=True)
def add_global_feature_3d_nb(global_map: np.ndarray,
                             wx: float, wy: float, wz: float,
                             fisher_value: float,
                             map_size: int,
                             resolution: float,
                             world_width: float,
                             world_length: float,
                             world_height: float):
    """
    Numba加速的3D全局特征地图更新
    
    Args:
        global_map: 3D全局特征地图
        wx, wy, wz: 世界坐标 (meters)
        fisher_value: Fisher信息值
        map_size: 地图尺寸
        resolution: 分辨率
        world_width, world_length, world_height: 世界尺寸
    
    Returns:
        更新后的地图（Numba要求）
    """
    gx = int(wx / resolution + map_size // 2 - world_width // (2 * resolution))
    gy = int(wy / resolution + map_size // 2 - world_length // (2 * resolution))
    gz = int(wz / resolution + map_size // 2 - world_height // (2 * resolution))
    
    if min(gx, gy, gz) >= 0 and max(gx, gy, gz) < map_size:
        # 更新中心
        if global_map[gx, gy, gz] < fisher_value:
            global_map[gx, gy, gz] = fisher_value
        
        # 26邻域扩散
        neighbor_val = fisher_value * NEIGHBOR_FISHER_RATIO
        x0, x1 = max(0, gx - 1), min(map_size, gx + 2)
        y0, y1 = max(0, gy - 1), min(map_size, gy + 2)
        z0, z1 = max(0, gz - 1), min(map_size, gz + 2)
        
        patch = global_map[x0:x1, y0:y1, z0:z1]
        global_map[x0:x1, y0:y1, z0:z1] = np.maximum(patch, neighbor_val)
    
    return global_map


# ========================= 使用示例 ========================= #

if __name__ == "__main__":
    # 示例1: 单点计算
    calc = FisherCalculator()
    fisher = calc.compute(
        distance=5.0,
        angle_deg=45.0,
        gaze_angle_deg=50.0,
        fov_angle_deg=90.0
    )
    print(f"Fisher value: {fisher:.4f}")
    
    # 示例2: 批量计算
    distances = np.array([1.0, 5.0, 10.0, 20.0])
    angles = np.array([0.0, 45.0, 90.0, 180.0])
    fishers = calc.compute_batch(distances, angles, gaze_angle_deg=45.0, fov_angle_deg=90.0)
    print(f"Batch Fisher values: {fishers}")
    
    # 示例3: Numba版本
    fisher_nb = compute_fisher_nb(
        distance=5.0,
        angle_rad=np.radians(45.0),
        gaze_angle_deg=50.0,
        fov_angle_deg=90.0
    )
    print(f"Numba Fisher value: {fisher_nb:.4f}")
