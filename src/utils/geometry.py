#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何工具函数

包含角度计算、坐标转换等常用函数
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


# ======================= Numba加速版本 ======================= #

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
    """
    gx = int(wx / resolution + map_size // 2 - world_width // (2 * resolution))
    gy = int(wy / resolution + map_size // 2 - world_length // (2 * resolution))
    gz = int(wz / resolution + map_size // 2 - world_height // (2 * resolution))
    
    if min(gx, gy, gz) >= 0 and max(gx, gy, gz) < map_size:
        if global_map[gx, gy, gz] < fisher_value:
            global_map[gx, gy, gz] = fisher_value
        
        neighbor_val = fisher_value * NEIGHBOR_FISHER_RATIO
        x0, x1 = max(0, gx - 1), min(map_size, gx + 2)
        y0, y1 = max(0, gy - 1), min(map_size, gy + 2)
        z0, z1 = max(0, gz - 1), min(map_size, gz + 2)
        
        patch = global_map[x0:x1, y0:y1, z0:z1]
        global_map[x0:x1, y0:y1, z0:z1] = np.maximum(patch, neighbor_val)
    
    return global_map
