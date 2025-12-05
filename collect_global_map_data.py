#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局地图预测数据收集脚本 (增强版)

目标：训练一个类似SLAM的全局地图重建模型
- 输入：时间窗口内的局部观测 + 全局累积信息
- 输出：完整的全局地图预测

增强特性：
1. 多样化障碍物形状：矩形、圆形、L形、T形、多边形等
2. 多样化地图边界：开放、封闭、部分墙壁、迷宫入口等
3. 障碍物重叠和聚合
4. 不同密度场景
"""

import numpy as np
import cv2
import os
import pickle
import argparse
import math
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from ring_sonar_simulator import RingSonarCore, RingSonarRenderer


class DiverseMapGenerator:
    """生成多样化的地图配置"""
    
    def __init__(self, world_width: float = 40.0, world_height: float = 40.0):
        self.world_width = world_width
        self.world_height = world_height
        
    def generate_obstacles(self, seed: int) -> List[Tuple[str, Tuple]]:
        """
        生成多样化的障碍物配置
        
        返回: List of (type, data)
            - ('rect', (x, y, w, h))
            - ('circle', (cx, cy, r))  # 会被转换为近似矩形
            - ('polygon', vertices)     # 会被转换为多个矩形
        """
        np.random.seed(seed)
        
        # 随机选择场景类型
        scene_type = np.random.choice([
            'sparse',           # 稀疏场景
            'dense',            # 密集场景
            'clustered',        # 聚集场景（障碍物成簇）
            'corridor',         # 走廊场景
            'rooms',            # 房间场景
            'mixed',            # 混合场景
            'maze_like',        # 迷宫状
            'open_center',      # 中心开阔
        ], p=[0.12, 0.12, 0.15, 0.12, 0.15, 0.14, 0.10, 0.10])
        
        obstacles = []
        
        # 生成边界墙（多样化）
        boundary_obstacles = self._generate_diverse_boundary(seed)
        obstacles.extend(boundary_obstacles)
        
        # 根据场景类型生成障碍物
        if scene_type == 'sparse':
            obstacles.extend(self._generate_sparse_obstacles(seed))
        elif scene_type == 'dense':
            obstacles.extend(self._generate_dense_obstacles(seed))
        elif scene_type == 'clustered':
            obstacles.extend(self._generate_clustered_obstacles(seed))
        elif scene_type == 'corridor':
            obstacles.extend(self._generate_corridor_obstacles(seed))
        elif scene_type == 'rooms':
            obstacles.extend(self._generate_room_obstacles(seed))
        elif scene_type == 'mixed':
            obstacles.extend(self._generate_mixed_obstacles(seed))
        elif scene_type == 'maze_like':
            obstacles.extend(self._generate_maze_obstacles(seed))
        elif scene_type == 'open_center':
            obstacles.extend(self._generate_open_center_obstacles(seed))
        
        return obstacles
    
    def _generate_diverse_boundary(self, seed: int) -> List[Tuple[str, Tuple]]:
        """生成多样化的地图边界"""
        np.random.seed(seed + 1000)
        
        boundary_type = np.random.choice([
            'full_walls',       # 完整四面墙
            'open_corners',     # 开放角落
            'gaps',             # 墙壁有缺口
            'thick_walls',      # 厚墙壁
            'irregular',        # 不规则边界
            'partial',          # 部分墙壁
        ], p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        obstacles = []
        wall_base = 0.5
        
        if boundary_type == 'full_walls':
            # 标准四面墙
            wall = wall_base
            obstacles = [
                ('rect', (0.0, 0.0, self.world_width, wall)),
                ('rect', (0.0, self.world_height - wall, self.world_width, wall)),
                ('rect', (0.0, 0.0, wall, self.world_height)),
                ('rect', (self.world_width - wall, 0.0, wall, self.world_height))
            ]
            
        elif boundary_type == 'open_corners':
            # 四面墙但角落开放
            wall = wall_base
            gap = 3.0
            # 下墙（中间部分）
            obstacles.append(('rect', (gap, 0.0, self.world_width - 2*gap, wall)))
            # 上墙（中间部分）
            obstacles.append(('rect', (gap, self.world_height - wall, self.world_width - 2*gap, wall)))
            # 左墙（中间部分）
            obstacles.append(('rect', (0.0, gap, wall, self.world_height - 2*gap)))
            # 右墙（中间部分）
            obstacles.append(('rect', (self.world_width - wall, gap, wall, self.world_height - 2*gap)))
            
        elif boundary_type == 'gaps':
            # 墙壁有随机缺口
            wall = wall_base
            num_gaps = np.random.randint(2, 5)
            
            for side in ['bottom', 'top', 'left', 'right']:
                if side == 'bottom':
                    self._add_wall_with_gaps(obstacles, 0, 0, self.world_width, wall, 
                                            horizontal=True, num_gaps=num_gaps)
                elif side == 'top':
                    self._add_wall_with_gaps(obstacles, 0, self.world_height - wall, 
                                            self.world_width, wall, horizontal=True, num_gaps=num_gaps)
                elif side == 'left':
                    self._add_wall_with_gaps(obstacles, 0, 0, wall, self.world_height, 
                                            horizontal=False, num_gaps=num_gaps)
                elif side == 'right':
                    self._add_wall_with_gaps(obstacles, self.world_width - wall, 0, 
                                            wall, self.world_height, horizontal=False, num_gaps=num_gaps)
                    
        elif boundary_type == 'thick_walls':
            # 厚墙壁（不规则厚度）
            for side in ['bottom', 'top', 'left', 'right']:
                thickness = np.random.uniform(0.5, 2.0)
                if side == 'bottom':
                    obstacles.append(('rect', (0, 0, self.world_width, thickness)))
                elif side == 'top':
                    obstacles.append(('rect', (0, self.world_height - thickness, self.world_width, thickness)))
                elif side == 'left':
                    obstacles.append(('rect', (0, 0, thickness, self.world_height)))
                elif side == 'right':
                    obstacles.append(('rect', (self.world_width - thickness, 0, thickness, self.world_height)))
                    
        elif boundary_type == 'irregular':
            # 不规则边界（锯齿状）
            wall = wall_base
            # 基础墙
            obstacles.append(('rect', (0.0, 0.0, self.world_width, wall)))
            obstacles.append(('rect', (0.0, self.world_height - wall, self.world_width, wall)))
            obstacles.append(('rect', (0.0, 0.0, wall, self.world_height)))
            obstacles.append(('rect', (self.world_width - wall, 0.0, wall, self.world_height)))
            
            # 添加随机凸起
            for _ in range(np.random.randint(4, 10)):
                side = np.random.choice(['bottom', 'top', 'left', 'right'])
                bump_size = np.random.uniform(1.0, 3.0)
                bump_length = np.random.uniform(2.0, 5.0)
                
                if side == 'bottom':
                    pos = np.random.uniform(2, self.world_width - bump_length - 2)
                    obstacles.append(('rect', (pos, wall, bump_length, bump_size)))
                elif side == 'top':
                    pos = np.random.uniform(2, self.world_width - bump_length - 2)
                    obstacles.append(('rect', (pos, self.world_height - wall - bump_size, bump_length, bump_size)))
                elif side == 'left':
                    pos = np.random.uniform(2, self.world_height - bump_length - 2)
                    obstacles.append(('rect', (wall, pos, bump_size, bump_length)))
                elif side == 'right':
                    pos = np.random.uniform(2, self.world_height - bump_length - 2)
                    obstacles.append(('rect', (self.world_width - wall - bump_size, pos, bump_size, bump_length)))
                    
        elif boundary_type == 'partial':
            # 只有部分墙壁
            wall = wall_base
            walls_to_add = np.random.choice(['bottom', 'top', 'left', 'right'], 
                                           size=np.random.randint(2, 4), replace=False)
            for side in walls_to_add:
                if side == 'bottom':
                    obstacles.append(('rect', (0.0, 0.0, self.world_width, wall)))
                elif side == 'top':
                    obstacles.append(('rect', (0.0, self.world_height - wall, self.world_width, wall)))
                elif side == 'left':
                    obstacles.append(('rect', (0.0, 0.0, wall, self.world_height)))
                elif side == 'right':
                    obstacles.append(('rect', (self.world_width - wall, 0.0, wall, self.world_height)))
        
        return obstacles
    
    def _add_wall_with_gaps(self, obstacles, x, y, w, h, horizontal, num_gaps):
        """添加有缺口的墙壁"""
        if horizontal:
            total_length = w
            gap_positions = sorted(np.random.uniform(2, total_length - 2, num_gaps))
            gap_sizes = np.random.uniform(2.0, 4.0, num_gaps)
            
            current_pos = x
            for gap_pos, gap_size in zip(gap_positions, gap_sizes):
                if gap_pos > current_pos + 1:
                    obstacles.append(('rect', (current_pos, y, gap_pos - current_pos, h)))
                current_pos = gap_pos + gap_size
            
            if current_pos < x + total_length - 1:
                obstacles.append(('rect', (current_pos, y, x + total_length - current_pos, h)))
        else:
            total_length = h
            gap_positions = sorted(np.random.uniform(2, total_length - 2, num_gaps))
            gap_sizes = np.random.uniform(2.0, 4.0, num_gaps)
            
            current_pos = y
            for gap_pos, gap_size in zip(gap_positions, gap_sizes):
                if gap_pos > current_pos + 1:
                    obstacles.append(('rect', (x, current_pos, w, gap_pos - current_pos)))
                current_pos = gap_pos + gap_size
            
            if current_pos < y + total_length - 1:
                obstacles.append(('rect', (x, current_pos, w, y + total_length - current_pos)))
    
    def _generate_sparse_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成稀疏障碍物"""
        np.random.seed(seed + 2000)
        obstacles = []
        num = np.random.randint(5, 12)
        
        for _ in range(num):
            obstacles.extend(self._generate_random_shape())
        
        return obstacles
    
    def _generate_dense_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成密集障碍物"""
        np.random.seed(seed + 3000)
        obstacles = []
        num = np.random.randint(25, 40)
        
        for _ in range(num):
            obstacles.extend(self._generate_random_shape(size_range=(1.0, 3.0)))
        
        return obstacles
    
    def _generate_clustered_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成聚集障碍物（成簇）"""
        np.random.seed(seed + 4000)
        obstacles = []
        
        # 生成几个聚集中心
        num_clusters = np.random.randint(3, 6)
        
        for _ in range(num_clusters):
            # 聚集中心
            cx = np.random.uniform(5, self.world_width - 5)
            cy = np.random.uniform(5, self.world_height - 5)
            
            # 每个聚集有多个障碍物（可能重叠）
            num_in_cluster = np.random.randint(3, 8)
            cluster_radius = np.random.uniform(3, 6)
            
            for _ in range(num_in_cluster):
                # 在聚集中心附近随机放置
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, cluster_radius)
                x = cx + dist * np.cos(angle)
                y = cy + dist * np.sin(angle)
                
                # 确保在地图内
                x = np.clip(x, 2.5, self.world_width - 2.5)
                y = np.clip(y, 2.5, self.world_height - 2.5)
                
                obstacles.extend(self._generate_random_shape(center=(x, y), size_range=(1.0, 3.0)))
        
        return obstacles
    
    def _generate_corridor_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成走廊场景"""
        np.random.seed(seed + 5000)
        obstacles = []
        
        # 走廊方向
        horizontal = np.random.random() > 0.5
        
        if horizontal:
            # 水平走廊
            corridor_y = self.world_height / 2
            corridor_width = np.random.uniform(3, 6)
            
            # 上半部分障碍物
            for _ in range(np.random.randint(8, 15)):
                x = np.random.uniform(2.5, self.world_width - 2.5)
                y = np.random.uniform(corridor_y + corridor_width / 2 + 1, self.world_height - 2.5)
                w = np.random.uniform(1.5, 4.0)
                h = np.random.uniform(1.5, 4.0)
                obstacles.append(('rect', (x, y, w, h)))
            
            # 下半部分障碍物
            for _ in range(np.random.randint(8, 15)):
                x = np.random.uniform(2.5, self.world_width - 2.5)
                y = np.random.uniform(2.5, corridor_y - corridor_width / 2 - 1)
                w = np.random.uniform(1.5, 4.0)
                h = np.random.uniform(1.5, 4.0)
                obstacles.append(('rect', (x, y, w, h)))
        else:
            # 垂直走廊
            corridor_x = self.world_width / 2
            corridor_width = np.random.uniform(3, 6)
            
            # 左半部分障碍物
            for _ in range(np.random.randint(8, 15)):
                x = np.random.uniform(2.5, corridor_x - corridor_width / 2 - 1)
                y = np.random.uniform(2.5, self.world_height - 2.5)
                w = np.random.uniform(1.5, 4.0)
                h = np.random.uniform(1.5, 4.0)
                obstacles.append(('rect', (x, y, w, h)))
            
            # 右半部分障碍物
            for _ in range(np.random.randint(8, 15)):
                x = np.random.uniform(corridor_x + corridor_width / 2 + 1, self.world_width - 2.5)
                y = np.random.uniform(2.5, self.world_height - 2.5)
                w = np.random.uniform(1.5, 4.0)
                h = np.random.uniform(1.5, 4.0)
                obstacles.append(('rect', (x, y, w, h)))
        
        return obstacles
    
    def _generate_room_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成房间场景"""
        np.random.seed(seed + 6000)
        obstacles = []
        
        # 划分为2x2或3x3的网格房间
        grid_size = np.random.choice([2, 3])
        cell_w = self.world_width / grid_size
        cell_h = self.world_height / grid_size
        wall_thickness = 0.5
        door_width = np.random.uniform(2.0, 3.5)
        
        # 生成内墙（带门洞）
        for i in range(1, grid_size):
            # 垂直内墙
            x = i * cell_w
            # 随机门洞位置
            for j in range(grid_size):
                y_start = j * cell_h
                y_end = (j + 1) * cell_h
                door_y = np.random.uniform(y_start + 1, y_end - door_width - 1)
                
                # 门洞上方的墙
                if door_y > y_start + 0.5:
                    obstacles.append(('rect', (x - wall_thickness/2, y_start, wall_thickness, door_y - y_start)))
                # 门洞下方的墙
                if door_y + door_width < y_end - 0.5:
                    obstacles.append(('rect', (x - wall_thickness/2, door_y + door_width, wall_thickness, y_end - door_y - door_width)))
            
            # 水平内墙
            y = i * cell_h
            for j in range(grid_size):
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                door_x = np.random.uniform(x_start + 1, x_end - door_width - 1)
                
                if door_x > x_start + 0.5:
                    obstacles.append(('rect', (x_start, y - wall_thickness/2, door_x - x_start, wall_thickness)))
                if door_x + door_width < x_end - 0.5:
                    obstacles.append(('rect', (door_x + door_width, y - wall_thickness/2, x_end - door_x - door_width, wall_thickness)))
        
        # 每个房间内添加一些家具/障碍物
        for i in range(grid_size):
            for j in range(grid_size):
                cx = (i + 0.5) * cell_w
                cy = (j + 0.5) * cell_h
                
                num_furniture = np.random.randint(0, 3)
                for _ in range(num_furniture):
                    fx = cx + np.random.uniform(-cell_w/3, cell_w/3)
                    fy = cy + np.random.uniform(-cell_h/3, cell_h/3)
                    fw = np.random.uniform(0.8, 2.0)
                    fh = np.random.uniform(0.8, 2.0)
                    obstacles.append(('rect', (fx, fy, fw, fh)))
        
        return obstacles
    
    def _generate_mixed_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成混合场景（各种形状）"""
        np.random.seed(seed + 7000)
        obstacles = []
        
        num = np.random.randint(15, 25)
        for _ in range(num):
            obstacles.extend(self._generate_random_shape(diverse=True))
        
        return obstacles
    
    def _generate_maze_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成迷宫状障碍物"""
        np.random.seed(seed + 8000)
        obstacles = []
        
        wall_thickness = np.random.uniform(0.3, 0.6)
        
        # 随机生成一些长墙壁（水平和垂直）
        num_walls = np.random.randint(8, 15)
        
        for _ in range(num_walls):
            if np.random.random() > 0.5:
                # 水平墙壁
                x = np.random.uniform(2, self.world_width - 8)
                y = np.random.uniform(2, self.world_height - 2)
                length = np.random.uniform(5, 15)
                obstacles.append(('rect', (x, y, length, wall_thickness)))
            else:
                # 垂直墙壁
                x = np.random.uniform(2, self.world_width - 2)
                y = np.random.uniform(2, self.world_height - 8)
                length = np.random.uniform(5, 15)
                obstacles.append(('rect', (x, y, wall_thickness, length)))
        
        # 添加一些小障碍物
        for _ in range(np.random.randint(5, 10)):
            obstacles.extend(self._generate_random_shape(size_range=(0.8, 2.0)))
        
        return obstacles
    
    def _generate_open_center_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成中心开阔、边缘密集的场景"""
        np.random.seed(seed + 9000)
        obstacles = []
        
        center_x, center_y = self.world_width / 2, self.world_height / 2
        safe_radius = np.random.uniform(6, 10)  # 中心安全区域半径
        
        num = np.random.randint(20, 35)
        for _ in range(num):
            # 在边缘区域放置障碍物
            for attempt in range(10):
                x = np.random.uniform(2.5, self.world_width - 2.5)
                y = np.random.uniform(2.5, self.world_height - 2.5)
                
                dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist_to_center > safe_radius:
                    obstacles.extend(self._generate_random_shape(center=(x, y), size_range=(1.0, 3.5)))
                    break
        
        return obstacles
    
    def _generate_random_shape(self, center: Optional[Tuple[float, float]] = None,
                               size_range: Tuple[float, float] = (1.5, 5.0),
                               diverse: bool = True) -> List[Tuple[str, Tuple]]:
        """
        生成随机形状的障碍物
        
        支持的形状：
        - 矩形（基础）
        - 正方形
        - L形
        - T形
        - 十字形
        - 近似圆形（多边形近似）
        - 三角形（多边形近似）
        """
        if center is None:
            cx = np.random.uniform(3, self.world_width - 3)
            cy = np.random.uniform(3, self.world_height - 3)
        else:
            cx, cy = center
        
        min_size, max_size = size_range
        
        if diverse:
            shape_type = np.random.choice([
                'rectangle', 'square', 'L_shape', 'T_shape', 
                'cross', 'circle_approx', 'triangle_approx', 'long_rect'
            ], p=[0.20, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10])
        else:
            shape_type = np.random.choice(['rectangle', 'square', 'long_rect'])
        
        obstacles = []
        
        if shape_type == 'rectangle':
            w = np.random.uniform(min_size, max_size)
            h = np.random.uniform(min_size, max_size)
            obstacles.append(('rect', (cx - w/2, cy - h/2, w, h)))
            
        elif shape_type == 'square':
            size = np.random.uniform(min_size, max_size)
            obstacles.append(('rect', (cx - size/2, cy - size/2, size, size)))
            
        elif shape_type == 'long_rect':
            # 长条形
            if np.random.random() > 0.5:
                w = np.random.uniform(max_size, max_size * 2)
                h = np.random.uniform(min_size * 0.3, min_size * 0.6)
            else:
                w = np.random.uniform(min_size * 0.3, min_size * 0.6)
                h = np.random.uniform(max_size, max_size * 2)
            obstacles.append(('rect', (cx - w/2, cy - h/2, w, h)))
            
        elif shape_type == 'L_shape':
            # L形：由两个矩形组成
            arm_length = np.random.uniform(min_size, max_size)
            arm_width = np.random.uniform(min_size * 0.3, min_size * 0.6)
            
            # 垂直臂
            obstacles.append(('rect', (cx - arm_width/2, cy - arm_length/2, arm_width, arm_length)))
            # 水平臂（底部）
            obstacles.append(('rect', (cx - arm_width/2, cy + arm_length/2 - arm_width, arm_length, arm_width)))
            
        elif shape_type == 'T_shape':
            # T形：由两个矩形组成
            stem_length = np.random.uniform(min_size, max_size)
            stem_width = np.random.uniform(min_size * 0.3, min_size * 0.5)
            top_length = np.random.uniform(min_size, max_size)
            
            # 垂直茎
            obstacles.append(('rect', (cx - stem_width/2, cy - stem_length/2, stem_width, stem_length)))
            # 水平顶部
            obstacles.append(('rect', (cx - top_length/2, cy - stem_length/2 - stem_width, top_length, stem_width)))
            
        elif shape_type == 'cross':
            # 十字形
            arm_length = np.random.uniform(min_size, max_size)
            arm_width = np.random.uniform(min_size * 0.3, min_size * 0.5)
            
            # 垂直臂
            obstacles.append(('rect', (cx - arm_width/2, cy - arm_length/2, arm_width, arm_length)))
            # 水平臂
            obstacles.append(('rect', (cx - arm_length/2, cy - arm_width/2, arm_length, arm_width)))
            
        elif shape_type == 'circle_approx':
            # 近似圆形（用多个小矩形）
            radius = np.random.uniform(min_size/2, max_size/2)
            num_segments = 8
            
            for i in range(num_segments):
                angle = i * 2 * np.pi / num_segments
                segment_cx = cx + radius * 0.7 * np.cos(angle)
                segment_cy = cy + radius * 0.7 * np.sin(angle)
                segment_size = radius * 0.6
                obstacles.append(('rect', (segment_cx - segment_size/2, segment_cy - segment_size/2, 
                                          segment_size, segment_size)))
            # 中心
            obstacles.append(('rect', (cx - radius*0.5, cy - radius*0.5, radius, radius)))
            
        elif shape_type == 'triangle_approx':
            # 近似三角形（用多个矩形）
            base = np.random.uniform(min_size, max_size)
            height = np.random.uniform(min_size, max_size)
            
            # 用3个矩形近似
            obstacles.append(('rect', (cx - base/2, cy - height/4, base, height/4)))
            obstacles.append(('rect', (cx - base/3, cy, base*2/3, height/4)))
            obstacles.append(('rect', (cx - base/6, cy + height/4, base/3, height/4)))
        
        return obstacles


class GlobalMapDataCollector:
    """收集全局地图预测训练数据（增强版）"""

    def __init__(self, data_dir: str = "./global_map_training_data", 
                 sequence_length: int = 5,
                 grid_size: int = 400,
                 # 新增：更真实的物理参数
                 robot_speed_range: Tuple[float, float] = (2.0, 6.0),  # 机器人速度范围 m/s
                 sensor_trigger_interval: int = 3,  # 传感器触发间隔（每N步触发一次）
                 dt: float = 0.05):  # 仿真时间步长（秒）
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.grid_size = grid_size
        os.makedirs(data_dir, exist_ok=True)
        
        # 边界排除（地图物理边界）
        self.border_margin = 10
        
        # 多样化地图生成器
        self.map_generator = DiverseMapGenerator()
        
        # 真实物理参数
        self.robot_speed_range = robot_speed_range  # 机器人速度更快
        self.sensor_trigger_interval = sensor_trigger_interval  # 传感器触发更慢
        self.dt = dt  # 更细的时间步长

    def collect_episode(self, episode_id: int, max_steps: int = 500) -> Dict:
        """收集一个episode的全局地图数据"""
        
        # 随机选择触发模式（更倾向于现实中常用的模式）
        # 现实中为避免串扰，通常使用 sequential 或 interleaved
        trigger_modes = ['sequential', 'interleaved', 'sector']
        trigger_weights = [0.4, 0.4, 0.2]  # sequential和interleaved更常用
        selected_trigger_mode = np.random.choice(trigger_modes, p=trigger_weights)
        
        # 创建环境（使用更细的时间步长）
        core = RingSonarCore(
            world_width=40.0,
            world_height=40.0,
            dt=self.dt,  # 更细的时间步长 (50ms)
            trigger_mode=selected_trigger_mode
        )
        renderer = RingSonarRenderer(core, render_mode=None, enable_prediction=False)
        
        # 【关键】使用多样化地图生成器替换默认障碍物生成
        core.obstacles = self.map_generator.generate_obstacles(seed=episode_id)
        core._have_map = True
        
        # 找安全起点并重置
        core.robot_pos = core._find_safe_start()
        core.robot_angle = float(np.random.randint(0, 360))
        core.velocity = 0.0
        core.angular_velocity = 0.0
        core.sim_time = 0.0
        core.feature_map.fill(0.0)
        core.global_feature_map.fill(0.0)
        core.sonar_readings.fill(core.sensor_max_range)

        # 【关键】创建完整的全局真实地图（Ground Truth）
        global_ground_truth = self._create_global_ground_truth(core, renderer)
        
        # 全局累积地图（随时间更新）
        global_accumulated = np.full((self.grid_size, self.grid_size), 127, dtype=np.uint8)
        global_visit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint16)
        
        # 帧缓冲区
        frame_buffer = []
        sequences = []
        
        # 传感器触发计数器
        sensor_trigger_counter = 0

        for step in range(max_steps):
            # 随机移动，增加探索多样性（更快的速度）
            if step % 40 == 0:  # 调整速度变化频率
                speed = np.random.uniform(*self.robot_speed_range)
                # 随机前进或后退
                if np.random.random() > 0.15:  # 85%概率前进
                    linear_vel = speed
                else:
                    linear_vel = -speed * 0.5  # 后退速度较慢
                    
                angular_vel = np.random.uniform(-2.0, 2.0)  # 更大的转向范围
                core.set_velocity(float(linear_vel), float(angular_vel))

            core.step()
            
            # 传感器触发控制（模拟真实传感器的触发间隔）
            sensor_trigger_counter += 1
            if sensor_trigger_counter >= self.sensor_trigger_interval:
                sensor_trigger_counter = 0
                renderer._update_occupancy_grid()
                
                # 更新全局累积地图
                self._update_global_accumulated(
                    global_accumulated, 
                    global_visit_count,
                    renderer.occupancy_grid,
                    renderer.visit_count
                )

            # 每隔一段时间保存样本（基于传感器触发后）
            # 采样间隔 = 传感器触发间隔 * 采样倍数
            sample_interval = self.sensor_trigger_interval * 3  # 每3次传感器触发采样一次
            if step % sample_interval == 0 and step > 0:
                frame = {
                    'local_occupancy': renderer.occupancy_grid.copy(),
                    'global_accumulated': global_accumulated.copy(),
                    'global_visit_count': global_visit_count.copy(),
                    'robot_pos': core.robot_pos.copy(),
                    'step': step
                }
                frame_buffer.append(frame)

                # 创建序列样本
                if len(frame_buffer) >= self.sequence_length:
                    sequence_frames = frame_buffer[-self.sequence_length:]
                    
                    # 计算当前的探索覆盖率
                    known_ratio = (global_accumulated != 127).sum() / (self.grid_size ** 2)
                    
                    # 只保存有一定探索量的样本（覆盖率5%-80%）
                    if 0.05 < known_ratio < 0.80:
                        sequence_data = {
                            'sequence_frames': sequence_frames,
                            'global_ground_truth': global_ground_truth,  # 完整真实地图
                            'current_known_mask': (global_accumulated != 127),  # 当前已知区域
                            'known_ratio': known_ratio,
                            'episode_id': episode_id,
                            'step': step
                        }
                        sequences.append(sequence_data)

        return {
            'episode_id': episode_id,
            'sequences': sequences,
            'total_steps': max_steps,
            'final_known_ratio': (global_accumulated != 127).sum() / (self.grid_size ** 2)
        }

    def _create_global_ground_truth(self, core: RingSonarCore, 
                                    renderer: RingSonarRenderer) -> np.ndarray:
        """
        创建完整的全局真实地图（上帝视角）
        
        返回：
            0 = 空闲区域
            1 = 障碍物
           -1 = 地图边界（不参与训练）
        """
        gt = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # 标注所有障碍物
        for kind, data in core.obstacles:
            if kind == 'rect':
                x, y, w, h = data
                
                gx1 = int(x / renderer.grid_resolution)
                gy1 = int(y / renderer.grid_resolution)
                gx2 = int((x + w) / renderer.grid_resolution)
                gy2 = int((y + h) / renderer.grid_resolution)
                
                gx1 = max(0, min(gx1, self.grid_size))
                gy1 = max(0, min(gy1, self.grid_size))
                gx2 = max(0, min(gx2, self.grid_size))
                gy2 = max(0, min(gy2, self.grid_size))
                
                gt[gy1:gy2, gx1:gx2] = 1
        
        # 地图物理边界设为-1（不参与训练）
        m = self.border_margin
        gt[:m, :] = -1
        gt[-m:, :] = -1
        gt[:, :m] = -1
        gt[:, -m:] = -1
        
        return gt

    def _update_global_accumulated(self, 
                                   global_acc: np.ndarray,
                                   global_visit: np.ndarray,
                                   local_occ: np.ndarray,
                                   local_visit: np.ndarray):
        """
        更新全局累积地图
        
        策略：使用访问次数加权的融合
        """
        # 找到新观测到的区域
        new_known = (local_occ != 127) & (local_visit > 0)
        
        # 更新全局地图：新观测覆盖旧数据
        global_acc[new_known] = local_occ[new_known]
        
        # 更新访问计数
        global_visit[new_known] = np.maximum(
            global_visit[new_known], 
            local_visit[new_known]
        )

    def collect_dataset(self, num_episodes: int, max_steps_per_episode: int = 500):
        """收集完整的数据集"""
        print("=" * 60)
        print("🗺️  全局地图预测数据收集 (增强版)")
        print("=" * 60)
        print(f"Episode数量: {num_episodes}")
        print(f"每episode最大步数: {max_steps_per_episode}")
        print(f"序列长度: {self.sequence_length}")
        print(f"地图尺寸: {self.grid_size}x{self.grid_size}")
        print("\n🎨 多样化特性:")
        print("  - 障碍物形状: 矩形、L形、T形、十字形、圆形近似等")
        print("  - 场景类型: 稀疏、密集、聚集、走廊、房间、迷宫等")
        print("  - 边界类型: 完整墙、开放角、缺口墙、厚墙、不规则等")

        all_episodes = []

        for episode_id in tqdm(range(num_episodes), desc="收集数据"):
            episode_data = self.collect_episode(episode_id, max_steps_per_episode)
            all_episodes.append(episode_data)

        # 保存数据
        output_path = os.path.join(self.data_dir, 'training_data.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_episodes, f)

        print(f"\n数据已保存至: {output_path}")

        # 统计信息
        total_sequences = sum(len(ep['sequences']) for ep in all_episodes)
        avg_known_ratio = np.mean([ep['final_known_ratio'] for ep in all_episodes])
        
        print(f"\n📊 数据集统计:")
        print(f"  总序列数: {total_sequences}")
        print(f"  平均最终探索覆盖率: {avg_known_ratio*100:.1f}%")
        
        # 数据质量检查
        self._check_data_quality(all_episodes)

    def _check_data_quality(self, episodes):
        """检查数据质量"""
        print("\n" + "=" * 50)
        print("📊 数据质量检查")
        print("=" * 50)
        
        known_ratios = []
        obstacle_ratios = []
        
        for ep in episodes[:10]:
            for seq in ep['sequences'][:10]:
                gt = seq['global_ground_truth']
                known_mask = seq['current_known_mask']
                
                # 统计
                valid_mask = (gt >= 0)  # 排除边界
                known_ratios.append(seq['known_ratio'])
                obstacle_ratios.append((gt == 1).sum() / valid_mask.sum())
        
        print(f"  探索覆盖率范围: {min(known_ratios)*100:.1f}% - {max(known_ratios)*100:.1f}%")
        print(f"  平均障碍物占比: {np.mean(obstacle_ratios)*100:.1f}%")
        print(f"  障碍物占比范围: {min(obstacle_ratios)*100:.1f}% - {max(obstacle_ratios)*100:.1f}%")
        print(f"  ✅ 数据收集完成!")


def main():
    parser = argparse.ArgumentParser(description='收集全局地图预测训练数据（增强版）')
    parser.add_argument('--episodes', type=int, default=100,
                       help='收集的episode数量')
    parser.add_argument('--max-steps', type=int, default=800,
                       help='每个episode的最大步数（更多步数因为dt更小）')
    parser.add_argument('--data-dir', type=str, default='./global_map_training_data',
                       help='数据保存目录')
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='时间序列长度')
    
    # 新增：真实物理参数
    parser.add_argument('--robot-speed-min', type=float, default=2.0,
                       help='机器人最小速度 (m/s)')
    parser.add_argument('--robot-speed-max', type=float, default=6.0,
                       help='机器人最大速度 (m/s)')
    parser.add_argument('--sensor-interval', type=int, default=3,
                       help='传感器触发间隔（每N步触发一次，模拟真实传感器延迟）')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='仿真时间步长（秒），更小的值=更精细的模拟')

    args = parser.parse_args()
    
    print("\n🔧 物理参数配置:")
    print(f"  机器人速度: {args.robot_speed_min} - {args.robot_speed_max} m/s")
    print(f"  传感器触发间隔: 每{args.sensor_interval}步 ({args.sensor_interval * args.dt * 1000:.0f}ms)")
    print(f"  仿真时间步长: {args.dt * 1000:.0f}ms")
    print(f"  单次完整扫描时间: ~{args.sensor_interval * args.dt * 12:.2f}s (sequential模式)")

    collector = GlobalMapDataCollector(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        robot_speed_range=(args.robot_speed_min, args.robot_speed_max),
        sensor_trigger_interval=args.sensor_interval,
        dt=args.dt
    )
    collector.collect_dataset(args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
