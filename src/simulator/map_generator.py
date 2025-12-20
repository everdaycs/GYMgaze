#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多样化地图生成器

生成多种类型的地图场景用于训练和测试
"""

import numpy as np
from typing import List, Tuple, Optional


class DiverseMapGenerator:
    """生成多样化的地图配置"""
    
    # 场景类型及其概率（简化版：减少复杂度）
    SCENE_TYPES = [
        'sparse',           # 稀疏场景（5-8个障碍物）
        'simple',           # 简单场景（8-12个障碍物，标准矩形）
        'corridor',         # 走廊场景（简化版）
        'rooms',            # 房间场景（简化版）
    ]
    
    # 调整概率：增加简单场景的概率
    SCENE_PROBABILITIES = [0.35, 0.40, 0.15, 0.10]
    
    def __init__(self, world_width: float = 20.0, world_height: float = 20.0):
        self.world_width = world_width
        self.world_height = world_height
        
    def generate_obstacles(self, seed: int, scene_type: Optional[str] = None) -> List[Tuple[str, Tuple]]:
        """
        生成多样化的障碍物配置
        
        参数:
            seed: 随机种子
            scene_type: 指定场景类型，None则随机选择
        
        返回: List of (type, data)
            - ('rect', (x, y, w, h))
        """
        np.random.seed(seed)
        
        # 随机选择或使用指定的场景类型
        if scene_type is None:
            scene_type = np.random.choice(self.SCENE_TYPES, p=self.SCENE_PROBABILITIES)
        elif scene_type not in self.SCENE_TYPES:
            raise ValueError(f"Unknown scene type: {scene_type}. Available: {self.SCENE_TYPES}")
        
        obstacles = []
        
        # 生成边界墙（多样化）
        boundary_obstacles = self._generate_diverse_boundary(seed)
        obstacles.extend(boundary_obstacles)
        
        # 根据场景类型生成障碍物
        generator_map = {
            'sparse': self._generate_sparse_obstacles,
            'simple': self._generate_simple_obstacles,
            'corridor': self._generate_corridor_obstacles,
            'rooms': self._generate_room_obstacles,
        }
        
        obstacles.extend(generator_map[scene_type](seed))
        
        return obstacles
    
    def get_scene_type_from_seed(self, seed: int) -> str:
        """根据种子返回会生成的场景类型（不实际生成）"""
        np.random.seed(seed)
        return np.random.choice(self.SCENE_TYPES, p=self.SCENE_PROBABILITIES)
    
    def _generate_diverse_boundary(self, seed: int) -> List[Tuple[str, Tuple]]:
        """生成边界墙（简化版：只使用标准四面墙）"""
        obstacles = []
        wall = 0.5
        
        # 标准四面墙
        obstacles = [
            ('rect', (0.0, 0.0, self.world_width, wall)),
            ('rect', (0.0, self.world_height - wall, self.world_width, wall)),
            ('rect', (0.0, 0.0, wall, self.world_height)),
            ('rect', (self.world_width - wall, 0.0, wall, self.world_height))
        ]
        
        return obstacles
    
    def _generate_sparse_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成稀疏障碍物（5-8个简单矩形）"""
        np.random.seed(seed + 2000)
        obstacles = []
        num = np.random.randint(5, 9)
        
        # 只生成简单矩形，不使用复杂形状
        for _ in range(num):
            x = np.random.uniform(3, self.world_width - 5)
            y = np.random.uniform(3, self.world_height - 5)
            w = np.random.uniform(1.0, 2.5)
            h = np.random.uniform(1.0, 2.5)
            obstacles.append(('rect', (x, y, w, h)))
        
        return obstacles
    
    def _generate_simple_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成简单障碍物（8-12个标准矩形）"""
        np.random.seed(seed + 2500)
        obstacles = []
        num = np.random.randint(8, 13)
        
        # 简单均匀分布的矩形
        for _ in range(num):
            x = np.random.uniform(3, self.world_width - 5)
            y = np.random.uniform(3, self.world_height - 5)
            w = np.random.uniform(1.0, 3.0)
            h = np.random.uniform(1.0, 3.0)
            obstacles.append(('rect', (x, y, w, h)))
        
        return obstacles
    
    def _generate_corridor_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成走廊场景（简化版：只有两侧障碍物）"""
        np.random.seed(seed + 5000)
        obstacles = []
        
        # 走廊方向
        horizontal = np.random.random() > 0.5
        
        if horizontal:
            # 水平走廊（中间清空）
            corridor_y = self.world_height / 2
            corridor_width = 4.0  # 固定走廊宽度
            
            # 上半部分障碍物（6-8个）
            for _ in range(np.random.randint(6, 9)):
                x = np.random.uniform(2.5, self.world_width - 5)
                y = np.random.uniform(corridor_y + corridor_width / 2 + 1, self.world_height - 2.5)
                w = np.random.uniform(1.0, 2.5)
                h = np.random.uniform(1.0, 2.5)
                obstacles.append(('rect', (x, y, w, h)))
            
            # 下半部分障碍物（6-8个）
            for _ in range(np.random.randint(6, 9)):
                x = np.random.uniform(2.5, self.world_width - 5)
                y = np.random.uniform(2.5, corridor_y - corridor_width / 2 - 1)
                w = np.random.uniform(1.0, 2.5)
                h = np.random.uniform(1.0, 2.5)
                obstacles.append(('rect', (x, y, w, h)))
        else:
            # 垂直走廊（中间清空）
            corridor_x = self.world_width / 2
            corridor_width = 4.0
            
            # 左半部分障碍物（6-8个）
            for _ in range(np.random.randint(6, 9)):
                x = np.random.uniform(2.5, corridor_x - corridor_width / 2 - 1)
                y = np.random.uniform(2.5, self.world_height - 5)
                w = np.random.uniform(1.0, 2.5)
                h = np.random.uniform(1.0, 2.5)
                obstacles.append(('rect', (x, y, w, h)))
            
            # 右半部分障碍物（6-8个）
            for _ in range(np.random.randint(6, 9)):
                x = np.random.uniform(corridor_x + corridor_width / 2 + 1, self.world_width - 2.5)
                y = np.random.uniform(2.5, self.world_height - 5)
                w = np.random.uniform(1.0, 2.5)
                h = np.random.uniform(1.0, 2.5)
                obstacles.append(('rect', (x, y, w, h)))
        
        return obstacles
    
    def _generate_room_obstacles(self, seed) -> List[Tuple[str, Tuple]]:
        """生成房间场景（简化版：2x2网格，简单隔离）"""
        np.random.seed(seed + 6000)
        obstacles = []
        
        # 固定2x2的网格房间
        grid_size = 2
        cell_w = self.world_width / grid_size
        cell_h = self.world_height / grid_size
        wall_thickness = 0.5
        
        # 生成简单的分隔墙（中心十字形）
        # 水平分隔线
        obstacles.append(('rect', (0, self.world_height/2 - wall_thickness/2, 
                                  self.world_width, wall_thickness)))
        # 垂直分隔线（带门洞）
        door_width = 3.0
        door_y = self.world_height / 2 - door_width / 2
        
        # 垂直线的上段
        obstacles.append(('rect', (self.world_width/2 - wall_thickness/2, 0, 
                                  wall_thickness, door_y)))
        # 垂直线的下段
        obstacles.append(('rect', (self.world_width/2 - wall_thickness/2, door_y + door_width, 
                                  wall_thickness, self.world_height - door_y - door_width)))
        
        # 每个房间内添加少量家具（2-3个障碍物）
        num_rooms = 4
        for room_id in range(num_rooms):
            i = room_id % 2
            j = room_id // 2
            cx = (i + 0.5) * cell_w
            cy = (j + 0.5) * cell_h
            
            # 每个房间2-3个简单障碍物
            num_furniture = np.random.randint(2, 4)
            for _ in range(num_furniture):
                fx = cx + np.random.uniform(-cell_w/4, cell_w/4)
                fy = cy + np.random.uniform(-cell_h/4, cell_h/4)
                fw = np.random.uniform(0.8, 1.8)
                fh = np.random.uniform(0.8, 1.8)
                obstacles.append(('rect', (fx, fy, fw, fh)))
        
        return obstacles


# 简单地图生成器（向后兼容）
class SimpleMapGenerator:
    """简单地图生成器（原始模拟器使用的方式）"""
    
    def __init__(self, world_width: float = 40.0, world_height: float = 40.0):
        self.world_width = world_width
        self.world_height = world_height
    
    def generate_obstacles(self, seed: int, num_obstacles: int = 20) -> List[Tuple[str, Tuple]]:
        """生成简单的随机障碍物"""
        import random
        random.seed(seed)
        
        obstacles = []
        wall = 0.5
        
        # 边界墙
        obstacles += [
            ('rect', (0.0, 0.0, self.world_width, wall)),
            ('rect', (0.0, self.world_height - wall, self.world_width, wall)),
            ('rect', (0.0, 0.0, wall, self.world_height)),
            ('rect', (self.world_width - wall, 0.0, wall, self.world_height))
        ]
        
        # 随机障碍物
        for _ in range(num_obstacles):
            x = random.uniform(2.5, self.world_width - 2.5)
            y = random.uniform(2.5, self.world_height - 2.5)
            w = random.uniform(1.0, 3.0)
            h = random.uniform(1.0, 3.0)
            obstacles.append(('rect', (x, y, w, h)))
        
        return obstacles
