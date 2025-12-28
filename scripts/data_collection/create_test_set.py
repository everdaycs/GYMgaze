#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局地图预测模型 - 测试集生成脚本

生成一个包含多种场景、高质量轨迹的固定测试集，用于评估模型的泛化能力。
使用 9000+ 的种子范围，确保与训练数据（通常使用 0-8000）不重叠。
"""

import numpy as np
import os
import sys
import pickle
import math
from tqdm import tqdm
from typing import Dict, List, Tuple

# 添加项目根目录到路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulator.ring_sonar_simulator import RingSonarCore
from configs import DATA_COLLECTION_CONFIG
from src.simulator.map_generator import DiverseMapGenerator

class TestSetGenerator:
    def __init__(self, output_dir: str = "./data/test_set"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.config = DATA_COLLECTION_CONFIG
        self.map_generator = DiverseMapGenerator(
            world_width=20.0,
            world_height=20.0
        )

    def generate_episode(self, seed: int, scene_type: str, steps: int = 1000) -> List[Dict]:
        """生成单个高质量测试回合"""
        # 初始化核心（使用 20x20 地图）
        core = RingSonarCore(
            world_width=20.0,
            world_height=20.0,
            trigger_mode="greedy", # 使用贪心策略收集数据，因为其覆盖率最高
            config=self.config
        )
        
        # 生成指定类型的地图
        core.obstacles = self.map_generator.generate_obstacles(seed=seed, scene_type=scene_type)
        core._have_map = True
        core.robot_pos = core._find_safe_start()
        center_pos = core.robot_pos.copy()
        
        episode_data = []
        
        # 智能移动逻辑
        speed = 1.2
        step_dist = speed * core.dt
        current_angle_rad = 0.0

        for i in range(steps):
            # 1. 记录当前状态（作为输入）
            # 注意：这里记录的是 update_maps 之前的状态，或者根据模型需求记录
            
            # 2. 移动机器人 (Lissajous 覆盖)
            t = i * 0.02 # 稍微加快时间流逝
            target_x = center_pos[0] + 7.0 * math.sin(1.0 * t)
            target_y = center_pos[1] + 7.0 * math.sin(0.8 * t + math.pi/3)
            
            dx = target_x - core.robot_pos[0]
            dy = target_y - core.robot_pos[1]
            dist = math.hypot(dx, dy)
            desired_angle = math.atan2(dy, dx) if dist > 0 else current_angle_rad

            # 避障
            final_pos = None
            for a in range(0, 180, 15):
                for sign in [1, -1]:
                    test_angle = desired_angle + sign * math.radians(a)
                    test_pos = core.robot_pos + np.array([math.cos(test_angle) * step_dist, math.sin(test_angle) * step_dist])
                    if core._position_safe(test_pos):
                        final_pos = test_pos
                        current_angle_rad = test_angle
                        break
                if final_pos is not None: break
            
            if final_pos is not None:
                core.robot_pos = final_pos
                core.robot_angle = math.degrees(current_angle_rad) % 360
            
            # 3. 执行仿真步
            core.step()
            core.update_maps()
            
            # 4. 收集数据帧
            # 格式需与 GlobalMapDataset 兼容
            frame = {
                'step': i,
                'robot_pos': core.robot_pos.copy(),
                'robot_angle': core.robot_angle,
                'sonar_readings': core.sonar_readings.copy(),
                'global_feature_map': core.global_feature_map.copy(),
                # 地图真值（用于评估）
                'ground_truth_map': self._get_gt_map(core)
            }
            episode_data.append(frame)
            
        return episode_data

    def _get_gt_map(self, core: RingSonarCore) -> np.ndarray:
        """生成当前地图的占用栅格真值"""
        res = core.feature_map_resolution
        size = core.global_feature_map_size
        gt = np.zeros((size, size), dtype=np.uint8) # 0=空闲, 1=障碍物
        
        for y in range(size):
            for x in range(size):
                # 转换回世界坐标
                wx = (x - size // 2) * res + core.world_width / 2
                wy = (y - size // 2) * res + core.world_height / 2
                if core._point_in_obstacle(wx, wy):
                    gt[y, x] = 1
        return gt

    def run(self, num_episodes_per_type: int = 5):
        scene_types = ["sparse", "simple", "corridor", "rooms"]
        start_seed = 9000
        
        print(f"🚀 开始生成测试集 (保存至: {self.output_dir})")
        
        for scene in scene_types:
            print(f"🎬 场景类型: {scene}")
            for i in tqdm(range(num_episodes_per_type)):
                seed = start_seed + i
                data = self.generate_episode(seed, scene)
                
                filename = f"test_data_{scene}_s{seed}.pkl"
                with open(os.path.join(self.output_dir, filename), 'wb') as f:
                    pickle.dump(data, f)
            start_seed += 100

        print(f"✅ 测试集生成完毕！")

if __name__ == "__main__":
    generator = TestSetGenerator()
    generator.run(num_episodes_per_type=5) # 每个场景生成5个回合，共20个文件
