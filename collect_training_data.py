#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局地图预测训练数据收集脚本

收集多样化的地图数据用于训练全局地图预测模型
"""

import numpy as np
import cv2
import os
import sys
import pickle
import argparse
import math
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ring_sonar_simulator import RingSonarCore, RingSonarRenderer
from configs import (
    SimulationConfig, RobotPhysicsConfig, SensorConfig, WorldConfig,
    DEFAULT_CONFIG, DATA_COLLECTION_CONFIG, print_config
)

# 使用共享的地图生成器
from src.simulator.map_generator import DiverseMapGenerator

class GlobalMapDataCollector:
    """收集全局地图预测训练数据（增强版）"""

    def __init__(self, data_dir: str = "./data/global_map_training_data", 
                 sequence_length: int = 5,
                 grid_size: int = None,  # 如果为None，则根据配置自动计算
                 config: SimulationConfig = None,  # 使用配置对象
                 trigger_mode: str = None,  # 触发模式：None表示随机，指定则使用固定模式
                 randomize_trigger: bool = True):  # 是否随机化触发模式
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        # 使用配置对象（如果提供）或使用默认数据收集配置
        if config is None:
            config = DATA_COLLECTION_CONFIG
        self.config = config
        
        # 触发模式设置
        self.trigger_mode = trigger_mode
        self.randomize_trigger = randomize_trigger
        
        # 根据配置计算栅格尺寸（如果未指定）
        if grid_size is None:
            grid_size = config.world.grid_size
        self.grid_size = grid_size
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 边界排除（地图物理边界）
        self.border_margin = 10
        
        # 多样化地图生成器
        self.map_generator = DiverseMapGenerator(
            world_width=config.world.world_width,
            world_height=config.world.world_height
        )

    def collect_episode(self, episode_id: int, max_steps: int = 500) -> Dict:
        """收集一个episode的全局地图数据"""
        
        # 创建环境（使用配置，根据参数设置触发模式）
        if self.trigger_mode is not None:
            # 使用指定的触发模式
            core = RingSonarCore(
                world_width=self.config.world.world_width,
                world_height=self.config.world.world_height,
                trigger_mode=self.trigger_mode,
                randomize_trigger=False,  # 不随机化，使用指定模式
                config=self.config
            )
        else:
            # 使用随机触发模式（默认行为）
            core = RingSonarCore(
                world_width=self.config.world.world_width,
                world_height=self.config.world.world_height,
                randomize_trigger=self.randomize_trigger,
                config=self.config
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
        # 速度变化计数器
        velocity_change_counter = 0

        for step in range(max_steps):
            # 使用配置中的速度变化间隔和随机速度
            velocity_change_counter += 1
            if velocity_change_counter >= self.config.robot.velocity_change_interval:
                velocity_change_counter = 0
                linear_vel, angular_vel = self.config.robot.get_random_velocity()
                core.set_velocity(linear_vel, angular_vel)

            core.step()
            
            # 传感器触发控制（使用配置中的间隔）
            sensor_trigger_counter += 1
            if sensor_trigger_counter >= self.config.robot.sensor_trigger_interval:
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
            sample_interval = self.config.robot.sensor_trigger_interval * 3  # 每3次传感器触发采样一次
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
                        # 创建当前已知区域掩码
                        current_known_mask = (global_accumulated != 127)
                        
                        # 创建局部 ground truth（只包含已知区域附近的障碍物）
                        local_ground_truth = self._create_local_ground_truth(
                            global_ground_truth=global_ground_truth,
                            known_mask=current_known_mask,
                            expansion_radius=20  # 向外扩展20像素（2米，基于0.1m分辨率）
                        )
                        
                        sequence_data = {
                            'sequence_frames': sequence_frames,
                            'local_ground_truth': local_ground_truth,  # 局部真实地图（已知区域附近）
                            'global_ground_truth': global_ground_truth,  # 保留完整地图用于评估
                            'current_known_mask': current_known_mask,  # 当前已知区域
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

    def _create_local_ground_truth(self, 
                                   global_ground_truth: np.ndarray,
                                   known_mask: np.ndarray,
                                   expansion_radius: int = 20) -> np.ndarray:
        """
        创建局部 ground truth（只包含已知区域附近的障碍物）
        
        参数:
            global_ground_truth: 完整的全局真实地图
            known_mask: 当前已知区域掩码 (bool array)
            expansion_radius: 从已知区域向外扩展的像素数（定义"附近"）
        
        返回:
            局部 ground truth:
                0 = 空闲区域（在扩展区域内）
                1 = 障碍物（在扩展区域内）
               -1 = 未知区域（超出扩展范围，不参与训练）
        
        核心思想:
            模型只需要预测已知区域"附近"的障碍物，而不是整个地图。
            这更符合实际SLAM场景：我们关心的是机器人当前可能遇到的障碍物。
        """
        import cv2
        
        # 从已知区域扩展，创建"感兴趣区域"
        known_uint8 = known_mask.astype(np.uint8) * 255
        
        # 使用形态学膨胀扩展已知区域
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (expansion_radius * 2 + 1, expansion_radius * 2 + 1)
        )
        expanded_region = cv2.dilate(known_uint8, kernel, iterations=1)
        expanded_mask = expanded_region > 0
        
        # 创建局部 ground truth
        local_gt = np.full_like(global_ground_truth, -1, dtype=np.int8)
        
        # 只在扩展区域内复制 ground truth
        local_gt[expanded_mask] = global_ground_truth[expanded_mask]
        
        # 保持边界为 -1
        m = self.border_margin
        local_gt[:m, :] = -1
        local_gt[-m:, :] = -1
        local_gt[:, :m] = -1
        local_gt[:, -m:] = -1
        
        return local_gt

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

    def _save_data_batch(self, episodes: list, batch_id: int = None):
        """Save batch to pickle file."""
        if batch_id is not None:
            filepath = os.path.join(self.data_dir, f'training_data_batch_{batch_id:03d}.pkl')
        else:
            filepath = os.path.join(self.data_dir, 'training_data.pkl')
        
        total_seqs = sum(len(ep['sequences']) for ep in episodes)
        print(f"  {len(episodes)} episodes ({total_seqs} seqs) -> {os.path.basename(filepath)}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(episodes, f)
        
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"  OK: {size_mb:.1f} MB")

    def collect_dataset(self, num_episodes: int, max_steps_per_episode: int = 500, batch_size: int = 50):
        """Collect with memory optimization."""
        batch_episodes = []
        batch_count = 0
        total_sequences = 0

        with tqdm(total=num_episodes, desc="Collecting") as pbar:
            for episode_id in range(num_episodes):
                episode_data = self.collect_episode(episode_id, max_steps_per_episode)
                batch_episodes.append(episode_data)
                total_sequences += len(episode_data['sequences'])
                pbar.update(1)
                
                if (episode_id + 1) % batch_size == 0:
                    batch_count += 1
                    print(f"Saving batch {batch_count}...")
                    self._save_data_batch(batch_episodes, batch_id=batch_count)
                    batch_episodes.clear()
        
        if batch_episodes:
            batch_count += 1
            print(f"Saving final batch...")
            self._save_data_batch(batch_episodes, batch_id=batch_count)
        
        print(f"Complete: {total_sequences} sequences in {batch_count} batches")

def main():
    parser = argparse.ArgumentParser(description='收集全局地图预测训练数据（增强版）')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='收集的episode数量')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='每个episode的最大步数（更多步数因为dt更小）')
    parser.add_argument('--data-dir', type=str, default='./data/global_map_training_data',
                       help='数据保存目录')
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='时间序列长度')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Memory optimization: save every N episodes')
    parser.add_argument('--trigger-mode', type=str, default=None,
                       choices=['sequential', 'interleaved', 'sector', 'all', 'greedy'],
                       help='传感器触发模式：None表示随机选择，指定则使用固定模式')
    parser.add_argument('--no-random-trigger', action='store_true',
                       help='禁用随机触发模式（仅在使用--trigger-mode时有效）')
    
    # 物理参数（可选覆盖配置）
    parser.add_argument('--robot-speed-min', type=float, default=None,
                       help='机器人最小速度 (m/s)，默认使用配置值')
    parser.add_argument('--robot-speed-max', type=float, default=None,
                       help='机器人最大速度 (m/s)，默认使用配置值')
    parser.add_argument('--sensor-interval', type=int, default=None,
                       help='传感器触发间隔（每N步触发一次），默认使用配置值')
    parser.add_argument('--dt', type=float, default=None,
                       help='仿真时间步长（秒），默认使用配置值')

    args = parser.parse_args()
    
    # 创建配置（使用默认数据收集配置，可通过参数覆盖）
    robot_config = RobotPhysicsConfig()
    
    # 覆盖指定的参数
    if args.robot_speed_min is not None:
        robot_config.linear_velocity_min = args.robot_speed_min
    if args.robot_speed_max is not None:
        robot_config.linear_velocity_max = args.robot_speed_max
    if args.sensor_interval is not None:
        robot_config.sensor_trigger_interval = args.sensor_interval
    if args.dt is not None:
        robot_config.dt = args.dt
    
    config = SimulationConfig(robot=robot_config)
    
    # 打印配置信息
    print_config(config, "数据收集配置")

    collector = GlobalMapDataCollector(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        config=config,
        trigger_mode=args.trigger_mode,
        randomize_trigger=not args.no_random_trigger
    )
    collector.collect_dataset(args.episodes, args.max_steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
