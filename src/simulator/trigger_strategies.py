#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器触发策略实现模块

包含具体的触发策略类，用于解耦 TriggerManager 和具体的算法逻辑。
"""

import numpy as np
import math
import random
import os
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Dict

from src.simulator.sensors import SonarSensor

@dataclass
class TriggerContext:
    """传递给策略的上下文信息"""
    step_count: int = 0
    sim_time: float = 0.0
    sonar_readings: np.ndarray = field(default_factory=lambda: np.zeros(12))
    robot_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    robot_angle: float = 0.0
    
    # 贪心策略需要的额外信息
    sensors: Optional[List[SonarSensor]] = None
    sensor_ready_times: np.ndarray = field(default_factory=lambda: np.zeros(12))
    global_feature_map: Optional[np.ndarray] = None
    feature_map_resolution: float = 0.1
    world_dims: Tuple[float, float] = (20.0, 20.0) # width, height
    sensor_max_range: float = 5.0
    
    # RL 策略需要的额外信息
    active_sensors_last_frame: List[int] = field(default_factory=list)

class BaseTriggerStrategy(ABC):
    """触发策略抽象基类"""
    
    def __init__(self, num_sensors: int):
        self.num_sensors = num_sensors

    @abstractmethod
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        """获取当前应激活的传感器ID列表"""
        pass

    @abstractmethod
    def advance(self) -> None:
        """推进一步内部状态"""
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """重置内部状态"""
        pass
        
    @abstractmethod
    def get_info(self) -> dict:
        """获取策略状态信息"""
        pass

class SequentialStrategy(BaseTriggerStrategy):
    """顺序扫描策略: 0->1->2..."""
    
    def __init__(self, num_sensors: int):
        super().__init__(num_sensors)
        self._index = 0
        
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        return [self._index]
        
    def advance(self) -> None:
        self._index = (self._index + 1) % self.num_sensors
        
    def reset(self) -> None:
        self._index = 0
        
    def get_info(self) -> dict:
        return {
            "name": "Sequential",
            "current_index": self._index
        }

class InterleavedStrategy(BaseTriggerStrategy):
    """交错扫描策略: 偶数组/奇数组交替"""
    
    def __init__(self, num_sensors: int, groups: Optional[List[List[int]]] = None):
        super().__init__(num_sensors)
        if groups is None:
            # 默认: 偶数ID一组, 奇数ID一组
            self.groups = [
                [i for i in range(num_sensors) if i % 2 == 0],
                [i for i in range(num_sensors) if i % 2 != 0]
            ]
        else:
            self.groups = groups
            
        self._group_index = 0
        self._sensor_index_in_group = 0
        
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        group = self.groups[self._group_index]
        # 确保索引有效
        if not group: return [0]
        idx = self._sensor_index_in_group % len(group)
        return [group[idx]]
        
    def advance(self) -> None:
        self._sensor_index_in_group += 1
        group = self.groups[self._group_index]
        if self._sensor_index_in_group >= len(group):
            self._sensor_index_in_group = 0
            self._group_index = (self._group_index + 1) % len(self.groups)
            
    def reset(self) -> None:
        self._group_index = 0
        self._sensor_index_in_group = 0
        
    def get_info(self) -> dict:
        return {
            "name": "Interleaved",
            "group_index": self._group_index,
            "sensor_index": self._sensor_index_in_group
        }

class SectorStrategy(BaseTriggerStrategy):
    """扇区轮询策略"""
    
    def __init__(self, num_sensors: int, 
                 sector_definition: Optional[Dict[str, List[int]]] = None,
                 sector_sequence: Optional[List[str]] = None):
        super().__init__(num_sensors)
        
        if sector_definition is None:
            # 默认扇区定义 (假设12个传感器)
            self.sectors = {
                "front": [11, 0, 1],
                "right": [2, 3, 4],
                "back": [5, 6, 7],
                "left": [8, 9, 10]
            }
        else:
            self.sectors = sector_definition
            
        if sector_sequence is None:
            self.sequence = ["front", "right", "back", "left"]
        else:
            self.sequence = sector_sequence
            
        self._sector_index = 0
        
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        sector_name = self.sequence[self._sector_index]
        return self.sectors.get(sector_name, [0])
        
    def advance(self) -> None:
        self._sector_index = (self._sector_index + 1) % len(self.sequence)
        
    def reset(self) -> None:
        self._sector_index = 0
        
    def get_info(self) -> dict:
        return {
            "name": "Sector",
            "current_sector": self.sequence[self._sector_index]
        }

class AllStrategy(BaseTriggerStrategy):
    """全部触发策略"""
    
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        return list(range(self.num_sensors))
        
    def advance(self) -> None:
        pass
        
    def reset(self) -> None:
        pass
        
    def get_info(self) -> dict:
        return {"name": "All"}

class GreedyStrategy(BaseTriggerStrategy):
    """
    贪心策略：优先扫描已知障碍物但Fisher信息较低的区域
    支持异步多传感器触发。
    """
    
    def __init__(self, num_sensors: int, exploration_prob: float = 0.2, max_active_sensors: int = 4):
        super().__init__(num_sensors)
        self.exploration_prob = exploration_prob
        self.max_active_sensors = max_active_sensors
        self._fallback_index = 0
        
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        # 检查必要上下文是否存在
        if (context.sonar_readings is None or 
            context.sensors is None or 
            context.global_feature_map is None or
            context.robot_pos is None):
            # 缺少上下文，回退到顺序扫描
            return [self._fallback_index]

        # 1. 找出所有处于就绪状态的传感器
        # 如果没有提供就绪时间，假设所有传感器都就绪（兼容旧代码）
        ready_sensors = []
        if context.sensor_ready_times is not None:
            for i in range(self.num_sensors):
                if context.sim_time >= context.sensor_ready_times[i]:
                    ready_sensors.append(i)
        else:
            ready_sensors = list(range(self.num_sensors))
                
        if not ready_sensors:
            return []
            
        # 2. 概率随机探索
        if random.random() < self.exploration_prob:
            # 随机选择最多 max_active_sensors 个就绪传感器
            count = min(len(ready_sensors), self.max_active_sensors)
            return random.sample(ready_sensors, count)

        # 3. 评估每个就绪传感器的"信息需求" (简化版：不使用 Fisher 信息)
        sensor_scores = []
        
        for sensor_id in ready_sensors:
            reading = context.sonar_readings[sensor_id]
            
            if reading >= context.sensor_max_range * 0.95:
                # 没探测到障碍物，赋予较低优先级
                score = 100.0 + random.random() * 10.0
            else:
                # 探测到障碍物，赋予高优先级。距离越近，优先级越高 (score 越小)
                # 这是一个纯粹的"障碍物追踪"策略，不考虑该位置是否已被建图
                score = reading
            
            sensor_scores.append((score, sensor_id))
            
        # 4. 选择分数最低（最有价值）的传感器
        # 随机打乱相同分数的传感器
        random.shuffle(sensor_scores)
        sensor_scores.sort(key=lambda x: x[0])
        
        selected_count = min(len(sensor_scores), self.max_active_sensors)
        selected_ids = [s[1] for s in sensor_scores[:selected_count]]
        
        return selected_ids
        
    def advance(self) -> None:
        self._fallback_index = (self._fallback_index + 1) % self.num_sensors
        
    def reset(self) -> None:
        self._fallback_index = 0
        
    def get_info(self) -> dict:
        return {
            "name": "Greedy",
            "exploration_prob": self.exploration_prob,
            "max_active_sensors": self.max_active_sensors
        }

class RLStrategy(BaseTriggerStrategy):
    """
    强化学习触发策略：使用训练好的 PPO 模型进行决策
    """
    def __init__(self, num_sensors: int, model_path: Optional[str] = None):
        super().__init__(num_sensors)
        self.model = None
        self.vec_normalize = None
        self.occupancy_map = None
        self.staleness_map = None
        self.map_size = 0
        self.stack_size = 3
        self.obs_history = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """延迟加载模型和归一化参数"""
        if os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import VecNormalize
                
                self.model = PPO.load(model_path)
                print(f"✅ 成功加载 RL 模型: {model_path}")
                
                # 尝试加载归一化参数
                stats_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
                if os.path.exists(stats_path):
                    # 创建一个具有相同观察空间的 Dummy 环境来加载 stats
                    from src.rl.trigger_env import SonarTriggerEnv
                    from stable_baselines3.common.vec_env import DummyVecEnv
                    dummy_env = DummyVecEnv([lambda: SonarTriggerEnv()])
                    self.vec_normalize = VecNormalize.load(stats_path, dummy_env)
                    # 禁用奖励归一化，只保留观察归一化
                    self.vec_normalize.training = False
                    self.vec_normalize.norm_reward = False
                    print(f"✅ 成功加载归一化参数: {stats_path}")
            except Exception as e:
                print(f"❌ 加载 RL 模型或归一化参数失败: {e}")
    
    def _update_occupancy_map(self, context: TriggerContext):
        """同步更新内部占据栅格地图"""
        if context.global_feature_map is None:
            return
            
        if self.occupancy_map is None:
            # 初始化地图 (0.5 = 未知)
            self.map_size = context.global_feature_map.shape[0]
            self.occupancy_map = np.full((self.map_size, self.map_size), 0.5, dtype=np.float32)
            self.staleness_map = np.ones((self.map_size, self.map_size), dtype=np.float32)
            
        # 更新陈旧度 (随时间增加)
        self.staleness_map = np.clip(self.staleness_map + 0.01, 0.0, 1.0)

        if not context.active_sensors_last_frame:
            return

        res = context.feature_map_resolution
        ms = self.map_size
        ww, wh = context.world_dims
        center_offset_x = ms // 2 - ww // (2 * res)
        center_offset_y = ms // 2 - wh // (2 * res)

        for sensor_id in context.active_sensors_last_frame:
            reading = context.sonar_readings[sensor_id]
            sensor = context.sensors[sensor_id]
            
            s_pos = sensor.get_world_position(context.robot_pos, context.robot_angle)
            s_angle = sensor.get_world_angle(context.robot_angle)
            
            sx_pix = int(s_pos[0] / res + center_offset_x)
            sy_pix = int(s_pos[1] / res + center_offset_y)
            
            start_angle = s_angle - sensor.fov_angle / 2
            end_angle = s_angle + sensor.fov_angle / 2
            reading_pix = int(reading / res)
            
            # 限制范围
            roi_radius = reading_pix + 5
            x_min, x_max = max(0, sx_pix - roi_radius), min(ms, sx_pix + roi_radius)
            y_min, y_max = max(0, sy_pix - roi_radius), min(ms, sy_pix + roi_radius)
            
            if x_max <= x_min or y_max <= y_min: continue
            
            roi_map = self.occupancy_map[y_min:y_max, x_min:x_max]
            rel_sx, rel_sy = sx_pix - x_min, sy_pix - y_min
            
            # 更新空闲区域
            mask_free = np.zeros_like(roi_map, dtype=np.uint8)
            cv2.ellipse(mask_free, (rel_sx, rel_sy), (reading_pix, reading_pix), 0, start_angle, end_angle, 255, -1)
            roi_map[mask_free > 0] = np.clip(roi_map[mask_free > 0] - 0.05, 0.0, 1.0)
            
            # 更新占用区域
            if reading < context.sensor_max_range * 0.95:
                mask_occ = np.zeros_like(roi_map, dtype=np.uint8)
                cv2.ellipse(mask_occ, (rel_sx, rel_sy), (reading_pix, reading_pix), 0, start_angle, end_angle, 255, 3)
                roi_map[mask_occ > 0] = np.clip(roi_map[mask_occ > 0] + 0.15, 0.0, 1.0)
            
            self.occupancy_map[y_min:y_max, x_min:x_max] = roi_map
            # 更新陈旧度
            self.staleness_map[y_min:y_max, x_min:x_max][mask_free > 0] = 0.0

    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        if self.model is None:
            return []
            
        # 1. 更新内部占据地图
        self._update_occupancy_map(context)
        
        # 如果地图尚未初始化或机器人位置缺失，返回空列表
        if self.occupancy_map is None or context.robot_pos is None:
            return []
            
        # 2. 构造 Observation
        res = context.feature_map_resolution
        ms = self.map_size
        ww, wh = context.world_dims
        
        gx = int(context.robot_pos[0] / res + ms // 2 - ww // (2 * res))
        gy = int(context.robot_pos[1] / res + ms // 2 - wh // (2 * res))
        
        # 裁剪 125x125 区域 (12.5m x 12.5m)，匹配传感器最大量程
        half_size = 62
        x1, x2 = max(0, gx - half_size), min(ms, gx + half_size + 1)
        y1, y2 = max(0, gy - half_size), min(ms, gy + half_size + 1)
        
        local_map = np.zeros((125, 125, 2), dtype=np.float32)
        local_map[:, :, 0] = 0.5
        local_map[:, :, 1] = 1.0
        
        crop_occ = self.occupancy_map[y1:y2, x1:x2]
        crop_stale = self.staleness_map[y1:y2, x1:x2]
        
        h, w = crop_occ.shape
        dy1 = half_size - (gy - y1)
        dx1 = half_size - (gx - x1)
        
        local_map[dy1:dy1+h, dx1:dx1+w, 0] = crop_occ
        local_map[dy1:dy1+h, dx1:dx1+w, 1] = crop_stale
        
        ready_status = np.zeros(12, dtype=np.float32)
        for i in range(12):
            wait_time = max(0, context.sensor_ready_times[i] - context.sim_time)
            ready_status[i] = min(1.0, wait_time / 0.1)
            
        obs_current = {
            "local_map": local_map,
            "sensor_ready": ready_status,
            "last_readings": context.sonar_readings.astype(np.float32)
        }
        
        # 3. 更新历史记录并堆叠
        if not self.obs_history:
            for _ in range(self.stack_size):
                self.obs_history.append(obs_current)
        else:
            self.obs_history.pop(0)
            self.obs_history.append(obs_current)
            
        stacked_local_map = np.concatenate([o["local_map"] for o in self.obs_history], axis=-1)
        stacked_ready = np.concatenate([o["sensor_ready"] for o in self.obs_history], axis=0)
        stacked_readings = np.concatenate([o["last_readings"] for o in self.obs_history], axis=0)
        
        obs_stacked = {
            "local_map": stacked_local_map,
            "sensor_ready": stacked_ready,
            "last_readings": stacked_readings
        }
        
        # 4. 应用归一化并增加 Batch 维度
        if self.vec_normalize is not None:
            obs_final = self.vec_normalize.normalize_obs(obs_stacked)
        else:
            obs_final = {k: np.expand_dims(v, 0) for k, v in obs_stacked.items()}
        
        # 5. 模型预测
        action, _ = self.model.predict(obs_final, deterministic=True)
        
        # 如果返回的是 Batch 结果 (1, 12)，取第一个 (12,)
        if len(action.shape) > 1:
            action = action[0]
        
        triggered_ids = []
        for i in range(12):
            if action[i] == 1:
                triggered_ids.append(i)
                
        return triggered_ids
    
    def reset(self) -> None:
        if self.occupancy_map is not None:
            self.occupancy_map.fill(0.5)
        if self.staleness_map is not None:
            self.staleness_map.fill(1.0)
        self.obs_history = []
        
    def advance(self) -> None:
        """RL 策略的状态更新在 get_active_sensors 中完成，此处无需操作"""
        pass
        
    def get_info(self) -> dict:
        return {"name": "RL_Strategy", "model_loaded": self.model is not None}
