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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Dict

from src.simulator.sensors import SonarSensor

@dataclass
class TriggerContext:
    """传递给策略的上下文信息"""
    step_count: int = 0
    sim_time: float = 0.0
    sonar_readings: Optional[np.ndarray] = None
    robot_pos: Optional[np.ndarray] = None
    robot_angle: float = 0.0
    
    # 贪心策略需要的额外信息
    sensors: Optional[List[SonarSensor]] = None
    sensor_ready_times: Optional[np.ndarray] = None
    global_feature_map: Optional[np.ndarray] = None
    feature_map_resolution: float = 0.1
    world_dims: Tuple[float, float] = (0.0, 0.0) # width, height
    sensor_max_range: float = 5.0

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

        # 3. 评估每个就绪传感器的"信息需求"
        sensor_scores = []
        
        for sensor_id in ready_sensors:
            reading = context.sonar_readings[sensor_id]
            
            # 如果没有探测到障碍物（最大量程），优先级较低
            if reading >= context.sensor_max_range * 0.95:
                # 赋予一个较低的基础分，但允许被选中（用于探索空区域）
                # 分数越高越好？这里我们用 min_fisher_val 越小越好。
                # 空区域: score = infinity? 或者一个大常数。
                score = 1000.0 
            else:
                # 计算障碍物位置估计
                sensor = context.sensors[sensor_id]
                sensor_pos = sensor.get_world_position(context.robot_pos, context.robot_angle)
                sensor_angle = sensor.get_world_angle(context.robot_angle)
                angle_rad = math.radians(sensor_angle)
                
                wx = sensor_pos[0] + math.cos(angle_rad) * reading
                wy = sensor_pos[1] + math.sin(angle_rad) * reading
                
                # 转换为地图坐标
                res = context.feature_map_resolution
                ms = context.global_feature_map.shape[0] # 假设是正方形
                ww, wh = context.world_dims
                
                val = 0.0
                # 使用传入的 world_dims 计算
                if ww > 0 and wh > 0:
                    gx = int(wx / res + ms // 2 - ww // (2 * res))
                    gy = int(wy / res + ms // 2 - wh // (2 * res))
                    
                    if 0 <= gx < ms and 0 <= gy < ms:
                        val = context.global_feature_map[gy, gx]
                
                # Fisher信息越低，不确定性越高，越需要扫描 -> score 越小越好
                score = val
            
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
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path)
                print(f"✅ 成功加载 RL 模型: {model_path}")
            except ImportError:
                print("⚠️ 未安装 stable-baselines3，RL 策略将回退到顺序扫描")
            except Exception as e:
                print(f"❌ 加载 RL 模型失败: {e}")
        
    def get_active_sensors(self, context: TriggerContext) -> List[int]:
        if self.model is None or context.global_feature_map is None:
            # 回退逻辑：如果模型没加载，每步发一个
            return [context.step_count % self.num_sensors]
            
        # 1. 构造与训练时一致的 Observation
        # 提取局部地图 (40x40)
        res = context.feature_map_resolution
        ms = context.global_feature_map.shape[0]
        ww, wh = context.world_dims
        
        gx = int(context.robot_pos[0] / res + ms // 2 - ww // (2 * res))
        gy = int(context.robot_pos[1] / res + ms // 2 - wh // (2 * res))
        
        half_size = 20
        x1, x2 = max(0, gx - half_size), min(ms, gx + half_size)
        y1, y2 = max(0, gy - half_size), min(ms, gy + half_size)
        
        local_map = np.zeros((40, 40, 1), dtype=np.float32)
        crop = context.global_feature_map[y1:y2, x1:x2]
        h, w = crop.shape
        dy1 = half_size - (gy - y1)
        dx1 = half_size - (gx - x1)
        local_map[dy1:dy1+h, dx1:dx1+w, 0] = crop
        
        # 传感器状态
        ready_status = np.zeros(12, dtype=np.float32)
        for i in range(12):
            wait_time = max(0, context.sensor_ready_times[i] - context.sim_time)
            ready_status[i] = min(1.0, wait_time / 0.1)
            
        obs = {
            "local_map": local_map,
            "sensor_ready": ready_status,
            "last_readings": context.sonar_readings.astype(np.float32)
        }
        
        # 2. 模型预测
        action, _ = self.model.predict(obs, deterministic=True)
        
        # 3. 转换动作为 ID 列表
        triggered_ids = []
        for i in range(12):
            if action[i] == 1 and context.sim_time >= context.sensor_ready_times[i]:
                triggered_ids.append(i)
                
        return triggered_ids
        
    def advance(self) -> None:
        pass
        
    def reset(self) -> None:
        pass
        
    def get_info(self) -> dict:
        return {"name": "RL_Strategy", "model_loaded": self.model is not None}
