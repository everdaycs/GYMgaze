#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器触发策略模块

提供多种传感器触发模式，支持随机选择和自定义扩展。

触发模式:
- sequential: 顺序扫描，每次1个传感器，完全无干扰
- interleaved: 交错扫描，60°间隔，低干扰
- sector: 扇区轮询，每次3个传感器，可能有干扰
- all: 全部触发，高干扰（仅用于仿真）

使用示例:
    from src.simulator.trigger import TriggerManager, TriggerMode
    
    # 创建触发管理器
    manager = TriggerManager(num_sensors=12)
    
    # 随机选择模式
    manager.set_random_mode()
    
    # 获取当前应触发的传感器
    active_ids = manager.get_active_sensors()
    manager.advance()  # 前进到下一步
"""

import random
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


class TriggerMode(Enum):
    """触发模式枚举"""
    SEQUENTIAL = "sequential"      # 顺序扫描：0→1→2→...→11→0
    INTERLEAVED = "interleaved"    # 交错扫描：偶数组/奇数组交替
    SECTOR = "sector"              # 扇区轮询：前/右/后/左
    ALL = "all"                    # 全部触发（仅仿真用）
    GREEDY = "greedy"              # 贪心策略：基于外部提供的价值函数选择
    
    @classmethod
    def from_string(cls, mode_str: str) -> 'TriggerMode':
        """从字符串创建枚举"""
        mode_map = {
            'sequential': cls.SEQUENTIAL,
            'interleaved': cls.INTERLEAVED,
            'sector': cls.SECTOR,
            'all': cls.ALL,
            'greedy': cls.GREEDY
        }
        return mode_map.get(mode_str.lower(), cls.SEQUENTIAL)
    
    def __str__(self) -> str:
        return self.value


@dataclass
class TriggerConfig:
    """触发配置"""
    # 模式权重（用于随机选择）
    mode_weights: Dict[TriggerMode, float] = field(default_factory=lambda: {
        TriggerMode.SEQUENTIAL: 0.4,
        TriggerMode.INTERLEAVED: 0.4,
        TriggerMode.SECTOR: 0.2,
        TriggerMode.ALL: 0.0,  # 默认不随机选择ALL模式
        TriggerMode.GREEDY: 0.0
    })
    
    # 扇区配置（用于SECTOR模式）
    # 传感器ID: 0(0°), 1(30°), 2(60°), ..., 11(330°)
    sector_definition: Dict[str, List[int]] = field(default_factory=lambda: {
        "front": [11, 0, 1],   # 330°, 0°, 30° (机器人前方±30°)
        "right": [2, 3, 4],    # 60°, 90°, 120° (右侧)
        "back": [5, 6, 7],     # 150°, 180°, 210° (后方)
        "left": [8, 9, 10]     # 240°, 270°, 300° (左侧)
    })
    sector_sequence: List[str] = field(default_factory=lambda: ["front", "right", "back", "left"])
    
    # 交错组配置（用于INTERLEAVED模式）
    interleaved_groups: List[List[int]] = field(default_factory=lambda: [
        [0, 2, 4, 6, 8, 10],   # 偶数ID，间隔60°
        [1, 3, 5, 7, 9, 11]    # 奇数ID，间隔60°
    ])
    
    def get_random_mode(self, rng: Optional[np.random.Generator] = None) -> TriggerMode:
        """随机选择触发模式"""
        if rng is None:
            rng = np.random.default_rng()
        
        # 过滤权重为0的模式
        valid_modes = [(m, w) for m, w in self.mode_weights.items() if w > 0]
        if not valid_modes:
            return TriggerMode.SEQUENTIAL
        
        modes, weights = zip(*valid_modes)
        weights = np.array(weights) / sum(weights)  # 归一化
        
        return rng.choice(modes, p=weights)
    
    def set_mode_weight(self, mode: TriggerMode, weight: float) -> None:
        """设置模式权重"""
        self.mode_weights[mode] = max(0.0, weight)
    
    def enable_all_mode(self, weight: float = 0.1) -> None:
        """启用ALL模式（仅用于特定场景）"""
        self.mode_weights[TriggerMode.ALL] = weight


class TriggerManager:
    """
    传感器触发管理器
    
    管理传感器的触发顺序和模式切换。
    """
    
    def __init__(self, 
                 num_sensors: int = 12,
                 config: Optional[TriggerConfig] = None,
                 initial_mode: Optional[TriggerMode] = None):
        """
        初始化触发管理器
        
        Args:
            num_sensors: 传感器数量
            config: 触发配置
            initial_mode: 初始触发模式，None则随机选择
        """
        self.num_sensors = num_sensors
        self.config = config or TriggerConfig()
        
        # 贪心策略回调函数
        self._greedy_callback: Optional[Callable[[], List[int]]] = None
        
        # 当前模式
        if initial_mode is not None:
            self._mode = initial_mode
        else:
            self._mode = TriggerMode.SEQUENTIAL
        
        # 状态索引
        self._sequential_index = 0
        self._sector_index = 0
        self._interleaved_group_index = 0
        self._interleaved_sensor_index = 0
        
        # 统计信息
        self._step_count = 0
        self._mode_history: List[TriggerMode] = []
    
    @property
    def mode(self) -> TriggerMode:
        """当前触发模式"""
        return self._mode
    
    @mode.setter
    def mode(self, value: TriggerMode) -> None:
        """设置触发模式"""
        if value != self._mode:
            self._mode = value
            self._reset_indices()
    
    def set_mode(self, mode: TriggerMode) -> None:
        """设置触发模式"""
        self.mode = mode
    
    def set_mode_from_string(self, mode_str: str) -> None:
        """从字符串设置触发模式"""
        self.mode = TriggerMode.from_string(mode_str)
    
    def set_greedy_callback(self, callback: Callable[[], List[int]]) -> None:
        """设置贪心策略回调函数"""
        self._greedy_callback = callback
    
    def set_random_mode(self, rng: Optional[np.random.Generator] = None) -> TriggerMode:
        """随机选择触发模式"""
        self._mode = self.config.get_random_mode(rng)
        self._reset_indices()
        self._mode_history.append(self._mode)
        return self._mode
    
    def _reset_indices(self) -> None:
        """重置所有索引"""
        self._sequential_index = 0
        self._sector_index = 0
        self._interleaved_group_index = 0
        self._interleaved_sensor_index = 0
    
    def get_active_sensors(self) -> List[int]:
        """
        获取当前帧应激活的传感器ID列表
        
        Returns:
            激活的传感器ID列表
        """
        if self._mode == TriggerMode.SEQUENTIAL:
            return [self._sequential_index]
        
        elif self._mode == TriggerMode.INTERLEAVED:
            group = self.config.interleaved_groups[self._interleaved_group_index]
            return [group[self._interleaved_sensor_index]]
        
        elif self._mode == TriggerMode.SECTOR:
            sector_name = self.config.sector_sequence[self._sector_index]
            return self.config.sector_definition[sector_name]
        
        elif self._mode == TriggerMode.GREEDY:
            if self._greedy_callback:
                return self._greedy_callback()
            else:
                # 如果没有回调，回退到顺序模式
                return [self._sequential_index]
        
        elif self._mode == TriggerMode.ALL:
            return list(range(self.num_sensors))
        
        else:
            return [0]  # 默认返回第一个传感器
    
    def advance(self) -> None:
        """前进到下一个触发状态"""
        self._step_count += 1
        
        if self._mode == TriggerMode.SEQUENTIAL:
            self._sequential_index = (self._sequential_index + 1) % self.num_sensors
        
        elif self._mode == TriggerMode.INTERLEAVED:
            self._interleaved_sensor_index += 1
            group = self.config.interleaved_groups[self._interleaved_group_index]
            if self._interleaved_sensor_index >= len(group):
                self._interleaved_sensor_index = 0
                self._interleaved_group_index = (self._interleaved_group_index + 1) % len(self.config.interleaved_groups)
        
        elif self._mode == TriggerMode.SECTOR:
            self._sector_index = (self._sector_index + 1) % len(self.config.sector_sequence)
        
        elif self._mode == TriggerMode.GREEDY:
            # 贪心模式下，如果回退到顺序模式，需要更新索引
            if not self._greedy_callback:
                self._sequential_index = (self._sequential_index + 1) % self.num_sensors
        
        # ALL模式不需要前进
    
    def get_mode_info(self) -> Dict:
        """获取当前模式的详细信息"""
        info = {
            'mode': str(self._mode),
            'step_count': self._step_count,
            'active_sensors': self.get_active_sensors(),
        }
        
        if self._mode == TriggerMode.SEQUENTIAL:
            info['description'] = "顺序扫描，每次1个传感器"
            info['current_index'] = self._sequential_index
            info['scan_order'] = "0 → 1 → 2 → ... → 11 → 0"
        
        elif self._mode == TriggerMode.INTERLEAVED:
            info['description'] = "交错扫描，60°间隔"
            info['current_group'] = self._interleaved_group_index
            info['current_sensor_in_group'] = self._interleaved_sensor_index
            info['groups'] = self.config.interleaved_groups
        
        elif self._mode == TriggerMode.SECTOR:
            sector_name = self.config.sector_sequence[self._sector_index]
            info['description'] = f"扇区轮询 ({sector_name})，每次3个传感器"
            info['current_sector'] = sector_name
            info['sectors'] = self.config.sector_definition
        
        elif self._mode == TriggerMode.GREEDY:
            info['description'] = "贪心策略，基于信息增益选择传感器"
            info['has_callback'] = self._greedy_callback is not None
        
        elif self._mode == TriggerMode.ALL:
            info['description'] = "全部触发"
            info['warning'] = "此模式在真实环境中会产生严重干扰"
        
        return info
    
    def reset(self, randomize_mode: bool = False) -> None:
        """
        重置触发管理器
        
        Args:
            randomize_mode: 是否随机选择新模式
        """
        self._reset_indices()
        self._step_count = 0
        
        if randomize_mode:
            self.set_random_mode()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_steps': self._step_count,
            'current_mode': str(self._mode),
            'mode_history_length': len(self._mode_history),
            'mode_distribution': self._calculate_mode_distribution()
        }
    
    def _calculate_mode_distribution(self) -> Dict[str, float]:
        """计算模式分布"""
        if not self._mode_history:
            return {}
        
        counts = {}
        for mode in self._mode_history:
            mode_str = str(mode)
            counts[mode_str] = counts.get(mode_str, 0) + 1
        
        total = len(self._mode_history)
        return {k: v / total for k, v in counts.items()}


# ============== 便捷函数 ==============

def create_default_trigger_manager(num_sensors: int = 12) -> TriggerManager:
    """创建默认配置的触发管理器"""
    return TriggerManager(num_sensors=num_sensors)


def create_data_collection_trigger_manager(num_sensors: int = 12) -> TriggerManager:
    """创建数据收集专用的触发管理器（随机模式）"""
    config = TriggerConfig()
    manager = TriggerManager(num_sensors=num_sensors, config=config)
    manager.set_random_mode()
    return manager


def create_demo_trigger_manager(num_sensors: int = 12, 
                                mode: TriggerMode = TriggerMode.SEQUENTIAL) -> TriggerManager:
    """创建演示用的触发管理器（固定模式）"""
    return TriggerManager(num_sensors=num_sensors, initial_mode=mode)


# ============== 预定义配置 ==============

# 默认触发配置
DEFAULT_TRIGGER_CONFIG = TriggerConfig()

# 数据收集触发配置（更均衡的权重）
DATA_COLLECTION_TRIGGER_CONFIG = TriggerConfig(
    mode_weights={
        TriggerMode.SEQUENTIAL: 0.35,
        TriggerMode.INTERLEAVED: 0.35,
        TriggerMode.SECTOR: 0.25,
        TriggerMode.ALL: 0.05  # 小概率使用ALL模式增加数据多样性
    }
)

# 真实环境触发配置（避免干扰）
REAL_WORLD_TRIGGER_CONFIG = TriggerConfig(
    mode_weights={
        TriggerMode.SEQUENTIAL: 0.7,
        TriggerMode.INTERLEAVED: 0.3,
        TriggerMode.SECTOR: 0.0,  # 扇区模式可能有干扰
        TriggerMode.ALL: 0.0
    }
)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("触发管理器测试")
    print("=" * 60)
    
    # 创建管理器
    manager = TriggerManager(num_sensors=12)
    
    # 测试各种模式
    for mode in TriggerMode:
        manager.set_mode(mode)
        print(f"\n模式: {mode}")
        info = manager.get_mode_info()
        print(f"  描述: {info['description']}")
        
        # 模拟几步
        for i in range(5):
            active = manager.get_active_sensors()
            print(f"  Step {i}: 激活传感器 {active}")
            manager.advance()
    
    # 测试随机选择
    print("\n" + "=" * 60)
    print("随机模式测试")
    print("=" * 60)
    
    manager = TriggerManager(num_sensors=12, config=DATA_COLLECTION_TRIGGER_CONFIG)
    
    for episode in range(5):
        mode = manager.set_random_mode()
        print(f"Episode {episode}: {mode}")
    
    print("\n统计:", manager.get_statistics())
