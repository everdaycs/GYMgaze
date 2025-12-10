"""
模拟器模块

包含:
- sensors: 传感器定义
- trigger: 触发策略管理
- map_generator: 地图生成器
- RingSonarCore: 环形超声波雷达核心仿真器（从主模块导入）
- RingSonarRenderer: 渲染器和可视化（从主模块导入）
"""

from .sensors import SonarSensor
from .trigger import (
    TriggerMode,
    TriggerConfig,
    TriggerManager,
    DEFAULT_TRIGGER_CONFIG,
    DATA_COLLECTION_TRIGGER_CONFIG,
    REAL_WORLD_TRIGGER_CONFIG,
    create_default_trigger_manager,
    create_data_collection_trigger_manager,
    create_demo_trigger_manager
)
from .map_generator import DiverseMapGenerator, SimpleMapGenerator

__all__ = [
    "SonarSensor",
    "TriggerMode",
    "TriggerConfig", 
    "TriggerManager",
    "DEFAULT_TRIGGER_CONFIG",
    "DATA_COLLECTION_TRIGGER_CONFIG",
    "REAL_WORLD_TRIGGER_CONFIG",
    "create_default_trigger_manager",
    "create_data_collection_trigger_manager",
    "create_demo_trigger_manager",
    "DiverseMapGenerator",
    "SimpleMapGenerator",
]
