"""
配置模块

包含:
- robot_config: 机器人物理参数配置
"""

from .robot_config import (
    SimulationConfig,
    RobotPhysicsConfig,
    SensorConfig,
    WorldConfig,
    DEFAULT_CONFIG,
    DEMO_CONFIG,
    DATA_COLLECTION_CONFIG,
    print_config
)

__all__ = [
    "SimulationConfig",
    "RobotPhysicsConfig",
    "SensorConfig",
    "WorldConfig",
    "DEFAULT_CONFIG",
    "DEMO_CONFIG",
    "DATA_COLLECTION_CONFIG",
    "print_config"
]
