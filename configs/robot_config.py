#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人物理参数配置

该文件定义了机器人和传感器的物理参数，供模拟器和数据收集共享使用。
确保训练数据和推理环境使用相同的参数。
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RobotPhysicsConfig:
    """机器人物理参数配置"""
    
    # ============== 运动参数 ==============
    # 线速度范围 (m/s)
    linear_velocity_min: float = 2.0
    linear_velocity_max: float = 5.0
    # 最大线速度限制
    max_linear_velocity: float = 6.0
    
    # 角速度范围 (rad/s)
    angular_velocity_min: float = -2.0
    angular_velocity_max: float = 2.0
    # 最大角速度限制
    max_angular_velocity: float = 2.5
    
    # 后退速度倍率（相对于前进速度）
    backward_speed_ratio: float = 0.5
    # 前进概率（剩余为后退概率）
    forward_probability: float = 0.85
    
    # ============== 时间参数 ==============
    # 仿真时间步长 (秒)
    dt: float = 0.05
    
    # 速度变化间隔（步数）
    velocity_change_interval: int = 40
    
    # ============== 传感器触发参数 ==============
    # 传感器触发间隔（每N步触发一次）
    sensor_trigger_interval: int = 3
    
    # ============== 机器人尺寸 ==============
    robot_size: float = 0.5  # 机器人直径 (m)
    
    def get_random_velocity(self, rng=None) -> Tuple[float, float]:
        """
        获取随机速度
        
        Returns:
            Tuple[float, float]: (linear_velocity, angular_velocity)
        """
        import numpy as np
        if rng is None:
            rng = np.random
        
        # 线速度
        speed = rng.uniform(self.linear_velocity_min, self.linear_velocity_max)
        if rng.random() > self.forward_probability:
            # 后退
            linear_vel = -speed * self.backward_speed_ratio
        else:
            # 前进
            linear_vel = speed
        
        # 角速度
        angular_vel = rng.uniform(self.angular_velocity_min, self.angular_velocity_max)
        
        return float(linear_vel), float(angular_vel)


@dataclass
class SensorConfig:
    """传感器配置"""
    
    # 传感器数量
    num_sensors: int = 12
    
    # 传感器环半径 (m)
    ring_radius: float = 0.15
    
    # 单个传感器视场角 (degrees)
    fov_angle: float = 65.0
    
    # 最大探测距离 (m)
    max_range: float = 12.5


@dataclass
class WorldConfig:
    """世界配置"""
    
    # 世界尺寸 (m)
    world_width: float = 40.0
    world_height: float = 40.0
    
    # 像素/米（渲染用）
    pixel_per_meter: int = 20
    
    # 栅格地图分辨率
    grid_resolution: float = 0.1  # 米/栅格
    
    @property
    def grid_size(self) -> int:
        """栅格地图尺寸"""
        return int(self.world_width / self.grid_resolution)


@dataclass
class SimulationConfig:
    """完整仿真配置"""
    
    robot: RobotPhysicsConfig = field(default_factory=RobotPhysicsConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    
    # 触发模式选择权重
    trigger_mode_weights: dict = field(default_factory=lambda: {
        'sequential': 0.4,
        'interleaved': 0.4,
        'sector': 0.2
    })
    
    def get_random_trigger_mode(self, rng=None) -> str:
        """随机选择触发模式"""
        import numpy as np
        if rng is None:
            rng = np.random
        
        modes = list(self.trigger_mode_weights.keys())
        weights = list(self.trigger_mode_weights.values())
        return rng.choice(modes, p=weights)


# ============== 预设配置 ==============

# 默认配置（用于训练和推理）
DEFAULT_CONFIG = SimulationConfig()

# 快速演示配置（较慢的速度，便于观察）
DEMO_CONFIG = SimulationConfig(
    robot=RobotPhysicsConfig(
        linear_velocity_min=1.0,
        linear_velocity_max=3.0,
        dt=0.1,
        velocity_change_interval=50,
        sensor_trigger_interval=1  # 每步都触发，便于观察
    )
)

# 数据收集配置（与DEFAULT_CONFIG相同，但可以单独调整）
DATA_COLLECTION_CONFIG = SimulationConfig(
    robot=RobotPhysicsConfig(
        linear_velocity_min=2.0,
        linear_velocity_max=5.0,
        dt=0.05,
        velocity_change_interval=40,
        sensor_trigger_interval=3
    )
)


def print_config(config: SimulationConfig, name: str = "Config"):
    """打印配置信息"""
    print(f"\n{'='*50}")
    print(f"📋 {name}")
    print(f"{'='*50}")
    print(f"🤖 机器人物理参数:")
    print(f"   线速度范围: [{config.robot.linear_velocity_min}, {config.robot.linear_velocity_max}] m/s")
    print(f"   角速度范围: [{config.robot.angular_velocity_min}, {config.robot.angular_velocity_max}] rad/s")
    print(f"   后退速度倍率: {config.robot.backward_speed_ratio}")
    print(f"   前进概率: {config.robot.forward_probability}")
    print(f"   时间步长 (dt): {config.robot.dt} s")
    print(f"   速度变化间隔: {config.robot.velocity_change_interval} 步")
    print(f"   传感器触发间隔: {config.robot.sensor_trigger_interval} 步")
    print(f"\n📡 传感器参数:")
    print(f"   数量: {config.sensor.num_sensors}")
    print(f"   环半径: {config.sensor.ring_radius} m")
    print(f"   视场角: {config.sensor.fov_angle}°")
    print(f"   最大探测距离: {config.sensor.max_range} m")
    print(f"\n🌍 世界参数:")
    print(f"   尺寸: {config.world.world_width} x {config.world.world_height} m")
    print(f"   栅格分辨率: {config.world.grid_resolution} m")
    print(f"   栅格尺寸: {config.world.grid_size} x {config.world.grid_size}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 测试配置
    print_config(DEFAULT_CONFIG, "默认配置 (DEFAULT_CONFIG)")
    print_config(DEMO_CONFIG, "演示配置 (DEMO_CONFIG)")
    print_config(DATA_COLLECTION_CONFIG, "数据收集配置 (DATA_COLLECTION_CONFIG)")
    
    # 测试随机速度生成
    import numpy as np
    rng = np.random.default_rng(42)
    print("随机速度测试 (5次):")
    for i in range(5):
        v, w = DEFAULT_CONFIG.robot.get_random_velocity(rng)
        print(f"  #{i+1}: linear={v:+.2f} m/s, angular={w:+.2f} rad/s")
