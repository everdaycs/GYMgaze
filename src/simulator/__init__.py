"""
模拟器模块

包含:
- sensors: 传感器定义
- RingSonarCore: 环形超声波雷达核心仿真器（从主模块导入）
- RingSonarRenderer: 渲染器和可视化（从主模块导入）
"""

from .sensors import SonarSensor

# RingSonarCore 和 RingSonarRenderer 由于与主文件紧密耦合
# 暂时从 ring_sonar_simulator.py 导入
# 将来可以进一步模块化

__all__ = ["SonarSensor"]
