"""
GYMgaze - 环形超声波雷达模拟器与全局地图预测

模块结构:
- simulator: 传感器定义
- models: 深度学习模型（GlobalMapPredictor）
- utils: 工具函数（Fisher计算、几何工具等）

注意: RingSonarCore 和 RingSonarRenderer 由于耦合较紧密
      暂时保留在 ring_sonar_simulator.py 中
"""

__version__ = "0.2.0"
__author__ = "GYMgaze Team"

from .models import GlobalMapPredictor
from .utils import SonarFisherCalculator
from .simulator.sensors import SonarSensor

__all__ = [
    "GlobalMapPredictor",
    "SonarFisherCalculator",
    "SonarSensor",
]
