"""
工具模块

包含:
- fisher: Fisher信息计算
- geometry: 几何工具函数
"""

from .fisher import SonarFisherCalculator
from .geometry import clamp, angnorm_deg, angdiff_deg, add_global_feature

__all__ = [
    "SonarFisherCalculator",
    "clamp", 
    "angnorm_deg", 
    "angdiff_deg",
    "add_global_feature"
]
