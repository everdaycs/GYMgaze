#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器策略测试实验室

用于测试和验证不同的传感器触发策略。
"""

import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator.trigger import TriggerManager, TriggerMode, TriggerConfig
from src.simulator.trigger_strategies import TriggerContext
from src.simulator.sensors import SonarSensor

def create_mock_context(num_sensors=12):
    """创建模拟上下文"""
    # 模拟传感器
    sensors = []
    for i in range(num_sensors):
        sensors.append(SonarSensor(
            id=i, angle=i*30, offset_x=0, offset_y=0, 
            fov_angle=60, max_range=5.0
        ))
        
    # 模拟读数 (假设传感器0和1检测到障碍物)
    readings = np.full(num_sensors, 5.0)
    readings[0] = 2.0
    readings[1] = 3.0
    
    # 模拟地图 (全0)
    global_map = np.zeros((100, 100))
    
    return TriggerContext(
        step_count=0,
        sim_time=0.0,
        sonar_readings=readings,
        robot_pos=np.array([20.0, 20.0]),
        robot_angle=0.0,
        sensors=sensors,
        global_feature_map=global_map,
        feature_map_resolution=0.1,
        world_dims=(40.0, 40.0),
        sensor_max_range=5.0
    )

def test_strategy(mode_name, steps=20):
    """测试特定策略"""
    print(f"\n🧪 测试策略: {mode_name}")
    print("-" * 40)
    
    manager = TriggerManager(num_sensors=12)
    manager.set_mode_from_string(mode_name)
    
    context = create_mock_context()
    
    history = []
    for i in range(steps):
        context.step_count = i
        active = manager.get_active_sensors(context)
        history.append(active)
        print(f"Step {i:02d}: {active}")
        manager.advance()
        
    return history

def visualize_history(history, title):
    """可视化触发历史"""
    data = np.zeros((len(history), 12))
    for t, active_ids in enumerate(history):
        for sensor_id in active_ids:
            data[t, sensor_id] = 1
            
    plt.figure(figsize=(10, 6))
    plt.imshow(data.T, aspect='auto', cmap='Greys', interpolation='nearest')
    plt.title(f"Sensor Trigger History - {title}")
    plt.xlabel("Time Step")
    plt.ylabel("Sensor ID")
    plt.yticks(range(12))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 保存图片
    filename = f"sensor_strategy_lab/history_{title.lower()}.png"
    try:
        plt.savefig(filename)
        print(f"📊 图表已保存至: {filename}")
    except Exception as e:
        print(f"⚠️ 无法保存图表: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    # 测试所有模式
    modes = ["sequential", "interleaved", "sector", "greedy"]
    
    for mode in modes:
        hist = test_strategy(mode)
        visualize_history(hist, mode)
        
    print("\n✅ 测试完成！")
