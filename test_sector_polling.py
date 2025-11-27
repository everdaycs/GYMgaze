#!/usr/bin/env python3
"""
测试扇区轮询触发模式
验证传感器按照扇区顺序触发
"""

import numpy as np
from ring_sonar_simulator import RingSonarCore

def test_sector_polling():
    """测试扇区轮询模式"""
    print("=" * 60)
    print("测试扇区轮询触发模式")
    print("=" * 60)
    
    # 创建模拟器（使用扇区轮询模式）
    sim = RingSonarCore(
        world_width=50.0,
        world_height=50.0,
        pixel_per_meter=10,
        robot_size=0.5,
        sensor_ring_radius=0.15,
        num_sensors=12,
        sensor_fov=65.0,
        sensor_max_range=12.5,
        feature_map_size=100,
        feature_map_resolution=0.25,
        control_frequency=10.0,
        trigger_mode="sector"  # 使用扇区轮询模式
    )
    
    # 重置环境（生成障碍物）
    sim.reset(regenerate_map=True)
    
    print("\n扇区配置:")
    for sector_name, sensor_ids in sim.sectors.items():
        print(f"  {sector_name}: 传感器 {sensor_ids}")
    
    print(f"\n轮询顺序: {sim.sector_sequence}")
    print(f"初始扇区索引: {sim.current_sector_index}")
    
    print("\n" + "=" * 60)
    print("执行10个步骤，观察扇区轮询")
    print("=" * 60)
    
    # 记录每个步骤的传感器读数变化
    prev_readings = sim.sonar_readings.copy()
    
    for step in range(10):
        # 记录当前扇区
        current_sector_name = sim.sector_sequence[sim.current_sector_index]
        active_sensors = sim.sectors[current_sector_name]
        
        print(f"\nStep {step + 1}:")
        print(f"  当前扇区: {current_sector_name}")
        print(f"  活跃传感器: {active_sensors}")
        
        # 执行一步
        sim.set_velocity(1.0, 0.1)
        sim.step()
        
        # 检查哪些传感器的读数发生了变化
        changed_sensors = []
        for i in range(12):
            if abs(sim.sonar_readings[i] - prev_readings[i]) > 0.01:
                changed_sensors.append(i)
        
        print(f"  读数变化的传感器: {changed_sensors}")
        
        # 验证：变化的传感器应该是当前扇区的传感器
        if set(changed_sensors) == set(active_sensors):
            print(f"  ✓ 验证通过：只有扇区 {current_sector_name} 的传感器被更新")
        elif set(changed_sensors).issubset(set(active_sensors)):
            print(f"  ✓ 部分验证：变化的传感器都在扇区 {current_sector_name} 内")
        else:
            unexpected = set(changed_sensors) - set(active_sensors)
            if unexpected:
                print(f"  ✗ 验证失败：传感器 {unexpected} 不在当前扇区内")
        
        # 显示当前扇区传感器的读数
        print(f"  扇区传感器读数:")
        for sensor_id in active_sensors:
            print(f"    传感器 {sensor_id:2d}: {sim.sonar_readings[sensor_id]:6.2f}m")
        
        # 更新上一次的读数
        prev_readings = sim.sonar_readings.copy()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 统计传感器读数范围
    print("\n传感器读数统计:")
    for i in range(12):
        print(f"  传感器 {i:2d}: {sim.sonar_readings[i]:6.2f}m")
    
    print(f"\n最终机器人位置: {sim.robot_pos}")
    print(f"最终机器人角度: {sim.robot_angle:.1f}°")

def test_all_sensors_mode():
    """测试同时触发所有传感器模式（对比）"""
    print("\n\n" + "=" * 60)
    print("对比测试：同时触发所有传感器模式")
    print("=" * 60)
    
    # 创建模拟器（使用默认模式 - 所有传感器同时触发）
    sim = RingSonarCore(
        world_width=50.0,
        world_height=50.0,
        pixel_per_meter=10,
        robot_size=0.5,
        sensor_ring_radius=0.15,
        num_sensors=12,
        sensor_fov=65.0,
        sensor_max_range=12.5,
        feature_map_size=100,
        feature_map_resolution=0.25,
        control_frequency=10.0,
        trigger_mode="all"  # 同时触发所有传感器
    )
    
    # 重置环境（生成障碍物）
    sim.reset(regenerate_map=True)
    
    print("\n执行3个步骤，观察所有传感器同时更新")
    
    # 记录每个步骤的传感器读数变化
    prev_readings = sim.sonar_readings.copy()
    
    for step in range(3):
        print(f"\nStep {step + 1}:")
        
        # 执行一步
        sim.set_velocity(1.0, 0.1)
        sim.step()
        
        # 检查哪些传感器的读数发生了变化
        changed_sensors = []
        for i in range(12):
            if abs(sim.sonar_readings[i] - prev_readings[i]) > 0.01:
                changed_sensors.append(i)
        
        print(f"  读数变化的传感器: {changed_sensors}")
        
        if len(changed_sensors) == 12:
            print(f"  ✓ 验证通过：所有12个传感器都被更新")
        else:
            print(f"  部分传感器更新: {len(changed_sensors)}/12")
        
        # 更新上一次的读数
        prev_readings = sim.sonar_readings.copy()
    
    print("\n" + "=" * 60)
    print("对比测试完成")
    print("=" * 60)

if __name__ == "__main__":
    # 测试扇区轮询模式
    test_sector_polling()
    
    # 测试同时触发所有传感器模式
    test_all_sensors_mode()
