#!/usr/bin/env python3
"""
测试不同传感器触发模式的性能
"""
import numpy as np
from ring_sonar_simulator import RingSonarCore
import time

def test_alternating_mode():
    """测试交替扫描模式（基线）"""
    print("\n" + "="*60)
    print("测试1: 交替扫描模式（奇偶两组）")
    print("="*60)
    
    # 创建环境
    core = RingSonarCore(
        sensor_max_range=12.5,
        sensor_fov=65,
        num_sensors=12,
        sensor_ring_radius=0.15,
        robot_size=0.5
    )
    
    # 验证默认模式
    state = core.state()
    print(f"\n初始状态:")
    print(f"  触发模式: {state['trigger_mode']}")
    print(f"  当前激活组: {state['active_group']}")
    print(f"  激活传感器: {core.get_active_sensors()}")
    
    # 执行5个步骤，观察组切换
    print(f"\n执行5个步骤，观察传感器激活模式:")
    core.set_velocity(0.2, 0)  # 直线前进
    for step in range(5):
        active_sensors = core.get_active_sensors()
        core.step()
        state = core.state()
        
        print(f"\n步骤 {step}:")
        print(f"  激活组: {state['active_group']}")
        print(f"  激活传感器: {active_sensors}")
        print(f"  传感器读数 (前2个):")
        for sensor_id in active_sensors[:2]:
            print(f"    传感器{sensor_id}: {core.sonar_readings[sensor_id]:.2f}m")

def test_simultaneous_mode():
    """测试同时扫描模式（对比）"""
    print("\n" + "="*60)
    print("测试2: 同时扫描模式（所有传感器）")
    print("="*60)
    
    core = RingSonarCore(
        sensor_max_range=12.5,
        sensor_fov=65,
        num_sensors=12,
        sensor_ring_radius=0.15,
        robot_size=0.5
    )
    
    # 切换到同时模式
    core.set_trigger_mode("simultaneous")
    
    # 执行3个步骤
    print(f"\n执行3个步骤:")
    core.set_velocity(0.2, 0)
    for step in range(3):
        active_sensors = core.get_active_sensors()
        core.step()
        
        print(f"\n步骤 {step}:")
        print(f"  激活传感器数量: {len(active_sensors)}")
        print(f"  传感器读数 (前4个): {core.sonar_readings[:4]}")

def test_custom_groups():
    """测试自定义分组（3组）"""
    print("\n" + "="*60)
    print("测试3: 自定义分组（3组，120度间隔）")
    print("="*60)
    
    core = RingSonarCore(
        sensor_max_range=12.5,
        sensor_fov=65,
        num_sensors=12,
        sensor_ring_radius=0.15,
        robot_size=0.5
    )
    
    # 定义3组：每组4个传感器，120度间隔
    custom_groups = [
        [0, 3, 6, 9],    # 0°, 90°, 180°, 270°
        [1, 4, 7, 10],   # 30°, 120°, 210°, 300°
        [2, 5, 8, 11]    # 60°, 150°, 240°, 330°
    ]
    
    core.set_trigger_mode("alternating", groups=custom_groups)
    
    # 执行6个步骤（2个完整周期）
    print(f"\n执行6个步骤（2个完整周期）:")
    core.set_velocity(0.2, 0)
    for step in range(6):
        active_sensors = core.get_active_sensors()
        core.step()
        state = core.state()
        
        print(f"\n步骤 {step}:")
        print(f"  激活组: {state['active_group']}")
        print(f"  激活传感器: {active_sensors}")

def test_performance_comparison():
    """性能对比：交替 vs 同时"""
    print("\n" + "="*60)
    print("测试4: 性能对比")
    print("="*60)
    
    num_steps = 200
    
    # 测试交替模式
    core_alt = RingSonarCore(sensor_max_range=12.5, sensor_fov=65, num_sensors=12)
    core_alt.set_velocity(0.2, 0.1)
    start = time.time()
    for _ in range(num_steps):
        core_alt.step()
    time_alt = time.time() - start
    
    # 测试同时模式
    core_sim = RingSonarCore(sensor_max_range=12.5, sensor_fov=65, num_sensors=12)
    core_sim.set_trigger_mode("simultaneous")
    core_sim.set_velocity(0.2, 0.1)
    start = time.time()
    for _ in range(num_steps):
        core_sim.step()
    time_sim = time.time() - start
    
    print(f"\n{num_steps}步性能对比:")
    print(f"  交替模式: {time_alt:.3f}秒 ({num_steps/time_alt:.1f} steps/s)")
    print(f"  同时模式: {time_sim:.3f}秒 ({num_steps/time_sim:.1f} steps/s)")
    print(f"  性能比: {time_alt/time_sim:.2f}x")

if __name__ == "__main__":
    test_alternating_mode()
    test_simultaneous_mode()
    test_custom_groups()
    test_performance_comparison()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
