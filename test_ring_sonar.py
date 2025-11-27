#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring Sonar Environment 测试脚本
验证12个超声波传感器环形阵列的功能
"""

import sys
import os
import numpy as np
import cv2
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ring_sonar_simulator import RingSonarCore, RingSonarRenderer


def test_sensor_layout():
    """测试1: 验证传感器布局"""
    print("\n=== 测试1: 传感器布局验证 ===")
    
    core = RingSonarCore(
        world_width=40.0,
        world_height=40.0,
        sensor_ring_radius=0.15,
        num_sensors=12
    )
    core.reset()
    
    print(f"传感器数量: {len(core.sensors)}")
    print(f"传感器环半径: {core.sensor_ring_radius}m")
    
    print("\n传感器布局 (相对机器人中心):")
    for sensor in core.sensors:
        print(f"  Sensor {sensor.id:2d}: "
              f"angle={sensor.angle:6.1f}°, "
              f"offset=({sensor.offset_x:6.3f}, {sensor.offset_y:6.3f})m, "
              f"fov={sensor.fov_angle}°")
    
    # 验证均匀分布
    angles = [s.angle for s in core.sensors]
    angle_diffs = np.diff(sorted(angles))
    expected_diff = 360.0 / core.num_sensors
    
    print(f"\n角度间隔统计:")
    print(f"  期望间隔: {expected_diff:.1f}°")
    print(f"  实际间隔: {angle_diffs}")
    print(f"  均匀性验证: {'通过' if np.allclose(angle_diffs, expected_diff) else '失败'}")
    
    return core


def test_sensor_scanning(core):
    """测试2: 验证传感器扫描功能"""
    print("\n=== 测试2: 传感器扫描功能 ===")
    
    # 执行几步仿真
    for _ in range(10):
        core.step()
    
    print(f"传感器读数 (距离, 米):")
    for i, distance in enumerate(core.sonar_readings):
        print(f"  Sensor {i:2d}: {distance:6.2f}m")
    
    # 统计
    print(f"\n统计信息:")
    print(f"  最小距离: {np.min(core.sonar_readings):.2f}m")
    print(f"  最大距离: {np.max(core.sonar_readings):.2f}m")
    print(f"  平均距离: {np.mean(core.sonar_readings):.2f}m")
    print(f"  标准差: {np.std(core.sonar_readings):.2f}m")
    
    # 检查是否有传感器检测到障碍物
    detected = np.sum(core.sonar_readings < core.sensor_max_range)
    print(f"  检测到障碍物的传感器数: {detected}/{core.num_sensors}")


def test_fisher_accumulation(core):
    """测试3: Fisher信息累积"""
    print("\n=== 测试3: Fisher信息累积 ===")
    
    # 重置并运行一段时间
    core.reset()
    core.set_velocity(2.0, 0.3)  # 设置运动
    
    fisher_history = []
    
    for step in range(50):
        core.step()
        core.update_maps()
        
        stats = core.fisher_map_stats()
        fisher_history.append(stats['total_features'])
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: "
                  f"features={stats['total_features']:6.0f}, "
                  f"mean={stats['mean_fisher']:6.3f}, "
                  f"density={stats['density']:6.3f}")
    
    # 验证Fisher信息是否累积
    initial_features = fisher_history[5]
    final_features = fisher_history[-1]
    
    print(f"\nFisher累积验证:")
    print(f"  初始特征数: {initial_features:.0f}")
    print(f"  最终特征数: {final_features:.0f}")
    print(f"  增长: {final_features - initial_features:.0f}")
    print(f"  累积测试: {'通过' if final_features > initial_features else '失败'}")


def test_visualization():
    """测试4: 可视化测试"""
    print("\n=== 测试4: 可视化测试 ===")
    print("运行可视化测试...")
    print("按 'q' 或 ESC 键退出")
    
    core = RingSonarCore(world_width=40.0, world_height=40.0)
    core.reset()
    renderer = RingSonarRenderer(core, render_mode="human")
    
    # 简单的随机移动
    for step in range(500):
        if step % 30 == 0:
            core.set_velocity(
                float(np.random.uniform(-2.0, 3.0)),
                float(np.random.uniform(-0.6, 0.6))
            )
        
        core.step()
        core.update_maps()
        renderer.render()
        
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q') or k == 27:
            break
    
    cv2.destroyAllWindows()
    print("可视化测试完成")


def test_coverage_analysis(core):
    """测试5: 覆盖范围分析"""
    print("\n=== 测试5: 传感器覆盖范围分析 ===")
    
    # 计算每个传感器的覆盖角度
    total_coverage = 0.0
    overlaps = []
    
    for i, sensor in enumerate(core.sensors):
        sensor_angle = sensor.angle
        half_fov = sensor.fov_angle / 2.0
        
        start_angle = (sensor_angle - half_fov) % 360.0
        end_angle = (sensor_angle + half_fov) % 360.0
        
        total_coverage += sensor.fov_angle
        
        # 检查与下一个传感器的重叠
        if i < len(core.sensors) - 1:
            next_sensor = core.sensors[i + 1]
            next_start = (next_sensor.angle - next_sensor.fov_angle / 2.0) % 360.0
            
            if end_angle > next_start or (end_angle < start_angle and next_start < start_angle):
                overlap = end_angle - next_start if end_angle > next_start else end_angle + 360.0 - next_start
                overlaps.append(overlap)
    
    print(f"总覆盖角度: {total_coverage:.1f}° (理论360°)")
    print(f"平均单传感器FoV: {total_coverage / core.num_sensors:.1f}°")
    
    if overlaps:
        print(f"相邻传感器重叠: 平均 {np.mean(overlaps):.1f}°")
    
    # 理论覆盖分析
    sensor_fov = core.sensor_fov
    num_sensors = core.num_sensors
    full_coverage = sensor_fov * num_sensors
    
    print(f"\n理论分析:")
    print(f"  {num_sensors}个传感器 × {sensor_fov}° FoV = {full_coverage}° 总覆盖")
    
    if full_coverage >= 360.0:
        overlap_total = full_coverage - 360.0
        print(f"  总重叠: {overlap_total:.1f}° (有重叠覆盖)")
    else:
        gap_total = 360.0 - full_coverage
        print(f"  总间隙: {gap_total:.1f}° (有覆盖盲区)")


def main():
    """主测试流程"""
    print("=" * 60)
    print("环形超声波雷达系统测试")
    print("=" * 60)
    
    # 测试1: 传感器布局
    core = test_sensor_layout()
    
    # 测试2: 传感器扫描
    test_sensor_scanning(core)
    
    # 测试3: Fisher累积
    test_fisher_accumulation(core)
    
    # 测试5: 覆盖范围分析
    test_coverage_analysis(core)
    
    # 测试4: 可视化 (可选)
    print("\n是否运行可视化测试? (y/n): ", end='')
    try:
        choice = input().strip().lower()
        if choice == 'y':
            test_visualization()
    except:
        print("跳过可视化测试")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
