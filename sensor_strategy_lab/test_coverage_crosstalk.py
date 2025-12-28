#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器策略覆盖率与串扰率测试

使用模拟器在固定地图和轨迹上测试不同触发策略的性能。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulator.ring_sonar_simulator import RingSonarCore
from src.simulator.trigger import TriggerMode

def run_simulation(strategy_name, steps=2000, seed=2):
    """运行单次仿真"""
    print(f"开始测试策略: {strategy_name}")
    
    # 初始化模拟器
    core = RingSonarCore(
        world_width=40.0,
        world_height=40.0,
        pixel_per_meter=10,
        trigger_mode=strategy_name,
        randomize_trigger=False
    )
    
    # 固定地图种子
    core.reset(regenerate_map=True, seed=seed, scene_type="corridor")
    
    print(f"  ⚙️ dt={core.dt}, max_range={core.sensor_max_range}, speed_of_sound={core.speed_of_sound}")
    
    # 使用 reset 后的安全位置作为轨迹中心
    center_pos = core.robot_pos.copy()
    
    # 记录数据
    coverage_history = []
    crosstalk_history = []
    trajectory = []
    
    # 运行仿真
    # 预计算移动步长
    speed = 1.2
    step_dist = speed * core.dt
    current_angle_rad = 0.0

    for i in range(steps):
        # 记录轨迹
        trajectory.append(core.robot_pos.copy())

        # 1. 生成期望目标点 (改进的 Lissajous 曲线，适应 40x40 地图)
        t = i * 0.05
        target_x = center_pos[0] + 15.0 * math.sin(0.11 * t)
        target_y = center_pos[1] + 15.0 * math.sin(0.05 * t + i * 0.0003 + math.pi/4)
        
        # 2. 计算期望移动向量
        dx = target_x - core.robot_pos[0]
        dy = target_y - core.robot_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            desired_angle = math.atan2(dy, dx)
        else:
            desired_angle = current_angle_rad

        # 3. 智能避障移动 (尝试不同角度直到找到无碰撞路径)
        final_pos = None
        
        # 尝试的角度偏移: 0, ±15, ±30, ... ±180
        angles_to_try = [0]
        for a in range(1, 13): 
            angles_to_try.append(math.radians(a * 15))
            angles_to_try.append(math.radians(-a * 15))
            
        for angle_offset in angles_to_try:
            # 旋转移动向量
            test_angle = desired_angle + angle_offset
            test_dx = math.cos(test_angle) * step_dist
            test_dy = math.sin(test_angle) * step_dist
            
            test_pos = core.robot_pos + np.array([test_dx, test_dy])
            
            # 检查是否安全 (使用私有方法 _position_safe)
            if core._position_safe(test_pos):
                final_pos = test_pos
                current_angle_rad = test_angle
                break
        
        # 4. 强制更新位置 (绕过物理引擎)
        if final_pos is not None:
            core.robot_pos = final_pos
            core.robot_angle = math.degrees(current_angle_rad) % 360
            
        # 5. 禁用物理引擎移动，仅用于更新时间和触发传感器
        core.velocity = 0.0
        core.angular_velocity = 0.0
        
        # 执行一步 (更新时间, 触发传感器)
        core.step()
        core.update_maps()
        
        # 记录指标
        # 覆盖率: 全局地图中非零元素的数量 (简单近似)
        coverage = np.count_nonzero(core.global_feature_map > 0.1)
        coverage_history.append(coverage)
        
        # 串扰率: 累计串扰次数 / 累计发射次数
        if core.total_sensor_firings > 0:
            rate = core.crosstalk_count / core.total_sensor_firings
        else:
            rate = 0.0
        crosstalk_history.append(rate)
        
    print(f"  ✅ 完成。最终覆盖: {coverage_history[-1]}, 总串扰: {core.crosstalk_count}, 串扰率: {crosstalk_history[-1]:.4f}")
    
    return {
        "coverage": coverage_history,
        "crosstalk_rate": crosstalk_history,
        "final_crosstalk_count": core.crosstalk_count,
        "total_firings": core.total_sensor_firings,
        "trajectory": np.array(trajectory),
        "obstacles": core.obstacles
    }

def plot_results(results):
    """绘制对比图表"""
    strategies = list(results.keys())
    steps = len(results[strategies[0]]["coverage"])
    x = range(steps)
    
    # 1. 覆盖率对比
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(x, data["coverage"], label=name)
    plt.title("Map Coverage Growth by Strategy")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Map Cells Covered (>0.1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sensor_strategy_lab/coverage_comparison.png")
    print("覆盖率图表已保存")
    
    # 2. 串扰率对比
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(x, data["crosstalk_rate"], label=name)
    plt.title("Crosstalk Rate by Strategy")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Crosstalk Rate (Cumulative)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sensor_strategy_lab/crosstalk_comparison.png")
    print("串扰率图表已保存")
    
    # 3. 轨迹与地图可视化
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 绘制障碍物 (使用第一个策略的数据，因为地图是一样的)
    obstacles = results[strategies[0]]["obstacles"]
    for kind, data in obstacles:
        if kind == 'rect':
            ox, oy, w, h = data
            rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)
            
    # 绘制轨迹
    # 绘制所有策略的轨迹，看看是否有区别
    colors = ['b', 'g', 'r', 'c']
    for idx, (name, data) in enumerate(results.items()):
        traj = data["trajectory"]
        plt.plot(traj[:, 0], traj[:, 1], color=colors[idx % len(colors)], 
                 linestyle='-', linewidth=1, alpha=0.6, label=f'Trajectory ({name})')
    
    plt.plot(results[strategies[0]]["trajectory"][0, 0], 
             results[strategies[0]]["trajectory"][0, 1], 'go', label='Start')
    
    plt.title("Test Environment & Trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("sensor_strategy_lab/trajectory_map.png")
    print("轨迹地图已保存")

    # 4. 综合性能柱状图
    plt.figure(figsize=(10, 6))
    final_coverages = [results[s]["coverage"][-1] for s in strategies]
    final_rates = [results[s]["crosstalk_rate"][-1] * 100 for s in strategies] # 百分比
    
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    rects1 = ax1.bar(x_pos - width/2, final_coverages, width, label='Coverage', color='skyblue')
    ax1.set_ylabel('Final Coverage (Cells)')
    ax1.set_title('Performance Comparison: Coverage vs Crosstalk')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(strategies)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x_pos + width/2, final_rates, width, label='Crosstalk Rate (%)', color='salmon')
    ax2.set_ylabel('Crosstalk Rate (%)')
    ax2.legend(loc='upper right')
    
    plt.savefig("sensor_strategy_lab/performance_summary.png")
    print("综合性能图表已保存")

if __name__ == "__main__":
    # 测试所有可用策略
    strategies = ["sequential", "interleaved", "sector", "greedy", "rl"]
    results = {}
    
    # 确保目录存在
    os.makedirs("sensor_strategy_lab", exist_ok=True)
    
    for s in strategies:
        try:
            results[s] = run_simulation(s, steps=2000)
        except Exception as e:
            print(f"❌ 策略 {s} 测试失败: {e}")
        
    if results:
        plot_results(results)
        print("\n🎉 所有测试完成！结果已保存至 sensor_strategy_lab/ 目录下。")
    else:
        print("❌ 没有成功的测试结果。")
