#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring Sonar Environment 交互式演示
展示12个超声波传感器环形阵列的工作原理
"""

import sys
import os
import numpy as np
import cv2
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ring_sonar_simulator import RingSonarCore, RingSonarRenderer


def demo_random_walk(args):
    """演示1: 随机游走"""
    print("\n=== 演示1: 随机游走 ===")
    print("机器人将进行随机运动，展示传感器工作原理")
    print("按 'q' 退出\n")
    
    core = RingSonarCore(
        world_width=args.world_size,
        world_height=args.world_size,
        sensor_ring_radius=0.15,
        num_sensors=12,
        sensor_fov=65.0
    )
    core.reset(regenerate_map=True)
    renderer = RingSonarRenderer(core, render_mode="human")
    
    start_time = time.time()
    
    for step in range(args.steps):
        # 随机改变速度
        if step % 50 == 0:
            core.set_velocity(
                float(np.random.uniform(-2.0, 3.0)),
                float(np.random.uniform(-0.8, 0.8))
            )
        
        core.step()
        core.update_maps()
        renderer.render()
        
        # 显示统计信息
        if step % 20 == 0:
            state = core.state()
            stats = core.fisher_map_stats()
            
            print(f"\rStep {step:4d} | "
                  f"Pos:[{state['position'][0]:5.1f},{state['position'][1]:5.1f}] | "
                  f"Vel:{state['linear_velocity']:5.2f} | "
                  f"Fisher:{stats['total_features']:5.0f} | "
                  f"MinSonar:{np.min(core.sonar_readings):5.2f}m",
                  end='', flush=True)
        
        # 控制帧率
        if args.realtime:
            time.sleep(0.05)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
    
    print(f"\n\n运行时间: {time.time() - start_time:.1f}秒")
    cv2.destroyAllWindows()


def demo_obstacle_avoidance(args):
    """演示2: 简单避障"""
    print("\n=== 演示2: 简单避障行为 ===")
    print("机器人将根据传感器读数尝试避开障碍物")
    print("按 'q' 退出\n")
    
    core = RingSonarCore(
        world_width=args.world_size,
        world_height=args.world_size
    )
    core.reset(regenerate_map=True)
    renderer = RingSonarRenderer(core, render_mode="human")
    
    for step in range(args.steps):
        # 简单的避障策略
        sonar = core.sonar_readings
        min_distance = np.min(sonar)
        min_idx = np.argmin(sonar)
        
        if min_distance < 2.0:
            # 太近，转向远离
            # 计算远离方向
            danger_angle = core.sensors[min_idx].angle
            
            # 转向相反方向
            if danger_angle < 180:
                angular_vel = 0.8  # 右转
            else:
                angular_vel = -0.8  # 左转
            
            core.set_velocity(1.0, angular_vel)
        else:
            # 前进
            core.set_velocity(2.5, 0.0)
        
        core.step()
        core.update_maps()
        renderer.render()
        
        if step % 20 == 0:
            state = core.state()
            print(f"\rStep {step:4d} | "
                  f"MinDist:{min_distance:5.2f}m | "
                  f"DangerSensor:{min_idx:2d} | "
                  f"Action:{'AVOID' if min_distance < 2.0 else 'FORWARD'}",
                  end='', flush=True)
        
        if args.realtime:
            time.sleep(0.05)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
    
    print()
    cv2.destroyAllWindows()


def demo_sensor_visualization(args):
    """演示3: 传感器可视化"""
    print("\n=== 演示3: 传感器详细可视化 ===")
    print("展示每个传感器的FoV和读数")
    print("按 'q' 退出\n")
    
    core = RingSonarCore(world_width=args.world_size, world_height=args.world_size)
    core.reset(regenerate_map=True)
    renderer = RingSonarRenderer(core, render_mode="human")
    
    # 机器人缓慢旋转以展示所有传感器
    for step in range(args.steps):
        # 缓慢旋转
        core.set_velocity(0.5, 0.2)
        
        core.step()
        core.update_maps()
        renderer.render()
        
        # 打印详细传感器信息
        if step % 30 == 0:
            print(f"\n--- Step {step} ---")
            state = core.state()
            print(f"机器人角度: {state['angle']:.1f}°")
            print("传感器读数:")
            for i, (sensor, distance) in enumerate(zip(core.sensors, core.sonar_readings)):
                world_angle = sensor.get_world_angle(core.robot_angle)
                print(f"  S{i:2d} [{world_angle:6.1f}°]: {distance:6.2f}m", end='')
                if i % 3 == 2:
                    print()
            print()
        
        if args.realtime:
            time.sleep(0.05)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
    
    cv2.destroyAllWindows()


def demo_fisher_exploration(args):
    """演示4: Fisher信息探索"""
    print("\n=== 演示4: Fisher信息探索 ===")
    print("机器人运动并累积Fisher信息")
    print("按 'q' 退出\n")
    
    core = RingSonarCore(world_width=args.world_size, world_height=args.world_size)
    core.reset(regenerate_map=True)
    renderer = RingSonarRenderer(core, render_mode="human")
    
    fisher_log = []
    
    for step in range(args.steps):
        # 探索策略：向Fisher值低的区域移动
        if step % 40 == 0:
            # 随机改变方向
            core.set_velocity(
                float(np.random.uniform(1.0, 3.0)),
                float(np.random.uniform(-0.5, 0.5))
            )
        
        core.step()
        core.update_maps()
        renderer.render()
        
        # 记录Fisher统计
        stats = core.fisher_map_stats()
        fisher_log.append(stats['total_features'])
        
        if step % 25 == 0:
            state = core.state()
            print(f"\rStep {step:4d} | "
                  f"Fisher:{stats['total_features']:6.0f} | "
                  f"Mean:{stats['mean_fisher']:6.3f} | "
                  f"Density:{stats['density']:6.3f} | "
                  f"Pos:[{state['position'][0]:5.1f},{state['position'][1]:5.1f}]",
                  end='', flush=True)
        
        if args.realtime:
            time.sleep(0.05)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
    
    print(f"\n\nFisher信息统计:")
    print(f"  初始: {fisher_log[5]:.0f}")
    print(f"  最终: {fisher_log[-1]:.0f}")
    print(f"  增长: {fisher_log[-1] - fisher_log[5]:.0f}")
    print(f"  增长率: {(fisher_log[-1] / max(fisher_log[5], 1) - 1) * 100:.1f}%")
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Ring Sonar Environment 交互式演示')
    parser.add_argument('--demo', type=int, default=1, choices=[1, 2, 3, 4],
                       help='演示模式: 1=随机游走, 2=避障, 3=传感器可视化, 4=Fisher探索')
    parser.add_argument('--steps', type=int, default=500, help='演示步数')
    parser.add_argument('--world-size', type=float, default=40.0, help='世界大小(米)')
    parser.add_argument('--realtime', action='store_true', help='实时速度运行')
    args = parser.parse_args()
    
    print("=" * 70)
    print("环形超声波雷达系统 - 交互式演示")
    print("=" * 70)
    print(f"\n系统配置:")
    print(f"  - 传感器数量: 12个")
    print(f"  - 传感器布局: 半径15cm圆环")
    print(f"  - 单传感器FoV: 65°")
    print(f"  - 最大探测距离: 12.5m")
    print(f"  - 世界大小: {args.world_size}m × {args.world_size}m")
    
    demos = {
        1: demo_random_walk,
        2: demo_obstacle_avoidance,
        3: demo_sensor_visualization,
        4: demo_fisher_exploration
    }
    
    demo_func = demos[args.demo]
    demo_func(args)
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
