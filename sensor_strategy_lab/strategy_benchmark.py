#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器策略基准测试集生成脚本

在多种地图场景和种子上运行不同策略，生成详细的性能对比报告。
用于评估策略的鲁棒性和泛化能力。
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ring_sonar_simulator import RingSonarCore
from src.simulator.trigger import TriggerMode

def run_benchmark_episode(strategy_name: str, seed: int, scene_type: str, steps: int = 4000):
    """运行单个基准测试回合"""
    core = RingSonarCore(
        world_width=20.0,
        world_height=20.0,
        trigger_mode=strategy_name,
        randomize_trigger=False
    )
    
    # 初始化地图
    core.reset(regenerate_map=True, seed=seed, scene_type=scene_type)
    center_pos = core.robot_pos.copy()
    
    # 记录数据
    trajectory = []
    
    # 智能避障移动逻辑 (改进的 Lissajous 曲线)
    speed = 1.5
    step_dist = speed * core.dt
    current_angle_rad = 0.0

    for i in range(steps):
        trajectory.append(core.robot_pos.copy())

        # 1. 生成期望目标点 (改进的 Lissajous 曲线，减少重合)
        t = i * 0.05
        target_x = center_pos[0] + 8.5 * math.sin(0.13 * t)
        target_y = center_pos[1] + 8.5 * math.sin(0.07 * t + i * 0.0005 + math.pi/4)
        
        dx = target_x - core.robot_pos[0]
        dy = target_y - core.robot_pos[1]
        dist = math.hypot(dx, dy)
        desired_angle = math.atan2(dy, dx) if dist > 0 else current_angle_rad

        # 2. 避障移动
        final_pos = None
        angles_to_try = [0]
        for a in range(1, 13): 
            angles_to_try.extend([math.radians(a * 15), math.radians(-a * 15)])
            
        for angle_offset in angles_to_try:
            test_angle = desired_angle + angle_offset
            test_pos = core.robot_pos + np.array([math.cos(test_angle) * step_dist, math.sin(test_angle) * step_dist])
            if core._position_safe(test_pos):
                final_pos = test_pos
                current_angle_rad = test_angle
                break
        
        if final_pos is not None:
            core.robot_pos = final_pos
            core.robot_angle = math.degrees(current_angle_rad) % 360
            
        core.velocity = 0.0
        core.step()
        core.update_maps()

    # 计算最终指标
    final_coverage = np.count_nonzero(core.global_feature_map > 0.1)
    crosstalk_rate = core.crosstalk_count / core.total_sensor_firings if core.total_sensor_firings > 0 else 0
    
    return {
        "strategy": strategy_name,
        "seed": seed,
        "scene_type": scene_type,
        "coverage": int(final_coverage),
        "crosstalk_count": int(core.crosstalk_count),
        "total_firings": int(core.total_sensor_firings),
        "crosstalk_rate": float(crosstalk_rate),
        "trajectory": np.array(trajectory),
        "obstacles": core.obstacles
    }

def save_map_visualization(res: Dict, output_path: str):
    """保存单个回合的地图和轨迹可视化"""
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # 绘制障碍物
    for kind, data in res["obstacles"]:
        if kind == 'rect':
            ox, oy, w, h = data
            rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
            
    # 绘制轨迹
    traj = res["trajectory"]
    plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.6, label='Trajectory')
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End')
    
    plt.title(f"Scene: {res['scene_type']} | Strategy: {res['strategy']} | Seed: {res['seed']}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_benchmark_summary(df: pd.DataFrame, output_dir: str):
    """生成基准测试汇总图表"""
    strategies = df['strategy'].unique()
    
    # 1. 覆盖率对比图 (柱状图 + 误差棒)
    plt.figure(figsize=(12, 6))
    summary = df.groupby('strategy')['coverage'].agg(['mean', 'std'])
    
    bars = plt.bar(summary.index, summary['mean'], yerr=summary['std'], 
                   capsize=10, color='skyblue', edgecolor='navy', alpha=0.8)
    plt.title("Average Map Coverage by Strategy (with Std Dev)", fontsize=14)
    plt.ylabel("Covered Cells", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上标注数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(f"{output_dir}/coverage_summary.png")
    plt.close()

    # 2. 串扰率对比图
    plt.figure(figsize=(12, 6))
    summary_rate = df.groupby('strategy')['crosstalk_rate'].agg(['mean', 'std'])
    # 转换为百分比
    mean_pct = summary_rate['mean'] * 100
    std_pct = summary_rate['std'] * 100
    
    bars = plt.bar(summary_rate.index, mean_pct, yerr=std_pct, 
                   capsize=10, color='salmon', edgecolor='darkred', alpha=0.8)
    plt.title("Average Crosstalk Rate by Strategy (%)", fontsize=14)
    plt.ylabel("Crosstalk Rate (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(f"{output_dir}/crosstalk_summary.png")
    plt.close()

def main():
    # 测试配置
    strategies = ["sequential", "interleaved", "sector", "greedy", "rl"]
    scene_types = ["sparse", "simple", "corridor"] # 移除 rooms 场景
    seeds = [i * 10 for i in range(1, 6)] # 每个场景测试5个种子 (减少总时间，但包含RL)
    steps = 1000
    
    results = []
    output_dir = "sensor_strategy_lab/benchmark"
    maps_dir = f"{output_dir}/test_maps"
    os.makedirs(maps_dir, exist_ok=True)
    
    print(f"🧪 开始生成策略基准测试集...")
    print(f"📊 策略: {strategies}")
    print(f"🌍 场景: {scene_types}")
    print(f"🔢 总回合数: {len(strategies) * len(scene_types) * len(seeds)}")
    
    pbar = tqdm(total=len(strategies) * len(scene_types) * len(seeds))
    
    for scene in scene_types:
        for seed in seeds:
            for strategy in strategies:
                res = run_benchmark_episode(strategy, seed, scene, steps)
                results.append(res)
                
                # 保存地图可视化 (每个场景的每个种子只保存一个代表性策略的地图，或者全部保存)
                # 这里我们为每个回合都保存，但在文件名中区分
                map_filename = f"{maps_dir}/{scene}_s{seed}_{strategy}.png"
                save_map_visualization(res, map_filename)
                
                pbar.update(1)
    
    pbar.close()
    
    # 保存结果
    # 移除 trajectory 和 obstacles 以便保存为 CSV/JSON
    clean_results = []
    for r in results:
        c = r.copy()
        del c["trajectory"]
        del c["obstacles"]
        clean_results.append(c)
        
    df = pd.DataFrame(clean_results)
    
    # 生成汇总图表
    plot_benchmark_summary(df, output_dir)
    
    # 计算汇总统计
    summary = df.groupby('strategy').agg({
        'coverage': ['mean', 'std'],
        'crosstalk_rate': ['mean', 'std'],
        'total_firings': 'mean'
    }).round(4)
    
    df.to_csv(f"{output_dir}/raw_results.csv", index=False)
    summary.to_csv(f"{output_dir}/summary_report.csv")
    
    with open(f"{output_dir}/full_data.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 基准测试完成！")
    print(f"📈 汇总图表已保存至: {output_dir}/")
    print(f"🗺️  测试地图已保存至: {maps_dir}/")
    print("\n--- 性能汇总 ---")
    print(summary)

if __name__ == "__main__":
    main()
