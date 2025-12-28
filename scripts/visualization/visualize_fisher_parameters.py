#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisher 信息参数可视化工具

展示不同参数设置对 Fisher 值的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_fisher_parameters():
    """可视化不同Fisher参数的效果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fisher 信息参数影响分析', fontsize=16, fontweight='bold')
    
    # 参数范围
    distances = np.linspace(0.1, 12.5, 100)
    max_range = 12.5
    
    # ========== 图 1: 距离衰减指数的影响 ==========
    ax = axes[0, 0]
    decay_powers = [0.5, 1.0, 1.5, 2.0, 2.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(decay_powers)))
    
    for power, color in zip(decay_powers, colors):
        normalized_dist = distances / max_range
        dist_factors = (1.0 - normalized_dist) ** power
        dist_factors = np.maximum(dist_factors, 0.1)
        ax.plot(distances, dist_factors, 'o-', label=f'power={power}', 
                color=color, linewidth=2, markersize=4)
    
    ax.set_xlabel('距离 (m)', fontsize=11)
    ax.set_ylabel('距离因子', fontsize=11)
    ax.set_title('DISTANCE_DECAY_POWER 的影响\n(指数越大，距离衰减越快)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ========== 图 2: 覆盖度因子的影响 ==========
    ax = axes[0, 1]
    coverage_counts = np.array([1, 2, 3, 4, 5])
    max_overlap = 5
    bonus_values = [1.5, 2.0, 3.0, 4.0, 5.0]
    
    x_pos = np.arange(len(coverage_counts))
    width = 0.15
    
    for i, bonus in enumerate(bonus_values):
        coverage_factors = 1.0 + (coverage_counts - 1) * (bonus - 1.0) / (max_overlap - 1)
        ax.bar(x_pos + i*width, coverage_factors, width, label=f'bonus={bonus}')
    
    ax.set_xlabel('传感器覆盖数量', fontsize=11)
    ax.set_ylabel('覆盖度因子', fontsize=11)
    ax.set_title('MAX_COVERAGE_BONUS 的影响\n(加成越高，重叠区域价值越大)', fontsize=11)
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels(coverage_counts)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    
    # ========== 图 3: 综合 Fisher 值（不同参数组合）==========
    ax = axes[1, 0]
    
    # 计算不同参数组合的 Fisher 值
    distance_powers = [1.0, 1.5, 2.0]
    coverage_bonuses = [1.5, 3.0, 5.0]
    
    for power in distance_powers:
        fisher_values = []
        for dist in distances:
            normalized_dist = dist / max_range
            dist_factor = (1.0 - normalized_dist) ** power
            dist_factor = max(dist_factor, 0.1)
            
            # 假设单个传感器覆盖
            coverage_factor = 1.0
            
            fisher = dist_factor * coverage_factor
            fisher = np.clip(fisher, 0.1, 10.0)
            fisher_values.append(fisher)
        
        ax.plot(distances, fisher_values, 'o-', label=f'power={power}', 
                linewidth=2, markersize=4)
    
    ax.set_xlabel('距离 (m)', fontsize=11)
    ax.set_ylabel('Fisher 值', fontsize=11)
    ax.set_title('综合 Fisher 值随距离变化\n(单传感器覆盖)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ========== 图 4: 衰减速率的影响 ==========
    ax = axes[1, 1]
    
    time_steps = np.arange(0, 100000, 1000)
    initial_value = 10.0
    decay_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    colors = plt.cm.plasma(np.linspace(0, 1, len(decay_rates)))
    
    for rate, color in zip(decay_rates, colors):
        values = initial_value * (1.0 - rate) ** time_steps
        ax.semilogy(time_steps, values, 'o-', label=f'decay={rate}', 
                   color=color, linewidth=2, markersize=4)
    
    ax.set_xlabel('时间步数', fontsize=11)
    ax.set_ylabel('特征值 (对数)', fontsize=11)
    ax.set_title('特征衰减速率的影响\n(衰减系数越大，遗忘越快)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/home/kaga/GYMgaze/docs/fisher_parameters_visualization.png', dpi=150)
    print("✅ 可视化已保存到: docs/fisher_parameters_visualization.png")
    plt.show()


def print_parameter_table():
    """打印参数参考表"""
    
    print("\n" + "="*80)
    print("Fisher 信息参数参考表")
    print("="*80 + "\n")
    
    # 距离衰减指数的影响
    print("1. DISTANCE_DECAY_POWER（距离衰减指数）")
    print("-" * 80)
    print(f"{'距离(m)':<12} {'power=0.5':<15} {'power=1.0':<15} {'power=1.5':<15} {'power=2.0':<15}")
    print("-" * 80)
    
    max_range = 12.5
    for dist in [3.1, 6.2, 9.4, 12.5]:
        normalized = dist / max_range
        values = []
        for power in [0.5, 1.0, 1.5, 2.0]:
            val = max((1.0 - normalized) ** power, 0.1)
            values.append(f"{val:.4f}")
        print(f"{dist:<12.1f} {values[0]:<15} {values[1]:<15} {values[2]:<15} {values[3]:<15}")
    
    print("\n📌 建议：")
    print("  • power=1.0: 线性衰减，对所有距离均衡")
    print("  • power=1.5: 默认值，接近真实传感器特性")
    print("  • power=2.0: 二次衰减，强调近距离精度")
    
    # 覆盖度因子的影响
    print("\n\n2. MAX_COVERAGE_BONUS（最大覆盖度加成）")
    print("-" * 80)
    print(f"{'覆盖数量':<12} {'bonus=1.5':<15} {'bonus=3.0':<15} {'bonus=5.0':<15}")
    print("-" * 80)
    
    max_overlap = 5
    for count in [1, 2, 3, 4, 5]:
        values = []
        for bonus in [1.5, 3.0, 5.0]:
            if count == 1:
                val = 1.0
            else:
                val = 1.0 + (bonus - 1.0) * (count - 1) / (max_overlap - 1)
            values.append(f"{val:.4f}")
        print(f"{count:<12} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
    
    print("\n📌 建议：")
    print("  • bonus=1.5: 轻微重视重叠区域")
    print("  • bonus=3.0: 默认值，中等重视重叠")
    print("  • bonus=5.0: 强烈重视重叠区域")
    
    # 衰减系数的影响
    print("\n\n3. 特征衰减系数（decay_rate）")
    print("-" * 80)
    print(f"{'步数':<12} {'rate=1e-6':<15} {'rate=5e-6':<15} {'rate=1e-5':<15} {'rate=5e-5':<15}")
    print("-" * 80)
    
    initial = 10.0
    for steps in [0, 10000, 50000, 100000]:
        values = []
        for rate in [1e-6, 5e-6, 1e-5, 5e-5]:
            val = initial * (1.0 - rate) ** steps
            values.append(f"{val:.4f}")
        print(f"{steps:<12} {values[0]:<15} {values[1]:<15} {values[2]:<15} {values[3]:<15}")
    
    print("\n📌 建议：")
    print("  • 1e-6: 长期记忆，适合静态环境")
    print("  • 5e-6: 默认值，平衡记忆与遗忘")
    print("  • 1e-5: 中期记忆，适合动态环境")
    print("  • 5e-5: 短期记忆，强调当前观测")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # 打印参数表
    print_parameter_table()
    
    # 可视化参数效果（需要matplotlib）
    try:
        visualize_fisher_parameters()
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过可视化")
        print("   运行: pip install matplotlib")
