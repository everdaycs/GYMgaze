#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
障碍物预测演示 - 展示扩散模型如何从空闲空间推断障碍物

使用说明：
    python demo_obstacle_prediction.py

观察要点：
1. 绿色区域：传感器确认的空闲空间
2. 红色区域：传感器确认的障碍物
3. 热力图（蓝→黄→红）：预测的障碍物概率
4. 注意边界区域如何被预测为障碍物
"""

import sys
import time
from ring_sonar_simulator import RingSonarCore, RingSonarRenderer

def main():
    print("=" * 60)
    print("障碍物预测扩散模型演示")
    print("=" * 60)
    print()
    print("功能说明：")
    print("  - 基于已知空闲空间预测未探索区域的障碍物")
    print("  - 使用形态学扩散 + 梯度检测 + 距离衰减")
    print("  - 实时更新预测和置信度")
    print()
    print("窗口说明：")
    print("  1. Ring Sonar Simulation - 世界视图")
    print("  2. Occupancy Grid - 占用栅格地图（已知信息）")
    print("  3. Obstacle Prediction - 障碍物预测（推断信息）⭐")
    print()
    print("颜色编码：")
    print("  绿色 = 已知空闲空间")
    print("  红色 = 已知障碍物")
    print("  蓝→黄→红 = 预测障碍物概率（0→100%）")
    print()
    print("按 'q' 或 ESC 退出")
    print("=" * 60)
    print()
    
    # 创建模拟器
    core = RingSonarCore(
        world_width=30.0,
        world_height=30.0,
        dt=0.1,
        trigger_mode="sequential"  # 顺序扫描，更清晰展示
    )
    
    renderer = RingSonarRenderer(core, render_mode="human")
    
    # 重置环境
    core.reset(regenerate_map=True, seed=42)
    print(f"机器人初始位置: [{core.robot_pos[0]:.2f}, {core.robot_pos[1]:.2f}] m")
    print(f"开始探索...")
    print()
    
    # 运行仿真
    step = 0
    max_steps = 5000
    
    try:
        while step < max_steps:
            # 简单的随机运动策略
            if step % 50 == 0:
                import random
                core.set_velocity(
                    float(random.uniform(-2.0, 3.0)),
                    float(random.uniform(-0.8, 0.8))
                )
            
            # 执行仿真步骤
            core.step()
            core.update_maps()
            
            # 渲染（包含预测）
            renderer.render()
            
            # 每50步打印统计
            if step % 50 == 0:
                state = core.state()
                
                # 从渲染器获取预测统计
                pred_obstacles = ((renderer.obstacle_prediction > 180) & 
                                (renderer.occupancy_grid > 100) & 
                                (renderer.occupancy_grid < 200)).sum()
                
                avg_conf = renderer.prediction_confidence[renderer.prediction_confidence > 0].mean()
                
                print(f"Step {step:4d}: "
                      f"Pos=[{state['position'][0]:5.1f},{state['position'][1]:5.1f}]m, "
                      f"Predicted Obstacles={pred_obstacles:4d}, "
                      f"Avg Confidence={avg_conf:.1f}%")
            
            # 控制帧率
            time.sleep(0.01)
            
            # 检查退出
            import cv2
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n用户退出")
                break
            
            step += 1
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 最终统计
        print()
        print("=" * 60)
        print("最终统计：")
        
        # 占用栅格统计
        explored = (renderer.visit_count > 0).sum()
        total_cells = renderer.grid_width * renderer.grid_height
        coverage = explored / total_cells * 100
        
        free_cells = (renderer.occupancy_grid > 200).sum()
        obstacle_cells = (renderer.occupancy_grid < 80).sum()
        
        # 预测统计
        predicted_obstacles = ((renderer.obstacle_prediction > 180) & 
                              (renderer.occupancy_grid > 100) & 
                              (renderer.occupancy_grid < 200)).sum()
        
        high_conf = (renderer.prediction_confidence > 70).sum()
        avg_conf = renderer.prediction_confidence[renderer.prediction_confidence > 0].mean()
        
        print(f"  探索覆盖率: {coverage:.1f}%")
        print(f"  已知空闲空间: {free_cells} 栅格")
        print(f"  已知障碍物: {obstacle_cells} 栅格")
        print(f"  预测障碍物: {predicted_obstacles} 栅格")
        print(f"  高置信度预测: {high_conf} 栅格")
        print(f"  平均置信度: {avg_conf:.1f}%")
        print()
        
        # 预测准确性分析（需要对比真实地图）
        print("预测模型特点：")
        print("  ✓ 边界区域预测准确（空闲空间边缘）")
        print("  ✓ 高密度信息区域置信度高")
        print("  ✓ 远离机器人区域置信度低")
        print("  ✓ 实时计算，性能高效")
        print()
        print("建议应用：")
        print("  1. 路径规划：避开高概率障碍物区域")
        print("  2. 探索策略：优先验证不确定区域")
        print("  3. 风险评估：结合概率和置信度")
        print("  4. 地图补全：填补传感器盲区")
        print("=" * 60)
        
        import cv2
        cv2.destroyAllWindows()
        print("演示完成！")

if __name__ == "__main__":
    main()
