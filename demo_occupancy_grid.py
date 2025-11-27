#!/usr/bin/env python3
"""
演示栅格占用图（Occupancy Grid Map）
展示类似SLAM的效果，每个传感器的圆锥形无障碍区域会逐步建立地图
"""

import cv2
import numpy as np
from ring_sonar_simulator import RingSonarCore, RingSonarRenderer

def main():
    print("=" * 60)
    print("栅格占用图演示（SLAM风格）")
    print("=" * 60)
    print()
    print("功能说明：")
    print("  - 显示3个窗口：")
    print("    1. Ring Sonar Simulation: 2D世界俯视图（传感器FoV扇区）")
    print("    2. Feature Map: Fisher信息地图")
    print("    3. Occupancy Grid: 栅格占用图（SLAM）")
    print()
    print("  - 全局栅格占用图特点：")
    print("    · 白色区域: 传感器确认的无障碍区域（累积）")
    print("    · 黑色区域: 检测到的障碍物位置")
    print("    · 灰色区域: 未探索或不确定区域")
    print("    · 红色标记: 机器人当前位置和朝向")
    print()
    print("  - 每个传感器的检测范围形成圆锥形无障碍区域")
    print("  - 随着机器人移动，全局地图会持续累积建立")
    print("  - 障碍物检测：在传感器检测距离处标记障碍物")
    print()
    print("按 'q' 退出, 'r' 重置环境")
    print("=" * 60)
    print()
    
    # 创建环境
    core = RingSonarCore(
        world_width=40.0,
        world_height=40.0,
        num_sensors=12,
        sensor_ring_radius=0.15,
        sensor_fov=65.0,
        sensor_max_range=12.5
    )
    
    core.reset(regenerate_map=True)
    
    renderer = RingSonarRenderer(core, render_mode="human")
    
    print(f"环境初始化完成")
    print(f"  机器人位置: [{core.robot_pos[0]:.2f}, {core.robot_pos[1]:.2f}] m")
    print(f"  传感器数量: {core.num_sensors}")
    print(f"  栅格分辨率: {renderer.grid_resolution}m/cell")
    print(f"  栅格大小: {renderer.grid_width} x {renderer.grid_height}")
    print()
    
    step = 0
    paused = False
    
    try:
        while True:
            if not paused:
                # 简单的随机探索策略
                if step % 50 == 0:
                    # 随机改变速度
                    import random
                    v = random.uniform(-2.0, 2.0)
                    w = random.uniform(-1.0, 1.0)
                    core.set_velocity(v, w)
                
                # 执行一步
                core.step()
                core.update_maps()
                step += 1
                
                # 每10步显示一次信息
                if step % 10 == 0:
                    state = core.state()
                    fisher_stats = core.fisher_map_stats()
                    explored = np.sum(renderer.visit_count > 0) / renderer.visit_count.size * 100
                    print(f"Step {step:4d}: "
                          f"Pos=[{state['position'][0]:5.2f}, {state['position'][1]:5.2f}]m, "
                          f"Fisher={int(fisher_stats['total_features']):4d}, "
                          f"Explored={explored:.1f}%")
            
            # 渲染
            renderer.render()
            
            # 处理键盘输入
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('r'):
                print("\n重置环境...")
                core.reset(regenerate_map=True)
                renderer.reset_grid()  # 重置全局栅格图
                step = 0
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'暂停' if paused else '继续'}仿真")
    
    except KeyboardInterrupt:
        print("\n\n收到中断信号，退出...")
    
    finally:
        cv2.destroyAllWindows()
        print("\n演示结束")
        print(f"总步数: {step}")
        explored = np.sum(renderer.visit_count > 0) / renderer.visit_count.size * 100
        free_cells = np.sum(renderer.occupancy_grid > 200)
        obstacle_cells = np.sum(renderer.occupancy_grid < 50)
        print(f"探索覆盖率: {explored:.1f}%")
        print(f"无障碍栅格: {free_cells}, 障碍物栅格: {obstacle_cells}")


if __name__ == "__main__":
    main()
