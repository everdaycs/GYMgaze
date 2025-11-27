#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring Sonar Environment - Gymnasium环境
12个超声波传感器环形阵列的强化学习环境
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ring_sonar_simulator import RingSonarCore, RingSonarRenderer


class RingSonarEnv(gym.Env):
    """
    环形超声波雷达环境
    
    观测空间:
        - 12维向量: 每个传感器的距离读数 [0, max_range]
        - 可选: Fisher地图的2D投影
    
    动作空间:
        - Discrete(5): 前进、后退、左转、右转、停止
        或
        - Box(2): [线速度, 角速度] 连续控制
    
    奖励:
        - Fisher信息增益
        - 探索奖励
        - 避障奖励
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 world_size: float = 40.0,
                 action_type: str = "discrete",  # "discrete" 或 "continuous"
                 include_fisher_map: bool = False):
        
        self.render_mode = render_mode
        self.action_type = action_type
        self.include_fisher_map = include_fisher_map
        
        # 创建核心模拟器
        self.core = RingSonarCore(
            world_width=world_size,
            world_height=world_size,
            sensor_ring_radius=0.15,  # 15cm
            num_sensors=12,
            sensor_fov=65.0,
            sensor_max_range=12.5
        )
        
        # 创建渲染器
        if render_mode is not None:
            self.renderer = RingSonarRenderer(self.core, render_mode=render_mode)
        else:
            self.renderer = None
        
        # 定义动作空间
        if action_type == "discrete":
            # 离散动作: 0=前进, 1=后退, 2=左转, 3=右转, 4=停止
            self.action_space = spaces.Discrete(5)
        else:
            # 连续动作: [线速度, 角速度]
            self.action_space = spaces.Box(
                low=np.array([-3.0, -1.0], dtype=np.float32),
                high=np.array([3.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
        # 定义观测空间
        if include_fisher_map:
            # 12个传感器读数 + 32x32 Fisher地图投影
            self.observation_space = spaces.Dict({
                'sonar': spaces.Box(
                    low=0.0,
                    high=self.core.sensor_max_range,
                    shape=(self.core.num_sensors,),
                    dtype=np.float32
                ),
                'fisher_map': spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(32, 32),
                    dtype=np.float32
                )
            })
        else:
            # 仅12个传感器读数
            self.observation_space = spaces.Box(
                low=0.0,
                high=self.core.sensor_max_range,
                shape=(self.core.num_sensors,),
                dtype=np.float32
            )
        
        # 奖励跟踪
        self._prev_fisher_count = 0
        self._prev_collision = False
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict]:
        super().reset(seed=seed, options=options)
        
        # 重置核心模拟器
        regenerate = options.get('regenerate_map', True) if options else True
        self.core.reset(regenerate_map=regenerate, seed=seed)
        
        # 重置奖励跟踪
        self._prev_fisher_count = 0
        self._prev_collision = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        # 应用动作
        self._apply_action(action)
        
        # 执行仿真步骤
        self.core.step()
        self.core.update_maps()
        
        # 获取观测和状态
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.renderer is not None:
            return self.renderer.render()
        return None
    
    def close(self):
        if self.renderer is not None:
            import cv2
            cv2.destroyAllWindows()
    
    # -------- 内部方法 -------- #
    
    def _apply_action(self, action):
        """将动作应用到模拟器"""
        if self.action_type == "discrete":
            # 离散动作映射
            if action == 0:  # 前进
                self.core.set_velocity(2.0, 0.0)
            elif action == 1:  # 后退
                self.core.set_velocity(-1.0, 0.0)
            elif action == 2:  # 左转
                self.core.set_velocity(1.0, 0.6)
            elif action == 3:  # 右转
                self.core.set_velocity(1.0, -0.6)
            elif action == 4:  # 停止
                self.core.set_velocity(0.0, 0.0)
        else:
            # 连续动作
            linear_vel, angular_vel = action
            self.core.set_velocity(float(linear_vel), float(angular_vel))
    
    def _get_obs(self) -> Any:
        """获取观测"""
        if self.include_fisher_map:
            # 返回字典观测
            sonar = self.core.sonar_readings.copy()
            fisher_map = self._get_fisher_map_projection()
            return {
                'sonar': sonar,
                'fisher_map': fisher_map
            }
        else:
            # 仅返回传感器读数
            return self.core.sonar_readings.copy()
    
    def _get_fisher_map_projection(self) -> np.ndarray:
        """获取Fisher地图的降采样投影"""
        # 将100x100的Fisher地图降采样到32x32
        import cv2
        fmap = self.core.feature_map
        projection = cv2.resize(fmap, (32, 32), interpolation=cv2.INTER_AREA)
        return projection.astype(np.float32)
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. Fisher信息增益奖励
        stats = self.core.fisher_map_stats()
        current_fisher_count = stats['total_features']
        
        if current_fisher_count > self._prev_fisher_count:
            # 发现新特征
            new_features = current_fisher_count - self._prev_fisher_count
            reward += 0.5 * (new_features / 10.0)  # 归一化
        
        # Fisher强度奖励
        fisher_intensity = min(stats['mean_fisher'] * 0.1, 1.0)
        reward += fisher_intensity * 0.3
        
        self._prev_fisher_count = current_fisher_count
        
        # 2. 避障奖励
        state = self.core.state()
        min_distance = np.min(self.core.sonar_readings)
        
        if min_distance < 1.0:
            # 太近，给予负奖励
            reward -= 0.5
        elif min_distance < 2.0:
            # 稍近，轻微负奖励
            reward -= 0.2
        
        # 3. 碰撞惩罚
        if state['collision_occurred']:
            reward -= 1.0
            if not self._prev_collision:
                self._prev_collision = True
        else:
            self._prev_collision = False
        
        # 4. 被卡住惩罚
        if state['stuck_counter'] > 10:
            reward -= 0.5
        
        # 5. 探索奖励（轻微基础奖励）
        reward += 0.1
        
        return float(reward)
    
    def _is_terminated(self) -> bool:
        """判断是否终止"""
        state = self.core.state()
        
        # 如果被卡住太久，终止
        if state['stuck_counter'] > 50:
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """判断是否截断"""
        # 时间限制
        if self.core.sim_time > 100.0:
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        state = self.core.state()
        stats = self.core.fisher_map_stats()
        
        return {
            'position': state['position'],
            'angle': state['angle'],
            'sim_time': state['sim_time'],
            'fisher_features': stats['total_features'],
            'fisher_mean': stats['mean_fisher'],
            'collision': state['collision_occurred'],
            'stuck_counter': state['stuck_counter'],
            'sonar_min': float(np.min(self.core.sonar_readings)),
            'sonar_mean': float(np.mean(self.core.sonar_readings))
        }


# 注册环境
if __name__ != "__main__":
    try:
        gym.register(
            id='RingSonar-v0',
            entry_point='gymnasium_env_gaze.envs:RingSonarEnv',
            max_episode_steps=1000,
        )
    except Exception as e:
        pass  # 可能已注册
