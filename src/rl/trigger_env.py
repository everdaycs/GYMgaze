import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
from ring_sonar_simulator import RingSonarCore

class SonarTriggerEnv(gym.Env):
    """
    专门用于训练传感器触发策略的 Gymnasium 环境
    """
    def __init__(self, render_mode=None):
        super(SonarTriggerEnv, self).__init__()
        
        # 初始化模拟器
        self.core = RingSonarCore(
            world_width=20.0,
            world_height=20.0,
            trigger_mode="manual", # 手动控制触发
            randomize_trigger=False
        )
        
        # 动作空间: 12个传感器的二进制触发 (0: 不触发, 1: 触发)
        self.action_space = spaces.MultiBinary(12)
        
        # 观察空间: 
        # 1. 局部地图 (40x40, 对应 4m x 4m 区域)
        # 2. 传感器就绪状态 (12维)
        # 3. 传感器上次读数 (12维)
        self.observation_space = spaces.Dict({
            "local_map": spaces.Box(low=0, high=10, shape=(40, 40, 1), dtype=np.float32),
            "sensor_ready": spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32),
            "last_readings": spaces.Box(low=0, high=5, shape=(12,), dtype=np.float32)
        })
        
        self.max_steps = 1000
        self.current_step = 0
        self.prev_coverage = 0

    def _get_obs(self):
        # 1. 提取局部地图 (以机器人为中心)
        res = self.core.feature_map_resolution
        ms = self.core.global_feature_map.shape[0]
        ww, wh = self.core.world_width, self.core.world_height
        
        # 机器人地图坐标
        gx = int(self.core.robot_pos[0] / res + ms // 2 - ww // (2 * res))
        gy = int(self.core.robot_pos[1] / res + ms // 2 - wh // (2 * res))
        
        # 裁剪 40x40 区域
        half_size = 20
        x1, x2 = max(0, gx - half_size), min(ms, gx + half_size)
        y1, y2 = max(0, gy - half_size), min(ms, gy + half_size)
        
        local_map = np.zeros((40, 40, 1), dtype=np.float32)
        crop = self.core.global_feature_map[y1:y2, x1:x2]
        
        # 填充到 local_map (处理边界情况)
        h, w = crop.shape
        dy1 = half_size - (gy - y1)
        dx1 = half_size - (gx - x1)
        local_map[dy1:dy1+h, dx1:dx1+w, 0] = crop
        
        # 2. 传感器就绪状态 (归一化: 0表示就绪, >0表示还需等待的时间)
        ready_status = np.zeros(12, dtype=np.float32)
        for i in range(12):
            wait_time = max(0, self.core.sensor_ready_times[i] - self.core.sim_time)
            ready_status[i] = min(1.0, wait_time / 0.1) # 假设最大等待0.1s
            
        # 3. 上次读数
        readings = self.core.sonar_readings.astype(np.float32)
        
        return {
            "local_map": local_map,
            "sensor_ready": ready_status,
            "last_readings": readings
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机选择场景
        scene = np.random.choice(["sparse", "simple", "corridor"])
        self.core.reset(regenerate_map=True, seed=seed, scene_type=scene)
        
        self.current_step = 0
        self.prev_coverage = 0
        self.prev_crosstalk = 0
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. 执行动作 (触发选中的传感器)
        # 注意: 只有 ready 的传感器才能被触发
        triggered_ids = []
        for i in range(12):
            if action[i] == 1 and self.core.sim_time >= self.core.sensor_ready_times[i]:
                triggered_ids.append(i)
        
        # 手动调用模拟器的触发处理
        self.core._process_sensor_trigger(triggered_ids)
        
        # 2. 机器人移动 (使用 Lissajous 轨迹模拟探索过程)
        # 在 RL 训练中，我们假设机器人是自主移动的，RL 只负责触发
        t = self.current_step * 0.05
        # 简单的圆周+扰动移动，确保覆盖不同区域
        speed = 1.5
        vx = speed * math.cos(0.1 * t)
        vy = speed * math.sin(0.07 * t)
        
        new_pos = self.core.robot_pos + np.array([vx, vy]) * self.core.dt
        if self.core._position_safe(new_pos):
            self.core.robot_pos = new_pos
            self.core.robot_angle = (self.core.robot_angle + 1.0) % 360
            
        # 3. 物理步进
        self.core.step()
        self.core.update_maps()
        
        # 4. 计算奖励
        current_coverage = np.count_nonzero(self.core.global_feature_map > 0.1)
        coverage_gain = current_coverage - self.prev_coverage
        self.prev_coverage = current_coverage
        
        # 串扰惩罚
        crosstalk_penalty = (self.core.crosstalk_count - self.prev_crosstalk) * 5.0
        self.prev_crosstalk = self.core.crosstalk_count
        
        # 发射惩罚 (能耗)
        firing_penalty = len(triggered_ids) * 0.1
        
        reward = (coverage_gain * 0.5) - crosstalk_penalty - firing_penalty
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"coverage": current_coverage}

    def render(self):
        pass
