import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
from ring_sonar_simulator import RingSonarCore

class SonarTriggerEnv(gym.Env):
    """
    专门用于训练传感器触发策略的 Gymnasium 环境
    
    修改版：不使用 Fisher 信息，而是模拟真实的低分辨率超声波栅格地图 (Occupancy Grid Map)。
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
        # 1. 局部地图 (80x80, 对应 8m x 8m 区域) - 现在是概率栅格地图
        # 2. 传感器就绪状态 (12维)
        # 3. 传感器上次读数 (12维)
        self.observation_space = spaces.Dict({
            "local_map": spaces.Box(low=0, high=1, shape=(80, 80, 1), dtype=np.float32),
            "sensor_ready": spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32),
            "last_readings": spaces.Box(low=0, high=15, shape=(12,), dtype=np.float32)
        })
        
        self.max_steps = 1000
        self.current_step = 0
        
        # 自定义概率栅格地图 (0.5=未知, 0.0=空闲, 1.0=占用)
        # 尺寸与 core.global_feature_map 保持一致
        self.map_size = self.core.global_feature_map.shape[0]
        self.resolution = self.core.feature_map_resolution
        self.occupancy_map = np.full((self.map_size, self.map_size), 0.5, dtype=np.float32)
        
        self.prev_confident_cells = 0

    def _get_obs(self):
        # 1. 提取局部地图 (以机器人为中心)
        res = self.resolution
        ms = self.map_size
        ww, wh = self.core.world_width, self.core.world_height
        
        # 机器人地图坐标
        gx = int(self.core.robot_pos[0] / res + ms // 2 - ww // (2 * res))
        gy = int(self.core.robot_pos[1] / res + ms // 2 - wh // (2 * res))
        
        # 裁剪 80x80 区域 (8m x 8m)
        half_size = 40
        x1, x2 = max(0, gx - half_size), min(ms, gx + half_size)
        y1, y2 = max(0, gy - half_size), min(ms, gy + half_size)
        
        local_map = np.full((80, 80, 1), 0.5, dtype=np.float32)
        crop = self.occupancy_map[y1:y2, x1:x2]
        
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

    def _update_occupancy_map(self, triggered_ids):
        """
        模拟低分辨率超声波传感器的建图过程
        不使用 Fisher 信息，而是使用简单的概率更新模型
        """
        if not triggered_ids:
            return

        # 获取地图参数
        res = self.resolution
        ms = self.map_size
        ww, wh = self.core.world_width, self.core.world_height
        center_offset_x = ms // 2 - ww // (2 * res)
        center_offset_y = ms // 2 - wh // (2 * res)

        for sensor_id in triggered_ids:
            reading = self.core.sonar_readings[sensor_id]
            sensor = self.core.sensors[sensor_id]
            
            # 获取传感器世界位置和角度
            s_pos = sensor.get_world_position(self.core.robot_pos, self.core.robot_angle)
            s_angle = sensor.get_world_angle(self.core.robot_angle)
            
            # 转换为地图像素坐标
            sx_pix = int(s_pos[0] / res + center_offset_x)
            sy_pix = int(s_pos[1] / res + center_offset_y)
            
            # 传感器参数
            fov_rad = math.radians(sensor.fov_angle)
            angle_rad = math.radians(s_angle)
            max_range_pix = int(sensor.max_range / res)
            reading_pix = int(reading / res)
            
            # 1. 更新空闲区域 (Free Space)
            # 扇形区域：从传感器位置开始，半径为 reading，角度范围 fov
            # 这里的逻辑是：如果读数是 d，那么 0 到 d 之间应该是空的
            # 使用 cv2 绘制扇形掩码
            
            # 限制绘制范围以加速
            roi_radius = reading_pix + 5
            x_min, x_max = max(0, sx_pix - roi_radius), min(ms, sx_pix + roi_radius)
            y_min, y_max = max(0, sy_pix - roi_radius), min(ms, sy_pix + roi_radius)
            
            if x_max <= x_min or y_max <= y_min:
                continue
                
            # 在 ROI 内操作
            roi_map = self.occupancy_map[y_min:y_max, x_min:x_max]
            mask = np.zeros_like(roi_map, dtype=np.uint8)
            
            # 相对坐标
            rel_sx = sx_pix - x_min
            rel_sy = sy_pix - y_min
            
            # 绘制空闲扇区 (Free)
            # 注意：cv2.ellipse 使用的角度是角度制
            start_angle = s_angle - sensor.fov_angle / 2
            end_angle = s_angle + sensor.fov_angle / 2
            
            cv2.ellipse(mask, (rel_sx, rel_sy), (reading_pix, reading_pix), 0, start_angle, end_angle, 255, -1)
            
            # 更新概率：空闲区域概率降低 (e.g. -0.1)
            # 只有当前是未知(0.5)或被错误标记为占用(>0.5)时才显著降低
            free_update = -0.05
            roi_map[mask > 0] = np.clip(roi_map[mask > 0] + free_update, 0.0, 1.0)
            
            # 2. 更新占用区域 (Occupied Space)
            # 如果读数小于最大量程，说明打到了障碍物
            # 超声波特性：障碍物在圆弧上的某处，但不知道具体在哪 -> "低分辨率"
            # 我们增加整个圆弧的占用概率
            if reading < sensor.max_range * 0.95:
                mask_occ = np.zeros_like(roi_map, dtype=np.uint8)
                # 绘制圆弧，厚度为 3 个像素 (模拟测量误差)
                cv2.ellipse(mask_occ, (rel_sx, rel_sy), (reading_pix, reading_pix), 0, start_angle, end_angle, 255, 3)
                
                # 更新概率：占用区域概率增加 (e.g. +0.15)
                # 这种"涂抹"效果迫使 RL 学会利用多传感器交叉来消除不确定性
                occ_update = 0.15
                roi_map[mask_occ > 0] = np.clip(roi_map[mask_occ > 0] + occ_update, 0.0, 1.0)

            # 写回
            self.occupancy_map[y_min:y_max, x_min:x_max] = roi_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机选择场景
        scene = np.random.choice(["sparse", "simple", "corridor"])
        self.core.reset(regenerate_map=True, seed=seed, scene_type=scene)
        
        # 重置概率地图 (0.5 = 未知)
        self.occupancy_map.fill(0.5)
        
        self.current_step = 0
        self.prev_confident_cells = 0
        self.prev_crosstalk = 0
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. 执行动作 (触发选中的传感器)
        triggered_ids = []
        for i in range(12):
            if action[i] == 1 and self.core.sim_time >= self.core.sensor_ready_times[i]:
                triggered_ids.append(i)
        
        # 手动调用模拟器的触发处理
        self.core._process_sensor_trigger(triggered_ids)
        
        # 2. 机器人移动
        t = self.current_step * 0.05
        speed = 1.5
        vx = speed * math.cos(0.1 * t)
        vy = speed * math.sin(0.07 * t)
        
        new_pos = self.core.robot_pos + np.array([vx, vy]) * self.core.dt
        if self.core._position_safe(new_pos):
            self.core.robot_pos = new_pos
            self.core.robot_angle = (self.core.robot_angle + 1.0) % 360
            
        # 3. 物理步进
        self.core.step()
        
        # 4. 更新自定义概率地图 (替代 Fisher 更新)
        # 注意：我们使用上一帧触发的传感器读数来更新地图
        # 实际上 core.step() 后 sonar_readings 已经更新了
        # 我们需要知道哪些传感器在这一步"完成"了测量并更新了读数
        # 简化起见，我们假设 triggered_ids 在这一帧立即产生读数 (虽然物理上有延迟)
        # 或者更准确地，我们应该检查 active_sensors_this_frame
        self._update_occupancy_map(list(self.core.active_sensors_this_frame))
        
        # 5. 计算奖励
        # 奖励基于"置信度高的栅格数量" (Confident Cells)
        # 即 p < 0.3 (确信为空) 或 p > 0.7 (确信为障碍)
        confident_mask = (self.occupancy_map < 0.3) | (self.occupancy_map > 0.7)
        current_confident_cells = np.count_nonzero(confident_mask)
        
        coverage_gain = current_confident_cells - self.prev_confident_cells
        self.prev_confident_cells = current_confident_cells
        
        # 串扰惩罚
        crosstalk_penalty = (self.core.crosstalk_count - self.prev_crosstalk) * 5.0
        self.prev_crosstalk = self.core.crosstalk_count
        
        # 发射惩罚
        firing_penalty = len(triggered_ids) * 0.1
        
        reward = (coverage_gain * 0.1) - crosstalk_penalty - firing_penalty
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"coverage": current_confident_cells}

    def render(self):
        pass
