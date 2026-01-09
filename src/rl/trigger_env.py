import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
from src.simulator.ring_sonar_simulator import RingSonarCore
from src.simulator.trigger_strategies import SectorStrategy, TriggerContext

class SonarTriggerEnv(gym.Env):
    """
    专门用于训练传感器触发策略的 Gymnasium 环境
    
    修改版：以 Sector 策略为基础，RL 作为其高级过滤器和调度器。
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
        
        # 辅助 Sector 策略 (用于产生 Proposal)
        # 初始化 Sector 策略
        self.sector_helper = SectorStrategy(num_sensors=12)
        self.last_proposal = np.zeros(12, dtype=np.float32)
        
        # 动作空间: 12个传感器的二进制触发 (0: 不触发, 1: 触发)
        self.action_space = spaces.MultiBinary(12)
        
        # 帧堆叠设置 (增加历史信息)
        self.stack_size = 3
        
        # 观察空间: 
        # 1. 局部地图 (125x125, 2通道 * 3帧 = 6通道)
        # 2. 传感器就绪状态 (12维 * 3帧 = 36维)
        # 3. 传感器上次读数 (12维 * 3帧 = 36维)
        # 4. Sector 建议 (12维)
        self.observation_space = spaces.Dict({
            "local_map": spaces.Box(low=0, high=1, shape=(125, 125, 2 * self.stack_size), dtype=np.float32),
            "sensor_ready": spaces.Box(low=0, high=1, shape=(12 * self.stack_size,), dtype=np.float32),
            "last_readings": spaces.Box(low=-1.0, high=15.0, shape=(12 * self.stack_size,), dtype=np.float32),
            "greedy_proposal": spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32) # key 保持不变以便兼容，实际含义变为 sector_proposal
        })
        
        self.max_steps = 1024
        self.current_step = 0
        
        # 自定义概率栅格地图 (0.5=未知, 0.0=空闲, 1.0=占用)
        self.map_size = self.core.global_feature_map.shape[0]
        self.resolution = self.core.feature_map_resolution
        self.occupancy_map = np.full((self.map_size, self.map_size), 0.5, dtype=np.float32)
        
        # 新增：扫描陈旧度地图 (0.0=刚刚扫过, 1.0=极度陈旧)
        self.staleness_map = np.ones((self.map_size, self.map_size), dtype=np.float32)
        
        # 历史记录缓存
        self.obs_history = []
        
        self.prev_fisher_coverage = 0
        self.prev_soft_coverage = 0
        self.prev_total_fisher = 0
        self.consecutive_trigger_counts = np.zeros(12, dtype=np.int32)
        self.last_fire_times = np.full(12, -1.0, dtype=np.float64) # 记录每个传感器上次发射的时间，初始化为-1.0

    def _get_obs(self):
        # 1. 提取当前帧局部地图 (以机器人为中心)
        res = self.resolution
        ms = self.map_size
        ww, wh = self.core.world_width, self.core.world_height
        
        gx = int(self.core.robot_pos[0] / res + ms // 2 - ww // (2 * res))
        gy = int(self.core.robot_pos[1] / res + ms // 2 - wh // (2 * res))
        
        half_size = 62
        x1, x2 = max(0, gx - half_size), min(ms, gx + half_size + 1)
        y1, y2 = max(0, gy - half_size), min(ms, gy + half_size + 1)
        
        current_local_map = np.zeros((125, 125, 2), dtype=np.float32)
        current_local_map[:, :, 0] = 0.5
        current_local_map[:, :, 1] = 1.0
        
        crop_occ = self.occupancy_map[y1:y2, x1:x2]
        crop_stale = self.staleness_map[y1:y2, x1:x2]
        
        h, w = crop_occ.shape
        dy1 = half_size - (gy - y1)
        dx1 = half_size - (gx - x1)
        
        current_local_map[dy1:dy1+h, dx1:dx1+w, 0] = crop_occ
        current_local_map[dy1:dy1+h, dx1:dx1+w, 1] = crop_stale
        
        # 2. 传感器就绪状态
        current_ready = np.zeros(12, dtype=np.float32)
        
        for i in range(12):
            # 物理就绪时间 (等待回波)
            phys_wait = max(0, self.core.sensor_ready_times[i] - self.core.sim_time)
            current_ready[i] = min(1.0, phys_wait / 0.1)
            
        # 3. 上次读数
        current_readings = self.core.sonar_readings.astype(np.float32)
        
        # 3.5 计算 Sector 建议 (替代原 Greedy)
        context = TriggerContext(
            step_count=self.core.step_counter,
            sim_time=self.core.sim_time,
            sonar_readings=self.core.sonar_readings,
            robot_pos=self.core.robot_pos,
            robot_angle=self.core.robot_angle,
            sensors=self.core.sensors,
            sensor_ready_times=self.core.sensor_ready_times,
            global_feature_map=self.core.global_feature_map,
            sensor_max_range=self.core.sensor_max_range,
            world_dims=(self.core.world_width, self.core.world_height)
        )
        # 获取 Sector 建议 (通常是3个传感器)
        proposal_ids = self.sector_helper.get_active_sensors(context)
        proposal_vec = np.zeros(12, dtype=np.float32)
        if proposal_ids:
            proposal_vec[proposal_ids] = 1.0
        self.last_proposal = proposal_vec.copy()

        # 4. 更新历史记录并堆叠
        current_obs = {
            "local_map": current_local_map,
            "sensor_ready": current_ready,
            "greedy_proposal": proposal_vec, # Key 保持不变
            "last_readings": current_readings
        }
        
        if not self.obs_history:
            for _ in range(self.stack_size):
                self.obs_history.append(current_obs)
        else:
            self.obs_history.pop(0)
            self.obs_history.append(current_obs)
            
        # 堆叠观察
        stacked_local_map = np.concatenate([o["local_map"] for o in self.obs_history], axis=-1)
        stacked_ready = np.concatenate([o["sensor_ready"] for o in self.obs_history], axis=0)
        stacked_readings = np.concatenate([o["last_readings"] for o in self.obs_history], axis=0)
        
        return {
            "local_map": stacked_local_map,
            "sensor_ready": stacked_ready,
            "last_readings": stacked_readings,
            "greedy_proposal": proposal_vec
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
            
            # 如果读数为无效 (发生串扰)，则跳过更新地图，但发射成本已扣除
            if reading <= 0:
                continue

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
            
            # 更新陈旧度：被扫描到的区域陈旧度清零
            self.staleness_map[y_min:y_max, x_min:x_max][mask > 0] = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机选择场景
        scene = np.random.choice(["sparse", "simple", "corridor"])
        self.core.reset(regenerate_map=True, seed=seed, scene_type=scene)
        
        # 重置概率地图 (0.5 = 未知)
        self.occupancy_map.fill(0.5)
        # 重置陈旧度地图 (1.0 = 陈旧)
        self.staleness_map.fill(1.0)
        
        self.greedy_helper = None # 已废弃
        self.sector_helper.reset()
        
        self.current_step = 0
        self.obs_history = []
        
        self.prev_fisher_coverage = np.count_nonzero(self.core.global_feature_map > 0.1)
        # 初始软覆盖率 (偏离 0.5 的程度)
        self.prev_soft_coverage = np.sum(np.abs(self.occupancy_map - 0.5))
        self.prev_total_fisher = np.sum(self.core.global_feature_map)
        self.prev_crosstalk = 0
        self.consecutive_trigger_counts = np.zeros(12, dtype=np.int32)
        
        # 记录初始中心位置和角度，用于 Lissajous 曲线
        self.center_pos = self.core.robot_pos.copy()
        self.current_angle_rad = math.radians(self.core.robot_angle)
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. 准备动作 (计算触发ID列表)
        triggered_ids = []
        # Masking: 移除 Sector 限制，允许 RL 探索触发所有传感器
        # 这里的 action 是 RL 自由决定的
        masked_action = action # * self.last_proposal (不再使用 Proposal 掩码)
        
        for i in range(12):
            # 物理就绪
            is_phys_ready = self.core.sim_time >= self.core.sensor_ready_times[i]
            
            if masked_action[i] > 0.5 and action[i] == 1 and is_phys_ready:
                triggered_ids.append(i)
                self.consecutive_trigger_counts[i] += 1
                self.last_fire_times[i] = self.core.sim_time
            else:
                if action[i] == 0:
                    self.consecutive_trigger_counts[i] = 0
        
        # 2. 机器人移动 (同步 Benchmark 中的智能避障 Lissajous 曲线)
        speed = 1.5
        step_dist = speed * self.core.dt
        
        # 生成期望目标点 (改进的 Lissajous 曲线)
        t = self.core.sim_time
        target_x = self.center_pos[0] + 8.5 * math.sin(0.13 * t)
        target_y = self.center_pos[1] + 8.5 * math.sin(0.07 * t + self.current_step * 0.0005 + math.pi/4)
        
        dx = target_x - self.core.robot_pos[0]
        dy = target_y - self.core.robot_pos[1]
        dist = math.hypot(dx, dy)
        desired_angle = math.atan2(dy, dx) if dist > 0 else self.current_angle_rad

        # 避障移动逻辑
        final_pos = None
        angles_to_try = [0]
        for a in range(1, 13): 
            angles_to_try.extend([math.radians(a * 15), math.radians(-a * 15)])
            
        for angle_offset in angles_to_try:
            test_angle = desired_angle + angle_offset
            test_pos = self.core.robot_pos + np.array([math.cos(test_angle) * step_dist, math.sin(test_angle) * step_dist])
            if self.core._position_safe(test_pos):
                final_pos = test_pos
                self.current_angle_rad = test_angle
                break
        
        if final_pos is not None:
            self.core.robot_pos = final_pos
            self.core.robot_angle = math.degrees(self.current_angle_rad) % 360
        self.sector_helper.advance()
            
        # 3. 物理步进 (更新时间, 清空 active_sensors_this_frame)
        self.core.step()
        
        # 4. 执行传感器触发 (动作) - 必须在 core.step() 之后!
        self.core._process_sensor_trigger(triggered_ids)
        
        # 5. 显式更新 Fisher 地图 (核心修复)
        self.core.update_maps()

        # 6. 更新自定义概率地图
        self._update_occupancy_map(list(self.core.active_sensors_this_frame))
        
        # 5. 更新陈旧度地图 (随时间增加，最大为 1.0)
        self.staleness_map = np.clip(self.staleness_map + 0.01, 0.0, 1.0)
        
        # 6. 计算奖励
        
        # A. 覆盖率奖励 (Coverage Gain) - 与 Benchmark 逻辑对齐
        # 统计 Fisher 信息图中值 > 0.1 的像素点数量增量
        current_fisher_coverage = np.count_nonzero(self.core.global_feature_map > 0.1)
        coverage_gain = current_fisher_coverage - self.prev_fisher_coverage
        self.prev_fisher_coverage = current_fisher_coverage
        
        # B. 串扰惩罚 (Crosstalk Penalty)
        # 设置极高的串扰惩罚，强制 RL 学习避免串扰
        new_crosstalk = self.core.crosstalk_count - self.prev_crosstalk
        self.prev_crosstalk = self.core.crosstalk_count
        crosstalk_penalty = new_crosstalk * 1.0  # -500 每帧
        
        # C. 发射成本 (Firing Cost)
        # 赋予传感器发射极小的负奖励
        num_fired = len(triggered_ids)
        firing_cost = num_fired * 0.5   # -0.5 每个传感器
        
        # 奖励组合：
        # 每个新增的 Fisher 覆盖像素点奖励 10.0
        reward = (coverage_gain * 10.0) - (crosstalk_penalty*0) - (firing_cost*0)
        # reward = (coverage_gain * 10.0) - crosstalk_penalty - firing_cost
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"coverage": current_fisher_coverage}

    def render(self):
        pass
