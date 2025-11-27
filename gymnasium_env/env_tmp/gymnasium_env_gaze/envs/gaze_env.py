from typing import Optional, Tuple
import numpy as np
import numba as nb
import gymnasium as gym
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import time
import sys
import os

# 添加父目录到路径以导入fisher_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from fisher_utils import compute_fisher_nb, add_global_feature_3d_nb, clamp, angnorm_deg, angdiff_deg

class GazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "bird"]}
    def __init__(self, render_mode = None):
        self.render_mode = render_mode
        # use spherical fibonacci sampling to generate gaze angle
        self.action_space = spaces.Discrete(32)
        # 12个超声波传感器的距离读数 (单位: 米)
        self.observation_space = spaces.Box(
            low = 0.0, high = 10.0, 
            shape = (12,), 
            dtype = np.float32
        )
        
        # 12个超声波传感器均匀分布在半径15cm的圆盘边缘
        self.ultrasonic_radius = 0.15  # 15cm
        self.num_sensors = 12
        self.sensor_angles = np.linspace(0, 2*np.pi, self.num_sensors, endpoint=False)
        
        self.pixel_per_meter = 5
        self.world_width = 20     # x
        self.world_length = 20   # y
        self.world_height = 20   # z
        self.obstacle_num : int = 20
        self.max_ray_distance : float = 100.0

        # feature map init
        self.feature_map_size : int = 80
        self.feature_map_res : float = 0.25
        self.feature_map : np.ndarray = np.zeros((self.feature_map_size, self.feature_map_size, self.feature_map_size), dtype = np.float32)
        self.global_feature_map_size : int = int(max(self.world_height, self.world_width, self.world_length)*2 / self.feature_map_res)
        self.global_feature_map : np.ndarray = np.zeros((self.global_feature_map_size, self.global_feature_map_size, self.global_feature_map_size), dtype = np.float32)

        self.dt = 0.1
        self.sim_time = 0.0
        self.control_frequency : float = 5.0
        self.control_period = 1.0 / self.control_frequency if self.control_frequency > 0 else 0.0
        self.last_control_update = 0.0

        # keep robot pos at z = 0 (on the ground)
        self.robot_pos = np.array([self.world_width / 2, self.world_height / 2, 0], dtype=np.float64)
        self.robot_angle : float = 0.0  # deg, body heading, 0 -> x+
        self.robot_vel : float = 0.0
        
        # 主动SLAM运动参数
        self.angular_velocity : float = 0.0  # rad/s
        self.max_linear_velocity = 3.0
        self.max_angular_velocity = 1.0
        self.robot_size = 0.5  # 机器人半径（米）
        self.velocity_sample_interval = 5.0
        self.last_movement_time = 0.0
        
        self._active_gaze_control = False
        self.reset()

    def init_env(self):
        self.space = np.zeros((self.world_length * self.pixel_per_meter, self.world_width * self.pixel_per_meter, self.world_height * self.pixel_per_meter), int)
        self.gaze_angles = self.sample_fibo_sphere()
        self.gaze_angle = self.gaze_angles[0]

    def generate_obstacles(self):
        nx, ny, nz = self.space.shape
        for indices in range (self.obstacle_num):
            x1 = np.random.randint(0, nx-1)
            y1 = np.random.randint(0, ny-1)
            z1 = np.random.randint(0, nz-1)
            x2 = x1 + np.random.randint(nx//8, nx//2)
            y2 = y1 + np.random.randint(ny//8, ny//2)
            z2 = z1 + np.random.randint(nz//8, nz//2)
            

            if x1==x2 : x2 = min(x2, nx - 1)
            if y1==y2 : y2 = min(y2, ny - 1)
            if z1==z2 : z2 = min(z2, nz - 1)

            self.space[x1:x2+1,y1:y2+1,z1:z2+1] = 1

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.sim_time += self.dt
        
        # 设置凝视角度（从动作空间）
        self.gaze_angle = self.gaze_angles[int(action)]
        
        # 主动SLAM运动逻辑
        self.sample_movement()
        self.move()
        
        # 获取传感器观测
        observation = self._get_obs()
        
        # 更新 Fisher 地图
        self.update_maps()
        
        reward = self.get_reward()
        info = {
            "current pos" : self.robot_pos,
            "current chassis angle" : self.robot_angle,
            "current gaze_angle" : self.gaze_angle    
        }
        truncated = False
        terminated = self.sim_time > 100
        return observation, reward, terminated, truncated, info

    def get_reward(self) -> float | int:
        """基于 Fisher 信息增益和探索奖励计算奖励值"""
        # Fisher 信息奖励
        fisher_stats = self.fisher_map_stats()
        
        # 特征数量奖励
        feature_reward = min(fisher_stats["nonzero_count"] * 0.01, 1.0)
        
        # 特征强度奖励
        intensity_reward = min(fisher_stats["mean_fisher"] * 0.1, 1.0)
        
        # 组合奖励，给予更高权重的特征强度
        fisher_reward = feature_reward * 0.3 + intensity_reward * 0.7
        
        # 如果没有特征，给予少量探索奖励
        if fisher_stats["nonzero_count"] == 0:
            # 轻微的探索奖励，避免完全无奖励的情况
            exploration_reward = 0.1
        else:
            exploration_reward = 0
            
        # 如果发现了新特征，额外给予奖励
        new_feature_reward = 0
        if hasattr(self, "_prev_nonzero_count"):
            if fisher_stats["nonzero_count"] > self._prev_nonzero_count:
                new_feature_reward = 0.5
        
        # 保存当前特征数量
        self._prev_nonzero_count = fisher_stats["nonzero_count"]
            
        # 总奖励
        total_reward = fisher_reward + exploration_reward + new_feature_reward
        
        return total_reward
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self.init_env()
        self.generate_obstacles()
        
        # 寻找安全的起始位置
        self.robot_pos = self._find_safe_start()
        self.robot_angle = float(np.random.randint(0, 360))
        self.robot_vel = 0.0
        self.angular_velocity = 0.0
        
        # 重置特征地图
        self.feature_map.fill(0.0)
        self.global_feature_map.fill(0.0)
        self._prev_nonzero_count = 0
        
        # 重置时间
        self.sim_time = 0.0
        self.last_movement_time = 0.0
        
        observation = self._get_obs()
        self.sample_movement(force=True)

        return observation, {}
    
    def _find_safe_start(self) -> np.ndarray:
        """寻找一个不会碰撞的起始位置"""
        margin = self.robot_size + 0.5
        
        # 尝试100次随机位置
        for _ in range(100):
            x = np.random.uniform(margin, self.world_width - margin)
            y = np.random.uniform(margin, self.world_height - margin)
            z = 0.0
            pos = np.array([x, y, z], dtype=np.float64)
            if not self.check_collision(pos):
                return pos
        
        # 如果找不到，尝试中心点
        center = np.array([self.world_width / 2, self.world_height / 2, 0.0], dtype=np.float64)
        if not self.check_collision(center):
            return center
        
        # 最后的备选：强制返回边界内的位置
        return np.array([
            clamp(self.world_width / 2, margin, self.world_width - margin),
            clamp(self.world_height / 2, margin, self.world_height - margin),
            0.0
        ], dtype=np.float64)
    
    def sample_fibo_sphere(self) -> np.ndarray:
        # remove gaze angle that point to lower hemisphere
        nth : int = self.action_space.n # type: ignore # fuck you pylance
        if nth == 0: return np.zeros((0,3), dtype=np.float32)
        indices = np.arange(0, nth)
        phi = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - (indices + 0.5) / nth
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        azimuth = phi * indices
        x = np.cos(azimuth) * np.sin(theta)
        y = np.sin(azimuth) * np.sin(theta)
        dirs = np.stack([x, y, z], axis=1)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        dirs = dirs / norms
        return dirs.astype(np.float32)


    
    def ray_marching(self):
        return ultrasonic_ray_marching(
            self.pixel_per_meter, 
            self.robot_angle, 
            self.robot_pos, 
            self.space, 
            self.global_feature_map, 
            self.global_feature_map_size, 
            self.feature_map_res,
            self.ultrasonic_radius,
            self.sensor_angles
        )

    def _get_obs(self):
        distances, self.global_feature_map = self.ray_marching()
        return distances

    def sample_movement(self, force: bool = False):
        """主动SLAM移动逻辑：速度采样"""
        r = np.random.rand()
        
        if (r >= 0.5 and self.sim_time - self.last_movement_time > self.velocity_sample_interval) or force:
            self.last_movement_time = self.sim_time
            # 线速度和角速度采样
            self.robot_vel = float(np.random.uniform(-self.max_linear_velocity, self.max_linear_velocity))
            self.angular_velocity = float(np.random.uniform(-self.max_angular_velocity, self.max_angular_velocity))

    def check_collision(self, target_pos: np.ndarray) -> bool:
        """检查目标位置是否碰撞"""
        ix = int(round(target_pos[0] * self.pixel_per_meter))
        iy = int(round(target_pos[1] * self.pixel_per_meter))
        iz = int(round(target_pos[2] * self.pixel_per_meter))
        
        # 边界检查
        if ix < 0 or ix >= self.space.shape[0] or iy < 0 or iy >= self.space.shape[1] or iz < 0 or iz >= self.space.shape[2]:
            return True
            
        # 障碍物检查
        return self.space[ix, iy, iz] == 1
        
    def handle_collision(self):
        """碰撞处理：反向运动并添加随机扰动"""
        self.robot_vel = clamp(-self.robot_vel * np.random.uniform(0.5, 1.0) + np.random.uniform(-0.5, 0.5),
                               -self.max_linear_velocity, self.max_linear_velocity)
        if abs(self.angular_velocity) < 0.1:
            self.angular_velocity = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8)
        else:
            self.angular_velocity = clamp(-self.angular_velocity + np.random.uniform(-0.2, 0.2),
                                          -self.max_angular_velocity, self.max_angular_velocity)

    def move(self):
        """主动SLAM移动更新：包含角速度、线速度和碰撞处理"""
        # 更新角度（角速度积分）
        new_angle = (self.robot_angle + np.rad2deg(self.angular_velocity * self.dt)) % 360.0
        
        # 计算新位置（线速度积分）
        new_pos = self.robot_pos.copy()
        new_pos[0] += np.cos(np.deg2rad(self.robot_angle)) * self.robot_vel * self.dt
        new_pos[1] += np.sin(np.deg2rad(self.robot_angle)) * self.robot_vel * self.dt
        
        # 限制在世界范围内
        new_pos[0] = clamp(new_pos[0], self.robot_size, self.world_width - self.robot_size)
        new_pos[1] = clamp(new_pos[1], self.robot_size, self.world_height - self.robot_size)
        
        # 碰撞检测和处理
        if self.check_collision(new_pos):
            self.handle_collision()
        else:
            self.robot_pos = new_pos
            self.robot_angle = new_angle

    def render(self):
        mode = self.render_mode
        # 获取12个传感器的距离读数
        distances = self._get_obs()
        if mode == "rgb_array":
            # 创建一个简单的可视化图像
            vis = np.zeros((200, 200, 3), dtype=np.uint8)
            center = (100, 100)
            
            # 绘制机器人（圆形）
            import cv2
            cv2.circle(vis, center, 10, (255, 255, 255), -1)
            
            # 绘制传感器读数
            yaw_rad = np.deg2rad(self.robot_angle)
            for i, dist in enumerate(distances):
                angle = self.sensor_angles[i] + yaw_rad
                # 传感器位置
                sx = int(center[0] + self.ultrasonic_radius * 100 * np.cos(angle))
                sy = int(center[1] + self.ultrasonic_radius * 100 * np.sin(angle))
                # 障碍物位置
                ex = int(sx + dist * 10 * np.cos(angle))
                ey = int(sy + dist * 10 * np.sin(angle))
                
                color = (0, 255, 0) if dist < 10.0 else (255, 0, 0)
                cv2.line(vis, (sx, sy), (ex, ey), color, 1)
                cv2.circle(vis, (sx, sy), 2, (0, 255, 255), -1)
            
            return vis
        elif mode == "human":
            # show using matplotlib if available
            try:
                import matplotlib.pyplot as plt
                if not hasattr(self, "_fig_human") or self._fig_human is None:
                    self._fig_human, self._ax_human = plt.subplots(1, 2, figsize=(12, 6))
                    plt.ion()
                    plt.show()
                
                # 重新布局：2x2子图
                if len(self._ax_human.shape) == 1:
                    plt.close(self._fig_human)
                    self._fig_human, self._ax_human = plt.subplots(2, 2, figsize=(14, 12))
                    plt.ion()
                
                # 左上：超声波传感器读数
                self._ax_human[0, 0].clear()
                angles_deg = np.rad2deg(self.sensor_angles)
                self._ax_human[0, 0].plot(angles_deg, distances, 'bo-', linewidth=2, markersize=8)
                self._ax_human[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Warning')
                self._ax_human[0, 0].set_xlabel("Sensor Angle (deg)")
                self._ax_human[0, 0].set_ylabel("Distance (m)")
                self._ax_human[0, 0].set_title(f"Ultrasonic (min: {np.min(distances):.2f}m)")
                self._ax_human[0, 0].set_ylim([0, 10])
                self._ax_human[0, 0].grid(True)
                self._ax_human[0, 0].legend()
                
                # 右上：Fisher信息地图
                self._ax_human[0, 1].clear()
                z_proj = np.max(self.feature_map, axis=2)
                vmax = np.max(z_proj)
                if vmax > 0:
                    self._ax_human[0, 1].imshow(z_proj, cmap="jet", vmin=0, vmax=vmax, origin='lower')
                else:
                    self._ax_human[0, 1].imshow(z_proj, cmap="jet", vmin=0, vmax=1, origin='lower')
                
                cx, cy = self.feature_map_size // 2, self.feature_map_size // 2
                self._ax_human[0, 1].plot(cx, cy, 'wo', markersize=8)
                
                L = 10
                ex = cx + np.cos(np.deg2rad(self.robot_angle)) * L
                ey = cy + np.sin(np.deg2rad(self.robot_angle)) * L
                self._ax_human[0, 1].arrow(cx, cy, ex-cx, ey-cy, head_width=3, head_length=2, 
                                          fc='white', ec='white', linewidth=2)
                
                stats = self.fisher_map_stats()
                self._ax_human[0, 1].set_title(f"Fisher Map ({int(stats['total_features'])} features)")
                self._ax_human[0, 1].axis('off')
                
                # 左下：状态信息
                self._ax_human[1, 0].clear()
                self._ax_human[1, 0].axis('off')
                status_text = (
                    f"=== Robot State ===\n"
                    f"Pos: [{self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}]m\n"
                    f"Heading: {self.robot_angle:.1f}°\n"
                    f"Lin Vel: {self.robot_vel:.2f} m/s\n"
                    f"Ang Vel: {np.rad2deg(self.angular_velocity):.2f} °/s\n\n"
                    f"=== Simulation ===\n"
                    f"Time: {self.sim_time:.1f}s\n\n"
                    f"=== Fisher Stats ===\n"
                    f"Features: {int(stats['total_features'])}\n"
                    f"Mean: {stats['mean_fisher']:.3f}\n"
                    f"Density: {stats['density']:.3f}"
                )
                self._ax_human[1, 0].text(0.1, 0.5, status_text, fontsize=10, family='monospace',
                                         verticalalignment='center')
                
                # 右下：极坐标雷达图
                self._ax_human[1, 1].clear()
                self._ax_human[1, 1].remove()
                self._ax_human[1, 1] = self._fig_human.add_subplot(2, 2, 4, projection='polar')
                theta = self.sensor_angles
                r = distances
                self._ax_human[1, 1].plot(theta, r, 'bo-', linewidth=2, markersize=8)
                self._ax_human[1, 1].fill(theta, r, alpha=0.25)
                self._ax_human[1, 1].set_ylim([0, 10])
                self._ax_human[1, 1].set_theta_zero_location('N')
                self._ax_human[1, 1].set_theta_direction(-1)
                self._ax_human[1, 1].set_title("Radar View", pad=20)
                self._ax_human[1, 1].grid(True)
                
                plt.tight_layout()
                plt.pause(0.01)
                
            except Exception as e:
                print(f" {e}")
                pass
        elif mode == "bird":
            # project along z-axis: any occupied voxel becomes True
            occupancy_z = np.any(self.space, axis=2)
            occupancy_z[self.robot_pos[0], self.robot_pos[1]] = 0.5
            # project alone y-axis
            occupancy_y = np.any(self.space, axis = 1)
            occupancy_y[self.robot_pos[0], self.robot_pos[2]] = 0.5
            # convert to 0/255 image
            img1 = (occupancy_z.astype(np.float16) * 255).astype(np.uint8)
            img2 = (occupancy_y.astype(np.float16) * 255).astype(np.uint8)
            # stack to RGB channels
            bird = np.stack([img1, img1, img1], axis=-1)
            side = np.stack([img2, img2, img2], axis=-1)
            
            try:
                import matplotlib.pyplot as plt
                if not hasattr(self, "_fig"):
                    self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2, figsize=(8, 4))
                    self._im1 = self._ax1.imshow(bird)
                    self._im2 = self._ax2.imshow(side)
                    self._ax1.axis('off')
                    self._ax2.axis('off')
                    plt.ion()
                    plt.show()

                self._im1.set_data(bird)
                self._im2.set_data(side)
                plt.pause(0.001)
            except Exception as e:
                print(f" {e}")
        
    def visualize_feature_map(self, show=False):
        # 对三维特征地图进行最大值投影
        z_proj = np.max(self.feature_map, axis=2)
        y_proj = np.max(self.feature_map, axis=1)
        x_proj = np.max(self.feature_map, axis=0)
        
        # 归一化到 [0, 255] 范围
        vmax_z = np.max(z_proj)
        vmax_y = np.max(y_proj)
        vmax_x = np.max(x_proj)
        
        if vmax_z > 0:
            z_norm = (z_proj / vmax_z * 255.0).astype(np.uint8)
        else:
            z_norm = np.zeros_like(z_proj, dtype=np.uint8)
            
        if vmax_y > 0:
            y_norm = (y_proj / vmax_y * 255.0).astype(np.uint8)
        else:
            y_norm = np.zeros_like(y_proj, dtype=np.uint8)
            
        if vmax_x > 0:
            x_norm = (x_proj / vmax_x * 255.0).astype(np.uint8)
        else:
            x_norm = np.zeros_like(x_proj, dtype=np.uint8)
        
        # 应用颜色映射
        import cv2
        z_heat = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
        y_heat = cv2.applyColorMap(y_norm, cv2.COLORMAP_JET)
        x_heat = cv2.applyColorMap(x_norm, cv2.COLORMAP_JET)
        
        # 标记机器人位置
        cz = self.feature_map_size // 2
        cy = self.feature_map_size // 2
        cx = self.feature_map_size // 2
        
        # 在 z 投影上标记机器人位置
        cv2.circle(z_heat, (cx, cy), 3, (255, 255, 255), -1)
        cv2.circle(z_heat, (cx, cy), 4, (0, 0, 0), 1)
        
        # 绘制机器人朝向和视线方向
        L = 10
        # 机器人朝向
        ex = int(cx + np.cos(np.deg2rad(self.robot_angle)) * L)
        ey = int(cy + np.sin(np.deg2rad(self.robot_angle)) * L)
        cv2.arrowedLine(z_heat, (cx, cy), (ex, ey), (255, 255, 255), 2)
        
        # 视线方向
        gx = int(cx + self.gaze_angle[0] * L)
        gy = int(cy + self.gaze_angle[1] * L)
        cv2.arrowedLine(z_heat, (cx, cy), (gx, gy), (0, 255, 255), 2)
        
        # 组合三个投影到一个图像
        combined = np.vstack((z_heat, np.vstack((y_heat, x_heat))))
        
        # 添加标签和统计信息
        stats = self.fisher_map_stats()
        cv2.putText(combined, "Fisher Information Map (3D)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Top: XY projection, Middle: XZ projection, Bottom: YZ projection", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Features: {int(stats['nonzero_count'])}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Avg Fisher: {stats['mean_fisher']:.2f}", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 12))
                plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title("3D Fisher Information Map")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"可视化失败: {e}")
                
        return combined

# ----- Fisher Map Methods ----- #
    def update_maps(self):
        self._apply_feature_decay()
        self._detect_and_add_features_to_global_map()
        self._extract_local_feature_map()

    def _apply_feature_decay(self):
        """对全局特征地图应用轻微的衰减"""
        self.global_feature_map *= (1.0 - 5e-6)
        self.global_feature_map[self.global_feature_map < 0.1] = 0.0
        

    def _detect_and_add_features_to_global_map(self):
        depth_map = self.ray_marching()
        

    def _fisher_at(self, hit_pos, distance, ang_rad):
        """计算特定位置的 Fisher 信息值"""
        # 距离因子：距离越远，特征值越小
        dist_factor = 1.0 / max(distance / 40.0, 0.1)
        dist_factor = min(dist_factor, 10.0)
        
        # 角度因子：与主轴的偏差越小，特征值越大
        angle_deg = (np.rad2deg(ang_rad) % 360.0)
        min_dev = min(angdiff_deg(angle_deg, d) for d in (0.0, 90.0, 180.0, 270.0))
        ang_factor = max(np.cos(np.deg2rad(min_dev)) ** 2, 0.1)
        
        # 视野中心因子：与视野中心的偏差越小，特征值越大
        dev = angdiff_deg(angle_deg, np.rad2deg(np.arccos(self.gaze_angle[2])) % 360.0)
        fov_factor = np.maximum(np.exp(-dev / 22.5), 0.2)  # 90度FOV的四分之一是22.5度
        print(clamp(dist_factor * ang_factor * fov_factor, 0.1, 10.0))
        return clamp(dist_factor * ang_factor * fov_factor, 0.1, 10.0)

    def _add_global_feature(self, pos, val):
        """将特征值添加到全局特征地图中"""
        # 使用统一的3D特征添加函数
        self.global_feature_map = add_global_feature_3d_nb(
            self.global_feature_map,
            float(pos[0]), float(pos[1]), float(pos[2]), val,
            self.global_feature_map_size, self.feature_map_res,
            float(self.world_width), float(self.world_length), float(self.world_height)
        )

    def _extract_local_feature_map(self):
        """从全局特征地图中提取以机器人为中心的局部特征地图"""
        # 重置局部特征地图
        self.feature_map.fill(0.0)
        # 暂时使用简化的实现，稍后优化
        m = self.global_feature_map
        size = self.global_feature_map_size
        res = self.feature_map_res
        half = self.feature_map_size // 2
        
        # 机器人在全局地图中的索引
        rx = int(self.robot_pos[0] / res + size // 2 - self.world_width // (2 * res))
        ry = int(self.robot_pos[1] / res + size // 2 - self.world_length // (2 * res))
        rz = int(self.robot_pos[2] / res + size // 2 - self.world_height // (2 * res))
        
        # 局部地图的边界
        gx0, gy0, gz0 = rx - half, ry - half, rz - half
        gx1, gy1, gz1 = gx0 + self.feature_map_size, gy0 + self.feature_map_size, gz0 + self.feature_map_size
        
        # 处理边界情况
        sx0 = max(0, -gx0)
        sy0 = max(0, -gy0)
        sz0 = max(0, -gz0)
        sx1 = self.feature_map_size - max(0, gx1 - size)
        sy1 = self.feature_map_size - max(0, gy1 - size)
        sz1 = self.feature_map_size - max(0, gz1 - size)
        
        Gx0 = max(0, gx0)
        Gy0 = max(0, gy0)
        Gz0 = max(0, gz0)
        Gx1 = min(size, gx1)
        Gy1 = min(size, gy1)
        Gz1 = min(size, gz1)
        
        # 填充局部特征地图
        if sx0 < sx1 and sy0 < sy1 and sz0 < sz1:
            self.feature_map[sx0:sx1, sy0:sy1, sz0:sz1] = m[Gx0:Gx1, Gy0:Gy1, Gz0:Gz1]
    
    def fisher_map_stats(self) -> dict:
        """计算 Fisher 地图的统计数据"""
        flat = self.feature_map.ravel()
        nonzero = flat[flat > 0]
        
        if len(nonzero) > 0:
            total = float(np.sum(nonzero))
            mean = float(np.mean(nonzero))
            density = float(len(nonzero) / len(flat))
            max_val = float(np.max(nonzero))
        else:
            total = 0.0
            mean = 0.0
            density = 0.0
            max_val = 0.0
            
        return {
            "total_features": total,
            "mean_fisher": mean,
            "density": density,
            "max_value": max_val,
            "nonzero_count": len(nonzero)
        }
@nb.njit(parallel=False, fastmath=True)
def ultrasonic_ray_marching(ppm, bot_angle, robot_pos, space, global_feature_map, global_size, global_res, ultrasonic_radius, sensor_angles):
    """
    12个超声波传感器的射线投射
    传感器均匀分布在半径15cm的圆盘边缘，平行于机器人前进方向发射
    """
    copy_global_feature = global_feature_map.copy()
    max_ray_distance = 10 * ppm  # 最大检测距离
    num_sensors = len(sensor_angles)
    distances = np.full(num_sensors, max_ray_distance / ppm, dtype=np.float32)
    
    # 机器人朝向（弧度）
    yaw_rad = np.deg2rad(bot_angle)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    for i in range(num_sensors):
        # 传感器相对机器人中心的位置（圆盘边缘）
        local_x = ultrasonic_radius * np.cos(sensor_angles[i])
        local_y = ultrasonic_radius * np.sin(sensor_angles[i])
        
        # 旋转到世界坐标系
        sensor_x = robot_pos[0] + local_x * cos_yaw - local_y * sin_yaw
        sensor_y = robot_pos[1] + local_x * sin_yaw + local_y * cos_yaw
        sensor_z = robot_pos[2]
        
        sensor_pos = np.array([sensor_x, sensor_y, sensor_z], dtype=np.float32)
        
        # 传感器朝向（与机器人朝向一致，水平发射）
        ray_dir = np.array([cos_yaw, sin_yaw, 0.0], dtype=np.float32)
        
        # 射线投射
        step = 0.1
        length = 0.0
        hit = -1.0
        
        while length <= max_ray_distance:
            pos = sensor_pos + ray_dir * length
            ix, iy, iz = np.round(pos).astype(np.int64)
            
            # 边界检查
            if ix < 0 or ix >= space.shape[0] or iy < 0 or iy >= space.shape[1] or iz < 0 or iz >= space.shape[2]:
                hit = length
                break
                
            # 碰撞检测
            if space[ix, iy, iz] == 1:
                hit = length
                # 添加Fisher信息
                fisher = compute_fisher_nb(hit, 0.0, bot_angle, 90.0)  # offset=0因为超声波是定向的
                copy_global_feature = add_global_feature_3d_nb(
                    copy_global_feature, float(ix), float(iy), float(iz), fisher,
                    global_size, global_res,
                    float(space.shape[0]), float(space.shape[1]), float(space.shape[2])
                )
                break
                
            length += step
        
        # 记录距离（转换为米）
        if hit > 0:
            distances[i] = hit / ppm
        else:
            distances[i] = max_ray_distance / ppm
    
    return distances, copy_global_feature
        
@nb.njit(parallel=True, fastmath=True)
def fast_ray_marching(ppm, bot_angle, gaze_angle, robot_pos, space, global_feature_map, global_size, global_res):
    # list of distance to obstacle, pixel_width*pixel_height
    # print(space.shape)
    copy_global_feature = global_feature_map
    horizontal_fov = np.deg2rad(75.0)
    vertical_fov = np.deg2rad(75.0)
    cam_pixel_width = 64
    cam_pixel_height = 64
    cam_focal = 6
    max_ray_distance = 10 * ppm
    hits = np.full((cam_pixel_height, cam_pixel_width), -1.0, dtype=np.float32)
    # Create camera basis vectors based on view angle
    yaw_rotation = np.deg2rad(bot_angle)
    Rz = np.array([[np.cos(yaw_rotation),  -np.sin(yaw_rotation),   0.0],
                    [np.sin(yaw_rotation),   np.cos(yaw_rotation),   0.0], 
                    [0.0,                    0.0,                    1.0]], dtype=np.float32)
    forward = (Rz @ gaze_angle).astype(np.float32)
    forward /= np.linalg.norm(forward)

    world_up = np.array([0,0,1], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1,0,0], dtype=np.float32)
    
    right /= np.linalg.norm(right)
    up = np.cross(right,forward)
    up /= np.linalg.norm(up)
    
    tan_hori = np.tan(horizontal_fov/2.0)
    tan_vert = np.tan(vertical_fov/2.0)
    for pixel_y in nb.prange(cam_pixel_height):
        ndc_y = (2.0 * (pixel_y + 0.5) / cam_pixel_height - 1.0) * tan_vert
        for pixel_x in nb.prange(cam_pixel_height):
            ndc_x = (2.0 * (pixel_x + 0.5) / cam_pixel_width - 1.0) * tan_hori
            ray_dir = forward * cam_focal + right * ndc_x + up * ndc_y
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # march
            step = 0.2  # tuning
            length = 0.0
            hit = -1.0
            while length <= max_ray_distance:
                pos = robot_pos + ray_dir * length
                ix, iy, iz = np.round(pos).astype(np.int64)
                if ix < 0 or ix >= space.shape[0] or iy < 0 or iy >= space.shape[1] or iz < 0 or iz >= space.shape[2]:
                    hit = length
                    # offset = np.arctan(np.sqrt(abs(pixel_x - 31)**2+abs(pixel_y - 31))/41.72)
                    # fisher = fast_fisher_at(ix, iy, iz, hit, offset, gaze_angle, bot_angle)
                if space[ix, iy, iz] == 1:
                    hit = length
                    offset = np.arctan(np.sqrt(abs(pixel_x - 31)**2+abs(pixel_y - 31)**2)/41.72)
                    # 使用统一的Fisher计算（将gaze_angle转换为角度）
                    gaze_angle_deg = np.rad2deg(np.arccos(np.clip(gaze_angle[2], -1, 1))) % 360.0
                    fisher = compute_fisher_nb(hit, offset, gaze_angle_deg, 90.0)  # 假设FOV=90度
                    copy_global_feature = add_global_feature_3d_nb(
                        global_feature_map, float(ix), float(iy), float(iz), fisher,
                        global_size, global_res,
                        float(space.shape[0]), float(space.shape[1]), float(space.shape[2])
                    )
                    break
                length += step
            hits[pixel_y, pixel_x] = hit
    # print(hits)
    return hits, copy_global_feature

# Fisher计算函数已移至fisher_utils.py
# 这里保留的函数已被compute_fisher_nb和add_global_feature_3d_nb替代
# @nb.njit (parallel = True, nogil = True)
# def ext_local_feature_map(wx, wy, wz, fisher_val, global_feature_map, gfm_size, gfm_res, 
#                     s_width, s_length, s_height, lfm_size, lfm):
#     m = global_feature_map
#     size = gfm_size
#     res = gfm_res
#     half = size // 2

#     rx = int(wx / res + size // 2 - s_width // (2 * res))
#     ry = int(wy / res + size // 2 - s_length // (2 * res))
#     rz = int(wz / res + size // 2 - s_height // (2 * res))

#     gx0, gy0, gz0 = rx - half, ry - half, rz - half
#     gx1, gy1, gz1 = gx0 + lfm_size, gy0 + lfm_size, gz0 + lfm_size

#     sx0 = max(0, -gx0)
#     sy0 = max(0, -gy0)
#     sz0 = max(0, -gz0)
#     sx1 = lfm_size - max(0, gx1 - size)
#     sy1 = lfm_size - max(0, gy1 - size)
#     sz1 = lfm_size - max(0, gz1 - size)

#     Gx0 = max(0, gx0)
#     Gy0 = max(0, gy0)
#     Gz0 = max(0, gz0)
#     Gx1 = min(size, gx1)
#     Gy1 = min(size, gy1)
#     Gz1 = min(size, gz1)

#     lfm.fill(0.0)
#     if sx0 < sx1 and sy0 < sy1 and sz0 < sz1:
#         lfm[sx0:sx1, sy0:sy1, sz0:sz1] = m[Gx0:Gx1, Gy0:Gy1, Gz0:Gz1]
#     return lfm


# 工具函数已从fisher_utils导入
