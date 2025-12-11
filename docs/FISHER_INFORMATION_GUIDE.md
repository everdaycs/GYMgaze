# Fisher 信息设置完整指南

## 📍 Fisher 信息的设置位置

Fisher 信息在系统中有两个主要设置位置：

### 1️⃣ **Fisher 计算器参数** (`src/utils/fisher.py`)

这是 Fisher 信息的核心计算模块。

#### A. 全局常量定义（第 20-31 行）

```python
DISTANCE_SCALE = 12.5           # 最大探测距离（米）
MIN_DISTANCE_FACTOR = 0.1       # 最小距离因子（避免近距离读数为0）
DISTANCE_DECAY_POWER = 1.5      # 距离衰减指数（控制距离对Fisher值的影响）

MAX_COVERAGE_BONUS = 3.0        # 最大覆盖度加成（倍数）
COVERAGE_SIGMA = 45.0           # 覆盖度计算的sigma（高斯分布）

MIN_FISHER_VALUE = 0.1          # Fisher值下界
MAX_FISHER_VALUE = 10.0         # Fisher值上界
```

**关键参数解释：**
- `DISTANCE_DECAY_POWER = 1.5`：距离越远，Fisher值衰减越快
  - 如果改为 1.0：线性衰减
  - 如果改为 2.0：二次方衰减（衰减更快）
  
- `MAX_COVERAGE_BONUS = 3.0`：多传感器覆盖时的增益
  - 值越大，重叠区域Fisher值越高
  - 当前设置：重叠区域可获得最多 3 倍的增益

#### B. 类初始化参数（`SonarFisherCalculator.__init__`，第 51-61 行）

```python
def __init__(self,
             num_sensors: int = 12,
             sensor_spacing: float = 30.0,    # 360/12 = 30度间隔
             sensor_fov: float = 65.0,        # 每个传感器65度视场
             max_range: float = 12.5):        # 最大探测距离12.5米
```

这些参数定义了传感器阵列的几何配置。

#### C. 主要计算方法（`compute` 方法，第 75-97 行）

```python
def compute(self, distance: float, angle_deg: float) -> float:
    """
    Fisher信息计算公式：
    Fisher = Distance_Factor × Coverage_Factor
    """
    # 距离因子计算
    normalized_dist = distance / self.max_range
    dist_factor = (1.0 - normalized_dist) ** DISTANCE_DECAY_POWER
    dist_factor = max(dist_factor, MIN_DISTANCE_FACTOR)
    
    # 覆盖度因子计算
    coverage_count = self._compute_coverage(angle_deg)
    coverage_factor = 1.0 + (MAX_COVERAGE_BONUS - 1.0) * \
                      (coverage_count - 1) / max(1, self.max_overlap_count - 1)
    
    # 最终Fisher值
    fisher = dist_factor * coverage_factor
    return clamp(fisher, MIN_FISHER_VALUE, MAX_FISHER_VALUE)
```

---

### 2️⃣ **模拟器中的 Fisher 集成** (`ring_sonar_simulator.py`)

#### A. Fisher 计算器初始化（第 165-171 行）

```python
self.fisher_calc = SonarFisherCalculator(
    num_sensors=self.num_sensors,
    sensor_spacing=360.0 / self.num_sensors,
    sensor_fov=self.sensor_fov,
    max_range=self.sensor_max_range
)
```

Fisher 计算器在模拟器初始化时创建，使用与传感器配置一致的参数。

#### B. Fisher 地图数据结构（第 111-120 行）

```python
# 局部特征地图（以机器人为中心）
self.feature_map = np.zeros((self.feature_map_size, self.feature_map_size), 
                           dtype=np.float32)

# 全局特征地图
self.global_feature_map = np.zeros((self.global_feature_map_size, 
                                   self.global_feature_map_size), 
                                  dtype=np.float32)
```

- `feature_map_size = 100`：100×100 的局部地图（分辨率 0.25m/格）
- `global_feature_map_size = 1600`：对应 40×40m 世界的全局地图（400 格 × 0.1m = 40m）

#### C. Fisher 信息计算流程（第 533-562 行）

```python
def update_maps(self) -> None:
    """更新Fisher信息地图的流程"""
    # 步骤1：特征衰减
    self._apply_feature_decay()
    
    # 步骤2：从传感器读数检测特征
    self._detect_and_add_features_to_global_map()
    
    # 步骤3：提取局部地图
    self._extract_local_feature_map()
```

核心计算方法：

```python
def _fisher_at(self, wx: float, wy: float, distance: float, angle_deg: float) -> float:
    """计算特定世界位置的Fisher信息值"""
    # 转换为相对机器人的角度
    relative_angle = (angle_deg - self.robot_angle) % 360.0
    
    # 调用Fisher计算器
    return self.fisher_calc.compute(
        distance=distance,
        angle_deg=relative_angle
    )
```

#### D. 特征衰减设置（第 537-540 行）

```python
def _apply_feature_decay(self):
    """应用特征衰减"""
    self.global_feature_map *= (1.0 - 5e-6)  # 每步衰减 5e-6
    self.global_feature_map[self.global_feature_map < 0.1] = 0.0  # 阈值清零
```

- 衰减系数 `5e-6`：控制信息遗忘速度
  - 更小的值：信息保留时间更长
  - 更大的值：信息快速衰减

---

## 🔧 如何修改 Fisher 信息

### 修改方案 1：调整距离敏感度

**目标：** 让远处的障碍物更有价值

```python
# 在 src/utils/fisher.py 第 24 行修改：
DISTANCE_DECAY_POWER = 1.0  # 从 1.5 改为 1.0（线性衰减，更缓和）
```

**效果对比：**

| 参数值 | 3m 处值 | 6m 处值 | 9m 处值 | 12.5m 处值 |
|--------|--------|--------|--------|-----------|
| 1.0    | 0.76   | 0.52   | 0.28   | 0.00      |
| 1.5    | 0.66   | 0.35   | 0.09   | 0.00      |
| 2.0    | 0.58   | 0.22   | 0.02   | 0.00      |

### 修改方案 2：调整覆盖度加成

**目标：** 增加多传感器重叠区域的价值

```python
# 在 src/utils/fisher.py 第 26 行修改：
MAX_COVERAGE_BONUS = 5.0  # 从 3.0 改为 5.0（重叠区域最多获得 5 倍增益）
```

### 修改方案 3：调整 Fisher 值范围

**目标：** 放大或压缩 Fisher 值的范围

```python
# 在 src/utils/fisher.py 第 29-30 行修改：
MIN_FISHER_VALUE = 0.05   # 从 0.1 改为 0.05（降低下界）
MAX_FISHER_VALUE = 20.0   # 从 10.0 改为 20.0（提高上界）
```

### 修改方案 4：调整特征衰减速度

**目标：** 控制地图记忆时长

```python
# 在 ring_sonar_simulator.py 第 539 行修改：
self.global_feature_map *= (1.0 - 1e-5)  # 从 5e-6 改为 1e-5（衰减更快）
```

**效果：** 衰减系数越大，信息保留时间越短

---

## 📊 Fisher 信息计算流程图

```
传感器读数 (distance, angle)
         ↓
    ┌────────────────────┐
    │  距离因子计算      │
    │ dist_factor =      │
    │ (1 - d/max_d)^1.5  │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │  覆盖度计算        │
    │ 统计有多少传感器   │
    │ 的FOV覆盖该角度    │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │  覆盖度因子计算    │
    │ coverage_factor =  │
    │ 1 + (count-1)*k    │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │  Fisher值计算      │
    │ fisher =           │
    │ dist_factor ×      │
    │ coverage_factor    │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │  值范围限制        │
    │ [MIN, MAX] =       │
    │ [0.1, 10.0]        │
    └────────────────────┘
         ↓
    最终 Fisher 值
```

---

## 🎯 实验建议

### 实验 1：测试距离敏感度的影响

```bash
# 运行模拟器并观察不同距离的特征值
python ring_sonar_simulator.py --trigger-mode all --headless --steps 500
```

观察 `发现特征: XXX` 和 `平均Fisher值: X.XXX` 的变化。

### 实验 2：比较不同的衰减速率

```python
# 修改 ring_sonar_simulator.py 第 539 行
# 尝试以下衰减系数：
# 1e-6, 5e-6, 1e-5, 5e-5

self.global_feature_map *= (1.0 - 1e-6)  # 保留最长
self.global_feature_map *= (1.0 - 5e-6)  # 默认
self.global_feature_map *= (1.0 - 1e-5)  # 快速衰减
```

### 实验 3：观察覆盖度对 Fisher 值的影响

```python
# 在 ring_sonar_simulator.py 中添加调试输出
def _fisher_at(self, wx, wy, distance, angle_deg):
    relative_angle = (angle_deg - self.robot_angle) % 360.0
    coverage = self.fisher_calc._compute_coverage(relative_angle)
    fisher = self.fisher_calc.compute(distance, relative_angle)
    
    print(f"Distance: {distance:.2f}m, Angle: {relative_angle:.1f}°, "
          f"Coverage: {coverage}, Fisher: {fisher:.2f}")
    
    return fisher
```

---

## 📋 总结

| 设置项 | 文件 | 位置 | 默认值 | 含义 |
|--------|------|------|--------|------|
| 距离衰减指数 | fisher.py | 第 24 行 | 1.5 | 控制距离对Fisher的影响 |
| 最大覆盖度加成 | fisher.py | 第 26 行 | 3.0 | 重叠区域增益倍数 |
| Fisher值下界 | fisher.py | 第 29 行 | 0.1 | 最小有效值 |
| Fisher值上界 | fisher.py | 第 30 行 | 10.0 | 最大有效值 |
| 特征衰减系数 | simulator.py | 第 539 行 | 5e-6 | 每步衰减速率 |
| 衰减阈值 | simulator.py | 第 540 行 | 0.1 | 清零的最小值 |

所有这些参数都可以根据实验需要进行调整！
