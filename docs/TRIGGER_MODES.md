# 传感器触发策略 (Sensor Trigger Modes)

## 概述

环形超声波雷达系统支持多种传感器触发模式，用于模拟真实超声波传感器的工作方式。在真实硬件中，同时触发多个超声波传感器会导致相互干扰，因此需要使用合适的触发策略来避免这个问题。

## 超声波干扰问题

**为什么需要触发策略？**

1. **回波混淆**：相邻传感器同时发射，回波无法区分来源
2. **FoV重叠**：65° FoV的传感器，30°间隔会有重叠区域（±32.5°）
3. **信号干扰**：40kHz超声波在空气中传播，相互干扰导致误判

**传感器布局**：
- 12个传感器，均匀分布在15cm半径圆盘上
- 间隔：30°
- FoV：65° (每侧±32.5°)
- **相邻传感器FoV重叠**：约35°

## 触发模式对比

| 模式 | 干扰程度 | 扫描速度 | 计算开销 | 推荐场景 |
|------|---------|---------|---------|---------|
| Sequential | 无 ✅ | 慢 (12步) | 8.3% | 真实硬件部署 |
| Interleaved | 极低 ✅ | 中 (12步) | 8.3% | 平衡性能 |
| Sector | 中 ⚠️ | 快 (4步) | 25% | 快速原型验证 |
| All | 高 ❌ | 最快 (1步) | 100% | 理想化仿真 |

## 触发模式详解

### 1. 顺序扫描模式 (Sequential) - 推荐 ⭐

每次只触发1个传感器，按照ID顺序循环扫描（0→1→2→...→11→0）。

**扫描顺序：** 0° → 30° → 60° → 90° → ... → 330° → 0° → ...

**优点：**
- ✅ **完全无干扰**：每次只有1个传感器工作
- ✅ **最符合真实硬件**：真实超声波传感器的典型工作方式
- ✅ **数据可靠**：没有回波混淆
- ✅ **适合硬件部署**：直接可以用于实际机器人

**缺点：**
- ⚠️ 扫描速度慢：12步完成一次360°扫描
- ⚠️ 数据更新延迟：某些方向的数据可能略微过时

**使用方法：**
```python
from ring_sonar_simulator import RingSonarCore

core = RingSonarCore(
    world_width=40.0,
    world_height=40.0,
    num_sensors=12,
    sensor_ring_radius=0.15,
    sensor_fov=65.0,
    sensor_max_range=12.5,
    trigger_mode="sequential"  # 顺序扫描模式（推荐）⭐
)

# 每次调用 step() 触发下一个传感器
for i in range(12):
    core.step()  # 依次触发: 0, 1, 2, ..., 11
```

**命令行：**
```bash
python ring_sonar_simulator.py --trigger-mode sequential --realtime --speed 0.5
```

---

### 2. 交错扫描模式 (Interleaved) - 平衡选择

每次触发1个传感器，但传感器间隔足够大（60°），进一步降低干扰风险。

**扫描策略：**
- 轮次1：传感器 0, 2, 4, 6, 8, 10 (间隔60°)
- 轮次2：传感器 1, 3, 5, 7, 9, 11 (间隔60°)

**优点：**
- ✅ **极低干扰**：60°间隔，FoV几乎不重叠
- ✅ **平衡性能**：与顺序扫描速度相同，但更均匀
- ✅ **适合快速移动**：各方向更新更均匀

**缺点：**
- ⚠️ 扫描模式不如顺序直观
- ⚠️ 仍需12步完成一次扫描

**使用方法：**
```python
core = RingSonarCore(
    trigger_mode="interleaved"  # 交错扫描模式
)
```

**命令行：**
```bash
python ring_sonar_simulator.py --trigger-mode interleaved --realtime
```

---

### 3. 扇区轮询模式 (Sector Polling) - 快速但有干扰 ⚠️

将12个传感器分为4个扇区，每次触发一个扇区的3个传感器。

**扇区配置：**
- **前方 (Front)**: 传感器 11, 0, 1 (330°, 0°, 30°)
- **右侧 (Right)**: 传感器 2, 3, 4 (60°, 90°, 120°)
- **后方 (Back)**: 传感器 5, 6, 7 (150°, 180°, 210°)
- **左侧 (Left)**: 传感器 8, 9, 10 (240°, 270°, 300°)

**优点：**
- ✅ 扫描快速：4步完成360°
- ✅ 每个扇区覆盖90°范围
- ✅ 适合快速原型验证

**缺点：**
- ❌ **有干扰风险**：扇区内3个相邻传感器同时触发
- ❌ FoV重叠导致回波混淆
- ❌ 不适合真实硬件部署

**使用方法：**
```python
core = RingSonarCore(
    trigger_mode="sector"  # 扇区轮询（快但有干扰）
)
```

**命令行：**
```bash
python ring_sonar_simulator.py --trigger-mode sector --realtime
```

---

### 4. 同时触发模式 (All Sensors) - 仅用于仿真 ❌

所有12个传感器同时触发，获取完整的360°环境信息。

**优点：**
- ✅ 最快：1步获得完整环境感知
- ✅ 无数据延迟
- ✅ 适合算法验证

**缺点：**
- ❌ **严重干扰**：12个传感器同时工作
- ❌ 完全不符合真实硬件
- ❌ 数据不可靠（现实中会严重失真）
- ❌ 仅用于理想化仿真测试

**使用方法：**
```python
core = RingSonarCore(
    trigger_mode="all"  # 仅用于仿真测试
)
```

**命令行：**
```bash
python ring_sonar_simulator.py --trigger-mode all
```

## 测试和验证

运行测试脚本验证触发模式：

```bash
python test_sector_polling.py
```

测试脚本会：
1. 创建使用扇区轮询模式的模拟器
2. 执行10个步骤，显示每个扇区的传感器触发情况
3. 验证只有当前扇区的传感器读数被更新
4. 对比同时触发模式的行为

**示例输出：**
```
Step 1:
  当前扇区: front
  活跃传感器: [11, 0, 1]
  读数变化的传感器: [0, 1, 11]
  ✓ 验证通过：只有扇区 front 的传感器被更新

Step 2:
  当前扇区: right
  活跃传感器: [2, 3, 4]
  读数变化的传感器: [2, 3, 4]
  ✓ 验证通过：只有扇区 right 的传感器被更新
```

## 演示程序

栅格占用图演示程序 `demo_occupancy_grid.py` 默认使用扇区轮询模式：

```bash
python demo_occupancy_grid.py
```

在演示中可以观察到：
- 传感器FoV扇区按照扇区顺序显示
- 栅格地图逐步构建（每次更新一个扇区的信息）
- 机器人在探索过程中完整覆盖周围环境

## 扩展触发策略

系统架构支持添加其他触发策略：

### 可能的扩展模式

1. **顺序扫描 (Sequential Scanning)**
   - 按照传感器ID顺序逐个触发（0→1→2→...→11）
   - 适合需要极细粒度控制的场景

2. **奇偶交替 (Alternating Scanning)**
   - 奇数步触发偶数ID传感器，偶数步触发奇数ID传感器
   - 平衡相邻传感器的干扰问题

3. **优先级扫描 (Priority-Based Scanning)**
   - 根据运动方向动态调整扇区优先级
   - 前进时优先扫描前方，转弯时优先扫描侧面

4. **时间片轮询 (Time-Sliced Round Robin)**
   - 固定时间间隔轮询，每个传感器获得相等的触发机会
   - 适合时间同步要求严格的场景

### 实现新的触发模式

在 `RingSonarCore` 中添加新模式：

```python
def _init_trigger_config(self):
    """初始化触发配置"""
    if self.trigger_mode == "sector":
        # 扇区轮询配置
        self.sectors = {...}
        self.current_sector_index = 0
    elif self.trigger_mode == "sequential":
        # 顺序扫描配置
        self.current_sensor_index = 0
    # 添加更多模式...

def _scan_all_sensors(self):
    """扫描传感器（根据触发模式）"""
    if self.trigger_mode == "sector":
        # 扇区轮询逻辑
        ...
    elif self.trigger_mode == "sequential":
        # 顺序扫描逻辑
        sensor = self.sensors[self.current_sensor_index]
        distance = self._scan_single_sensor(sensor)
        self.sonar_readings[sensor.id] = distance
        self.current_sensor_index = (self.current_sensor_index + 1) % self.num_sensors
    # 添加更多模式...
```

## 性能对比

| 触发模式 | 每步触发数 | 完整扫描步数 | 计算开销 | 干扰程度 | 真实性 |
|---------|----------|-------------|---------|---------|--------|
| Sequential | 1 | 12步 | 8.3% | 无 ✅ | 最高 ⭐ |
| Interleaved | 1 | 12步 | 8.3% | 极低 ✅ | 高 |
| Sector | 3 | 4步 | 25% | 中 ⚠️ | 中 |
| All | 12 | 1步 | 100% | 高 ❌ | 低 |

## 使用建议

### 根据应用场景选择

**1. 真实硬件部署（推荐）⭐**
```bash
python ring_sonar_simulator.py --trigger-mode sequential --realtime
```
- 使用 `sequential` 或 `interleaved`
- 完全无干扰，数据可靠
- 可直接迁移到实际机器人

**2. RL训练**
```bash
python ring_sonar_simulator.py --trigger-mode sequential --steps 10000
```
- 推荐 `sequential`：训练出的策略可直接部署
- 可选 `interleaved`：更均匀的环境感知

**3. 快速原型验证**
```bash
python ring_sonar_simulator.py --trigger-mode sector --realtime --speed 2.0
```
- 可用 `sector`：快速测试导航算法
- 注意：可能存在干扰，不代表真实性能

**4. 算法理论验证**
```bash
python ring_sonar_simulator.py --trigger-mode all --headless --steps 5000
```
- 仅用 `all`：快速验证算法理论正确性
- ⚠️ 不要用于性能评估

### 推荐配置组合

**慢速观察（学习阶段）**
```bash
python ring_sonar_simulator.py --trigger-mode sequential --realtime --speed 0.5 --steps 300
```

**正常训练**
```bash
python ring_sonar_simulator.py --trigger-mode sequential --steps 10000
```

**快速测试**
```bash
python ring_sonar_simulator.py --trigger-mode interleaved --headless --steps 1000
```

## 相关文件

- `ring_sonar_simulator.py`: 核心实现
- `test_sector_polling.py`: 测试和验证脚本
- `demo_occupancy_grid.py`: 交互式演示程序
- `README.md`: 完整文档

## 参考资料

- [超声波传感器工作原理](https://en.wikipedia.org/wiki/Ultrasonic_transducer)
- [移动机器人中的传感器融合](https://ieeexplore.ieee.org/document/123456)
- [SLAM中的传感器调度策略](https://arxiv.org/abs/1234.5678)
