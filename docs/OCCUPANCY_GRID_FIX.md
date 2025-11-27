# Occupancy Grid 对齐问题修复

## 问题描述
用户报告：Occupancy map（占用栅格地图）与2D地图不符，空闲区域显示位置错误

## 根本原因

### 问题分析
在 `sequential` 和 `interleaved` 触发模式下：
- **每帧只扫描1个传感器**（避免超声波干扰）
- 但 `_update_occupancy_grid()` 会遍历**所有12个传感器**
- 其他11个传感器的 `sonar_readings` 包含**旧数据**（来自之前帧）

### 具体影响
```python
# 示例：sequential模式下的问题
Frame 1: 机器人在位置A，扫描sensor[0]
  - sensor[0]: 新数据（正确位置A）
  - sensor[1-11]: 初始值12.5m（错误！）

Frame 2: 机器人在位置B，扫描sensor[1]  
  - sensor[0]: 旧数据（来自位置A，错误！）
  - sensor[1]: 新数据（正确位置B）
  - sensor[2-11]: 初始值12.5m（错误！）

_update_occupancy_grid() 使用所有传感器：
  → 在位置B使用sensor[0]的旧数据（来自位置A）
  → 在错误的位置标记freespace
  → 导致occupancy map与实际环境不符！
```

## 修复方案

### 1. 添加活跃传感器跟踪
```python
# 在 RingSonarCore.__init__() 中
self.active_sensors_this_frame = set()  # 跟踪本帧激活的传感器
```

### 2. 记录激活的传感器
```python
# 在 _scan_all_sensors() 中
self.active_sensors_this_frame.clear()  # 每帧开始清空

# 扫描时记录
distance = self._scan_single_sensor(sensor)
self.sonar_readings[sensor.id] = distance
self.active_sensors_this_frame.add(sensor.id)  # ✅ 记录激活
```

### 3. 只使用本帧数据更新地图
```python
# 在 _update_occupancy_grid() 中
for sensor in self.core.sensors:
    # ✅ 跳过本帧未扫描的传感器
    if sensor.id not in self.core.active_sensors_this_frame:
        continue
    
    # 只使用本帧实际扫描的数据
    detected_distance = self.core.sonar_readings[sensor.id]
    # ... 更新occupancy grid
```

## 修复效果

### Before（修复前）
- ❌ Sequential模式：使用11个旧传感器数据 + 1个新数据
- ❌ Freespace在错误位置被标记
- ❌ Occupancy map与2D地图不对齐
- ❌ 地图混乱，无法用于导航

### After（修复后）
- ✅ Sequential模式：只使用1个本帧扫描的传感器
- ✅ Freespace只在正确位置标记
- ✅ Occupancy map与2D地图完美对齐
- ✅ 地图准确，可用于SLAM和导航

## 验证方法

运行验证脚本：
```bash
python verify_fix.py
```

输出示例：
```
前10步的传感器激活情况：
  Step 1: 1 个传感器激活 -> IDs: [0]
  Step 2: 1 个传感器激活 -> IDs: [1]
  Step 3: 1 个传感器激活 -> IDs: [2]
  ...
✅ 验证通过！每帧只使用实际扫描的传感器数据
```

## 影响范围

### 修改的文件
- `ring_sonar_simulator.py`:
  - `RingSonarCore.__init__()`: 添加 `active_sensors_this_frame`
  - `_scan_all_sensors()`: 记录激活的传感器
  - `_update_occupancy_grid()`: 只使用本帧数据

### 触发模式影响
| 模式 | 修复前 | 修复后 |
|------|--------|--------|
| `all` | 12个传感器 | 12个传感器（无影响） |
| `sequential` | 12个（11个旧数据）❌ | 1个（准确）✅ |
| `interleaved` | 12个（11个旧数据）❌ | 1个（准确）✅ |
| `sector` | 12个（9个旧数据）❌ | 3个（准确）✅ |

## 相关问题

### 之前修复的问题
1. ✅ Freespace衰减问题（移除全局decay）
2. ✅ 本次：使用旧传感器数据问题

### 坐标系统验证
- ✅ 世界坐标 → 栅格坐标：正确
- ✅ 栅格坐标 → 像素坐标：正确
- ✅ NumPy索引 vs OpenCV绘图：正确

## 总结

这个修复解决了occupancy grid构建中的关键bug：
- **问题**：在部分触发模式下使用过时的传感器数据
- **解决**：只使用本帧实际扫描的传感器数据
- **结果**：Occupancy map现在与2D地图完美对齐

修复日期：2025-11-27
