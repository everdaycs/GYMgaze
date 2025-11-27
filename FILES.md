# 项目文件结构

## 📁 核心代码

| 文件 | 说明 |
|------|------|
| **ring_sonar_simulator.py** ⭐ | 主模拟器：12传感器SLAM、障碍物预测、Fisher地图 |
| **sonar_fisher_calculator.py** | Fisher信息计算（超声波优化版） |
| **fisher_utils.py** | 通用工具函数 |
| **run.py** | 快速启动脚本 |

---

## 🧪 演示和测试

| 文件 | 说明 |
|------|------|
| **demo_obstacle_prediction.py** ⭐ | 障碍物预测演示 |
| **demo_occupancy_grid.py** | 占用栅格地图演示 |
| **demo_ring_sonar.py** | 基础功能演示 |
| **test_ring_sonar.py** | 单元测试 |
| **test_sector_polling.py** | 触发模式测试 |

---

## 🤖 机器学习（可选）

| 文件 | 说明 |
|------|------|
| **train_diffusion_model.py** | 神经网络训练脚本（需PyTorch）|
| **requirements_training.txt** | 训练依赖 |

---

## 📚 文档

### 根目录
- **README.md** - 项目概述和快速开始
- **FILES.md** - 本文件（项目导航）

### docs/ 目录

| 文档 | 说明 |
|------|------|
| **OBSTACLE_PREDICTION.md** ⭐ | 障碍物预测扩散模型算法详解 |
| **TRIGGER_MODES.md** | 传感器触发模式策略 |
| **SONAR_FISHER_REDESIGN.md** | Fisher信息计算优化 |
| **OCCUPANCY_GRID_FIX.md** | 占用地图对齐修复说明 |
| **MOVEMENT_INDICATOR.md** | 移动方向显示功能 |
| **OPTIMIZATION_LOG.md** | 代码优化记录 |
| **TRAINING_GUIDE.md** | 神经网络训练指南（高级）|

---

## 🎯 快速开始

```bash
# 基础演示
python ring_sonar_simulator.py --steps 200 --realtime

# 障碍物预测演示（推荐）
python demo_obstacle_prediction.py

# 自定义参数
python ring_sonar_simulator.py \
    --steps 500 \
    --trigger-mode sequential \
    --speed 1.0
```

---

## 📂 完整目录结构

```
gymGaze/
├── ring_sonar_simulator.py      ⭐ 主模拟器
├── sonar_fisher_calculator.py      Fisher计算
├── fisher_utils.py                 工具函数
├── run.py                          启动脚本
├── demo_obstacle_prediction.py  ⭐ 障碍物预测演示
├── demo_occupancy_grid.py          占用地图演示
├── demo_ring_sonar.py              基础演示
├── test_ring_sonar.py              单元测试
├── test_sector_polling.py          触发测试
├── train_diffusion_model.py        神经网络训练
├── requirements_training.txt       训练依赖
├── README.md                       项目概述
├── FILES.md                        本文件
├── docs/                        📚 技术文档
│   ├── OBSTACLE_PREDICTION.md   ⭐ 算法详解
│   ├── TRIGGER_MODES.md            触发策略
│   ├── SONAR_FISHER_REDESIGN.md    Fisher优化
│   ├── OCCUPANCY_GRID_FIX.md       地图修复
│   ├── MOVEMENT_INDICATOR.md       方向显示
│   ├── OPTIMIZATION_LOG.md         优化记录
│   └── TRAINING_GUIDE.md           训练指南
└── gymnasium_env/                  强化学习环境

总计：11个核心代码 + 7个文档 = 18个文件
```  
