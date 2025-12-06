# GYMgaze 项目结构

```
GYMgaze/
├── README.md                      # 项目主说明文档
├── PROJECT_STRUCTURE.md           # 项目结构说明（本文件）
├── requirements.txt               # 基础依赖
├── .gitignore                     # Git忽略规则
│
├── configs/                       # 配置模块
│   ├── __init__.py               # 导出所有配置
│   └── robot_config.py           # 机器人物理参数配置
│       ├── RobotPhysicsConfig    # 机器人物理配置
│       ├── SensorConfig          # 传感器配置
│       ├── WorldConfig           # 世界配置
│       ├── SimulationConfig      # 组合配置
│       ├── DEFAULT_CONFIG        # 默认配置（训练/推理）
│       ├── DEMO_CONFIG           # 演示配置（慢速）
│       └── DATA_COLLECTION_CONFIG # 数据收集配置
│
├── src/                           # 核心源码模块
│   ├── __init__.py               # 顶层导出
│   │
│   ├── models/                   # 深度学习模型
│   │   ├── __init__.py
│   │   └── global_map.py         # GlobalMapPredictor U-Net模型
│   │
│   ├── simulator/                # 仿真组件
│   │   ├── __init__.py
│   │   └── sensors.py            # SonarSensor 数据类
│   │
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── geometry.py           # 几何工具（角度处理、特征更新）
│       └── fisher.py             # SonarFisherCalculator
│
├── scripts/                       # 脚本入口（预留）
│   └── (entry point scripts)
│
├── data/                          # 数据存储（gitignore）
│   └── global_map_training_data/ # 训练数据
│
├── checkpoints/                   # 模型存储（gitignore）
│   ├── global_map_model.pth      # 训练好的模型
│   └── global_map_training_history.png  # 训练历史图
│
├── docs/                          # 文档
│   ├── MOVEMENT_INDICATOR.md
│   ├── OBSTACLE_PREDICTION.md
│   ├── OCCUPANCY_GRID_FIX.md
│   ├── OPTIMIZATION_LOG.md
│   ├── SONAR_FISHER_REDESIGN.md
│   ├── TRAINING_GUIDE.md
│   └── TRIGGER_MODES.md
│
├── gymnasium_env/                 # Gymnasium环境（独立包）
│   ├── setup.py
│   └── env_tmp/
│       └── gymnasium_env_gaze/
│           ├── envs/
│           │   ├── gaze_env.py
│           │   └── ring_sonar_env.py
│           └── wrappers/
│
│  # ============== 主要脚本 ==============
│
├── ring_sonar_simulator.py        # 主仿真器（RingSonarCore, RingSonarRenderer）
├── collect_global_map_data.py     # 数据收集脚本
├── train_global_map_model.py      # 模型训练脚本
│
│  # ============== 演示脚本 ==============
│
├── demo_ring_sonar.py             # 环形声纳演示
├── demo_global_map_prediction.py  # 全局地图预测演示
│
│  # ============== 旧版兼容（待清理）==============
│
├── fisher_utils.py                # 旧版Fisher工具（已迁移到src/utils）
├── sonar_fisher_calculator.py     # 旧版Fisher计算器（已迁移到src/utils）
├── collect_training_data.py       # 旧版数据收集
└── train_model.py                 # 旧版训练脚本
```

## 配置使用

```python
from configs import DEFAULT_CONFIG, DEMO_CONFIG, print_config

# 使用默认配置
print_config(DEFAULT_CONFIG)

# 创建核心仿真器
from ring_sonar_simulator import RingSonarCore
core = RingSonarCore(config=DEFAULT_CONFIG)
```

## 模块导入

```python
# 从src模块导入
from src.models import GlobalMapPredictor
from src.utils import SonarFisherCalculator, clamp, angnorm_deg
from src.simulator import SonarSensor

# 从configs导入
from configs import SimulationConfig, DEFAULT_CONFIG
```

## 数据流

1. **数据收集**: `python collect_global_map_data.py --episodes 100`
   - 输出: `data/global_map_training_data/`

2. **模型训练**: `python train_global_map_model.py --epochs 100`
   - 输入: `data/global_map_training_data/`
   - 输出: `checkpoints/global_map_model.pth`

3. **运行仿真**: `python ring_sonar_simulator.py --demo-mode`
   - 加载: `checkpoints/global_map_model.pth`
