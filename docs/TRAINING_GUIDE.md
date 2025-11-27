# 障碍物预测扩散模型训练指南

## 概述

本指南介绍如何从**基于规则的扩散模型**升级到**学习型神经网络模型**。

## 两种方法对比

### 当前方法：基于规则（Rule-Based）

**优点：**
- ✅ 无需训练数据
- ✅ 计算高效（5-10ms/帧）
- ✅ 可解释性强
- ✅ 即插即用

**缺点：**
- ❌ 规则固定，无法适应不同环境
- ❌ 参数手动调优
- ❌ 预测精度受限于规则设计

**实现：** `RingSonarRenderer._predict_obstacles()`

---

### 学习方法：神经网络（Neural Network）

**优点：**
- ✅ 自动学习复杂模式
- ✅ 可适应不同环境
- ✅ 预测精度更高（理论上）
- ✅ 端到端优化

**缺点：**
- ❌ 需要大量训练数据
- ❌ 计算开销更大
- ❌ 需要GPU加速
- ❌ 训练和调优复杂

**实现：** `train_diffusion_model.py`

---

## 训练流程

### 1️⃣ 环境准备

```bash
# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install tqdm matplotlib scikit-learn
```

### 2️⃣ 数据收集

运行模拟器收集训练数据：

```bash
# 收集100个episode（约1500+个样本）
python train_diffusion_model.py --mode collect --episodes 100

# 收集更多数据以提高性能
python train_diffusion_model.py --mode collect --episodes 500
```

**数据结构：**
```
每个样本包含：
- 输入：occupancy_grid (已知空闲空间)
        visit_count (探索置信度)
        known_mask (已知/未知区域)
- 输出：ground_truth_obstacles (真实障碍物位置)
```

**数据增强策略：**
- 不同随机种子 → 不同环境布局
- 不同探索阶段 → 不同占用率快照
- 旋转/翻转（可选）

### 3️⃣ 模型训练

```bash
# 基础训练（50轮）
python train_diffusion_model.py --mode train --epochs 50

# 调整超参数
python train_diffusion_model.py --mode train \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

**训练监控：**
- 训练损失（Train Loss）
- 验证损失（Val Loss）
- 学习率衰减
- 最佳模型保存

### 4️⃣ 模型评估

```bash
python train_diffusion_model.py --mode evaluate
```

**评估指标：**
- **准确率（Accuracy）**: 正确预测的栅格占比
- **精确率（Precision）**: 预测为障碍物中真正是障碍物的比例
- **召回率（Recall）**: 真实障碍物中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **IoU**: 预测与真实障碍物的交并比

### 5️⃣ 模型部署

将训练好的模型集成到模拟器：

```python
# 在 RingSonarRenderer 中添加
class RingSonarRenderer:
    def __init__(self, ...):
        # 加载训练好的模型
        self.learned_model = torch.load('obstacle_diffusion_model.pth')
        self.use_learned_model = True  # 切换为学习模型
    
    def _predict_obstacles(self):
        if self.use_learned_model:
            return self._predict_obstacles_neural()
        else:
            return self._predict_obstacles_rule_based()
```

---

## 网络架构

### U-Net 架构

```
输入: [B, 3, H, W]
  ├─ 通道0: occupancy (空闲空间)
  ├─ 通道1: visit_count (探索置信度)
  └─ 通道2: known_mask (已知区域)

编码器 (Encoder):
  ├─ Conv Block 1: [3, 64]   + MaxPool
  ├─ Conv Block 2: [64, 128] + MaxPool
  └─ Conv Block 3: [128, 256] + MaxPool

瓶颈 (Bottleneck):
  └─ Conv Block: [256, 512]

解码器 (Decoder):
  ├─ UpConv + Skip + Conv Block: [512, 256]
  ├─ UpConv + Skip + Conv Block: [256, 128]
  └─ UpConv + Skip + Conv Block: [128, 64]

输出: [B, 1, H, W]
  └─ Sigmoid → 障碍物概率 [0, 1]
```

**关键设计：**
- **跳跃连接（Skip Connections）**: 保留细节信息
- **批归一化（Batch Normalization）**: 加速训练
- **Sigmoid激活**: 输出概率分布

---

## 损失函数

### 二元交叉熵损失（BCE Loss）

```python
loss = -[y * log(p) + (1-y) * log(1-p)]
```

**带掩码的损失：**
```python
# 只计算未知区域的损失
mask = (target >= 0)  # 已知区域标记为-1
loss = BCE(pred, target) * mask
loss = loss.sum() / mask.sum()
```

**改进方向：**
1. **Focal Loss**: 关注难样本
2. **Dice Loss**: 处理类别不平衡
3. **组合损失**: BCE + Dice

---

## 超参数调优

### 关键超参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| 学习率 | 1e-3 | [1e-4, 1e-2] | 过大→震荡，过小→收敛慢 |
| 批大小 | 16 | [8, 64] | 显存允许的最大值 |
| 基础通道数 | 64 | [32, 128] | 影响模型容量 |
| 训练轮数 | 50 | [30, 200] | 观察收敛情况 |

### 学习率策略

```python
# ReduceLROnPlateau: 验证损失停滞时降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # 监控最小值
    patience=5,      # 5轮不改善后降低
    factor=0.5       # 降低50%
)
```

---

## 训练技巧

### 1. 数据平衡

**问题**: 障碍物栅格远少于空闲栅格（类别不平衡）

**解决方案**:
- 加权损失: `pos_weight` 参数
- 过采样障碍物边界区域
- 使用Focal Loss

### 2. 数据增强

```python
# 旋转和翻转
def augment(occupancy, target):
    # 随机旋转 90°/180°/270°
    k = np.random.randint(0, 4)
    occupancy = np.rot90(occupancy, k)
    target = np.rot90(target, k)
    
    # 随机翻转
    if np.random.rand() > 0.5:
        occupancy = np.fliplr(occupancy)
        target = np.fliplr(target)
    
    return occupancy, target
```

### 3. 早停（Early Stopping）

```python
# 验证损失不再改善时停止训练
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= max_patience:
        print("Early stopping")
        break
else:
    patience_counter = 0
    best_val_loss = val_loss
```

### 4. 模型集成

训练多个模型，集成预测：

```python
predictions = []
for model in models:
    pred = model(input)
    predictions.append(pred)

# 平均预测
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

---

## 性能优化

### GPU加速

```python
# 使用CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 混合精度训练（更快）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 推理加速

```python
# 模型量化（减少内存和计算）
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# TorchScript编译（JIT加速）
scripted_model = torch.jit.script(model)
scripted_model.save('model_optimized.pt')
```

---

## 评估和可视化

### 定量评估

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 二值化预测
pred_binary = (pred > 0.5).float()
target_binary = (target > 0.5).float()

# 计算指标
accuracy = accuracy_score(target_binary.flatten(), 
                          pred_binary.flatten())
precision, recall, f1, _ = precision_recall_fscore_support(
    target_binary.flatten(), 
    pred_binary.flatten(), 
    average='binary'
)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

### 定性评估

```python
# 可视化对比
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(occupancy, cmap='gray')
axes[0].set_title('Input: Occupancy Grid')

axes[1].imshow(ground_truth, cmap='hot')
axes[1].set_title('Ground Truth')

axes[2].imshow(rule_based_pred, cmap='hot')
axes[2].set_title('Rule-Based Prediction')

axes[3].imshow(neural_pred, cmap='hot')
axes[3].set_title('Neural Network Prediction')

plt.tight_layout()
plt.savefig('comparison.png')
```

---

## 实验建议

### 阶段1: 基线实验

1. 收集100 episodes数据
2. 训练基础U-Net (50 epochs)
3. 评估基线性能

### 阶段2: 改进实验

**数据方面：**
- 增加到500 episodes
- 尝试数据增强

**模型方面：**
- 调整网络深度（更多/更少层）
- 调整通道数（32/64/128）
- 尝试其他架构（ResNet, Transformer）

**训练方面：**
- 调整学习率和批大小
- 尝试不同优化器（Adam, SGD, AdamW）
- 使用不同损失函数

### 阶段3: 部署实验

1. 集成到模拟器
2. 实时性能测试
3. 对比规则模型

---

## 常见问题

### Q1: 训练损失不下降？

**可能原因：**
- 学习率过大 → 降低学习率
- 数据问题 → 检查数据质量
- 模型过小 → 增加网络容量

### Q2: 验证损失上升（过拟合）？

**解决方案：**
- 增加Dropout层
- 使用数据增强
- 减小模型容量
- 早停

### Q3: 推理速度慢？

**优化方法：**
- 模型量化
- TorchScript编译
- 使用更小的模型
- 批处理推理

### Q4: 预测不准确？

**检查清单：**
- 数据质量（ground truth正确？）
- 类别平衡（障碍物占比？）
- 模型容量（是否足够？）
- 训练充分（是否收敛？）

---

## 总结

### 规则模型 vs 学习模型

| 维度 | 规则模型 | 学习模型 |
|------|---------|---------|
| 开发成本 | 低 | 高 |
| 数据需求 | 无 | 大量 |
| 计算开销 | 小 | 大 |
| 精度 | 中等 | 较高 |
| 可解释性 | 高 | 低 |
| 适应性 | 差 | 好 |

### 建议

**使用规则模型，如果：**
- 快速原型开发
- 计算资源受限
- 需要高可解释性
- 环境相对简单

**使用学习模型，如果：**
- 有充足训练数据
- 有GPU计算资源
- 需要高精度预测
- 环境复杂多变

**混合方案：**
- 初期使用规则模型快速部署
- 收集真实运行数据
- 训练学习模型逐步替换
- 保留规则模型作为fallback

---

## 下一步

1. ✅ 完成数据收集脚本
2. ✅ 实现训练流程
3. ⬜ 实现评估模块（TODO）
4. ⬜ 实现可视化对比（TODO）
5. ⬜ 集成到模拟器（TODO）

---

**注意**: 这是一个完整的机器学习项目，需要耐心调试和优化。建议先使用规则模型，确认需求后再投入学习模型的开发。
