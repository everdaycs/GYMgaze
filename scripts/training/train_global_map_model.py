#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局地图预测模型训练脚本

任务：从局部观测+历史累积信息 → 预测完整全局地图
类似SLAM中的地图构建，但使用深度学习进行预测
"""

import os
import sys
import pickle
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 导入AMP，兼容不同PyTorch版本
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

# 添加项目根目录到路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 从 src/models 导入模型定义
from src.models.global_map import GlobalMapPredictor, ConvBlock


# ============== 数据集 ==============

class GlobalMapDataset(Dataset):
    """全局地图预测数据集"""
    
    def __init__(self, data_path: str, sequence_length: int = 5, use_local_gt: bool = True):
        """
        初始化数据集
        
        参数:
            data_path: 数据文件路径或目录路径
                      如果是目录，自动加载所有 training_data_batch_*.pkl 文件
                      如果是文件，加载单个 .pkl 文件
            sequence_length: 序列长度
            use_local_gt: 是否使用局部 ground truth（已知区域附近的障碍物）
                         如果为 False，则使用全局 ground truth
        """
        self.sequence_length = sequence_length
        self.use_local_gt = use_local_gt
        self.samples = []
        
        # 检查是目录还是文件
        import glob
        if os.path.isdir(data_path):
            # 从目录加载所有批次文件
            batch_files = sorted(glob.glob(os.path.join(data_path, 'training_data_batch_*.pkl')))
            if not batch_files:
                # 如果没有批次文件，尝试加载单个文件
                single_file = os.path.join(data_path, 'training_data.pkl')
                batch_files = [single_file] if os.path.exists(single_file) else []
            
            if not batch_files:
                raise FileNotFoundError(f"未找到数据文件: {data_path}")
            
            # 加载所有批次
            for batch_file in batch_files:
                with open(batch_file, 'rb') as f:
                    episodes = pickle.load(f)
                    # 展开所有序列
                    for ep in episodes:
                        for seq in ep['sequences']:
                            self.samples.append(seq)
        else:
            # 加载单个文件
            with open(data_path, 'rb') as f:
                episodes = pickle.load(f)
            
            # 展开所有序列
            for ep in episodes:
                for seq in ep['sequences']:
                    self.samples.append(seq)
        
        gt_type = "局部GT (已知区域附近)" if use_local_gt else "全局GT"
        print(f"加载数据集: {len(self.samples)} 个样本, 使用 {gt_type}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构建输入张量
        # 输入通道：
        # 1. 序列帧的local_occupancy (T帧)
        # 2. 当前的global_accumulated
        # 3. global_visit_count (归一化)
        # 4. 已知区域掩码
        # 5. 边界掩码（新增）
        
        frames = sample['sequence_frames']
        T = len(frames)
        
        # 获取最后一帧的全局累积信息
        last_frame = frames[-1]
        global_acc = last_frame['global_accumulated'].astype(np.float32) / 255.0
        global_visit = np.clip(last_frame['global_visit_count'].astype(np.float32) / 100.0, 0, 1)
        
        # 构建时间序列输入 (T, H, W)
        local_seq = np.stack([
            f['local_occupancy'].astype(np.float32) / 255.0 
            for f in frames
        ], axis=0)
        
        # 创建已知区域掩码 (H, W)
        known_mask = (last_frame['global_accumulated'] != 127).astype(np.float32)
        
        # Ground Truth: 选择使用局部或全局
        if self.use_local_gt and 'local_ground_truth' in sample:
            gt = sample['local_ground_truth'].astype(np.float32)
        else:
            gt = sample['global_ground_truth'].astype(np.float32)
        
        # 创建边界掩码：-1的区域是边界
        # border_mask: 1=边界区域（不参与训练），0=有效区域
        border_mask = (gt == -1).astype(np.float32)
        
        # 有效区域掩码（排除边界区域，只在有效区域计算损失）
        # valid_mask: 1=有效区域（参与训练），0=边界区域
        valid_mask = (gt >= 0).astype(np.float32)
        
        # 将-1(边界)转为0（空闲），避免影响学习
        # 边界区域的预测不参与损失计算，所以target值不重要
        # 但使用0比0.5更合理，因为边界外通常没有障碍物
        gt_tensor = np.where(gt == -1, 0.0, gt)
        
        # 组合输入 (T+4, H, W) - 保持与之前兼容
        # - T帧局部观测
        # - 1帧全局累积
        # - 1帧访问计数
        # - 1帧已知掩码
        # - 1帧边界掩码
        input_tensor = np.concatenate([
            local_seq,                          # (T, H, W)
            global_acc[np.newaxis, :, :],       # (1, H, W)
            global_visit[np.newaxis, :, :],     # (1, H, W)
            known_mask[np.newaxis, :, :],       # (1, H, W)
            border_mask[np.newaxis, :, :]       # (1, H, W)
        ], axis=0)
        
        return (
            torch.from_numpy(input_tensor),
            torch.from_numpy(gt_tensor),
            torch.from_numpy(valid_mask),
            torch.from_numpy(known_mask)
        )


# ============== 训练器 ==============

class GlobalMapTrainer:
    """全局地图预测训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 损失函数：结合BCE和已知区域约束
        self.bce = nn.BCELoss(reduction='none')
        
        # 已知区域权重更高（确保已知区域预测准确）
        self.known_weight = 2.0
        # 未知区域中障碍物权重
        self.unknown_obs_weight = 5.0
    
    def compute_loss(self, pred, target, valid_mask, known_mask):
        """
        计算损失（改进的边界处理）
        
        参数:
            pred: (B, H, W) 模型预测值，范围[0,1]
            target: (B, H, W) 真实标签，0=空闲，1=障碍物
            valid_mask: (B, H, W) 有效区域掩码，1=有效（非边界），0=边界
            known_mask: (B, H, W) 已知区域掩码，1=已探索，0=未探索
        
        边界处理策略:
            1. valid_mask=0 的区域（边界）完全不参与损失计算
            2. 边界区域的梯度为0，不会影响模型学习
            3. 已知区域权重更高，确保已探索区域预测准确
            4. 未知区域中的障碍物权重更高，鼓励模型预测未知区域的障碍物
        
        返回:
            loss: 标量损失值
        """
        # 基础BCE损失（逐像素）
        bce_loss = self.bce(pred, target)
        
        # 权重矩阵初始化为1
        weight = torch.ones_like(pred)
        
        # 已知区域权重更高（确保已探索区域预测准确）
        weight = weight + known_mask * (self.known_weight - 1)
        
        # 未知区域中的障碍物权重更高（鼓励预测未知障碍物）
        unknown_mask = (1 - known_mask) * valid_mask  # 未知且有效的区域
        unknown_obstacle = unknown_mask * target       # 未知区域中的真实障碍物
        weight = weight + unknown_obstacle * (self.unknown_obs_weight - 1)
        
        # 应用有效区域掩码（边界区域权重为0，不参与损失）
        # 这是关键：valid_mask=0 的区域（边界）损失为0
        weighted_loss = bce_loss * weight * valid_mask
        
        # 平均（只在有效区域上平均）
        valid_count = valid_mask.sum() + 1e-8  # 防止除零
        loss = weighted_loss.sum() / valid_count
        
        return loss
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for inputs, targets, valid_masks, known_masks in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            valid_masks = valid_masks.to(self.device)
            known_masks = known_masks.to(self.device)
            
            # Forward
            outputs = self.model(inputs, known_masks.unsqueeze(1))
            
            # Loss
            loss = self.compute_loss(outputs, targets, valid_masks, known_masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train_epoch_amp(self, dataloader, optimizer, scaler):
        """使用自动混合精度(AMP)训练一个epoch - 更快更省显存"""
        self.model.train()
        total_loss = 0
        
        for inputs, targets, valid_masks, known_masks in tqdm(dataloader, desc="Training (AMP)"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            valid_masks = valid_masks.to(self.device)
            known_masks = known_masks.to(self.device)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播（兼容不同版本）
            try:
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs, known_masks.unsqueeze(1))
                    loss = self.compute_loss(outputs, targets, valid_masks, known_masks)
            except TypeError:
                # 旧版本的autocast语法
                with autocast():
                    outputs = self.model(inputs, known_masks.unsqueeze(1))
                    loss = self.compute_loss(outputs, targets, valid_masks, known_masks)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        # 分别统计已知区域和未知区域的准确率
        known_correct = 0
        known_total = 0
        unknown_correct = 0
        unknown_total = 0
        
        with torch.no_grad():
            for inputs, targets, valid_masks, known_masks in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                valid_masks = valid_masks.to(self.device)
                known_masks = known_masks.to(self.device)
                
                outputs = self.model(inputs, known_masks.unsqueeze(1))
                
                loss = self.compute_loss(outputs, targets, valid_masks, known_masks)
                total_loss += loss.item()
                
                # 计算准确率（只在有效区域内）
                pred_binary = (outputs > 0.5).float()
                correct = (pred_binary == targets) * valid_masks
                
                # 已知区域准确率
                known_correct += (correct * known_masks).sum().item()
                known_total += (valid_masks * known_masks).sum().item()
                
                # 未知区域准确率
                unknown_mask = (1 - known_masks) * valid_masks
                unknown_correct += (correct * unknown_mask).sum().item()
                unknown_total += unknown_mask.sum().item()
        
        known_acc = known_correct / (known_total + 1e-8)
        unknown_acc = unknown_correct / (unknown_total + 1e-8)
        
        return total_loss / len(dataloader), known_acc, unknown_acc


def main():
    parser = argparse.ArgumentParser(description='训练全局地图预测模型')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='./data/global_map_training_data')
    parser.add_argument('--model-path', type=str, default='./checkpoints/global_map_model.pth')
    parser.add_argument('--sequence-length', type=int, default=5)
    parser.add_argument('--use-global-gt', action='store_true',
                       help='使用全局GT训练（默认使用局部GT，只预测已知区域附近）')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='设备选择: auto(自动)/cuda(强制GPU)/cpu(强制CPU), 默认auto')
    parser.add_argument('--no-cuda', action='store_true',
                       help='禁用CUDA（强制使用CPU）')
    parser.add_argument('--cuda-device', type=int, default=0,
                       help='指定使用的GPU设备ID，默认0')
    parser.add_argument('--amp', action='store_true',
                       help='启用自动混合精度训练（加快速度，节省显存）')
    parser.add_argument('--benchmark', action='store_true',
                       help='启用cuDNN benchmark（可能加快训练，但占用更多显存）')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🗺️  训练全局地图预测模型")
    print("=" * 70)
    
    use_local_gt = not args.use_global_gt
    gt_type = "局部GT (已知区域附近)" if use_local_gt else "全局GT (完整地图)"
    print(f"Ground Truth 类型: {gt_type}")
    
    # ============== 设备配置 ==============
    if args.no_cuda:
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"\n💻 设备配置:")
    print(f"   使用设备: {device}")
    
    # CUDA信息
    if device.type == 'cuda':
        print(f"   GPU设备: NVIDIA {torch.cuda.get_device_name(device.index)}")
        print(f"   GPU显存: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.1f} GB")
        
        # cuDNN优化
        if args.benchmark:
            torch.backends.cudnn.benchmark = True
            print(f"   cuDNN benchmark: 启用 (可能提升速度)")
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print(f"   cuDNN benchmark: 禁用 (更稳定)")
        
        # AMP配置
        if args.amp:
            print(f"   混合精度训练: 启用 ✓")
        else:
            print(f"   混合精度训练: 禁用")
    else:
        print(f"   ⚠️  使用CPU训练，速度会很慢")
    
    # 加载数据（自动支持单个文件或批次文件）
    dataset = GlobalMapDataset(args.data_dir, args.sequence_length, use_local_gt=use_local_gt)
    
    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"训练集: {train_size} 样本")
    print(f"验证集: {val_size} 样本")
    
    # 创建模型
    # 输入通道：T帧局部观测 + 全局累积 + 访问计数 + 已知掩码 + 边界掩码
    in_channels = args.sequence_length + 4  # T + 4 channels
    model = GlobalMapPredictor(in_channels=in_channels, base_channels=32)
    
    print(f"\n📊 模型配置:")
    print(f"   输入通道数: {in_channels} (序列长度{args.sequence_length} + 4)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 训练器
    trainer = GlobalMapTrainer(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 自动混合精度（AMP）
    scaler = None
    if args.amp and device.type == 'cuda':
        scaler = GradScaler()
        print(f"   自动混合精度: 启用 ✓")
    else:
        print(f"   自动混合精度: 禁用")
    
    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'known_acc': [], 'unknown_acc': []}
    
    print(f"\n{'='*70}")
    print(f"🚀 开始训练...")
    print(f"{'='*70}\n")
    
    for epoch in range(args.epochs):
        print(f"📊 Epoch {epoch+1}/{args.epochs}")
        
        # 训练循环（支持AMP）
        if scaler is not None:
            train_loss = trainer.train_epoch_amp(train_loader, optimizer, scaler)
        else:
            train_loss = trainer.train_epoch(train_loader, optimizer)
        
        val_loss, known_acc, unknown_acc = trainer.evaluate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['known_acc'].append(known_acc)
        history['unknown_acc'].append(unknown_acc)
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   已知区域准确率: {known_acc*100:.1f}%")
        print(f"   未知区域准确率: {unknown_acc*100:.1f}%")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, args.model_path)
            print(f"💾 保存最佳模型 (Val Loss: {val_loss:.4f})")
    
    print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
    
    # 保存训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curve')
    
    axes[1].plot(history['known_acc'], label='Known Region')
    axes[1].plot(history['unknown_acc'], label='Unknown Region')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy by Region')
    
    plt.tight_layout()
    plt.savefig('global_map_training_history.png', dpi=150)
    print("📈 训练曲线已保存")


if __name__ == "__main__":
    main()
