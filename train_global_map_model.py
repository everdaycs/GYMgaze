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
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 从 src/models 导入模型定义
from src.models.global_map import GlobalMapPredictor, ConvBlock


# ============== 数据集 ==============

class GlobalMapDataset(Dataset):
    """全局地图预测数据集"""
    
    def __init__(self, data_path: str, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.samples = []
        
        # 加载数据
        with open(data_path, 'rb') as f:
            episodes = pickle.load(f)
        
        # 展开所有序列
        for ep in episodes:
            for seq in ep['sequences']:
                self.samples.append(seq)
        
        print(f"加载数据集: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构建输入张量
        # 输入通道：
        # 1. 序列帧的local_occupancy (T帧)
        # 2. 当前的global_accumulated
        # 3. global_visit_count (归一化)
        
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
        
        # 组合输入 (T+3, H, W)
        # - T帧局部观测
        # - 1帧全局累积
        # - 1帧访问计数
        # - 1帧已知掩码
        input_tensor = np.concatenate([
            local_seq,                          # (T, H, W)
            global_acc[np.newaxis, :, :],       # (1, H, W)
            global_visit[np.newaxis, :, :],     # (1, H, W)
            known_mask[np.newaxis, :, :]        # (1, H, W)
        ], axis=0)
        
        # Ground Truth: 完整全局地图
        gt = sample['global_ground_truth'].astype(np.float32)
        # 将-1(边界)转为0.5，0(空闲)保持0，1(障碍物)保持1
        gt_tensor = np.where(gt == -1, 0.5, gt)
        
        # 有效区域掩码（排除边界）
        valid_mask = (sample['global_ground_truth'] >= 0).astype(np.float32)
        
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
        计算损失
        
        pred: (B, H, W) 预测
        target: (B, H, W) 真实标签
        valid_mask: (B, H, W) 有效区域（排除边界）
        known_mask: (B, H, W) 已知区域
        """
        # 基础BCE损失
        bce_loss = self.bce(pred, target)
        
        # 权重矩阵
        weight = torch.ones_like(pred)
        
        # 已知区域权重更高
        weight = weight + known_mask * (self.known_weight - 1)
        
        # 未知区域中的障碍物权重更高
        unknown_mask = (1 - known_mask) * valid_mask
        unknown_obstacle = unknown_mask * target
        weight = weight + unknown_obstacle * (self.unknown_obs_weight - 1)
        
        # 加权损失
        weighted_loss = bce_loss * weight * valid_mask
        
        # 平均
        loss = weighted_loss.sum() / (valid_mask.sum() + 1e-8)
        
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
                
                # 计算准确率
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='./data/global_map_training_data')
    parser.add_argument('--model-path', type=str, default='./checkpoints/global_map_model.pth')
    parser.add_argument('--sequence-length', type=int, default=5)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🗺️  训练全局地图预测模型")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_path = os.path.join(args.data_dir, 'training_data.pkl')
    dataset = GlobalMapDataset(data_path, args.sequence_length)
    
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
    # 输入通道：T帧局部观测 + 全局累积 + 访问计数 + 已知掩码
    in_channels = args.sequence_length + 3
    model = GlobalMapPredictor(in_channels=in_channels, base_channels=32)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练器
    trainer = GlobalMapTrainer(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'known_acc': [], 'unknown_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        
        train_loss = trainer.train_epoch(train_loader, optimizer)
        val_loss, known_acc, unknown_acc = trainer.evaluate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['known_acc'].append(known_acc)
        history['unknown_acc'].append(unknown_acc)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"已知区域准确率: {known_acc*100:.1f}%")
        print(f"未知区域准确率: {unknown_acc*100:.1f}%")
        
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
