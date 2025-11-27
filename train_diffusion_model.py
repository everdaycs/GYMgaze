#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练障碍物预测扩散模型

将基于规则的模型升级为学习型神经网络模型

训练流程：
1. 数据收集：运行模拟器收集 (occupancy_grid, ground_truth_obstacles) 对
2. 模型定义：U-Net或类似架构
3. 训练：监督学习，预测障碍物概率分布
4. 评估：计算预测准确率、F1分数等
5. 部署：替换规则模型

使用方法：
    # 收集数据
    python train_diffusion_model.py --mode collect --episodes 1000
    
    # 训练模型
    python train_diffusion_model.py --mode train --epochs 50
    
    # 评估模型
    python train_diffusion_model.py --mode evaluate
    
    # 可视化对比
    python train_diffusion_model.py --mode visualize
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from ring_sonar_simulator import RingSonarCore, RingSonarRenderer


# ======================== 数据收集 ======================== #

class DataCollector:
    """收集训练数据：从模拟器中收集 occupancy map 和真实障碍物标签"""
    
    def __init__(self, data_dir: str = "./diffusion_training_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def collect_episode(self, episode_id: int, max_steps: int = 300) -> Dict:
        """收集一个episode的数据"""
        # 创建环境
        core = RingSonarCore(
            world_width=40.0,
            world_height=40.0,
            dt=0.1,
            trigger_mode="sequential"
        )
        renderer = RingSonarRenderer(core, render_mode=None)
        
        # 重置环境
        core.reset(regenerate_map=True, seed=episode_id)
        
        # 创建真实障碍物地图
        ground_truth = self._create_ground_truth_map(core, renderer)
        
        # 收集不同探索阶段的快照
        snapshots = []
        
        for step in range(max_steps):
            # 随机移动
            if step % 50 == 0:
                core.set_velocity(
                    float(np.random.uniform(-2.0, 3.0)),
                    float(np.random.uniform(-0.8, 0.8))
                )
            
            core.step()
            
            # 每隔一段时间保存快照
            if step % 20 == 0:
                # 输入：占用栅格地图（已知空闲空间）
                occupancy = renderer.occupancy_grid.copy()
                
                # 输出：真实障碍物位置
                # 注意：只标记已知区域外的障碍物（模拟未探索区域）
                snapshot = {
                    'occupancy': occupancy,
                    'ground_truth': ground_truth,
                    'robot_pos': core.robot_pos.copy(),
                    'visit_count': renderer.visit_count.copy(),
                    'step': step
                }
                snapshots.append(snapshot)
            
            # 更新占用栅格
            renderer.render()
        
        return {
            'episode_id': episode_id,
            'snapshots': snapshots,
            'obstacles': core.obstacles
        }
    
    def _create_ground_truth_map(self, core: RingSonarCore, 
                                  renderer: RingSonarRenderer) -> np.ndarray:
        """创建真实障碍物地图（从环境障碍物列表生成）"""
        gt_map = np.zeros((renderer.grid_height, renderer.grid_width), dtype=np.uint8)
        
        # 将所有障碍物光栅化到栅格地图
        for kind, data in core.obstacles:
            if kind == 'rect':
                x, y, w, h = data
                # 转换为栅格坐标
                gx1 = int(x / renderer.grid_resolution)
                gy1 = int(y / renderer.grid_resolution)
                gx2 = int((x + w) / renderer.grid_resolution)
                gy2 = int((y + h) / renderer.grid_resolution)
                
                # 限制在边界内
                gx1 = max(0, min(gx1, renderer.grid_width - 1))
                gy1 = max(0, min(gy1, renderer.grid_height - 1))
                gx2 = max(0, min(gx2, renderer.grid_width - 1))
                gy2 = max(0, min(gy2, renderer.grid_height - 1))
                
                # 标记为障碍物
                gt_map[gy1:gy2+1, gx1:gx2+1] = 255
        
        return gt_map
    
    def collect_dataset(self, num_episodes: int = 100):
        """收集完整数据集"""
        print(f"开始收集数据：{num_episodes} episodes")
        
        all_data = []
        
        for ep in tqdm(range(num_episodes), desc="收集数据"):
            episode_data = self.collect_episode(ep)
            all_data.append(episode_data)
        
        # 保存数据
        output_path = os.path.join(self.data_dir, 'training_data.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"数据已保存至: {output_path}")
        print(f"总快照数: {sum(len(ep['snapshots']) for ep in all_data)}")
        
        return all_data


# ======================== 数据集定义 ======================== #

class ObstaclePredictionDataset(Dataset):
    """障碍物预测数据集"""
    
    def __init__(self, data_path: str, transform=None):
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        
        # 展开所有快照
        self.samples = []
        for episode in self.raw_data:
            for snapshot in episode['snapshots']:
                self.samples.append(snapshot)
        
        self.transform = transform
        print(f"加载数据集: {len(self.samples)} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        snapshot = self.samples[idx]
        
        # 输入：占用栅格地图 (1通道)
        occupancy = snapshot['occupancy'].astype(np.float32) / 255.0
        
        # 输出：真实障碍物地图 (1通道，二值化)
        ground_truth = snapshot['ground_truth'].astype(np.float32) / 255.0
        
        # 访问计数（辅助信息）
        visit_count = snapshot['visit_count'].astype(np.float32)
        visit_count = np.clip(visit_count / 100.0, 0, 1)  # 归一化
        
        # 合并为多通道输入
        # 通道0: occupancy (空闲空间信息)
        # 通道1: visit_count (探索置信度)
        # 通道2: known_mask (已知/未知区域)
        known_mask = ((snapshot['occupancy'] < 80) | 
                     (snapshot['occupancy'] > 200)).astype(np.float32)
        
        input_tensor = np.stack([occupancy, visit_count, known_mask], axis=0)
        
        # 目标：只预测未知区域的障碍物
        # 已知区域设为-1（忽略）
        target = ground_truth.copy()
        target[known_mask > 0.5] = -1  # 已知区域不参与loss计算
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target = self.transform(target)
        
        return torch.FloatTensor(input_tensor), torch.FloatTensor(target[np.newaxis, :, :])


# ======================== 模型定义 ======================== #

class UNetBlock(nn.Module):
    """U-Net基础块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ObstacleDiffusionNet(nn.Module):
    """
    障碍物预测扩散网络
    基于U-Net架构，从occupancy map预测障碍物分布
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = UNetBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = UNetBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, 1, 1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output (sigmoid for probability)
        out = torch.sigmoid(self.out(dec1))
        
        return out


# ======================== 训练器 ======================== #

class DiffusionTrainer:
    """扩散模型训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss(reduction='none')  # 逐元素loss，方便mask
        
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for inputs, targets in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = self.model(inputs)
            
            # 计算loss（只考虑未知区域，target != -1）
            mask = (targets >= 0).float()
            loss = self.criterion(outputs, targets.clamp(0, 1))
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # 计算loss
                mask = (targets >= 0).float()
                loss = self.criterion(outputs, targets.clamp(0, 1))
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                
                total_loss += loss.item()
                
                # 收集预测和真实值
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        return total_loss / len(dataloader), all_preds, all_targets


# ======================== 主程序 ======================== #

def main():
    parser = argparse.ArgumentParser(description='训练障碍物预测扩散模型')
    parser.add_argument('--mode', type=str, default='collect',
                       choices=['collect', 'train', 'evaluate', 'visualize'],
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=100,
                       help='数据收集episode数')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--data-dir', type=str, default='./diffusion_training_data',
                       help='数据目录')
    parser.add_argument('--model-path', type=str, default='./obstacle_diffusion_model.pth',
                       help='模型保存路径')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        print("=" * 60)
        print("模式：数据收集")
        print("=" * 60)
        
        collector = DataCollector(args.data_dir)
        collector.collect_dataset(args.episodes)
        
    elif args.mode == 'train':
        print("=" * 60)
        print("模式：模型训练")
        print("=" * 60)
        
        # 加载数据
        data_path = os.path.join(args.data_dir, 'training_data.pkl')
        dataset = ObstaclePredictionDataset(data_path)
        
        # 划分训练/验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)
        
        # 创建模型
        model = ObstacleDiffusionNet(in_channels=3, base_channels=64)
        trainer = DiffusionTrainer(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 训练
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            train_loss = trainer.train_epoch(train_loader, optimizer)
            val_loss, _, _ = trainer.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, args.model_path)
                print(f"保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig('training_history.png')
        print("训练曲线已保存至 training_history.png")
        
    elif args.mode == 'evaluate':
        print("=" * 60)
        print("模式：模型评估")
        print("=" * 60)
        
        # TODO: 实现详细评估（准确率、F1、可视化对比）
        
    elif args.mode == 'visualize':
        print("=" * 60)
        print("模式：可视化对比")
        print("=" * 60)
        
        # TODO: 对比规则模型 vs 学习模型


if __name__ == "__main__":
    main()
