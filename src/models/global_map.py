#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局地图预测模型

U-Net架构的全局地图预测网络
- 输入：时间序列局部观测 + 全局累积信息
- 输出：预测的全局地图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GlobalMapPredictor(nn.Module):
    """
    全局地图预测网络
    
    架构：U-Net with Skip Connections
    - 编码器：处理时空输入
    - 解码器：生成全局地图预测
    - 关键：利用已知区域信息引导未知区域预测
    
    输入通道说明（默认8通道）：
    - 5帧局部观测 (T=5)
    - 1帧全局累积
    - 1帧访问计数
    - 1帧已知掩码
    """
    
    def __init__(self, in_channels: int = 8, base_channels: int = 32):
        super().__init__()
        
        C = base_channels
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, C)      # -> C
        self.enc2 = ConvBlock(C, C*2)              # -> C*2
        self.enc3 = ConvBlock(C*2, C*4)            # -> C*4
        self.enc4 = ConvBlock(C*4, C*8)            # -> C*8
        
        # 瓶颈层
        self.bottleneck = ConvBlock(C*8, C*8)      # -> C*8
        
        # 解码器 (上采样后与对应encoder拼接)
        self.up4 = nn.ConvTranspose2d(C*8, C*8, 2, stride=2)
        self.dec4 = ConvBlock(C*8 + C*8, C*4)
        
        self.up3 = nn.ConvTranspose2d(C*4, C*4, 2, stride=2)
        self.dec3 = ConvBlock(C*4 + C*4, C*2)
        
        self.up2 = nn.ConvTranspose2d(C*2, C*2, 2, stride=2)
        self.dec2 = ConvBlock(C*2 + C*2, C)
        
        self.up1 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.dec1 = ConvBlock(C + C, C)
        
        # 输出层
        self.out_conv = nn.Conv2d(C, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x, known_mask=None):
        """
        前向传播
        
        Args:
            x: (B, C, H, W) 输入特征
            known_mask: (B, 1, H, W) 已知区域掩码（可选）
        
        Returns:
            (B, H, W) 预测的全局地图
        """
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e4))
        
        # 解码 (with skip connections)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 输出
        out = torch.sigmoid(self.out_conv(d1))
        
        return out.squeeze(1)  # (B, H, W)
