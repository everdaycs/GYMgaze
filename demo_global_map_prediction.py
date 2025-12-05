"""
全局地图预测演示

展示模型如何从局部观测和历史累积信息预测完整的全局地图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys

# 添加训练脚本路径
sys.path.insert(0, str(Path(__file__).parent))
from train_global_map_model import GlobalMapPredictor, GlobalMapDataset


def visualize_prediction(model, sample, device='cpu'):
    """
    可视化单个预测结果
    """
    model.eval()
    
    # 准备输入
    frames = sample['sequence_frames']
    gt = sample['global_ground_truth']
    
    # 构建输入张量
    input_channels = []
    
    # 1. 时序局部occupancy (5帧)
    for frame in frames:
        local_occ = frame['local_occupancy'].astype(np.float32)
        local_occ = np.where(local_occ == 127, 0.5, local_occ / 255.0)
        input_channels.append(local_occ)
    
    # 2. 全局累积map
    global_acc = frames[-1]['global_accumulated'].astype(np.float32)
    global_acc = np.where(global_acc == 127, 0.5, global_acc / 255.0)
    input_channels.append(global_acc)
    
    # 3. 访问计数 (归一化)
    visit_count = frames[-1]['global_visit_count'].astype(np.float32)
    visit_count = np.clip(visit_count / 10.0, 0, 1)
    input_channels.append(visit_count)
    
    # 4. 已知区域掩码
    known_mask = (frames[-1]['global_accumulated'] != 127).astype(np.float32)
    input_channels.append(known_mask)
    
    # Stack and prepare
    inputs = torch.FloatTensor(np.stack(input_channels, axis=0)).unsqueeze(0).to(device)
    known_mask_tensor = torch.FloatTensor(known_mask).unsqueeze(0).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        pred = model(inputs, known_mask_tensor)
        pred = pred.cpu().numpy()[0]
    
    # 准备Ground Truth
    gt_display = gt.copy().astype(np.float32)
    gt_display[gt == -1] = 0.5  # 边界显示为灰色
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：输入
    ax = axes[0, 0]
    local_last = frames[-1]['local_occupancy']
    local_display = np.where(local_last == 127, 0.5, local_last / 255.0)
    ax.imshow(local_display, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Current Local Observation')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(global_acc, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Accumulated Map (History)')
    ax.axis('off')
    
    ax = axes[0, 2]
    ax.imshow(known_mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Known Mask ({known_mask.mean()*100:.1f}% Explored)')
    ax.axis('off')
    
    # 第二行：输出
    ax = axes[1, 0]
    ax.imshow(gt_display, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Ground Truth (Full Map)')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Model Prediction')
    ax.axis('off')
    
    # 对比图
    ax = axes[1, 2]
    comparison = np.zeros((gt.shape[0], gt.shape[1], 3))
    
    valid_mask = (gt >= 0)
    unknown_mask = ~(known_mask.astype(bool)) & valid_mask
    
    # 已知区域：蓝色通道显示
    comparison[..., 2] = np.where(known_mask.astype(bool), gt_display, 0)
    
    # 预测正确的未知区域：绿色
    pred_binary = (pred > 0.5).astype(np.float32)
    correct = (pred_binary == gt) & unknown_mask
    comparison[..., 1] = correct * 0.7
    
    # 预测错误的未知区域：红色
    incorrect = (pred_binary != gt) & unknown_mask
    comparison[..., 0] = incorrect * 0.7
    
    ax.imshow(comparison)
    unknown_acc = correct.sum() / max(unknown_mask.sum(), 1) * 100
    ax.set_title(f'Comparison (Blue=Known, Green=Correct, Red=Error)\nUnknown Acc: {unknown_acc:.1f}%')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='global_map_models/best_model.pth',
                        help='模型路径')
    parser.add_argument('--data', type=str, default='global_map_training_data/training_data.pkl',
                        help='数据集路径')
    parser.add_argument('--samples', type=int, default=5, help='展示样本数量')
    parser.add_argument('--save', action='store_true', help='保存图片')
    args = parser.parse_args()
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = GlobalMapPredictor(in_channels=8, base_channels=32)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型训练于 epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f}")
    
    # 加载数据
    print(f"加载数据: {args.data}")
    with open(args.data, 'rb') as f:
        raw_data = pickle.load(f)
    
    # 展开所有序列
    all_samples = []
    for episode in raw_data:
        all_samples.extend(episode['sequences'])
    
    print(f"总样本数: {len(all_samples)}")
    
    # 随机选择样本
    np.random.seed(42)
    indices = np.random.choice(len(all_samples), min(args.samples, len(all_samples)), replace=False)
    
    # 创建输出目录
    output_dir = Path('global_map_prediction_demo')
    output_dir.mkdir(exist_ok=True)
    
    # 可视化
    total_unknown_acc = []
    for i, idx in enumerate(indices):
        sample = all_samples[idx]
        
        # 计算准确率用于统计
        fig = visualize_prediction(model, sample, device)
        
        if args.save:
            fig.savefig(output_dir / f'prediction_{i+1}.png', dpi=150, bbox_inches='tight')
            print(f"已保存: prediction_{i+1}.png")
        else:
            plt.show()
        
        plt.close(fig)
    
    print("\n✅ 演示完成!")
    if args.save:
        print(f"图片已保存至: {output_dir}/")


if __name__ == '__main__':
    main()
