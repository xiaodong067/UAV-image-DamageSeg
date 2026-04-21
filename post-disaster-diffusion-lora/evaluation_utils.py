"""
训练可视化工具
保存等级转换对比图：原图 | mask | L1 | L2 | L3 | L4
"""

import os
import torch
import numpy as np
from PIL import Image


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "eval_images")
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"✅ 可视化器初始化完成，输出目录: {self.images_dir}")
    
    def save_level_comparison(self, step: int, originals: torch.Tensor, masks: torch.Tensor,
                               level_outputs: dict):
        """
        保存等级转换对比图
        
        Args:
            step: 训练步数
            originals: 原始图像 [N, 3, H, W] in [-1, 1]
            masks: 掩码 [N, 1, H, W]
            level_outputs: {1: [N, 3, H, W], 2: ..., 3: ..., 4: ...} in [0, 1]
        """
        N = originals.shape[0]
        
        rows = []
        for i in range(N):
            # 原图 [-1,1] -> [0,1]
            orig = (originals[i] / 2 + 0.5).clamp(0, 1)
            # mask 转 3 通道
            mask = masks[i].repeat(3, 1, 1)
            
            # 拼接：原图 | mask | L1 | L2 | L3 | L4
            row_images = [orig, mask]
            for level in [1, 2, 3, 4]:
                row_images.append(level_outputs[level][i])
            
            row = torch.cat(row_images, dim=2)  # 水平拼接
            rows.append(row)
        
        # 垂直拼接所有行
        grid = torch.cat(rows, dim=1)
        
        # 保存
        grid_np = (grid.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(os.path.join(self.images_dir, f"step_{step:06d}.png"))
        print(f"📸 保存可视化: step_{step:06d}.png (原图|mask|L1|L2|L3|L4)")
