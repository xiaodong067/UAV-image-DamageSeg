"""
Step 2: 训练数据准备

功能：
1. 从 training_manifest.json 读取实例清单
2. 按建筑实例裁剪 ROI（带上下文边界）
3. 生成 inpainting 训练三元组：image, masked_image, mask
4. 统一 resize 到训练分辨率
5. 保存为训练可用的格式

输出目录结构：
disaster_lora/training_data/
├── L2/
│   ├── 0001_image.png
│   ├── 0001_masked.png
│   ├── 0001_mask.png
│   └── ...
├── L3/
├── L4/
└── metadata.json
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict


@dataclass
class TrainingSample:
    """训练样本"""
    sample_id: str
    level: int
    image_path: str      # 裁剪后的灾后图
    masked_path: str     # 建筑区域抹掉的图
    mask_path: str       # 建筑 mask
    prompt: str          # 训练 prompt
    original_image: str  # 原始图像路径
    bbox: Tuple[int, int, int, int]  # 在原图中的位置


class TrainingDataPreparer:
    """训练数据准备器"""
    
    # Prompt 模板
    PROMPT_TEMPLATE = "aerial view of post-disaster building damage <L{level}>"
    
    def __init__(
        self,
        manifest_path: str,
        output_dir: str = "./disaster_lora/training_data",
        target_size: int = 512,
        context_ratio: float = 0.2,  # ROI 外扩比例
        mask_dilate_kernel: int = 5,  # mask 膨胀核大小
    ):
        """
        Args:
            manifest_path: training_manifest.json 路径
            output_dir: 输出目录
            target_size: 目标训练分辨率
            context_ratio: ROI 外扩比例（0.2 = 外扩 20%）
            mask_dilate_kernel: mask 膨胀核大小（避免边缘问题）
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.context_ratio = context_ratio
        self.mask_dilate_kernel = mask_dilate_kernel
        
        # 加载 manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # 创建输出目录
        for level in [1, 2, 3, 4]:
            (self.output_dir / f"L{level}").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "masked_images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
    
    def _expand_bbox(
        self, 
        bbox: Tuple[int, int, int, int], 
        img_shape: Tuple[int, int],
        ratio: float
    ) -> Tuple[int, int, int, int]:
        """
        扩展 bbox，添加上下文边界
        
        Args:
            bbox: (x, y, w, h)
            img_shape: (height, width)
            ratio: 外扩比例
        
        Returns:
            expanded bbox (x, y, w, h)
        """
        x, y, w, h = bbox
        img_h, img_w = img_shape
        
        # 计算外扩量
        expand_w = int(w * ratio)
        expand_h = int(h * ratio)
        
        # 扩展并裁剪到图像边界
        new_x = max(0, x - expand_w)
        new_y = max(0, y - expand_h)
        new_w = min(img_w - new_x, w + 2 * expand_w)
        new_h = min(img_h - new_y, h + 2 * expand_h)
        
        return (new_x, new_y, new_w, new_h)
    
    def _make_square_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        img_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        将 bbox 调整为正方形（取较大边）
        """
        x, y, w, h = bbox
        img_h, img_w = img_shape
        
        # 取较大边作为正方形边长
        size = max(w, h)
        
        # 计算中心点
        cx = x + w // 2
        cy = y + h // 2
        
        # 从中心扩展为正方形
        new_x = max(0, cx - size // 2)
        new_y = max(0, cy - size // 2)
        
        # 确保不超出图像边界
        if new_x + size > img_w:
            new_x = img_w - size
        if new_y + size > img_h:
            new_y = img_h - size
        
        # 如果图像本身比 size 小，调整 size
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        size = min(size, img_w - new_x, img_h - new_y)
        
        return (new_x, new_y, size, size)
    
    def _crop_and_resize(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: int
    ) -> np.ndarray:
        """裁剪并 resize 到目标尺寸"""
        x, y, w, h = bbox
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _crop_and_resize_mask(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: int
    ) -> np.ndarray:
        """裁剪并 resize mask（使用最近邻插值保持边界清晰）"""
        x, y, w, h = bbox
        cropped = mask[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        return resized
    
    def _create_instance_mask(
        self,
        full_mask: np.ndarray,
        instance_id: int,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        创建单实例的二值 mask
        
        Args:
            full_mask: 完整的 label mask (0/1/2/3/4)
            instance_id: 连通域 ID
            bbox: 扩展后的 bbox
        
        Returns:
            二值 mask (0/255)
        """
        # 创建建筑区域二值 mask
        building_mask = (full_mask > 0).astype(np.uint8)
        
        # 连通域分析找到对应实例
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            building_mask, connectivity=8
        )
        
        # 获取该实例的 mask
        instance_mask = (labels == instance_id).astype(np.uint8) * 255
        
        # 轻微膨胀（避免边缘问题）
        if self.mask_dilate_kernel > 0:
            kernel = np.ones((self.mask_dilate_kernel, self.mask_dilate_kernel), np.uint8)
            instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)
        
        return instance_mask
    
    def _create_masked_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        fill_value: int = 128  # 灰色填充
    ) -> np.ndarray:
        """
        创建 masked image（建筑区域被抹掉）
        
        Args:
            image: 原图
            mask: 二值 mask (0/255)
            fill_value: 填充值
        
        Returns:
            masked image
        """
        masked = image.copy()
        mask_bool = mask > 127
        masked[mask_bool] = fill_value
        return masked
    
    def process_instance(
        self,
        instance_info: Dict,
        level: int,
        sample_idx: int
    ) -> Optional[TrainingSample]:
        """
        处理单个实例
        
        Returns:
            TrainingSample or None if failed
        """
        try:
            # 读取原图和 mask
            image_path = instance_info['image_path']
            mask_path = instance_info['mask_path']
            instance_id = instance_info['instance_id']
            bbox = tuple(instance_info['bbox'])
            
            image = cv2.imread(image_path)
            full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or full_mask is None:
                return None
            
            img_shape = image.shape[:2]
            
            # 扩展 bbox（添加上下文）
            expanded_bbox = self._expand_bbox(bbox, img_shape, self.context_ratio)
            
            # 调整为正方形
            square_bbox = self._make_square_bbox(expanded_bbox, img_shape)
            
            # 检查 bbox 是否有效
            x, y, w, h = square_bbox
            if w < 64 or h < 64:
                return None
            
            # 创建实例 mask
            instance_mask = self._create_instance_mask(full_mask, instance_id, square_bbox)
            
            # 裁剪并 resize
            image_crop = self._crop_and_resize(image, square_bbox, self.target_size)
            mask_crop = self._crop_and_resize_mask(instance_mask, square_bbox, self.target_size)
            
            # 创建 masked image
            masked_crop = self._create_masked_image(image_crop, mask_crop)
            
            # 生成文件名和路径
            sample_id = f"{sample_idx:05d}"
            level_dir = self.output_dir / f"L{level}"
            
            image_save_path = level_dir / f"{sample_id}_image.png"
            masked_save_path = level_dir / f"{sample_id}_masked.png"
            mask_save_path = level_dir / f"{sample_id}_mask.png"
            
            # 保存
            cv2.imwrite(str(image_save_path), image_crop)
            cv2.imwrite(str(masked_save_path), masked_crop)
            cv2.imwrite(str(mask_save_path), mask_crop)
            
            # 生成 prompt
            prompt = self.PROMPT_TEMPLATE.format(level=level)
            
            return TrainingSample(
                sample_id=sample_id,
                level=level,
                image_path=str(image_save_path),
                masked_path=str(masked_save_path),
                mask_path=str(mask_save_path),
                prompt=prompt,
                original_image=image_path,
                bbox=square_bbox,
            )
            
        except Exception as e:
            print(f"Error processing instance: {e}")
            return None
    
    def prepare_all(self, max_per_level: int = None) -> List[TrainingSample]:
        """
        处理所有实例
        
        Args:
            max_per_level: 每个等级最大样本数（用于平衡/测试）
        
        Returns:
            所有训练样本列表
        """
        all_samples = []
        level_counts = defaultdict(int)
        
        for level in [1, 2, 3, 4]:
            level_key = str(level)
            if level_key not in self.manifest['instances']:
                print(f"Warning: No instances for level {level}")
                continue
            
            instances = self.manifest['instances'][level_key]
            
            # 限制数量（如果指定）
            if max_per_level:
                instances = instances[:max_per_level]
            
            print(f"\nProcessing Level {level}: {len(instances)} instances")
            
            for idx, inst_info in enumerate(tqdm(instances, desc=f"L{level}")):
                sample = self.process_instance(inst_info, level, level_counts[level])
                if sample:
                    all_samples.append(sample)
                    level_counts[level] += 1
        
        # 保存 metadata
        self._save_metadata(all_samples, level_counts)
        
        return all_samples
    
    def _save_metadata(self, samples: List[TrainingSample], level_counts: Dict):
        """保存训练数据元信息"""
        metadata = {
            "config": {
                "target_size": self.target_size,
                "context_ratio": self.context_ratio,
                "mask_dilate_kernel": self.mask_dilate_kernel,
                "prompt_template": self.PROMPT_TEMPLATE,
            },
            "summary": {
                "total": len(samples),
                "by_level": dict(level_counts),
            },
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "level": s.level,
                    "image_path": s.image_path,
                    "masked_path": s.masked_path,
                    "mask_path": s.mask_path,
                    "prompt": s.prompt,
                }
                for s in samples
            ]
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n训练数据元信息已保存: {metadata_path}")
        print(f"总样本数: {len(samples)}")
        for level, count in sorted(level_counts.items()):
            print(f"  Level {level}: {count}")


def main():
    """主函数"""
    
    # ========== 配置 ==========
    MANIFEST_PATH = "./disaster_lora/data_analysis/training_manifest.json"
    OUTPUT_DIR = "./disaster_lora/training_data"
    TARGET_SIZE = 512       # 训练分辨率
    CONTEXT_RATIO = 0.2     # ROI 外扩 20%
    MAX_PER_LEVEL = None    # None = 全部，或设置数字限制
    # ==========================
    
    preparer = TrainingDataPreparer(
        manifest_path=MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        context_ratio=CONTEXT_RATIO,
    )
    
    samples = preparer.prepare_all(max_per_level=MAX_PER_LEVEL)
    
    print(f"\n完成！共生成 {len(samples)} 个训练样本")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
