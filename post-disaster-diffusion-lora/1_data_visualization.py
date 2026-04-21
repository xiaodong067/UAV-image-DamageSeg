"""
Step 1: 数据可视化 & 单实例 mask/level 对齐检查

功能：
1. 可视化灾后图像 + mask 叠加
2. 提取建筑连通域（单实例）
3. 统计每个实例的等级分布
4. 过滤等级纯度低的实例
5. 生成训练样本清单

验收标准：
- mask 与建筑对齐
- 每个实例有明确的 level 标签
- 等级分布可视化正确
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class BuildingInstance:
    """单个建筑实例的信息"""
    image_path: str
    mask_path: str
    instance_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    level: int  # 主导等级 (2/3/4)
    level_distribution: Dict[int, float]  # 各等级占比
    purity: float  # 主导等级纯度
    pixel_count: int  # 实例像素数


class DisasterDatasetAnalyzer:
    """灾后数据集分析器"""
    
    # 等级颜色映射（用于可视化）
    LEVEL_COLORS = {
        0: (128, 128, 128),  # 背景 - 灰色
        1: (0, 255, 0),      # Level 1 - 绿色（无损/轻微）
        2: (255, 255, 0),    # Level 2 - 黄色（轻度）
        3: (255, 165, 0),    # Level 3 - 橙色（中度）
        4: (255, 0, 0),      # Level 4 - 红色（严重）
    }
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        output_dir: str = "./disaster_lora/data_analysis",
        purity_threshold: float = 0.7,
        min_instance_pixels: int = 500,
        target_levels: List[int] = [1, 2, 3, 4],
    ):
        """
        Args:
            image_dir: 灾后图像目录
            mask_dir: 灾后 mask 目录
            output_dir: 输出目录
            purity_threshold: 等级纯度阈值（低于此值的实例被过滤）
            min_instance_pixels: 最小实例像素数（过滤太小的建筑）
            target_levels: 目标训练等级
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.purity_threshold = purity_threshold
        self.min_instance_pixels = min_instance_pixels
        self.target_levels = target_levels
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_image_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """查找图像-mask 配对（原图 {id}.jpg，掩码 {id}_mask.png）"""
        pairs = []
        
        # 获取所有 jpg 图像文件
        image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.JPG'))
        
        for img_path in sorted(image_files):
            # mask 文件名：{id}_mask.png
            stem = img_path.stem
            mask_path = self.mask_dir / f"{stem}_mask.png"
            
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                # 尝试大写
                mask_path_upper = self.mask_dir / f"{stem}_mask.PNG"
                if mask_path_upper.exists():
                    pairs.append((img_path, mask_path_upper))
                else:
                    print(f"Warning: No mask found for {img_path.name} (expected {stem}_mask.png)")
        
        print(f"Found {len(pairs)} image-mask pairs")
        return pairs
    
    def extract_instances(self, mask: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        从 mask 中提取建筑连通域实例
        
        Returns:
            List of (instance_mask, info_dict)
        """
        instances = []
        
        # 创建建筑区域二值 mask（所有非背景像素）
        building_mask = (mask > 0).astype(np.uint8)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            building_mask, connectivity=8
        )
        
        # 跳过背景（label 0）
        for label_id in range(1, num_labels):
            # 该实例的 mask
            instance_mask = (labels == label_id).astype(np.uint8)
            pixel_count = stats[label_id, cv2.CC_STAT_AREA]
            
            # 过滤太小的实例
            if pixel_count < self.min_instance_pixels:
                continue
            
            # 计算该实例内的等级分布
            instance_pixels = mask[instance_mask > 0]
            level_counts = {}
            for level in range(1, 5):  # 1, 2, 3, 4
                count = np.sum(instance_pixels == level)
                if count > 0:
                    level_counts[level] = count
            
            if not level_counts:
                continue
            
            # 计算主导等级和纯度
            total = sum(level_counts.values())
            level_distribution = {k: v / total for k, v in level_counts.items()}
            dominant_level = max(level_counts, key=level_counts.get)
            purity = level_distribution[dominant_level]
            
            # bbox
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            
            info = {
                'label_id': label_id,
                'bbox': (x, y, w, h),
                'pixel_count': pixel_count,
                'level_distribution': level_distribution,
                'dominant_level': dominant_level,
                'purity': purity,
            }
            
            instances.append((instance_mask, info))
        
        return instances
    
    def visualize_single_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instances: List[Tuple[np.ndarray, Dict]],
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """可视化单张图像的 mask 和实例"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 原图
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Mask 等级可视化
        mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for level, color in self.LEVEL_COLORS.items():
            mask_colored[mask == level] = color
        axes[0, 1].imshow(mask_colored)
        axes[0, 1].set_title('Mask (Level Colors: G=1, Y=2, O=3, R=4)')
        axes[0, 1].axis('off')
        
        # 3. 原图 + Mask 叠加
        overlay = image.copy()
        alpha = 0.4
        for level, color in self.LEVEL_COLORS.items():
            if level == 0:
                continue
            level_mask = (mask == level)
            overlay[level_mask] = (
                overlay[level_mask] * (1 - alpha) + 
                np.array(color[::-1]) * alpha  # BGR
            ).astype(np.uint8)
        axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Overlay (Image + Mask)')
        axes[1, 0].axis('off')
        
        # 4. 实例分割 + 等级标注
        instance_vis = image.copy()
        valid_count = 0
        for inst_mask, info in instances:
            level = info['dominant_level']
            purity = info['purity']
            bbox = info['bbox']
            
            # 只标注目标等级且纯度达标的实例
            is_valid = (level in self.target_levels and purity >= self.purity_threshold)
            
            if is_valid:
                valid_count += 1
                color = self.LEVEL_COLORS[level][::-1]  # BGR
                # 绘制轮廓
                contours, _ = cv2.findContours(
                    inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(instance_vis, contours, -1, color, 2)
                # 标注等级和纯度
                x, y, w, h = bbox
                text = f"L{level} ({purity:.0%})"
                cv2.putText(
                    instance_vis, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        axes[1, 1].imshow(cv2.cvtColor(instance_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Valid Instances: {valid_count} (purity >= {self.purity_threshold:.0%})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def analyze_dataset(self, max_samples: int = None, visualize_count: int = 10):
        """
        分析整个数据集
        
        Args:
            max_samples: 最大处理样本数（None 表示全部）
            visualize_count: 可视化样本数
        """
        pairs = self.find_image_mask_pairs()
        if max_samples:
            pairs = pairs[:max_samples]
        
        all_instances: List[BuildingInstance] = []
        level_stats = defaultdict(int)
        purity_stats = defaultdict(list)
        
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        print("Analyzing dataset...")
        for idx, (img_path, mask_path) in enumerate(tqdm(pairs)):
            # 读取图像和 mask
            image = cv2.imread(str(img_path))
            if mask_path.suffix == '.npy':
                mask = np.load(str(mask_path))
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Warning: Failed to load {img_path.name}")
                continue
            
            # 提取实例
            instances = self.extract_instances(mask)
            
            # 可视化前 N 张
            if idx < visualize_count:
                save_path = vis_dir / f"{img_path.stem}_analysis.png"
                self.visualize_single_image(image, mask, instances, save_path)
            
            # 收集有效实例
            for inst_mask, info in instances:
                level = info['dominant_level']
                purity = info['purity']
                
                level_stats[level] += 1
                purity_stats[level].append(purity)
                
                # 只保留目标等级且纯度达标的实例
                if level in self.target_levels and purity >= self.purity_threshold:
                    instance = BuildingInstance(
                        image_path=str(img_path),
                        mask_path=str(mask_path),
                        instance_id=info['label_id'],
                        bbox=info['bbox'],
                        level=level,
                        level_distribution=info['level_distribution'],
                        purity=purity,
                        pixel_count=info['pixel_count'],
                    )
                    all_instances.append(instance)
        
        # 统计报告
        self._generate_report(all_instances, level_stats, purity_stats)
        
        # 保存训练样本清单
        self._save_training_manifest(all_instances)
        
        return all_instances
    
    def _generate_report(
        self,
        instances: List[BuildingInstance],
        level_stats: Dict[int, int],
        purity_stats: Dict[int, List[float]],
    ):
        """生成统计报告"""
        report_path = self.output_dir / "analysis_report.txt"
        
        lines = [
            "=" * 60,
            "灾后数据集分析报告",
            "=" * 60,
            "",
            f"数据目录: {self.image_dir}",
            f"Mask 目录: {self.mask_dir}",
            f"纯度阈值: {self.purity_threshold:.0%}",
            f"最小实例像素: {self.min_instance_pixels}",
            "",
            "-" * 40,
            "各等级实例统计（过滤前）",
            "-" * 40,
        ]
        
        for level in sorted(level_stats.keys()):
            count = level_stats[level]
            purities = purity_stats[level]
            avg_purity = np.mean(purities) if purities else 0
            lines.append(f"Level {level}: {count} 个实例, 平均纯度 {avg_purity:.1%}")
        
        lines.extend([
            "",
            "-" * 40,
            f"有效训练实例（等级 {self.target_levels}, 纯度 >= {self.purity_threshold:.0%}）",
            "-" * 40,
        ])
        
        valid_by_level = defaultdict(int)
        for inst in instances:
            valid_by_level[inst.level] += 1
        
        for level in self.target_levels:
            count = valid_by_level[level]
            lines.append(f"Level {level}: {count} 个有效实例")
        
        lines.extend([
            "",
            f"总有效实例数: {len(instances)}",
            "",
            "=" * 60,
        ])
        
        report = "\n".join(lines)
        print(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")
    
    def _save_training_manifest(self, instances: List[BuildingInstance]):
        """保存训练样本清单（JSON）"""
        manifest_path = self.output_dir / "training_manifest.json"
        
        def convert_to_native(obj):
            """递归转换 numpy 类型为 Python 原生类型"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 按等级分组
        by_level = defaultdict(list)
        for inst in instances:
            inst_dict = convert_to_native(asdict(inst))
            by_level[inst.level].append(inst_dict)
        
        manifest = {
            "config": {
                "purity_threshold": self.purity_threshold,
                "min_instance_pixels": self.min_instance_pixels,
                "target_levels": self.target_levels,
            },
            "summary": {
                "total": len(instances),
                "by_level": {str(k): len(v) for k, v in by_level.items()},
            },
            "instances": {str(k): v for k, v in by_level.items()},
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"训练清单已保存: {manifest_path}")


def main():
    """主函数 - 根据你的数据路径修改"""
    
    # ========== 修改这里的路径 ==========
    IMAGE_DIR = "./data/images"
    MASK_DIR = "./data/masks"
    OUTPUT_DIR = "./disaster_lora/data_analysis"
    # ====================================
    
    analyzer = DisasterDatasetAnalyzer(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        purity_threshold=0.7,      # 等级纯度阈值
        min_instance_pixels=500,   # 最小建筑像素数
        target_levels=[1, 2, 3, 4],   # 目标训练等级（含 L1）
    )
    
    # 分析数据集
    # max_samples=None 表示处理全部
    instances = analyzer.analyze_dataset(
        max_samples=None,      # 处理全部 3595 张
        visualize_count=20,    # 可视化前 20 张
    )
    
    print(f"\n完成！共找到 {len(instances)} 个有效训练实例")


if __name__ == "__main__":
    main()
