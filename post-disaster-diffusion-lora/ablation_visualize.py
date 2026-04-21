"""
消融实验可视化模块

功能：
1. Control Grid（对比 A0/A1/A2）
2. L4 Adapter Zoom（对比 A2/A3 的边界）
3. Sweep Grid（参数扫描结果）

输出：
- viz/control_grid.png
- viz/l4_adapter_zoom.png
- viz/sweep_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2


class AblationVisualizer:
    """消融实验可视化器"""
    
    def __init__(self, overlay_mask: bool = False, mask_color: Tuple[int, int, int] = (255, 0, 0), mask_alpha: float = 0.35):
        self.overlay_mask = overlay_mask
        self.mask_color = mask_color
        self.mask_alpha = mask_alpha

        # 字体（如果系统没有，用默认）
        try:
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def create_control_grid(self, root_dir: Path, output_path: Path):
        """
        创建 Control Grid（对比 A0/A1/A2）
        
        布局（4 rows x 8 cols）：
        Row 1: srcL1->tgtL1 | A0 | A1 | A2 | srcL1->tgtL2 | A0 | A1 | A2
        Row 2: srcL1->tgtL3 | A0 | A1 | A2 | srcL1->tgtL4 | A0 | A1 | A2
        Row 3: srcL2->tgtL1 | A0 | A1 | A2 | srcL2->tgtL2 | A0 | A1 | A2
        Row 4: srcL3->tgtL3 | A0 | A1 | A2 | srcL4->tgtL4 | A0 | A1 | A2
        """
        print("  生成 Control Grid...")

        roots = self._normalize_roots(root_dir)
        exp_dir_a0 = self._resolve_exp_dir(roots, "A0")
        exp_dir_a1 = self._resolve_exp_dir(roots, "A1")
        exp_dir_a2 = self._resolve_exp_dir(roots, "A2")
        if exp_dir_a0 is None or exp_dir_a1 is None or exp_dir_a2 is None:
            print("  缺少 A0/A1/A2 目录，无法生成 Control Grid")
            return
        
        # 选择样本（代表性的 sample_id）
        configs = [
            ('L1', 1, 1), ('L1', 1, 2), ('L1', 1, 3), ('L1', 1, 4),
            ('L2', 2, 1), ('L2', 2, 2), ('L2', 2, 3), ('L2', 2, 4),
        ]
        
        cell_size = 192
        rows = 2
        cols = 16  # 每个配置 4 张（原图+A0+A1+A2）
        
        canvas = Image.new('RGB', (cell_size * cols, cell_size * rows), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        for row, (src_label, src_level, tgt_level) in enumerate(configs):
            base_col = (row % 4) * 4
            row_idx = row // 4
            
            # 查找样本
            sample = self._find_sample(exp_dir_a0, src_level, tgt_level)
            if sample is None:
                continue
            sample_id = sample["sample_id"]
            mask_path = sample.get("mask_path")
            
            # 加载原图
            orig_path = sample.get("image_path")
            if orig_path:
                orig_img = self._load_image_with_mask(orig_path, cell_size, mask_path)
                canvas.paste(orig_img, (base_col * cell_size, row_idx * cell_size))
            
            # 加载 A0/A1/A2
            for i, exp_id in enumerate(['A0', 'A1', 'A2']):
                exp_dir = self._resolve_exp_dir(roots, exp_id)
                if exp_dir is None:
                    continue
                gen_path = exp_dir / "gen" / f"tgt_L{tgt_level}" / f"src_L{src_level}_{sample_id}.png"
                if gen_path.exists():
                    img = self._load_image_with_mask(gen_path, cell_size, mask_path)
                    canvas.paste(img, ((base_col + i + 1) * cell_size, row_idx * cell_size))
            
            # 标题
            title = f"L{src_level}→L{tgt_level}"
            draw.text((base_col * cell_size + 10, row_idx * cell_size + 10), title, fill=(255, 255, 255), font=self.font_medium)
        
        canvas.save(output_path, quality=95)
        print(f"    ✅ 保存到: {output_path}")
    
    def create_l4_adapter_zoom(self, root_dir: Path, output_path: Path):
        """
        创建 L4 Adapter Zoom（对比 A2/A3 的边界细节）
        
        布局：
        Row 1: 原图 | Mask | A2 (w/ L4) | A3 (wo/ L4)
        Row 2: 边界放大 x3（每个方法）
        """
        print("  生成 L4 Adapter Zoom...")

        roots = self._normalize_roots(root_dir)
        exp_dir_a2 = self._resolve_exp_dir(roots, "A2")
        exp_dir_a3 = self._resolve_exp_dir(roots, "A3")
        if exp_dir_a2 is None or exp_dir_a3 is None:
            print("  缺少 A2/A3 目录，无法生成 L4 Adapter Zoom")
            return
        
        # 选择 L1->L4, L2->L4, L3->L4, L4->L4 各一个样本
        samples = [
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4)
        ]
        
        cell_size = 256
        rows = len(samples) * 2  # 每个样本 2 行
        cols = 4
        
        canvas = Image.new('RGB', (cell_size * cols, cell_size * rows), (255, 255, 255))
        
        for i, (src_level, tgt_level) in enumerate(samples):
            sample = self._find_sample(exp_dir_a3, src_level, tgt_level)
            if sample is None:
                continue
            sample_id = sample["sample_id"]
            orig_path = sample.get("image_path")
            mask_path = sample.get("mask_path")
            
            row_base = i * 2
            
            # Row 1: 原图 | Mask | A2 | A3
            # 原图
            if orig_path:
                orig_img = self._load_image_with_mask(orig_path, cell_size, mask_path)
                canvas.paste(orig_img, (0, row_base * cell_size))
            
            # Mask
            if mask_path:
                mask_img = Image.open(mask_path).convert('L').resize((cell_size, cell_size))
                mask_rgb = Image.merge('RGB', (mask_img, mask_img, mask_img))
                canvas.paste(mask_rgb, (cell_size, row_base * cell_size))
            
            # A2
            a2_path = exp_dir_a2 / "gen" / f"tgt_L{tgt_level}" / f"src_L{src_level}_{sample_id}.png"
            if a2_path.exists():
                a2_img = self._load_image_with_mask(a2_path, cell_size, mask_path)
                canvas.paste(a2_img, (cell_size * 2, row_base * cell_size))
            
            # A3
            a3_path = exp_dir_a3 / "gen" / f"tgt_L{tgt_level}" / f"src_L{src_level}_{sample_id}.png"
            if a3_path.exists():
                a3_img = self._load_image_with_mask(a3_path, cell_size, mask_path)
                canvas.paste(a3_img, (cell_size * 3, row_base * cell_size))
            
            # Row 2: 边界放大（取 mask 边界区域）
            if orig_path and mask_path and a2_path.exists() and a3_path.exists():
                zoom_orig = self._extract_boundary_zoom(orig_path, mask_path, cell_size)
                zoom_a2 = self._extract_boundary_zoom(a2_path, mask_path, cell_size)
                zoom_a3 = self._extract_boundary_zoom(a3_path, mask_path, cell_size)
                
                canvas.paste(zoom_orig, (0, (row_base + 1) * cell_size))
                canvas.paste(zoom_orig, (cell_size, (row_base + 1) * cell_size))  # Mask 列用原图
                canvas.paste(zoom_a2, (cell_size * 2, (row_base + 1) * cell_size))
                canvas.paste(zoom_a3, (cell_size * 3, (row_base + 1) * cell_size))
        
        canvas.save(output_path, quality=95)
        print(f"    ✅ 保存到: {output_path}")
    
    def create_sweep_grid(self, root_dir: Path, exp_id: str, param_name: str, param_values: List, output_path: Path):
        """
        创建 Sweep Grid（参数扫描对比）
        
        布局（rows = len(param_values), cols = 4 个转换）：
        每行：param_value | L1->L3 | L2->L2 | L3->L3 | L4->L4
        """
        print(f"  生成 Sweep Grid: {param_name}...")

        roots = self._normalize_roots(root_dir)
        
        # 选择代表性转换
        configs = [
            (1, 3),
            (2, 2),
            (3, 3),
            (4, 4)
        ]
        
        cell_size = 192
        rows = len(param_values)
        cols = len(configs) + 1  # +1 for label
        
        canvas = Image.new('RGB', (cell_size * cols, cell_size * rows), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        for row, param_value in enumerate(param_values):
            # 左侧标签
            label = f"{param_name}={param_value}"
            draw.text((10, row * cell_size + cell_size // 2 - 10), label, fill=(0, 0, 0), font=self.font_medium)
            
            # 实验目录（分组结构：A4/guidance_scale_10/）
            if isinstance(param_value, float):
                setting_name = f"{param_name}_{param_value:.2f}".replace('.', '_')
            else:
                setting_name = f"{param_name}_{param_value}"
            
            exp_dir = self._resolve_exp_dir(roots, exp_id, setting_name)
            if exp_dir is None:
                continue
            
            for col, (src_level, tgt_level) in enumerate(configs):
                sample = self._find_sample(exp_dir, src_level, tgt_level)
                if sample is None:
                    continue
                sample_id = sample["sample_id"]
                mask_path = sample.get("mask_path")
                
                gen_path = exp_dir / "gen" / f"tgt_L{tgt_level}" / f"src_L{src_level}_{sample_id}.png"
                if gen_path.exists():
                    img = self._load_image_with_mask(gen_path, cell_size, mask_path)
                    canvas.paste(img, ((col + 1) * cell_size, row * cell_size))
        
        canvas.save(output_path, quality=95)
        print(f"    ✅ 保存到: {output_path}")
    
    def _find_sample(self, exp_dir: Path, src_level: int, tgt_level: int) -> Optional[Dict]:
        """查找样本"""
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for sample in manifest['samples']:
            if sample['source_level'] == src_level and sample['target_level'] == tgt_level:
                return sample
        
        return None

    def _find_sample_id(self, exp_dir: Path, src_level: int, tgt_level: int) -> Optional[str]:
        sample = self._find_sample(exp_dir, src_level, tgt_level)
        if not sample:
            return None
        return sample.get('sample_id')
    
    def _find_original_image(self, exp_dir: Path, src_level: int, tgt_level: int, sample_id: str) -> str:
        """查找原始图像"""
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for sample in manifest['samples']:
            if sample['sample_id'] == sample_id:
                return sample['image_path']
        
        return None
    
    def _find_mask_image(self, exp_dir: Path, src_level: int, tgt_level: int, sample_id: str) -> str:
        """查找 mask"""
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for sample in manifest['samples']:
            if sample['sample_id'] == sample_id:
                return sample['mask_path']
        
        return None

    def _normalize_roots(self, root_dir: Path) -> List[Path]:
        if isinstance(root_dir, (list, tuple)):
            return [Path(p) for p in root_dir]
        return [Path(root_dir)]

    def _resolve_exp_dir(self, roots: Sequence[Path], exp_id: str, setting: Optional[str] = None) -> Optional[Path]:
        for root in roots:
            if setting:
                candidate = root / exp_id / setting
            else:
                candidate = root / exp_id
            if candidate.exists():
                return candidate
        return None

    def _load_image_with_mask(self, image_path: str, cell_size: int, mask_path: Optional[str]) -> Image.Image:
        img = Image.open(image_path).convert('RGB').resize((cell_size, cell_size))
        if self.overlay_mask and mask_path:
            img = self._overlay_mask(img, mask_path)
        return img

    def _overlay_mask(self, image: Image.Image, mask_path: str) -> Image.Image:
        if not mask_path or not Path(mask_path).exists():
            return image
        mask = Image.open(mask_path).convert('L').resize(image.size, Image.NEAREST)
        mask_np = np.array(mask)
        alpha = int(255 * self.mask_alpha)

        unique_vals = set(np.unique(mask_np).tolist())
        is_multiclass = any(v in unique_vals for v in (2, 3, 4))

        if not is_multiclass:
            mask_alpha = (mask_np > 0).astype(np.uint8) * alpha
            overlay = Image.new("RGBA", image.size, self.mask_color + (0,))
            overlay.putalpha(Image.fromarray(mask_alpha, mode="L"))
            base = image.convert("RGBA")
            return Image.alpha_composite(base, overlay).convert("RGB")

        color_map = {
            1: (0, 255, 0),     # green
            2: (255, 255, 0),   # yellow
            3: (0, 0, 255),     # blue
            4: (255, 0, 0)      # red
        }
        overlay_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        overlay_alpha = np.zeros(mask_np.shape, dtype=np.uint8)
        for value, color in color_map.items():
            value_mask = mask_np == value
            overlay_rgb[value_mask] = color
            overlay_alpha[value_mask] = alpha

        overlay = Image.fromarray(overlay_rgb, mode="RGB").convert("RGBA")
        overlay.putalpha(Image.fromarray(overlay_alpha, mode="L"))
        base = image.convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")
    
    def _extract_boundary_zoom(self, image_path: str, mask_path: str, output_size: int) -> Image.Image:
        """提取边界区域并放大"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 512))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 找到 mask 的边界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((output_size, output_size))
        
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # 扩展边界框（+30px）
        margin = 30
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(512, x + w + margin)
        y2 = min(512, y + h + margin)
        
        # 裁剪
        crop = img[y1:y2, x1:x2]
        
        # 转为 PIL 并 resize
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_resized = crop_pil.resize((output_size, output_size), Image.LANCZOS)
        
        return crop_resized


# ============== 命令行测试 ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="runs_ablation 根目录")
    parser.add_argument("--root_dirs", type=str, nargs='+', help="多个 runs_ablation 根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--mode", type=str, choices=['control', 'l4_zoom', 'sweep'], required=True)
    parser.add_argument("--exp_id", type=str, help="实验 ID（sweep 模式需要）")
    parser.add_argument("--param_name", type=str, help="参数名（sweep 模式需要）")
    parser.add_argument("--param_values", nargs='+', help="参数值列表（sweep 模式需要）")
    parser.add_argument("--overlay_mask", action="store_true", help="叠加 mask 区域")
    
    args = parser.parse_args()

    root_dirs = args.root_dirs or ([args.root_dir] if args.root_dir else None)
    if not root_dirs:
        parser.error("需要提供 --root_dir 或 --root_dirs")

    visualizer = AblationVisualizer(overlay_mask=args.overlay_mask)
    root_dir = [Path(p) for p in root_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'control':
        visualizer.create_control_grid(root_dir, output_dir / "control_grid.png")
    
    elif args.mode == 'l4_zoom':
        visualizer.create_l4_adapter_zoom(root_dir, output_dir / "l4_adapter_zoom.png")
    
    elif args.mode == 'sweep':
        # 解析参数值
        param_values = []
        for v in args.param_values:
            try:
                param_values.append(float(v))
            except:
                param_values.append(int(v))
        
        visualizer.create_sweep_grid(root_dir, args.exp_id, args.param_name, param_values, output_dir / f"sweep_{args.param_name}.png")

