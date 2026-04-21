"""
完整消融实验运行器 (A0-A7)

功能：
1. 按统一配额生成样本（4×4 源→目标等级矩阵）
2. 严格目录结构（refs/ + 每个实验的 config/manifest/gen/viz/metrics）
3. 自动化指标计算（KID, FID, LPIPS, 背景保持, 边界伪影）
4. 论文图生成（网格图、边界放大、扫参图）
5. 最终汇总表格

Usage:
    python disaster_lora/ablation_runner.py --experiment A0 --output_dir runs_ablation
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import cv2
from PIL import Image

# 导入推理相关模块
sys.path.insert(0, str(Path(__file__).parent))


# ============== 配置定义 ==============

@dataclass
class BestParams:
    """Best 参数（默认基准）"""
    padding: int = 50
    strength: float = 0.9
    guidance_scale: float = 10.0
    num_inference_steps: int = 50
    mask_dilate_kernel: int = 25
    feather_pixels: int = 25


@dataclass
class GenerationQuota:
    """生成配额（4×4 源→目标矩阵）"""
    L1_to: List[int]  # [L1->L1, L1->L2, L1->L3, L1->L4]
    L2_to: List[int]
    L3_to: List[int]
    L4_to: List[int]
    
    def total(self) -> int:
        return sum(self.L1_to + self.L2_to + self.L3_to + self.L4_to)
    
    def by_target(self) -> Dict[int, int]:
        """按目标等级统计"""
        return {
            1: self.L1_to[0] + self.L2_to[0] + self.L3_to[0] + self.L4_to[0],
            2: self.L1_to[1] + self.L2_to[1] + self.L3_to[1] + self.L4_to[1],
            3: self.L1_to[2] + self.L2_to[2] + self.L3_to[2] + self.L4_to[2],
            4: self.L1_to[3] + self.L2_to[3] + self.L3_to[3] + self.L4_to[3],
        }


# 预定义配额
QUOTA_MAIN = GenerationQuota(
    L1_to=[100, 100, 100, 100],
    L2_to=[100, 100, 100, 100],
    L3_to=[100, 100, 100, 100],
    L4_to=[50, 50, 50, 50],
)  # 总 1400 张

QUOTA_L4_ONLY = GenerationQuota(
    L1_to=[0, 0, 0, 150],
    L2_to=[0, 0, 0, 150],
    L3_to=[0, 0, 0, 150],
    L4_to=[0, 0, 0, 50],
)  # 总 500 张（只生成 tgtL4）

QUOTA_SWEEP = GenerationQuota(
    L1_to=[20, 20, 20, 20],
    L2_to=[20, 20, 20, 20],
    L3_to=[20, 20, 20, 20],
    L4_to=[10, 10, 10, 10],
)  # 总 280 张


# Prompt 规则（完整四级）
PROMPTS_BASELINE = {
    1: {
        "positive": "aerial view of intact building, no damage, complete structure",
        "negative": "damaged, collapsed, destroyed, debris, rubble"
    },
    2: {
        "positive": "aerial view of minor damaged building, slight roof damage",
        "negative": "collapsed, destroyed, rubble, debris field"
    },
    3: {
        "positive": "aerial view of major damaged building, severe structural damage",
        "negative": "intact roof, undamaged, perfect condition"
    },
    4: {
        "positive": "aerial view of completely destroyed building, rubble and debris",
        "negative": "house, structure, standing walls, intact roof"
    },
}

PROMPTS_WITH_TOKEN = {
    1: {
        "positive": "aerial view of intact building <L1>",
        "negative": "damaged, collapsed, destroyed <L4>"
    },
    2: {
        "positive": "aerial view of minor damaged building <L2>",
        "negative": "collapsed, destroyed, rubble, debris field"
    },
    3: {
        "positive": "aerial view of major damaged building <L3>",
        "negative": "intact roof, undamaged, perfect condition"
    },
    4: {
        "positive": "aerial view, debris field, completely destroyed building, no intact roof, scattered rubble, dust, irregular boundary <L4>",
        "negative": "intact roof, floorplan, rooms, clean interior, architectural drawing, blueprint, undamaged, organized structure"
    },
}


# ============== 实验配置 ==============

class ExperimentConfig:
    """实验配置基类"""
    def __init__(self, exp_id: str, name: str, description: str):
        self.exp_id = exp_id
        self.name = name
        self.description = description
        self.params = BestParams()
        self.quota = QUOTA_MAIN
        self.prompts = PROMPTS_WITH_TOKEN
        self.seed = 0
        
        # 模型相关
        self.use_baseline = False  # A0 用
        self.stage1_only = False   # A1 用
        self.force_shared_only = False  # A3 用
        
        # Checkpoint 路径（需要在运行时设置）
        self.stage1_checkpoint = None
        self.stage2_checkpoint = None
    
    def to_dict(self) -> Dict:
        """转为字典（保存到 config.json）"""
        return {
            "exp_id": self.exp_id,
            "name": self.name,
            "description": self.description,
            "params": asdict(self.params),
            "quota": {
                "L1_to": self.quota.L1_to,
                "L2_to": self.quota.L2_to,
                "L3_to": self.quota.L3_to,
                "L4_to": self.quota.L4_to,
                "total": self.quota.total(),
                "by_target": self.quota.by_target(),
            },
            "prompts": self.prompts,
            "seed": self.seed,
            "model": {
                "use_baseline": self.use_baseline,
                "stage1_only": self.stage1_only,
                "force_shared_only": self.force_shared_only,
                "stage1_checkpoint": self.stage1_checkpoint,
                "stage2_checkpoint": self.stage2_checkpoint,
            }
        }


# ============== 具体实验配置 ==============

def get_experiment_config(exp_id: str, args) -> ExperimentConfig:
    """获取实验配置"""
    
    if exp_id == "A0":
        config = ExperimentConfig(
            "A0", 
            "Overall Baseline",
            "Original SD-Inpainting without LoRA"
        )
        config.use_baseline = True
        config.prompts = PROMPTS_BASELINE
        config.quota = QUOTA_MAIN
        return config
    
    elif exp_id == "A1":
        config = ExperimentConfig(
            "A1",
            "Stage1 Shared Adapter",
            "Stage1 shared LoRA only"
        )
        config.stage1_only = True
        config.stage1_checkpoint = args.stage1_checkpoint
        config.quota = QUOTA_MAIN
        return config
    
    elif exp_id == "A2":
        config = ExperimentConfig(
            "A2",
            "Final Method (Stage 2 + Routed L4)",
            "Stage 2 merged base + L4 adapter routing"
        )
        config.stage2_checkpoint = args.stage2_checkpoint
        config.quota = QUOTA_MAIN
        return config
    
    elif exp_id == "A3":
        config = ExperimentConfig(
            "A3",
            "L4 Adapter Ablation (Shared Only)",
            "Force shared adapter for all targets (no L4 adapter)"
        )
        config.stage2_checkpoint = args.stage2_checkpoint
        config.force_shared_only = True
        config.quota = QUOTA_L4_ONLY  # 只生成 tgtL4
        return config
    
    elif exp_id.startswith("A4"):
        # CFG Sweep: A4_cfg0, A4_cfg5, A4_cfg7, A4_cfg10, A4_cfg12
        cfg_value = float(exp_id.split("_cfg")[1])
        config = ExperimentConfig(
            exp_id,
            f"CFG Sweep (cfg={cfg_value})",
            f"Guidance scale sweep: cfg={cfg_value}"
        )
        config.stage2_checkpoint = args.stage2_checkpoint
        config.params.guidance_scale = cfg_value
        config.quota = QUOTA_SWEEP
        return config
    
    elif exp_id.startswith("A5"):
        # Strength Sweep: A5_s075, A5_s085, A5_s090, A5_s095
        strength_value = float(exp_id.split("_s")[1]) / 100
        config = ExperimentConfig(
            exp_id,
            f"Strength Sweep (strength={strength_value})",
            f"Denoising strength sweep: {strength_value}"
        )
        config.stage2_checkpoint = args.stage2_checkpoint
        config.params.strength = strength_value
        config.quota = QUOTA_SWEEP
        return config
    
    elif exp_id.startswith("A6"):
        # Steps: A6_steps25, A6_steps50
        steps_value = int(exp_id.split("_steps")[1])
        config = ExperimentConfig(
            exp_id,
            f"Steps Comparison (steps={steps_value})",
            f"Inference steps: {steps_value}"
        )
        config.stage2_checkpoint = args.stage2_checkpoint
        config.params.num_inference_steps = steps_value
        config.quota = QUOTA_SWEEP
        return config
    
    elif exp_id.startswith("A7"):
        # Dilate: A7_k0, A7_k15, A7_k25, A7_k40
        kernel_value = int(exp_id.split("_k")[1])
        config = ExperimentConfig(
            exp_id,
            f"Mask Dilation Sweep (kernel={kernel_value})",
            f"Mask dilate kernel: {kernel_value}"
        )
        config.stage2_checkpoint = args.stage2_checkpoint
        config.params.mask_dilate_kernel = kernel_value
        config.quota = QUOTA_SWEEP
        return config
    
    else:
        raise ValueError(f"Unknown experiment ID: {exp_id}")


# ============== 主运行器 ==============

class AblationRunner:
    """消融实验运行器"""
    
    def __init__(self, root_dir: str, data_dir: str):
        self.root_dir = Path(root_dir)
        self.data_dir = Path(data_dir)
        
        # 创建根目录
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # 公共引用目录（只生成一次）
        self.refs_dir = self.root_dir / "refs"
        self.refs_dir.mkdir(exist_ok=True)
    
    def prepare_refs(self):
        """准备公共引用数据（只运行一次）"""
        print("\n" + "="*60)
        print("准备公共引用数据...")
        print("="*60)
        
        # 读取 metadata.json
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 按等级分组
        samples_by_level = {1: [], 2: [], 3: [], 4: []}
        for sample in metadata['samples']:
            level = sample['level']
            samples_by_level[level].append(sample)
        
        # 复制或链接到 refs/
        for level in [1, 2, 3, 4]:
            level_dir = self.refs_dir / f"real_by_level/L{level}"
            level_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  L{level}: {len(samples_by_level[level])} samples")
        
        # 保存 refs manifest
        refs_manifest = {
            "samples_by_level": {
                str(k): [s['image_path'] for s in v] 
                for k, v in samples_by_level.items()
            },
            "total": sum(len(v) for v in samples_by_level.values())
        }
        
        with open(self.refs_dir / "manifest.json", 'w') as f:
            json.dump(refs_manifest, f, indent=2)
        
        print(f"✅ 公共引用数据准备完成: {self.refs_dir}")
        return samples_by_level
    
    def run_experiment(self, config: ExperimentConfig, samples_by_level: Dict):
        """运行单个实验"""
        print("\n" + "="*80)
        print(f"运行实验: {config.exp_id} - {config.name}")
        print("="*80)
        print(f"描述: {config.description}")
        print(f"配额: {config.quota.total()} 张")
        print(f"参数: {asdict(config.params)}")
        
        # 创建实验目录
        exp_dir = self.root_dir / config.exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # 保存 config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"✅ Config 已保存: {config_path}")
        
        # 采样样本（按配额）
        manifest = self._sample_manifest(samples_by_level, config.quota, config.seed)
        manifest_path = exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ Manifest 已保存: {manifest_path} ({len(manifest['samples'])} samples)")
        
        # 创建生成目录
        gen_dir = exp_dir / "gen"
        for target_level in [1, 2, 3, 4]:
            (gen_dir / f"tgt_L{target_level}").mkdir(parents=True, exist_ok=True)
        
        # TODO: 调用推理（下一步实现）
        print(f"\n⚠️  推理部分待实现（需要加载模型并生成）")
        print(f"   生成目录: {gen_dir}")
        
        # TODO: 可视化（下一步实现）
        viz_dir = exp_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        print(f"   可视化目录: {viz_dir}")
        
        # TODO: 指标计算（下一步实现）
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        print(f"   指标目录: {metrics_dir}")
        
        print(f"\n✅ 实验 {config.exp_id} 框架搭建完成")
        return exp_dir
    
    def _sample_manifest(self, samples_by_level: Dict, quota: GenerationQuota, seed: int) -> Dict:
        """采样生成 manifest"""
        np.random.seed(seed)
        
        manifest_samples = []
        
        # L1 源
        for target_idx, target_level in enumerate([1, 2, 3, 4]):
            n_needed = quota.L1_to[target_idx]
            if n_needed > 0:
                sampled = np.random.choice(samples_by_level[1], n_needed, replace=False)
                for s in sampled:
                    manifest_samples.append({
                        "source_level": 1,
                        "target_level": target_level,
                        "image_path": s['image_path'],
                        "mask_path": s['mask_path'],
                        "sample_id": s['sample_id']
                    })
        
        # L2 源
        for target_idx, target_level in enumerate([1, 2, 3, 4]):
            n_needed = quota.L2_to[target_idx]
            if n_needed > 0:
                sampled = np.random.choice(samples_by_level[2], n_needed, replace=False)
                for s in sampled:
                    manifest_samples.append({
                        "source_level": 2,
                        "target_level": target_level,
                        "image_path": s['image_path'],
                        "mask_path": s['mask_path'],
                        "sample_id": s['sample_id']
                    })
        
        # L3 源
        for target_idx, target_level in enumerate([1, 2, 3, 4]):
            n_needed = quota.L3_to[target_idx]
            if n_needed > 0:
                sampled = np.random.choice(samples_by_level[3], n_needed, replace=False)
                for s in sampled:
                    manifest_samples.append({
                        "source_level": 3,
                        "target_level": target_level,
                        "image_path": s['image_path'],
                        "mask_path": s['mask_path'],
                        "sample_id": s['sample_id']
                    })
        
        # L4 源
        for target_idx, target_level in enumerate([1, 2, 3, 4]):
            n_needed = quota.L4_to[target_idx]
            if n_needed > 0:
                sampled = np.random.choice(samples_by_level[4], n_needed, replace=False)
                for s in sampled:
                    manifest_samples.append({
                        "source_level": 4,
                        "target_level": target_level,
                        "image_path": s['image_path'],
                        "mask_path": s['mask_path'],
                        "sample_id": s['sample_id']
                    })
        
        return {
            "total": len(manifest_samples),
            "by_source_target": {
                f"L{src}->L{tgt}": sum(1 for s in manifest_samples if s['source_level']==src and s['target_level']==tgt)
                for src in [1,2,3,4] for tgt in [1,2,3,4]
            },
            "samples": manifest_samples
        }


# ============== 命令行入口 ==============

def main():
    parser = argparse.ArgumentParser(description="消融实验运行器 (A0-A7)")
    parser.add_argument("--experiment", type=str, required=True, 
                        help="实验ID: A0, A1, A2, A3, A4_cfg7, A5_s090, etc.")
    parser.add_argument("--output_dir", type=str, default="runs_ablation",
                        help="输出根目录")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="训练数据目录（含 metadata.json）")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage1 checkpoint 路径（A1用）")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Stage 2 checkpoint 路径（A2/A3用）")
    parser.add_argument("--prepare_refs_only", action="store_true",
                        help="只准备公共引用数据")
    
    args = parser.parse_args()
    
    # 初始化 Runner
    runner = AblationRunner(args.output_dir, args.data_dir)
    
    # 准备公共引用（第一次运行时）
    refs_manifest_path = runner.refs_dir / "manifest.json"
    if not refs_manifest_path.exists() or args.prepare_refs_only:
        samples_by_level = runner.prepare_refs()
        if args.prepare_refs_only:
            print("\n✅ 公共引用数据准备完成，退出")
            return
    else:
        # 加载已有的 refs manifest
        with open(refs_manifest_path, 'r') as f:
            refs_manifest = json.load(f)
        
        # 重新加载 samples_by_level（简化版）
        metadata_path = Path(args.data_dir) / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        samples_by_level = {1: [], 2: [], 3: [], 4: []}
        for sample in metadata['samples']:
            samples_by_level[sample['level']].append(sample)
    
    # 获取实验配置
    config = get_experiment_config(args.experiment, args)
    
    # 运行实验
    runner.run_experiment(config, samples_by_level)
    
    print("\n" + "="*80)
    print("✅ 实验框架搭建完成！")
    print("="*80)
    print(f"输出目录: {args.output_dir}/{args.experiment}")
    print("\n下一步：")
    print("  1. 实现推理部分（加载模型并生成）")
    print("  2. 实现可视化部分（网格图、边界放大）")
    print("  3. 实现指标计算部分（KID/FID/LPIPS/背景保持/边界伪影）")
    print("  4. 实现汇总部分（summary表格和图表）")


if __name__ == "__main__":
    main()

