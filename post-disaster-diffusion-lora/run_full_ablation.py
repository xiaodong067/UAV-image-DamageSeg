"""
完整消融实验一键运行脚本

一个命令完成所有任务：
1. 准备公共引用数据
2. 运行所有实验（A0-A7）
3. 批量推理生成样本
4. 计算所有指标
5. 生成论文可视化图
6. 汇总结果表格

Usage:
    python disaster_lora/run_full_ablation.py \
        --config disaster_lora/ablation_config.yaml \
        --output runs_ablation
"""

import os
import sys
import json
import yaml
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


class FullAblationPipeline:
    """完整消融实验流程"""
    
    def __init__(self, config_path: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)
        
        self.start_time = datetime.now()
        self.log_file = self.output_dir / "pipeline.log"
        
        # 实验列表
        self.experiments = self._build_experiment_list()
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def _build_experiment_list(self) -> List[Dict]:
        """构建实验列表"""
        experiments = []
        
        # A0-A3 主对比
        experiments.extend([
            {"id": "A0", "name": "Baseline", "type": "main", "group": "A0"},
            {"id": "A1", "name": "Stage1", "type": "main", "group": "A1"},
            {"id": "A2", "name": "Final", "type": "main", "group": "A2"},
            {"id": "A3", "name": "L4Ablation", "type": "l4only", "group": "A3"},
        ])
        
        # A4: CFG Sweep（分组结构：A4/guidance_scale_X/）
        for cfg in [0, 5, 7, 10, 12]:
            experiments.append({
                "id": "A4",
                "group": "A4",
                "setting": f"guidance_scale_{cfg}",
                "name": f"CFG{cfg}",
                "type": "sweep",
                "param": "guidance_scale",
                "value": cfg
            })
        
        # A5: Strength Sweep（分组结构：A5/strength_X/）
        for s in [0.75, 0.85, 0.90, 0.95]:
            experiments.append({
                "id": "A5",
                "group": "A5",
                "setting": f"strength_{s:.2f}".replace('.', '_'),
                "name": f"Strength{int(s*100)}",
                "type": "sweep",
                "param": "strength",
                "value": s
            })
        
        # A6: Steps（分组结构：A6/num_inference_steps_X/）
        for steps in [25, 50]:
            experiments.append({
                "id": "A6",
                "group": "A6",
                "setting": f"num_inference_steps_{steps}",
                "name": f"Steps{steps}",
                "type": "sweep",
                "param": "num_inference_steps",
                "value": steps
            })
        
        # A7: Dilation（分组结构：A7/mask_dilate_kernel_X/）
        for k in [0, 15, 25, 40]:
            experiments.append({
                "id": "A7",
                "group": "A7",
                "setting": f"mask_dilate_kernel_{k}",
                "name": f"Dilate{k}",
                "type": "sweep",
                "param": "mask_dilate_kernel",
                "value": k
            })
        
        return experiments
    
    def run_all(self):
        """运行完整流程"""
        self.log("="*80)
        self.log("完整消融实验流程启动")
        self.log("="*80)
        self.log(f"输出目录: {self.output_dir}")
        self.log(f"总实验数: {len(self.experiments)}")
        self.log("")
        
        try:
            # Step 1: 准备公共引用
            self.log("Step 1/5: 准备公共引用数据...")
            self.prepare_refs()
            
            # Step 2: 运行所有实验（推理）
            self.log("\nStep 2/5: 运行所有实验...")
            for i, exp in enumerate(self.experiments, 1):
                self.log(f"\n[{i}/{len(self.experiments)}] 实验 {exp['id']}: {exp['name']}")
                self.run_single_experiment(exp)
            # Step 3: metrics
            if not getattr(self, "skip_metrics", False):
                self.log("\nStep 3/5: Metrics...")
                self.compute_all_metrics()
            else:
                self.log("\nStep 3/5: Skip metrics")

            # Step 4: visualization
            if not getattr(self, "skip_viz", False):
                self.log("\nStep 4/5: Visualization...")
                self.generate_all_visualizations()
            else:
                self.log("\nStep 4/5: Skip visualization")

            # Step 5: summary
            if not getattr(self, 'skip_summary', False):
                self.log("\nStep 5/5: Summary...")
                self.summarize_results()
            
            # 完成
            elapsed = datetime.now() - self.start_time
            self.log("")
            self.log("="*80)
            self.log("✅ 完整流程完成！")
            self.log("="*80)
            self.log(f"总耗时: {elapsed}")
            self.log(f"输出目录: {self.output_dir}")
            self.log(f"日志文件: {self.log_file}")
            
        except Exception as e:
            self.log(f"\n❌ 错误: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise
    
    def prepare_refs(self):
        """准备公共引用数据"""
        from ablation_runner import AblationRunner
        
        runner = AblationRunner(
            str(self.output_dir),
            self.config['data_dir']
        )
        
        samples_by_level = runner.prepare_refs()
        
        # 保存到实例变量，供后续使用
        self.samples_by_level = samples_by_level
        
        self.log(f"  L1: {len(samples_by_level[1])} samples")
        self.log(f"  L2: {len(samples_by_level[2])} samples")
        self.log(f"  L3: {len(samples_by_level[3])} samples")
        self.log(f"  L4: {len(samples_by_level[4])} samples")
        self.log(f"  ✅ 公共引用准备完成")
    

    def _load_metadata_samples(self) -> Dict[int, List[Dict]]:
        """??metadata.json ????????????"""
        metadata_path = Path(self.config['data_dir']) / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        samples_by_level = {1: [], 2: [], 3: [], 4: []}
        for sample in metadata['samples']:
            level = sample['level']
            samples_by_level[level].append(sample)
        return samples_by_level

    def _prepare_base_samples(self):
        """??? '?????' ?????"""
        base_cfg = self.config.get('base_sampling', {})
        if not base_cfg.get('enabled', False):
            return

        base_manifest_path = base_cfg.get('base_manifest')
        if base_manifest_path:
            base_manifest_path = Path(base_manifest_path)
        else:
            base_manifest_path = self.output_dir / "base_manifest.json"

        if base_manifest_path.exists():
            with open(base_manifest_path, 'r') as f:
                base_manifest = json.load(f)
            self.base_samples_by_level = {
                int(k): v for k, v in base_manifest.get('source_samples', {}).items()
            }
            return

        per_level = int(base_cfg.get('per_level', 150))
        seed = int(base_cfg.get('seed', 0))
        samples_by_level = self._load_metadata_samples()

        np.random.seed(seed)
        base_samples_by_level = {}
        for level in [1, 2, 3, 4]:
            pool = samples_by_level[level]
            if len(pool) < per_level:
                sampled = np.random.choice(pool, per_level, replace=True)
            else:
                sampled = np.random.choice(pool, per_level, replace=False)
            base_samples_by_level[level] = list(sampled)

        base_manifest = {
            "base_per_level": per_level,
            "seed": seed,
            "source_samples": {
                str(k): v for k, v in base_samples_by_level.items()
            }
        }
        with open(base_manifest_path, 'w') as f:
            json.dump(base_manifest, f, indent=2)

        self.base_samples_by_level = base_samples_by_level

    def _select_subset(self, samples: List[Dict], n: int, seed: int) -> List[Dict]:
        """??base ?????????"""
        if n >= len(samples):
            return samples
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))[:n]
        return [samples[i] for i in indices]


    def _create_manifest(self, exp_config: Dict) -> Dict:
        """?????? manifest"""
        quota_dict = exp_config['quota']
        seed = exp_config.get('seed', 0)
        base_cfg = self.config.get('base_sampling', {})

        # ???
        np.random.seed(seed)
        manifest_samples = []

        if base_cfg.get('enabled', False):
            if not hasattr(self, 'base_samples_by_level'):
                self._prepare_base_samples()

            subset_seed = int(base_cfg.get('subset_seed', seed))
            sweep_per_level = int(base_cfg.get('sweep_per_level', 50))

            for src_level in [1, 2, 3, 4]:
                base_samples = self.base_samples_by_level[src_level]
                if exp_config['type'] == 'sweep':
                    selected = self._select_subset(base_samples, sweep_per_level, subset_seed + src_level)
                else:
                    selected = base_samples

                for target_level in [1, 2, 3, 4]:
                    if exp_config['type'] == 'l4only' and target_level != 4:
                        continue
                    for s in selected:
                        manifest_samples.append({
                            "source_level": src_level,
                            "target_level": target_level,
                            "image_path": s['image_path'],
                            "mask_path": s['mask_path'],
                            "sample_id": s['sample_id']
                        })
        else:
            # ???????????refs??????
            if not hasattr(self, 'samples_by_level'):
                self.samples_by_level = self._load_metadata_samples()

            # ?????????????????
            for src_level in [1, 2, 3, 4]:
                quota_key = f"L{src_level}_to"
                quota_list = quota_dict[quota_key]

                for target_idx, target_level in enumerate([1, 2, 3, 4]):
                    n_needed = quota_list[target_idx]
                    if n_needed > 0:
                        available_samples = self.samples_by_level[src_level]
                        if len(available_samples) < n_needed:
                            # ?????????????????????
                            sampled = np.random.choice(available_samples, n_needed, replace=True)
                        else:
                            sampled = np.random.choice(available_samples, n_needed, replace=False)

                        for s in sampled:
                            manifest_samples.append({
                                "source_level": src_level,
                                "target_level": target_level,
                                "image_path": s['image_path'],
                                "mask_path": s['mask_path'],
                                "sample_id": s['sample_id']
                            })

        return {
            "exp_id": exp_config['exp_id'],
            "total": len(manifest_samples),
            "samples": manifest_samples
        }


    def run_single_experiment(self, exp: Dict):
        """????????? + ?? + ???"""
        exp_start = time.time()

        # ????????? sweep ????
        if exp['type'] == 'sweep':
            exp_dir = self.output_dir / exp['group'] / exp['setting']
            exp_id_display = f"{exp['group']}/{exp['setting']}"
        else:
            exp_dir = self.output_dir / exp['id']
            exp_id_display = exp['id']

        # 1. ??????
        if not (exp_dir / "config.json").exists():
            self.log(f"  1/3: ??????...")
            exp_config = self._build_exp_config(exp)
            exp_dir.mkdir(parents=True, exist_ok=True)

            with open(exp_dir / "config.json", 'w') as f:
                json.dump(exp_config, f, indent=2)

            manifest = self._create_manifest(exp_config)
            with open(exp_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)

            for target_level in [1, 2, 3, 4]:
                (exp_dir / "gen" / f"tgt_L{target_level}").mkdir(parents=True, exist_ok=True)
        else:
            self.log(f"  1/3: ??????????")

        # 2. ??
        gen_root = exp_dir / "gen"
        has_generated = gen_root.exists() and any(gen_root.rglob("*.png"))
        skip_inference = getattr(self, "skip_inference", False)
        if skip_inference or has_generated:
            reason = "????" if skip_inference else "???????????"
            self.log(f"  2/3: ???? ({reason})")
        else:
            self.log(f"  2/3: ????...")
            self._run_inference(exp_dir)

        # 3. ??
        if not getattr(self, 'skip_metrics', False):
            self.log(f"  3/3: ????...")
            self._run_metrics(exp_dir)
        else:
            self.log(f"  3/3: ????")

        elapsed = time.time() - exp_start
        self.log(f"  ? {exp_id_display} ?? (??: {elapsed:.1f}s)")

    def _build_exp_config(self, exp: Dict) -> Dict:

        """构建实验配置"""
        exp_id = exp.get('group', exp['id'])  # 使用 group 作为实验 ID
        
        config = {
            "exp_id": exp_id,
            "name": exp['name'],
            "type": exp['type'],
            "params": self.config['best_params'].copy(),
            "prompts": {},
            "model": {},
            "seed": self.config['best_params'].get('seed', 0)
        }
        
        # 模型配置
        if exp_id == "A0":
            config['model']['use_baseline'] = True
        elif exp_id == "A1":
            config['model']['use_baseline'] = False
            config['model']['stage1_only'] = True
            config['model']['stage1_checkpoint'] = self.config['stage1_checkpoint']
        else:
            # A2, A3, A4-A7 都用 Stage2
            config['model']['use_baseline'] = False
            config['model']['stage1_only'] = False
            config['model']['stage2_checkpoint'] = self.config['stage2_checkpoint']
        
        # A3: 强制 shared
        if exp_id == "A3":
            config['model']['force_shared_only'] = True
        
        # Sweep 参数覆盖
        if exp['type'] == "sweep":
            param_name = exp['param']
            param_value = exp['value']
            config['params'][param_name] = param_value
        
        # Prompts
        if exp_id == "A0":
            prompts_dict = self.config['prompts_A0_baseline']
        else:
            prompts_dict = self.config['prompts_A1_A2_A3']
        
        for level in [1, 2, 3, 4]:
            config['prompts'][level] = {
                'positive': prompts_dict[f'L{level}_pos'],
                'negative': prompts_dict[f'L{level}_neg']
            }
        
        # Quota（根据类型）
        if exp['type'] == 'l4only':
            config['quota'] = self.config['quotas']['l4_only']
        elif exp['type'] == 'sweep':
            config['quota'] = self.config['quotas']['sweep']
        else:
            config['quota'] = self.config['quotas']['main']
        
        return config
    
    def _run_inference(self, exp_dir: Path):
        """运行推理"""
        from ablation_inference import AblationInferenceEngine
        
        # 加载配置
        with open(exp_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        with open(exp_dir / "manifest.json", 'r') as f:
            manifest = json.load(f)
        
        # 初始化引擎
        engine = AblationInferenceEngine(config, device="cuda")
        
        # 生成
        engine.generate_all(manifest, exp_dir)
    
    def _run_metrics(self, exp_dir: Path):
        """计算指标"""
        from ablation_metrics import MetricCalculator
        
        with open(exp_dir / "manifest.json", 'r') as f:
            manifest = json.load(f)
        
        # 传入 refs_dir 用于 KID 计算
        refs_dir = self.output_dir / "refs"
        
        calculator = MetricCalculator(device="cuda")
        calculator.calculate_all_metrics(exp_dir, manifest, refs_dir)
    
    def compute_all_metrics(self):
        """计算所有实验的指标"""
        # 已在 run_single_experiment 中完成
        self.log("  指标已在各实验中计算完成")
        self.log("  ✅ 指标计算完成")
    
    def generate_all_visualizations(self):
        """生成所有论文可视化图"""
        from ablation_visualize import AblationVisualizer
        viz_roots = getattr(self, "viz_roots", None)
        if viz_roots:
            root_dirs = [Path(p) for p in viz_roots]
        else:
            root_dirs = [self.output_dir]

        visualizer = AblationVisualizer(overlay_mask=getattr(self, "viz_overlay_mask", False))
        viz_dir = self.output_dir / "summary" / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Control Grid（A0/A1/A2 对比）
        self.log("  生成 Control Grid...")
        try:
            visualizer.create_control_grid(root_dirs, viz_dir / "control_grid.png")
        except Exception as e:
            self.log(f"    警告: Control Grid 生成失败: {e}")
        
        # 2. L4 Adapter 对比（A2 vs A3）
        self.log("  生成 L4 Adapter 对比...")
        try:
            visualizer.create_l4_adapter_zoom(root_dirs, viz_dir / "l4_adapter_zoom.png")
        except Exception as e:
            self.log(f"    警告: L4 Adapter Zoom 生成失败: {e}")
        
        # 3. Sweep Grid（参数扫描）
        sweep_configs = [
            ("A4", "guidance_scale", [0, 5, 7, 10, 12]),
            ("A5", "strength", [0.75, 0.85, 0.90, 0.95]),
            ("A6", "num_inference_steps", [25, 50]),
            ("A7", "mask_dilate_kernel", [0, 15, 25, 40])
        ]
        
        for exp_id, param_name, param_values in sweep_configs:
            self.log(f"  生成 Sweep Grid: {param_name}...")
            try:
                visualizer.create_sweep_grid(
                    root_dirs, exp_id, param_name, param_values,
                    viz_dir / f"sweep_{param_name}.png"
                )
            except Exception as e:
                self.log(f"    警告: {param_name} Sweep 生成失败: {e}")
        
        self.log("  ✅ 可视化生成完成")
    
    def summarize_results(self):
        """汇总所有结果"""
        from ablation_summary import AblationSummarizer
        
        summarizer = AblationSummarizer(self.output_dir)
        summarizer.generate_summary()
        
        self.log(f"  ✅ 汇总完成: {self.output_dir / 'summary'}")


def main():
    parser = argparse.ArgumentParser(description="完整消融实验一键运行")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径 (YAML or JSON)")
    parser.add_argument("--output", type=str, default="runs_ablation",
                        help="输出根目录")
    parser.add_argument("--experiments", type=str, nargs='+', default=None,
                        help="指定运行的实验（默认全部）。例如: A0 A1 A2")
    parser.add_argument("--skip_refs", action="store_true",
                        help="跳过准备公共引用（如果已准备）")
    parser.add_argument("--skip_inference", action="store_true",
                        help="跳过推理（只计算指标和可视化）")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="??????")
    parser.add_argument("--skip_viz", action="store_true",
                        help="?????")
    parser.add_argument("--skip_summary", action="store_true",
                        help="????")
    parser.add_argument("--viz_roots", type=str, nargs='+', default=None,
                        help="可视化输入根目录列表（默认使用 --output）")
    parser.add_argument("--viz_overlay_mask", action="store_true",
                        help="可视化叠加 mask 区域")

    args = parser.parse_args()
    
    # 运行完整流程
    pipeline = FullAblationPipeline(args.config, args.output)
    pipeline.skip_inference = args.skip_inference
    pipeline.skip_metrics = args.skip_metrics
    pipeline.skip_viz = args.skip_viz
    pipeline.skip_summary = args.skip_summary
    pipeline.viz_roots = args.viz_roots
    pipeline.viz_overlay_mask = args.viz_overlay_mask
    
    # 应用过滤器（支持组匹配）
    if args.experiments:
        filtered = []
        for exp in pipeline.experiments:
            exp_group = exp.get('group', exp['id'])
            # 精确匹配 ID 或 group
            if exp['id'] in args.experiments or exp_group in args.experiments:
                filtered.append(exp)
        pipeline.experiments = filtered
        pipeline.log(f"过滤后实验数: {len(pipeline.experiments)}")
    
    # 运行
    pipeline.run_all()


if __name__ == "__main__":
    main()
