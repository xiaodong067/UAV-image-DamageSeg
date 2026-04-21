"""
消融实验汇总工具

功能：
1. 读取所有实验的 metrics/*.csv
2. 生成 summary_metrics.csv（汇总表格）
3. 生成对比图表

输出：
- summary/summary_metrics.csv
- summary/plots/kid_comparison.png
- summary/plots/background_preservation.png
- summary/plots/sweep_curves.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


class AblationSummarizer:
    """消融实验汇总器"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.summary_dir = root_dir / "summary"
        self.summary_dir.mkdir(exist_ok=True)
        self.plots_dir = self.summary_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def generate_summary(self):
        """生成完整汇总"""
        print("\n生成汇总报告...")
        
        # 1. 收集所有实验的指标
        print("  [1/4] 收集指标...")
        all_metrics = self._collect_all_metrics()
        
        # 2. 生成汇总表格
        print("  [2/4] 生成汇总表格...")
        self._generate_summary_table(all_metrics)
        
        # 3. 生成对比图表
        print("  [3/4] 生成对比图表...")
        self._generate_comparison_plots(all_metrics)
        
        # 4. 生成扫参曲线
        print("  [4/4] 生成扫参曲线...")
        self._generate_sweep_curves(all_metrics)
        
        print(f"\n✅ 汇总完成: {self.summary_dir}")
    
    def _collect_all_metrics(self) -> Dict[str, Dict]:
        """收集所有实验的指标"""
        all_metrics = {}
        
        for exp_dir in sorted(self.root_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name == "summary":
                continue
            
            # 读取该实验的所有指标
            metrics_dir = exp_dir / "metrics"
            if not metrics_dir.exists():
                continue
            
            exp_metrics = {}
            
            # KID
            kid_path = metrics_dir / "kid_per_level.csv"
            if kid_path.exists():
                exp_metrics['kid'] = pd.read_csv(kid_path)

            fid_path = metrics_dir / "fid_per_level.csv"
            if fid_path.exists():
                exp_metrics['fid'] = pd.read_csv(fid_path)
            
            # LPIPS
            lpips_path = metrics_dir / "lpips.csv"
            if lpips_path.exists():
                exp_metrics['lpips'] = pd.read_csv(lpips_path)

            tm_path = metrics_dir / "transformation_magnitude.csv"
            if tm_path.exists():
                exp_metrics['transformation_magnitude'] = pd.read_csv(tm_path)

            clip_path = metrics_dir / "clip_score.csv"
            if clip_path.exists():
                exp_metrics['clip'] = pd.read_csv(clip_path)
            
            # Background Preservation
            bg_path = metrics_dir / "background_preservation.csv"
            if bg_path.exists():
                exp_metrics['background'] = pd.read_csv(bg_path)
            
            # Boundary Artifacts
            boundary_path = metrics_dir / "boundary_artifacts.csv"
            if boundary_path.exists():
                exp_metrics['boundary'] = pd.read_csv(boundary_path)
            
            all_metrics[exp_dir.name] = exp_metrics
        
        return all_metrics
    
    def _generate_summary_table(self, all_metrics: Dict):
        """生成汇总表格"""
        rows = []
        
        for exp_id, metrics in all_metrics.items():
            row = {'exp_id': exp_id}
            
            # KID（平均）
            if 'kid' in metrics:
                kid_df = metrics['kid']
                row['kid_mean_avg'] = kid_df['kid_mean'].mean()
                row['kid_L1'] = kid_df[kid_df['target_level'] == 1]['kid_mean'].values[0] if len(kid_df[kid_df['target_level'] == 1]) > 0 else np.nan
                row['kid_L2'] = kid_df[kid_df['target_level'] == 2]['kid_mean'].values[0] if len(kid_df[kid_df['target_level'] == 2]) > 0 else np.nan
                row['kid_L3'] = kid_df[kid_df['target_level'] == 3]['kid_mean'].values[0] if len(kid_df[kid_df['target_level'] == 3]) > 0 else np.nan
                row['kid_L4'] = kid_df[kid_df['target_level'] == 4]['kid_mean'].values[0] if len(kid_df[kid_df['target_level'] == 4]) > 0 else np.nan
            
            # FID????
            if 'fid' in metrics:
                fid_df = metrics['fid']
                row['fid_mean_avg'] = fid_df['fid'].mean()
                row['fid_L1'] = fid_df[fid_df['target_level'] == 1]['fid'].values[0] if len(fid_df[fid_df['target_level'] == 1]) > 0 else np.nan
                row['fid_L2'] = fid_df[fid_df['target_level'] == 2]['fid'].values[0] if len(fid_df[fid_df['target_level'] == 2]) > 0 else np.nan
                row['fid_L3'] = fid_df[fid_df['target_level'] == 3]['fid'].values[0] if len(fid_df[fid_df['target_level'] == 3]) > 0 else np.nan
                row['fid_L4'] = fid_df[fid_df['target_level'] == 4]['fid'].values[0] if len(fid_df[fid_df['target_level'] == 4]) > 0 else np.nan

            
            # LPIPS（平均）
            if 'lpips' in metrics:
                lpips_df = metrics['lpips']
                row['lpips_mean'] = lpips_df['lpips'].mean()

            if 'transformation_magnitude' in metrics:
                tm_df = metrics['transformation_magnitude']
                row['transformation_magnitude_mean'] = tm_df['transformation_magnitude'].mean()

            if 'clip' in metrics:
                clip_df = metrics['clip']
                row['clip_score_mean'] = clip_df['clip_score'].mean()
            
            # Background Preservation（平均）
            if 'background' in metrics:
                bg_df = metrics['background']
                row['bg_ssim_mean'] = bg_df['bg_ssim'].mean()
                row['bg_psnr_mean'] = bg_df['bg_psnr'].mean()
                b_df = metrics['building']
            
            # Boundary Artifacts（平均）
            if 'boundary' in metrics:
                boundary_df = metrics['boundary']
                row['boundary_grad_mean'] = boundary_df['boundary_grad_mean'].mean()
            
            rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        
        # 保存
        output_path = self.summary_dir / "summary_metrics.csv"
        summary_df.to_csv(output_path, index=False, float_format='%.4f')
        
        # 同时保存为 Excel（如果有 openpyxl）
        try:
            xlsx_path = self.summary_dir / "summary_metrics.xlsx"
            summary_df.to_excel(xlsx_path, index=False, float_format='%.4f')
        except ImportError:
            pass
        
        print(f"    ✅ 保存到: {output_path}")
    
    def _generate_comparison_plots(self, all_metrics: Dict):
        """生成对比图表（A0/A1/A2/A3）"""
        main_exps = ['A0', 'A1', 'A2', 'A3']
        
        # 1. KID Comparison（每个 level）
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for level_idx, target_level in enumerate([1, 2, 3, 4]):
            ax = axes[level_idx]
            
            exp_names = []
            kid_means = []
            kid_stds = []
            
            for exp_id in main_exps:
                if exp_id not in all_metrics or 'kid' not in all_metrics[exp_id]:
                    continue
                
                kid_df = all_metrics[exp_id]['kid']
                level_data = kid_df[kid_df['target_level'] == target_level]
                
                if len(level_data) == 0:
                    continue
                
                exp_names.append(exp_id)
                kid_means.append(level_data['kid_mean'].values[0])
                kid_stds.append(level_data['kid_std'].values[0])
            
            if len(exp_names) > 0:
                x = np.arange(len(exp_names))
                ax.bar(x, kid_means, yerr=kid_stds, capsize=5, alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(exp_names)
                ax.set_title(f'KID (Target L{target_level})')
                ax.set_ylabel('KID')
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "kid_comparison.png", dpi=150)
        plt.close()
        
        # 2. Preservation Comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        exp_names = []
        ssim_means = []
        psnr_means = []
        
        for exp_id in main_exps:
            if exp_id not in all_metrics or 'background' not in all_metrics[exp_id]:
                continue

            bg_df = all_metrics[exp_id]['background']
            exp_names.append(exp_id)
            ssim_means.append(bg_df['bg_ssim'].mean())
            psnr_means.append(bg_df['bg_psnr'].mean())
        
        if len(exp_names) > 0:
            x = np.arange(len(exp_names))
            
            axes[0].bar(x, ssim_means, alpha=0.7)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(exp_names)
            axes[0].set_title('Preservation SSIM')
            axes[0].set_ylabel('SSIM')
            axes[0].grid(axis='y', alpha=0.3)
            
            axes[1].bar(x, psnr_means, alpha=0.7, color='orange')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(exp_names)
            axes[1].set_title('Preservation PSNR')
            axes[1].set_ylabel('PSNR (dB)')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "background_preservation.png", dpi=150)
        plt.close()
        
        print(f"    ✅ 保存对比图表")
    
    def _generate_sweep_curves(self, all_metrics: Dict):
        """生成扫参曲线（A4-A7）"""
        sweep_exps = {
            'A4': ('guidance_scale', [0, 5, 7, 10, 12]),
            'A5': ('strength', [0.75, 0.85, 0.90, 0.95]),
            'A6': ('num_inference_steps', [25, 50]),
            'A7': ('mask_dilate_kernel', [0, 15, 25, 40])
        }
        
        for exp_id, (param_name, param_values) in sweep_exps.items():
            if exp_id not in all_metrics:
                continue
            
            # 查找该实验下的所有 setting 子目录
            exp_dir = self.root_dir / exp_id
            if not exp_dir.exists():
                continue
            
            # 收集每个参数值的 KID
            kid_data = {level: [] for level in [1, 2, 3, 4]}
            actual_values = []
            
            for param_value in param_values:
                if isinstance(param_value, float):
                    setting_name = f"{param_name}_{param_value:.2f}".replace('.', '_')
                else:
                    setting_name = f"{param_name}_{param_value}"
                
                setting_dir = exp_dir / setting_name
                kid_path = setting_dir / "metrics" / "kid_per_level.csv"
                
                if not kid_path.exists():
                    continue
                
                kid_df = pd.read_csv(kid_path)
                actual_values.append(param_value)
                
                for level in [1, 2, 3, 4]:
                    level_data = kid_df[kid_df['target_level'] == level]
                    if len(level_data) > 0:
                        kid_data[level].append(level_data['kid_mean'].values[0])
                    else:
                        kid_data[level].append(np.nan)
            
            if len(actual_values) == 0:
                continue
            
            # 绘制曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for level in [1, 2, 3, 4]:
                if len(kid_data[level]) > 0:
                    ax.plot(actual_values, kid_data[level], marker='o', label=f'L{level}', linewidth=2)
            
            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel('KID', fontsize=12)
            ax.set_title(f'{exp_id}: {param_name} Sweep', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"sweep_{param_name}.png", dpi=150)
            plt.close()
        
        print(f"    ✅ 保存扫参曲线")


# ============== 命令行测试 ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="runs_ablation 根目录")
    
    args = parser.parse_args()
    
    summarizer = AblationSummarizer(Path(args.root_dir))
    summarizer.generate_summary()

