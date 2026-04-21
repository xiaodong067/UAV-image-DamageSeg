"""
消融实验指标计算模块

功能：
1. KID/FID（生成质量）
2. LPIPS（感知距离）
3. Background Preservation（背景保持 SSIM/PSNR）
4. Boundary Artifact Score（边界伪影）

输出：
- metrics/kid_per_level.csv
- metrics/background_preservation.csv
- metrics/background_preservation_per_level.csv
- metrics/boundary_artifacts.csv
- metrics/lpips.csv
- metrics/transformation_magnitude_per_level.csv
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor
from scipy import linalg


class MetricCalculator:
    """指标计算器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # InceptionV3 for KID/FID
        self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval()
        
        self.clip_model = None
        self.clip_processor = None

        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_all_metrics(self, exp_dir: Path, manifest: Dict, refs_dir: Path = None):
        """计算所有指标
        
        Args:
            exp_dir: 实验目录
            manifest: 生成清单
            refs_dir: 公共引用数据目录（用于 KID 计算）
        """
        print(f"\n计算指标: {exp_dir.name}")
        
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # 1. KID (per target level) - masked building region
        print("  [1/3] Compute KID (building region)...")
        kid_results = self._calculate_kid_per_level(exp_dir, manifest, refs_dir)
        kid_df = pd.DataFrame(kid_results)
        kid_df.to_csv(metrics_dir / "kid_per_level.csv", index=False)
        
        # 1.1 FID (per target level) - masked building region
        print("  [1/3] Compute FID (building region)...")
        fid_results = self._calculate_fid_per_level(exp_dir, manifest, refs_dir)
        fid_df = pd.DataFrame(fid_results)
        fid_df.to_csv(metrics_dir / "fid_per_level.csv", index=False)
        
        # 1.2 CLIP Score (prompt consistency)
        print("  [1/3] Compute CLIP Score (building region)...")
        clip_results = self._calculate_clip_score(exp_dir, manifest)
        clip_df = pd.DataFrame(clip_results)
        clip_df.to_csv(metrics_dir / "clip_score.csv", index=False)


        
        # 2. Masked LPIPS (building region)
        print("  [2/3] Compute Transformation Magnitude (Masked LPIPS)...")
        tm_results = self._calculate_lpips_masked(exp_dir, manifest)
        tm_df = pd.DataFrame(tm_results)
        tm_df.to_csv(metrics_dir / "transformation_magnitude.csv", index=False)
        self._save_per_level_stats(
            tm_df,
            metrics_dir / "transformation_magnitude_per_level.csv",
            value_col="transformation_magnitude"
        )
        
        # 3.1 Background Preservation (mask-out region)
        print("  [3/3] Compute background preservation...")
        bg_results = self._calculate_background_preservation(exp_dir, manifest)
        bg_df = pd.DataFrame(bg_results)
        bg_df.to_csv(metrics_dir / "background_preservation.csv", index=False)
        self._save_per_level_stats(
            bg_df,
            metrics_dir / "background_preservation_per_level.csv",
            value_col="background_ssim",
            extra_cols=["background_psnr"]
        )

        
        print("  ✅ 指标计算完成")
    
    
    def _save_per_level_stats(
        self,
        df: pd.DataFrame,
        output_path: Path,
        value_col: str,
        extra_cols: List[str] = None
    ) -> None:
        """Save per-target-level mean/std/count for a metric."""
        if df.empty or "target_level" not in df.columns or value_col not in df.columns:
            return
        agg_map = {value_col: ["mean", "std", "count"]}
        if extra_cols:
            for col in extra_cols:
                if col in df.columns:
                    agg_map[col] = ["mean", "std"]
        summary = df.groupby(["target_level"]).agg(agg_map)
        summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]
        summary = summary.reset_index()
        summary.to_csv(output_path, index=False)

    def _build_image_mask_index(self, manifest: Dict) -> Dict[str, str]:
        """Build image_path -> mask_path index from metadata.json"""
        if not manifest.get('samples'):
            return {}
        sample_path = Path(manifest['samples'][0]['image_path'])
        data_dir = sample_path.parent.parent
        metadata_path = data_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        index = {}
        for s in metadata.get('samples', []):
            index[s['image_path']] = s['mask_path']
        return index

    def _extract_inception_features_masked(self, image_path: str, mask_path: str) -> np.ndarray:
        """Extract InceptionV3 features inside building mask"""
        img = Image.open(image_path).convert('RGB').resize((299, 299), Image.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize((299, 299), Image.NEAREST)

        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = (np.array(mask).astype(np.float32) / 255.0) > 0.5
        mask_np = mask_np.astype(np.float32)

        img_np = img_np * mask_np[:, :, None]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.inception(img_t)
        return features.cpu().numpy().flatten()


    def _calculate_kid_per_level(self, exp_dir: Path, manifest: Dict, refs_dir: Path = None) -> List[Dict]:
        """Compute KID per target level (masked building region)"""
        results = []
        self._image_mask_index = self._build_image_mask_index(manifest)

        for target_level in [1, 2, 3, 4]:
            gen_samples = [s for s in manifest.get('samples', []) if s['target_level'] == target_level]
            if len(gen_samples) == 0:
                continue

            ref_samples = self._get_real_level_samples(target_level, refs_dir, manifest)
            if len(ref_samples) == 0:
                print(f"    Warning: no reference samples for L{target_level}, skip KID")
                continue

            gen_features = []
            for s in tqdm(gen_samples, desc=f"  Level {target_level} Gen", leave=False):
                gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{s['source_level']}_{s['sample_id']}.png"
                if not gen_path.exists():
                    continue
                feat = self._extract_inception_features_masked(str(gen_path), s['mask_path'])
                gen_features.append(feat)

            ref_features = []
            for s in tqdm(ref_samples, desc=f"  Level {target_level} Real", leave=False):
                feat = self._extract_inception_features_masked(s['image_path'], s['mask_path'])
                ref_features.append(feat)

            if len(gen_features) == 0 or len(ref_features) == 0:
                continue

            gen_features = np.stack(gen_features)
            ref_features = np.stack(ref_features)

            kid_mean, kid_std = self._compute_kid(gen_features, ref_features)

            results.append({
                'target_level': target_level,
                'kid_mean': kid_mean,
                'kid_std': kid_std,
                'num_generated': len(gen_features),
                'num_reference': len(ref_features)
            })

        return results


    def _calculate_fid_per_level(self, exp_dir: Path, manifest: Dict, refs_dir: Path = None) -> List[Dict]:
        '''Compute FID per target level (masked building region)'''
        results = []
        self._image_mask_index = self._build_image_mask_index(manifest)

        for target_level in [1, 2, 3, 4]:
            gen_samples = [s for s in manifest.get('samples', []) if s['target_level'] == target_level]
            if len(gen_samples) == 0:
                continue

            ref_samples = self._get_real_level_samples(target_level, refs_dir, manifest)
            if len(ref_samples) == 0:
                print(f"    Warning: no reference samples for L{target_level}, skip FID")
                continue

            gen_features = []
            for s in tqdm(gen_samples, desc=f"  Level {target_level} Gen (FID)", leave=False):
                gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{s['source_level']}_{s['sample_id']}.png"
                if not gen_path.exists():
                    continue
                feat = self._extract_inception_features_masked(str(gen_path), s['mask_path'])
                gen_features.append(feat)

            ref_features = []
            for s in tqdm(ref_samples, desc=f"  Level {target_level} Real (FID)", leave=False):
                feat = self._extract_inception_features_masked(s['image_path'], s['mask_path'])
                ref_features.append(feat)

            if len(gen_features) == 0 or len(ref_features) == 0:
                continue

            gen_features = np.stack(gen_features)
            ref_features = np.stack(ref_features)

            fid_value = self._compute_fid(ref_features, gen_features)

            results.append({
                'target_level': target_level,
                'fid': fid_value,
                'num_generated': len(gen_features),
                'num_reference': len(ref_features)
            })

        return results

    def _compute_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        '''Compute Fr?chet Inception Distance (FID)'''
        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(fake_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(fake_features, rowvar=False)

        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    
    def _get_real_level_samples(self, target_level: int, refs_dir: Path, manifest: Dict) -> List[Dict]:
        """Get real samples for target level with mask paths"""
        results = []

        if refs_dir and refs_dir.exists():
            refs_manifest_path = refs_dir / "manifest.json"
            if refs_manifest_path.exists():
                with open(refs_manifest_path, 'r') as f:
                    refs_manifest = json.load(f)

                level_key = str(target_level)
                if level_key in refs_manifest.get('samples_by_level', {}):
                    for img_path in refs_manifest['samples_by_level'][level_key]:
                        mask_path = self._image_mask_index.get(img_path)
                        if mask_path:
                            results.append({'image_path': img_path, 'mask_path': mask_path})

        if results:
            return results

        for sample in manifest.get('samples', []):
            if sample['source_level'] == target_level:
                results.append({'image_path': sample['image_path'], 'mask_path': sample['mask_path']})

        seen = set()
        uniq = []
        for s in results:
            if s['image_path'] in seen:
                continue
            seen.add(s['image_path'])
            uniq.append(s)
        return uniq
    
    def _extract_inception_features(self, image_path: str) -> np.ndarray:
        """提取 Inception V3 特征"""
        img = Image.open(image_path).convert('RGB')
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.inception(img_t)
        
        return features.cpu().numpy().flatten()
    
    def _compute_kid(self, real_features: np.ndarray, fake_features: np.ndarray, num_subsets: int = 100, subset_size: int = 100) -> Tuple[float, float]:
        """计算 Kernel Inception Distance (KID)"""
        n = min(len(real_features), len(fake_features))
        if n < subset_size:
            subset_size = n
        
        mmd2_list = []
        for _ in range(num_subsets):
            # 随机采样
            real_subset = real_features[np.random.choice(len(real_features), subset_size, replace=False)]
            fake_subset = fake_features[np.random.choice(len(fake_features), subset_size, replace=False)]
            
            # 计算 MMD^2
            mmd2 = self._polynomial_mmd(real_subset, fake_subset)
            mmd2_list.append(mmd2)
        
        return np.mean(mmd2_list), np.std(mmd2_list)
    
    def _polynomial_mmd(self, X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: float = None, coef0: float = 1) -> float:
        """Polynomial kernel MMD"""
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        K_XX = (gamma * (X @ X.T) + coef0) ** degree
        K_YY = (gamma * (Y @ Y.T) + coef0) ** degree
        K_XY = (gamma * (X @ Y.T) + coef0) ** degree
        
        m = K_XX.shape[0]
        n = K_YY.shape[0]
        
        mmd2 = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
        mmd2 += (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
        mmd2 -= 2 * K_XY.sum() / (m * n)
        
        return mmd2
    
    def _calculate_lpips(self, exp_dir: Path, manifest: Dict) -> List[Dict]:
        """计算 LPIPS（感知距离）"""
        results = []
        
        for sample in tqdm(manifest['samples'], desc="  LPIPS"):
            target_level = sample['target_level']
            source_level = sample['source_level']
            sample_id = sample['sample_id']
            
            # 生成图像路径
            gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{source_level}_{sample_id}.png"
            if not gen_path.exists():
                continue
            
            # 参考图像
            ref_path = sample['image_path']
            
            # 加载图像
            gen_img = self._load_image_for_lpips(gen_path)
            ref_img = self._load_image_for_lpips(ref_path)
            
            # 计算 LPIPS
            with torch.no_grad():
                lpips_score = self.lpips_fn(gen_img, ref_img).item()
            
            results.append({
                'source_level': source_level,
                'target_level': target_level,
                'sample_id': sample_id,
                'lpips': lpips_score
            })
        
        return results


    
    def _init_clip(self):
        if self.clip_model is None or self.clip_processor is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_model.eval()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _load_image_for_clip_masked(self, image_path: str, mask_path: str):
        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L').resize(img.size, Image.NEAREST)
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = (np.array(mask).astype(np.float32) / 255.0) > 0.5
        mask_np = mask_np.astype(np.float32)
        img_np = img_np * mask_np[:, :, None]
        img_masked = Image.fromarray((img_np * 255).astype(np.uint8))
        return img_masked

    def _calculate_clip_score(self, exp_dir: Path, manifest: Dict) -> List[Dict]:
        '''Compute CLIP score between generated image and prompt (building region).'''
        self._init_clip()

        # Load prompt mapping from experiment config
        config_path = exp_dir / "config.json"
        prompt_by_level = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            prompt_by_level = {int(k): v['positive'] for k, v in cfg.get('prompts', {}).items()}

        results = []
        for sample in tqdm(manifest['samples'], desc="  CLIP Score"):
            target_level = sample['target_level']
            source_level = sample['source_level']
            sample_id = sample['sample_id']

            gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{source_level}_{sample_id}.png"
            if not gen_path.exists():
                continue

            prompt = prompt_by_level.get(target_level)
            if not prompt:
                prompt = f"aerial view of post-disaster building damage <L{target_level}>"

            img_masked = self._load_image_for_clip_masked(str(gen_path), sample['mask_path'])
            inputs = self.clip_processor(text=[prompt], images=img_masked, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_emb = outputs.image_embeds
                text_emb = outputs.text_embeds
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                score = (image_emb * text_emb).sum(dim=-1).item()

            results.append({
                'source_level': source_level,
                'target_level': target_level,
                'sample_id': sample_id,
                'clip_score': score
            })

        return results

    def _calculate_lpips_masked(self, exp_dir: Path, manifest: Dict) -> List[Dict]:
        '''?? Masked LPIPS?????????'''
        results = []
        
        for sample in tqdm(manifest['samples'], desc="  Masked LPIPS"):
            target_level = sample['target_level']
            source_level = sample['source_level']
            sample_id = sample['sample_id']
            
            gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{source_level}_{sample_id}.png"
            if not gen_path.exists():
                continue
            
            ref_path = sample['image_path']
            mask_path = sample['mask_path']
            
            gen_img = self._load_image_for_lpips_masked(gen_path, mask_path)
            ref_img = self._load_image_for_lpips_masked(ref_path, mask_path)
            
            with torch.no_grad():
                lpips_score = self.lpips_fn(gen_img, ref_img).item()
            
            results.append({
                'source_level': source_level,
                'target_level': target_level,
                'sample_id': sample_id,
                'transformation_magnitude': lpips_score
            })
        
        return results
    
    def _load_image_for_lpips(self, path: str) -> torch.Tensor:
        """加载图像用于 LPIPS（范围 [-1, 1]）"""
        img = Image.open(path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_t = img_t * 2 - 1  # [0, 1] -> [-1, 1]
        return img_t


    def _load_image_for_lpips_masked(self, image_path: str, mask_path: str) -> torch.Tensor:
        '''????+mask??? LPIPS?????????'''
        img = Image.open(image_path).convert('RGB').resize((512, 512), Image.LANCZOS)
        mask = Image.open(mask_path).convert('L').resize((512, 512), Image.NEAREST)
        
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = (np.array(mask).astype(np.float32) / 255.0) > 0.5
        mask_np = mask_np.astype(np.float32)
        
        img_np = img_np * mask_np[:, :, None]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_t = img_t * 2 - 1
        return img_t
    
    def _calculate_background_preservation(self, exp_dir: Path, manifest: Dict) -> List[Dict]:
        """计算背景保持（mask外区域的 SSIM/PSNR）"""
        results = []
        
        for sample in tqdm(manifest['samples'], desc="  背景保持"):
            target_level = sample['target_level']
            source_level = sample['source_level']
            sample_id = sample['sample_id']
            
            # 生成图像
            gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{source_level}_{sample_id}.png"
            if not gen_path.exists():
                continue
            
            # 原始图像
            orig_path = sample['image_path']
            
            # Mask（背景 = 0，前景 = 255）
            mask_path = sample['mask_path']
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            background_mask = (mask == 0).astype(np.uint8)  # 背景区域
            
            # 加载图像
            gen_img = cv2.imread(str(gen_path))
            gen_img = cv2.resize(gen_img, (512, 512))
            
            orig_img = cv2.imread(orig_path)
            orig_img = cv2.resize(orig_img, (512, 512))
            
            # 只计算背景区域
            gen_bg = gen_img * background_mask[:, :, None]
            orig_bg = orig_img * background_mask[:, :, None]
            
            # SSIM
            ssim_score = ssim(orig_bg, gen_bg, multichannel=True, data_range=255, channel_axis=2)
            
            # PSNR
            psnr_score = psnr(orig_bg, gen_bg, data_range=255)
            
            results.append({
                'source_level': source_level,
                'target_level': target_level,
                'sample_id': sample_id,
                'bg_ssim': ssim_score,
                'bg_psnr': psnr_score
            })
        
        return results
    
    def _calculate_boundary_artifacts(self, exp_dir: Path, manifest: Dict) -> List[Dict]:
        """计算边界伪影（ring 区域的梯度统计）"""
        results = []
        
        for sample in tqdm(manifest['samples'], desc="  边界伪影"):
            target_level = sample['target_level']
            source_level = sample['source_level']
            sample_id = sample['sample_id']
            
            # 生成图像
            gen_path = exp_dir / "gen" / f"tgt_L{target_level}" / f"src_L{source_level}_{sample_id}.png"
            if not gen_path.exists():
                continue
            
            # Mask
            mask_path = sample['mask_path']
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            # 生成 ring mask（外环10像素）
            kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            mask_outer = cv2.dilate(mask, kernel_outer)
            mask_inner = cv2.erode(mask, kernel_inner)
            ring_mask = mask_outer - mask_inner
            ring_mask = (ring_mask > 0).astype(np.uint8)
            
            # 加载图像
            gen_img = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE)
            gen_img = cv2.resize(gen_img, (512, 512))
            
            # 计算梯度
            grad_x = cv2.Sobel(gen_img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gen_img, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Ring 区域梯度统计
            ring_grad = grad_mag[ring_mask > 0]
            
            if len(ring_grad) == 0:
                continue
            
            results.append({
                'source_level': source_level,
                'target_level': target_level,
                'sample_id': sample_id,
                'boundary_grad_mean': np.mean(ring_grad),
                'boundary_grad_std': np.std(ring_grad),
                'boundary_grad_max': np.max(ring_grad)
            })
        
        return results


# ============== 命令行测试 ==============
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="实验目录")
    parser.add_argument("--manifest", type=str, required=True, help="manifest.json 路径")
    
    args = parser.parse_args()
    
    # 加载 manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    # 初始化计算器
    calculator = MetricCalculator(device="cuda")
    
    # 计算指标
    calculator.calculate_all_metrics(Path(args.exp_dir), manifest)

