"""
消融实验推理引擎

功能：
1. 加载模型（Baseline / Stage1 / Stage2）
2. Adapter 路由（L4用l4，其他用shared）
3. 按 manifest 批量生成
4. 保存到 gen/tgt_L{1-4}/

Usage:
    from ablation_inference import AblationInferenceEngine
    
    engine = AblationInferenceEngine(config)
    engine.generate_all(manifest, output_dir)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
import cv2
import json

# Diffusers 导入
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipeline
from peft import PeftModel


class SeverityEmbedding(torch.nn.Module):
    """Severity Embedding（从 stage2_train.py 复制）"""
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_classes, embed_dim)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, severity_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(severity_ids)


class AblationInferenceEngine:
    """消融实验推理引擎"""
    
    def __init__(self, config: Dict, device: str = "cuda"):
        """
        Args:
            config: 实验配置字典（从 config.json 读取）
            device: 推理设备
        """
        self.config = config
        self.device = device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model_config = config['model']
        self.params = config['params']
        self.prompts = config['prompts']
        self.seed = config.get('seed', 0)
        
        # 加载模型
        self._load_models()
    
    
    def _enable_low_vram(self):
        """Enable memory-saving options when available."""
        if hasattr(self.unet, "enable_attention_slicing"):
            self.unet.enable_attention_slicing()
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _load_models(self):
        """加载模型"""
        print(f"加载模型: {self.config['exp_id']}")
        
        if self.model_config['use_baseline']:
            # A0: Baseline（原始 SD-Inpainting）
            print("  加载 Baseline (SD-Inpainting)...")
            self._load_baseline()
        elif self.model_config['stage1_only']:
            # A1: Stage1 Shared LoRA
            print("  加载 Stage1...")
            self._load_stage1()
        else:
            # A2/A3: Stage2
            print("  加载 Stage2...")
            self._load_stage2()
        
        print("  ✅ 模型加载完成")
    
    def _load_baseline(self):
        """加载 Baseline（原始 SD-Inpainting）"""
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
        self.severity_embedding = None  # Baseline 不用
        self.available_adapters = []
    
    def _load_stage1(self):
        """加载 Stage1 Shared LoRA"""
        checkpoint_path = self.model_config['stage1_checkpoint']
        
        # Tokenizer & Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(checkpoint_path, "tokenizer"))
        self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(checkpoint_path, "text_encoder"), torch_dtype=self.dtype)
        self.text_encoder.to(self.device, dtype=self.dtype).eval()
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="vae", torch_dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype).eval()
        
        # UNet + Shared LoRA
        unet_base = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="unet", torch_dtype=self.dtype)
        lora_path = os.path.join(checkpoint_path, "unet_lora")
        self.unet = PeftModel.from_pretrained(unet_base, lora_path, adapter_name="shared")
        self.unet.to(self.device, dtype=self.dtype).eval()
        
        # Severity Embedding
        time_embed_dim = self.unet.base_model.model.config.block_out_channels[0] * 4
        self.severity_embedding = SeverityEmbedding(5, time_embed_dim)
        sev_path = os.path.join(checkpoint_path, "severity_embedding.pt")
        if os.path.exists(sev_path):
            self.severity_embedding.load_state_dict(torch.load(sev_path, map_location=self.device))
        self.severity_embedding.to(self.device, dtype=self.dtype).eval()
        
        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")
        self._enable_low_vram()
        
        self.available_adapters = ["shared"]
        self._enable_low_vram()
    
    def _load_stage2(self):
        """加载 Stage2（Merged Base + L4 LoRA）"""
        checkpoint_path = self.model_config['stage2_checkpoint']
        
        # 读取 config 获取 merged_base_path
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'r') as f:
            ckpt_config = json.load(f)
        
        merged_base_path = ckpt_config.get("merged_base_path")
        if merged_base_path is None:
            merged_base_path = os.path.join(os.path.dirname(checkpoint_path), "merged_base_unet")
        
        # Tokenizer & Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(checkpoint_path, "tokenizer"))
        self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(checkpoint_path, "text_encoder"), torch_dtype=self.dtype)
        self.text_encoder.to(self.device, dtype=self.dtype).eval()
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="vae", torch_dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype).eval()
        
        # Merged Base UNet
        print(f"    加载 Merged Base: {merged_base_path}")
        self.unet = UNet2DConditionModel.from_pretrained(merged_base_path, torch_dtype=self.dtype)
        
        # 加载 L4 adapter
        l4_path = os.path.join(checkpoint_path, "unet_lora", "l4")
        self.available_adapters = []
        
        if os.path.exists(l4_path):
            print(f"    加载 L4 LoRA: {l4_path}")
            self.unet = PeftModel.from_pretrained(self.unet, l4_path, adapter_name="l4")
            self.available_adapters.append("l4")
        
        # 加载 Shared adapter（备用）
        shared_path = os.path.join(checkpoint_path, "unet_lora", "shared")
        if os.path.exists(shared_path):
            print(f"    加载 Shared LoRA: {shared_path}")
            self.unet.load_adapter(shared_path, adapter_name="shared")
            self.available_adapters.append("shared")
        
        self.unet.to(self.device, dtype=self.dtype).eval()
        
        # Severity Embedding
        time_embed_dim = self.unet.base_model.model.config.block_out_channels[0] * 4
        self.severity_embedding = SeverityEmbedding(5, time_embed_dim)
        sev_path = os.path.join(checkpoint_path, "severity_embedding.pt")
        if os.path.exists(sev_path):
            self.severity_embedding.load_state_dict(torch.load(sev_path, map_location=self.device))
        self.severity_embedding.to(self.device, dtype=self.dtype).eval()
        
        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")
        self._enable_low_vram()
    
    def set_adapter(self, target_level: int):
        """设置 Adapter（路由逻辑）"""
        if not self.available_adapters:
            return  # Baseline 没有 adapter
        
        # A3: force_shared_only（强制用 shared）
        if self.model_config.get('force_shared_only', False):
            if "shared" in self.available_adapters:
                self.unet.set_adapter("shared")
            return
        
        # A2: 路由逻辑（L4 用 l4，其他用 shared）
        if target_level == 4 and "l4" in self.available_adapters:
            self.unet.set_adapter("l4")
        elif "shared" in self.available_adapters:
            self.unet.set_adapter("shared")
    
    def generate_single(
        self,
        image_path: str,
        mask_path: str,
        target_level: int,
        output_path: str
    ) -> bool:
        """
        生成单张图像
        
        Returns:
            成功返回 True，失败返回 False
        """
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # 确保尺寸为 512x512
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.LANCZOS)
            if mask.size != (512, 512):
                mask = mask.resize((512, 512), Image.NEAREST)
            
            # 设置 adapter
            self.set_adapter(target_level)
            
            # 设置随机种子
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            
            # 获取 prompt
            prompt_dict = self.prompts.get(target_level, self.prompts.get(str(target_level)))
            positive_prompt = prompt_dict['positive']
            negative_prompt = prompt_dict['negative']
            
            # 生成
            if self.model_config['use_baseline']:
                # Baseline 用 pipeline
                result = self._generate_baseline(image, mask, positive_prompt, negative_prompt)
            else:
                # Stage1/Stage2 用自定义 forward
                result = self._generate_custom(image, mask, target_level, positive_prompt, negative_prompt)
            
            # 保存
            result.save(output_path, quality=95)
            return True
            
        except Exception as e:
            print(f"    ❌ 生成失败: {e}")
            return False
    
    def _generate_baseline(self, image: Image.Image, mask: Image.Image, positive: str, negative: str) -> Image.Image:
        """Baseline 生成"""
        result = self.pipeline(
            prompt=positive,
            negative_prompt=negative,
            image=image,
            mask_image=mask,
            num_inference_steps=self.params['num_inference_steps'],
            guidance_scale=self.params['guidance_scale'],
            strength=self.params['strength'],
        ).images[0]
        return result

    
    def _generate_custom(self, image: Image.Image, mask: Image.Image, target_level: int, positive: str, negative: str) -> Image.Image:
        """??????Stage1/Stage2?"""
        with torch.no_grad():
            # ??? tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            mask_np = np.array(mask).astype(np.float32) / 255.0

            image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.dtype)

            # ???? [-1, 1]
            image_t = image_t * 2 - 1

            # Dilate mask
            mask_dilated = self._dilate_mask(mask_t, self.params['mask_dilate_kernel']).to(self.dtype)

            # Masked image
            masked_image = image_t * (1 - mask_dilated)

            # Encode
            image_latents = self.vae.encode(image_t).latent_dist.sample().to(self.dtype) * self.vae.config.scaling_factor
            masked_latents = self.vae.encode(masked_image).latent_dist.sample().to(self.dtype) * self.vae.config.scaling_factor

            mask_latent = F.interpolate(mask_dilated, size=image_latents.shape[-2:], mode='nearest')

            # Text encoding
            text_inputs = self.tokenizer(positive, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_emb = self.text_encoder(text_inputs.input_ids.to(self.device))[0].to(self.dtype)

            neg_inputs = self.tokenizer(negative, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_emb = self.text_encoder(neg_inputs.input_ids.to(self.device))[0].to(self.dtype)

            # Severity ID
            level_to_severity = {1: 0, 2: 1, 3: 2, 4: 3}
            severity_id = torch.tensor([level_to_severity.get(target_level, 0)], device=self.device)
            uncond_severity_id = torch.tensor([4], device=self.device)

            # Denoising
            self.scheduler.set_timesteps(self.params['num_inference_steps'], device=self.device)
            init_timestep = int(self.params['num_inference_steps'] * self.params['strength'])
            t_start = max(self.params['num_inference_steps'] - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start:]

            noise = torch.randn_like(image_latents, dtype=self.dtype)
            latents = self.scheduler.add_noise(image_latents, noise, timesteps[:1])

            for t in timesteps:
                unet_input = torch.cat([latents, mask_latent, masked_latents], dim=1)

                # Forward with severity
                noise_cond = self._unet_forward(unet_input, t, text_emb, severity_id)
                noise_uncond = self._unet_forward(unet_input, t, uncond_emb, uncond_severity_id)

                noise_pred = noise_uncond + self.params['guidance_scale'] * (noise_cond - noise_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode
            generated = self.vae.decode(latents / self.vae.config.scaling_factor).sample

            # Feather mask
            soft_mask = self._create_feather_mask(mask_dilated, self.params['feather_pixels']).to(self.dtype)
            result = generated * soft_mask + image_t * (1 - soft_mask)

            # ?? [0, 1]
            result = (result / 2 + 0.5).clamp(0, 1)

            # ?? PIL
            result_np = (result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(result_np)


    def _unet_forward(self, sample, timestep, encoder_hidden_states, severity_id):
        """UNet Forward with Severity Embedding"""
        base = self.unet.base_model.model if hasattr(self.unet, 'base_model') else self.unet
        
        t_emb = base.time_proj(timestep.unsqueeze(0)).to(dtype=sample.dtype)
        emb = base.time_embedding(t_emb)
        
        if self.severity_embedding is not None:
            emb = emb + self.severity_embedding(severity_id).to(dtype=emb.dtype)
        
        sample = base.conv_in(sample)
        down_res = (sample,)
        
        for block in base.down_blocks:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(sample, emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res = block(sample, emb)
            down_res += res
        
        sample = base.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        
        for block in base.up_blocks:
            res = down_res[-len(block.resnets):]
            down_res = down_res[:-len(block.resnets)]
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(sample, res, emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample = block(sample, res, emb)
        
        sample = base.conv_norm_out(sample)
        sample = base.conv_act(sample)
        return base.conv_out(sample)
    
    def _dilate_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """膨胀 mask"""
        if kernel_size <= 0:
            return mask
        k = kernel_size * 2 + 1
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    
    def _create_feather_mask(self, mask: torch.Tensor, feather_pixels: int) -> torch.Tensor:
        """创建羽化 mask"""
        if feather_pixels <= 0:
            return mask
        
        k = feather_pixels * 2 + 1
        sigma = feather_pixels / 3.0
        x = torch.arange(k, device=mask.device, dtype=mask.dtype) - k // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.view(1, 1, -1, 1) * gauss_1d.view(1, 1, 1, -1)
        
        padded = F.pad(mask, (k//2, k//2, k//2, k//2), mode='replicate')
        soft_mask = F.conv2d(padded, gauss_2d.view(1, 1, k, k))
        return torch.max(soft_mask, mask).clamp(0, 1)
    
    def generate_all(self, manifest: Dict, output_dir: Path):
        """按 manifest 批量生成"""
        samples = manifest['samples']
        total = len(samples)
        
        print(f"\n开始生成: {total} 张图像")
        print(f"  参数: {self.params}")
        print(f"  Adapters: {self.available_adapters}")
        
        success_count = 0
        fail_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="生成中")):
            source_level = sample['source_level']
            target_level = sample['target_level']
            sample_id = sample['sample_id']
            
            # 输出路径
            output_subdir = output_dir / "gen" / f"tgt_L{target_level}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_path = output_subdir / f"src_L{source_level}_{sample_id}.png"
            
            # 生成
            success = self.generate_single(
                sample['image_path'],
                sample['mask_path'],
                target_level,
                str(output_path)
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n✅ 生成完成:")
        print(f"  成功: {success_count}/{total}")
        print(f"  失败: {fail_count}/{total}")


# ============== 命令行测试 ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验 config.json 路径")
    parser.add_argument("--manifest", type=str, required=True, help="manifest.json 路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    # 初始化引擎
    engine = AblationInferenceEngine(config, device="cuda")
    
    # 生成
    engine.generate_all(manifest, Path(args.output))
