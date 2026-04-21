"""
Stage 2 训练脚本：Merge Shared LoRA + 训练 L4 专用 LoRA

方案 2 实现：
1. 加载 Stage 1 的 shared LoRA
2. 把 shared LoRA merge 到 UNet 基座权重（变成"新基座"）
3. 新增 l4 adapter，只训练 l4
4. 推理时 L4 效果 = (base + shared_merged) + l4

使用方式：
    HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes=2 \
        disaster_lora/stage2_train.py --checkpoint ./disaster_lora/output/stage1/checkpoint-30000
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from torchvision import transforms
from typing import List
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluation_utils import TrainingVisualizer


# ============== 配置 ==============
class Config:
    pretrained_model = "runwayml/stable-diffusion-inpainting"
    data_dir = "./disaster_lora/training_data"
    output_dir = "./disaster_lora/output/stage2"
    checkpoint_path = None
    val_ratio = 0.1
    min_val_per_level = 1
    
    resolution = 512
    train_batch_size = 4
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    l4_lora_lr = 1e-4
    seed = 42
    mixed_precision = "fp16"
    gradient_checkpointing = True
    save_steps = 2000
    max_steps = 20000
    
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"]
    
    l4_lora_rank = 8
    l4_lora_alpha = 16
    l4_lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    num_severity_classes = 5
    severity_uncond_id = 4
    p_uncond_severity = 0.1
    level_to_severity = {1: 0, 2: 1, 3: 2, 4: 3}
    
    weight_background = 0.1
    weight_interior = 1.0
    l4_loss_weight = 2.0
    
    # ring mask（在 latent 空间做，latent 尺度约等于图像/8）
    # P1: 加大 ring 宽度，逼模型学"破碎边界 + 外溢碎片"
    ring_dilate_kernel = 21   # 从 11 提升到 21（约 80 原图像素）
    weight_ring_l4 = 0.9      # 从 0.4 提升到 0.9，接近 interior 权重
    
    # 按等级动态膨胀 mask（在图像空间，训练时动态应用）
    # P0: L4 膨胀量大幅提升，消除建筑边界线索
    mask_dilate_by_level = {
        1: 0,    # L1 不额外膨胀
        2: 0,    # L2 不额外膨胀
        3: 21,   # L3 中等膨胀 (~21 像素)
        4: 80,   # L4 大膨胀 (~80 像素) - 从 21 提升到 80
    }
    
    # 软边界（feather）配置
    use_soft_mask = True      # 是否使用软边界回填
    feather_pixels = 25       # 羽化像素数（边缘渐变宽度）
    
    level_tokens = ["<L1>", "<L2>", "<L3>", "<L4>"]
    l4_oversample_ratio = 0.5


class EvalConfig:
    eval_steps = 500
    num_inference_steps = 50
    
    # 按"目标等级"设置推理强度：越重损伤越需要更强重绘
    strength_by_level = {
        1: 0.75,
        2: 0.85,
        3: 0.93,
        4: 0.98,
    }
    
    # P2: L4 降低 CFG，避免把"building"语义拉回来
    guidance_by_level = {
        1: 7.5,
        2: 8.5,
        3: 9.0,
        4: 7.0,   # 从 12 降到 7
    }


# L4 专用 prompt 配置（P2: 抑制"语义拉回 building / floorplan"）
L4_PROMPTS = {
    "positive": "aerial view, debris field, completely destroyed building, no intact roof, scattered rubble, dust, irregular boundary <L4>",
    "negative": "intact roof, floorplan, rooms, clean interior, architectural drawing, blueprint, undamaged, organized structure"
}


config = Config()
eval_config = EvalConfig()


def load_metadata_samples(data_dir: str) -> List[dict]:
    data_dir = Path(data_dir)
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    return [
        {
            "image_path": sample["image_path"],
            "masked_path": sample["masked_path"],
            "mask_path": sample["mask_path"],
            "level": sample["level"],
        }
        for sample in metadata["samples"]
    ]


def stratified_train_val_split(samples: List[dict], val_ratio: float, seed: int, min_val_per_level: int = 1):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["level"]].append(sample)

    train_samples, val_samples = [], []
    for level, level_samples in sorted(grouped.items()):
        level_samples = list(level_samples)
        rng = np.random.default_rng(seed + int(level))
        indices = np.arange(len(level_samples))
        rng.shuffle(indices)

        if len(level_samples) <= 1 or val_ratio <= 0:
            val_count = 0
        else:
            val_count = max(min_val_per_level, int(round(len(level_samples) * val_ratio)))
            val_count = min(val_count, len(level_samples) - 1)

        val_index_set = set(indices[:val_count].tolist())
        for idx, sample in enumerate(level_samples):
            if idx in val_index_set:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, val_samples


# ============== 模块（支持多卡同步）==============
class SeverityEmbedding(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, severity_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(severity_ids)


class NewTokenEmbedding(nn.Module):
    """新 token embedding 包装成 Module，支持 DDP 多卡同步"""
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_tokens, embed_dim))
    
    def init_from_pretrained(self, pretrained_weight: torch.Tensor):
        self.weight.data.copy_(pretrained_weight)
    
    def forward(self) -> torch.Tensor:
        return self.weight


# ============== Dataset ==============
class DisasterDataset(Dataset):
    def __init__(self, samples: List[dict], tokenizer: CLIPTokenizer, resolution: int = 512, split_name: str = "train"):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.split_name = split_name
        self.samples = list(samples)
        self.level_indices = defaultdict(list)

        for idx, sample in enumerate(self.samples):
            self.level_indices[sample['level']].append(idx)

        print(f"{self.split_name} dataset: {len(self.samples)} samples")
        for level in sorted(self.level_indices.keys()):
            print(f"  L{level}: {len(self.level_indices[level])} samples")
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = self.image_transforms(Image.open(sample['image_path']).convert('RGB'))
        masked_image = self.image_transforms(Image.open(sample['masked_path']).convert('RGB'))
        mask = self.mask_transforms(Image.open(sample['mask_path']).convert('L'))
        
        level = sample['level']
        
        # 关键改动：L4 使用专用 prompt（debris field），而不是 "building damage"
        if level == 4:
            prompt = L4_PROMPTS["positive"]
        else:
            prompt = f"aerial view of post-disaster building damage <L{level}>"
        
        text_inputs = self.tokenizer(prompt, padding="max_length", 
                                      max_length=self.tokenizer.model_max_length, 
                                      truncation=True, return_tensors="pt")
        
        return {
            "image": image, "masked_image": masked_image, "mask": mask,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "level": level, 
            "severity_id": config.level_to_severity.get(level, config.severity_uncond_id),
            "is_l4": level == 4,
        }
    
    def get_sample_weights(self, l4_ratio: float = 0.5) -> List[float]:
        n_l4 = len(self.level_indices[4])
        n_other = len(self.samples) - n_l4
        if n_l4 == 0 or n_other == 0:
            return [1.0] * len(self.samples)
        w_l4 = (l4_ratio * n_other) / ((1 - l4_ratio) * n_l4)
        return [w_l4 if s['level'] == 4 else 1.0 for s in self.samples]


# ============== UNet Forward with Severity ==============
def unet_forward_with_severity(unet, noisy_latents, timesteps, encoder_hidden_states, 
                                severity_embedding, severity_ids, mask_latent, masked_latents):
    """带 severity embedding 的 UNet forward（手动注入 time embedding）"""
    _unet = unet.module if hasattr(unet, 'module') else unet
    # 处理 PEFT 包装
    if hasattr(_unet, 'base_model'):
        base = _unet.base_model.model if hasattr(_unet.base_model, 'model') else _unet.base_model
    else:
        base = _unet
    
    unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
    
    t_emb = base.time_proj(timesteps).to(dtype=unet_input.dtype)
    emb = base.time_embedding(t_emb)
    
    if severity_embedding is not None and severity_ids is not None:
        sev_emb = severity_embedding(severity_ids).to(dtype=emb.dtype)
        emb = emb + sev_emb
    
    sample = base.conv_in(unet_input)
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


# ============== Utils ==============
def dilate_mask_by_level(masks: torch.Tensor, levels: torch.Tensor, device) -> torch.Tensor:
    """
    按等级动态膨胀 mask（在图像空间）
    
    Args:
        masks: [B, 1, H, W] in [0, 1]
        levels: [B] 等级标签
        device: 设备
    
    Returns:
        dilated masks: [B, 1, H, W]
    """
    B = masks.shape[0]
    dilated_masks = []
    
    for i in range(B):
        mask = masks[i:i+1]  # [1, 1, H, W]
        level = levels[i].item()
        k = config.mask_dilate_by_level.get(level, 0)
        
        if k > 0:
            if k % 2 == 0:
                k += 1
            # 用 max_pool2d 做膨胀
            mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
        
        dilated_masks.append(mask)
    
    return torch.cat(dilated_masks, dim=0)


def create_feather_mask(mask_tensor: torch.Tensor, feather_pixels: int = 25) -> torch.Tensor:
    """
    创建软边界 mask（feather）- P0 关键改动
    
    把硬边界的二值 mask 转换为边缘渐变的 soft mask，
    让外扩 ring 区域不是"原图硬贴"，而是"生成主导的过渡带"
    
    Args:
        mask_tensor: [B, 1, H, W] 二值 mask (0 或 1)
        feather_pixels: 羽化像素数（边缘渐变宽度）
    
    Returns:
        soft_mask: [B, 1, H, W] 边缘渐变的 mask，值域 [0, 1]
    """
    if feather_pixels <= 0:
        return mask_tensor
    
    # 用高斯模糊实现羽化效果
    k = feather_pixels * 2 + 1
    sigma = feather_pixels / 3.0
    
    # 创建高斯核
    x = torch.arange(k, device=mask_tensor.device, dtype=mask_tensor.dtype) - k // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.view(1, 1, -1, 1) * gauss_1d.view(1, 1, 1, -1)
    gauss_2d = gauss_2d.view(1, 1, k, k)
    
    # 对每个 batch 应用高斯模糊
    B = mask_tensor.shape[0]
    soft_masks = []
    for i in range(B):
        m = mask_tensor[i:i+1]  # [1, 1, H, W]
        padded = F.pad(m, (k//2, k//2, k//2, k//2), mode='replicate')
        soft = F.conv2d(padded, gauss_2d)
        soft_masks.append(soft)
    
    return torch.cat(soft_masks, dim=0).clamp(0, 1)


def sync_new_token_embedding(text_encoder, new_token_ids, new_token_module):
    """同步新 token embedding 到 text encoder"""
    te = text_encoder.module if hasattr(text_encoder, 'module') else text_encoder
    ntm = new_token_module.module if hasattr(new_token_module, 'module') else new_token_module
    te.text_model.embeddings.token_embedding.weight.data[new_token_ids] = ntm().detach()


def encode_with_new_tokens(text_encoder, input_ids, new_token_ids, new_token_emb):
    te = text_encoder.module if hasattr(text_encoder, 'module') else text_encoder
    tok_emb_layer = te.text_model.embeddings.token_embedding
    inputs_embeds = tok_emb_layer(input_ids)
    inputs_embeds = inputs_embeds.clone()
    for i, tid in enumerate(new_token_ids):
        mask = input_ids == tid
        if mask.any():
            inputs_embeds[mask] = new_token_emb[i].to(dtype=inputs_embeds.dtype)
    return te(inputs_embeds=inputs_embeds)[0]


def load_stage1_checkpoint(checkpoint_path: str, device):
    """加载 Stage 1 checkpoint"""
    print(f"Loading Stage 1 checkpoint from: {checkpoint_path}")
    
    tok_path = os.path.join(checkpoint_path, "tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(tok_path) if os.path.exists(tok_path) else None
    
    te_path = os.path.join(checkpoint_path, "text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(te_path) if os.path.exists(te_path) else None
    
    sev_path = os.path.join(checkpoint_path, "severity_embedding.pt")
    severity_state = torch.load(sev_path, map_location=device) if os.path.exists(sev_path) else None
    
    lora_path = os.path.join(checkpoint_path, "unet_lora")
    
    return tokenizer, text_encoder, severity_state, lora_path


def save_checkpoint(unet, severity_embedding, new_token_module, tokenizer, text_encoder, 
                    save_path, accelerator, merged_base_path=None):
    """保存 checkpoint"""
    os.makedirs(save_path, exist_ok=True)
    
    te = text_encoder.module if hasattr(text_encoder, 'module') else text_encoder
    ntm = new_token_module.module if hasattr(new_token_module, 'module') else new_token_module
    
    # 保存 l4 LoRA adapter
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(os.path.join(save_path, "unet_lora"))
    
    if severity_embedding is not None:
        sev = severity_embedding.module if hasattr(severity_embedding, 'module') else severity_embedding
        torch.save(sev.state_dict(), os.path.join(save_path, "severity_embedding.pt"))
    
    torch.save(ntm.state_dict(), os.path.join(save_path, "new_token_embedding.pt"))
    
    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    te.save_pretrained(os.path.join(save_path, "text_encoder"))
    
    # 记录 merged base 路径，推理时需要
    config_data = {
        "l4_lora_rank": config.l4_lora_rank,
        "stage": "2",
        "note": "shared LoRA merged into base, l4 is incremental adapter",
        "merged_base_path": merged_base_path,
    }
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config_data, f, indent=2)


def print_trainable_params(model, name_prefix="Model"):
    """打印可训练参数统计"""
    _model = model.module if hasattr(model, 'module') else model
    trainable = 0
    total = 0
    trainable_names = []
    for name, param in _model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)
    
    print(f"{name_prefix}: {trainable:,} / {total:,} trainable params ({100*trainable/total:.2f}%)")
    print(f"  Trainable layers: {len(trainable_names)}")
    for n in trainable_names[:5]:
        print(f"    - {n}")
    if len(trainable_names) > 5:
        print(f"    ... and {len(trainable_names)-5} more")


# ============== 可视化函数 ==============
@torch.no_grad()
def run_visualization(unet, vae, text_encoder, noise_scheduler, severity_embedding,
                      dataset, tokenizer, visualizer, accelerator, global_step, 
                      new_token_ids, new_token_module, num_samples=4):
    """
    可视化：所有等级都用同一个模型（base+shared_merged+l4）
    
    改进：
    1. L4 使用专用 prompt（debris field）和 negative prompt
    2. L3/L4 使用软边界回填（feather mask）
    3. L4 降低 CFG
    """
    unet.eval()
    device = accelerator.device
    if len(dataset) == 0:
        return
    ntm = new_token_module.module if hasattr(new_token_module, 'module') else new_token_module
    new_token_emb = ntm()
    
    sample_count = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_count, replace=False)
    
    all_originals, all_masks = [], []
    level_outputs = {1: [], 2: [], 3: [], 4: []}
    
    # 通用 negative prompt
    neg_inputs = tokenizer("blurry, low quality, distorted", padding="max_length", 
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_text_emb_general = encode_with_new_tokens(
        text_encoder, neg_inputs.input_ids.to(device), new_token_ids, new_token_emb
    )
    
    # L4 专用 negative prompt（P2: 抑制 floorplan/rooms）
    l4_neg_inputs = tokenizer(L4_PROMPTS["negative"], padding="max_length",
                              max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_text_emb_l4 = encode_with_new_tokens(
        text_encoder, l4_neg_inputs.input_ids.to(device), new_token_ids, new_token_emb
    )
    
    uncond_severity_id = torch.tensor([config.severity_uncond_id], device=device)
    
    sev_emb = severity_embedding.module if hasattr(severity_embedding, 'module') else severity_embedding
    
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device, dtype=torch.float32)
        mask = sample["mask"].unsqueeze(0).to(device, dtype=torch.float32)
        
        image_latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        
        all_originals.append(image.cpu())
        
        # 显示 L4 的膨胀 mask（最大膨胀），让用户能直观看到膨胀效果
        level_tensor_l4 = torch.tensor([4], device=device)
        mask_dilated_l4 = dilate_mask_by_level(mask, level_tensor_l4, device)
        all_masks.append(mask_dilated_l4.cpu())
        
        for level in [1, 2, 3, 4]:
            # 按目标等级动态调整 strength 和 CFG
            local_strength = eval_config.strength_by_level.get(level, 0.8)
            local_guidance = eval_config.guidance_by_level.get(level, 7.5)
            
            # ===== 按目标等级动态膨胀 mask =====
            level_tensor = torch.tensor([level], device=device)
            mask_dilated = dilate_mask_by_level(mask, level_tensor, device)
            
            # 调试：打印膨胀前后的 mask 面积变化
            orig_area = mask.sum().item()
            dilated_area = mask_dilated.sum().item()
            if level in [3, 4] and idx == indices[0]:
                print(f"  L{level} mask: {orig_area:.0f} -> {dilated_area:.0f} pixels (+{(dilated_area/orig_area-1)*100:.1f}%)")
            
            masked_image_dilated = image * (1 - mask_dilated)
            
            masked_latents = vae.encode(masked_image_dilated).latent_dist.sample() * vae.config.scaling_factor
            mask_latent = F.interpolate(mask_dilated, size=image_latents.shape[-2:], mode='nearest')
            
            # P2: L4 使用专用 prompt
            if level == 4:
                prompt = L4_PROMPTS["positive"]
                uncond_text_emb = uncond_text_emb_l4
            else:
                prompt = f"aerial view of post-disaster building damage <L{level}>"
                uncond_text_emb = uncond_text_emb_general
            
            text_inputs = tokenizer(prompt, padding="max_length", 
                                    max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            cond_text_emb = encode_with_new_tokens(
                text_encoder, text_inputs.input_ids.to(device), new_token_ids, new_token_emb
            )
            cond_severity_id = torch.tensor([config.level_to_severity[level]], device=device)
            
            noise_scheduler.set_timesteps(eval_config.num_inference_steps, device=device)
            init_timestep = int(eval_config.num_inference_steps * local_strength)
            t_start = max(eval_config.num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            
            noise = torch.randn_like(image_latents)
            latents = noise_scheduler.add_noise(image_latents, noise, timesteps[:1])
            
            for t in timesteps:
                noise_cond = unet_forward_with_severity(
                    unet, latents, t.unsqueeze(0), cond_text_emb, 
                    sev_emb, cond_severity_id, mask_latent, masked_latents
                )
                noise_uncond = unet_forward_with_severity(
                    unet, latents, t.unsqueeze(0), uncond_text_emb, 
                    sev_emb, uncond_severity_id, mask_latent, masked_latents
                )
                noise_pred = noise_uncond + local_guidance * (noise_cond - noise_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            generated = vae.decode(latents / vae.config.scaling_factor).sample
            
            # P0: L3/L4 使用软边界回填（feather mask），L1/L2 用硬边界
            if level >= 3 and config.use_soft_mask:
                soft_mask = create_feather_mask(mask_dilated, config.feather_pixels)
                generated = generated * soft_mask + image * (1 - soft_mask)
            else:
                generated = generated * mask_dilated + image * (1 - mask_dilated)
            
            generated = (generated / 2 + 0.5).clamp(0, 1)
            level_outputs[level].append(generated.cpu())
    
    all_originals = torch.cat(all_originals, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    for level in [1, 2, 3, 4]:
        level_outputs[level] = torch.cat(level_outputs[level], dim=0)
    
    if accelerator.is_main_process:
        visualizer.save_level_comparison(global_step, all_originals, all_masks, level_outputs)
    
    unet.train()


# ============== 训练步骤（只训练 L4 样本）==============
def train_step_l4_only(
    unet, vae, text_encoder, noise_scheduler, severity_embedding, new_token_module,
    batch, new_token_ids, device, accelerator
):
    """
    方案 2：shared 已 merge 到基座，只用 l4 adapter
    所有样本都走同一个 forward，但只有 L4 样本产生 loss
    + 按等级动态膨胀 mask（L3/L4 膨胀更多）
    
    关键改动：非 L4 样本的 loss 权重设为 0，只有 L4 才更新 l4 adapter
    """
    images = batch["image"]
    masked_images = batch["masked_image"]
    masks = batch["mask"]
    input_ids = batch["input_ids"]
    severity_ids = batch["severity_id"].to(device)
    is_l4 = batch["is_l4"].to(device).float()
    levels = batch["level"].to(device)  # 需要等级信息做动态膨胀
    
    B = images.shape[0]
    
    # ===== 按等级动态膨胀 mask =====
    # 在图像空间膨胀，然后再下采样到 latent 空间
    masks_dilated = dilate_mask_by_level(masks, levels, device)
    
    # 用膨胀后的 mask 重新生成 masked_image（只对 L3/L4 有效果）
    # masked_image = image * (1 - mask_dilated) + 0.5 * mask_dilated  # 0.5 对应归一化后的灰色
    # 注意：images 已经是 [-1, 1] 范围，灰色是 0
    masked_images_dilated = images * (1 - masks_dilated)
    
    # Severity dropout
    if config.p_uncond_severity > 0:
        drop = torch.rand(B, device=device) < config.p_uncond_severity
        severity_ids = torch.where(drop, config.severity_uncond_id, severity_ids)
    
    # VAE encode（使用膨胀后的 mask 和 masked_image）
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        masked_latents = vae.encode(masked_images_dilated).latent_dist.sample() * vae.config.scaling_factor
    
    mask_latent = F.interpolate(masks_dilated, size=latents.shape[-2:], mode='nearest')
    
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Text encoding (inject new token embeddings)
    ntm = new_token_module.module if hasattr(new_token_module, 'module') else new_token_module
    encoder_hidden_states = encode_with_new_tokens(
        text_encoder, input_ids, new_token_ids, ntm()
    )
    
    sev_emb = severity_embedding.module if hasattr(severity_embedding, 'module') else severity_embedding
    
    # Forward（所有样本用同一个模型）
    noise_pred = unet_forward_with_severity(
        unet, noisy_latents, timesteps, encoder_hidden_states,
        sev_emb, severity_ids, mask_latent, masked_latents
    )
    
    # ===== 三段权重：background / ring / interior =====
    mask_bin = (mask_latent > 0.5).float()  # (B,1,H,W)
    
    k = int(config.ring_dilate_kernel)
    if k % 2 == 0:
        k += 1  # 保证奇数
    
    # 用 max_pool 做二值 mask 的膨胀（比逐样本 opencv 简洁，GPU 友好）
    mask_dilated_latent = F.max_pool2d(mask_bin, kernel_size=k, stride=1, padding=k // 2)
    ring = (mask_dilated_latent - mask_bin).clamp(0, 1)  # dilated 去掉 interior 的部分
    
    # ring 权重：只对 L4 提升；非 L4 时 ring=background 权重（避免背景漂移）
    is_l4_4d = is_l4.view(-1, 1, 1, 1)  # (B,1,1,1)
    ring_w = config.weight_background + (config.weight_ring_l4 - config.weight_background) * is_l4_4d
    
    # 三段合成 base_weight
    # background 区域是：既不在 interior 也不在 ring
    bg = (1.0 - mask_bin - ring).clamp(0, 1)
    base_weight = (
        config.weight_interior * mask_bin
        + ring_w * ring
        + config.weight_background * bg
    )
    
    # 关键改动：非 L4 样本的 loss 权重设为 0，只有 L4 才更新 l4 adapter
    # 这样 l4 adapter 不会被 L1-L3 的"保结构"分布拉扯
    sample_weight = torch.where(is_l4_4d > 0.5, config.l4_loss_weight, 0.0)
    weight = base_weight * sample_weight
    
    loss = (F.mse_loss(noise_pred, noise, reduction='none').mean(dim=1, keepdim=True) * weight).mean()
    
    return loss


# ============== Stage 2 训练 ==============
def train_stage2(args):
    print("\n" + "="*60)
    print("Stage 2: Merge Shared LoRA + Train L4 Adapter")
    print("="*60 + "\n")
    
    # ===== 版本校验：确保跑的是新版代码 =====
    assert hasattr(eval_config, "guidance_by_level"), "错误：跑到了旧版脚本！缺少 guidance_by_level"
    assert hasattr(config, "mask_dilate_by_level"), "错误：跑到了旧版脚本！缺少 mask_dilate_by_level"
    assert hasattr(config, "use_soft_mask"), "错误：跑到了旧版脚本！缺少 use_soft_mask"
    assert config.mask_dilate_by_level[4] >= 60, f"错误：L4 膨胀量太小！当前={config.mask_dilate_by_level[4]}，应>=60"
    assert config.weight_ring_l4 >= 0.8, f"错误：L4 ring 权重太小！当前={config.weight_ring_l4}，应>=0.8"
    
    print("✓ 版本校验通过：大膨胀 + 高 ring 权重 + 软边界 + L4 专用 prompt")
    print(f"\n关键配置（P0 - 消除建筑边界线索）:")
    print(f"  mask_dilate_by_level: {config.mask_dilate_by_level}")
    print(f"  use_soft_mask: {config.use_soft_mask}, feather_pixels: {config.feather_pixels}")
    print(f"\n关键配置（P1 - 强化边界破碎学习）:")
    print(f"  ring_dilate_kernel: {config.ring_dilate_kernel}")
    print(f"  weight_ring_l4: {config.weight_ring_l4}")
    print(f"\n关键配置（P2 - 抑制语义拉回）:")
    print(f"  guidance_by_level: {eval_config.guidance_by_level}")
    print(f"  L4 positive: {L4_PROMPTS['positive'][:60]}...")
    print(f"  L4 negative: {L4_PROMPTS['negative'][:60]}...")
    print()
    
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        mixed_precision=config.mixed_precision
    )
    set_seed(config.seed)
    
    visualizer = TrainingVisualizer(output_dir)
    device = accelerator.device
    
    # ========== 加载 Stage 1 checkpoint ==========
    tokenizer_ckpt, text_encoder_ckpt, severity_state, lora_path = load_stage1_checkpoint(
        config.checkpoint_path, device
    )
    
    # ========== 初始化模型 ==========
    if tokenizer_ckpt is not None:
        tokenizer = tokenizer_ckpt
    else:
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model, subfolder="tokenizer")
        tokenizer.add_tokens(config.level_tokens)
    
    original_vocab_size = len(tokenizer) - len(config.level_tokens)
    new_token_ids = list(range(original_vocab_size, len(tokenizer)))
    
    if text_encoder_ckpt is not None:
        text_encoder = text_encoder_ckpt
    else:
        text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model, subfolder="text_encoder")
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    vae = AutoencoderKL.from_pretrained(config.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    
    # ========== 关键：加载 shared LoRA 并 merge 到基座 ==========
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model, subfolder="unet")
    
    if os.path.exists(lora_path):
        print(f"Loading shared LoRA from: {lora_path}")
        # 用 PEFT 加载 shared adapter
        unet = PeftModel.from_pretrained(unet, lora_path, adapter_name="shared")
        print(f"✓ Loaded shared LoRA")
        
        # Merge shared LoRA 到基座权重
        print("Merging shared LoRA into base weights...")
        unet = unet.merge_and_unload()
        print(f"✓ Shared LoRA merged into base (now part of UNet weights)")
    else:
        print(f"Warning: No shared LoRA found at {lora_path}, starting from pretrained")
    
    # 保存 merged base（可选，用于推理时加载）
    merged_base_path = os.path.join(output_dir, "merged_base_unet")
    if accelerator.is_main_process:
        os.makedirs(merged_base_path, exist_ok=True)
        unet.save_pretrained(merged_base_path)
        print(f"✓ Saved merged base UNet to: {merged_base_path}")
    
    # ========== 添加 L4 专用 LoRA（在 merged base 上）==========
    l4_lora_config = LoraConfig(
        r=config.l4_lora_rank, 
        lora_alpha=config.l4_lora_alpha, 
        lora_dropout=config.lora_dropout,
        init_lora_weights="gaussian", 
        target_modules=config.l4_lora_target_modules
    )
    unet = get_peft_model(unet, l4_lora_config, adapter_name="l4")
    print(f"✓ Added L4 LoRA adapter (rank={config.l4_lora_rank}) on merged base")
    
    # Severity embedding
    time_embed_dim = unet.base_model.model.config.block_out_channels[0] * 4
    severity_embedding = SeverityEmbedding(config.num_severity_classes, time_embed_dim)
    if severity_state is not None:
        severity_embedding.load_state_dict(severity_state)
        print(f"✓ Loaded severity embedding from Stage 1")
    
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")
    
    # 新 token embedding
    text_encoder.requires_grad_(False)
    embed_dim = text_encoder.text_model.embeddings.token_embedding.weight.shape[1]
    new_token_module = NewTokenEmbedding(len(new_token_ids), embed_dim)
    new_token_module.init_from_pretrained(
        text_encoder.text_model.embeddings.token_embedding.weight[new_token_ids].clone()
    )
    
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    print("\n参数统计:")
    print_trainable_params(unet, "UNet (l4 adapter)")
    
    # ========== 数据集 ==========
    all_samples = load_metadata_samples(config.data_dir)
    train_samples, val_samples = stratified_train_val_split(
        all_samples, config.val_ratio, config.seed, config.min_val_per_level
    )
    train_l4_samples = [sample for sample in train_samples if sample["level"] == 4]
    if not train_l4_samples:
        raise ValueError("Stage 2 requires at least one L4 sample in the training split.")

    train_dataset = DisasterDataset(train_l4_samples, tokenizer, config.resolution, split_name="train_l4")
    val_dataset = DisasterDataset(val_samples, tokenizer, config.resolution, split_name="val")
    vis_dataset = val_dataset if len(val_dataset) > 0 else train_dataset
    if len(val_dataset) == 0:
        print("Warning: validation split is empty, visualization falls back to the training split.")

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # ========== 优化器 ==========
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    print(f"\nL4 adapter trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW([
        {"params": trainable_params, "lr": config.l4_lora_lr},
        {"params": severity_embedding.parameters(), "lr": config.learning_rate},
        {"params": new_token_module.parameters(), "lr": config.learning_rate},
    ])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps, eta_min=1e-6)
    
    # ========== Accelerate 准备 ==========
    unet, text_encoder, severity_embedding, new_token_module, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, severity_embedding, new_token_module, optimizer, dataloader, lr_scheduler
    )
    vae.to(device, dtype=torch.float32)
    
    print(f"\nTraining config:")
    print(f"  max_steps={config.max_steps}, save_steps={config.save_steps}, eval_steps={eval_config.eval_steps}")
    print(f"  Dataset split: train_l4={len(train_dataset)}, val={len(val_dataset)}")
    print(f"  L4 loss weight={config.l4_loss_weight}")
    print(f"  Strategy: shared merged into base, only l4 adapter trainable (lr={config.l4_lora_lr})")
    
    global_step = 0
    
    # 初始可视化
    accelerator.wait_for_everyone()
    run_visualization(unet, vae, text_encoder, noise_scheduler, severity_embedding, 
                      vis_dataset, tokenizer, visualizer, accelerator, 0, 
                      new_token_ids, new_token_module, num_samples=4)
    accelerator.wait_for_everyone()
    
    # ========== 训练循环 ==========
    while global_step < config.max_steps:
        unet.train()
        severity_embedding.train()
        new_token_module.train()
        
        for batch in tqdm(dataloader, desc=f"Stage2 | Step {global_step}", 
                          disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(unet):
                loss = train_step_l4_only(
                    unet, vae, text_encoder, noise_scheduler, severity_embedding, new_token_module,
                    batch, new_token_ids, device, accelerator
                )
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    all_params = trainable_params + list(severity_embedding.parameters()) + list(new_token_module.parameters())
                    accelerator.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % eval_config.eval_steps == 0:
                accelerator.wait_for_everyone()
                run_visualization(unet, vae, text_encoder, noise_scheduler, severity_embedding, 
                                  vis_dataset, tokenizer, visualizer, accelerator, global_step, 
                                  new_token_ids, new_token_module, num_samples=4)
                accelerator.wait_for_everyone()
            
            if global_step % config.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    sync_new_token_embedding(text_encoder, new_token_ids, new_token_module)
                    save_checkpoint(unet, severity_embedding, new_token_module, tokenizer, text_encoder, 
                                    os.path.join(output_dir, f"checkpoint-{global_step}"), accelerator,
                                    merged_base_path=merged_base_path)
                    print(f"\n💾 Saved checkpoint-{global_step}")
                accelerator.wait_for_everyone()
            
            if global_step >= config.max_steps:
                break
    
    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        sync_new_token_embedding(text_encoder, new_token_ids, new_token_module)
        save_checkpoint(unet, severity_embedding, new_token_module, tokenizer, text_encoder, 
                        os.path.join(output_dir, "final"), accelerator,
                        merged_base_path=merged_base_path)
        print(f"\n✅ Stage 2 complete! Saved to: {output_dir}/final")
        print(f"   Merged base UNet: {merged_base_path}")
        print(f"   L4 LoRA adapter: {output_dir}/final/unet_lora")
    accelerator.wait_for_everyone()


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="Stage 2: Merge Shared + Train L4 LoRA")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--l4_lr", type=float, default=None)
    parser.add_argument("--l4_ratio", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    
    args = parser.parse_args()
    
    config.checkpoint_path = args.checkpoint
    if args.output_dir: config.output_dir = args.output_dir
    if args.save_steps: config.save_steps = args.save_steps
    if args.eval_steps: eval_config.eval_steps = args.eval_steps
    if args.batch_size: config.train_batch_size = args.batch_size
    if args.lr: config.learning_rate = args.lr
    if args.l4_lr: config.l4_lora_lr = args.l4_lr
    if args.l4_ratio is not None:
        print("Warning: --l4_ratio is deprecated because Stage 2 now trains on the L4-only split.")
    if args.max_steps: config.max_steps = args.max_steps
    if args.val_ratio is not None: config.val_ratio = args.val_ratio
    
    train_stage2(args)


if __name__ == "__main__":
    main()
