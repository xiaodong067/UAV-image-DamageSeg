"""
Stage 1 训练脚本：Severity Embedding + LoRA + 新 token embedding（所有等级混合训练）

使用方式：
    HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes=2 \
        disaster_lora/stage1_train.py --max_steps 50000 --save_steps 5000 --eval_steps 500
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
from peft import LoraConfig, get_peft_model
import numpy as np
from torchvision import transforms
from typing import Dict, List, Optional
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from evaluation_utils import TrainingVisualizer


# ============== 配置 ==============
class Config:
    pretrained_model = "runwayml/stable-diffusion-inpainting"
    data_dir = "./disaster_lora/training_data"
    output_dir = "./disaster_lora/output/stage1"
    val_ratio = 0.1
    min_val_per_level = 1
    
    resolution = 512
    train_batch_size = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    seed = 42
    mixed_precision = "fp16"
    gradient_checkpointing = True
    save_steps = 5000
    max_steps = 30000
    
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"]
    
    num_severity_classes = 5
    severity_uncond_id = 4
    p_uncond_severity = 0.1
    level_to_severity = {1: 0, 2: 1, 3: 2, 4: 3}
    
    weight_background = 0.1
    weight_interior = 1.0
    level_tokens = ["<L1>", "<L2>", "<L3>", "<L4>"]


class EvalConfig:
    eval_steps = 500
    eval_batch_size = 4
    num_eval_samples = 64
    num_inference_steps = 50


config = Config()
eval_config = EvalConfig()


def load_metadata_samples(data_dir: str) -> List[Dict]:
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


def stratified_train_val_split(samples: List[Dict], val_ratio: float, seed: int, min_val_per_level: int = 1):
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


# ============== 模块 ==============
class SeverityEmbeddingWrapper(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, severity_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(severity_ids)


class NewTokenEmbedding(nn.Module):
    """New token embedding module for DDP-safe training."""
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_tokens, embed_dim))
    
    def init_from_pretrained(self, pretrained_weight: torch.Tensor):
        self.weight.data.copy_(pretrained_weight)
    
    def forward(self) -> torch.Tensor:
        return self.weight


# ============== Dataset ==============
class DisasterDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: CLIPTokenizer, resolution: int = 512, split_name: str = "train"):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.split_name = split_name
        self.samples = list(samples)
        
        # 加载metadata.json获取样本信息
        grouped = defaultdict(int)
        for sample in self.samples:
            grouped[sample["level"]] += 1
        
        # 适配你的数据格式：从metadata中的samples获取路径信息
        print(f"{self.split_name} dataset: {len(self.samples)} samples")
        for level in sorted(grouped.keys()):
            # 你的数据格式中路径已经是完整路径，直接使用
            print(f"  L{level}: {grouped[level]} samples")
        
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
        
        # 直接使用完整路径加载图像
        image = self.image_transforms(Image.open(sample['image_path']).convert('RGB'))
        masked_image = self.image_transforms(Image.open(sample['masked_path']).convert('RGB'))
        mask = self.mask_transforms(Image.open(sample['mask_path']).convert('L'))
        
        level = sample['level']
        prompt = f"aerial view of post-disaster building damage <L{level}>"
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        return {
            "image": image, "masked_image": masked_image, "mask": mask,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "level": level, "severity_id": config.level_to_severity.get(level, config.severity_uncond_id),
        }


# ============== Utils ==============
def compute_mask_weights(mask_latent: torch.Tensor) -> torch.Tensor:
    return torch.where(mask_latent > 0.5, config.weight_interior, config.weight_background)


def sync_new_token_embedding(text_encoder, new_token_ids, new_token_module):
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


def unet_forward_with_severity(unet, sample, timestep, encoder_hidden_states, severity_wrapper, severity_ids):
    _unet = unet.module if hasattr(unet, 'module') else unet
    base = _unet.base_model.model if hasattr(_unet, 'base_model') else _unet
    
    t_emb = base.time_proj(timestep).to(dtype=sample.dtype)
    t_emb = base.time_embedding(t_emb)
    if severity_wrapper is not None:
        t_emb = t_emb + severity_wrapper(severity_ids).to(dtype=t_emb.dtype)
    
    return unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


def save_checkpoint(unet, severity_wrapper, new_token_module, new_token_ids, tokenizer, text_encoder, save_path, accelerator):
    os.makedirs(save_path, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet_lora"))
    
    if severity_wrapper is not None:
        sw = severity_wrapper.module if hasattr(severity_wrapper, 'module') else severity_wrapper
        torch.save(sw.state_dict(), os.path.join(save_path, "severity_embedding.pt"))
    
    sync_new_token_embedding(text_encoder, new_token_ids, new_token_module)
    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    te = text_encoder.module if hasattr(text_encoder, 'module') else text_encoder
    te.save_pretrained(os.path.join(save_path, "text_encoder"))
    
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump({"lora_rank": config.lora_rank, "stage": 1}, f, indent=2)


# ============== 可视化函数 ==============
# 每个等级的 strength 配置（统一使用相同值，让模型控制等级差异）
LEVEL_STRENGTH = {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8}


def unet_forward_with_severity_manual(unet, sample, timestep, encoder_hidden_states, severity_wrapper, severity_ids):
    """手动注入 severity embedding 的 UNet forward（和推理脚本一致）"""
    _unet = unet.module if hasattr(unet, 'module') else unet
    base = _unet.base_model.model if hasattr(_unet, 'base_model') else _unet
    
    t_emb = base.time_proj(timestep).to(dtype=sample.dtype)
    emb = base.time_embedding(t_emb)
    if severity_wrapper is not None and severity_ids is not None:
        sw = severity_wrapper.module if hasattr(severity_wrapper, 'module') else severity_wrapper
        emb = emb + sw(severity_ids).to(dtype=emb.dtype)
    
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


@torch.no_grad()
def run_visualization(unet, vae, text_encoder, noise_scheduler, severity_wrapper,
                      dataset, tokenizer, visualizer, accelerator, global_step, 
                      new_token_ids=None, new_token_module=None, num_samples=4):
    """
    随机选 4 张图，每张生成 L1-L4 四个等级，保存对比图
    使用和推理脚本一致的设置：CFG + img2img strength + composite
    """
    unet.eval()
    device = accelerator.device
    if len(dataset) == 0:
        return
    
    new_token_emb = None
    if new_token_ids is not None and new_token_module is not None:
        ntm = new_token_module.module if hasattr(new_token_module, 'module') else new_token_module
        new_token_emb = ntm()
    
    sample_count = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_count, replace=False)
    
    all_originals = []
    all_masks = []
    level_outputs = {1: [], 2: [], 3: [], 4: []}
    
    guidance_scale = 7.5
    num_inference_steps = 50
    negative_prompt = "blurry, low quality, distorted"
    
    neg_inputs = tokenizer(negative_prompt, padding="max_length", 
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    if new_token_emb is not None:
        uncond_emb = encode_with_new_tokens(text_encoder, neg_inputs.input_ids.to(device), new_token_ids, new_token_emb)
    else:
        uncond_emb = text_encoder(neg_inputs.input_ids.to(device))[0]
    uncond_severity_id = torch.tensor([config.severity_uncond_id], device=device)
    
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device, dtype=torch.float32)
        masked_image = sample["masked_image"].unsqueeze(0).to(device, dtype=torch.float32)
        mask = sample["mask"].unsqueeze(0).to(device, dtype=torch.float32)
        
        image_latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        masked_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor
        mask_latent = F.interpolate(mask, size=image_latents.shape[-2:], mode='nearest')
        
        all_originals.append(image.cpu())
        all_masks.append(mask.cpu())
        
        for level in [1, 2, 3, 4]:
            prompt = f"aerial view of post-disaster building damage <L{level}>"
            text_inputs = tokenizer(prompt, padding="max_length", 
                                    max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            if new_token_emb is not None:
                text_emb = encode_with_new_tokens(text_encoder, text_inputs.input_ids.to(device), new_token_ids, new_token_emb)
            else:
                text_emb = text_encoder(text_inputs.input_ids.to(device))[0]
            
            severity_id = torch.tensor([config.level_to_severity[level]], device=device)
            
            strength = LEVEL_STRENGTH[level]
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            init_timestep = int(num_inference_steps * strength)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            
            noise = torch.randn_like(image_latents)
            latents = noise_scheduler.add_noise(image_latents, noise, timesteps[:1])
            
            for t in timesteps:
                unet_input = torch.cat([latents, mask_latent, masked_latents], dim=1)
                noise_cond = unet_forward_with_severity_manual(
                    unet, unet_input, t.unsqueeze(0), text_emb, severity_wrapper, severity_id
                )
                noise_uncond = unet_forward_with_severity_manual(
                    unet, unet_input, t.unsqueeze(0), uncond_emb, severity_wrapper, uncond_severity_id
                )
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            generated = vae.decode(latents / vae.config.scaling_factor).sample
            generated = generated * mask + image * (1 - mask)
            generated = (generated / 2 + 0.5).clamp(0, 1)
            level_outputs[level].append(generated.cpu())
    
    all_originals = torch.cat(all_originals, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    for level in [1, 2, 3, 4]:
        level_outputs[level] = torch.cat(level_outputs[level], dim=0)
    
    if accelerator.is_main_process:
        visualizer.save_level_comparison(global_step, all_originals, all_masks, level_outputs)
    
    unet.train()


# ============== Stage 1 训练 ==============
def train_stage1(args):
    print("\n" + "="*60)
    print("Stage 1: Severity Embedding + LoRA + Token Embedding")
    print("="*60 + "\n")
    
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps, mixed_precision=config.mixed_precision)
    set_seed(config.seed)
    
    visualizer = TrainingVisualizer(output_dir)
    
    # 加载模型
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model, subfolder="tokenizer")
    original_vocab_size = len(tokenizer)
    tokenizer.add_tokens(config.level_tokens)
    new_token_ids = list(range(original_vocab_size, len(tokenizer)))
    print(f"Added tokens {config.level_tokens} -> ids {new_token_ids}")
    
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model, subfolder="text_encoder")
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    vae = AutoencoderKL.from_pretrained(config.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")
    
    lora_cfg = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                          init_lora_weights="gaussian", target_modules=config.lora_target_modules)
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()
    
    time_embed_dim = unet.base_model.model.config.block_out_channels[0] * 4
    severity_wrapper = SeverityEmbeddingWrapper(config.num_severity_classes, time_embed_dim)
    
    # 冻结 text encoder，只训练新 token
    text_encoder.requires_grad_(False)
    embed_dim = text_encoder.text_model.embeddings.token_embedding.weight.shape[1]
    new_token_module = NewTokenEmbedding(len(new_token_ids), embed_dim)
    new_token_module.init_from_pretrained(
        text_encoder.text_model.embeddings.token_embedding.weight[new_token_ids].clone()
    )
    print(f"New token embedding: {new_token_module.weight.numel()} params")
    
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    all_samples = load_metadata_samples(config.data_dir)
    train_samples, val_samples = stratified_train_val_split(
        all_samples, config.val_ratio, config.seed, config.min_val_per_level
    )
    train_dataset = DisasterDataset(train_samples, tokenizer, config.resolution, split_name="train")
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
    
    trainable_params = (
        list(filter(lambda p: p.requires_grad, unet.parameters()))
        + list(new_token_module.parameters())
        + list(severity_wrapper.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps, eta_min=1e-6)
    
    unet, text_encoder, severity_wrapper, new_token_module, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, severity_wrapper, new_token_module, optimizer, dataloader, lr_scheduler)
    vae.to(accelerator.device, dtype=torch.float32)
    
    print(f"Training: max_steps={config.max_steps}, save_steps={config.save_steps}, eval_steps={eval_config.eval_steps}")
    print(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}")
    global_step = 0
    
    # 初始可视化
    accelerator.wait_for_everyone()
    run_visualization(unet, vae, text_encoder, noise_scheduler, severity_wrapper, 
                      vis_dataset, tokenizer, visualizer, accelerator, 0, 
                      new_token_ids, new_token_module, num_samples=4)
    accelerator.wait_for_everyone()
    
    while global_step < config.max_steps:
        unet.train()
        severity_wrapper.train()
        
        for batch in tqdm(dataloader, desc=f"Stage1 | Step {global_step}", disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(unet):
                images, masked_images, masks = batch["image"], batch["masked_image"], batch["mask"]
                input_ids = batch["input_ids"]
                severity_ids = batch["severity_id"].to(accelerator.device)
                
                if config.p_uncond_severity > 0:
                    drop = torch.rand(severity_ids.shape[0], device=severity_ids.device) < config.p_uncond_severity
                    severity_ids = torch.where(drop, config.severity_uncond_id, severity_ids)
                
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                    masked_latents = vae.encode(masked_images).latent_dist.sample() * vae.config.scaling_factor
                
                mask_latent = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
                weight_map = compute_mask_weights(mask_latent)
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                new_token_emb = new_token_module()
                encoder_hidden_states = encode_with_new_tokens(text_encoder, input_ids, new_token_ids, new_token_emb)
                
                unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
                noise_pred = unet_forward_with_severity_manual(
                    unet, unet_input, timesteps, encoder_hidden_states, severity_wrapper, severity_ids
                )
                
                loss = (F.mse_loss(noise_pred, noise, reduction='none').mean(dim=1, keepdim=True) * weight_map).mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % eval_config.eval_steps == 0:
                accelerator.wait_for_everyone()
                run_visualization(unet, vae, text_encoder, noise_scheduler, severity_wrapper, 
                                  vis_dataset, tokenizer, visualizer, accelerator, global_step, 
                                  new_token_ids, new_token_module, num_samples=4)
                accelerator.wait_for_everyone()
            
            if global_step % config.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(
                        unet, severity_wrapper, new_token_module, new_token_ids,
                        tokenizer, text_encoder, os.path.join(output_dir, f"checkpoint-{global_step}"), accelerator
                    )
                    print(f"\n💾 Saved checkpoint-{global_step}")
                accelerator.wait_for_everyone()
            
            if global_step >= config.max_steps:
                break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            unet, severity_wrapper, new_token_module, new_token_ids,
            tokenizer, text_encoder, os.path.join(output_dir, "final"), accelerator
        )
        print(f"\n✅ Stage 1 complete! Saved to: {output_dir}/final")
    accelerator.wait_for_everyone()


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training Script")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.save_steps:
        config.save_steps = args.save_steps
    if args.eval_steps:
        eval_config.eval_steps = args.eval_steps
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.max_steps:
        config.max_steps = args.max_steps
    
    train_stage1(args)


if __name__ == "__main__":
    main()
