"""
大图按建筑实例推理脚本

转换规则：L1→L3, L2→L2, L4→L4
输出三个文件夹：original/, mask/, result/
文件命名：1.jpg, 2.jpg... 和 1_mask.png, 2_mask.png...

使用方式：
    HF_ENDPOINT=https://hf-mirror.com python disaster_lora/inference_fullres.py \
        --checkpoint ./disaster_lora/output/stage2/checkpoint-14000 \
        --image_dir ./data/images \
        --mask_dir ./data/masks \
        --output ./inference_output/fullres_test \
        --num_images 10 \
        --seed 456
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import PeftModel
import json
import glob
from scipy import ndimage


class Config:
    pretrained_model = "runwayml/stable-diffusion-inpainting"
    num_severity_classes = 5
    severity_uncond_id = 4
    level_to_severity = {1: 0, 2: 1, 3: 2, 4: 3}
    
    # 转换规则：L1→L3, L2→L2, L4→L4
    transitions = {
        1: 3,  # L1 → L3
        2: 2,  # L2 → L2 (重新生成)
        4: 4,  # L4 → L4 (重新生成)
    }
    
    # 推理配置
    infer_size = 512
    padding = 50
    strength = 0.9
    cfg = 10
    dilate = 25
    feather = 25
    steps = 50


PROMPTS = {
    2: {
        "positive": "aerial view of minor damaged building <L2>",
        "negative": "collapsed, destroyed, rubble, debris field"
    },
    3: {
        "positive": "aerial view of major damaged building <L3>",
        "negative": "intact roof, undamaged, perfect condition"
    },
    4: {
        "positive": "aerial view of post-disaster damage <L4>",
        "negative": "house, structure, standing walls, intact roof, roof tiles"
    },
}

# 掩码颜色：转换后等级
TARGET_COLORS = {
    2: (255, 255, 0),    # 黄色 - L2
    3: (0, 128, 255),    # 蓝色 - L3
    4: (255, 0, 0),      # 红色 - L4
}


def load_models(checkpoint_path: str, device: str = "cuda"):
    print(f"Loading checkpoint from: {checkpoint_path}")
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        ckpt_config = json.load(f)
    merged_base_path = ckpt_config.get("merged_base_path")
    if merged_base_path is None:
        merged_base_path = os.path.join(os.path.dirname(checkpoint_path), "merged_base_unet")

    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(checkpoint_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(checkpoint_path, "text_encoder"))
    text_encoder.to(device).eval()

    vae = AutoencoderKL.from_pretrained(Config.pretrained_model, subfolder="vae", local_files_only=True)
    vae.to(device).eval()

    # Base UNet (merged shared already in weights)
    base_unet = UNet2DConditionModel.from_pretrained(merged_base_path)

    # Optional shared adapter (if present)
    shared_path = os.path.join(checkpoint_path, "unet_lora", "shared")
    if os.path.exists(shared_path):
        print(f"Loading shared LoRA from: {shared_path}")
        base_unet = PeftModel.from_pretrained(base_unet, shared_path, adapter_name="shared")
        if hasattr(base_unet, "set_adapter"):
            base_unet.set_adapter("shared")

    # L4-only UNet (separate instance to avoid adapter bleed)
    l4_unet = None
    l4_path = os.path.join(checkpoint_path, "unet_lora", "l4")
    if os.path.exists(l4_path):
        print(f"Loading L4 LoRA from: {l4_path}")
        l4_base = UNet2DConditionModel.from_pretrained(merged_base_path)
        l4_unet = PeftModel.from_pretrained(l4_base, l4_path, adapter_name="l4")

    base_unet.to(device).eval()
    if l4_unet is not None:
        l4_unet.to(device).eval()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from stage2_train import SeverityEmbedding
    time_embed_dim = unet.base_model.model.config.block_out_channels[0] * 4
    severity_embedding = SeverityEmbedding(Config.num_severity_classes, time_embed_dim)
    sev_path = os.path.join(checkpoint_path, "severity_embedding.pt")
    if os.path.exists(sev_path):
        severity_embedding.load_state_dict(torch.load(sev_path, map_location=device))
    severity_embedding.to(device).eval()

    scheduler = DDPMScheduler.from_pretrained(Config.pretrained_model, subfolder="scheduler", local_files_only=True)
    return tokenizer, text_encoder, vae, base_unet, l4_unet, severity_embedding, scheduler


def unet_forward_with_severity(unet, sample, timestep, encoder_hidden_states, severity_embedding, severity_ids):
    base = unet.base_model.model if hasattr(unet, 'base_model') else unet
    t_emb = base.time_proj(timestep).to(dtype=sample.dtype)
    emb = base.time_embedding(t_emb)
    if severity_embedding is not None and severity_ids is not None:
        emb = emb + severity_embedding(severity_ids).to(dtype=emb.dtype)

    sample_in = base.conv_in(sample)
    down_res = (sample_in,)
    for block in base.down_blocks:
        if hasattr(block, "has_cross_attention") and block.has_cross_attention:
            sample_in, res = block(sample_in, emb, encoder_hidden_states=encoder_hidden_states)
        else:
            sample_in, res = block(sample_in, emb)
        down_res += res
    sample_in = base.mid_block(sample_in, emb, encoder_hidden_states=encoder_hidden_states)
    for block in base.up_blocks:
        res = down_res[-len(block.resnets):]
        down_res = down_res[:-len(block.resnets)]
        if hasattr(block, "has_cross_attention") and block.has_cross_attention:
            sample_in = block(sample_in, res, emb, encoder_hidden_states=encoder_hidden_states)
        else:
            sample_in = block(sample_in, res, emb)
    sample_in = base.conv_norm_out(sample_in)
    sample_in = base.conv_act(sample_in)
    return base.conv_out(sample_in)


def dilate_mask(mask_tensor, dilate_pixels: int = 0):
    if dilate_pixels <= 0:
        return mask_tensor
    k = dilate_pixels * 2 + 1
    return F.max_pool2d(mask_tensor, kernel_size=k, stride=1, padding=k // 2)


def create_feather_mask(mask_tensor, feather_pixels: int = 25):
    if feather_pixels <= 0:
        return mask_tensor
    k = feather_pixels * 2 + 1
    sigma = feather_pixels / 3.0
    x = torch.arange(k, device=mask_tensor.device, dtype=mask_tensor.dtype) - k // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.view(1, 1, -1, 1) * gauss_1d.view(1, 1, 1, -1)
    padded = F.pad(mask_tensor, (k//2, k//2, k//2, k//2), mode='replicate')
    soft_mask = F.conv2d(padded, gauss_2d.view(1, 1, k, k))
    return torch.max(soft_mask, mask_tensor).clamp(0, 1)


def find_building_instances(mask_np):
    """找到需要转换的建筑实例（L1, L2, L4）"""
    instances = []
    
    for level in [1, 2, 4]:  # L1→L3, L2→L2, L4→L4
        if level not in Config.transitions:
            continue
        
        level_mask = (mask_np == level).astype(np.uint8)
        labeled, num_features = ndimage.label(level_mask)
        
        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) == 0:
                continue
            
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            area = len(ys)
            
            if area < 500:
                continue
            
            instances.append({
                'bbox': (y1, x1, y2, x2),
                'area': area,
                'mask': (labeled == i).astype(np.float32),
                'source_level': level,
                'target_level': Config.transitions[level],
            })
    
    return instances


@torch.no_grad()
def inpaint_512(image_tensor, mask_tensor, target_level, tokenizer, text_encoder, vae, base_unet, l4_unet, severity_embedding, scheduler, device):
    """对 512x512 图像进行 inpainting"""
    # L4 用专用 UNet，其余等级只用 base UNet（避免 l4 adapter 串用）
    unet = l4_unet if target_level == 4 and l4_unet is not None else base_unet
    
    prompt = PROMPTS[target_level]
    
    mask_dilated = dilate_mask(mask_tensor, Config.dilate)
    masked_image = image_tensor * (1 - mask_dilated)

    image_latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
    masked_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor
    mask_latent = F.interpolate(mask_dilated, size=image_latents.shape[-2:], mode='nearest')

    text_inputs = tokenizer(prompt["positive"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_emb = text_encoder(text_inputs.input_ids.to(device))[0]
    neg_inputs = tokenizer(prompt["negative"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_emb = text_encoder(neg_inputs.input_ids.to(device))[0]

    severity_id = torch.tensor([Config.level_to_severity[target_level]], device=device)
    uncond_severity_id = torch.tensor([Config.severity_uncond_id], device=device)

    scheduler.set_timesteps(Config.steps, device=device)
    init_timestep = int(Config.steps * Config.strength)
    t_start = max(Config.steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    noise = torch.randn_like(image_latents)
    latents = scheduler.add_noise(image_latents, noise, timesteps[:1])

    for t in timesteps:
        unet_input = torch.cat([latents, mask_latent, masked_latents], dim=1)
        noise_cond = unet_forward_with_severity(unet, unet_input, t.unsqueeze(0), text_emb, severity_embedding, severity_id)
        noise_uncond = unet_forward_with_severity(unet, unet_input, t.unsqueeze(0), uncond_emb, severity_embedding, uncond_severity_id)
        noise_pred = noise_uncond + Config.cfg * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    generated = vae.decode(latents / vae.config.scaling_factor).sample
    soft_mask = create_feather_mask(mask_dilated, Config.feather)
    result = generated * soft_mask + image_tensor * (1 - soft_mask)
    return (result / 2 + 0.5).clamp(0, 1)


def inpaint_instance(image_np, instance, tokenizer, text_encoder, vae, base_unet, l4_unet, severity_embedding, scheduler, device):
    """对单个建筑实例进行 inpainting"""
    H, W = image_np.shape[:2]
    y1, x1, y2, x2 = instance['bbox']
    target_level = instance['target_level']
    
    pad = Config.padding
    y1_pad = max(0, y1 - pad)
    x1_pad = max(0, x1 - pad)
    y2_pad = min(H, y2 + pad)
    x2_pad = min(W, x2 + pad)
    
    crop_image = image_np[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    crop_mask = instance['mask'][y1_pad:y2_pad, x1_pad:x2_pad].copy()
    crop_h, crop_w = crop_image.shape[:2]
    
    resized_image = Image.fromarray(crop_image).resize((Config.infer_size, Config.infer_size), Image.LANCZOS)
    resized_mask = Image.fromarray((crop_mask * 255).astype(np.uint8)).resize((Config.infer_size, Config.infer_size), Image.NEAREST)
    
    image_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float() / 255.0
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0).to(device)
    
    mask_tensor = torch.from_numpy(np.array(resized_mask)).float() / 255.0
    mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0).unsqueeze(0).to(device)
    
    result_tensor = inpaint_512(
        image_tensor, mask_tensor, target_level,
        tokenizer, text_encoder, vae, base_unet, l4_unet, severity_embedding, scheduler, device
    )
    
    result_np = (result_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    result_resized = Image.fromarray(result_np).resize((crop_w, crop_h), Image.LANCZOS)
    
    return np.array(result_resized), (y1_pad, x1_pad, y2_pad, x2_pad)


def process_image(image_path, mask_path, tokenizer, text_encoder, vae, base_unet, l4_unet, severity_embedding, scheduler, device):
    """处理单张图"""
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    instances = find_building_instances(mask_np)
    
    if len(instances) == 0:
        return None, None, None
    
    result_np = image_np.copy()
    target_mask_np = np.zeros_like(mask_np)
    
    for instance in instances:
        crop_result, bbox_pad = inpaint_instance(
            image_np, instance, tokenizer, text_encoder, vae, base_unet, l4_unet,
            severity_embedding, scheduler, device
        )
        
        y1, x1, y2, x2 = bbox_pad
        instance_mask = instance['mask'][y1:y2, x1:x2]
        
        # 更新目标掩码
        target_mask_np[instance['mask'] > 0.5] = instance['target_level']
        
        # 羽化混合
        mask_tensor = torch.from_numpy(instance_mask).float().unsqueeze(0).unsqueeze(0)
        soft_mask = create_feather_mask(mask_tensor, Config.feather // 2)
        soft_mask_np = soft_mask[0, 0].numpy()
        
        for c in range(3):
            result_np[y1:y2, x1:x2, c] = (
                crop_result[:, :, c] * soft_mask_np + 
                result_np[y1:y2, x1:x2, c] * (1 - soft_mask_np)
            ).astype(np.uint8)
    
    # 创建彩色掩码
    H, W = mask_np.shape
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    for level, color in TARGET_COLORS.items():
        mask_vis[target_mask_np == level] = color
    
    return image_np, mask_vis, result_np


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    tokenizer, text_encoder, vae, base_unet, l4_unet, severity_embedding, scheduler = load_models(args.checkpoint, device)
    
    # 创建输出目录
    original_dir = os.path.join(args.output, "original")
    mask_dir = os.path.join(args.output, "mask")
    result_dir = os.path.join(args.output, "result")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 获取图像列表
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    np.random.shuffle(image_files)
    
    print(f"Transitions: L1→L3, L2→L2, L4→L4")
    print(f"L4 adapter loaded: {l4_unet is not None}")
    
    saved_count = 0
    img_idx = 0
    
    while saved_count < args.num_images and img_idx < len(image_files):
        image_path = image_files[img_idx]
        img_idx += 1
        
        image_id = Path(image_path).stem
        mask_path = os.path.join(args.mask_dir, f"{image_id}_mask.png")
        
        if not os.path.exists(mask_path):
            continue
        
        print(f"[{saved_count+1}/{args.num_images}] Processing {image_id}")
        
        try:
            original_np, mask_vis, result_np = process_image(
                image_path, mask_path, tokenizer, text_encoder, vae, base_unet, l4_unet,
                severity_embedding, scheduler, device
            )
            
            if original_np is None:
                print(f"  No valid buildings, skipping")
                continue
            
            saved_count += 1
            
            # 保存：编号从 1 开始
            Image.fromarray(original_np).save(os.path.join(original_dir, f"{saved_count}.jpg"), quality=95)
            Image.fromarray(mask_vis).save(os.path.join(mask_dir, f"{saved_count}_mask.png"))
            Image.fromarray(result_np).save(os.path.join(result_dir, f"{saved_count}.jpg"), quality=95)
            
            print(f"  Saved as {saved_count}.jpg")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Saved {saved_count} images to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="./inference_output/fullres_test")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--cfg", type=float, default=10)
    parser.add_argument("--dilate", type=int, default=25)
    parser.add_argument("--feather", type=int, default=25)
    parser.add_argument("--steps", type=int, default=50)
    
    args = parser.parse_args()
    
    Config.strength = args.strength
    Config.cfg = args.cfg
    Config.dilate = args.dilate
    Config.feather = args.feather
    Config.steps = args.steps
    
    run_inference(args)


if __name__ == "__main__":
    main()
