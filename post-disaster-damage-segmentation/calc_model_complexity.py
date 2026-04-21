#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型复杂度计算脚本
计算参数量 (Parameters)、计算量 (FLOPs)、模型大小
用于论文对比表格
"""

import os
import torch

# 尝试导入 thop（计算 FLOPs）
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ thop 未安装，无法计算 FLOPs")
    print("   安装命令: pip install thop")


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calc_flops(model, input_shape=(1, 3, 256, 256)):
    """计算 FLOPs"""
    if not HAS_THOP:
        return 0
    
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops


def get_file_size(file_path):
    """获取文件大小 (MB)"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0


def analyze_model(model, model_name, input_shape=(256, 256)):
    """分析单个模型"""
    model.eval()
    model = model.to('cpu')
    
    # 参数量
    total_params, _ = count_parameters(model)
    
    # FLOPs
    flops = calc_flops(model, input_shape=(1, 3, input_shape[0], input_shape[1]))
    
    return {
        'name': model_name,
        'params': total_params,
        'params_m': total_params / 1e6,
        'flops': flops,
        'flops_g': flops / 1e9 if flops > 0 else 0,
    }


def main():
    print("=" * 70)
    print("   模型复杂度分析工具 (参数量 / FLOPs / 模型大小)")
    print("=" * 70)
    
    input_shape = (256, 256)
    num_classes = 5
    results = []
    
    # ==================== 模型路径配置 ====================
    MODEL_DIR = "model"
    ONNX_DIR = "model_onnx"
    TRT_DIR = "model_trt"
    
    # 模型配置: (pth文件名, onnx文件名, trt文件名, 模型类型, 显示名称)
    models_config = [
        ("mobilenet.pth", "mobilenet.onnx", "mobilenet_fp16.trt",
         "original_mobilenet", "DeepLabV3+ (MobileNetV2)"),
        ("xception.pth", "xception.onnx", "xception_fp16.trt",
         "original_xception", "DeepLabV3+ (Xception)"),
        ("mobilenetV2_t1w.pth", "mobilenetV2_t1w.onnx", "mobilenetV2_t1w_fp16.trt",
         "enhanced_building_damage", "Enhanced Building Damage"),
    ]
    # =====================================================

    for pth_name, onnx_name, trt_name, model_type, display_name in models_config:
        pth_path = os.path.join(MODEL_DIR, pth_name)
        onnx_path = os.path.join(ONNX_DIR, onnx_name)
        trt_path = os.path.join(TRT_DIR, trt_name)
        
        if not os.path.exists(pth_path):
            print(f"⚠️ 跳过 {pth_name}，文件不存在")
            continue
        
        print(f"\n🔄 分析: {display_name}")
        
        # 加载模型
        try:
            if model_type == "original_mobilenet":
                from nets.deeplabv3_plus import DeepLab
                model = DeepLab(num_classes=num_classes, backbone="mobilenet", 
                               downsample_factor=8, pretrained=False)
            elif model_type == "original_xception":
                from nets.deeplabv3_plus import DeepLab
                model = DeepLab(num_classes=num_classes, backbone="xception", 
                               downsample_factor=8, pretrained=False)
            elif model_type == "building_damage":
                from nets.building_damage_deeplab import create_building_damage_deeplab
                model = create_building_damage_deeplab(num_classes=num_classes, pretrained=False)
            elif model_type == "enhanced_building_damage":
                from nets.enhanced_building_damage import create_enhanced_building_damage_deeplab
                model = create_enhanced_building_damage_deeplab(num_classes=num_classes, pretrained=False)
            else:
                print(f"❌ 未知模型类型: {model_type}")
                continue
            
            # 计算参数量和 FLOPs
            result = analyze_model(model, display_name, input_shape)
            
            # 获取文件大小
            result['pth_size'] = get_file_size(pth_path)
            result['onnx_size'] = get_file_size(onnx_path)
            result['trt_size'] = get_file_size(trt_path)
            
            results.append(result)
            
            print(f"   参数量: {result['params_m']:.2f}M")
            print(f"   FLOPs: {result['flops_g']:.2f}G")
            print(f"   PTH: {result['pth_size']:.2f}MB | ONNX: {result['onnx_size']:.2f}MB | TRT: {result['trt_size']:.2f}MB")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    # 汇总表格
    if results:
        print("\n")
        print("=" * 90)
        print("📋 模型复杂度对比表格")
        print("=" * 90)
        print(f"{'Model':<30} {'Params(M)':<12} {'FLOPs(G)':<12} {'PTH(MB)':<12} {'ONNX(MB)':<12} {'TRT(MB)':<12}")
        print("-" * 90)
        for r in results:
            print(f"{r['name']:<30} {r['params_m']:<12.2f} {r['flops_g']:<12.2f} {r['pth_size']:<12.2f} {r['onnx_size']:<12.2f} {r['trt_size']:<12.2f}")
        print("=" * 90)
        
        # 保存到文件
        with open('model_complexity.txt', 'w', encoding='utf-8') as f:
            f.write("模型复杂度对比\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'Model':<30} {'Params(M)':<12} {'FLOPs(G)':<12} {'PTH(MB)':<12} {'ONNX(MB)':<12} {'TRT(MB)':<12}\n")
            f.write("-" * 90 + "\n")
            for r in results:
                f.write(f"{r['name']:<30} {r['params_m']:<12.2f} {r['flops_g']:<12.2f} {r['pth_size']:<12.2f} {r['onnx_size']:<12.2f} {r['trt_size']:<12.2f}\n")
            f.write("=" * 90 + "\n")
            f.write(f"\n输入尺寸: {input_shape[0]}x{input_shape[1]}\n")
            f.write(f"类别数: {num_classes}\n")
        
        print("\n💾 结果已保存到 model_complexity.txt")


if __name__ == "__main__":
    main()
