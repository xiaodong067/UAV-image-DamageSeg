#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型转ONNX格式 - 256x256 版本
用于 Jetson Nano 部署优化
"""

import torch
import onnx
import os
from deeplab import DeeplabV3

def convert_model_to_onnx_256():
    """
    将训练好的PyTorch模型转换为256x256输入的ONNX格式
    """
    # 模型配置参数
    config = {
        "model_path": r"E:\Student\Code_reproduction\Dajournal\seg_test1\model\mobilenetV2_t1w.pth",
        "num_classes": 5,
        "backbone": "mobilenet",
        "model_type": "enhanced_building_damage",
        "input_shape": [256, 256],  # 改为 256x256
        "downsample_factor": 8,
        "cuda": True,
    }
    
    # 输出ONNX文件路径
    onnx_path = "mobilenetV2_t1w.onnx"
    
    print("🚀 开始转换PyTorch模型到ONNX格式 (256x256)...")
    print(f"📁 源模型路径: {config['model_path']}")
    print(f"📁 输出ONNX路径: {onnx_path}")
    print(f"📐 输入尺寸: {config['input_shape']}")
    
    if not os.path.exists(config["model_path"]):
        print(f"❌ 错误：找不到模型文件 {config['model_path']}")
        return False
    
    try:
        # 创建DeepLab模型实例
        deeplab = DeeplabV3(**config)
        
        # 转换为ONNX格式
        print("🔄 正在转换模型...")
        deeplab.convert_to_onnx(
            simplify=True,
            model_path=onnx_path
        )
        
        # 验证ONNX模型
        print("✅ 验证ONNX模型...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✅ 转换成功！")
        print(f"📄 ONNX模型文件: {onnx_path}")
        print(f"📊 文件大小: {file_size:.2f} MB")
        print(f"📐 输入形状: [1, 3, 256, 256]")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🏗️ DeepLabV3+ ONNX 转换 (256x256 优化版)")
    print("=" * 60)
    
    success = convert_model_to_onnx_256()
    
    if success:
        print("\n🎉 转换完成！")
        print("\n📋 下一步操作:")
        print("   1. 将 building_damage_deeplab_256.onnx 传到 Jetson Nano")
        print("   2. 在 Jetson 上转换 TensorRT:")
        print("      /usr/src/tensorrt/bin/trtexec --onnx=building_damage_deeplab_256.onnx --saveEngine=building_damage_256_fp16.trt --fp16 --workspace=1024")
        print("   3. 修改推理脚本的 input_shape 为 (256, 256)")
