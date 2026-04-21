"""
增强版建筑损伤检测模型
基于building_damage进行针对性改进，提升中等损伤类别的识别精度
保持轻量化设计，适合实际部署

支持多种注意力机制的消融实验：
- none: 无注意力（Baseline）
- cbam: CBAM注意力
- eca: ECA-Net通道注意力
- esam: ESAM（自设计，推荐）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.deeplabv3_plus import MobileNetV2
from nets.attention_modules import get_attention_module

# 注意力模块已移至 nets/attention_modules.py
# 通过 get_attention_module() 统一获取

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 轻量化基础组件"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DamageGradientModule(nn.Module):
    """
    损伤梯度感知模块 - 解决slight/moderate damage混淆问题
    通过梯度信息增强损伤边界和过渡区域的表征
    """
    def __init__(self, in_channels, out_channels=64):
        super(DamageGradientModule, self).__init__()
        
        # 梯度计算分支
        self.gradient_x = nn.Conv2d(in_channels, out_channels//2, 3, padding=1, bias=False)
        self.gradient_y = nn.Conv2d(in_channels, out_channels//2, 3, padding=1, bias=False)
        
        # 梯度融合
        self.gradient_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 损伤强度感知
        self.intensity_aware = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 计算梯度信息
        grad_x = self.gradient_x(x)
        grad_y = self.gradient_y(x)
        gradient_feat = self.gradient_fusion(torch.cat([grad_x, grad_y], dim=1))
        
        # 强度感知加权
        intensity_weight = self.intensity_aware(x)
        enhanced_feat = gradient_feat * intensity_weight
        
        return enhanced_feat

class MultiScaleDamageASPP(nn.Module):
    """
    多尺度损伤感知ASPP - 消融实验版本
    针对slight/moderate damage的尺度特征优化
    
    参数：
        dim_in: 输入通道数
        dim_out: 输出通道数
        attention_type: 注意力类型
            - 'none': 无注意力（Baseline）
            - 'cbam': CBAM注意力
            - 'eca': ECA-Net通道注意力
            - 'esam': ESAM（推荐）
    """
    def __init__(self, dim_in, dim_out=256, attention_type='esam'):
        super(MultiScaleDamageASPP, self).__init__()
        
        self.attention_type = attention_type
        
        # 原有6个分支保持不变
        self.pointwise_branch = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        # 针对性优化膨胀率，增强中等损伤检测
        self.slight_damage_branch = nn.Sequential(
            DepthwiseSeparableConv(dim_in, dim_out, kernel_size=3, padding=2, dilation=2),  # 调整为2
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        self.moderate_damage_branch = nn.Sequential(
            DepthwiseSeparableConv(dim_in, dim_out, kernel_size=3, padding=4, dilation=4),  # 调整为4  
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        self.severe_damage_branch = nn.Sequential(
            DepthwiseSeparableConv(dim_in, dim_out, kernel_size=3, padding=8, dilation=8),  # 调整为8
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        self.damage_boundary_branch = nn.Sequential(
            DepthwiseSeparableConv(dim_in, dim_out, kernel_size=3, padding=12, dilation=12), # 调整为12
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        self.global_context_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        # 新增：损伤梯度感知分支
        self.damage_gradient_branch = DamageGradientModule(dim_in, dim_out)
        
        # 7分支融合 (6原有 + 1梯度)
        self.damage_aware_fusion = nn.Sequential(
            nn.Conv2d(dim_out * 7, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 注意力模块（通过工厂函数创建）
        if attention_type == 'cbam':
            self.attention = get_attention_module('cbam', dim_out, reduction=16)
        elif attention_type in ['eca', 'esam']:
            self.attention = get_attention_module(attention_type, dim_out, gamma=2, b=1)
        else:  # 'none'
            self.attention = get_attention_module('none', dim_out)
        
    def forward(self, x):
        # 6个原有分支
        pointwise_out = self.pointwise_branch(x)
        slight_out = self.slight_damage_branch(x)
        moderate_out = self.moderate_damage_branch(x)
        severe_out = self.severe_damage_branch(x)
        boundary_out = self.damage_boundary_branch(x)
        
        # 全局上下文 + 上采样
        global_out = self.global_context_branch(x)
        global_out = F.interpolate(global_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # 新增：梯度感知分支
        gradient_out = self.damage_gradient_branch(x)
        
        # 7分支融合
        concat_features = torch.cat([
            pointwise_out, slight_out, moderate_out, severe_out, 
            boundary_out, global_out, gradient_out
        ], dim=1)
        
        fused_features = self.damage_aware_fusion(concat_features)
        
        # 应用注意力机制
        final_features = self.attention(fused_features)
        
        return final_features


class EnhancedDamageDecoder(nn.Module):
    """
    增强损伤解码器 - 优化特征融合和上采样
    """
    def __init__(self, low_level_channels, high_level_channels, num_classes=5, decoder_channels=128):
        super(EnhancedDamageDecoder, self).__init__()
        
        # 低级特征处理 - 保留细节信息
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 多级特征融合
        self.feature_fusion = nn.Sequential(
            DepthwiseSeparableConv(high_level_channels + 48, decoder_channels, 3, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(decoder_channels, decoder_channels, 3, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # 损伤类别专用分支
        self.damage_specific_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels, 1, 1) for _ in range(num_classes)
        ])
        
        # 最终分类器
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels//2, num_classes, 1)
        )
        
    def forward(self, high_level_feat, low_level_feat):
        # 上采样高级特征
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.shape[2:], 
                                      mode='bilinear', align_corners=True)
        
        # 处理低级特征
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # 特征融合
        concat_feat = torch.cat([high_level_feat, low_level_feat], dim=1)
        fused_feat = self.feature_fusion(concat_feat)
        
        # 主分类输出
        main_output = self.final_conv(fused_feat)
        
        # 损伤类别专用输出 (暂时不使用，避免复杂性)
        # damage_outputs = []
        # for head in self.damage_specific_heads:
        #     damage_outputs.append(head(fused_feat))
        
        return main_output

class EnhancedBuildingDamageDeepLab(nn.Module):
    """
    增强版建筑损伤DeepLab模型
    
    参数：
        num_classes: 类别数量（默认5）
        pretrained: 是否使用预训练权重
        attention_type: 注意力类型
            - 'none': 无注意力（Baseline）
            - 'cbam': CBAM注意力
            - 'eca': ECA-Net通道注意力
            - 'esam': ESAM（推荐）
    """
    def __init__(self, num_classes=5, pretrained=False, attention_type='esam'):
        super(EnhancedBuildingDamageDeepLab, self).__init__()
        
        self.attention_type = attention_type
        
        # MobileNetV2骨干网络
        self.backbone = MobileNetV2(downsample_factor=8, pretrained=pretrained)
        
        # 增强多尺度损伤ASPP
        self.aspp = MultiScaleDamageASPP(
            dim_in=320, 
            dim_out=256, 
            attention_type=attention_type
        )
        
        # 增强损伤解码器
        self.decoder = EnhancedDamageDecoder(
            low_level_channels=24,  # MobileNetV2的低级特征通道
            high_level_channels=256,
            num_classes=num_classes,
            decoder_channels=128
        )
        
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # 特征提取
        low_level_features, high_level_features = self.backbone(x)
        
        # ASPP处理
        aspp_features = self.aspp(high_level_features)
        
        # 解码
        main_output = self.decoder(aspp_features, low_level_features)
        
        # 上采样到原始尺寸
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=True)
        
        return main_output

def create_enhanced_building_damage_deeplab(num_classes=5, pretrained=False, attention_type='esam'):
    """
    创建增强版建筑损伤DeepLab模型
    
    参数：
        num_classes: 类别数量（默认5）
        pretrained: 是否使用预训练权重
        attention_type: 注意力类型
            - 'none': 无注意力（Baseline）
            - 'cbam': CBAM注意力
            - 'eca': ECA-Net通道注意力
            - 'esam': ESAM（推荐，默认）
    
    消融实验示例：
        # 1. Baseline（无注意力）
        model_baseline = create_enhanced_building_damage_deeplab(attention_type='none')
        
        # 2. CBAM（传统方法）
        model_cbam = create_enhanced_building_damage_deeplab(attention_type='cbam')
        
        # 3. ECA-Net（仅通道注意力）
        model_eca = create_enhanced_building_damage_deeplab(attention_type='eca')
        
        # 4. ESAM（推荐方案）
        model_esam = create_enhanced_building_damage_deeplab(attention_type='esam')
    """
    return EnhancedBuildingDamageDeepLab(
        num_classes=num_classes, 
        pretrained=pretrained,
        attention_type=attention_type
    )

if __name__ == "__main__":
    print("=" * 80)
    print("测试增强版建筑损伤DeepLab模型 - 消融实验")
    print("=" * 80)
    
    # 测试所有注意力配置
    attention_types = ['none', 'cbam', 'eca', 'esam']
    
    # 模拟输入
    x = torch.randn(1, 3, 512, 512)
    
    print(f"\n输入形状: {x.shape}\n")
    
    for att_type in attention_types:
        print(f"{'='*60}")
        print(f"测试配置: {att_type.upper()}")
        print(f"{'='*60}")
        
        # 创建模型
        model = create_enhanced_building_damage_deeplab(
            num_classes=5, 
            pretrained=False, 
            attention_type=att_type
        )
        
        # 前向传播
        with torch.no_grad():
            output = model(x)
        
        print(f"输出形状: {output.shape}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        aspp_params = sum(p.numel() for p in model.aspp.parameters())
        attention_params = sum(p.numel() for p in model.aspp.attention.parameters())
        
        print(f"总参数量:      {total_params:,}")
        print(f"ASPP参数:      {aspp_params:,} ({aspp_params/total_params*100:.1f}%)")
        print(f"注意力参数:    {attention_params:,} ({attention_params/total_params*100:.2f}%)")
        print(f"模型大小:      {total_params * 4 / (1024**2):.2f} MB")
        
        # 注意力描述
        descriptions = {
            'none': '无注意力（Baseline）',
            'cbam': 'CBAM - 通道MLP(降维) + 空间注意力',
            'eca': 'ECA-Net - 高效通道注意力（无降维）',
            'esam': 'ESAM - ECA通道 + 空间注意力（推荐）'
        }
        print(f"描述:          {descriptions[att_type]}")
        print()
    
    print("="*80)
    print("测试完成！")
    print("="*80)
    print("\n消融实验建议：")
    print("1. Baseline（无注意力）: --attention none")
    print("2. CBAM（传统方法）:     --attention cbam")
    print("3. ECA-Net（仅通道）:    --attention eca")
    print("4. ESAM（推荐方案）:     --attention esam")
    print("\n训练命令示例：")
    print("CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train.py --attention esam")
