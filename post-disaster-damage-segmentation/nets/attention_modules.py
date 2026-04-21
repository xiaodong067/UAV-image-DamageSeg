"""
注意力模块集合 - 用于消融实验
包含：CBAM、ECA、ESAM（自设计）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """
    CBAM注意力模块 (Baseline)
    论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
    
    特点：
    - 通道注意力：使用MLP，有降维操作（reduction=16）
    - 空间注意力：7x7卷积
    - 串行结构：Channel -> Spatial
    
    缺点：
    - MLP降维可能丢失细微损伤特征
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        
        # 通道注意力（使用MLP，有降维）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        x = x * sa_weight
        
        return x


class ECANet(nn.Module):
    """
    ECA-Net通道注意力模块
    论文: ECA-Net: Efficient Channel Attention for Deep CNNs (CVPR 2020)
    
    特点：
    - 无降维操作，避免信息丢失
    - 使用1D卷积进行局部跨通道交互
    - 参数极少（~100个参数）
    
    优势：
    - 保留细微损伤特征
    - 计算效率高
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        
        # 自适应确定1D卷积核大小
        # 论文公式：k = |log2(C) + b| / gamma
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1  # 保证是奇数
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # 1D卷积：跨通道交互 [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid生成通道权重
        y = self.sigmoid(y)
        
        # 重标定
        return x * y.expand_as(x)


class ESAM(nn.Module):
    """
    ESAM: Efficient Spatial-Channel Attention Module
    自设计模块 - 针对建筑物损伤检测优化
    
    设计动机：
    1. 损伤特征细微（裂缝仅几个像素宽）
    2. 形状不规则（长条状、片状）
    3. 易受背景干扰（瓷砖缝隙、树影）
    
    核心创新：
    1. 第一阶段：高效通道注意力（ECA）
       - 用1D卷积替代MLP，避免降维导致的信息丢失
       - 自适应核大小，学习局部跨通道依赖
    
    2. 第二阶段：空间注意力（SA）
       - 大感受野（7x7卷积）捕获裂缝连通性
       - Avg+Max池化同时保留显著特征和背景信息
    
    结构：
        Input [B, C, H, W]
          ↓
        ECA (通道筛选) - 识别"哪些通道"重要
          ↓
        SA (空间定位) - 识别"哪些位置"重要
          ↓
        Output [B, C, H, W]
    
    优势对比：
    - vs CBAM：无降维，保留细微特征
    - vs ECA：增加空间定位能力
    - vs Coordinate Attention：更轻量，更适合损伤检测
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ESAM, self).__init__()
        
        # ========== 第一阶段：高效通道注意力 (ECA) ==========
        # 核心创新：无降维的跨通道交互
        
        # 自适应确定1D卷积核大小
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.channel_sigmoid = nn.Sigmoid()
        
        # ========== 第二阶段：空间注意力 (SA) ==========
        # 核心创新：大感受野捕获损伤连通性
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),  # 增加BN稳定训练
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # ========== 阶段1：通道注意力 (The "Heart") ==========
        # 问题：哪些特征通道对损伤检测重要？
        # 例如：混凝土纹理通道、边缘响应通道
        
        # 全局平均池化：压缩空间信息 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # 1D卷积：局部跨通道交互（无降维！）
        # 学习模式："如果通道i激活，邻居通道i±k也应该被激活"
        y = self.channel_conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        # 生成通道权重 [B, C, 1, 1]
        channel_weight = self.channel_sigmoid(y)
        
        # 通道重标定
        x = x * channel_weight.expand_as(x)
        
        # ========== 阶段2：空间注意力 (The "Eyes") ==========
        # 问题：在特征图的哪些位置应该关注？
        # 例如：裂缝位置、损伤边界
        
        # 统计特征：同时保留显著性和整体性
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 背景信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 显著特征
        
        # 拼接并生成空间权重
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_weight = self.spatial_conv(spatial_input)  # [B, 1, H, W]
        
        # 空间重标定
        out = x * spatial_weight
        
        return out


class IdentityAttention(nn.Module):
    """
    恒等注意力（消融实验基线）
    直接返回输入，用于测试"无注意力"的baseline
    """
    def __init__(self, channels, **kwargs):
        super(IdentityAttention, self).__init__()
        
    def forward(self, x):
        return x


# ========== 注意力模块工厂函数 ==========

def get_attention_module(attention_type, channels, **kwargs):
    """
    获取注意力模块的工厂函数
    
    参数：
        attention_type: 注意力类型
            - 'none': 无注意力（恒等映射）
            - 'cbam': CBAM注意力
            - 'eca': ECA-Net通道注意力
            - 'esam': ESAM（自设计）
        channels: 输入通道数
        **kwargs: 额外参数（如reduction, gamma, b等）
    
    返回：
        注意力模块实例
    
    使用示例：
        attention = get_attention_module('esam', channels=256)
        output = attention(input_features)
    """
    attention_dict = {
        'none': IdentityAttention,
        'cbam': CBAM,
        'eca': ECANet,
        'esam': ESAM,
    }
    
    if attention_type not in attention_dict:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available: {list(attention_dict.keys())}")
    
    return attention_dict[attention_type](channels, **kwargs)


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 80)
    print("注意力模块消融实验 - 参数量和性能对比")
    print("=" * 80)
    
    # 测试配置
    batch_size = 2
    channels = 256
    height, width = 32, 32
    
    # 模拟输入
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试所有注意力模块
    attention_types = ['none', 'cbam', 'eca', 'esam']
    
    for att_type in attention_types:
        print(f"\n{'='*60}")
        print(f"测试: {att_type.upper()}")
        print(f"{'='*60}")
        
        # 创建模块
        if att_type == 'cbam':
            attention = get_attention_module(att_type, channels, reduction=16)
        elif att_type in ['eca', 'esam']:
            attention = get_attention_module(att_type, channels, gamma=2, b=1)
        else:
            attention = get_attention_module(att_type, channels)
        
        # 前向传播
        with torch.no_grad():
            output = attention(x)
        
        # 统计参数
        total_params = sum(p.numel() for p in attention.parameters())
        trainable_params = sum(p.numel() for p in attention.parameters() if p.requires_grad)
        
        print(f"输入形状:  {x.shape}")
        print(f"输出形状:  {output.shape}")
        print(f"总参数量:  {total_params:,}")
        print(f"可训参数:  {trainable_params:,}")
        print(f"模型大小:  {total_params * 4 / 1024:.2f} KB")
        
        # 模块描述
        descriptions = {
            'none': '无注意力（Baseline）',
            'cbam': 'CBAM - 通道MLP(降维) + 空间注意力',
            'eca': 'ECA-Net - 高效通道注意力（无降维）',
            'esam': 'ESAM - ECA通道 + 空间注意力（串行）'
        }
        print(f"描述:      {descriptions[att_type]}")
    
    # 参数量对比总结
    print("\n" + "="*80)
    print("参数量对比总结")
    print("="*80)
    print(f"{'模块':<15} {'参数量':<15} {'相对CBAM':<15} {'特点'}")
    print("-"*80)
    
    # 实际计算（C=256, reduction=16, k=7）
    cbam_params = 256*16 + 16*256 + 2*1*7*7  # MLP + spatial conv
    eca_params = 1*1*7  # 1D conv (k=7 for C=256)
    esam_params = 1*1*7 + 2*1*7*7 + 1  # ECA + spatial conv + BN
    
    print(f"{'None':<15} {0:<15} {'0%':<15} {'无注意力'}")
    print(f"{'CBAM':<15} {cbam_params:<15,} {'100%':<15} {'有降维'}")
    print(f"{'ECA':<15} {eca_params:<15,} {f'{eca_params/cbam_params*100:.1f}%':<15} {'无降维，仅通道'}")
    print(f"{'ESAM (推荐)':<15} {esam_params:<15,} {f'{esam_params/cbam_params*100:.1f}%':<15} {'无降维，通道+空间'}")
    
    print("\n" + "="*80)
    print("消融实验建议")
    print("="*80)
    print("""
1. Baseline (无注意力):
   python train.py --attention none
   
2. CBAM (传统方法):
   python train.py --attention cbam
   
3. ECA-Net (仅通道):
   python train.py --attention eca
   
4. ESAM (推荐方案):
   python train.py --attention esam

预期效果：
- None < CBAM ≈ ECA < ESAM
- ESAM 应该在 slight/moderate damage 混淆问题上表现最好
""")

