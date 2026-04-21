"""
增强损失函数 - 针对建筑损伤检测优化
解决slight_damage和moderate_damage混淆问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡，专注难分类样本
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            # 建筑损伤任务的类别权重
            self.alpha = torch.tensor([0.5, 1.5, 3.0, 3.0, 2.0])  # background, undamaged, slight, moderate, severe
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        device = inputs.device
        self.alpha = self.alpha.to(device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 获取alpha权重
        alpha_t = self.alpha[targets]
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DamageBoundaryLoss(nn.Module):
    """
    损伤边界损失 - 增强不同损伤类别之间的边界清晰度
    """
    def __init__(self, num_classes=5):
        super(DamageBoundaryLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
        # 计算梯度
        pred_grad_x = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
        pred_grad_y = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])
        
        target_grad_x = torch.abs(targets[:, :, 1:].float() - targets[:, :, :-1].float())
        target_grad_y = torch.abs(targets[:, 1:, :].float() - targets[:, :-1, :].float())
        
        # 边界loss
        boundary_loss_x = F.mse_loss(pred_grad_x.sum(1), target_grad_x)
        boundary_loss_y = F.mse_loss(pred_grad_y.sum(1), target_grad_y)
        
        return (boundary_loss_x + boundary_loss_y) / 2

class DamageConsistencyLoss(nn.Module):
    """
    损伤一致性损失 - 解决slight/moderate damage混淆
    通过区域一致性约束提高分类准确性
    """
    def __init__(self, kernel_size=5):
        super(DamageConsistencyLoss, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, padding=kernel_size//2)
        
    def forward(self, predictions, targets):
        batch_size, num_classes, height, width = predictions.shape
        
        # 获取预测的类别
        pred_classes = torch.argmax(predictions, dim=1, keepdim=True).float()
        targets_expanded = targets.unsqueeze(1).float()
        
        # 展开邻域
        pred_patches = self.unfold(pred_classes).view(batch_size, -1, height, width)
        target_patches = self.unfold(targets_expanded).view(batch_size, -1, height, width)
        
        # 计算邻域一致性
        pred_consistency = torch.std(pred_patches, dim=1)
        target_consistency = torch.std(target_patches, dim=1)
        
        # 一致性损失
        consistency_loss = F.mse_loss(pred_consistency, target_consistency)
        
        return consistency_loss

class EnhancedDamageLoss(nn.Module):
    """
    增强损伤损失函数 - 组合多种损失（简化版）
    """
    def __init__(self, num_classes=5, 
                 ce_weight=1.0, focal_weight=1.5, dice_weight=1.0):
        super(EnhancedDamageLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight 
        self.dice_weight = dice_weight
        
        # 损失函数组件（简化版，只使用最核心的）
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss(num_classes)
        
    def forward(self, predictions, targets):
        # 确保targets是正确的格式 (batch_size, height, width)
        # 处理可能的维度问题
        if len(targets.shape) == 4:
            # 如果是4D tensor，可能是one-hot编码，转换为索引
            if targets.shape[-1] > 1:
                targets = torch.argmax(targets, dim=-1)
            else:
                targets = targets.squeeze(-1)
        elif len(targets.shape) == 2:
            # 如果是2D tensor (batch_size, height)，可能是被错误压缩了
            # 尝试恢复width维度（假设是方形图像）
            batch_size, height = targets.shape
            if height == 512:  # 如果高度是512，假设width也是512
                # 这种情况说明数据有问题，我们不能随意添加维度
                # 暂时使用原始损失函数的方式
                # 对于2D targets，使用简单的CrossEntropyLoss
                ce_loss = nn.CrossEntropyLoss()
                # 需要reshape predictions为 (batch_size * height, num_classes)
                # 和targets为 (batch_size * height)
                batch_size, num_classes, pred_height, pred_width = predictions.shape
                predictions_flat = predictions.permute(0, 2, 3, 1).reshape(-1, num_classes)
                targets_flat = targets.reshape(-1).long()
                loss = ce_loss(predictions_flat, targets_flat)
                return loss, {'ce_loss': loss.item(), 'focal_loss': 0.0, 'dice_loss': 0.0}
        
        # 基础交叉熵损失
        ce_loss = self.ce_loss(predictions, targets.long())
        
        # Focal损失 - 处理困难样本
        focal_loss = self.focal_loss(predictions, targets)
        
        # Dice损失 - 处理类别不平衡
        dice_loss = self.dice_loss(predictions, targets)
        
        # 总损失（简化版）
        total_loss = (self.ce_weight * ce_loss + 
                     self.focal_weight * focal_loss +
                     self.dice_weight * dice_loss)
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'focal_loss': focal_loss.item(), 
            'dice_loss': dice_loss.item()
        }

class DiceLoss(nn.Module):
    """
    Dice Loss - 适合分割任务
    """
    def __init__(self, num_classes, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # 转换为one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        predictions_soft = F.softmax(predictions, dim=1)
        
        # 计算每个类别的Dice
        dice_scores = []
        for i in range(self.num_classes):
            pred_i = predictions_soft[:, i, :, :]
            target_i = targets_one_hot[:, i, :, :]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # 返回1 - dice作为损失
        return 1 - torch.stack(dice_scores).mean()

class AdaptiveLossScheduler:
    """
    自适应损失权重调度器
    根据训练进度动态调整损失权重
    """
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        
    def get_loss_weights(self, epoch):
        progress = epoch / self.total_epochs
        
        # 训练前期重点关注基础分类
        if progress < 0.3:
            return {
                'ce_weight': 2.0,
                'focal_weight': 1.0, 
                'dice_weight': 1.0,
                'boundary_weight': 0.2,
                'consistency_weight': 0.1
            }
        # 训练中期增强边界和一致性
        elif progress < 0.7:
            return {
                'ce_weight': 1.5,
                'focal_weight': 2.0,
                'dice_weight': 1.5, 
                'boundary_weight': 0.5,
                'consistency_weight': 0.3
            }
        # 训练后期精细调优
        else:
            return {
                'ce_weight': 1.0,
                'focal_weight': 2.5,
                'dice_weight': 2.0,
                'boundary_weight': 0.8,
                'consistency_weight': 0.5
            }
