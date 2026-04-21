import os
import csv
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免GUI问题
import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
Server environment specific building damage degree semantic segmentation task mIoU evaluation

Features:
1. Uses non-interactive matplotlib backend to avoid Qt errors
2. All charts are saved only, not displayed
3. Suitable for running in GUI-less server environments
4. Focuses on building damage category evaluation, excluding background class
'''

if __name__ == "__main__":
    print("=== Server Environment Semantic Segmentation Evaluation ===")
    
    #---------------------------------------------------------------------------#
    #   miou_mode specifies what this file will calculate
    #   miou_mode=0: complete mIoU calculation process, including prediction and mIoU calculation
    #   miou_mode=1: only get prediction results
    #   miou_mode=2: only calculate mIoU
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    
    #------------------------------#
    #   Building damage segmentation task configuration
    #   Please modify according to your training configuration
    #------------------------------#
    num_classes     = 5  # background + 4 damage levels
    name_classes    = ["background", "undamaged_building", "slight_damage", "moderate_damage", "severe_damage"]
    
    #-------------------------------------------------------#
    #   Exclude background class, focus on damage category evaluation
    #-------------------------------------------------------#
    include_background = False

    #===========================================================================#
    #   模型评估配置（核心！只需修改这里，不用改 deeplab.py）
    #   
    #   model_type: 模型类型
    #       - "original"                : 原版 DeepLabV3+
    #       - "enhanced_building_damage": 增强版建筑损伤模型
    #
    #   backbone: 主干网络（仅 original 模型使用，增强版模型忽略此项）
    #       - "mobilenet" : MobileNetV2 主干
    #       - "xception"  : Xception 主干
    #
    #   attention_type: 注意力模块类型（仅 enhanced_building_damage 有效）
    #       - 'none': 无注意力（Baseline）
    #       - 'cbam': CBAM注意力
    #       - 'eca':  ECA-Net通道注意力
    #       - 'esam': ESAM（推荐）
    #
    #   downsample_factor: 下采样倍数，与训练时保持一致（8 或 16）
    #   input_shape: 输入图片尺寸，与训练时保持一致
    #   model_path: 对应训练出的权重文件路径
    #
    #   示例：
    #   评估原版mobilenet:  model_type="original", backbone="mobilenet", model_path="logs/原版mob/best_epoch_weights.pth"
    #   评估原版xception:   model_type="original", backbone="xception",  model_path="logs/原版xce/best_epoch_weights.pth"
    #   评估增强 ESAM:      model_type="enhanced_building_damage", attention_type="esam", model_path="logs/esam实验/best_epoch_weights.pth"
    #   评估增强 CBAM:      model_type="enhanced_building_damage", attention_type="cbam", model_path="logs/cbam实验/best_epoch_weights.pth"
    #   评估增强 ECA:       model_type="enhanced_building_damage", attention_type="eca",  model_path="logs/eca实验/best_epoch_weights.pth"
    #   评估增强 None:      model_type="enhanced_building_damage", attention_type="none", model_path="logs/none实验/best_epoch_weights.pth"
    #===========================================================================#
    model_type          = "enhanced_building_damage"
    backbone            = "mobilenet"           # 仅 original 模型生效，增强版忽略
    attention_type      = "esam"                # 仅 enhanced_building_damage 生效
    downsample_factor   = 8                     # 与训练时一致
    input_shape         = [512, 512]            # 与训练时一致
    model_path          = '/home/luminous/Hxd/Disaster/Code/Disaster_seg/deeplab/MAG_deeplab_test1/logs/loss_2026_01_18_12_36_24/best_epoch_weights.pth'

    #-------------------------------------------------------#
    #   Dataset path configuration
    #   Please modify according to your actual dataset path
    #-------------------------------------------------------#
    VOCdevkit_path  = r'/home/luminous/Hxd/Disaster/Code/Disaster_seg/deeplab/MAG_deeplab_test1/VOCdevkit'

    # If your dataset structure is different, please modify the following paths:
    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    #-------------------------------------------------------#
    #   输出目录自动按模型类型区分，避免不同实验结果互相覆盖
    #   原版mobilenet:  miou_out_original_mobilenet/
    #   原版xception:   miou_out_original_xception/
    #   增强版模型:     miou_out_esam/ / miou_out_cbam/ / ...
    #-------------------------------------------------------#
    if model_type == "original":
        miou_out_path = f"miou_out_original_{backbone}"
    else:
        miou_out_path = f"miou_out_{attention_type}"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')
    
    print(f"{'='*60}")
    print(f"  评估配置")
    print(f"{'='*60}")
    print(f"  模型类型:       {model_type}")
    if model_type == "original":
        print(f"  主干网络:       {backbone}")
    if model_type == "enhanced_building_damage":
        print(f"  注意力类型:     {attention_type.upper()}")
    print(f"  下采样倍数:     {downsample_factor}")
    print(f"  输入尺寸:       {input_shape}")
    print(f"  类别数量:       {num_classes}")
    print(f"  类别名称:       {name_classes}")
    print(f"  模型权重:       {model_path}")
    print(f"  数据集路径:     {VOCdevkit_path}")
    print(f"  输出目录:       {miou_out_path}")
    print(f"{'='*60}")

    #-------------------------------------------------------#
    #   可视化颜色映射（类别索引 -> RGB颜色）
    #   0-background: 黑色, 1-undamaged: 绿色, 
    #   2-slight: 黄色, 3-moderate: 蓝色, 4-severe: 红色
    #-------------------------------------------------------#
    color_map = np.array([
        [0,   0,   0],     # 0 - background  黑色
        [0,   255, 0],     # 1 - undamaged    绿色
        [255, 255, 0],     # 2 - slight       黄色
        [0,   0,   255],   # 3 - moderate     蓝色
        [255, 0,   0],     # 4 - severe       红色
    ], dtype=np.uint8)

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        vis_dir = os.path.join(miou_out_path, 'visualization')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        print("\n=== Loading Model ===")
        try:
            deeplab = DeeplabV3(
                model_type          = model_type,
                backbone            = backbone,
                attention_type      = attention_type,
                downsample_factor   = downsample_factor,
                input_shape         = input_shape,
                model_path          = model_path,
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Please check model_path and attention_type configuration")
            exit(1)

        #-------------------------------------------------------#
        #   逐图预测 + 可视化 + 逐图逐类别精度统计
        #-------------------------------------------------------#
        per_image_stats = []

        print("\n=== Generating Prediction Results ===")
        for image_id in tqdm(image_ids, desc="Processing images"):
            try:
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                pred_img    = deeplab.get_miou_png(image)
                
                # 1) 保存类别索引图（用于 mIoU 计算）
                pred_img.save(os.path.join(pred_dir, image_id + ".png"))
                
                # 2) 保存可视化彩色图（分割颜色叠加到原图上，方便对比）
                pred_np     = np.array(pred_img, dtype=np.uint8)
                vis_rgb     = color_map[pred_np]                         # 纯分割着色图
                seg_overlay = Image.fromarray(vis_rgb)
                orig_rgb    = Image.open(image_path).convert('RGB')      # 重新读取原图（RGB）
                # 确保尺寸一致
                if orig_rgb.size != seg_overlay.size:
                    seg_overlay = seg_overlay.resize(orig_rgb.size, Image.NEAREST)
                # 混合：原图 30% + 分割着色 70%，与 detect_image 效果一致
                blended = Image.blend(orig_rgb, seg_overlay, alpha=0.7)
                blended.save(os.path.join(vis_dir, image_id + ".png"))
                
                # 3) 逐图逐类别精度统计
                gt_path = os.path.join(gt_dir, image_id + ".png")
                if os.path.exists(gt_path):
                    gt_np = np.array(Image.open(gt_path), dtype=np.uint8)
                    # 如果尺寸不一致，resize 预测结果到标签尺寸
                    if pred_np.shape != gt_np.shape:
                        pred_resized = np.array(
                            Image.fromarray(pred_np).resize(
                                (gt_np.shape[1], gt_np.shape[0]), Image.NEAREST
                            ), dtype=np.uint8
                        )
                    else:
                        pred_resized = pred_np
                    
                    row = {"image_id": image_id}
                    for cls_idx in range(1, num_classes):  # 跳过背景
                        gt_mask   = (gt_np == cls_idx)
                        total     = gt_mask.sum()
                        if total > 0:
                            correct = ((pred_resized == cls_idx) & gt_mask).sum()
                            acc     = round(correct / total * 100, 2)
                        else:
                            acc     = -1  # 该图中没有此类别，用 -1 标记
                        row[name_classes[cls_idx]] = acc
                    per_image_stats.append(row)
                    
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
        
        #-------------------------------------------------------#
        #   保存逐图逐类别精度 CSV
        #-------------------------------------------------------#
        csv_path = os.path.join(miou_out_path, "per_image_class_accuracy.csv")
        csv_columns = ["image_id"] + name_classes[1:]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(per_image_stats)
        print(f"Per-image class accuracy saved to {csv_path}")
        print("Prediction results generation completed!")

    if miou_mode == 0 or miou_mode == 2:
        print("\n=== Computing Evaluation Metrics ===")
        try:
            print("Calling compute_mIoU...")
            result = compute_mIoU(
                gt_dir, pred_dir, image_ids, num_classes, name_classes, 
                include_background=include_background
            )
            print(f"compute_mIoU returned {len(result)} values")
            hist, IoUs, PA, Recall, Precision, F1_Scores = result
            print("Evaluation metrics computation completed!")
            
            print("\n=== Generating Visualization Results ===")
            show_results(
                miou_out_path, hist, IoUs, PA, Recall, Precision, name_classes, 
                include_background=include_background, F1_Scores=F1_Scores
            )
            
            print(f"\n=== Result Files Description ===")
            print(f"All results saved to: {miou_out_path}/")
            print(f"  result files:")
            print(f"  1. detection-results/          : prediction class-index images (for mIoU computation)")
            print(f"  2. visualization/              : colorized prediction images")
            print(f"     green=undamaged, yellow=slight, blue=moderate, red=severe")
            print(f"  3. per_image_class_accuracy.csv: per-image per-class recall")
            print(f"  4. mIoU_without_bg.png         : mIoU bar chart")
            print(f"  5. confusion_matrix_heatmap_without_bg.png  : confusion matrix")
            print(f"  6. confusion_matrix_normalized_without_bg.png: normalized confusion matrix")
            print(f"  7. confusion_matrix_without_bg.csv          : confusion matrix CSV")
            print(f"  8. mPA / Recall / Precision / F1 charts")
            
        except Exception as e:
            print(f"Error during evaluation process: {e}")
            print("Please check if dataset path and label files are correct")

    print("\n=== Evaluation Completed ===") 