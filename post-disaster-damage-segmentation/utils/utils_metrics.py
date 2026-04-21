import csv
import os
from os.path import join

import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免Qt错误
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import seaborn as sns


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    """
    计算每个类别的像素准确率 (Class-wise Pixel Accuracy)
    PA_i = TP_i / (TP_i + FN_i)
    即：该类别被正确分类的像素数 / 该类别在真实标注中的总像素数
    等价于 Recall（召回率）
    """
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Recall(hist):
    """
    计算每个类别的召回率 (Recall)
    Recall = TP / (TP + FN)
    """
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    """
    计算每个类别的精确率 (Precision)  
    Precision = TP / (TP + FP)
    """
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_class_F1(hist):
    """
    计算每个类别的F1-Score
    F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    precision = per_class_Precision(hist)
    recall = per_class_Recall(hist)
    # 避免除零错误，当precision + recall = 0时，F1 = 0
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    return f1

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, include_background=False):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，将预测结果resize到标签尺寸
        if len(label.flatten()) != len(pred.flatten()):  
            
            
            # 将预测结果resize到标签的尺寸
            from PIL import Image as PILImage
            pred_img = PILImage.fromarray(pred.astype(np.uint8))
            pred_img_resized = pred_img.resize((label.shape[1], label.shape[0]), PILImage.NEAREST)
            pred = np.array(pred_img_resized)

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            IoUs_temp = per_class_iu(hist)
            if include_background:
                mIoU_temp = np.nanmean(IoUs_temp)
            else:
                mIoU_temp = np.nanmean(IoUs_temp[1:])  # 排除背景类
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * mIoU_temp,
                    100 * np.nanmean(per_class_PA(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    #------------------------------------------------#
    #   根据include_background参数决定计算基础矩阵
    #------------------------------------------------#
    if include_background:
        # 使用完整矩阵计算
        compute_hist = hist
        classes_to_show = range(num_classes)
        suffix_text = "(包含背景)"
    else:
        # 排除背景类，使用4×4矩阵计算
        compute_hist = hist[1:, 1:]  # 去掉第0行第0列
        classes_to_show = range(1, num_classes)  # 对应原始类别索引
        suffix_text = "(不包含背景)"
    
    #------------------------------------------------#
    #   基于选定矩阵计算所有验证集图片的逐类别指标
    #------------------------------------------------#
    IoUs_compute        = per_class_iu(compute_hist)
    PA_compute          = per_class_PA(compute_hist)
    Recall_compute      = per_class_Recall(compute_hist)
    Precision_compute   = per_class_Precision(compute_hist)
    F1_Scores_compute   = per_class_F1(compute_hist)
    
    # 计算平均值
    mIoU = np.nanmean(IoUs_compute)
    mPA = np.nanmean(PA_compute)
    mF1 = np.nanmean(F1_Scores_compute)
    
    print('===> mIoU ' + suffix_text + ': ' + str(round(mIoU * 100, 2)) + '%')
    
    # 为了保持返回格式兼容，需要构造完整的数组
    if include_background:
        IoUs, PA, Recall, Precision, F1_Scores = IoUs_compute, PA_compute, Recall_compute, Precision_compute, F1_Scores_compute
    else:
        # 构造完整数组，但实际只有非背景类的值有意义
        IoUs = np.zeros(num_classes)
        PA = np.zeros(num_classes) 
        Recall = np.zeros(num_classes)
        Precision = np.zeros(num_classes)
        F1_Scores = np.zeros(num_classes)
        
        # 填入计算得到的非背景类数值
        IoUs[1:] = IoUs_compute
        PA[1:] = PA_compute
        Recall[1:] = Recall_compute
        Precision[1:] = Precision_compute
        F1_Scores[1:] = F1_Scores_compute
    
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    print('===> mF1: ' + str(round(mF1 * 100, 2)) + '%')
    
    if name_classes is not None:
        for ind_class in classes_to_show:
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; PA-' + str(round(PA[ind_class] * 100, 2)) + '; Recall-' + str(round(Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)) \
                + '; F1-Score-' + str(round(F1_Scores[ind_class] * 100, 2)))

    print('===> mPA: ' + str(round(mPA * 100, 2)) + '%; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)) + '%')
    return np.array(hist, int), IoUs, PA, Recall, Precision, F1_Scores

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    # if plt_show:
    #     plt.show()  # 注释掉避免在服务器环境中的Qt错误
    plt.close()

def draw_confusion_matrix(confusion_matrix, name_classes, output_path, include_background=False, normalize=False):
    """
    绘制混淆矩阵热力图
    """
    # 根据是否包含背景类来选择显示的类别
    if include_background:
        classes_to_show = name_classes
        cm = confusion_matrix
    else:
        classes_to_show = name_classes[1:]  # 排除背景类
        cm = confusion_matrix[1:, 1:]  # 排除第0行第0列
    
    # 是否归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # 将NaN替换为0
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
        cmap = 'Blues'
    else:
        title = 'Confusion Matrix (Pixel Count)'
        fmt = 'd'
        cmap = 'viridis'
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, 
                annot=True,           # 显示数值
                fmt=fmt,              # 数值格式
                cmap=cmap,            # 颜色映射
                xticklabels=classes_to_show,
                yticklabels=classes_to_show,
                cbar_kws={'shrink': 0.8})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()  # 注释掉避免在服务器环境中的Qt错误
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA, Recall, Precision, name_classes, include_background=False, tick_font_size = 12, F1_Scores=None):
    # 根据是否包含背景类来选择显示的数据
    if include_background:
        classes_to_show = name_classes
        IoUs_to_show = IoUs
        PA_to_show = PA
        Recall_to_show = Recall
        Precision_to_show = Precision
        F1_to_show = F1_Scores if F1_Scores is not None else None
        suffix = "_with_bg"
    else:
        classes_to_show = name_classes[1:]  # 排除背景类
        IoUs_to_show = IoUs[1:]
        PA_to_show = PA[1:]
        Recall_to_show = Recall[1:]
        Precision_to_show = Precision[1:]
        F1_to_show = F1_Scores[1:] if F1_Scores is not None else None
        suffix = "_without_bg"
    
    draw_plot_func(IoUs_to_show, classes_to_show, "mIoU = {0:.2f}%".format(np.nanmean(IoUs_to_show)*100), "Intersection over Union", \
        os.path.join(miou_out_path, f"mIoU{suffix}.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, f"mIoU{suffix}.png"))

    draw_plot_func(PA_to_show, classes_to_show, "mPA = {0:.2f}%".format(np.nanmean(PA_to_show)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, f"mPA{suffix}.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, f"mPA{suffix}.png"))
    
    draw_plot_func(Recall_to_show, classes_to_show, "mRecall = {0:.2f}%".format(np.nanmean(Recall_to_show)*100), "Recall", \
        os.path.join(miou_out_path, f"Recall{suffix}.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, f"Recall{suffix}.png"))

    draw_plot_func(Precision_to_show, classes_to_show, "mPrecision = {0:.2f}%".format(np.nanmean(Precision_to_show)*100), "Precision", \
        os.path.join(miou_out_path, f"Precision{suffix}.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, f"Precision{suffix}.png"))

    # 绘制F1分数图表
    if F1_to_show is not None:
        draw_plot_func(F1_to_show, classes_to_show, "mF1 = {0:.2f}%".format(np.nanmean(F1_to_show)*100), "F1-Score", \
            os.path.join(miou_out_path, f"F1{suffix}.png"), tick_font_size = tick_font_size, plt_show = False)
        print("Save F1-Score out to " + os.path.join(miou_out_path, f"F1{suffix}.png"))

    # 绘制混淆矩阵热力图
    draw_confusion_matrix(hist, name_classes, 
                         os.path.join(miou_out_path, f"confusion_matrix_heatmap{suffix}.png"),
                         include_background=include_background, normalize=False)
    print("Save confusion matrix heatmap to " + os.path.join(miou_out_path, f"confusion_matrix_heatmap{suffix}.png"))
    
    # 绘制归一化混淆矩阵热力图
    draw_confusion_matrix(hist, name_classes, 
                         os.path.join(miou_out_path, f"confusion_matrix_normalized{suffix}.png"),
                         include_background=include_background, normalize=True)
    print("Save normalized confusion matrix heatmap to " + os.path.join(miou_out_path, f"confusion_matrix_normalized{suffix}.png"))

    # 保存CSV格式的混淆矩阵
    with open(os.path.join(miou_out_path, f"confusion_matrix{suffix}.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        if include_background:
            writer_list.append([' '] + [str(c) for c in name_classes])
            for i in range(len(hist)):
                writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        else:
            writer_list.append([' '] + [str(c) for c in classes_to_show])
            for i in range(1, len(hist)):  # 从1开始，排除背景类
                writer_list.append([name_classes[i]] + [str(x) for x in hist[i][1:]])  # 排除背景类的列
        writer.writerows(writer_list)
    print("Save confusion matrix CSV to " + os.path.join(miou_out_path, f"confusion_matrix{suffix}.csv"))
            