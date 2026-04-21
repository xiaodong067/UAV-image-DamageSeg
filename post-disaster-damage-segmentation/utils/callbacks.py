import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1, name_classes=None, train_image_ids=None):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        self.name_classes       = name_classes if name_classes is not None else [f"class_{i}" for i in range(num_classes)]
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        
        # 训练集图片ID（用于对比过拟合）
        self.train_image_ids    = None
        if train_image_ids is not None:
            self.train_image_ids = [image_id.split()[0] for image_id in train_image_ids]
            if len(self.train_image_ids) > 300:
                import random
                random.seed(42)
                self.train_image_ids = random.sample(self.train_image_ids, 300)
        
        # 验证集mIoU跟踪
        self.mious      = [0]
        self.epoches    = [0]
        
        # 训练集mIoU跟踪（用于对比过拟合）
        self.train_mious = [0]
        
        # 新增：每个类别的指标跟踪（验证集）
        self.class_ious = [[0] for _ in range(num_classes)]      # 每个类别的IoU历史
        self.class_pas = [[0] for _ in range(num_classes)]       # 每个类别的像素准确率历史
        self.class_recalls = [[0] for _ in range(num_classes)]   # 每个类别的召回率历史
        self.class_precisions = [[0] for _ in range(num_classes)] # 每个类别的精确率历史
        self.class_f1_scores = [[0] for _ in range(num_classes)]  # 每个类别的F1分数历史
        
        # 训练集每个类别的IoU跟踪
        self.train_class_ious = [[0] for _ in range(num_classes)]
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
            
            # 训练集mIoU记录文件
            if self.train_image_ids is not None:
                with open(os.path.join(self.log_dir, "epoch_train_miou.txt"), 'a') as f:
                    f.write(str(0))
                    f.write("\n")
            
            # 为每个类别创建记录文件
            for i in range(num_classes):
                class_name = self.name_classes[i]
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_iou.txt"), 'a') as f:
                    f.write(str(0))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_pa.txt"), 'a') as f:
                    f.write(str(0))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_recall.txt"), 'a') as f:
                    f.write(str(0))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_precision.txt"), 'a') as f:
                    f.write(str(0))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_f1.txt"), 'a') as f:
                    f.write(str(0))
                    f.write("\n")

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            
            #------------------------------------------------#
            #   评估验证集
            #------------------------------------------------#
            print("Get val miou.")
            for image_id in tqdm(self.image_ids, desc="Val"):
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate val metrics.")
            _, IoUs, PA, Recall, Precision, F1_Scores = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)
            # 排除背景类计算mIoU（与compute_mIoU内部打印一致）
            temp_miou = np.nanmean(IoUs[1:]) * 100  # IoUs[1:] 排除背景类

            # 记录验证集总体mIoU
            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            #------------------------------------------------#
            #   评估训练集（用于检测过拟合）
            #------------------------------------------------#
            train_miou = 0
            train_IoUs = None
            if self.train_image_ids is not None:
                # 清空预测目录
                shutil.rmtree(pred_dir)
                os.makedirs(pred_dir)
                
                print("Get train miou (sampled).")
                for image_id in tqdm(self.train_image_ids, desc="Train"):
                    image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                    image       = Image.open(image_path)
                    image       = self.get_miou_png(image)
                    image.save(os.path.join(pred_dir, image_id + ".png"))
                
                print("Calculate train metrics.")
                _, train_IoUs, _, _, _, _ = compute_mIoU(gt_dir, pred_dir, self.train_image_ids, self.num_classes, None)
                # 排除背景类计算mIoU
                train_miou = np.nanmean(train_IoUs[1:]) * 100  # train_IoUs[1:] 排除背景类
                
                self.train_mious.append(train_miou)
                
                with open(os.path.join(self.log_dir, "epoch_train_miou.txt"), 'a') as f:
                    f.write(str(train_miou))
                    f.write("\n")
                
                # 记录训练集每个类别的IoU
                for i in range(self.num_classes):
                    train_class_iou = train_IoUs[i] * 100 if not np.isnan(train_IoUs[i]) else 0
                    self.train_class_ious[i].append(train_class_iou)
                
                # 打印过拟合差距
                gap = train_miou - temp_miou
                print(f"📊 Epoch {epoch}: Train mIoU={train_miou:.2f}%, Val mIoU={temp_miou:.2f}%, Gap={gap:.2f}%")
                if gap > 10:
                    print(f"⚠️ 警告: 过拟合差距较大 ({gap:.2f}%)，建议增加正则化或数据增强")
            
            # 记录验证集每个类别的指标
            for i in range(self.num_classes):
                class_name = self.name_classes[i]
                
                class_iou = IoUs[i] * 100 if not np.isnan(IoUs[i]) else 0
                class_pa = PA[i] * 100 if not np.isnan(PA[i]) else 0
                class_recall = Recall[i] * 100 if not np.isnan(Recall[i]) else 0
                class_precision = Precision[i] * 100 if not np.isnan(Precision[i]) else 0
                class_f1 = F1_Scores[i] * 100 if not np.isnan(F1_Scores[i]) else 0
                
                self.class_ious[i].append(class_iou)
                self.class_pas[i].append(class_pa)
                self.class_recalls[i].append(class_recall)
                self.class_precisions[i].append(class_precision)
                self.class_f1_scores[i].append(class_f1)
                
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_iou.txt"), 'a') as f:
                    f.write(str(class_iou))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_pa.txt"), 'a') as f:
                    f.write(str(class_pa))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_recall.txt"), 'a') as f:
                    f.write(str(class_recall))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_precision.txt"), 'a') as f:
                    f.write(str(class_precision))
                    f.write("\n")
                with open(os.path.join(self.log_dir, f"epoch_{class_name}_f1.txt"), 'a') as f:
                    f.write(str(class_f1))
                    f.write("\n")
            
            #------------------------------------------------#
            #   绘制对比曲线图
            #------------------------------------------------#
            # 绘制训练集vs验证集mIoU对比曲线
            self._plot_train_val_comparison()
            
            # 绘制每个类别的训练集vs验证集IoU对比
            if self.train_image_ids is not None:
                self._plot_class_train_val_comparison()
            
            # 绘制总体mIoU曲线
            plt.figure(figsize=(10, 6))
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='Val mIoU')
            if self.train_image_ids is not None:
                plt.plot(self.epoches, self.train_mious, 'blue', linewidth = 2, label='Train mIoU')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('mIoU (%)')
            plt.title('Mean IoU Curve (Train vs Val)')
            plt.legend(loc="upper left")
            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")
            
            # 绘制每个类别的IoU曲线
            self._plot_class_metrics(self.class_ious, 'IoU', 'IoU (%)', 'Class IoU Curves')
            
            # 绘制每个类别的像素准确率曲线
            self._plot_class_metrics(self.class_pas, 'PA', 'Pixel Accuracy (%)', 'Class Pixel Accuracy Curves')
            
            # 绘制每个类别的召回率曲线
            self._plot_class_metrics(self.class_recalls, 'Recall', 'Recall (%)', 'Class Recall Curves')
            
            # 绘制每个类别的精确率曲线
            self._plot_class_metrics(self.class_precisions, 'Precision', 'Precision (%)', 'Class Precision Curves')
            
            # 绘制每个类别的F1分数曲线
            self._plot_class_metrics(self.class_f1_scores, 'F1', 'F1-Score (%)', 'Class F1-Score Curves')

            print("Get metrics done.")
            shutil.rmtree(self.miou_out_path)
    
    def _plot_train_val_comparison(self):
        """绘制训练集vs验证集mIoU对比曲线"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.epoches, self.mious, 'r-', linewidth=2, label='Val mIoU', marker='o', markersize=4)
        if self.train_image_ids is not None and len(self.train_mious) > 1:
            plt.plot(self.epoches, self.train_mious, 'b-', linewidth=2, label='Train mIoU', marker='s', markersize=4)
            
            # 填充过拟合区域
            train_arr = np.array(self.train_mious)
            val_arr = np.array(self.mious)
            plt.fill_between(self.epoches, val_arr, train_arr, 
                           where=(train_arr > val_arr), alpha=0.3, color='orange', label='Overfit Gap')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mIoU (%)', fontsize=12)
        plt.title('Train vs Val mIoU (Overfitting Detection)', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        
        # 显示最新的差距
        if self.train_image_ids is not None and len(self.train_mious) > 1:
            gap = self.train_mious[-1] - self.mious[-1]
            plt.text(0.98, 0.02, f'Latest Gap: {gap:.2f}%', 
                    transform=plt.gca().transAxes, fontsize=11, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow' if gap > 10 else 'lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_train_val_miou.png"), dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close("all")
        print(f"Save train vs val mIoU comparison to {os.path.join(self.log_dir, 'epoch_train_val_miou.png')}")
    
    def _plot_class_train_val_comparison(self):
        """绘制每个类别的训练集vs验证集IoU对比"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors_train = plt.cm.Blues(0.7)
        colors_val = plt.cm.Reds(0.7)
        
        for i in range(min(self.num_classes, 5)):
            ax = axes[i]
            class_name = self.name_classes[i]
            
            ax.plot(self.epoches, self.class_ious[i], 'r-', linewidth=2, label='Val', marker='o', markersize=3)
            ax.plot(self.epoches, self.train_class_ious[i], 'b-', linewidth=2, label='Train', marker='s', markersize=3)
            
            # 填充过拟合区域
            train_arr = np.array(self.train_class_ious[i])
            val_arr = np.array(self.class_ious[i])
            ax.fill_between(self.epoches, val_arr, train_arr, 
                          where=(train_arr > val_arr), alpha=0.3, color='orange')
            
            ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU (%)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 显示差距
            if len(self.train_class_ious[i]) > 1:
                gap = self.train_class_ious[i][-1] - self.class_ious[i][-1]
                ax.text(0.98, 0.02, f'Gap: {gap:.1f}%', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='yellow' if gap > 10 else 'lightgreen', alpha=0.7))
        
        # 隐藏多余的子图
        for i in range(self.num_classes, 6):
            axes[i].axis('off')
        
        plt.suptitle('Per-Class Train vs Val IoU (Overfitting Detection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_class_train_val_iou.png"), dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close("all")
        print(f"Save per-class train vs val IoU to {os.path.join(self.log_dir, 'epoch_class_train_val_iou.png')}")

    def _plot_class_metrics(self, class_metrics, metric_name, ylabel, title):
        """
        绘制每个类别的指标曲线
        
        Args:
            class_metrics: 每个类别的指标历史列表
            metric_name: 指标名称，用于文件命名
            ylabel: y轴标签
            title: 图表标题
        """
        # 设置颜色映射，为每个类别分配不同颜色
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))
        
        plt.figure(figsize=(12, 8))
        
        # 为每个类别绘制曲线
        for i in range(self.num_classes):
            class_name = self.name_classes[i]
            plt.plot(self.epoches, class_metrics[i], 
                    color=colors[i], linewidth=2, label=f'{class_name}', 
                    marker='o', markersize=3, alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # 设置图例
        if self.num_classes <= 10:
            # 类别较少时显示所有图例
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            # 类别较多时不显示图例，避免过于拥挤
            plt.text(0.02, 0.98, f'Total {self.num_classes} classes', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 调整布局以适应图例
        plt.tight_layout()
        
        # 保存图片
        filename = f"epoch_class_{metric_name.lower()}.png"
        plt.savefig(os.path.join(self.log_dir, filename), dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close("all")
        
        print(f"Save class {metric_name} curves to {os.path.join(self.log_dir, filename)}")
        
        # 额外：为每个类别单独绘制一张图（可选，如果需要更清晰的单类别视图）
        if self.num_classes <= 5:  # 只有类别数较少时才单独绘制，避免生成过多文件
            for i in range(self.num_classes):
                class_name = self.name_classes[i]
                plt.figure(figsize=(10, 6))
                plt.plot(self.epoches, class_metrics[i], 
                        color=colors[i], linewidth=3, label=f'{class_name} {metric_name}',
                        marker='o', markersize=5)
                
                plt.grid(True, alpha=0.3)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.title(f'{class_name} {metric_name} Curve', fontsize=14, fontweight='bold')
                plt.legend(loc="upper right", fontsize=12)
                
                # 显示数值
                if len(self.epoches) > 1:
                    latest_value = class_metrics[i][-1]
                    plt.text(0.02, 0.98, f'Latest: {latest_value:.2f}%', 
                            transform=plt.gca().transAxes, fontsize=11, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                
                plt.tight_layout()
                
                # 保存单类别图片
                single_filename = f"epoch_{class_name}_{metric_name.lower()}.png"
                plt.savefig(os.path.join(self.log_dir, single_filename), dpi=300, bbox_inches='tight')
                plt.cla()
                plt.close("all")
