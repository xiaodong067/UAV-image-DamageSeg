"""
Jetson Nano TensorRT 优化版测试脚本
优化预处理和后处理，提升端到端性能
支持显存和功耗监控
"""
import os
import sys
import time
import cv2
import numpy as np
import subprocess
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = "/home/z5cy/Desktop/HXD/seg_test1"
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# 导入 TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"✅ TensorRT 版本: {trt.__version__}")
except ImportError as e:
    print(f"❌ TensorRT 导入失败: {e}")
    print("请安装: pip3 install pycuda")
    sys.exit(1)


def get_jetson_stats():
    """获取 Jetson 的显存和功耗信息"""
    memory_used = 0
    power_usage = 0
    
    try:
        # 通过 tegrastats 或读取系统文件获取信息
        # 方法1: 读取 GPU 内存
        with open('/sys/kernel/debug/nvmap/iovmm/maps', 'r') as f:
            content = f.read()
            # 解析内存使用
    except:
        pass
    
    try:
        # 方法2: 使用 jtop 库（如果安装了）
        from jtop import jtop
        with jtop() as jetson:
            if jetson.ok():
                stats = jetson.stats
                memory_used = stats.get('RAM', 0)
                power_usage = stats.get('Power TOT', 0)
    except:
        pass
    
    try:
        # 方法3: 读取功耗文件 (Jetson Nano)
        power_paths = [
            '/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input',
            '/sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power0_input',
            '/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power0_input'
        ]
        for path in power_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    power_usage = float(f.read().strip()) / 1000  # mW to W
                break
    except:
        pass
    
    return memory_used, power_usage


def get_gpu_memory_tegrastats():
    """通过 tegrastats 获取 GPU 内存使用"""
    try:
        result = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True, text=True, timeout=1
        )
        # 解析 tegrastats 输出
        # 格式类似: RAM 2345/3964MB (lfb 1x4MB) SWAP 0/1982MB ...
        output = result.stdout
        if 'RAM' in output:
            ram_part = output.split('RAM')[1].split('MB')[0]
            used = int(ram_part.split('/')[0].strip())
            return used
    except:
        pass
    return 0


def get_power_usage():
    """获取 Jetson 功耗 (mW)"""
    power_paths = [
        # Jetson Nano
        '/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input',
        '/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power0_input',
        # Jetson Xavier NX
        '/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input',
    ]
    
    for path in power_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return float(f.read().strip())  # mW
        except:
            continue
    
    # 尝试使用 tegrastats
    try:
        result = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True, text=True, timeout=1
        )
        output = result.stdout
        # 解析功耗信息
        if 'POM_5V_IN' in output:
            power_part = output.split('POM_5V_IN')[1].split('/')[0]
            return float(power_part.strip())
    except:
        pass
    
    return 0


class TensorRTInferenceOptimized:
    """优化版 TensorRT 推理类"""
    
    def __init__(self, trt_path, input_shape=(300, 300)):
        self.input_shape = input_shape
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"🔄 加载 TensorRT 引擎: {trt_path}")
        
        with open(trt_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        
        self.colors_bgr = np.array([
            [0, 0, 0],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [0, 0, 255]
        ], dtype=np.uint8)
        
        print("✅ TensorRT 优化版引擎加载完成")
    
    def _allocate_buffers(self):
        self.h_input = cuda.pagelocked_empty(
            trt.volume((1, 3, *self.input_shape)), dtype=np.float32
        )
        self.h_output = cuda.pagelocked_empty(
            trt.volume((1, 5, *self.input_shape)), dtype=np.float32
        )
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()
    
    def preprocess_fast(self, frame_bgr):
        original_h, original_w = frame_bgr.shape[:2]
        ih, iw = self.input_shape
        
        scale = min(iw / original_w, ih / original_h)
        nw, nh = int(original_w * scale), int(original_h * scale)
        
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        canvas = np.full((ih, iw, 3), 128, dtype=np.uint8)
        dx, dy = (iw - nw) // 2, (ih - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized_rgb
        
        blob = canvas.astype(np.float32) * (1.0 / 255.0)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1).reshape(-1))
        
        return blob, (original_h, original_w), (nw, nh, dx, dy)
    
    def postprocess_fast(self, output, original_shape, pad_info):
        original_h, original_w = original_shape
        nw, nh, dx, dy = pad_info
        ih, iw = self.input_shape
        
        pr = output.reshape(5, ih, iw)
        pr = np.argmax(pr, axis=0)
        pr = pr[dy:dy+nh, dx:dx+nw]
        pr = cv2.resize(pr.astype(np.uint8), (original_w, original_h), 
                       interpolation=cv2.INTER_NEAREST)
        return pr
    
    def predict(self, frame_bgr):
        blob, original_shape, pad_info = self.preprocess_fast(frame_bgr)
        np.copyto(self.h_input, blob)
        
        infer_start = time.time()
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        infer_time = time.time() - infer_start
        
        pred = self.postprocess_fast(self.h_output, original_shape, pad_info)
        colored = self.colors_bgr[pred]
        result_bgr = cv2.addWeighted(frame_bgr, 0.6, colored, 0.4, 0)
        
        return result_bgr, infer_time


def save_fps_results(video_path, frame_count, infer_fps_list, e2e_fps_list, memory_list, power_list):
    """保存 FPS、显存、功耗数据和可视化图表"""
    save_dir = os.path.join(PROJECT_DIR, "fps_results_trt")
    os.makedirs(save_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{video_name}_{timestamp}"
    
    avg_infer_fps = np.mean(infer_fps_list)
    avg_e2e_fps = np.mean(e2e_fps_list)
    infer_times = [1000/fps for fps in infer_fps_list]
    e2e_times = [1000/fps for fps in e2e_fps_list]
    avg_memory = np.mean(memory_list) if memory_list and any(memory_list) else 0
    max_memory = max(memory_list) if memory_list and any(memory_list) else 0
    avg_power = np.mean(power_list) if power_list and any(power_list) else 0
    max_power = max(power_list) if power_list and any(power_list) else 0
    
    # 保存 txt
    txt_path = os.path.join(save_dir, f"{base_name}_fps_data.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Jetson Nano TensorRT 建筑损伤检测 - 性能数据\n")
        f.write("=" * 50 + "\n")
        f.write(f"视频名称: {video_name}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总帧数: {frame_count}\n")
        f.write(f"推理引擎: TensorRT FP16\n\n")
        
        f.write("性能统计:\n")
        f.write(f"  --- 纯推理 ---\n")
        f.write(f"  平均 FPS: {avg_infer_fps:.2f}\n")
        f.write(f"  最高 FPS: {max(infer_fps_list):.2f}\n")
        f.write(f"  最低 FPS: {min(infer_fps_list):.2f}\n")
        f.write(f"  平均延迟: {np.mean(infer_times):.2f} ms\n")
        f.write(f"  --- 端到端 ---\n")
        f.write(f"  平均 FPS: {avg_e2e_fps:.2f}\n")
        f.write(f"  最高 FPS: {max(e2e_fps_list):.2f}\n")
        f.write(f"  最低 FPS: {min(e2e_fps_list):.2f}\n")
        f.write(f"  平均延迟: {np.mean(e2e_times):.2f} ms\n")
        f.write(f"  --- 资源占用 ---\n")
        f.write(f"  平均显存: {avg_memory:.2f} MB\n")
        f.write(f"  峰值显存: {max_memory:.2f} MB\n")
        f.write(f"  平均功耗: {avg_power:.2f} mW\n")
        f.write(f"  峰值功耗: {max_power:.2f} mW\n\n")
        
        f.write("详细数据:\n")
        f.write("帧号\t推理FPS\t端到端FPS\t推理时间(ms)\t端到端时间(ms)\t显存(MB)\t功耗(mW)\n")
        for i in range(len(infer_fps_list)):
            mem = memory_list[i] if i < len(memory_list) else 0
            pwr = power_list[i] if i < len(power_list) else 0
            f.write(f"{i+1}\t{infer_fps_list[i]:.2f}\t{e2e_fps_list[i]:.2f}\t")
            f.write(f"{infer_times[i]:.2f}\t{e2e_times[i]:.2f}\t{mem:.2f}\t{pwr:.2f}\n")
    
    print(f"💾 FPS数据已保存: {txt_path}")
    
    # 生成图表
    try:
        frames = list(range(1, len(infer_fps_list) + 1))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. FPS 曲线图
        axes[0, 0].plot(frames, e2e_fps_list, 'b-', label='End-to-End FPS', linewidth=0.8, alpha=0.7)
        axes[0, 0].plot(frames, infer_fps_list, 'r-', label='Pure Inference FPS', linewidth=0.8, alpha=0.7)
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].set_title('FPS Performance (Jetson TensorRT)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 处理时间曲线图
        axes[0, 1].plot(frames, e2e_times, 'g-', label='End-to-End Time', linewidth=0.8, alpha=0.7)
        axes[0, 1].plot(frames, infer_times, 'orange', label='Pure Inference Time', linewidth=0.8, alpha=0.7)
        axes[0, 1].set_xlabel('Frame Number')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].set_title('Processing Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 功耗曲线图
        if power_list and any(power_list):
            axes[0, 2].plot(frames[:len(power_list)], power_list, 'red', label='Power', linewidth=0.8)
            axes[0, 2].axhline(avg_power, color='blue', linestyle='--', label=f'Avg: {avg_power:.1f} mW')
            axes[0, 2].set_xlabel('Frame Number')
            axes[0, 2].set_ylabel('Power (mW)')
            axes[0, 2].set_title('Power Consumption')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Power data not available', ha='center', va='center')
            axes[0, 2].set_title('Power Consumption')
        
        # 4. FPS 分布直方图
        axes[1, 0].hist(e2e_fps_list, bins=30, alpha=0.7, color='blue', label='End-to-End FPS')
        axes[1, 0].axvline(avg_e2e_fps, color='red', linestyle='--', label=f'Mean: {avg_e2e_fps:.2f}')
        axes[1, 0].set_xlabel('FPS')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('E2E FPS Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 显存曲线图
        if memory_list and any(memory_list):
            axes[1, 1].plot(frames[:len(memory_list)], memory_list, 'purple', label='Memory', linewidth=0.8)
            axes[1, 1].axhline(avg_memory, color='red', linestyle='--', label=f'Avg: {avg_memory:.1f} MB')
            axes[1, 1].set_xlabel('Frame Number')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Memory data not available', ha='center', va='center')
            axes[1, 1].set_title('Memory Usage')
        
        # 6. 性能统计表格
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        stats_data = [
            ['Metric', 'Value'],
            ['Avg Inference FPS', f'{avg_infer_fps:.2f}'],
            ['Avg E2E FPS', f'{avg_e2e_fps:.2f}'],
            ['Avg Inference Time', f'{np.mean(infer_times):.2f} ms'],
            ['Avg E2E Time', f'{np.mean(e2e_times):.2f} ms'],
            ['Avg Memory', f'{avg_memory:.2f} MB'],
            ['Peak Memory', f'{max_memory:.2f} MB'],
            ['Avg Power', f'{avg_power:.2f} mW'],
            ['Peak Power', f'{max_power:.2f} mW'],
            ['Total Frames', f'{frame_count}']
        ]
        table = axes[1, 2].table(cellText=stats_data, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        axes[1, 2].set_title('Performance Summary (Jetson TensorRT)', pad=20)
        
        for i in range(len(stats_data)):
            for j in range(len(stats_data[0])):
                if i == 0:
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{base_name}_performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 性能图表已保存: {plot_path}")
    except Exception as e:
        print(f"⚠️ 生成图表时出错: {e}")


def test_video(video_path, output_path=None):
    """测试视频"""
    trt_path = os.path.join(PROJECT_DIR, "model/building_damage_fp16.trt")
    
    if not os.path.exists(trt_path):
        print(f"❌ 找不到 TensorRT 引擎: {trt_path}")
        return
    
    model = TensorRTInferenceOptimized(trt_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 视频: {width}x{height} @ {fps:.1f}fps, 共 {total} 帧")
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 输出: {output_path}")
    
    frame_count = 0
    infer_fps_list = []
    e2e_fps_list = []
    memory_list = []
    power_list = []
    
    SHOW_WINDOW = True
    
    print("\n🎬 TensorRT 优化版推理开始... (按 'q' 退出)\n")
    
    try:
        while True:
            e2e_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            result_bgr, infer_time = model.predict(frame)
            
            infer_fps = 1.0 / infer_time
            infer_fps_list.append(infer_fps)
            
            e2e_time = time.time() - e2e_start
            e2e_fps = 1.0 / e2e_time
            e2e_fps_list.append(e2e_fps)
            
            # 每30帧记录一次资源使用
            if frame_count % 30 == 0:
                mem = get_gpu_memory_tegrastats()
                pwr = get_power_usage()
                memory_list.append(mem)
                power_list.append(pwr)
            elif memory_list:
                memory_list.append(memory_list[-1])
                power_list.append(power_list[-1])
            else:
                memory_list.append(0)
                power_list.append(0)
            
            # 叠加信息
            info1 = f"Infer FPS: {infer_fps:.1f} | Infer: {infer_time*1000:.1f}ms"
            info2 = f"E2E FPS: {e2e_fps:.1f} | E2E: {e2e_time*1000:.1f}ms | Frame: {frame_count}/{total}"
            cv2.putText(result_bgr, info1, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_bgr, info2, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if SHOW_WINDOW:
                cv2.imshow("Building Damage - TensorRT Optimized", result_bgr)
            
            if writer:
                writer.write(result_bgr)
            
            if frame_count % 30 == 0:
                avg_infer = np.mean(infer_fps_list[-30:])
                avg_e2e = np.mean(e2e_fps_list[-30:])
                progress = frame_count / total * 100
                pwr_str = f" | 功耗: {power_list[-1]:.0f}mW" if power_list[-1] > 0 else ""
                print(f"📊 {progress:.1f}% | 帧 {frame_count}/{total} | 推理: {avg_infer:.1f} FPS | 端到端: {avg_e2e:.1f} FPS{pwr_str}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if infer_fps_list:
            avg_infer_fps = np.mean(infer_fps_list)
            avg_e2e_fps = np.mean(e2e_fps_list)
            avg_memory = np.mean([m for m in memory_list if m > 0]) if any(memory_list) else 0
            avg_power = np.mean([p for p in power_list if p > 0]) if any(power_list) else 0
            
            print(f"\n{'='*60}")
            print(f"📈 TensorRT 优化版性能统计:")
            print(f"   处理帧数: {frame_count}")
            print(f"   --- 纯推理 ---")
            print(f"   平均 FPS: {avg_infer_fps:.2f}")
            print(f"   最高 FPS: {max(infer_fps_list):.2f}")
            print(f"   最低 FPS: {min(infer_fps_list):.2f}")
            print(f"   平均延迟: {1000/avg_infer_fps:.1f} ms")
            print(f"   --- 端到端 ---")
            print(f"   平均 FPS: {avg_e2e_fps:.2f}")
            print(f"   最高 FPS: {max(e2e_fps_list):.2f}")
            print(f"   最低 FPS: {min(e2e_fps_list):.2f}")
            print(f"   平均延迟: {1000/avg_e2e_fps:.1f} ms")
            print(f"   --- 资源占用 ---")
            print(f"   平均显存: {avg_memory:.2f} MB")
            print(f"   平均功耗: {avg_power:.2f} mW")
            print(f"{'='*60}")
            
            save_fps_results(video_path, frame_count, infer_fps_list, e2e_fps_list, memory_list, power_list)


if __name__ == "__main__":
    print("="*60)
    print("   Jetson Nano TensorRT 优化版建筑损伤检测")
    print("="*60)
    
    VIDEO_PATH = "/home/z5cy/Desktop/HXD/seg_test1/test.mp4"
    OUTPUT_PATH = "/home/z5cy/Desktop/HXD/seg_test1/result_trt_opt.mp4"
    
    if os.path.exists(VIDEO_PATH):
        test_video(VIDEO_PATH, OUTPUT_PATH)
    else:
        print(f"❌ 找不到视频: {VIDEO_PATH}")
        print("请修改 VIDEO_PATH 变量")
