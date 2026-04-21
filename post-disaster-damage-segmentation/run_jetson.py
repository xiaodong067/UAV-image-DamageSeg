#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin Nano 一键启动脚本
自动检测平台、配置环境变量、查找模型、启动 Web App 或 桌面 App

使用方式:
    # 启动 Web App (推荐，适合 ToDesk 远程)
    python3 run_jetson.py --mode web

    # 启动 Desktop App
    python3 run_jetson.py --mode desktop

    # 手动指定 TRT 模型
    python3 run_jetson.py --model /path/to/engine.trt

    # 仅查看平台信息
    python3 run_jetson.py --info
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ---- 模型搜索路径 (按优先级) ----
_user = os.environ.get("USER", "jetson")
JETSON_MODEL_SEARCH_PATHS = [
    SCRIPT_DIR / "model",
    Path(f"/home/{_user}/Desktop/HXD/seg_test1/model"),
    Path(f"/home/{_user}/model"),
    Path.home() / "model",
    SCRIPT_DIR,
]

TRT_ENGINE_NAMES = [
    "building_damage_256_fp16.trt",
    "building_damage_fp16.trt",
    "best_epoch_weights.trt",
]

PTH_MODEL_NAMES = [
    "best_epoch_weights.pth",
    "mobilenetV2_t1w.pth",
]


# ===========================================================================
#  Platform Detection
# ===========================================================================

def detect_jetson() -> dict:
    """检测是否运行在 Jetson 平台上，返回平台详细信息。"""
    info = {
        "is_jetson": False,
        "model_name": "unknown",
        "l4t_release": "",
        "cuda_available": False,
        "tensorrt_available": False,
        "tensorrt_version": "",
        "arch": platform.machine(),
        "python": platform.python_version(),
    }

    # /etc/nv_tegra_release
    tegra_release = Path("/etc/nv_tegra_release")
    if tegra_release.exists():
        info["is_jetson"] = True
        try:
            info["l4t_release"] = tegra_release.read_text().strip().lstrip("# ").split("\n")[0]
        except Exception:
            pass

    # /proc/device-tree/model
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            info["model_name"] = dt_model.read_text().strip().replace("\x00", "")
            info["is_jetson"] = True
        except Exception:
            pass

    # aarch64 架构也可能是 Jetson
    if platform.machine() == "aarch64":
        info["is_jetson"] = True

    # CUDA
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # TensorRT
    try:
        import tensorrt as trt
        info["tensorrt_available"] = True
        info["tensorrt_version"] = trt.__version__
    except ImportError:
        pass

    return info


def find_model(prefer_trt: bool = True):
    """在常见路径中查找模型文件，返回 (path, backend)。"""
    search_order = []
    if prefer_trt:
        # TRT engines first
        for sp in JETSON_MODEL_SEARCH_PATHS:
            for name in TRT_ENGINE_NAMES:
                search_order.append((sp / name, "trt"))
    # PTH models
    for sp in JETSON_MODEL_SEARCH_PATHS:
        for name in PTH_MODEL_NAMES:
            search_order.append((sp / name, "pth"))
    # If not prefer_trt, add TRT as fallback
    if not prefer_trt:
        for sp in JETSON_MODEL_SEARCH_PATHS:
            for name in TRT_ENGINE_NAMES:
                search_order.append((sp / name, "trt"))

    for path, backend in search_order:
        if path.exists():
            return str(path.resolve()), backend

    # Wildcard scan
    for sp in JETSON_MODEL_SEARCH_PATHS:
        if not sp.exists():
            continue
        if prefer_trt:
            for f in sorted(sp.glob("*.trt")):
                return str(f.resolve()), "trt"
        for f in sorted(sp.glob("*.pth")):
            return str(f.resolve()), "pth"
        if not prefer_trt:
            for f in sorted(sp.glob("*.trt")):
                return str(f.resolve()), "trt"

    return None, None


def set_performance_mode(enable: bool = True) -> None:
    """设置 Jetson 最大性能模式 (需要 sudo 权限)。"""
    if not enable:
        return
    print("⚡ 设置 Jetson 最大性能模式...")
    try:
        subprocess.run(["sudo", "nvpmodel", "-m", "0"],
                       check=False, capture_output=True, timeout=5)
        print("   ✅ nvpmodel -m 0 (MAXN)")
    except Exception:
        print("   ⚠️ nvpmodel 设置失败")

    try:
        subprocess.run(["sudo", "jetson_clocks"],
                       check=False, capture_output=True, timeout=5)
        print("   ✅ jetson_clocks 已开启")
    except Exception:
        print("   ⚠️ jetson_clocks 设置失败")


def print_platform_info(info: dict) -> None:
    """打印平台检测结果。"""
    print("=" * 56)
    print("   Jetson 平台检测报告")
    print("=" * 56)
    print(f"  设备识别:    {'✅ Jetson' if info['is_jetson'] else '❌ 非 Jetson 平台'}")
    print(f"  设备型号:    {info['model_name']}")
    print(f"  L4T 版本:    {info.get('l4t_release', 'N/A')}")
    print(f"  架构:        {info['arch']}")
    print(f"  Python:      {info['python']}")
    print(f"  CUDA:        {'✅ 可用' if info['cuda_available'] else '❌ 不可用'}")
    if info.get("cuda_device"):
        print(f"  CUDA 设备:   {info['cuda_device']}")
    print(f"  TensorRT:    {'✅ ' + info['tensorrt_version'] if info['tensorrt_available'] else '❌ 不可用'}")
    print("-" * 56)


def print_model_info(model_path, backend):
    """打印找到的模型信息。"""
    if model_path:
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"  模型路径:    {model_path}")
        print(f"  推理后端:    {backend.upper()}")
        print(f"  文件大小:    {size_mb:.1f} MB")
    else:
        print("  ⚠️ 未找到模型文件！")
        print("     请将模型放入以下目录之一:")
        for sp in JETSON_MODEL_SEARCH_PATHS:
            print(f"       - {sp}")


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Jetson Orin Nano 一键启动脚本 — MGA-ESAM 无人机建筑灾损检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 run_jetson.py --mode web                # 启动 Web App
  python3 run_jetson.py --mode web --port 9090    # 自定义端口
  python3 run_jetson.py --mode desktop             # 启动桌面 App
  python3 run_jetson.py --model model/engine.trt   # 指定模型
  python3 run_jetson.py --info                     # 仅显示平台信息
        """,
    )
    parser.add_argument("--mode", choices=["web", "desktop"], default="web",
                        help="启动模式: web (推荐, 适合 ToDesk 远程) 或 desktop (默认: web)")
    parser.add_argument("--host", default="0.0.0.0", help="Web 服务绑定地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Web 服务端口 (默认: 8080)")
    parser.add_argument("--model", default=None, help="手动指定模型文件路径 (.pth / .trt)")
    parser.add_argument("--backend", choices=["auto", "pth", "trt"], default="auto",
                        help="推理后端 (默认: auto)")
    parser.add_argument("--no-perf", action="store_true", help="不设置 Jetson 最大性能模式")
    parser.add_argument("--no-cuda", action="store_true", help="禁用 CUDA (仅 CPU 推理)")
    parser.add_argument("--info", action="store_true", help="仅显示平台信息，不启动应用")

    args = parser.parse_args()

    # ---- 平台检测 ----
    info = detect_jetson()
    print_platform_info(info)

    # ---- 查找模型 ----
    if args.model:
        model_path = str(Path(args.model).resolve())
        backend = args.backend
        if backend == "auto":
            backend = "trt" if model_path.endswith(".trt") else "pth"
    else:
        prefer_trt = info["tensorrt_available"]
        model_path, backend = find_model(prefer_trt=prefer_trt)
        if args.backend != "auto":
            backend = args.backend

    print_model_info(model_path, backend)
    print("=" * 56)

    if args.info:
        sys.exit(0)

    if model_path is None:
        print("\n❌ 无法找到模型文件，请通过 --model 参数指定路径")
        sys.exit(1)

    # ---- 设置环境变量 ----
    os.environ["MGA_MODEL_PATH"] = model_path
    os.environ["MGA_USE_CUDA"] = "0" if args.no_cuda else ("1" if info["cuda_available"] else "0")
    os.environ["MGA_MODEL_TYPE"] = "enhanced_building_damage"
    os.environ["MGA_BACKBONE"] = "mobilenet"
    os.environ["MGA_NUM_CLASSES"] = "5"
    os.environ["MGA_DOWNSAMPLE_FACTOR"] = "8"
    os.environ["MGA_ATTENTION"] = "esam"
    os.environ["MGA_WEB_HOST"] = args.host
    os.environ["MGA_WEB_PORT"] = str(args.port)

    print(f"\n🔧 环境配置:")
    print(f"   MGA_MODEL_PATH        = {os.environ['MGA_MODEL_PATH']}")
    print(f"   MGA_USE_CUDA          = {os.environ['MGA_USE_CUDA']}")
    print(f"   MGA_MODEL_TYPE        = enhanced_building_damage")
    print(f"   Backend               = {backend}")

    # ---- 性能模式 ----
    if info["is_jetson"] and not args.no_perf:
        set_performance_mode(True)

    # ---- 启动应用 ----
    os.chdir(str(SCRIPT_DIR))
    sys.path.insert(0, str(SCRIPT_DIR))

    if args.mode == "web":
        print(f"\n🌐 启动 Web App — http://{args.host}:{args.port}")
        print(f"   📱 通过 ToDesk 远程访问: 在浏览器打开 http://localhost:{args.port}")
        print(f"   🔗 局域网访问: http://<Jetson_IP>:{args.port}")
        print("-" * 56)
        import uvicorn
        uvicorn.run("uav_webapp.main:app", host=args.host, port=args.port, reload=False)
    else:
        print("\n🖥️ 启动桌面 App...")
        print("-" * 56)
        from uav_desktop_app import main as desktop_main
        desktop_main()


if __name__ == "__main__":
    main()

