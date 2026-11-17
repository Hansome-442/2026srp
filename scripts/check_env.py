# -*- coding: utf-8 -*-
"""环境自检：一条命令确认「这台机器能不能跑这个项目」。

用法
----
python scripts/check_env.py              # 只查依赖 + 目录 + GPU
python scripts/check_env.py --forward    # 额外真跑一次推理（会加载权重）
python scripts/check_env.py --forward --weights runs/best.pt --source test_images

退出码：0 = 全部通过；1 = 有 FAIL（多半是环境没装对）。
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OK, WARN, FAIL = "✅", "⚠️ ", "❌"
results: List[Tuple[str, str, str]] = []   # (状态, 项目, 说明)


def add(state: str, item: str, detail: str = "") -> None:
    results.append((state, item, detail))


# ---------------------------------------------------------------- 依赖

PACKAGES = [
    ("torch", "深度学习框架"),
    ("torchvision", "视觉模型库"),
    ("ultralytics", "YOLO 训练/推理"),
    ("cv2", "OpenCV 图像处理"),
    ("numpy", "数值计算"),
    ("PIL", "Pillow 中文标签绘制"),
    ("pandas", "报表统计"),
    ("matplotlib", "图表"),
    ("openpyxl", "Excel 导出"),
    ("yaml", "读取配置"),
    ("tqdm", "进度条"),
    ("PyQt5", "桌面端界面"),
    ("streamlit", "网页端界面"),
]


def check_packages() -> None:
    for mod, desc in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", None)
            if ver is None and mod == "PyQt5":
                from PyQt5.QtCore import PYQT_VERSION_STR
                ver = PYQT_VERSION_STR
            add(OK, f"{mod:<12} {desc}", str(ver) if ver else "")
        except Exception as exc:                     # noqa: BLE001
            add(FAIL, f"{mod:<12} {desc}", f"导入失败：{exc.__class__.__name__}")


def check_python() -> None:
    v = sys.version_info
    state = OK if v >= (3, 10) else FAIL
    add(state, f"Python {platform.python_version()}", f"{platform.architecture()[0]} / {sys.executable}")


def check_gpu() -> bool:
    try:
        import torch
    except Exception:
        add(FAIL, "GPU", "torch 没装上，无法检查")
        return False
    if not torch.cuda.is_available():
        add(FAIL, "CUDA 可用", "torch.cuda.is_available() = False（装了 CPU 版 torch，或驱动太旧）")
        add(WARN, "运行建议", "只能跑 CPU：训练会非常慢，界面检测可以用但帧率低")
        return False
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    add(OK, f"GPU {name}", f"算力 {cap[0]}.{cap[1]} / {mem:.1f} GB / CUDA {torch.version.cuda}")
    if cap[0] >= 12:
        add(OK, "Blackwell(50 系)适配", "需要 cu128 及以上版本的 torch，当前满足")
    return True


# ---------------------------------------------------------------- 文件

def check_files() -> None:
    must = [
        ("configs/classes.yaml", "8 类定义"),
        ("configs/app.yaml", "界面/模型配置"),
        ("configs/train.yaml", "训练配置"),
        ("src/detector.py", "检测核心"),
    ]
    for rel, desc in must:
        p = PROJECT_ROOT / rel
        add(OK if p.exists() else FAIL, rel, desc if p.exists() else "缺少文件")

    ds = PROJECT_ROOT / "data/dataset.yaml"
    add(OK if ds.exists() else WARN, "data/dataset.yaml",
        "已生成" if ds.exists() else "没有——要先跑 scripts/prepare_dataset.py")

    imgs = list((PROJECT_ROOT / "data/processed/images/train").glob("*")) if (PROJECT_ROOT / "data/processed/images/train").exists() else []
    if imgs:
        add(OK, "data/processed/images/train", f"{len(imgs)} 个文件")
    else:
        add(WARN, "data/processed/images/train", "空或不存在——数据没放进来，见 docs/10 第 3 节")

    for rel in ("runs/best.pt", "weights"):
        p = PROJECT_ROOT / rel
        if p.exists():
            add(OK, rel, "存在")
            break
    else:
        add(WARN, "训练权重", "runs/best.pt 不存在：界面会用 COCO 预训练权重兜底（测不出课堂行为）")


# ---------------------------------------------------------------- 推理

def check_forward(weights: str, source: str) -> None:
    try:
        import cv2
        import numpy as np
        from src.detector import BehaviorDetector
        from src.utils import load_classes

        ci = load_classes()
        det = BehaviorDetector(weights=weights, device="0", imgsz=640, conf=0.25,
                               class_names=ci["zh"], per_class_conf=None)
        det.load()

        src = Path(source)
        if src.is_file():
            frame = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        else:
            cands = [p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            if not cands:
                add(WARN, "推理自检", f"{source} 下没有图片，跳过")
                return
            frame = cv2.imdecode(np.fromfile(str(cands[0]), np.uint8), cv2.IMREAD_COLOR)

        import time
        t0 = time.time()
        dets = det.predict(frame)
        ms = (time.time() - t0) * 1000
        names = sorted({d.cls_name for d in dets}) or ["（无目标）"]
        add(OK, "推理自检", f"{len(dets)} 个框 / {ms:.0f} ms / 类别：{'、'.join(names[:4])}")
    except Exception as exc:                          # noqa: BLE001
        add(FAIL, "推理自检", f"{exc.__class__.__name__}: {exc}")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="课堂行为检测项目 环境自检")
    ap.add_argument("--forward", action="store_true", help="额外跑一次真实推理")
    ap.add_argument("--weights", default="runs/best.pt")
    ap.add_argument("--source", default="test_images")
    args = ap.parse_args()

    print("=" * 78)
    print(" 课堂行为检测系统 · 环境自检")
    print(" 项目目录：%s" % PROJECT_ROOT)
    print("=" * 78)

    check_python()
    check_packages()
    check_gpu()
    check_files()
    if args.forward:
        check_forward(args.weights, args.source)

    for state, item, detail in results:
        print(f"{state} {item:<34} {detail}")

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print("-" * 78)
    print(f"通过 {len(results) - len(fails) - len(warns)} 项 / 警告 {len(warns)} 项 / 失败 {len(fails)} 项")
    if fails:
        print("\n先修失败项，最常见的三个原因：")
        print("  1) 没激活环境（要 conda activate classroom），用的是系统 python")
        print("  2) torch 装成了 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("  3) 缺包：pip install -r requirements.lock.txt")
        return 1
    if warns:
        print("\n警告项不影响代码运行，通常只是数据/权重还没放到位。")
    print("\n环境就绪。下一步：python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
