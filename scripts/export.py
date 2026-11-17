"""模型导出脚本：导出 ONNX（可选 OpenVINO / TensorRT）用于加速部署。

用法
----
python scripts/export.py --weights runs/best.pt --format onnx
python scripts/export.py --weights runs/best.pt --format openvino --half
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import RUNS_DIR, ensure_dir, get_logger, resolve_path  # noqa: E402

LOG = get_logger("export")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="导出模型")
    parser.add_argument("--weights", type=str, default="runs/best.pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "openvino", "torchscript", "engine"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16（ONNX/engine 支持）")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args(argv)

    weights = resolve_path(args.weights)
    if not Path(weights).exists():
        LOG.error("权重不存在：%s", weights)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = model.export(
        format=args.format, imgsz=args.imgsz, half=args.half,
        dynamic=args.dynamic, simplify=args.simplify, opset=args.opset,
    )
    LOG.info("导出完成：%s", out)

    # 统一拷贝到 runs/export/ 方便查找
    dest_dir = ensure_dir(RUNS_DIR / "export")
    src = Path(str(out))
    if src.exists():
        target = dest_dir / src.name
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
        LOG.info("已复制到：%s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
