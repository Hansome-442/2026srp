"""模型验证 / 精度评估脚本：输出 mAP、每类指标，并生成中文评估报告。

用法
----
python scripts/val.py --weights runs/train/classroom_yolo11s/weights/best.pt
python scripts/val.py --weights runs/best.pt --split test --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 与 train.py 一致：切到项目根，保证 dataset.yaml 里的相对 path 能找到数据
os.chdir(PROJECT_ROOT)

from src.utils import REPORTS_DIR, ensure_dir, get_logger, load_classes, load_yaml, resolve_path  # noqa: E402

LOG = get_logger("val")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="验证 YOLOv11 课堂行为模型")
    parser.add_argument("--weights", type=str, default="runs/best.pt")
    parser.add_argument("--data", type=str, default="data/dataset.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="评估时建议保留低阈值")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--tta", action="store_true", help="开启 TTA 多尺度/翻转增强评估（更准但更慢）")
    args = parser.parse_args(argv)

    weights = resolve_path(args.weights)
    data = resolve_path(args.data)
    if not Path(weights).exists():
        LOG.error("权重不存在：%s", weights)
        return 2
    if not Path(data).exists():
        LOG.error("数据配置不存在：%s", data)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=args.conf, project=str(resolve_path("runs/val")),
        name=args.name or f"{Path(weights).parent.parent.name}_{args.split}", exist_ok=True,
        augment=bool(args.tta),
    )
    if args.tta:
        LOG.info("已开启 TTA 评估（augment=True）")

    class_info = load_classes()
    zh = class_info["zh"]
    names = getattr(metrics, "names", {}) or {}

    def _as_list(value):
        """numpy 数组不能直接做布尔判断，统一转成 list。"""
        if value is None:
            return []
        try:
            return [float(v) for v in value]
        except TypeError:
            return [float(value)]

    maps = _as_list(getattr(metrics.box, "maps", None))
    ap50 = _as_list(getattr(metrics.box, "ap50", None))
    ap = _as_list(getattr(metrics.box, "ap", None))
    p = _as_list(getattr(metrics.box, "p", None))
    r = _as_list(getattr(metrics.box, "r", None))

    lines = [
        "# 模型验证报告",
        "",
        f"- 权重：`{weights}`",
        f"- 数据：`{data}`（split={args.split}）",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 总体 mAP50：**{float(getattr(metrics.box, 'map50', 0)):.4f}**",
        f"- 总体 mAP50-95：**{float(getattr(metrics.box, 'map', 0)):.4f}**",
        "",
        "## 每类指标",
        "",
        "| id | 类别 | 精确率 P | 召回率 R | mAP50 | mAP50-95 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i in range(len(maps) if maps else len(zh)):
        label = zh[i] if i < len(zh) else str(names.get(i, i))
        row = [
            f"{p[i]:.4f}" if i < len(p) else "-",
            f"{r[i]:.4f}" if i < len(r) else "-",
            f"{ap50[i]:.4f}" if i < len(ap50) else "-",
            f"{ap[i]:.4f}" if i < len(ap) else "-",
        ]
        lines.append(f"| {i} | {label} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines.append("")

    out = ensure_dir(REPORTS_DIR) / f"val_report_{args.split}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("评估报告：%s", out)
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
