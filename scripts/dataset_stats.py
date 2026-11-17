"""数据集统计脚本：对任意 YOLO 数据集目录做体检，输出中文统计报告与图表。

用法
----
python scripts/dataset_stats.py --data data/dataset.yaml
python scripts/dataset_stats.py --data data/dataset.yaml --charts
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import REPORTS_DIR, dump_json, ensure_dir, get_logger, load_classes, load_yaml, resolve_path, setup_matplotlib  # noqa: E402

LOG = get_logger("stats")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_split(root: Path, split: str, data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    entry = data_cfg.get(split)
    if not entry:
        return {}
    if isinstance(entry, str):
        entry = [entry]
    class_counter: Counter = Counter()
    image_class_counter: Counter = Counter()
    images = 0
    boxes = 0
    boxes_per_image: List[int] = []
    sizes: Counter = Counter()
    missing_labels = 0
    for item in entry:
        base = Path(item)
        if not base.is_absolute():
            base = Path(data_cfg.get("path", "")) / item
        if base.is_file():
            names = [line.strip() for line in base.read_text(encoding="utf-8").splitlines() if line.strip()]
            img_paths = [Path(n) for n in names]
        else:
            img_paths = [p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        for img in img_paths:
            images += 1
            # 依次尝试几种常见的 YOLO 目录布局
            candidates = [
                base.parent.parent / "labels" / base.name / f"{img.stem}.txt",  # <root>/images/<split> -> <root>/labels/<split>
                base.parent.parent / "labels" / f"{img.stem}.txt",               # <root>/images/<split> -> <root>/labels
                img.with_suffix(".txt"),                                        # 与图片同目录
                img.parent / f"{img.stem}.txt",
                img.parent.parent / "labels" / f"{img.stem}.txt",
            ]
            label = next((c for c in candidates if c.exists()), None)
            if label is None:
                missing_labels += 1
                continue
            n = 0
            present: set = set()
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                class_counter[cid] += 1
                present.add(cid)
                n += 1
            boxes += n
            boxes_per_image.append(n)
            for cid in present:
                image_class_counter[cid] += 1
            try:
                from PIL import Image

                with Image.open(img) as im:
                    sizes[f"{im.width}x{im.height}"] += 1
            except Exception:
                pass
    avg_boxes = round(boxes / images, 2) if images else 0.0
    return {
        "images": images,
        "boxes": boxes,
        "avg_boxes_per_image": avg_boxes,
        "missing_labels": missing_labels,
        "instances_per_class": dict(class_counter),
        "images_per_class": dict(image_class_counter),
        "top_sizes": dict(sizes.most_common(10)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="数据集统计")
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--charts", action="store_true", help="额外生成统计图")
    parser.add_argument("--out", default=None, help="报告输出路径（.md）")
    args = parser.parse_args(argv)

    data_path = resolve_path(args.data)
    if not Path(data_path).exists():
        LOG.error("找不到数据集配置：%s", data_path)
        return 2
    data_cfg = load_yaml(data_path)
    class_info = load_classes()
    zh = class_info["zh"]

    stats: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_yaml": str(data_path),
        "num_classes": data_cfg.get("nc", class_info["num_classes"]),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        info = scan_split(Path(data_path).parent, split, data_cfg)
        if info:
            stats["splits"][split] = info
            LOG.info("%s：%d 张，%d 框，平均 %.2f 框/图%s",
                     split, info["images"], info["boxes"], info["avg_boxes_per_image"],
                     f"，缺标注 {info['missing_labels']}" if info["missing_labels"] else "")

    total_images = sum(s["images"] for s in stats["splits"].values())
    total_boxes = sum(s["boxes"] for s in stats["splits"].values())

    lines = [
        "# 数据集体检报告",
        "",
        f"- 生成时间：{stats['generated_at']}",
        f"- 配置文件：`{data_path}`",
        f"- 图片总数：**{total_images}**，标注框总数：**{total_boxes}**",
        "",
        "## 各划分统计",
        "",
        "| 划分 | 图片数 | 标注框 | 平均框/图 | 缺标注 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split, info in stats["splits"].items():
        lines.append(f"| {split} | {info['images']} | {info['boxes']} | {info['avg_boxes_per_image']} | {info['missing_labels']} |")

    lines += ["", "## 类别实例数（train/val/test）", "", "| id | 类别 | train | val | test | 合计 | 图片数 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    images_per_class: Counter = Counter()
    for info in stats["splits"].values():
        for k, v in info.get("images_per_class", {}).items():
            images_per_class[int(k)] += v
    for i, name in enumerate(zh):
        row = [stats["splits"].get(s, {}).get("instances_per_class", {}).get(i, 0) for s in ["train", "val", "test"]]
        flag = " ⚠️" if sum(row) < 300 else ""
        lines.append(f"| {i} | {name}{flag} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} | {images_per_class.get(i, 0)} |")
    lines += ["", "> 带 ⚠️ 的类别实例数少于 300，建议继续补采样本（稀有类会显著拉低 mAP）。", ""]

    sizes: Counter = Counter()
    for info in stats["splits"].values():
        for k, v in info.get("top_sizes", {}).items():
            sizes[k] += v
    if sizes:
        lines += ["## 图片尺寸分布（Top 10）", "", "| 尺寸 | 数量 |", "| --- | --- |"]
        for k, v in sizes.most_common(10):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    out = Path(args.out) if args.out else (ensure_dir(REPORTS_DIR) / "dataset_check.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    dump_json(stats, REPORTS_DIR / "dataset_check.json")
    LOG.info("报告已生成：%s", out)

    if args.charts:
        setup_matplotlib()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = [sum(stats["splits"].get(s, {}).get("instances_per_class", {}).get(i, 0) for s in stats["splits"]) for i in range(len(zh))]
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
        bars = ax.bar(zh, counts, color="#4C8BF5")
        ax.set_ylabel("实例数")
        ax.set_title("数据集类别分布")
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        chart = ensure_dir(REPORTS_DIR) / "dataset_class_distribution.png"
        fig.savefig(chart)
        plt.close(fig)
        LOG.info("分布图：%s", chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
