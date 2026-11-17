"""数据集去重 / 泄漏检查（dHash 感知哈希）。

为什么需要：多来源数据集（Kaggle / Roboflow / SCB 系列 / 自采）之间常常存在
**同一课堂视频的不同抽帧**，甚至同一张图被重复上传。如果它们被随机划分到
train 和 val，验证指标会虚高（数据泄漏），答辩时一换教室就露馅。

本脚本：
1. 对每张图计算 64 位 dHash（缩放到 9x8 灰度后比较相邻像素）；
2. 汉明距离 <= 阈值的图片归为同一簇；
3. 统计「跨划分（train/val/test）重复」与「同一划分内重复」；
4. 输出报告 `reports/dataset_dedup.md` + 重复清单 CSV；
5. 可选 `--drop`：按「保留 test > val > train、同划分内保留第一张」的规则生成剔除清单。

用法
----
python scripts/dataset_dedup.py --data data/dataset.yaml
python scripts/dataset_dedup.py --data data/dataset.yaml --threshold 6 --drop
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import REPORTS_DIR, ensure_dir, get_logger, load_yaml, resolve_path  # noqa: E402

LOG = get_logger("dedup")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path: Path, size: int = 8) -> Optional[int]:
    """64 位差值哈希，对亮度变化不敏感。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            pixels = list(im.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect(data_yaml: Path) -> Dict[str, List[Path]]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    split_files: Dict[str, List[Path]] = {}
    for split in ["train", "val", "test"]:
        entry = cfg.get(split)
        if not entry:
            continue
        base = Path(entry)
        if not base.is_absolute():
            base = root / entry
        split_files[split] = [p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return split_files


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="数据集去重 / 泄漏检查")
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--threshold", type=int, default=6, help="汉明距离阈值，越小越严格")
    parser.add_argument("--drop", action="store_true", help="生成剔除清单 CSV（不实际删除图片）")
    parser.add_argument("--max-images", type=int, default=0, help="调试：只算前 N 张")
    args = parser.parse_args(argv)

    data_yaml = resolve_path(args.data)
    if not Path(data_yaml).exists():
        LOG.error("找不到数据集配置：%s", data_yaml)
        return 2

    split_files = collect(Path(data_yaml))
    total = sum(len(v) for v in split_files.values())
    LOG.info("待检查图片：%d 张（%s）", total, {k: len(v) for k, v in split_files.items()})

    records: List[Tuple[str, Path, int]] = []
    count = 0
    for split, files in split_files.items():
        for f in files:
            if args.max_images and count >= args.max_images:
                break
            h = dhash(f)
            count += 1
            if h is not None:
                records.append((split, f, h))
            if count % 2000 == 0:
                LOG.info("  已计算 %d / %d", count, total)

    LOG.info("计算完成，共 %d 张有效哈希，开始聚类比较", len(records))

    # 按哈希前缀分桶，避免 O(n^2) 全比较
    buckets: Dict[int, List[Tuple[str, Path, int]]] = defaultdict(list)
    for item in records:
        buckets[item[2] >> 16].append(item)  # 高 16 位分桶

    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    index_of = {(s, p): i for i, (s, p, _) in enumerate(records)}
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                if hamming(bucket[i][2], bucket[j][2]) <= args.threshold:
                    union(index_of[(bucket[i][0], bucket[i][1])], index_of[(bucket[j][0], bucket[j][1])])

    clusters: Dict[int, List[Tuple[str, Path, int]]] = defaultdict(list)
    for i, item in enumerate(records):
        clusters[find(i)].append(item)

    dup_clusters = [c for c in clusters.values() if len(c) > 1]
    cross_split = [c for c in dup_clusters if len({s for s, _, _ in c}) > 1]
    same_split = [c for c in dup_clusters if len({s for s, _, _ in c}) == 1]
    leaked_images = sum(len(c) for c in cross_split)

    LOG.info("重复簇 %d 个｜跨划分簇 %d 个（涉及 %d 张）｜同划分簇 %d 个",
             len(dup_clusters), len(cross_split), leaked_images, len(same_split))

    ensure_dir(REPORTS_DIR)
    drop_rows: List[Dict[str, str]] = []
    priority = {"test": 0, "val": 1, "train": 2}
    for cluster in dup_clusters:
        ordered = sorted(cluster, key=lambda x: (priority.get(x[0], 9), str(x[1])))
        keep = ordered[0]
        for split, path, _ in ordered[1:]:
            drop_rows.append({"去掉": str(path), "保留": str(keep[1]), "划分_去掉": split, "划分_保留": keep[0]})

    csv_path = REPORTS_DIR / "dataset_duplicates.csv"
    if drop_rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(drop_rows[0].keys()))
            writer.writeheader()
            writer.writerows(drop_rows)

    lines = [
        "# 数据集去重 / 泄漏检查报告",
        "",
        f"- 数据配置：`{data_yaml}`",
        f"- 检查图片：**{len(records)}** 张",
        f"- 汉明距离阈值：{args.threshold}（64 位 dHash，越小越严格）",
        f"- 重复簇：**{len(dup_clusters)}** 个（涉及 {sum(len(c) for c in dup_clusters)} 张图）",
        f"- **跨划分重复簇：{len(cross_split)} 个，涉及 {leaked_images} 张图**"
        f"{' ⚠️ 存在数据泄漏风险' if cross_split else ' ✅ 未发现泄漏'}",
        f"- 同划分内重复簇：{len(same_split)} 个",
        "",
    ]
    if cross_split:
        lines += ["## 跨划分重复样例（最多 20 组）", "", "| train | val/test |", "| --- | --- |"]
        for cluster in cross_split[:20]:
            by_split: Dict[str, List[str]] = defaultdict(list)
            for split, path, _ in cluster:
                by_split[split].append(Path(path).name)
            lines.append(f"| {', '.join(by_split.get('train', [])[:2])} | "
                         f"{', '.join((by_split.get('val', []) + by_split.get('test', []))[:2])} |")
        lines += ["", "> 建议：按 `reports/dataset_duplicates.csv` 去掉重复图片后重新划分，"
                      "或直接运行 `python scripts/dataset_dedup.py --data data/dataset.yaml --drop` 生成清单。"]
    else:
        lines += ["未发现跨划分的近似重复图片，验证指标可直接使用。", ""]

    report = REPORTS_DIR / "dataset_dedup.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("报告：%s", report)
    if drop_rows:
        LOG.info("剔除清单：%s（共 %d 条）", csv_path, len(drop_rows))
    return 0 if not cross_split else 1


if __name__ == "__main__":
    raise SystemExit(main())
