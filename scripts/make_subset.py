# -*- coding: utf-8 -*-
"""从完整数据集里抽一个「最小可用子集」，用来让成员先把流程跑通。

为什么需要：完整数据 5.3 GB / 9.3 万个文件，非百度网盘会员下载会被限速到无法接受。
而这个子集约 400 MB，可以直接扔群文件/微信，十几分钟拿全。
用途仅限「验证环境与训练链路」，正式指标必须用完整数据集。

用法
----
python scripts/make_subset.py                       # 生成 data/subset + data/subset.yaml
python scripts/make_subset.py --train 3000 --val 800 --test 800
python scripts/make_subset.py --zip                 # 顺便打成 dist/subset_日期.zip

产物
----
data/subset/images|labels/{train,val,test}/...
data/subset.yaml      （path 写成相对路径 data/subset，须在项目根目录执行训练）
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_classes  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_labels(labels_dir: Path) -> Dict[int, List[Path]]:
    """类别 id -> 含该类别的标签文件列表。"""
    per_cls: Dict[int, List[Path]] = defaultdict(list)
    for txt in labels_dir.glob("*.txt"):
        try:
            ids: Set[int] = set()
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line:
                    ids.add(int(line.split()[0]))
        except Exception:
            continue
        for c in ids:
            per_cls[c].append(txt)
    return per_cls


def pick(labels_dir: Path, images_dir: Path, total: int, nc: int, seed: int) -> List[Path]:
    per_cls = scan_labels(labels_dir)
    rng = random.Random(seed)
    for files in per_cls.values():
        rng.shuffle(files)

    per_class = max(1, total // nc)
    chosen: Dict[str, Path] = {}
    for c in range(nc):
        for txt in per_cls.get(c, [])[:per_class]:
            chosen[txt.stem] = txt

    # 补足到 total（按全部标签顺序补，保证不超）
    if len(chosen) < total:
        rest = [p for p in sorted(labels_dir.glob("*.txt")) if p.stem not in chosen]
        rng.shuffle(rest)
        for txt in rest[: total - len(chosen)]:
            chosen[txt.stem] = txt

    out = []
    for stem, txt in chosen.items():
        for ext in IMG_EXT:
            img = images_dir / (stem + ext)
            if img.exists():
                out.append(txt)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="从完整数据集抽最小可用子集")
    ap.add_argument("--src", default="data/processed")
    ap.add_argument("--out", default="data/subset")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=700)
    ap.add_argument("--test", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip", action="store_true", help="顺便打成 dist/subset_日期.zip")
    args = ap.parse_args()

    src = ROOT / args.src if not Path(args.src).is_absolute() else Path(args.src)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    ci = load_classes()
    nc = ci["num_classes"]
    names = [c["en"] for c in ci["classes"]]

    if not (src / "labels" / "train").exists():
        print(f"[错误] 找不到 {src / 'labels' / 'train'}，先把完整数据放好")
        return 1

    if out.exists():
        shutil.rmtree(out)

    stat = {}
    for split, cap in (("train", args.train), ("val", args.val), ("test", args.test)):
        labels_in, images_in = src / "labels" / split, src / "images" / split
        picked = pick(labels_in, images_in, cap, nc, args.seed)

        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)

        n = 0
        for txt in picked:
            shutil.copy2(txt, out / "labels" / split / txt.name)
            for ext in IMG_EXT:
                img = images_in / (txt.stem + ext)
                if img.exists():
                    shutil.copy2(img, out / "images" / split / img.name)
                    n += 1
                    break
        stat[split] = n
        print(f"  {split:<6} 抽了 {n} 张")

    yaml_path = out.parent / "subset.yaml"
    yaml_path.write_text(
        "# 最小可用子集（仅用于跑通流程，正式训练请用 data/dataset.yaml）\n"
        f"path: {args.out.replace(os.sep, '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {nc}\n"
        "names:\n" + "".join(f"- {n}\n" for n in names),
        encoding="utf-8", newline="\n")

    total = sum(stat.values())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print("-" * 58)
    print(f"合计 {total} 张，{size / 1024 ** 2:.0f} MB")
    print(f"数据：{out}")
    print(f"配置：{yaml_path}")

    if args.zip:
        dist = ROOT / "dist"
        dist.mkdir(exist_ok=True)
        zp = dist / f"2026srp_subset_{time.strftime('%Y%m%d')}.zip"
        if zp.exists():
            zp.unlink()
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in out.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT).as_posix())
            z.write(yaml_path, yaml_path.relative_to(ROOT).as_posix())
        print(f"压缩包：{zp}  ({zp.stat().st_size / 1024 ** 2:.0f} MB)")
        print("解压到项目根目录即可（会生成 data/subset 与 data/subset.yaml）")

    print("\n成员用法：")
    print("  python scripts/train.py --data data/subset.yaml --epochs 10 --name subset_try")
    return 0


if __name__ == "__main__":
    sys.exit(main())
