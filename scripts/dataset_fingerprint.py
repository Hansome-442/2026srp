# -*- coding: utf-8 -*-
"""给数据集打指纹，解决「三台机器训练、数据到底一不一样」的问题。

数据集 5.3 GB 不进 git，谁补了标注、谁删了图，光看目录根本看不出来。
这个脚本把数据集的关键信息压成一个版本号 + 指纹，两边一对就知道是否同一份。

用法
----
python scripts/dataset_fingerprint.py                # 计算并写入 data/DATASET_VERSION.txt
python scripts/dataset_fingerprint.py --check        # 只比对，不写入（不一致时退出码 1）
python scripts/dataset_fingerprint.py --bump         # 强制升版本号（例如 v3 -> v4）

输出 data/DATASET_VERSION.txt 内容示例
----
version  : v3
images   : 46760
labels   : 46763
boxes    : 449209
fingerprint: 8f3c2a1d9e   （由每张图的「文件名+大小」和每个标签的「行数」聚合而来）
updated  : 2026-11-17 21:40
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
VERSION_FILE = DATA / "DATASET_VERSION.txt"


def collect(src: Path, count_boxes: bool = False):
    """统计图片/标签（以及可选框数），并算一个稳定的指纹。

    性能：默认只读目录项和文件大小，4.6 万张图约 20 秒。
    加 --boxes 才会逐个打开标签文件数行数（约 7 分钟），只在出正式报告时用。
    """
    images = labels = boxes = 0
    h = hashlib.sha1()
    img_dir, lbl_dir = src / "images", src / "labels"
    if not img_dir.exists():
        return None

    for base, kind in ((img_dir, "img"), (lbl_dir, "lbl")):
        if not base.exists():
            continue
        for split in sorted(os.listdir(base)):
            d = base / split
            if not d.is_dir():
                continue
            with os.scandir(d) as it:
                for e in it:
                    if not e.is_file():
                        continue
                    if kind == "img":
                        images += 1
                    else:
                        if not e.name.endswith(".txt"):
                            continue
                        labels += 1
                        if count_boxes:
                            try:
                                with open(e.path, encoding="utf-8", errors="ignore") as fh:
                                    boxes += sum(1 for line in fh if line.strip())
                            except Exception:
                                pass
                    st = e.stat()
                    h.update(e.name.encode("utf-8", "ignore"))
                    h.update(str(st.st_size).encode())

    return {"images": images, "labels": labels, "boxes": boxes, "fingerprint": h.hexdigest()[:12]}


def read_version() -> dict:
    if not VERSION_FILE.exists():
        return {}
    out = {}
    for line in VERSION_FILE.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().split("（")[0].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集指纹 / 版本")
    ap.add_argument("--src", default=str(PROCESSED))
    ap.add_argument("--check", action="store_true", help="只比对，不写入")
    ap.add_argument("--bump", action="store_true", help="强制升版本号")
    ap.add_argument("--boxes", action="store_true", help="同时统计标注框数（慢，约 7 分钟）")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[错误] 没有数据集目录：{src}")
        return 2

    print(f"正在扫描 {src} …（4.6 万张图，约 30 秒）")
    t0 = time.time()
    cur = collect(src, count_boxes=args.boxes)
    if cur is None:
        print("[错误] 目录结构不对，应该有 images/ 和 labels/")
        return 2
    print(f"扫描完成，用时 {time.time() - t0:.0f}s")

    old = read_version()
    same = old.get("fingerprint") == cur["fingerprint"]

    print("-" * 56)
    print(f"  图片      : {cur['images']}")
    print(f"  标签文件  : {cur['labels']}")
    print(f"  标注框    : {cur['boxes']}")
    print(f"  指纹      : {cur['fingerprint']}")
    if old:
        print(f"  已记录版本 : {old.get('version', '?')} / 指纹 {old.get('fingerprint', '?')}")
        print(f"  对比结果   : {'一致 ✅' if same else '不一致 ❗ 数据集被改动过'}")

    if args.check:
        return 0 if same else 1

    # 决定版本号：指纹一致就沿用；不一致则升一版
    ver = old.get("version", "v0")
    if not same or args.bump:
        m = re.match(r"v(\d+)$", ver or "v0")
        ver = "v%d" % ((int(m.group(1)) if m else 0) + 1)

    text = (
        f"version    : {ver}\n"
        f"images     : {cur['images']}\n"
        f"labels     : {cur['labels']}\n"
        f"boxes      : {cur['boxes'] if args.boxes else '未统计'}\n"
        f"fingerprint: {cur['fingerprint']}\n"
        f"updated    : {time.strftime('%Y-%m-%d %H:%M')}\n"
    )
    VERSION_FILE.write_text(text, encoding="utf-8", newline="\n")
    print("-" * 56)
    print(f"已写入 {VERSION_FILE}")
    print(f"数据集版本：{ver}")
    print("\n训练报告里请写明这个版本号，别人才能确认你的结果基于哪份数据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
