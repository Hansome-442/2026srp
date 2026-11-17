# -*- coding: utf-8 -*-
"""把组员交上来的新样本合并进训练集。

为什么需要：手动往 data/processed/images/train 里拖文件，很容易
漏标签、图片重名被覆盖、或者误加到 val/test 里导致指标不可比。

这个脚本做的事：
  1. 校验每个图片都有对应的 .txt 标签
  2. 只往 **train** 里加（val/test 保持不变，这样历史指标仍然可比）
  3. 重名自动跳过并报出来，不覆盖已有数据
  4. 合并完提示你去跑 dataset_fingerprint.py 升版本

用法
----
python scripts/merge_new_data.py --src data/incoming/1125_小王_趴桌
python scripts/merge_new_data.py --src <目录> --dry-run     # 只看会怎么合，不实际拷贝

交上来的目录结构（两种都支持）：
  A)  <目录>/images/*.jpg  +  <目录>/labels/*.txt     （YOLO 标准格式）
  B)  <目录>/*.jpg         +  <目录>/*.txt            （平铺）
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def locate(src: Path):
    """返回 (图片目录, 标签目录)。"""
    if (src / "images").is_dir() and (src / "labels").is_dir():
        return src / "images", src / "labels"
    return src, src


def main() -> int:
    ap = argparse.ArgumentParser(description="把新样本合并进训练集")
    ap.add_argument("--src", required=True, help="交上来的样本目录")
    ap.add_argument("--split", default="train", choices=["train"], help="目前只允许合进 train")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不实际拷贝")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    if not src.is_dir():
        print(f"[错误] 目录不存在：{src}")
        return 2

    img_dir, lbl_dir = locate(src)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        print(f"[错误] {img_dir} 里没有图片")
        return 2

    dst_img = PROCESSED / "images" / args.split
    dst_lbl = PROCESSED / "labels" / args.split
    if not dst_img.is_dir():
        print(f"[错误] 找不到 {dst_img}，请确认数据集已经准备好")
        return 2
    dst_lbl.mkdir(parents=True, exist_ok=True)

    # 已有文件名（防止重名覆盖）
    existing = {p.stem for p in dst_img.iterdir() if p.is_file()}

    ok, no_label, dup, skipped = [], [], [], []
    for p in imgs:
        stem = p.stem
        lbl = lbl_dir / (stem + ".txt")
        if stem in existing:
            dup.append(p.name)
            continue
        if not lbl.exists():
            no_label.append(p.name)
            continue
        # 标签必须是非空的合法 YOLO 行
        try:
            lines = [l for l in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        except Exception:
            lines = []
        if not lines:
            skipped.append(p.name + "（标签为空）")
            continue
        bad = [l for l in lines if len(l.split()) < 5]
        if bad:
            skipped.append(p.name + "（标签格式不对）")
            continue
        ok.append((p, lbl))

    print("=" * 60)
    print(f"来源：{src}")
    print(f"图片总数：{len(imgs)}")
    print(f"  可合并    ：{len(ok)}")
    print(f"  缺标签    ：{len(no_label)}")
    print(f"  重名跳过  ：{len(dup)}")
    print(f"  标签有问题：{len(skipped)}")

    for name, items in (("缺标签", no_label), ("重名", dup), ("标签异常", skipped)):
        if items:
            print(f"\n【{name}】前 5 个：")
            for x in items[:5]:
                print("   ", x)

    if args.dry_run:
        print("\n--dry-run：没有实际拷贝")
        return 0

    for p, lbl in ok:
        shutil.copy2(p, dst_img / p.name)
        shutil.copy2(lbl, dst_lbl / lbl.name)

    print("\n" + "=" * 60)
    print(f"已合并 {len(ok)} 组到 {dst_img.parent.parent}（split={args.split}）")
    print("\n下一步必须做：")
    print("   python scripts/dataset_fingerprint.py      # 升版本号")
    print("   python scripts/pack_dataset.py             # 重打包传网盘")
    if no_label or skipped:
        print("\n⚠️ 上面被跳过的那几张，记得让组员补标签后再交一次。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
