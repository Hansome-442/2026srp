# -*- coding: utf-8 -*-
"""把数据集打成「可分卷上传网盘」的 zip 包，并生成 MD5 校验清单。

为什么需要：data/processed 有 5.4 GB、9 万多个小文件，
直接拖进百度网盘会传到崩溃（小文件速度极慢、还容易失败）。
先打包成 2~3 个分卷，上传快、下载稳、还能校验完整性。

用法
----
# 打包（默认 data/processed -> dist/dataset，每卷 2000 MB）
python scripts/pack_dataset.py

# 自定义
python scripts/pack_dataset.py --src data/processed --out dist/dataset --max-mb 2000

# 校验（成员下载后、解压前先跑这个）
python scripts/pack_dataset.py --verify dist/dataset

产物
----
dist/dataset/
  2026srp_dataset_part01.zip … partNN.zip
  MD5SUMS.txt        每个分卷的 md5
  解压说明.txt        成员照着做就行
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# 这些后缀已经是压缩格式，再压缩只是浪费时间
COMPRESSED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".pt", ".pth"}

READ_ME = """数据集分卷包 —— 解压说明
================================

【里面是什么】
  {name}_part01.zip … part{parts:02d}.zip   共 {parts} 卷，合计约 {size:.1f} GB
  MD5SUMS.txt                              每个分卷的校验值

【怎么校验（强烈建议）】
  把 MD5SUMS.txt 和分卷放在同一目录，然后：
    Windows:  certutil -hashfile {name}_part01.zip MD5
    Linux/Mac: md5sum -c MD5SUMS.txt
  逐个和 MD5SUMS.txt 里的值对比，不一致就重新下载那一卷。

【怎么解压】
  1. 所有分卷放同一个文件夹
  2. 从 part01 开始，依次解压到项目根的 data 目录：
       Windows 右键 part01 → 解压到 data\\
       （每卷解压到同一个 data 目录，提示覆盖就选“是”）
  3. 解压完应该得到：
       data/processed/images/train|val|test/...
       data/processed/labels/train|val|test/...
       data/dataset.yaml
  4. 验证：python scripts/check_env.py  ← 应显示 train 图片数量

【为什么分卷】
  9 万多个小文件直接上传网盘极易中断；分卷后每卷 2 GB 左右，
  断了只重传那一卷。

生成时间：{time}
"""


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def iter_files(src: Path) -> List[Tuple[Path, str, int]]:
    """返回 [(绝对路径, 打包内相对路径, 大小)]，相对路径以 src 的父目录为根。"""
    base = src.parent
    out = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            out.append((p, rel, p.stat().st_size))
    return out


def pack(src: Path, out: Path, max_bytes: int, name: str) -> int:
    files = iter_files(src)
    if not files:
        print(f"[错误] {src} 里没有文件")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.zip"):
        old.unlink()

    total = sum(s for _, _, s in files)
    print(f"源目录：{src}")
    print(f"文件数：{len(files)} 个，合计 {human(total)}")
    print(f"每卷上限：{human(max_bytes)}")

    sums: List[Tuple[str, str]] = []
    part, used, written = 1, 0, 0

    def new_zip(idx: int) -> Tuple[zipfile.ZipFile, Path]:
        path = out / f"{name}_part{idx:02d}.zip"
        return zipfile.ZipFile(path, "w", zipfile.ZIP_STORED), path

    zf, path = new_zip(part)
    md5 = hashlib.md5()
    t0 = time.time()

    for abs_path, rel, size in files:
        if used + size > max_bytes and used > 0:
            zf.close()
            print(f"  ✅ {path.name:<40} {human(path.stat().st_size):>10}")
            part += 1
            zf, path = new_zip(part)
            used = 0

        zf.write(abs_path, rel)
        used += size
        written += 1
        if written % 5000 == 0:
            print(f"     …已打包 {written}/{len(files)} 个文件（{human(used)}/{human(total)}）")

    # 把数据集版本文件也打进去：解压到 data/ 后版本号自动对齐，
    # 否则新人从全量包解压出来没有版本文件，fingerprint 会重新从 v1 开始算。
    ver_file = ROOT / "data" / "DATASET_VERSION.txt"
    if ver_file.exists() and src.name == "processed":
        zf.write(ver_file, "DATASET_VERSION.txt")
        print("  已附带 DATASET_VERSION.txt")

    zf.close()
    print(f"  ✅ {path.name:<40} {human(path.stat().st_size):>10}")

    # 分卷 md5 必须按最终落盘文件重算
    print("  计算分卷校验值…")
    sums = []
    for zpath in sorted(out.glob("*.zip")):
        h = hashlib.md5()
        with open(zpath, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
        sums.append((zpath.name, h.hexdigest()))

    (out / "MD5SUMS.txt").write_text(
        "\n".join(f"{d}  {n}" for n, d in sums) + "\n", encoding="utf-8", newline="\n")
    (out / "解压说明.txt").write_text(
        READ_ME.format(name=name, parts=len(sums), size=total / 1024 ** 3,
                       time=time.strftime("%Y-%m-%d %H:%M")),
        encoding="utf-8", newline="\n")

    print("-" * 60)
    print(f"完成：{len(sums)} 个分卷，{human(total)}，耗时 {time.time() - t0:.0f}s")
    print(f"输出目录：{out}")
    print("把整个目录上传百度网盘即可（含 MD5SUMS.txt 与 解压说明.txt）")
    return 0


MERGE_LOG = ROOT / "data" / "_merge_log.txt"


def pack_increment(out: Path) -> int:
    """只打包 data/_merge_log.txt 里记录的新增文件（merge_new_data.py 写入）。"""
    if not MERGE_LOG.exists():
        print("[提示] 没有待打包的新增文件。")
        print("       先合并新数据：python scripts/merge_new_data.py --src <目录>")
        return 1

    rels = [l.strip() for l in MERGE_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rels:
        print("[提示] _merge_log.txt 是空的，没有新增文件需要打包。")
        return 1

    data_dir = ROOT / "data"
    entries, missing = [], []
    for rel in rels:
        p = data_dir / rel
        (entries if p.is_file() else missing).append(p if p.is_file() else rel)
    if missing:
        print(f"[警告] {len(missing)} 个文件已不存在（可能被删了），跳过：")
        for m in missing[:5]:
            print("   ", m)

    if not entries:
        print("[错误] 记录的文件都不存在了")
        return 1

    total = sum(p.stat().st_size for p in entries)
    ver_file = data_dir / "DATASET_VERSION.txt"
    ver = "?"
    if ver_file.exists():
        for line in ver_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("version"):
                ver = line.split(":", 1)[1].strip()

    out.mkdir(parents=True, exist_ok=True)
    zp = out / f"2026srp_dataset_update_{ver}.zip"
    if zp.exists():
        zp.unlink()

    print(f"增量打包：{len(entries)} 个文件，合计 {human(total)}")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for p in entries:
            z.write(p, p.relative_to(data_dir).as_posix())
        # 把新的版本文件一起带上，解压后版本号自动对齐
        if ver_file.exists():
            z.write(ver_file, ver_file.relative_to(data_dir).as_posix())
        note = (
            f"数据集增量包 {ver}\n"
            "=========================\n\n"
            "【装法】\n"
            "  解压到项目根的 data 目录，例如：\n"
            "     D:\\2026srp\\data\\\n"
            "  会覆盖同名文件，不会动到没变的部分。\n\n"
            "【必须按顺序】\n"
            "  增量包只能依次应用：v1→v2→v3，不能跳过中间版本。\n"
            "  不确定自己是什么版本：\n"
            "     python scripts/dataset_fingerprint.py --check\n\n"
            "【懒人办法】\n"
            "  直接下最新的全量包，一次到位。\n"
        )
        z.writestr("增量说明.txt", note.encode("utf-8"))

    print(f"  ✅ {zp.name}  {human(zp.stat().st_size)}")
    MERGE_LOG.write_text("", encoding="utf-8")
    print("  已清空 _merge_log.txt（下次合并会重新记录）")
    print("\n下一步：把增量包传网盘，群里说明「只下这个，解压到 data 目录」。")
    return 0


def verify(folder: Path) -> int:
    sums_file = folder / "MD5SUMS.txt"
    if not sums_file.exists():
        print(f"[错误] 找不到 {sums_file}")
        return 1
    bad = 0
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        p = folder / name
        if not p.exists():
            print(f"❌ 缺失 {name}")
            bad += 1
            continue
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == digest:
            print(f"✅ {name}")
        else:
            print(f"❌ {name} 校验不符（重新下载这一卷）")
            bad += 1
    print("-" * 60)
    print("全部通过" if bad == 0 else f"{bad} 个分卷有问题")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集分卷打包 / 校验")
    ap.add_argument("--src", default="data/processed")
    ap.add_argument("--out", default="dist/dataset")
    ap.add_argument("--max-mb", type=int, default=2000, help="每个分卷上限 MB")
    ap.add_argument("--name", default="2026srp_dataset_" + time.strftime("%Y%m%d"))
    ap.add_argument("--verify", metavar="DIR", default=None, help="校验已有的分卷目录")
    ap.add_argument("--only-new", action="store_true", help="只打包 merge_new_data.py 记录的新增文件")
    args = ap.parse_args()

    if args.verify:
        return verify(Path(args.verify))

    if args.only_new:
        return pack_increment(Path(args.out) if args.out else ROOT / 'dist' / 'dataset')

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if not src.is_dir():
        print(f"[错误] 源目录不存在：{src}")
        return 1
    return pack(src, out, args.max_mb * 1024 * 1024, args.name)


if __name__ == "__main__":
    sys.exit(main())
