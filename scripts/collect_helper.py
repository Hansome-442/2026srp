"""数据采集 / 标注辅助工具（7 人分工协作使用）。

子命令
------
frames  从课堂视频抽帧成待标注图片（自动过滤模糊帧和近乎重复的帧）
assign  把待标注图片平均分给 7 位同学，生成每人一份的任务清单（CSV）
check   校验标注文件合法性（类别 id、坐标范围、空文件、越界框）
merge   把多人标注好的结果合并成一个 YOLO 数据集目录

示例
----
python scripts/collect_helper.py frames --video data/videos --out data/to_label --stride 30 --scene 教室后排 --stage 大学
python scripts/collect_helper.py assign --src data/to_label --out data/assignments --members 7
python scripts/collect_helper.py check  --labels data/to_label/labels
python scripts/collect_helper.py merge  --src data/to_label/labels_checked --images data/to_label --out data/raw/self_collected
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_DIR, REPORTS_DIR, ensure_dir, get_logger, imread, imwrite, load_classes  # noqa: E402

LOG = get_logger("collect")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


# --------------------------------------------------------------------------- #
def cmd_frames(args: argparse.Namespace) -> int:
    import cv2

    src = Path(args.video)
    out_dir = ensure_dir(Path(args.out) / f"{args.scene}_{args.stage}")
    videos: List[Path] = []
    if src.is_dir():
        videos = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in VIDEO_EXTS]
    elif src.suffix.lower() in VIDEO_EXTS:
        videos = [src]
    if not videos:
        LOG.error("没找到视频文件：%s", src)
        return 2

    saved = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        LOG.info("处理 %s（%d 帧，%.1f fps）", video.name, total, fps)
        idx = 0
        kept_for_video = 0
        prev_small = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                small = cv2.resize(gray, (32, 32))
                diff = 999.0
                if prev_small is not None:
                    diff = float(abs(small.astype("int16") - prev_small.astype("int16")).mean())
                if sharpness >= args.min_sharpness and diff >= args.min_diff:
                    name = f"{video.stem}_{idx:06d}.jpg"
                    imwrite(out_dir / name, frame)
                    saved += 1
                    kept_for_video += 1
                    prev_small = small
                if args.max_per_video and kept_for_video >= args.max_per_video:
                    break
            idx += 1
        cap.release()
        LOG.info("  -> 保存 %d 张", kept_for_video)
    LOG.info("抽帧完成：共 %d 张，输出目录 %s", saved, out_dir)
    LOG.info("下一步：用 labelImg 标注，类别名请使用 configs/classes.yaml 中的中文类别名")
    return 0


# --------------------------------------------------------------------------- #
def cmd_assign(args: argparse.Namespace) -> int:
    src = Path(args.src)
    images = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        LOG.error("没找到待标注图片：%s", src)
        return 2
    out_dir = ensure_dir(args.out)
    random.Random(args.seed).shuffle(images)

    members = [f"成员{i + 1}" for i in range(args.members)]
    buckets: Dict[str, List[Path]] = defaultdict(list)
    for i, img in enumerate(images):
        buckets[members[i % args.members]].append(img)

    lines = ["# 数据采集/标注分工表", "", f"待标注图片总数：**{len(images)}**，共 {args.members} 位成员", ""]
    lines += ["| 成员 | 任务张数 | 清单文件 |", "| --- | --- | --- |"]
    for member in members:
        items = buckets[member]
        csv_path = out_dir / f"{member}_任务清单.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "图片路径", "是否已标注", "备注"])
            for j, p in enumerate(items, 1):
                writer.writerow([j, str(p), "", ""])
        lines.append(f"| {member} | {len(items)} | {csv_path.name} |")
        if args.copy:
            dst = ensure_dir(out_dir / member)
            for p in items:
                shutil.copy2(p, dst / p.name)
    (out_dir / "分工表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("分工完成，输出目录：%s", out_dir)
    for line in lines[4:]:
        print(line)
    return 0


# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    class_info = load_classes(args.classes)
    nc = class_info["num_classes"]
    zh = class_info["zh"]

    if args.labels:
        label_files = [p for p in Path(args.labels).rglob("*.txt") if p.name != "classes.txt"]
    else:
        label_files = [p for p in Path(args.src).rglob("*.txt") if p.name != "classes.txt"]

    problems: List[str] = []
    cls_counter: Counter = Counter()
    empty = 0
    boxes = 0
    per_file: Dict[str, int] = {}
    for path in label_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            empty += 1
            continue
        file_boxes = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) < 5:
                problems.append(f"{path.name}:{lineno} 字段不足 5 个：{line}")
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                problems.append(f"{path.name}:{lineno} 数值解析失败：{line}")
                continue
            if not (0 <= cid < nc):
                problems.append(f"{path.name}:{lineno} 类别 id 越界：{cid}（应在 0~{nc - 1}）")
                continue
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                problems.append(f"{path.name}:{lineno} 坐标不在 0~1：{line}")
                continue
            if bw <= 0.002 or bh <= 0.002:
                problems.append(f"{path.name}:{lineno} 框过小（可能误标）：{line}")
                continue
            cls_counter[zh[cid]] += 1
            file_boxes += 1
            boxes += 1
        per_file[path.name] = file_boxes

    lines = [
        "# 标注校验报告",
        "",
        f"- 标注文件数：{len(label_files)}（空文件 {empty} 个）",
        f"- 有效标注框：{boxes}",
        f"- 问题条数：{len(problems)}",
        "",
        "## 类别分布",
        "",
        "| 类别 | 实例数 |",
        "| --- | --- |",
    ]
    for name in zh:
        lines.append(f"| {name} | {cls_counter.get(name, 0)} |")
    if problems:
        lines += ["", "## 问题明细（最多显示 100 条）", ""]
        lines += [f"- {p}" for p in problems[:100]]
    ensure_dir(REPORTS_DIR)
    out = REPORTS_DIR / "label_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("校验完成：%d 个文件，%d 个框，%d 个问题 -> %s", len(label_files), boxes, len(problems), out)
    if args.fail_on_problem and problems:
        return 1
    return 0


# --------------------------------------------------------------------------- #
def cmd_merge(args: argparse.Namespace) -> int:
    src_dirs = [Path(p) for p in args.src]
    out_dir = ensure_dir(args.out)
    img_out = ensure_dir(out_dir / "images")
    lbl_out = ensure_dir(out_dir / "labels")
    copied = 0
    for src in src_dirs:
        if not src.exists():
            LOG.warning("目录不存在，跳过：%s", src)
            continue
        for img in src.rglob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            label = None
            for cand in [img.with_suffix(".txt"), src / "labels" / f"{img.stem}.txt", img.parent.parent / "labels" / f"{img.stem}.txt"]:
                if cand.exists():
                    label = cand
                    break
            if label is None:
                LOG.warning("缺少标注，跳过：%s", img.name)
                continue
            name = img.name if not (img_out / img.name).exists() else f"{img.parent.name}_{img.name}"
            shutil.copy2(img, img_out / name)
            shutil.copy2(label, lbl_out / f"{Path(name).stem}.txt")
            copied += 1
    LOG.info("合并完成：%d 组「图片+标注」-> %s", copied, out_dir)
    LOG.info("接下来运行：python scripts/prepare_dataset.py --yolo-dir %s", out_dir)
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="课堂行为数据采集/标注辅助工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("frames", help="从视频抽帧")
    p1.add_argument("--video", required=True, help="视频文件或目录")
    p1.add_argument("--out", default=str(DATA_DIR / "to_label"))
    p1.add_argument("--stride", type=int, default=30, help="每 N 帧取一帧")
    p1.add_argument("--scene", default="教室后排", help="场景名（写入目录名）")
    p1.add_argument("--stage", default="大学", help="学段（写入目录名）")
    p1.add_argument("--min-sharpness", type=float, default=60.0, help="拉普拉斯方差阈值，越小越宽松")
    p1.add_argument("--min-diff", type=float, default=2.0, help="与上一张的差异阈值，用于去重")
    p1.add_argument("--max-per-video", type=int, default=0, help="每个视频最多保存多少张（0=不限）")
    p1.set_defaults(func=cmd_frames)

    p2 = sub.add_parser("assign", help="把待标注图片分给 N 位成员")
    p2.add_argument("--src", required=True)
    p2.add_argument("--out", default=str(DATA_DIR / "assignments"))
    p2.add_argument("--members", type=int, default=7)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--copy", action="store_true", help="同时把图片复制到每人目录")
    p2.set_defaults(func=cmd_assign)

    p3 = sub.add_parser("check", help="校验标注文件")
    p3.add_argument("--labels", default=None)
    p3.add_argument("--src", default=str(DATA_DIR / "to_label"))
    p3.add_argument("--classes", default=None)
    p3.add_argument("--fail-on-problem", action="store_true")
    p3.set_defaults(func=cmd_check)

    p4 = sub.add_parser("merge", help="合并多人标注结果")
    p4.add_argument("--src", nargs="+", required=True)
    p4.add_argument("--out", default=str(DATA_DIR / "raw" / "self_collected"))
    p4.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
