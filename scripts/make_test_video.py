"""从数据集中找一段连续抽帧，合成一个可直接用于「本地视频检测」的 mp4。"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, r"E:\2026srp")
from src.utils import imread

SRC = Path(r"E:\2026srp\data\processed\images\train")
OUT = Path(r"E:\2026srp\test_images\课堂测试视频.mp4")
TARGET_FRAMES = 240
FPS = 12

# 解析文件名：<前缀>_<编号>.jpg
groups = defaultdict(list)
for p in SRC.glob("*.jpg"):
    m = re.match(r"^(.*?)_?(\d{4,})\.jpg$", p.name)
    if not m:
        continue
    groups[m.group(1)].append((int(m.group(2)), p))

best = None
for prefix, items in groups.items():
    items.sort()
    run = [items[0]]
    for prev, cur in zip(items, items[1:]):
        if cur[0] == prev[0] + 1:
            run.append(cur)
        else:
            if best is None or len(run) > len(best[1]):
                best = (prefix, run)
            run = [cur]
    if best is None or len(run) > len(best[1]):
        best = (prefix, run)

if not best:
    print("没找到连续帧序列")
    raise SystemExit(1)

prefix, run = best
print(f"最长连续序列: {prefix}  共 {len(run)} 帧  起止 {run[0][0]}~{run[-1][0]}")

frames = run[:TARGET_FRAMES]
first = imread(frames[0][1])
h, w = first.shape[:2]
# 缩放到 960 宽，便于播放
scale = 960 / w
size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
count = 0
for _, path in frames:
    img = imread(path)
    if img is None:
        continue
    writer.write(cv2.resize(img, size))
    count += 1
writer.release()
print(f"已生成 {OUT}  帧数 {count}  尺寸 {size}  {FPS}fps  时长 {count/FPS:.1f}s")
