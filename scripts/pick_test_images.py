"""挑选最合适的演示图片：从验证集采样 -> 推理 -> 按"检出数量+类别丰富度"排序 -> 复制到 test_images。"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"E:\2026srp")
from src.detector import BehaviorDetector
from src.utils import imread, imwrite, load_app_config, load_classes

ROOT = Path(r"E:\2026srp")
VAL = ROOT / "data/processed/images/val"
OUT = ROOT / "test_images"
OUT.mkdir(exist_ok=True)

cfg = load_app_config()
ci = load_classes()
det = BehaviorDetector(
    weights="runs/best.pt", device="0", imgsz=640,
    conf=cfg["model"]["conf"], class_names=ci["zh"],
    per_class_conf=cfg["model"].get("per_class_conf"),
)
det.load()

# 按来源前缀采样
groups = {
    "课堂全景(中小学)": "student_classroom_behavior_",
    "手机/睡觉(海外教室)": "yolov9_classroom_",
    "高校课堂": "0.671k_university_yolo_Dataset_",
    "谈话/站立(SCB5)": "SCB5-Talk",
}
cands = []
for tag, prefix in groups.items():
    files = sorted([p for p in VAL.glob("*.jpg") if p.name.startswith(prefix)])[:25]
    print(f"{tag}: 候选 {len(files)} 张")
    for f in files:
        img = imread(f)
        if img is None:
            continue
        dets = det.predict(img)
        if not dets:
            continue
        cnt = Counter(d.cls_name for d in dets)
        score = len(dets) + 3 * len(cnt)
        cands.append((score, len(dets), len(cnt), dict(cnt), tag, f, img, dets))

cands.sort(key=lambda x: -x[0])
print(f"\n有效候选 {len(cands)} 张，Top 6：")
for i, (score, n, k, cnt, tag, f, img, dets) in enumerate(cands[:6], 1):
    print(f"  {i}. {f.name}  检出 {n} 个 / {k} 类 {cnt}  [{tag}]  {img.shape[1]}x{img.shape[0]}")

# 每个来源挑 1 张最好的，保证覆盖不同场景
best_by_tag = {}
for item in cands:
    tag = item[4]
    if tag not in best_by_tag or item[0] > best_by_tag[tag][0]:
        best_by_tag[tag] = item
picked = [best_by_tag[t] for t in groups if t in best_by_tag]

print("\n导出：")
for i, (score, n, k, cnt, tag, f, img, dets) in enumerate(picked, 1):
    safe_tag = tag.replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "")
    src_dst = OUT / f"测试图片{i}_{safe_tag}.jpg"
    shutil.copy2(f, src_dst)
    annotated = det.annotate(img.copy(), dets, colors=ci["colors"])
    preview = OUT / f"测试图片{i}_{safe_tag}_带框预览.jpg"
    imwrite(preview, annotated)
    print(f"  {src_dst.name}  ({img.shape[1]}x{img.shape[0]})  检出 {cnt}")
    print(f"    带框预览: {preview.name}")
