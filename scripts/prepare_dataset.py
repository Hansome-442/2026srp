"""数据集转换与划分脚本：VOC(XML) / YOLO(txt) -> 标准 YOLO 训练集。

用法示例
--------
# 1) 课程提供的 VOC 格式数据集（Annotations + JPEGImages）
python scripts/prepare_dataset.py --voc-dir E:/yolov11/VOCdevkit/VOC2007

# 2) 已经是 YOLO 格式（images/ + labels/ + classes.txt）
python scripts/prepare_dataset.py --yolo-dir data/raw/yolo_dataset

# 3) 混合/增量：把新标好的数据合并进来后重新划分
python scripts/prepare_dataset.py --voc-dir data/raw/voc_all --oversample-min 300

输出
----
data/processed/images/{train,val,test}/*.jpg
data/processed/labels/{train,val,test}/*.txt
data/dataset.yaml          <- 训练脚本用的配置文件
data/processed/manifest.csv
reports/dataset_stats.md / reports/dataset_stats.json
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (  # noqa: E402
    DATA_DIR,
    REPORTS_DIR,
    dhash,
    dump_json,
    dump_yaml,
    ensure_dir,
    get_logger,
    hamming,
    imread,
    load_classes,
    normalize_label,
)

LOG = get_logger("prepare")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# 读取原始标注
# --------------------------------------------------------------------------- #
def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[str, float, float, float, float]]]:
    """解析 VOC XML，返回 (宽, 高, [(类别名, x1, y1, x2, y2), ...])。"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    objects: List[Tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        try:
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append((name, x1, y1, x2, y2))
    return width, height, objects


def collect_voc_samples(voc_dir: Path) -> List[Dict[str, Any]]:
    """扫描 VOC 目录（支持嵌套），返回样本列表。"""
    img_dirs = [voc_dir / "JPEGImages", voc_dir / "images", voc_dir]
    ann_dirs = [voc_dir / "Annotations", voc_dir / "annotations"]
    images: Dict[str, Path] = {}
    for d in img_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                images.setdefault(p.stem, p)
    annotations: Dict[str, Path] = {}
    for d in ann_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.xml"):
            annotations.setdefault(p.stem, p)
    if not annotations and voc_dir.exists():
        for p in voc_dir.rglob("*.xml"):
            annotations.setdefault(p.stem, p)

    samples: List[Dict[str, Any]] = []
    for stem, xml_path in sorted(annotations.items()):
        img_path = images.get(stem)
        if img_path is None:
            continue
        group = _guess_group(img_path, voc_dir)
        samples.append({"image": img_path, "annotation": xml_path, "kind": "voc", "group": group})
    return samples


def collect_yolo_samples(yolo_dir: Path) -> List[Dict[str, Any]]:
    """扫描 YOLO 目录（images/ + labels/），返回样本列表。"""
    img_root = None
    for cand in [yolo_dir / "images", yolo_dir / "JPEGImages", yolo_dir]:
        if cand.exists():
            img_root = cand
            break
    if img_root is None:
        return []
    samples: List[Dict[str, Any]] = []
    for img in sorted(img_root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(img_root)
        label = None
        for cand in [
            yolo_dir / "labels" / rel.with_suffix(".txt"),           # <root>/images/x.jpg -> <root>/labels/x.txt
            img.parent.parent / "labels" / f"{img.stem}.txt",       # <root>/<split>/images/x.jpg -> <root>/<split>/labels/x.txt（Roboflow 布局）
            img_root.parent / "labels" / rel.with_suffix(".txt"),
            img.with_suffix(".txt"),
            img.parent / f"{img.stem}.txt",
        ]:
            if cand.exists():
                label = cand
                break
        samples.append(
            {"image": img, "annotation": label, "kind": "yolo", "group": _guess_group(img, img_root)}
        )
    return samples


def _guess_group(path: Path, root: Path) -> str:
    """用相对路径的第一级目录当作「场景/学段」分组，用于统计覆盖度。"""
    try:
        rel = path.relative_to(root)
        parts = rel.parts[:-1]
        return parts[0] if parts else "默认"
    except ValueError:
        return "默认"


# --------------------------------------------------------------------------- #
# 类别映射
# --------------------------------------------------------------------------- #
def build_name_to_id(class_info: Dict[str, Any]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for text, cls_id in class_info["alias_zh"].items():
        mapping[text] = cls_id
    return mapping


def map_class(name: str, name_to_id: Dict[str, int]) -> Optional[int]:
    return name_to_id.get(normalize_label(name))


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def split_samples(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """按「主类别」分层划分，保证稀有类在三份里都出现。"""
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        buckets[s.get("dominant", "__none__")].append(s)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:  # 小类至少保证 train/val 各一条
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        n_test = n - n_train - n_val
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train : n_train + n_val]
        splits["test"] += items[n_train + n_val : n_train + n_val + n_test]
    for key in splits:
        rng.shuffle(splits[key])
    return splits


# --------------------------------------------------------------------------- #
# 类别映射（支持「源类别名」映射，用于 Kaggle/Roboflow 等外部数据集）
# --------------------------------------------------------------------------- #
def resolve_yolo_cls_id(raw_id: int, src_names: Optional[Sequence[str]],
                        class_info: Dict[str, Any], name_to_id: Dict[str, int]):
    """把源数据集的类别 id 映射成本项目的 0~7。

    * 给了 src_names：按「源 id -> 源类别名 -> 本项目类别」映射，返回 (本项目id, 源类别名或None)
    * 没给 src_names：认为源 id 就是本项目 id（要求 0~nc-1 合法）
    """
    if src_names:
        if not (0 <= raw_id < len(src_names)):
            return None, f"id_{raw_id}"
        src_name = str(src_names[raw_id])
        cls_id = map_class(src_name, name_to_id)
        if cls_id is None:
            return None, src_name
        return cls_id, src_name
    if not (0 <= raw_id < class_info["num_classes"]):
        return None, f"id_{raw_id}"
    return raw_id, None


def mapped_class_names(sample: Dict[str, Any], class_info: Dict[str, Any],
                       name_to_id: Dict[str, int]) -> List[str]:
    """返回某样本里所有能映射到本项目类别的中文类别名（用于分层划分）。"""
    names: List[str] = []
    if sample["kind"] == "voc":
        _, _, objects = parse_voc_xml(sample["annotation"])
        for raw_name, *_ in objects:
            cid = map_class(raw_name, name_to_id)
            if cid is not None:
                names.append(class_info["zh"][cid])
        return names

    label_path = sample.get("annotation")
    if not label_path or not Path(label_path).exists():
        return names
    src_names = sample.get("yolo_names")
    for line in Path(label_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            raw_id = int(float(parts[0]))
        except ValueError:
            continue
        cid, _ = resolve_yolo_cls_id(raw_id, src_names, class_info, name_to_id)
        if cid is not None:
            names.append(class_info["zh"][cid])
    return names


# --------------------------------------------------------------------------- #
# 感知哈希去重：把「同一视频的相邻帧 / 重复上传图」聚成一组，整组进同一个划分
# --------------------------------------------------------------------------- #
def assign_dup_groups(samples: Sequence[Dict[str, Any]], threshold: int = 6) -> Dict[str, Any]:
    """计算 dHash 并用并查集聚类，把结果写进 sample["group_id"]。

    返回统计信息：{'groups': 组数, 'dup_images': 重复图片数, 'singletons': 单图组数}
    """
    import numpy as np

    LOG.info("计算感知哈希（%d 张）…", len(samples))
    hashes: List[Optional[int]] = []
    for i, s in enumerate(samples):
        hashes.append(dhash(s["image"]))
        if (i + 1) % 2000 == 0:
            LOG.info("  哈希 %d/%d", i + 1, len(samples))

    parent = list(range(len(samples)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    valid = [(i, h) for i, h in enumerate(hashes) if h is not None]
    if valid:
        popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        arr = np.array([h for _, h in valid], dtype=np.uint64)
        idx_map = [i for i, _ in valid]
        LOG.info("近似重复聚类（%d 张，阈值 %d）…", len(arr), threshold)
        for a in range(len(arr)):
            diff = arr ^ arr[a]
            bits = popcnt[diff.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for b in np.nonzero(bits <= threshold)[0]:
                if b > a:
                    union(idx_map[a], idx_map[int(b)])

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(samples)):
        groups[find(i)].append(i)
    for root, members in groups.items():
        for i in members:
            samples[i]["group_id"] = root

    dup_images = sum(len(m) for m in groups.values() if len(m) > 1)
    return {
        "groups": len(groups),
        "dup_images": dup_images,
        "singletons": sum(1 for m in groups.values() if len(m) == 1),
        "largest_group": max((len(m) for m in groups.values()), default=1),
    }


def split_samples_grouped(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """按「重复组」划分：同一组的图片只能进入同一个划分，杜绝数据泄漏。

    组内主类别用于分层，保证稀有类在三份里都出现。
    """
    rng = random.Random(seed)
    by_group: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_group[int(s.get("group_id", id(s)))].append(s)

    group_dom: Dict[int, str] = {}
    for gid, items in by_group.items():
        group_dom[gid] = Counter(s.get("dominant", "__none__") for s in items).most_common(1)[0][0]

    # 关键：按「来源数据集 + 主类别」双层分层，保证每个来源在 train/val/test 里都出现，
    # 否则会出现「某个来源整个落进 test、train 一张都没有」的极端划分（实测踩过。
    group_src: Dict[int, str] = {}
    for gid, items in by_group.items():
        group_src[gid] = Counter(s.get("source", "unknown") for s in items).most_common(1)[0][0]

    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for gid in by_group:
        buckets[(group_src[gid], group_dom[gid])].append(gid)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for dom, gids in buckets.items():
        gids = list(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        for split, part in (
            ("train", gids[:n_train]),
            ("val", gids[n_train : n_train + n_val]),
            ("test", gids[n_train + n_val :]),
        ):
            for gid in part:
                splits[split].extend(by_group[gid])
    for key in splits:
        rng.shuffle(splits[key])
    return splits


# --------------------------------------------------------------------------- #
# 写出
# --------------------------------------------------------------------------- #
def write_split(
    split: str,
    samples: Sequence[Dict[str, Any]],
    out_dir: Path,
    class_info: Dict[str, Any],
    name_to_id: Dict[str, int],
    link_mode: str = "copy",
    oversample_min: int = 0,
) -> Dict[str, Any]:
    """把某个划分写成 YOLO 格式，返回统计信息。"""
    img_out = ensure_dir(out_dir / "images" / split)
    lbl_out = ensure_dir(out_dir / "labels" / split)

    class_counter: Counter = Counter()
    image_class_counter: Counter = Counter()
    unknown_counter: Counter = Counter()
    boxes_total = 0
    written = 0
    manifest_rows: List[Dict[str, Any]] = []
    oversample_pool: List[Tuple[Path, Path, List[str]]] = []

    for sample in samples:
        image_path: Path = sample["image"]
        img = imread(image_path)
        if img is None:
            LOG.warning("无法读取图片，跳过：%s", image_path)
            continue
        h, w = img.shape[:2]
        lines: List[str] = []
        names_in_image: List[str] = []

        if sample["kind"] == "voc":
            width, height, objects = parse_voc_xml(sample["annotation"])
            if width <= 0 or height <= 0:
                width, height = w, h
            for raw_name, x1, y1, x2, y2 in objects:
                cls_id = map_class(raw_name, name_to_id)
                if cls_id is None:
                    unknown_counter[raw_name] += 1
                    continue
                # 归一化 + 裁剪到图像范围内
                x1 = max(0.0, min(x1, width))
                x2 = max(0.0, min(x2, width))
                y1 = max(0.0, min(y1, height))
                y2 = max(0.0, min(y2, height))
                bw, bh = x2 - x1, y2 - y1
                if bw < 2 or bh < 2:
                    continue
                cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
                class_counter[class_info["zh"][cls_id]] += 1
                names_in_image.append(class_info["zh"][cls_id])
        else:
            label_path = sample["annotation"]
            src_names = sample.get("yolo_names")
            if label_path and Path(label_path).exists():
                for raw_line in Path(label_path).read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = raw_line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        raw_id = int(float(parts[0]))
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    except ValueError:
                        continue
                    cls_id, src_name = resolve_yolo_cls_id(raw_id, src_names, class_info, name_to_id)
                    if cls_id is None:
                        unknown_counter[src_name] += 1
                        continue
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    class_counter[class_info["zh"][cls_id]] += 1
                    names_in_image.append(class_info["zh"][cls_id])

        if not lines:
            continue

        # 关键：输出文件名加「来源前缀」。不同数据集常有同名图片（实测 SCB5-Talk 的 4539 张
        # 与 Talk-Teacher-Behavior 完全重名），直接 image_path.name 会互相覆盖，
        # 42,695 条记录最后只剩 23,575 张文件。加前缀后保证全局唯一。
        src_tag = "".join(ch for ch in str(sample.get("source", "") or "") if ch.isalnum() or ch in "-_")
        stem = f"{src_tag}_{image_path.stem}" if src_tag else image_path.stem
        dst_img = img_out / f"{stem}{image_path.suffix.lower()}"
        dst_lbl = lbl_out / f"{stem}.txt"
        if link_mode == "copy":
            shutil.copy2(image_path, dst_img)
        elif link_mode == "move":
            shutil.move(str(image_path), str(dst_img))
        else:  # none
            dst_img = image_path
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
        boxes_total += len(lines)
        for n in set(names_in_image):
            image_class_counter[n] += 1
        manifest_rows.append(
            {
                "split": split,
                "image": dst_img.name,
                "boxes": len(lines),
                "classes": "|".join(sorted(set(names_in_image))),
                "group": sample.get("group", ""),
                "source": str(image_path),
            }
        )
        if split == "train":
            oversample_pool.append((image_path, dst_lbl, sorted(set(names_in_image)), stem))

    # 稀有类过采样（仅 train）：复制含稀有类的图片若干份
    oversampled = 0
    if split == "train" and oversample_min > 0:
        need = {n: oversample_min - image_class_counter.get(n, 0) for n in class_info["zh"]}
        for image_path, dst_lbl, names, stem in oversample_pool:
            rare = [n for n in names if need.get(n, 0) > 0]
            if not rare:
                continue
            copies = min(2, max(need[n] for n in rare))  # 最多复制 2 份，避免过拟合
            for k in range(copies):
                suffix = f"_os{k + 1}"
                new_img = img_out / f"{stem}{suffix}{image_path.suffix.lower()}"
                new_lbl = lbl_out / f"{stem}{suffix}.txt"
                if link_mode != "none":
                    shutil.copy2(image_path, new_img)
                dst_lbl_text = Path(dst_lbl).read_text(encoding="utf-8")
                new_lbl.write_text(dst_lbl_text, encoding="utf-8")
                oversampled += 1
                for n in rare:
                    image_class_counter[n] += 1

    return {
        "split": split,
        "images": written,
        "boxes": boxes_total,
        "instances_per_class": dict(class_counter),
        "images_per_class": dict(image_class_counter),
        "unknown_labels": dict(unknown_counter),
        "oversampled": oversampled,
        "manifest": manifest_rows,
    }


def render_stats_markdown(stats: Dict[str, Any], class_info: Dict[str, Any]) -> str:
    lines: List[str] = ["# 数据集统计报告", ""]
    lines += [f"- 生成时间：{stats['generated_at']}", f"- 类别数：{stats['num_classes']}", f"- 源目录：`{stats['source']}`", ""]
    lines += ["## 一、划分概览", "", "| 划分 | 图片数 | 标注框数 | 过采样图片 |", "| --- | --- | --- | --- |"]
    for split in ["train", "val", "test"]:
        s = stats["splits"].get(split, {})
        lines.append(f"| {split} | {s.get('images', 0)} | {s.get('boxes', 0)} | {s.get('oversampled', 0)} |")
    total_images = sum(stats["splits"].get(s, {}).get("images", 0) for s in ["train", "val", "test"])
    total_boxes = sum(stats["splits"].get(s, {}).get("boxes", 0) for s in ["train", "val", "test"])
    lines += ["", f"- 图片总数：**{total_images}**（预期成果要求 >= {stats.get('target_samples', 5000)}）"]
    lines += [f"- 标注框总数：**{total_boxes}**", ""]

    dedup = stats.get("dedup")
    if dedup:
        lines += [
            "### 去重与泄漏检查",
            "",
            f"- 感知哈希（dHash）聚类：**{dedup['groups']}** 组，其中多图组共含 **{dedup['dup_images']}** 张，最大组 {dedup['largest_group']} 张",
            "- 划分规则：**同一重复组只会进入同一个划分**（train/val/test 互斥），不存在跨划分数据泄漏",
            "",
        ]

    lines += ["## 二、各类别实例数", "", "| id | 类别 | 训练 | 验证 | 测试 | 合计 | 图片数 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    images_per_class: Counter = Counter()
    for split in ["train", "val", "test"]:
        for k, v in stats["splits"].get(split, {}).get("images_per_class", {}).items():
            images_per_class[k] += v
    for i, name in enumerate(class_info["zh"]):
        row = [stats["splits"].get(s, {}).get("instances_per_class", {}).get(name, 0) for s in ["train", "val", "test"]]
        lines.append(f"| {i} | {name} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} | {images_per_class.get(name, 0)} |")

    unknown = Counter()
    for split in ["train", "val", "test"]:
        unknown.update(stats["splits"].get(split, {}).get("unknown_labels", {}))
    lines += ["", "## 三、未识别标签（需检查标注或补充 aliases）", ""]
    if unknown:
        lines += ["| 原始标签 | 出现次数 |", "| --- | --- |"]
        for k, v in unknown.most_common(30):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("无（全部标签都能映射到 8 个类别）")

    lines += ["", "## 四、场景/学段覆盖（按子目录统计）", "", "| 分组 | 图片数 |", "| --- | --- |"]
    for k, v in sorted(stats.get("groups", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    sources = stats.get("sources") or {}
    if sources:
        all_src = sorted({s for d in sources.values() for s in d})
        lines += ["## 五、来源数据集分布（每个划分都含全部来源）", "",
                  "| 来源 | " + " | ".join(["train", "val", "test"]) + " |",
                  "| --- | --- | --- | --- |"]
        for src in all_src:
            row = [sources.get(sp, {}).get(src, 0) for sp in ["train", "val", "test"]]
            lines.append(f"| {src} | {row[0]} | {row[1]} | {row[2]} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="把 VOC / YOLO 标注转换成 YOLOv11 训练集")
    parser.add_argument("--voc-dir", type=str, default=None, help="VOC 数据集目录（含 Annotations/JPEGImages）")
    parser.add_argument("--yolo-dir", type=str, action="append", default=None,
                        help="YOLO 数据集目录（含 images/labels），可重复传多个")
    parser.add_argument("--yolo-names", type=str, action="append", default=None,
                        help="某个 YOLO 数据集的源类别名，逗号分隔、顺序=源类别 id 顺序；"
                             "可重复，与 --yolo-dir 按出现顺序一一对应。"
                             "例：--yolo-names \"Eating,Fighting,Sleeping,Talking,UOP\"")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "processed"), help="输出目录")
    parser.add_argument("--dataset-yaml", type=str, default=str(DATA_DIR / "dataset.yaml"), help="生成的训练配置")
    parser.add_argument("--classes", type=str, default=None, help="类别定义文件，默认 configs/classes.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oversample-min", type=int, default=0, help="稀有类在训练集中至少达到的图片数（0=关闭）")
    parser.add_argument("--dedup", dest="dedup", action="store_true", default=True,
                        help="默认开启：用感知哈希把重复/相邻帧聚成一组，整组进同一划分，防止数据泄漏")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="关闭去重分组（不推荐）")
    parser.add_argument("--dedup-threshold", type=int, default=6, help="dHash 汉明距离阈值，越小越严格（默认 6）")
    parser.add_argument("--link-mode", choices=["copy", "move", "none"], default="copy")
    parser.add_argument("--max-samples", type=int, default=0, help="调试用：只处理前 N 张")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = parser.parse_args(argv)

    if not args.voc_dir and not args.yolo_dir:
        parser.error("至少提供 --voc-dir 或 --yolo-dir 之一")

    class_info = load_classes(args.classes)
    name_to_id = build_name_to_id(class_info)
    LOG.info("类别：%s", "、".join(class_info["zh"]))

    samples: List[Dict[str, Any]] = []
    source_desc: List[str] = []
    if args.voc_dir:
        voc_dir = Path(args.voc_dir)
        part = collect_voc_samples(voc_dir)
        for s in part:
            s["source"] = voc_dir.name
        LOG.info("VOC 目录 %s：找到 %d 个「图片+XML」配对", voc_dir, len(part))
        samples += part
        source_desc.append(str(voc_dir))
    yolo_dirs = args.yolo_dir or []
    yolo_names_list = args.yolo_names or []
    for idx, dir_str in enumerate(yolo_dirs):
        yolo_dir = Path(dir_str)
        src_names: Optional[List[str]] = None
        if idx < len(yolo_names_list) and yolo_names_list[idx]:
            src_names = [x.strip() for x in str(yolo_names_list[idx]).split(",") if x.strip()]
        part = collect_yolo_samples(yolo_dir)
        for s in part:
            s["yolo_names"] = src_names
            s["source"] = yolo_dir.name
        LOG.info("YOLO 目录 %s：找到 %d 张图片%s", yolo_dir, len(part),
                 f"（源类别：{'、'.join(src_names)}）" if src_names else "（按源 id 直接对应）")
        samples += part
        source_desc.append(str(yolo_dir))

    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        LOG.error("没有找到任何样本，请检查目录结构（Annotations/JPEGImages 或 images/labels）")
        return 2

    # 先解析每张图的主类别，用于分层划分
    for s in samples:
        mapped = mapped_class_names(s, class_info, name_to_id)
        s["dominant"] = Counter(mapped).most_common(1)[0][0] if mapped else "__none__"

    dedup_stats: Optional[Dict[str, Any]] = None
    if args.dedup:
        dedup_stats = assign_dup_groups(samples, threshold=args.dedup_threshold)
        LOG.info("去重分组：%d 组（多图组含 %d 张，最大组 %d 张），同组只会进入同一个划分",
                 dedup_stats["groups"], dedup_stats["dup_images"], dedup_stats["largest_group"])
        splits = split_samples_grouped(samples, args.train_ratio, args.val_ratio, args.seed)
    else:
        LOG.warning("已关闭去重分组：近似重复图片可能被分到 train/val，验证指标会虚高！")
        splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)

    if args.dry_run:
        for k, v in splits.items():
            print(f"{k}: {len(v)} 张")
        return 0

    out_dir = Path(args.out)
    stats: Dict[str, Any] = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "source": " + ".join(source_desc),
        "num_classes": class_info["num_classes"],
        "target_samples": class_info["dataset"].get("target_samples", 0),
        "splits": {},
        "dedup": dedup_stats,
        "sources": {k: dict(Counter(s.get("source", "unknown") for s in v)) for k, v in splits.items()},
        "groups": dict(Counter(s.get("group", "") for s in samples)),
    }
    manifest_rows: List[Dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        LOG.info("写出 %s：%d 张", split, len(splits[split]))
        info = write_split(
            split,
            splits[split],
            out_dir,
            class_info,
            name_to_id,
            link_mode=args.link_mode,
            oversample_min=args.oversample_min if split == "train" else 0,
        )
        manifest_rows += info.pop("manifest")
        stats["splits"][split] = info

    # dataset.yaml
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": class_info["num_classes"],
        "names": class_info["names"],
    }
    if not args.dry_run:
        dump_yaml(dataset_yaml, args.dataset_yaml)
        LOG.info("已生成训练配置：%s", args.dataset_yaml)

    # manifest.csv
    manifest_path = out_dir / "manifest.csv"
    if manifest_rows:
        ensure_dir(manifest_path.parent)
        with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    ensure_dir(REPORTS_DIR)
    dump_json(stats, REPORTS_DIR / "dataset_stats.json")
    (REPORTS_DIR / "dataset_stats.md").write_text(render_stats_markdown(stats, class_info), encoding="utf-8")

    total = sum(stats["splits"][s]["images"] for s in stats["splits"])
    LOG.info("完成：共 %d 张图片，%d 个标注框", total, sum(stats["splits"][s]["boxes"] for s in stats["splits"]))
    LOG.info("统计报告：%s", REPORTS_DIR / "dataset_stats.md")
    target = class_info["dataset"].get("target_samples", 5000)
    if total < target:
        LOG.warning("当前样本量 %d < 目标 %d，请继续采集/标注（见 docs/03_采集与标注分工.md）", total, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
