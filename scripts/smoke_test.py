"""端到端自检脚本：模型 -> 检测 -> 跟踪 -> 行为分析 -> 异常预警 -> 报表导出。

用途：换电脑、换环境、训练完新模型后，一条命令确认整条链路可用。

用法
----
python scripts/smoke_test.py                                  # 自动挑一张图片自检
python scripts/smoke_test.py --weights runs/best.pt --source D:/data/test.jpg
python scripts/smoke_test.py --source data/processed/images/val --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (  # noqa: E402
    REPORTS_DIR,
    ensure_dir,
    get_logger,
    imread,
    load_app_config,
    load_classes,
    resolve_path,
)

LOG = get_logger("smoke")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_image(source: Optional[str]) -> Optional[Path]:
    if source:
        p = Path(source)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
        if p.is_dir():
            for cand in sorted(p.rglob("*")):
                if cand.suffix.lower() in IMAGE_EXTS:
                    return cand
    # 兜底：用划分好的验证集
    for cand in sorted(resolve_path("data/processed/images/val").rglob("*")) if resolve_path("data/processed/images/val").exists() else []:
        if cand.suffix.lower() in IMAGE_EXTS:
            return cand
    # 再兜底：课程自带 VOC 数据
    voc = Path("E:/yolov11/VOCdevkit/VOC2007/JPEGImages")
    if voc.exists():
        for cand in sorted(voc.glob("*.jpg"))[:1]:
            return cand
    return None


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="课堂行为检测系统端到端自检")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--source", type=str, default=None, help="图片或图片目录")
    parser.add_argument("--limit", type=int, default=3, help="最多检测几张图")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_app_config(args.config)
    class_info = load_classes()
    weights = args.weights or cfg.get("model", {}).get("weights", "yolo11n.pt")
    resolved = resolve_path(weights)
    if not resolved.exists():
        fallback = cfg.get("model", {}).get("fallback_weights", "yolo11n.pt")
        LOG.warning("权重 %s 不存在，回退到 %s", weights, fallback)
        resolved = resolve_path(fallback)
    ok_all = check("模型权重存在", resolved.exists(), str(resolved))

    image = pick_image(args.source)
    if not check("找到测试图片", image is not None, str(image)):
        return 1

    from src.alerts import AlertManager
    from src.analytics import BehaviorAnalytics
    from src.detector import BehaviorDetector
    from src.tracker import SimpleTracker
    from src import report as report_mod

    detector = BehaviorDetector(
        weights=str(resolved),
        device=str(cfg.get("model", {}).get("device", "0")),
        imgsz=int(cfg.get("model", {}).get("imgsz", 640)),
        conf=float(cfg.get("model", {}).get("conf", 0.35)),
        class_names=class_info["zh"],
    )
    detector.load()
    ok_all &= check("模型加载", True, f"{resolved.name}，类别数 {len(detector.names)}")

    analytics = BehaviorAnalytics(
        zh_names=class_info["zh"],
        attention_classes=cfg.get("analysis", {}).get("attention_classes", []),
        anomaly_classes=cfg.get("analysis", {}).get("anomaly_classes", []),
    )
    alerts = AlertManager(rules=class_info["anomaly_rules"], cooldown=1.0)
    tracker = SimpleTracker()

    frame = imread(image)
    ok_all &= check("图片读取", frame is not None, f"{image.name} {None if frame is None else frame.shape}")
    if frame is None:
        return 1

    detections = detector.predict(frame)
    annotated = detector.annotate(frame.copy(), detections, colors=class_info["colors"])
    out_img = report_mod.save_annotated_image(annotated, resolve_path(REPORTS_DIR) / "smoke_detect.jpg")
    ok_all &= check("推理 + 绘制", Path(out_img).exists(), f"检测到 {len(detections)} 个目标 -> {out_img}")

    # 模拟 30 帧序列，验证跟踪 + 时长统计 + 预警链路
    for i in range(30):
        dets = tracker.update(detector.predict(frame)) if i % 10 == 0 else tracker.update(detections)
        analytics.update(dets, ts=time.time() + i * 0.04)
        alerts.evaluate(analytics, ts=time.time() + i * 0.04)

    summary = analytics.summary()
    students = analytics.per_student_table()
    ok_all &= check("行为分析", summary["frames"] == 30, f"帧数={summary['frames']}，跟踪目标={summary['students']}")

    csv_path = report_mod.export_timeline_csv(analytics, resolve_path(REPORTS_DIR) / "smoke_timeline.csv")
    charts = report_mod.save_charts(summary, resolve_path(REPORTS_DIR) / "smoke_charts")
    md_path = report_mod.write_markdown_report(summary, students, alerts.alerts, resolve_path(REPORTS_DIR) / "smoke_report.md")
    ok_all &= check("报表导出", all(Path(p).exists() for p in [csv_path, md_path] + list(charts)),
                    f"{Path(csv_path).name}, {Path(md_path).name}, {len(charts)} 张图")

    print("\n===== 自检结果 =====")
    print(f"权重        : {resolved}")
    print(f"测试图片    : {image}")
    print(f"检测目标数  : {len(detections)}")
    print(f"行为统计    : {summary['behavior_seconds']}")
    print(f"抬头率      : {summary['attention_rate']}%")
    print(f"预警条数    : {len(alerts.alerts)}")
    print(f"结论        : {'全部通过 ✅' if ok_all else '存在失败项 ❌'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
