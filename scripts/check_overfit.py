"""过拟合 / 欠拟合诊断：读取 ultralytics 的 results.csv，给出中文诊断结论。

判断依据
--------
1. **损失曲线**：train 与 val 的 box/cls/dfl 损失是否同步下降；
   val 损失持续上升而 train 持续下降 → 典型过拟合。
2. **mAP 曲线**：mAP50-95 是否在后期走平/回落；回落幅度 > 3% 视为过拟合信号。
3. **泛化间隙**：最后 10 轮 val 损失均值 / 最低 val 损失 的比值，> 1.15 提示过拟合。
4. **训练充分性**：mAP 曲线在最后一轮仍在上升 → 欠拟合（应加轮次）。
5. **数据泄漏排查**：若 val mAP 高得异常（如 > 0.97）而实测效果差，优先怀疑泄漏，
   跑 scripts/dataset_dedup.py 检查。

用法
----
python scripts/check_overfit.py --run runs/train/classroom_v2_100ep
python scripts/check_overfit.py --run runs/train/classroom_v3_150ep --json reports/overfit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import REPORTS_DIR, ensure_dir, get_logger, setup_matplotlib  # noqa: E402

LOG = get_logger("overfit")


def read_results(run_dir: Path) -> List[Dict[str, float]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}")
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if k is None:
                    continue
                key = k.strip()
                try:
                    parsed[key] = float(v)
                except (TypeError, ValueError):
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def series(rows: List[Dict[str, float]], key: str) -> List[float]:
    return [r[key] for r in rows if key in r]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def diagnose(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"epochs": len(rows)}
    if len(rows) < 5:
        out["verdict"] = "轮次太少，无法判断"
        return out

    train_box = series(rows, "train/box_loss")
    train_cls = series(rows, "train/cls_loss")
    val_box = series(rows, "val/box_loss")
    val_cls = series(rows, "val/cls_loss")
    map50 = series(rows, "metrics/mAP50(B)")
    map5095 = series(rows, "metrics/mAP50-95(B)")

    def trend(vals: List[float], window: int = 10) -> float:
        """近 window 轮的相对变化率（正=上升，负=下降）。"""
        if len(vals) < window * 2:
            window = max(3, len(vals) // 2)
        head = mean(vals[-window * 2 : -window]) if len(vals) >= window * 2 else mean(vals[:window])
        tail = mean(vals[-window:])
        return (tail - head) / head if head else 0.0

    out["train_box_trend"] = round(trend(train_box), 4)
    out["val_box_trend"] = round(trend(val_box), 4)
    out["train_cls_trend"] = round(trend(train_cls), 4)
    out["val_cls_trend"] = round(trend(val_cls), 4)
    out["map50_trend"] = round(trend(map50), 4)
    out["map5095_trend"] = round(trend(map5095), 4)

    if val_box:
        best_val = min(val_box)
        out["val_box_min"] = round(best_val, 4)
        out["val_box_last10_ratio"] = round(mean(val_box[-10:]) / best_val, 3) if best_val else 0.0

    if map5095:
        out["map5095_best"] = round(max(map5095), 4)
        out["map5095_best_epoch"] = int(map5095.index(max(map5095)) + 1)
        out["map5095_last"] = round(map5095[-1], 4)
        out["map5095_drop_from_best"] = round(max(map5095) - map5095[-1], 4)
    if map50:
        out["map50_best"] = round(max(map50), 4)
        out["map50_last"] = round(map50[-1], 4)

    # ---- 结论 ----
    verdicts: List[str] = []
    signals: List[str] = []
    if out.get("val_box_trend", 0) > 0.03 and out.get("train_box_trend", 0) < -0.05:
        verdicts.append("过拟合")
        signals.append(f"val box_loss 上升 {out['val_box_trend']*100:.1f}% 而 train box_loss 下降 {abs(out['train_box_trend'])*100:.1f}%")
    if out.get("val_box_last10_ratio", 1.0) > 1.15:
        verdicts.append("过拟合")
        signals.append(f"最后 10 轮 val loss 是最低值的 {out['val_box_last10_ratio']} 倍（>1.15 视为过拟合）")
    if out.get("map5095_drop_from_best", 0) > 0.03:
        verdicts.append("过拟合")
        signals.append(f"mAP50-95 从最好 {out['map5095_best']} 回落到 {out['map5095_last']}（跌幅 {out['map5095_drop_from_best']}）")
    if out.get("map5095_trend", 0) > 0.005 and len(rows) > 30:
        verdicts.append("欠拟合（可继续训）")
        signals.append(f"最后阶段 mAP50-95 仍在上升（+{out['map5095_trend']*100:.1f}%），说明还没训够")
    if not verdicts:
        verdicts.append("曲线正常（未见过拟合信号）")
        signals.append("train/val 损失同步下降，mAP 已收敛")

    out["verdict"] = " + ".join(dict.fromkeys(verdicts))
    out["signals"] = signals
    return out


def make_chart(rows: List[Dict[str, float]], out_png: Path) -> Optional[Path]:
    setup_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=130, sharex=True)
    for key, label in [("train/box_loss", "train box_loss"), ("val/box_loss", "val box_loss")]:
        values = series(rows, key)
        if values:
            axes[0].plot(epochs[: len(values)], values, label=label)
    axes[0].set_ylabel("box_loss")
    axes[0].legend()
    axes[0].set_title("训练/验证损失（看是否分叉）")
    for key, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
        values = series(rows, key)
        if values:
            axes[1].plot(epochs[: len(values)], values, label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()
    axes[1].set_title("精度曲线（看是否走平/回落）")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="过拟合 / 欠拟合诊断")
    parser.add_argument("--run", type=str, required=True, help="训练输出目录，如 runs/train/classroom_v3_150ep")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--chart", default=None, help="曲线图输出路径")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    try:
        rows = read_results(run_dir)
    except FileNotFoundError as exc:
        LOG.error(str(exc))
        return 2

    info = diagnose(rows)
    ensure_dir(REPORTS_DIR)
    chart = make_chart(rows, Path(args.chart) if args.chart else REPORTS_DIR / f"overfit_{run_dir.name}.png")

    lines = [
        "# 过拟合 / 欠拟合诊断报告",
        "",
        f"- 训练目录：`{run_dir}`",
        f"- 轮数：{info['epochs']}",
        f"- **结论：{info['verdict']}**",
        "",
        "## 一、判断依据",
        "",
        *[f"- {s}" for s in info.get("signals", [])],
        "",
        "## 二、关键数字",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| train box_loss 近期变化 | {info.get('train_box_trend', 0)*100:+.1f}% |",
        f"| val box_loss 近期变化 | {info.get('val_box_trend', 0)*100:+.1f}% |",
        f"| val box_loss 最低值 | {info.get('val_box_min', '-')} |",
        f"| 最后 10 轮 val loss / 最低值 | {info.get('val_box_last10_ratio', '-')} |",
        f"| mAP50 最好 / 最后 | {info.get('map50_best', '-')} / {info.get('map50_last', '-')} |",
        f"| mAP50-95 最好（第几轮） | {info.get('map5095_best', '-')}（第 {info.get('map5095_best_epoch', '-')} 轮）|",
        f"| mAP50-95 最后 | {info.get('map5095_last', '-')} |",
        f"| 距最佳跌幅 | {info.get('map5095_drop_from_best', '-')} |",
        "",
        f"曲线图：`{chart}`",
        "",
        "## 三、处理建议",
        "",
    ]
    if "过拟合" in info["verdict"]:
        lines += [
            "- 加数据（最有效）：补采真实教室场景，尤其是误检多的类别；",
            "- 增强加强：`mixup 0.15~0.2`、`scale 0.6`、`erasing 0.5`、`degrees 8`；",
            "- 容量控制：换更小模型（yolo11n）或加大 `weight_decay`（0.0005→0.001）；",
            "- 早停：用最佳轮次权重（best.pt）而不是最后一轮。",
        ]
    elif "欠拟合" in info["verdict"]:
        lines += [
            "- 加轮次（+50~100 轮）；",
            "- 加大模型（yolo11s→yolo11m，注意 8GB 显存要用 batch 4）；",
            "- 提高 `imgsz`（640→768），后排小目标更清楚；",
            "- 检查标注质量与类别定义是否一致。",
        ]
    else:
        lines += ["- 当前训练健康，可直接部署；下一步收益主要来自补数据与推理侧调优（TTA / 按类阈值）。"]

    report = REPORTS_DIR / f"overfit_{run_dir.name}.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    if args.json:
        Path(args.json).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(lines[:16]))
    LOG.info("报告：%s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
