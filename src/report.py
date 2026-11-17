"""报表与图表导出：CSV / Excel / Markdown / 统计图。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils import REPORTS_DIR, ensure_dir, imwrite, setup_matplotlib

# 中文字体在 matplotlib 里需要显式设置
setup_matplotlib()


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def export_timeline_csv(analytics, path: Optional[Any] = None) -> Path:
    """逐帧统计明细 CSV。"""
    df = analytics.to_dataframe()
    path = Path(path) if path else ensure_dir(REPORTS_DIR) / f"timeline_{_stamp()}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_excel(
    analytics,
    alerts: Optional[Sequence[Any]] = None,
    path: Optional[Any] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """导出 Excel：总览 / 逐帧明细 / 学生画像 / 预警记录。"""
    import pandas as pd

    path = Path(path) if path else ensure_dir(REPORTS_DIR) / f"课堂行为报表_{_stamp()}.xlsx"
    summary = summary or analytics.summary()

    overview = pd.DataFrame(
        [
            {"指标": "统计帧数", "数值": summary.get("frames")},
            {"指标": "累计行为时长(秒)", "数值": summary.get("duration_seconds")},
            {"指标": "出现学生数(跟踪目标)", "数值": summary.get("students")},
            {"指标": "画面峰值人数", "数值": summary.get("peak_persons")},
            {"指标": "平均人数", "数值": summary.get("avg_persons")},
            {"指标": "抬头率(%)", "数值": summary.get("head_up_rate")},
            {"指标": "专注/参与率(%)", "数值": summary.get("attention_rate")},
            {"指标": "异常行为时长(秒)", "数值": summary.get("anomaly_seconds")},
        ]
    )
    behavior = pd.DataFrame(
        [{"行为": k, "时长(秒)": v, "占比(%)": summary.get("behavior_share", {}).get(k, 0.0)}
         for k, v in summary.get("behavior_seconds", {}).items()]
    )
    students = pd.DataFrame(analytics.per_student_table())
    timeline = analytics.to_dataframe()
    alert_rows = [a.as_dict() for a in (alerts or [])]
    alert_df = pd.DataFrame(alert_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="总览", index=False)
        behavior.to_excel(writer, sheet_name="行为分布", index=False)
        if not students.empty:
            students.to_excel(writer, sheet_name="学生画像", index=False)
        if not timeline.empty:
            timeline.to_excel(writer, sheet_name="逐帧明细", index=False)
        if not alert_df.empty:
            alert_df.to_excel(writer, sheet_name="预警记录", index=False)
    return path


def save_charts(summary: Dict[str, Any], out_dir: Optional[Any] = None) -> List[Path]:
    """保存行为分布柱状图 + 抬头率饼图，返回图片路径列表。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ensure_dir(out_dir or (REPORTS_DIR / f"charts_{_stamp()}"))
    paths: List[Path] = []

    behavior = summary.get("behavior_seconds", {})
    if behavior:
        names = list(behavior.keys())
        values = [behavior[n] for n in names]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
        bars = ax.bar(names, values, color="#4C8BF5")
        ax.set_ylabel("累计时长（秒）")
        ax.set_title("课堂行为时长分布")
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        p = out / "behavior_distribution.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        attention = float(summary.get("attention_rate", 0.0))
        fig2, ax2 = plt.subplots(figsize=(4.6, 4.6), dpi=130)
        ax2.pie(
            [max(attention, 0.0), max(100.0 - attention, 0.0)],
            labels=["专注/参与", "非专注"],
            autopct="%1.1f%%",
            colors=["#41B883", "#E0E0E0"],
            startangle=90,
            wedgeprops={"width": 0.42},
        )
        ax2.set_title("课堂抬头/参与率")
        p2 = out / "attention_rate.png"
        fig2.savefig(p2)
        plt.close(fig2)
        paths.append(p2)
    return paths


def save_trend_chart(analytics, out_path: Optional[Any] = None) -> Optional[Path]:
    """抬头率随时间变化曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = analytics.timeline_records()
    if not records:
        return None
    path = Path(out_path) if out_path else ensure_dir(REPORTS_DIR) / f"trend_{_stamp()}.png"
    xs = [r["帧号"] for r in records]
    ys = [r["抬头率(%)"] for r in records]
    ps = [r["人数"] for r in records]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
    ax.plot(xs, ys, color="#41B883", label="抬头率(%)")
    ax.set_xlabel("帧号")
    ax.set_ylabel("抬头率(%)")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.plot(xs, ps, color="#4C8BF5", alpha=0.5, label="画面人数")
    ax2.set_ylabel("人数")
    ax.set_title("抬头率 / 人数随时间变化")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_markdown_report(
    summary: Dict[str, Any],
    students: Sequence[Dict[str, Any]],
    alerts: Sequence[Any],
    path: Optional[Any] = None,
    title: str = "课堂行为检测报告",
) -> Path:
    """生成 Markdown 文本报告。"""
    path = Path(path) if path else ensure_dir(REPORTS_DIR) / f"报告_{_stamp()}.md"
    lines: List[str] = [f"# {title}", "", f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines += ["## 一、总体指标", "", "| 指标 | 数值 |", "| --- | --- |"]
    for key, label in [
        ("frames", "统计帧数"),
        ("duration_seconds", "累计行为时长(秒)"),
        ("students", "跟踪到学生数"),
        ("peak_persons", "画面峰值人数"),
        ("avg_persons", "平均人数"),
        ("head_up_rate", "抬头率(%)"),
        ("attention_rate", "专注/参与率(%)"),
        ("anomaly_seconds", "异常行为时长(秒)"),
    ]:
        value = summary.get(key)
        if isinstance(value, float):
            value = round(value, 2)
        lines.append(f"| {label} | {value} |")

    lines += ["", "## 二、行为分布", "", "| 行为 | 累计时长(秒) | 占比(%) |", "| --- | --- | --- |"]
    for name, sec in summary.get("behavior_seconds", {}).items():
        lines.append(f"| {name} | {sec} | {summary.get('behavior_share', {}).get(name, 0)} |")

    if students:
        lines += ["", "## 三、学生行为画像（按总时长排序）", "", "| 学生 | 主要行为 | 总时长(秒) | 异常时长(秒) |", "| --- | --- | --- | --- |"]
        for row in students[:50]:
            lines.append(
                f"| {row.get('学生编号')} | {row.get('主要行为')} | {row.get('总时长(秒)')} | {row.get('异常时长(秒)')} |"
            )

    lines += ["", f"## 四、异常预警（共 {len(alerts)} 条）", ""]
    if alerts:
        lines += ["| 时间 | 等级 | 对象 | 行为 | 持续(秒) | 说明 |", "| --- | --- | --- | --- | --- | --- |"]
        for a in alerts[:200]:
            d = a.as_dict() if hasattr(a, "as_dict") else dict(a)
            lines.append(
                f"| {d.get('time_str')} | {d.get('level')} | #{d.get('track_id')} | {d.get('cls_name')} | "
                f"{d.get('duration', 0):.1f} | {d.get('message')} |"
            )
    else:
        lines.append("本次统计未触发异常预警。")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_annotated_image(image, path: Optional[Any] = None) -> Path:
    """保存标注后的图片（支持中文路径）。"""
    path = Path(path) if path else ensure_dir(REPORTS_DIR) / f"detect_{_stamp()}.jpg"
    imwrite(path, image)
    return path
