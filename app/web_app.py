"""课堂行为检测系统 —— Streamlit 网页端（可视化）。

功能
----
* 图片检测 / 视频检测 / 摄像头拍照检测
* 检测结果可视化、行为统计、异常预警、报表导出
* 与桌面端共用 src/ 下的检测与分析模块

运行
----
streamlit run app/web_app.py
（在 E:\\2026srp 目录下执行；默认浏览器打开 http://localhost:8501）
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import streamlit as st  # noqa: E402

from src import report as report_mod  # noqa: E402
from src.alerts import AlertManager  # noqa: E402
from src.analytics import BehaviorAnalytics  # noqa: E402
from src.detector import BehaviorDetector  # noqa: E402
from src.tracker import SimpleTracker  # noqa: E402
from src.utils import (  # noqa: E402
    REPORTS_DIR,
    imread,
    load_app_config,
    load_classes,
    resolve_path,
    setup_matplotlib,
)

setup_matplotlib()
CLASS_INFO = load_classes()
ZH_NAMES: List[str] = CLASS_INFO["zh"]
COLORS = CLASS_INFO["colors"]
CFG = load_app_config()

st.set_page_config(page_title="课堂行为检测系统", page_icon="🏫", layout="wide")


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="正在加载模型…")
def get_detector(weights: str, device: str, imgsz: int, conf: float) -> BehaviorDetector:
    det = BehaviorDetector(weights=weights, device=device, imgsz=imgsz, conf=conf, class_names=ZH_NAMES, per_class_conf=CFG.get("model", {}).get("per_class_conf"))
    det.load()
    return det


def make_analytics(fps: float = 25.0) -> BehaviorAnalytics:
    analysis_cfg = CFG.get("analysis", {})
    return BehaviorAnalytics(
        zh_names=ZH_NAMES,
        attention_classes=analysis_cfg.get("attention_classes", []),
        anomaly_classes=analysis_cfg.get("anomaly_classes", []),
        fps_assume=fps,
        window_seconds=float(analysis_cfg.get("window_seconds", 60)),
    )


def make_alerts() -> AlertManager:
    return AlertManager(
        rules=CLASS_INFO["anomaly_rules"],
        cooldown=float(CFG.get("analysis", {}).get("alert_cooldown", 10)),
    )


def show_behavior_panel(snapshot: Dict[str, Any]) -> None:
    counts = snapshot.get("counts", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("画面人数", snapshot.get("persons", 0))
    c2.metric("当前抬头率", f"{snapshot.get('head_up_rate', 0):.1f}%")
    c3.metric("专注/参与率", f"{snapshot.get('attention_rate', 0):.1f}%")
    st.bar_chart({name: counts.get(name, 0) for name in ZH_NAMES})


def render_alerts(alerts: List[Any]) -> None:
    if not alerts:
        st.success("暂未发现异常行为")
        return
    for a in alerts[-15:][::-1]:
        icon = "🔴" if a.level == "高" else "🟠"
        st.write(f"{icon} {a.text()}")


# --------------------------------------------------------------------------- #
def tab_image(detector: BehaviorDetector) -> None:
    st.subheader("本地图片检测")
    uploaded = st.file_uploader("上传课堂图片", type=["jpg", "jpeg", "png", "bmp"], key="img")
    if uploaded is None:
        st.info("支持 jpg / png / bmp；建议使用教室全景、后排视角的图片。")
        return
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("图片解码失败")
        return
    with st.spinner("检测中…"):
        detections = detector.predict(frame)
        annotated = detector.annotate(frame.copy(), detections, colors=COLORS)
    show_boxes = st.checkbox("显示识别框（取消勾选看原图）", value=True, key="img_show_boxes")
    left, right = st.columns([3, 2])
    with left:
        shown = annotated if show_boxes else frame
        st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                 caption="检测结果" if show_boxes else "原始图片", use_container_width=True)
    with right:
        st.write("**本帧行为统计**")
        st.dataframe(
            [{"行为": d.cls_name, "置信度": round(d.conf, 3)} for d in detections] or [{"行为": "未检测到目标", "置信度": 0}],
            use_container_width=True, hide_index=True,
        )
    if st.button("保存检测图片"):
        path = report_mod.save_annotated_image(annotated)
        st.success(f"已保存：{path}")


def tab_video(detector: BehaviorDetector) -> None:
    st.subheader("本地视频检测")
    uploaded = st.file_uploader("上传课堂视频", type=["mp4", "avi", "mov", "mkv"], key="vid")
    stride = st.slider("抽帧间隔（每 N 帧检测一次，越大越快）", 1, 30, 5)
    if uploaded is None:
        st.info("视频会逐帧（按间隔抽帧）检测并合成结果视频，长视频请调大抽帧间隔。")
        return
    if not st.button("开始检测视频"):
        return

    tmp_in = Path(tempfile.gettempdir()) / f"cls_in_{int(time.time())}{Path(uploaded.name).suffix}"
    tmp_in.write_bytes(uploaded.getvalue())
    cap = cv2.VideoCapture(str(tmp_in))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_out = Path(tempfile.gettempdir()) / f"cls_out_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(stride, 1), (w, h))

    analytics = make_analytics(fps / max(stride, 1))
    alerts = make_alerts()
    tracker = SimpleTracker(max_age=int(CFG.get("tracking", {}).get("max_age", 30)))
    use_track = bool(CFG.get("tracking", {}).get("enable", True))
    detector.reset_tracking()

    progress = st.progress(0.0, text="检测中…")
    preview = st.empty()
    idx = 0
    det_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            ts = idx / fps
            dets = detector.track(frame) if use_track else tracker.update(detector.predict(frame))
            det_count += len(dets)
            analytics.update(dets, ts=ts)
            alerts.evaluate(analytics, ts=ts)
            writer.write(detector.annotate(frame.copy(), dets, colors=COLORS))
            if idx % (stride * 10) == 0:
                preview.image(cv2.cvtColor(detector.annotate(frame.copy(), dets, colors=COLORS), cv2.COLOR_BGR2RGB),
                              caption=f"第 {idx} 帧", use_container_width=True)
        idx += 1
        if total:
            progress.progress(min(idx / total, 1.0), text=f"检测中… {idx}/{total} 帧")
    cap.release()
    writer.release()
    progress.progress(1.0, text="检测完成")
    st.video(str(tmp_out))
    render_summary(analytics, alerts, f"视频检测（{total} 帧，检测目标累计 {det_count} 次）")


def tab_camera(detector: BehaviorDetector) -> None:
    st.subheader("摄像头实时检测（网页拍照模式）")
    st.caption("浏览器摄像头按帧抓拍检测；如需连续实时检测，请使用桌面端 app/desktop_app.py。")
    shot = st.camera_input("点击拍照并检测")
    if shot is None:
        return
    data = np.frombuffer(shot.getvalue(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("图像解码失败")
        return
    detections = detector.predict(frame)
    annotated = detector.annotate(frame.copy(), detections, colors=COLORS)
    show_boxes2 = st.checkbox("显示识别框（取消勾选看原图）", value=True, key="cam_show_boxes")
    shown2 = annotated if show_boxes2 else frame
    st.image(cv2.cvtColor(shown2, cv2.COLOR_BGR2RGB),
             caption="检测结果" if show_boxes2 else "原始画面", use_container_width=True)
    analytics = make_analytics()
    analytics.update(detections, ts=time.time())
    alerts = make_alerts()
    render_summary(analytics, alerts, "摄像头拍照检测")


def render_summary(analytics: BehaviorAnalytics, alerts: AlertManager, title: str) -> None:
    summary = analytics.summary()
    st.markdown(f"### {title}")
    show_behavior_panel(summary.get("current", {}))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("统计帧数", summary["frames"])
    c2.metric("跟踪学生数", summary["students"])
    c3.metric("整体抬头率", f"{summary['head_up_rate']:.1f}%")
    c4.metric("异常行为时长(秒)", f"{summary['anomaly_seconds']:.1f}")

    behavior = summary.get("behavior_seconds", {})
    if behavior:
        st.bar_chart(behavior)
    st.write("**逐学生行为画像**")
    st.dataframe(analytics.per_student_table(), use_container_width=True, hide_index=True)
    st.write("**异常预警**")
    render_alerts(alerts.alerts)

    if st.button("导出报表（CSV / Markdown / 图表）"):
        csv_path = report_mod.export_timeline_csv(analytics)
        md = report_mod.write_markdown_report(summary, analytics.per_student_table(), alerts.alerts)
        charts = report_mod.save_charts(summary)
        st.success("已生成：" + "，".join([str(csv_path), str(md)] + [str(c) for c in charts]))
        with open(md, "rb") as f:
            st.download_button("下载 Markdown 报告", f, file_name=Path(md).name)


# --------------------------------------------------------------------------- #
def render_legend() -> None:
    """页面顶部：颜色 → 课堂行为类别 的图例（与画面里框/标签同色）。"""
    chips = []
    for i, name in enumerate(ZH_NAMES):
        b, g, r = (int(v) for v in COLORS[i])
        lum = 0.114 * b + 0.587 * g + 0.299 * r
        fg = "#000" if lum >= 150 else "#fff"
        chips.append(
            f'<span style="background:rgb({r},{g},{b});color:{fg};'
            "border-radius:9px;padding:3px 11px;margin:0 6px 6px 0;"
            f'display:inline-block;font-size:13px;font-weight:600">{name}</span>'
        )
    st.markdown(
        '<div style="line-height:2.2;margin-bottom:4px">'
        '<span style="color:#888;font-size:12px;margin-right:8px">类别图例</span>'
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("🏫 课堂行为检测系统（YOLOv11）")
    st.caption("多场景 · 多学段 · 8 类课堂行为识别｜支持图片 / 视频 / 摄像头 · 行为分析 · 异常预警 · 数据可视化")
    render_legend()

    with st.sidebar:
        st.header("模型设置")
        default_weights = str(CFG.get("model", {}).get("weights", "yolo11n.pt"))
        if not resolve_path(default_weights).exists():
            default_weights = str(CFG.get("model", {}).get("fallback_weights", "yolo11n.pt"))
        weights = st.text_input("权重路径", default_weights)
        device = st.selectbox("推理设备", ["0", "cpu"], index=0)
        imgsz = st.select_slider("推理尺寸", [416, 512, 640, 768, 960], value=int(CFG.get("model", {}).get("imgsz", 640)))
        conf = st.slider("置信度阈值", 0.05, 0.95, float(CFG.get("model", {}).get("conf", 0.35)), 0.05)
        st.markdown("---")
        st.markdown("**类别定义**")
        for i, name in enumerate(ZH_NAMES):
            kind = CLASS_INFO["kinds"][i]
            tag = {"anomaly": "⚠️ 异常", "positive": "⭐ 积极", "normal": "正常"}.get(kind, "正常")
            st.markdown(f"`{i}` {name} — {tag}")
        st.markdown(f"报告目录：`{resolve_path(REPORTS_DIR)}`")

    detector = get_detector(weights, device, imgsz, conf)
    st.success(f"模型已就绪：{Path(detector.weights).name}｜设备：{device}｜类别数：{CLASS_INFO['num_classes']}")

    t1, t2, t3, t4 = st.tabs(["🖼 图片检测", "🎞 视频检测", "📷 摄像头检测", "📊 数据看板"])
    with t1:
        tab_image(detector)
    with t2:
        tab_video(detector)
    with t3:
        tab_camera(detector)
    with t4:
        st.subheader("数据集与系统说明")
        stats_md = resolve_path(REPORTS_DIR) / "dataset_stats.md"
        if stats_md.exists():
            st.markdown(stats_md.read_text(encoding="utf-8"))
        else:
            st.info("还没有数据集统计报告。运行：python scripts/prepare_dataset.py --voc-dir <数据目录>")
        val_md = resolve_path(REPORTS_DIR) / "val_report_val.md"
        if val_md.exists():
            st.markdown("---")
            st.markdown(val_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
