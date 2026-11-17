"""课堂行为检测系统 —— PyQt5 桌面端（可视化主程序）。

功能
----
* 本地图片检测 / 本地视频检测 / 摄像头实时检测
* 行为识别 + 目标跟踪 + 抬头率、参与度、行为时长统计
* 异常行为预警（趴桌睡觉 / 玩手机等，阈值可配）
* 实时图表 + 逐学生画像 + CSV/Excel/Markdown 报表导出

运行
----
python app/desktop_app.py
python app/desktop_app.py --weights runs/best.pt --source demo.mp4
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QFont, QImage, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src import report as report_mod  # noqa: E402
from src.alerts import Alert, AlertManager  # noqa: E402
from src.analytics import BehaviorAnalytics  # noqa: E402
from src.detector import BehaviorDetector  # noqa: E402
from src.tracker import SimpleTracker  # noqa: E402
from src.utils import (  # noqa: E402
    REPORTS_DIR,
    FPSCounter,
    ensure_dir,
    imread,
    load_app_config,
    load_classes,
    resolve_path,
    setup_matplotlib,
)

setup_matplotlib()


# --------------------------------------------------------------------------- #
class InferenceWorker(QThread):
    """独立线程做推理，避免界面卡顿。"""

    finished_frame = pyqtSignal(object, object, object, object)  # 标注图, 原始图, 检测框, 快照
    new_alerts = pyqtSignal(object)                       # List[Alert]

    def __init__(
        self,
        detector: BehaviorDetector,
        analytics: BehaviorAnalytics,
        alerts: AlertManager,
        tracker: Optional[SimpleTracker],
        use_track: bool,
        colors: List[Any],
    ) -> None:
        super().__init__()
        self.detector = detector
        self.analytics = analytics
        self.alerts = alerts
        self.tracker = tracker
        self.use_track = use_track
        self.colors = colors
        self.jobs: "queue.Queue" = queue.Queue(maxsize=2)
        self._running = True

    # ------------------------------------------------------------------ #
    def submit(self, frame: np.ndarray, ts: Optional[float] = None) -> None:
        """投递一帧；队列满时丢弃最旧的一帧，保证实时性。"""
        try:
            self.jobs.put_nowait((frame, ts or time.time()))
        except queue.Full:
            try:
                self.jobs.get_nowait()
                self.jobs.put_nowait((frame, ts or time.time()))
            except queue.Empty:
                pass

    def stop(self) -> None:
        self._running = False
        self.jobs.put(None) if self.jobs.empty() else None

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D401
        while self._running:
            try:
                item = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, ts = item
            try:
                if self.use_track and self.tracker is None:
                    detections = self.detector.track(frame)
                else:
                    detections = self.detector.predict(frame)
                    if self.tracker is not None:
                        detections = self.tracker.update(detections)
                snapshot = self.analytics.update(detections, ts=ts)
                fired = self.alerts.evaluate(self.analytics, ts=ts)
                annotated = self.detector.annotate(frame.copy(), detections, colors=self.colors)
                self.finished_frame.emit(annotated, frame, detections, snapshot)
                if fired:
                    self.new_alerts.emit(fired)
            except Exception as exc:  # pragma: no cover - 单帧异常不应终止整个界面
                print(f"[worker] 处理失败：{exc}")


# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, cfg: Dict[str, Any], class_info: Dict[str, Any], weights: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_info = class_info
        self.zh_names: List[str] = class_info["zh"]
        self.colors = class_info["colors"]

        model_cfg = cfg.get("model", {})
        self.weights = str(weights or model_cfg.get("weights", "yolo11n.pt"))
        fallback = model_cfg.get("fallback_weights", "yolo11n.pt")
        if not Path(self.weights).exists() and not Path(resolve_path(self.weights)).exists():
            print(f"[提示] 未找到权重 {self.weights}，回退到 {fallback}")
            self.weights = fallback
        resolved = resolve_path(self.weights)
        self.weights = str(resolved if Path(resolved).exists() else self.weights)

        self.detector = BehaviorDetector(
            weights=self.weights,
            device=str(model_cfg.get("device", "0")),
            imgsz=int(model_cfg.get("imgsz", 640)),
            conf=float(model_cfg.get("conf", 0.35)),
            iou=float(model_cfg.get("iou", 0.45)),
            half=bool(model_cfg.get("half", False)),
            max_det=int(model_cfg.get("max_det", 300)),
            class_names=self.zh_names,
            per_class_conf=model_cfg.get("per_class_conf"),
        )

        analysis_cfg = cfg.get("analysis", {})
        self.analytics = BehaviorAnalytics(
            zh_names=self.zh_names,
            attention_classes=analysis_cfg.get("attention_classes", []),
            head_up_classes=analysis_cfg.get("head_up_classes", []),
            anomaly_classes=analysis_cfg.get("anomaly_classes", []),
            fps_assume=float(analysis_cfg.get("fps_assume", 25)),
            window_seconds=float(analysis_cfg.get("window_seconds", 60)),
        )
        self.alerts = AlertManager(
            rules=class_info["anomaly_rules"],
            cooldown=float(analysis_cfg.get("alert_cooldown", 10)),
            log_path=resolve_path(REPORTS_DIR) / "alerts.jsonl",
        )

        tracking_cfg = cfg.get("tracking", {})
        self.use_track = bool(tracking_cfg.get("enable", True))
        self.tracker = SimpleTracker(max_age=int(tracking_cfg.get("max_age", 30)))

        self.cap: Optional[cv2.VideoCapture] = None
        self.mode: Optional[str] = None
        self.worker: Optional[InferenceWorker] = None
        self.fps = FPSCounter()
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.last_frame: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._last_annotated: Optional[np.ndarray] = None
        self.chart_every = 12   # 图表刷新降频，避免拖慢视频播放
        self._chart_tick = 0
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.paused = False
        self._video_total = 0
        self._video_fps = 25.0
        self._frame_interval_ms = 40   # 按视频真实帧率换算的取帧间隔（毫秒）
        self._play_start_ts = None     # 播放起点（墙上时钟），用于对齐真实时长
        self._play_start_frame = 0
        self.video_out_path: Optional[Path] = None

        self._init_ui()
        self.detector.load()
        self._start_worker()

    # ------------------------------------------------------------------ #
    def _build_legend(self) -> QWidget:
        """窗口顶部：颜色 → 课堂行为类别 的图例。"""
        bar = QFrame()
        bar.setObjectName("legendBar")
        bar.setStyleSheet("#legendBar{background:#f6f7f9;border:1px solid #e3e5e9;border-radius:6px;}")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(6)

        title = QLabel("类别图例")
        title.setStyleSheet("color:#666;font-size:12px;font-weight:600")
        lay.addWidget(title)

        classes = self.class_info.get("classes") or []
        for i, name in enumerate(self.zh_names):
            bgr = self.colors[i] if i < len(self.colors) else (0, 200, 0)
            b, g, r = (int(v) for v in bgr)
            lum = 0.114 * b + 0.587 * g + 0.299 * r
            fg = "#000000" if lum >= 150 else "#ffffff"
            chip = QLabel(name)
            chip.setStyleSheet(
                f"background:rgb({r},{g},{b});color:{fg};border-radius:9px;"
                "padding:3px 11px;font-size:13px;font-weight:600;"
            )
            desc = str(classes[i].get("desc") or "") if i < len(classes) else ""
            chip.setToolTip(f"{name}：{desc}" if desc else name)
            lay.addWidget(chip)

        lay.addStretch(1)
        tip = QLabel("画面里框和标签的颜色与这里一一对应")
        tip.setStyleSheet("color:#9aa0a6;font-size:11px")
        lay.addWidget(tip)
        return bar

    def _init_ui(self) -> None:
        self.setWindowTitle(self.cfg.get("ui", {}).get("window_title", "课堂行为检测系统"))
        self.resize(1480, 880)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addWidget(self._build_legend())     # 顶部类别图例
        root = QHBoxLayout()
        outer.addLayout(root, 1)

        # ---------------- 左侧控制区 ---------------- #
        left = QVBoxLayout()
        box_ctrl = QGroupBox("功能")
        ctrl = QVBoxLayout(box_ctrl)
        self.btn_image = QPushButton("本地图片检测")
        self.btn_video = QPushButton("本地视频检测")
        self.btn_camera = QPushButton("摄像头实时检测")
        self.btn_stop = QPushButton("停止")
        self.btn_snapshot = QPushButton("抓拍当前画面（存为图片）")
        self.btn_report = QPushButton("导出统计报表")
        for b in [self.btn_image, self.btn_video, self.btn_camera, self.btn_stop, self.btn_snapshot, self.btn_report]:
            b.setMinimumHeight(34)
            ctrl.addWidget(b)
        self.chk_record = QCheckBox("保存检测结果视频（mp4）")
        ctrl.addWidget(self.chk_record)
        left.addWidget(box_ctrl)

        box_param = QGroupBox("参数")
        param = QVBoxLayout(box_param)
        self.chk_boxes = QCheckBox("显示识别框（取消勾选 = 看原始画面）")
        self.chk_boxes.setChecked(True)
        param.addWidget(self.chk_boxes)

        self.chk_track = QCheckBox("启用目标跟踪（行为时长统计）")
        self.chk_track.setChecked(self.use_track)
        param.addWidget(self.chk_track)

        param.addWidget(QLabel("置信度阈值"))
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(5, 95)
        self.sld_conf.setValue(int(float(self.cfg.get("model", {}).get("conf", 0.35)) * 100))
        param.addWidget(self.sld_conf)
        self.lbl_conf = QLabel(f"{self.sld_conf.value() / 100:.2f}")
        param.addWidget(self.lbl_conf)

        param.addWidget(QLabel("推理尺寸"))
        self.cmb_imgsz = QComboBox()
        self.cmb_imgsz.addItems(["416", "512", "640", "768", "960"])
        self.cmb_imgsz.setCurrentText(str(self.cfg.get("model", {}).get("imgsz", 640)))
        param.addWidget(self.cmb_imgsz)
        left.addWidget(box_param)

        box_alert = QGroupBox("异常预警")
        al = QVBoxLayout(box_alert)
        self.list_alerts = QListWidget()
        al.addWidget(self.list_alerts)
        left.addWidget(box_alert, 1)

        # ---------------- 中间视频区 ---------------- #
        middle = QVBoxLayout()
        self.lbl_video = QLabel("请选择 图片 / 视频 / 摄像头 开始检测")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(880, 620)
        self.lbl_video.setStyleSheet("background:#000;color:#888;border:1px solid #333;font-size:16px")
        self.lbl_video.setScaledContents(False)   # 居中显示，不平铺拉伸
        middle.addWidget(self.lbl_video, 1)
        # 视频播放控制条：暂停/继续 + 进度拖动 + 时间显示
        ctrl_bar = QHBoxLayout()
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setMinimumWidth(70)
        self.btn_pause.setEnabled(False)
        self.sld_progress = QSlider(Qt.Horizontal)
        self.sld_progress.setRange(0, 0)
        self.sld_progress.setEnabled(False)
        self.lbl_progress = QLabel("--:-- / --:--")
        self.lbl_progress.setMinimumWidth(110)
        ctrl_bar.addWidget(self.btn_pause)
        ctrl_bar.addWidget(self.sld_progress, 1)
        ctrl_bar.addWidget(self.lbl_progress)
        middle.addLayout(ctrl_bar)
        self.lbl_status = QLabel("模型：未加载")
        # 关键：这行状态文字每帧都在变（FPS、目标数、0 目标时还会追加长提示）。
        # QLabel 默认的最小宽度 = 文字宽度，文字变长就会把中间的显示区撑大，
        # 导致画面被重新缩放、左右黑边边界来回跳（实测窗口 1340px 时跳 365px）。
        # 改成 Ignored 后，宽度只由布局决定，与文字长短无关。
        self.lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_status.setMinimumWidth(240)
        middle.addWidget(self.lbl_status)

        # ---------------- 右侧统计区 ---------------- #
        right = QVBoxLayout()
        box_counts = QGroupBox("实时行为统计")
        bc = QVBoxLayout(box_counts)
        self.table = QTableWidget(len(self.zh_names), 2)
        self.table.setHorizontalHeaderLabels(["行为", "人数"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, name in enumerate(self.zh_names):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem("0"))
        bc.addWidget(self.table)
        self.lbl_kpi = QLabel("人数：0    抬头率：0.0%")
        self.lbl_kpi.setStyleSheet("font-size:15px;font-weight:bold;color:#1a7f37")
        bc.addWidget(self.lbl_kpi)
        right.addWidget(box_counts)

        box_chart = QGroupBox("可视化")
        ch = QVBoxLayout(box_chart)
        self.figure = Figure(figsize=(4.6, 5.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        ch.addWidget(self.canvas)
        right.addWidget(box_chart, 1)

        splitter = QSplitter()
        w_left, w_mid, w_right = QWidget(), QWidget(), QWidget()
        w_left.setLayout(left)
        w_mid.setLayout(middle)
        w_right.setLayout(right)
        splitter.addWidget(w_left)
        splitter.addWidget(w_mid)
        splitter.addWidget(w_right)
        splitter.setSizes([260, 880, 360])
        root.addWidget(splitter)

        # ---------------- 信号 ---------------- #
        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.sld_progress.sliderReleased.connect(self._on_slider_released)
        self.btn_snapshot.clicked.connect(self.snapshot)
        self.btn_report.clicked.connect(self.export_report)
        self.sld_conf.valueChanged.connect(lambda v: (self.lbl_conf.setText(f"{v / 100:.2f}"), self._apply_conf()))
        self.cmb_imgsz.currentTextChanged.connect(self._apply_imgsz)
        self.chk_track.stateChanged.connect(self._apply_track)
        self.chk_boxes.stateChanged.connect(self._display_current)

        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)

    # ------------------------------------------------------------------ #
    def _start_worker(self) -> None:
        use_simple_tracker = not self.use_track
        self.worker = InferenceWorker(
            self.detector,
            self.analytics,
            self.alerts,
            self.tracker if use_simple_tracker else None,
            use_track=self.use_track,
            colors=self.colors,
        )
        self.worker.finished_frame.connect(self._on_frame)
        self.worker.new_alerts.connect(self._on_alerts)
        self.worker.start()

    def _restart_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1500)
        self.analytics.reset()
        self.tracker.reset()
        self.detector.reset_tracking()
        self._start_worker()

    # ------------------------------------------------------------------ #
    # 按钮回调
    # ------------------------------------------------------------------ #
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str) -> None:
        """读取单张图片 -> 检测 -> 画框 -> 居中显示（也被命令行/自测调用）。"""
        self.stop_detect(quiet=True)
        self.analytics.reset()
        self.detector.reset_tracking()
        self.mode = "image"
        frame = imread(path)
        if frame is None:
            frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", "图片读取失败")
            return
        self.lbl_status.setText(f"当前模式：图片检测｜{Path(path).name}")
        detections = self.detector.predict(frame)
        self.analytics.update(detections, ts=time.time())
        annotated = self.detector.annotate(frame.copy(), detections, colors=self.colors)
        self._last_raw = frame
        self._last_annotated = annotated
        self._display_current()
        self._refresh_panels(self.analytics.snapshot(), detections)
        self._draw_charts()
        out = report_mod.save_annotated_image(annotated)
        names = {}
        for d in detections:
            names[d.cls_name] = names.get(d.cls_name, 0) + 1
        self.lbl_status.setText(
            f"图片检测完成：{len(detections)} 个目标 {names if names else ''}｜结果已保存 {out}"
        )

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str) -> None:
        """打开视频文件并初始化播放控制（暂停/进度条）。"""
        self.stop_detect(quiet=True)
        self._clear_display("正在打开视频…")
        self._restart_worker()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "视频打开失败")
            self._clear_display()
            return
        self.mode = "video"
        self.paused = False
        self._play_start_ts = None
        self._play_start_frame = 0
        self._video_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.sld_progress.setRange(0, max(self._video_total, 1))
        self.sld_progress.setValue(0)
        self.sld_progress.setEnabled(True)
        self.btn_pause.setText("暂停")
        self.btn_pause.setEnabled(True)
        self.lbl_progress.setText(f"00:00 / {self._fmt_time(self._video_total / max(self._video_fps, 1e-3))}")
        # 关键：按视频真实帧率播放。之前固定 15ms 取一帧 = 66fps，12fps 的视频会播快 5 倍。
        self._frame_interval_ms = int(round(1000.0 / max(self._video_fps, 1.0)))
        self._frame_interval_ms = max(5, min(self._frame_interval_ms, 200))
        self.lbl_status.setText(
            f"当前模式：视频检测｜{Path(path).name}｜{self._video_total} 帧 / {self._video_fps:.1f}fps"
            f"｜按原速播放（每帧 {self._frame_interval_ms}ms）"
        )
        # 用半帧间隔轮询（到点才真正读帧），把 QTimer 抖动的影响降到最低
        self.timer.start(max(15, self._frame_interval_ms // 2))

    def open_camera(self) -> None:
        cam_cfg = self.cfg.get("camera", {})
        self.stop_detect(quiet=True)
        self._clear_display("正在打开摄像头…")
        self._restart_worker()
        self.cap = cv2.VideoCapture(int(cam_cfg.get("index", 0)))
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "摄像头打开失败（检查是否被其它程序占用）")
            self._clear_display()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg.get("width", 1280)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg.get("height", 720)))
        self.mode = "camera"
        self.paused = False
        self._video_total = 0
        self.sld_progress.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.lbl_progress.setText("--:-- / --:--")
        cam_fps = float(cam_cfg.get("target_fps", 15))
        self._frame_interval_ms = max(5, int(round(1000.0 / max(cam_fps, 1.0))))
        self.lbl_status.setText(f"当前模式：摄像头实时检测｜取帧 {cam_fps:.0f}fps")
        # 摄像头没有"视频时长"概念，按 target_fps 正常节奏取帧即可
        self.timer.start(self._frame_interval_ms)

    def stop_detect(self, quiet: bool = False) -> None:
        self.timer.stop()
        self._release_video_writer()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = None
        self.paused = False
        self._video_total = 0
        self.sld_progress.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.lbl_progress.setText("--:-- / --:--")
        # 停止后清空画面，回到黑屏待机（否则会残留摄像头最后一帧）
        self._clear_display("已停止检测（黑屏待机）")
        if not quiet:
            self.lbl_status.setText("已停止检测")

    def _clear_display(self, text: str = "请选择 图片 / 视频 / 摄像头 开始检测") -> None:
        """把中间显示区恢复成黑屏 + 提示文字。"""
        self.last_frame = None
        self._last_raw = None
        self._last_annotated = None
        self.lbl_video.clear()
        self.lbl_video.setPixmap(QPixmap())
        self.lbl_video.setText(text)
        self.lbl_video.setStyleSheet("background:#000;color:#888;border:1px solid #333;font-size:16px")

    def _on_timer(self) -> None:
        if self.cap is None:
            return
        if self.mode == "video" and self.paused:
            return
        if self.mode == "video" and self._video_fps > 0:
            # 按墙上时钟推算"此刻应该播到第几帧"，避免 QTimer 抖动导致播放偏慢/偏快
            now = time.time()
            if self._play_start_ts is None:
                self._play_start_ts = now
                self._play_start_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            expected = self._play_start_frame + int((now - self._play_start_ts) * self._video_fps)
            current = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if expected <= current:
                return  # 还没到下一帧的时间点，本次 tick 跳过
            # 落后时用 grab() 快速跳过中间帧（最多补 4 帧）：
            # 这样即使推理/绘图拖慢了界面，整体播放时长仍与视频一致。
            skip = min(expected - current, 4) - 1
            for _ in range(max(0, skip)):
                if not self.cap.grab():
                    break
        ok, frame = self.cap.read()
        if not ok:
            # 视频结束：不退出、不清屏，停在最后一帧，允许拖动进度条回看或重播
            if self.mode == "video":
                self.timer.stop()
                self.paused = True
                self.btn_pause.setText("重播")
                self.btn_pause.setEnabled(True)
                self._draw_charts()
                self.lbl_status.setText("视频播放结束（可拖动进度条回看，或点「重播」从头播放）")
            else:
                self.stop_detect()
                self.lbl_status.setText("摄像头已断开")
            return
        self.fps.tick()
        if self.mode == "video" and self._video_total > 0:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.sld_progress.blockSignals(True)
            self.sld_progress.setValue(min(pos, self._video_total))
            self.sld_progress.blockSignals(False)
            self.lbl_progress.setText(f"{self._fmt_time(pos / max(self._video_fps, 1e-3))} / "
                                      f"{self._fmt_time(self._video_total / max(self._video_fps, 1e-3))}")
        if self.worker is not None:
            self.worker.submit(frame)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _toggle_pause(self) -> None:
        """暂停 / 继续 / 重播。"""
        if self.mode != "video" or self.cap is None:
            return
        if self._video_ended():
            self._seek_video(0)
            self.paused = False
            self.btn_pause.setText("暂停")
            self.timer.start(15)
            return
        self.paused = not self.paused
        self._play_start_ts = None   # 暂停后重新计时，保证继续时不跳帧
        self.btn_pause.setText("继续" if self.paused else "暂停")

    def _video_ended(self) -> bool:
        return self.cap is not None and self._video_total > 0 and \
            self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._video_total - 0.5

    def _seek_video(self, value: int) -> None:
        """拖动进度条跳到指定帧（保持暂停状态）。"""
        if self.mode != "video" or self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(value)))
        self._play_start_ts = None
        # 跳转后重置跟踪，避免 ID 串号；累计统计保留
        self.tracker.reset()
        self.detector.reset_tracking()
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.lbl_progress.setText(f"{self._fmt_time(pos / max(self._video_fps, 1e-3))} / "
                                  f"{self._fmt_time(self._video_total / max(self._video_fps, 1e-3))}")
        if self.paused:
            self.btn_pause.setText("继续")

    def _on_slider_released(self) -> None:
        self._seek_video(self.sld_progress.value())

    # ------------------------------------------------------------------ #
    # worker 回调
    # ------------------------------------------------------------------ #
    def _on_frame(self, annotated: np.ndarray, raw: np.ndarray, detections: List[Any], snapshot: Dict[str, Any]) -> None:
        # 只有视频/摄像头模式才接收 worker 的画面；否则（图片模式或已停止）忽略，
        # 避免上一路视频/摄像头的残留帧覆盖当前显示。
        if self.mode not in {"video", "camera"}:
            return
        self._last_annotated = annotated
        self._last_raw = raw
        self.last_frame = annotated
        self._display_current()
        self._refresh_panels(snapshot, detections)
        if self.chk_record.isChecked() and self.mode in {"video", "camera"}:
            self._write_video_frame(annotated)
        self._chart_tick += 1
        if self._chart_tick % self.chart_every == 0:
            self._draw_charts()

    # ------------------------------------------------------------------ #
    def _write_video_frame(self, frame: np.ndarray) -> None:
        """把标注后的帧写入结果视频（首次调用时创建写入器）。"""
        try:
            if self.video_writer is None:
                out_dir = ensure_dir(resolve_path(REPORTS_DIR) / "videos")
                stamp = time.strftime("%Y%m%d_%H%M%S")
                self.video_out_path = out_dir / f"检测结果_{stamp}.mp4"
                h, w = frame.shape[:2]
                fps = float(self.cfg.get("camera", {}).get("target_fps", 15))
                self.video_writer = cv2.VideoWriter(
                    str(self.video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            self.video_writer.write(frame)
        except Exception as exc:  # pragma: no cover
            print(f"[record] 录制失败：{exc}")
            self._release_video_writer()

    def _release_video_writer(self) -> None:
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
            if self.video_out_path is not None:
                QMessageBox.information(self, "录制完成", f"检测结果视频已保存：\n{self.video_out_path}")
                self.video_out_path = None

    def snapshot(self) -> None:
        """抓拍当前画面（摄像头 / 视频 / 图片均可），保存为图片并给出当前帧统计。"""
        if self.last_frame is None:
            QMessageBox.information(self, "提示", "还没有可抓拍的画面，请先打开图片 / 视频 / 摄像头")
            return
        path = report_mod.save_annotated_image(self.last_frame)
        snap = self.analytics.snapshot()
        QMessageBox.information(
            self,
            "抓拍完成",
            f"已保存：{path}\n\n当前画面：{snap.get('persons', 0)} 人，"
            f"抬头率 {snap.get('head_up_rate', 0):.1f}%  专注率 {snap.get('attention_rate', 0):.1f}%",
        )
        self.lbl_status.setText(f"已抓拍：{path}")

    def _on_alerts(self, alerts: List[Alert]) -> None:
        for alert in alerts:
            item = QListWidgetItem(alert.text())
            if alert.level == "高":
                item.setForeground(Qt.red)
            self.list_alerts.insertItem(0, item)
        while self.list_alerts.count() > int(self.cfg.get("ui", {}).get("max_alert_rows", 200)):
            self.list_alerts.takeItem(self.list_alerts.count() - 1)

    # ------------------------------------------------------------------ #
    # 界面刷新
    # ------------------------------------------------------------------ #
    def _display_current(self) -> None:
        """按「显示识别框」勾选状态切换显示：勾选=带框画面，取消=原始画面。

        这样演示时可以先给观众看"原图"，再勾上显示识别框，避免被误认为图片本来就带框。
        """
        frame = self._last_annotated if self.chk_boxes.isChecked() else self._last_raw
        if frame is not None:
            self._show_frame(frame)
    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_panels(self, snapshot: Dict[str, Any], detections: List[Any]) -> None:
        counts = snapshot.get("counts", {})
        for i, name in enumerate(self.zh_names):
            self.table.item(i, 1).setText(str(counts.get(name, 0)))
        self.lbl_kpi.setText(
            f"人数：{snapshot.get('persons', 0)}"
            f"    抬头率：{snapshot.get('head_up_rate', 0):.1f}%"
            f"    专注/参与率：{snapshot.get('attention_rate', 0):.1f}%"
            f"    活跃轨迹：{snapshot.get('active_tracks', 0)}"
        )
        fps = self.fps.fps if self.mode in {"video", "camera"} else 0.0
        hint = ""
        if len(detections) == 0:
            # 0 目标时给出可操作提示：这是「课堂行为」模型，非教室画面（自拍/室内近景）
            # 不会有合理输出，实测最高置信度只有 0.05 左右。
            hint = "（未检出：本模型只识别课堂场景，请用教室图片/视频，或把置信度调到 0.15 以下试）"
        self.lbl_status.setText(
            f"模型：{Path(self.weights).name}｜跟踪：{'开' if self.use_track else '关'}"
            f"｜推理尺寸：{self.detector.imgsz}｜播放：{1000.0/max(self._frame_interval_ms,1):.0f}fps｜推理FPS：{fps:.1f}｜本帧目标：{len(detections)}{hint}"
        )

    def _draw_charts(self) -> None:
        summary = self.analytics.summary()
        snap = self.analytics.snapshot()
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        behavior = summary.get("behavior_seconds", {})
        total_time = sum(behavior.values()) if behavior else 0.0

        if total_time > 0.05:
            # 视频/摄像头模式：画累计时长
            names = list(behavior.keys())
            values = [behavior[n] for n in names]
            colors = ["#4C8BF5" if v > 0 else "#DDDDDD" for v in values]
            ax1.set_title("课堂行为时长累计", fontsize=10)
            ax1.set_ylabel("累计秒")
            attention = summary.get("head_up_rate", 0.0)
            pie_title = "课堂抬头率（只算抬头姿态）"
        else:
            # 图片模式（或刚启动，还没累计时长）：改画「当前这帧各行为人数」，
            # 否则柱子高度全是 0，看起来像没画出图。
            counts = snap.get("counts", {})
            names = list(self.zh_names)
            values = [counts.get(n, 0) for n in names]
            colors = ["#41B883" if v > 0 else "#DDDDDD" for v in values]
            ax1.set_title("当前画面各行为人数", fontsize=10)
            ax1.set_ylabel("人数")
            attention = snap.get("head_up_rate", 0.0)
            pie_title = "本帧抬头率（只算抬头姿态）"

        ax1.bar(range(len(names)), values, color=colors)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=30, fontsize=8)
        if values:
            for i, v in enumerate(values):
                if v > 0:
                    ax1.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)

        ax2 = self.figure.add_subplot(212)
        ax2.pie([max(attention, 0), max(100 - attention, 0)],
                labels=["专注", "非专注"], autopct="%1.1f%%",
                colors=["#41B883", "#E6E6E6"], startangle=90,
                textprops={"fontsize": 9}, wedgeprops={"width": 0.45})
        ax2.set_title(pie_title, fontsize=10)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def export_report(self) -> None:
        summary = self.analytics.summary()
        students = self.analytics.per_student_table()
        try:
            csv_path = report_mod.export_timeline_csv(self.analytics)
            charts = report_mod.save_charts(summary)
            trend = report_mod.save_trend_chart(self.analytics)
            md = report_mod.write_markdown_report(summary, students, self.alerts.alerts)
            xlsx = None
            if self.cfg.get("report", {}).get("excel", True):
                xlsx = report_mod.export_excel(self.analytics, self.alerts.alerts, summary=summary)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{exc}")
            return
        paths = [str(csv_path), str(md)] + [str(p) for p in charts]
        if trend:
            paths.append(str(trend))
        if xlsx:
            paths.insert(0, str(xlsx))
        QMessageBox.information(self, "导出完成", "已生成：\n" + "\n".join(paths))

    # ------------------------------------------------------------------ #
    def _apply_conf(self) -> None:
        self.detector.conf = self.sld_conf.value() / 100.0

    def _apply_imgsz(self, text: str) -> None:
        try:
            self.detector.imgsz = int(text)
        except ValueError:
            pass

    def _apply_track(self) -> None:
        self.use_track = self.chk_track.isChecked()
        self._restart_worker()
        self.lbl_status.setText(f"跟踪已{'开启' if self.use_track else '关闭'}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_detect(quiet=True)
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1500)
        event.accept()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="课堂行为检测系统（PyQt5 桌面端）")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    parser.add_argument("--source", type=str, default=None, help="启动后直接打开的视频/图片")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_app_config(args.config)
    class_info = load_classes()

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow(cfg, class_info, weights=args.weights)
    window.show()

    if args.source:
        src = Path(args.source)
        if src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            window.cap = cv2.VideoCapture(str(src))
            window.mode = "video"
            window.timer.start(15)
        else:
            window.load_image(str(src))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
