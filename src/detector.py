"""YOLOv11 检测 / 跟踪封装。

对 ultralytics 的调用集中在这里，界面层（PyQt5 / Streamlit）和脚本层只使用
``BehaviorDetector`` 这一个入口，方便后续替换成 ONNX / TensorRT 推理。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    """单个检测框（可带跟踪 id）。"""

    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]
    track_id: Optional[int] = None

    # ----------------------------------------------------------------- #
    @property
    def x1(self) -> float:
        return float(self.xyxy[0])

    @property
    def y1(self) -> float:
        return float(self.xyxy[1])

    @property
    def x2(self) -> float:
        return float(self.xyxy[2])

    @property
    def y2(self) -> float:
        return float(self.xyxy[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def aspect(self) -> float:
        """高宽比，可用于区分站立/趴桌等粗略形态。"""
        return self.height / self.width if self.width > 1e-6 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.cls_id,
            "class_name": self.cls_name,
            "conf": round(float(self.conf), 4),
            "xyxy": [round(float(v), 2) for v in self.xyxy],
        }


class BehaviorDetector:
    """课堂行为检测器。

    参数
    ----
    weights : 权重路径（.pt / .onnx），相对路径按项目根目录解析
    device  : '0' / 'cpu'
    conf    : 置信度阈值
    iou     : NMS IoU 阈值
    imgsz   : 推理尺寸
    class_names : 中文类别名列表（configs/classes.yaml）；用于界面显示
    """

    def __init__(
        self,
        weights: Any = "yolo11n.pt",
        device: Optional[str] = "0",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
        half: bool = False,
        max_det: int = 300,
        class_names: Optional[Sequence[str]] = None,
        per_class_conf: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ) -> None:
        from src.utils import PROJECT_ROOT

        path = Path(str(weights)).expanduser()
        self.weights = str(path if path.is_absolute() else (PROJECT_ROOT / path))
        self.device = _resolve_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.half = bool(half)
        self.max_det = int(max_det)
        self.zh_names = list(class_names) if class_names else None
        # 按类阈值：强类（听讲）用高阈值压误检，弱类（趴桌/手机）用低阈值保召回。
        # 实现方式：推理用"所有类阈值里的最小值"拿候选框，再在 _to_detections 里按类过滤。
        self.per_class_conf: Dict[str, float] = {
            str(k): float(v) for k, v in (per_class_conf or {}).items() if v is not None
        }
        self.verbose = bool(verbose)
        self._model = None
        self._tracking = False

    # ----------------------------------------------------------------- #
    # 模型
    # ----------------------------------------------------------------- #
    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    @property
    def names(self) -> Dict[int, str]:
        """权重自带的类别名。"""
        try:
            return dict(self.model.names)
        except Exception:  # pragma: no cover - 权重异常时兜底
            return {}

    def display_name(self, cls_id: int) -> str:
        """优先使用 configs/classes.yaml 的中文名，其次用权重自带名字。"""
        if self.zh_names and 0 <= cls_id < len(self.zh_names):
            return self.zh_names[cls_id]
        return str(self.names.get(cls_id, f"class_{cls_id}"))

    def load(self) -> "BehaviorDetector":
        """显式加载（界面启动时预热，避免首帧卡顿）。"""
        _ = self.model
        return self

    # ----------------------------------------------------------------- #
    # 推理
    # ----------------------------------------------------------------- #
    def _common_kwargs(self) -> Dict[str, Any]:
        # 若配置了按类阈值，推理阈值取「全局 conf」与「各类阈值最小值」中更小的那个，
        # 先拿到候选框，再在 _to_detections 里按类过滤，避免弱类被全局阈值一刀切掉。
        infer_conf = self.conf
        if self.per_class_conf:
            infer_conf = min([infer_conf] + list(self.per_class_conf.values()))
        kwargs: Dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": infer_conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "verbose": self.verbose,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.half:
            kwargs["half"] = True
        return kwargs

    def _passes_threshold(self, name: str, conf: float) -> bool:
        """按类阈值过滤：没有单独配置的类沿用全局 conf。"""
        thr = self.per_class_conf.get(name, self.conf)
        return conf >= thr

    def _to_detections(self, result: Any, with_id: bool = False) -> List[Detection]:
        dets: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = None
        if with_id and getattr(boxes, "id", None) is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        for i in range(len(clss)):
            name = self.display_name(int(clss[i]))
            conf = float(confs[i])
            if not self._passes_threshold(name, conf):
                continue
            dets.append(
                Detection(
                    cls_id=int(clss[i]),
                    cls_name=name,
                    conf=conf,
                    xyxy=tuple(float(v) for v in xyxy[i][:4]),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return dets

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """单帧检测（不跟踪）。"""
        results = self.model.predict(source=frame, **self._common_kwargs())
        return self._to_detections(results[0], with_id=False) if results else []

    def track(self, frame: np.ndarray, tracker: str = "bytetrack.yaml", persist: bool = True) -> List[Detection]:
        """单帧检测 + 目标跟踪（返回带 track_id 的检测框）。"""
        kwargs = self._common_kwargs()
        kwargs.update({"persist": persist, "tracker": tracker})
        results = self.model.track(source=frame, **kwargs)
        self._tracking = True
        return self._to_detections(results[0], with_id=True) if results else []

    def reset_tracking(self) -> None:
        """切换视频/摄像头时清空跟踪状态。"""
        try:
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None:
                predictor.trackers = []
        except Exception:  # pragma: no cover
            pass
        self._tracking = False

    # ----------------------------------------------------------------- #
    # 绘制
    # ----------------------------------------------------------------- #
    def annotate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        colors: Optional[Sequence[Tuple[int, int, int]]] = None,
        show_conf: bool = True,
        show_id: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        """在原图上画框 + 中文类别标签（BGR 图，原地返回）。"""
        import cv2

        for det in detections:
            if colors and 0 <= det.cls_id < len(colors):
                color = tuple(int(v) for v in colors[det.cls_id])
            else:
                color = (0, 200, 0)
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            # 先垫一条黑边再加彩色框：纯黄/青绿这类亮色框压在浅色墙面、白板上会看不见
            cv2.rectangle(frame, p1, p2, (0, 0, 0), thickness + 2)
            cv2.rectangle(frame, p1, p2, color, thickness)
            label = det.cls_name
            if show_conf:
                label += f" {det.conf:.2f}"
            if show_id and det.track_id is not None:
                label += f" #{det.track_id}"
            self._put_text(frame, label, (p1[0], max(0, p1[1] - 6)), color)
        return frame

    @staticmethod
    def _put_text(frame: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """带底色背景的文字绘制，使用 PIL 以支持中文。

        底色 = 类别颜色；文字颜色按底色亮度自动选黑/白，
        否则遇到黄色（站立回答 [0,255,255]）这类高亮色，白字会糊成一片。
        """
        fg = _text_color_for(color)
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.fromarray(frame[:, :, ::-1])
            draw = ImageDraw.Draw(img)
            font = _load_cjk_font(18)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = int(org[0]), int(org[1]) - th - 4
            y = max(0, y)
            box = [x, y, x + tw + 8, y + th + 8]
            # 先描一圈黑边：亮色底牌压在浅色墙面上也能看清边界
            draw.rectangle([box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1], fill=(0, 0, 0))
            draw.rectangle(box, fill=tuple(int(v) for v in color[::-1]))
            draw.text((x + 4, y + 3), text, font=font, fill=fg)
            frame[:, :, :] = np.array(img)[:, :, ::-1]
        except Exception:
            import cv2

            size, base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x, y = int(org[0]), max(int(org[1]), size + 8)
            cv2.rectangle(frame, (x, y - size - 6), (x + size + 8, y + 4), color, -1)
            cv2.rectangle(frame, (x, y - size - 6), (x + size + 8, y + 4), (0, 0, 0), 1)
            cv2.putText(frame, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)


def _text_color_for(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """按 BGR 底色的感知亮度选前景色：亮底用黑字，暗底用白字。"""
    b, g, r = (float(v) for v in tuple(color)[:3])
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance >= 150 else (255, 255, 255)


def _resolve_device(device: Any) -> Any:
    """没有可用 GPU 时自动退回 CPU。

    为什么需要：ultralytics 收到 device='0' 会直接抛
    `Invalid CUDA 'device=0' requested but no CUDA devices available`，
    没有显卡的组员连界面都打不开。这里静默降级，只打印一行提示。
    """
    if device is None:
        return None
    text = str(device).strip().lower()
    if text in ("", "cpu"):
        return device
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "[提示] 未检测到可用的 CUDA 设备，已自动改用 CPU 推理。\n"
                "       CPU 下：单张图片检测约 0.5~1 秒/张，可用；\n"
                "       视频和摄像头会明显变慢（约 1~3 FPS），建议只用来演示界面。"
            )
            return "cpu"
    except Exception:
        return "cpu"
    return device


_FONT_CACHE: Dict[int, Any] = {}


def _load_cjk_font(size: int = 18):
    """加载一个可显示中文的字体（Windows 自带 msyh.ttc / simhei.ttf）。"""
    from PIL import ImageFont

    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    font = None
    for c in candidates:
        if Path(c).exists():
            try:
                font = ImageFont.truetype(c, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font
