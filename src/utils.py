"""通用工具函数：配置加载、类别表、中文路径图片读写、日志、绘图字体。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

_LOGGER_NAME = "classroom"


# --------------------------------------------------------------------------- #
# 目录 / 文件
# --------------------------------------------------------------------------- #
def ensure_dir(path: Any) -> Path:
    """创建目录（已存在则忽略），返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: Any, base: Optional[Path] = None) -> Path:
    """相对路径按项目根目录解析。"""
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_yaml(path: Any) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: Any, path: Any) -> None:
    import yaml

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def dump_json(obj: Any, path: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label(text: Any) -> str:
    """标注文本归一化：去空格、下划线，英文转小写，便于别名匹配。"""
    return str(text).strip().replace(" ", "").replace("_", "").lower()


def load_classes(config_path: Any = None) -> Dict[str, Any]:
    """读取 configs/classes.yaml，返回统一的类别信息字典。

    返回键：
        classes     原始类别列表（含 id/en/zh/kind/color/desc）
        names       英文类别名列表（顺序 = 类别 id 顺序）
        zh          中文类别名列表
        colors      BGR 颜色元组列表（OpenCV 绘制用）
        kinds       类别属性列表（normal/positive/anomaly）
        num_classes 类别数
        alias       标注文本 -> 中文类别名
        alias_zh    标注文本 -> 类别 id
        alias_en    英文类别名 -> 类别 id
        anomaly_rules / dataset
    """
    cfg = load_yaml(config_path or (CONFIG_DIR / "classes.yaml"))
    classes: List[Dict[str, Any]] = sorted(cfg.get("classes", []), key=lambda c: int(c["id"]))

    ids = [int(c["id"]) for c in classes]
    if ids != list(range(len(classes))):
        raise ValueError(f"configs/classes.yaml 中类别 id 必须从 0 连续编号，当前为 {ids}")

    zh_names = [str(c["zh"]) for c in classes]
    alias: Dict[str, str] = {}
    for key, value in (cfg.get("aliases") or {}).items():
        if value not in zh_names:
            continue
        alias[normalize_label(key)] = str(value)
    for c in classes:
        alias[normalize_label(c["zh"])] = str(c["zh"])
        alias[normalize_label(c["en"])] = str(c["zh"])

    alias_zh = {k: zh_names.index(v) for k, v in alias.items()}

    return {
        "raw": cfg,
        "classes": classes,
        "names": [str(c["en"]) for c in classes],
        "zh": zh_names,
        "colors": [tuple(int(v) for v in c.get("color", [0, 255, 0])) for c in classes],
        "kinds": [str(c.get("kind", "normal")) for c in classes],
        "num_classes": len(classes),
        "alias": alias,
        "alias_zh": alias_zh,
        "alias_en": {normalize_label(c["en"]): i for i, c in enumerate(classes)},
        "anomaly_rules": cfg.get("anomaly_rules") or [],
        "dataset": cfg.get("dataset") or {},
    }


def load_app_config(config_path: Any = None) -> Dict[str, Any]:
    return load_yaml(config_path or (CONFIG_DIR / "app.yaml"))


# --------------------------------------------------------------------------- #
# 中文路径图片读写（cv2.imread 不支持中文路径）
# --------------------------------------------------------------------------- #
def imread(path: Any, flags: Optional[int] = None) -> Optional[np.ndarray]:
    import cv2

    flags = cv2.IMREAD_COLOR if flags is None else flags
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite(path: Any, image: np.ndarray) -> bool:
    import cv2

    p = Path(path)
    ensure_dir(p.parent)
    ext = p.suffix if p.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# 绘图（中文）
# --------------------------------------------------------------------------- #
def setup_matplotlib() -> None:
    """让 matplotlib 正常显示中文与负号。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
class FPSCounter:
    """滑动平均帧率统计。"""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self.times: List[float] = []
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        self.times.append(now)
        if len(self.times) > self.window:
            self.times.pop(0)
        if len(self.times) >= 2:
            dt = self.times[-1] - self.times[0]
            self.fps = (len(self.times) - 1) / dt if dt > 0 else 0.0
        return self.fps


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1].copy()


# --------------------------------------------------------------------------- #
# 感知哈希（数据去重 / 泄漏检查）
# --------------------------------------------------------------------------- #
def dhash(path: Any, size: int = 8) -> Optional[int]:
    """64 位差值哈希：缩放到 (size+1)×size 灰度图，逐行比较相邻像素。

    对亮度/对比度变化不敏感，适合检测「同一课堂视频的相邻抽帧」这类近似重复。
    """
    try:
        from PIL import Image

        with Image.open(str(path)) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            pixels = list(im.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    """两个整数的汉明距离。"""
    return bin(int(a) ^ int(b)).count("1")
