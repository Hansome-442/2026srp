"""YOLOv11 训练脚本。

用法
----
python scripts/train.py                       # 使用 configs/train.yaml
python scripts/train.py --model yolo11n.pt --epochs 100 --batch 8 --name try_n
python scripts/train.py --smoke               # 1 轮小图快跑，只验证流程是否通
python scripts/train.py --resume              # 断点续训（自动找最近一次）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# dataset.yaml 里的 path 写成相对路径（data/processed），ultralytics 是按
# 「当前工作目录」去找的。这里统一切到项目根目录，保证不管从哪个目录执行
# 训练都能找到数据，也让项目可以放在任意盘符（不必是作者机器上的 E:\）。
os.chdir(PROJECT_ROOT)

from src.utils import CONFIG_DIR, RUNS_DIR, get_logger, load_yaml, resolve_path  # noqa: E402

LOG = get_logger("train")

# 配置文件里除了这几个"由脚本自己处理"的键，其余全部原样透传给 ultralytics.train()
# （此前用白名单，导致 cls_pw / multi_scale / momentum 等参数被静默丢弃）
RESERVED_KEYS = {"data", "model", "project", "name"}


def build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv11 课堂行为检测模型")
    parser.add_argument("--config", type=str, default=str(CONFIG_DIR / "train.yaml"))
    parser.add_argument("--data", type=str, default=None, help="dataset.yaml 路径")
    parser.add_argument("--model", type=str, default=None, help="预训练权重或模型 yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：1 轮 + 小图 + 小 batch")
    parser.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    return parser.parse_args(argv)


def resolve_weights(model: str) -> str:
    """本地权重优先，找不到就交给 ultralytics 自动下载。"""
    p = Path(str(model))
    if p.is_absolute() and p.exists():
        return str(p)
    local = PROJECT_ROOT / p
    if local.exists():
        return str(local)
    course = Path("E:/yolov11/weights") / p.name
    if course.exists():
        LOG.info("使用课程自带权重：%s", course)
        return str(course)
    LOG.info("本地未找到 %s，将按名称自动下载/解析", model)
    return str(model)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_args(argv)
    cfg: Dict[str, Any] = load_yaml(args.config)

    data = str(resolve_path(args.data or cfg.get("data", "data/dataset.yaml")))
    if not Path(data).exists():
        LOG.error("找不到数据集配置 %s，请先运行：python scripts/prepare_dataset.py --voc-dir <你的数据目录>", data)
        return 2

    # ---- 打印数据集版本：三台机器各自训练时，靠它判断数据是不是同一份 ----
    ver_file = PROJECT_ROOT / "data" / "DATASET_VERSION.txt"
    if ver_file.exists():
        info = {}
        for line in ver_file.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        LOG.info(
            "数据集版本 %s ｜ 图片 %s ｜ 指纹 %s ｜ 更新于 %s",
            info.get("version", "?"), info.get("images", "?"),
            info.get("fingerprint", "?"), info.get("updated", "?"),
        )
        LOG.info("（三台机器训练时请核对指纹一致，否则指标不可比）")
    else:
        LOG.warning("没有 data/DATASET_VERSION.txt，建议先跑：python scripts/dataset_fingerprint.py")

    model = resolve_weights(args.model or cfg.get("model", "yolo11s.pt"))
    train_kwargs: Dict[str, Any] = {k: v for k, v in cfg.items() if k not in RESERVED_KEYS}

    overrides = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
    }
    for key, value in overrides.items():
        if value is not None:
            train_kwargs[key] = value

    project = str(resolve_path(args.project or cfg.get("project", RUNS_DIR / "train")))
    name = args.name or cfg.get("name", "classroom_yolo11s")

    # ---- 断点续训：自动定位 last.pt，并只把 resume 交给 ultralytics ----
    if args.resume:
        last_pt = Path(project) / name / "weights" / "last.pt"
        if args.model:
            last_pt = Path(resolve_weights(args.model))
        if not last_pt.exists():
            LOG.error("找不到断点文件 %s，请确认 --project/--name 是否正确", last_pt)
            return 2
        LOG.info("断点续训：%s", last_pt)
        from ultralytics import YOLO

        model_obj = YOLO(str(last_pt))
        model_obj.train(resume=True)
        save_dir = Path(getattr(model_obj.trainer, "save_dir", last_pt.parent.parent))
        _report_result(save_dir)
        return 0

    if args.smoke:
        train_kwargs.update({"epochs": 1, "imgsz": 320, "batch": 2, "workers": 0, "patience": 1, "plots": True})
        name = args.name or "smoke"
        LOG.info("冒烟模式：epochs=1, imgsz=320, batch=2")
    if args.no_amp:
        train_kwargs["amp"] = False

    train_kwargs.update({"data": data, "project": project, "name": name, "exist_ok": True})

    LOG.info("模型：%s", model)
    LOG.info("数据：%s", data)
    LOG.info("关键超参：epochs=%s imgsz=%s batch=%s device=%s workers=%s",
             train_kwargs.get("epochs"), train_kwargs.get("imgsz"), train_kwargs.get("batch"),
             train_kwargs.get("device"), train_kwargs.get("workers"))

    from ultralytics import YOLO

    model_obj = YOLO(model)
    results = model_obj.train(**train_kwargs)

    save_dir = Path(getattr(model_obj.trainer, "save_dir", Path(project) / name))
    _report_result(save_dir)

    try:
        metrics = results.results_dict  # type: ignore[attr-defined]
        LOG.info("最终指标：%s", {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception:
        pass
    return 0


def _report_result(save_dir: Path) -> None:
    """打印最佳权重位置，并复制一份到 runs/best.pt 便于界面直接引用。"""
    best = save_dir / "weights" / "best.pt"
    LOG.info("训练结束，结果目录：%s", save_dir)
    if not best.exists():
        LOG.warning("未找到 best.pt，请检查训练日志")
        return
    LOG.info("最佳权重：%s", best)
    LOG.info("把它填到 configs/app.yaml 的 model.weights 即可在界面中使用：\n  weights: %s", best.as_posix())
    try:
        target = RUNS_DIR / "best.pt"
        shutil.copy2(best, target)
        LOG.info("已复制一份到 %s", target)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
