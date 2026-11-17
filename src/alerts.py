"""异常行为预警：规则匹配 + 冷却去重 + 日志留档。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.analytics import BehaviorAnalytics
from src.utils import ensure_dir


@dataclass
class Alert:
    """一条预警记录。"""

    ts: float
    time_str: str
    track_id: int
    cls_name: str
    duration: float
    level: str
    message: str
    frame: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        who = f"学生#{self.track_id}" if self.track_id and self.track_id > 0 else "画面中"
        return f"[{self.time_str}] {self.level}｜{who} {self.cls_name} {self.duration:.1f}s — {self.message}"


class AlertManager:
    """按 configs/classes.yaml 的 anomaly_rules 触发预警。

    规则字段：class（中文类别名）、min_seconds、level、message。
    """

    def __init__(
        self,
        rules: Sequence[Dict[str, Any]],
        cooldown: float = 10.0,
        log_path: Optional[Any] = None,
        callbacks: Optional[Sequence[Callable[[Alert], None]]] = None,
    ) -> None:
        self.rules: Dict[str, Dict[str, Any]] = {}
        for rule in rules or []:
            name = str(rule.get("class", "")).strip()
            if not name:
                continue
            self.rules[name] = {
                "min_seconds": float(rule.get("min_seconds", 5.0)),
                "level": str(rule.get("level", "中")),
                "message": str(rule.get("message", f"检测到{name}")),
            }
        self.cooldown = float(cooldown)
        self.alerts: List[Alert] = []
        self._last_fired: Dict[tuple, float] = {}
        self.callbacks: List[Callable[[Alert], None]] = list(callbacks or [])
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            ensure_dir(self.log_path.parent)

    # ------------------------------------------------------------------ #
    def evaluate(self, analytics: BehaviorAnalytics, ts: Optional[float] = None) -> List[Alert]:
        """检查所有活跃学生当前行为是否超阈值，返回本次新增的预警。"""
        if not self.rules:
            return []
        ts = float(ts if ts is not None else time.time())
        new_alerts: List[Alert] = []

        for tid, state in analytics.tracks.items():
            rule = self.rules.get(state.cls_name)
            if not rule:
                continue
            duration = state.current_duration(ts)
            if duration < rule["min_seconds"]:
                continue
            key = (tid, state.cls_name)
            last = self._last_fired.get(key, 0.0)
            if ts - last < self.cooldown:
                continue
            self._last_fired[key] = ts
            alert = Alert(
                ts=ts,
                time_str=time.strftime("%H:%M:%S", time.localtime(ts)),
                track_id=int(tid),
                cls_name=state.cls_name,
                duration=float(duration),
                level=rule["level"],
                message=rule["message"],
                frame=analytics.frame_index,
            )
            new_alerts.append(alert)
            self.alerts.append(alert)
            self._write(alert)
            for cb in self.callbacks:
                try:
                    cb(alert)
                except Exception:  # pragma: no cover - 回调异常不影响主流程
                    pass
        return new_alerts

    def _write(self, alert: Alert) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.as_dict(), ensure_ascii=False) + "\n")
        except OSError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        by_level: Dict[str, int] = {}
        by_class: Dict[str, int] = {}
        for a in self.alerts:
            by_level[a.level] = by_level.get(a.level, 0) + 1
            by_class[a.cls_name] = by_class.get(a.cls_name, 0) + 1
        return {
            "total": len(self.alerts),
            "by_level": by_level,
            "by_class": by_class,
            "records": [a.as_dict() for a in self.alerts],
        }

    def reset(self) -> None:
        self.alerts.clear()
        self._last_fired.clear()
