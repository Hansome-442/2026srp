"""课堂行为分析：行为时长统计、抬头率/参与度、逐学生画像、时间序列数据。

输入是「每帧的检测框（带 track_id）」，输出是界面与报表需要的统计结果。

设计说明
--------
* 有 track_id（视频 / 摄像头 + 跟踪）时，按学生累计各类行为的持续时间，得到
  「抬头率」「参与度」「逐学生行为画像」；
* 没有 track_id（单张图片检测）时，只统计当前帧各行为的人次，不产生时长；
* 所有时长均按相邻帧时间差累加，并对单帧最大间隔做截断，避免卡顿导致虚高。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from src.detector import Detection


@dataclass
class TrackState:
    """一个学生（跟踪目标）的行为状态。"""

    track_id: int
    cls_name: str
    start_ts: float
    last_seen: float
    durations: Dict[str, float] = field(default_factory=dict)
    box: Optional[Sequence[float]] = None
    frames: int = 0
    anomaly_seconds: float = 0.0

    def accumulate(self, ts: float) -> None:
        dt = max(0.0, ts - self.last_seen)
        if dt > 0:
            self.durations[self.cls_name] = self.durations.get(self.cls_name, 0.0) + dt
        self.last_seen = ts

    def switch_to(self, cls_name: str, ts: float) -> None:
        self.accumulate(ts)
        self.cls_name = cls_name

    def current_duration(self, ts: float) -> float:
        return max(0.0, ts - self.last_seen)

    @property
    def total_seconds(self) -> float:
        return float(sum(self.durations.values()))

    def dominant(self) -> str:
        if not self.durations:
            return self.cls_name
        return max(self.durations.items(), key=lambda kv: kv[1])[0]


class BehaviorAnalytics:
    """行为统计器。"""

    def __init__(
        self,
        zh_names: Sequence[str],
        attention_classes: Sequence[str] = (),
        anomaly_classes: Sequence[str] = (),
        head_up_classes: Sequence[str] = (),
        fps_assume: float = 25.0,
        window_seconds: float = 60.0,
        track_timeout: float = 2.0,
    ) -> None:
        self.zh_names = list(zh_names)
        # 专注/参与率：听讲 + 阅读 + 书写 + 举手 + 站立（"在参与课堂"）
        self.attention_classes = set(attention_classes) or set(self.zh_names)
        # 抬头率：只算真正抬头的姿态（听讲/举手/站立），阅读和书写是低头行为，不算
        self.head_up_classes = set(head_up_classes) or {
            "抬头听讲", "举手发言", "站立回答",
        } & set(self.zh_names)
        self.anomaly_classes = set(anomaly_classes)
        self.fps_assume = float(fps_assume) if fps_assume else 25.0
        self.window_seconds = float(window_seconds)
        self.track_timeout = float(track_timeout)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.tracks: Dict[int, TrackState] = {}
        self.finished: List[TrackState] = []
        self.frame_records: List[Dict[str, Any]] = []
        self.window: Deque[Dict[str, Any]] = deque()
        self.last_ts: Optional[float] = None
        self.frame_index = 0
        self.peak_persons = 0

    # ------------------------------------------------------------------ #
    def update(self, detections: Sequence[Detection], ts: Optional[float] = None) -> Dict[str, Any]:
        """处理一帧检测结果，返回当前帧快照。"""
        ts = float(ts if ts is not None else time.time())
        dt = (ts - self.last_ts) if self.last_ts is not None else (1.0 / self.fps_assume)
        dt = min(max(dt, 1.0 / 120.0), 1.0)  # 截断，避免长时间卡顿把时长算爆
        self.last_ts = ts
        self.frame_index += 1

        counts: Dict[str, int] = {name: 0 for name in self.zh_names}
        seen_ids: set = set()
        for det in detections:
            name = det.cls_name
            counts[name] = counts.get(name, 0) + 1
            tid = det.track_id
            if tid is None:
                continue
            seen_ids.add(tid)
            state = self.tracks.get(tid)
            if state is None:
                state = TrackState(track_id=tid, cls_name=name, start_ts=ts, last_seen=ts, box=det.xyxy)
                self.tracks[tid] = state
            else:
                if state.cls_name != name:
                    state.switch_to(name, ts)
                else:
                    state.accumulate(ts)
                state.box = det.xyxy
            state.frames += 1

        self._expire_tracks(ts, seen_ids)

        persons = sum(counts.values())
        self.peak_persons = max(self.peak_persons, persons)
        attention_now = sum(counts.get(c, 0) for c in self.attention_classes)
        head_up_now = sum(counts.get(c, 0) for c in self.head_up_classes)
        record = {
            "frame": self.frame_index,
            "ts": ts,
            "persons": persons,
            "attention_rate": (attention_now / persons * 100.0) if persons else 0.0,
            "head_up_rate": (head_up_now / persons * 100.0) if persons else 0.0,
            "dt": dt,
            **{f"count_{k}": v for k, v in counts.items()},
        }
        self.frame_records.append(record)
        self.window.append(record)
        cutoff = ts - self.window_seconds
        while self.window and self.window[0]["ts"] < cutoff:
            self.window.popleft()
        return self.snapshot()

    def _expire_tracks(self, ts: float, seen_ids: set) -> None:
        for tid in list(self.tracks.keys()):
            state = self.tracks[tid]
            if tid in seen_ids:
                continue
            if ts - state.last_seen > self.track_timeout:
                state.accumulate(state.last_seen)  # 结算最后一段
                self._finalize(state)
                del self.tracks[tid]

    def _finalize(self, state: TrackState) -> None:
        state.anomaly_seconds = float(sum(state.durations.get(c, 0.0) for c in self.anomaly_classes))
        self.finished.append(state)

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """当前状态（界面实时刷新用）。"""
        last = self.frame_records[-1] if self.frame_records else {}
        counts = {name: int(last.get(f"count_{name}", 0)) for name in self.zh_names}
        return {
            "frame": self.frame_index,
            "persons": int(last.get("persons", 0)),
            "counts": counts,
            "attention_rate": float(last.get("attention_rate", 0.0)),
            "head_up_rate": float(last.get("head_up_rate", 0.0)),
            "window_attention_rate": self.window_attention_rate(),
            "active_tracks": len(self.tracks),
        }

    def window_attention_rate(self) -> float:
        """最近 window_seconds 内按人数加权的抬头率。"""
        if not self.window:
            return 0.0
        persons = sum(r["persons"] for r in self.window)
        if persons == 0:
            return 0.0
        attention = sum(r["attention_rate"] * r["persons"] for r in self.window)
        return attention / persons

    def window_head_up_rate(self) -> float:
        """最近窗口内按人数加权的「抬头率」（只含听讲/举手/站立）。"""
        if not self.window:
            return 0.0
        persons = sum(r["persons"] for r in self.window)
        if persons == 0:
            return 0.0
        head_up = sum(r.get("head_up_rate", 0.0) * r["persons"] for r in self.window)
        return head_up / persons

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """整体统计（报表 / 看板用）。"""
        states = list(self.tracks.values()) + self.finished
        durations: Dict[str, float] = {name: 0.0 for name in self.zh_names}
        for st in states:
            for name, sec in st.durations.items():
                durations[name] = durations.get(name, 0.0) + sec
        total_time = sum(durations.values())
        share = {k: (v / total_time * 100.0 if total_time > 0 else 0.0) for k, v in durations.items()}
        attention_seconds = sum(durations.get(c, 0.0) for c in self.attention_classes)
        head_up_seconds = sum(durations.get(c, 0.0) for c in self.head_up_classes)
        anomaly_seconds = sum(durations.get(c, 0.0) for c in self.anomaly_classes)
        frames = len(self.frame_records)
        avg_persons = (
            sum(r["persons"] for r in self.frame_records) / frames if frames else 0.0
        )
        return {
            "frames": frames,
            "duration_seconds": round(float(total_time), 2),
            "students": len(states),
            "peak_persons": int(self.peak_persons),
            "avg_persons": round(avg_persons, 2),
            "behavior_seconds": {k: round(v, 2) for k, v in durations.items()},
            "behavior_share": {k: round(v, 2) for k, v in share.items()},
            "attention_rate": round((attention_seconds / total_time * 100.0) if total_time else 0.0, 2),
            "head_up_rate": round((head_up_seconds / total_time * 100.0) if total_time else 0.0, 2),
            "window_attention_rate": round(self.window_attention_rate(), 2),
            "window_head_up_rate": round(self.window_head_up_rate(), 2),
            "anomaly_seconds": round(anomaly_seconds, 2),
            "current": self.snapshot(),
        }

    def per_student_table(self) -> List[Dict[str, Any]]:
        """逐学生行为画像（导出 Excel / 界面表格）。"""
        rows: List[Dict[str, Any]] = []
        states = list(self.tracks.values()) + self.finished
        states.sort(key=lambda s: s.total_seconds, reverse=True)
        for st in states:
            row: Dict[str, Any] = {
                "学生编号": f"#{st.track_id}",
                "出现帧数": st.frames,
                "总时长(秒)": round(st.total_seconds, 1),
                "主要行为": st.dominant(),
                "异常时长(秒)": round(sum(st.durations.get(c, 0.0) for c in self.anomaly_classes), 1),
            }
            for name in self.zh_names:
                row[f"{name}(秒)"] = round(st.durations.get(name, 0.0), 1)
            rows.append(row)
        return rows

    def timeline_records(self) -> List[Dict[str, Any]]:
        """逐帧统计记录（画趋势图 / 导出 CSV）。"""
        return [
            {
                "帧号": r["frame"],
                "时间戳": round(r["ts"], 3),
                "人数": r["persons"],
                "抬头率(%)": round(r["attention_rate"], 2),
                **{name: int(r.get(f"count_{name}", 0)) for name in self.zh_names},
            }
            for r in self.frame_records
        ]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.timeline_records())
