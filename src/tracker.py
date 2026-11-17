"""轻量 IOU 跟踪器。

当使用 ONNX 权重或不需要 ultralytics 内置跟踪器时使用；课堂场景人数多、
遮挡重，这里采用「贪心 IOU 匹配 + 最大丢失帧数」的经典 SORT 简化策略，
足以支撑行为时长的近似统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.detector import Detection


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的交并比。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    cls_id: int
    lost: int = 0
    age: int = 1
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)


class SimpleTracker:
    """最小可用的多目标跟踪器（按类别分别匹配）。"""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 1) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> List[Detection]:
        """为当前帧检测框分配 track_id，返回新的 Detection 列表。"""
        unmatched = list(range(len(detections)))
        matched_track_ids = set()

        # 1) 对每条已有轨迹，找同类别中 IOU 最大的未匹配检测框
        for track in list(self.tracks.values()):
            best_idx, best_iou = -1, self.iou_threshold
            for idx in unmatched:
                det = detections[idx]
                if det.cls_id != track.cls_id:
                    continue
                score = iou(track.xyxy, det.xyxy)
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_idx >= 0:
                det = detections[best_idx]
                track.xyxy = tuple(float(v) for v in det.xyxy)
                track.cls_id = det.cls_id
                track.lost = 0
                track.age += 1
                track.history.append(track.xyxy)
                if len(track.history) > 50:
                    track.history.pop(0)
                det.track_id = track.track_id
                unmatched.remove(best_idx)
                matched_track_ids.add(track.track_id)

        # 2) 未匹配的检测框新建轨迹
        for idx in unmatched:
            det = detections[idx]
            track = _Track(track_id=self._next_id, xyxy=tuple(float(v) for v in det.xyxy), cls_id=det.cls_id)
            track.history.append(track.xyxy)
            self.tracks[self._next_id] = track
            det.track_id = self._next_id
            matched_track_ids.add(self._next_id)
            self._next_id += 1

        # 3) 未命中的轨迹累计丢失帧数，超时删除
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid].lost += 1
                if self.tracks[tid].lost > self.max_age:
                    del self.tracks[tid]

        return list(detections)

    @property
    def active_count(self) -> int:
        return len(self.tracks)
