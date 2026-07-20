"""
ByteTrack wrapper via the boxmot library.

WARNING: boxmot is NOT in the competition Docker requirements.txt.
         This tracker will raise CJLogisticsError on import in the evaluation
         environment. Use SortTracker (sort_tracker.py) instead for Docker-
         compatible tracking, or add boxmot to a custom Docker image.

Connection point (Phase 2 / Phase 3):
    If boxmot becomes available in the environment:
        configs/default.yaml → tracker.type: bytetrack
"""
from __future__ import annotations
from typing import List
import numpy as np

from src.schema import BBox, Detection, Track
from src.exceptions import CJLogisticsError
from .base import BaseTracker


class ByteTrackWrapper(BaseTracker):
    """
    Thin wrapper around ByteTrack via boxmot.
    NOT available in the competition Docker environment (boxmot not installed).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.cfg = dict(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )
        self._impl = None

    def _lazy_init(self) -> None:
        try:
            from boxmot import ByteTrack
            self._impl = ByteTrack(**self.cfg)
        except ImportError as e:
            raise CJLogisticsError(
                "boxmot is not installed in the competition Docker environment. "
                "Use SortTracker (tracker.type: sort) instead, or install boxmot "
                "manually: pip install boxmot"
            ) from e

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        if self._impl is None:
            self._lazy_init()

        dets_np = (
            np.array(
                [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.score, d.class_id]
                 for d in detections],
                dtype=np.float32,
            )
            if detections
            else np.empty((0, 6), dtype=np.float32)
        )

        tracks_raw = self._impl.update(dets_np, frame=None)
        tracks: dict[int, Track] = {}
        for row in tracks_raw:
            x1, y1, x2, y2, tid = row[:5]
            tid = int(tid)
            if tid not in tracks:
                tracks[tid] = Track(track_id=tid, is_confirmed=True)
            det = Detection(
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                score=float(row[5]) if len(row) > 5 else 1.0,
                frame_id=frame_id,
            )
            tracks[tid].detections.append(det)
        return list(tracks.values())

    def reset(self) -> None:
        self._impl = None
