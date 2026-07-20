from typing import List
from src.schema import Detection, Track
from .base import BaseTracker


class DummyTracker(BaseTracker):
    """
    Positional dummy tracker: detection index → track id.
    Detection 0 → track 1, detection 1 → track 2, etc.
    Subsequent frames add detections to the same positional track.

    Supports multi-object scenarios (one track per detection slot).
    Does not perform any IoU or Kalman-based association.
    Intended for Phase 1 baseline only.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        for i, det in enumerate(detections):
            det.frame_id = frame_id
            tid = i + 1  # 1-indexed: slot 0 → track_id 1
            if tid not in self._tracks:
                self._tracks[tid] = Track(track_id=tid, is_confirmed=True)
            self._tracks[tid].detections.append(det)
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
