"""
SORT tracker (Simple Online and Realtime Tracking).

Dependencies — both available in the competition Docker environment:
    filterpy==1.4.5   (Kalman filter)
    scipy==1.14.1     (Hungarian matching via linear_sum_assignment)

State vector per track: [cx, cy, s, r, vx, vy, vs]
    cx, cy : bounding-box centre
    s      : box area (w * h)
    r      : aspect ratio (w / h)
    vx, vy, vs : constant-velocity components

Measurement: [cx, cy, s, r]
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from src.schema import BBox, Detection, Track
from .base import BaseTracker


def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between N boxes (N×4) and M boxes (M×4) → N×M matrix."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class _KFTrack:
    """Single-object constant-velocity Kalman filter tracker."""

    def __init__(self, bbox: np.ndarray, score: float, track_id: int) -> None:
        from filterpy.kalman import KalmanFilter

        self.id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.last_score = score

        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7, dtype=np.float32)
        kf.F[0, 4] = 1.0  # cx  += vx
        kf.F[1, 5] = 1.0  # cy  += vy
        kf.F[2, 6] = 1.0  # s   += vs

        kf.H = np.zeros((4, 7), dtype=np.float32)
        kf.H[:4, :4] = np.eye(4, dtype=np.float32)

        kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        kf.P = np.eye(7, dtype=np.float32) * 10.0
        kf.P[4:, 4:] *= 100.0  # high velocity uncertainty on init
        kf.Q = np.eye(7, dtype=np.float32)
        kf.Q[4:, 4:] *= 0.01
        kf.Q[6, 6] *= 0.01

        kf.x[:4] = self._to_z(bbox)
        self.kf = kf

    def predict(self) -> np.ndarray:
        # Guard against negative area blowing up
        if float(self.kf.x[6, 0]) + float(self.kf.x[2, 0]) <= 0:
            self.kf.x[6, 0] = np.float32(0.0)
        self.kf.predict()
        self.time_since_update += 1
        return self._to_bbox(self.kf.x[:4].flatten())

    def update(self, bbox: np.ndarray, score: float) -> None:
        self.hits += 1
        self.time_since_update = 0
        self.last_score = score
        self.kf.update(self._to_z(bbox))

    def get_bbox(self) -> np.ndarray:
        return self._to_bbox(self.kf.x[:4].flatten())

    @staticmethod
    def _to_z(bbox: np.ndarray) -> np.ndarray:
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        return np.array(
            [[(bbox[0] + bbox[2]) / 2.0],
             [(bbox[1] + bbox[3]) / 2.0],
             [w * h],
             [w / max(h, 1e-6)]],
            dtype=np.float32,
        )

    @staticmethod
    def _to_bbox(z: np.ndarray) -> np.ndarray:
        cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        w = np.sqrt(max(s * r, 0.0))
        h = np.sqrt(max(s / max(r, 1e-6), 0.0))
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class SortTracker(BaseTracker):
    """
    SORT tracker.  Connection point for Phase 2:
        - Set config tracker.type = "sort" to activate.
        - Tune max_age, min_hits, iou_threshold from train video analysis.
        - Can be replaced by ByteTrack (tracker.type = "bytetrack") once
          the detection confidence score distribution is understood.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: List[_KFTrack] = []
        self._frame_count = 0
        self._next_id = 1

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        self._frame_count += 1

        # --- Step 1: predict ---
        predicted: List[np.ndarray] = []
        alive: List[_KFTrack] = []
        for trk in self._trackers:
            bbox = trk.predict()
            if not np.any(np.isnan(bbox)):
                predicted.append(bbox)
                alive.append(trk)
        self._trackers = alive
        pred_np = np.array(predicted) if predicted else np.empty((0, 4))

        # ByteTrack-style two-stage association: split by confidence tier.
        # Low-confidence detections (is_high_conf=False, only present if the
        # detector was configured with a low_conf_thresh) may only extend an
        # existing track through a momentary confidence dip — they never
        # spawn a new track, so a stray low-score false positive can't
        # create a ghost.
        high = [d for d in detections if d.is_high_conf]
        low = [d for d in detections if not d.is_high_conf]

        def _to_np(dets: List[Detection]) -> np.ndarray:
            if not dets:
                return np.empty((0, 4))
            return np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in dets])

        high_np = _to_np(high)

        # --- Step 2a: match high-confidence detections against all tracks ---
        matched, unmatched_high = self._match(high_np, pred_np)
        matched_track_idx = {ti for _, ti in matched}
        for di, ti in matched:
            self._trackers[ti].update(high_np[di], high[di].score)

        # --- Step 2b: rescue still-unmatched tracks with low-confidence dets ---
        if low:
            remaining_idx = [i for i in range(len(self._trackers)) if i not in matched_track_idx]
            if remaining_idx:
                low_np = _to_np(low)
                remaining_pred = pred_np[remaining_idx]
                low_matched, _ = self._match(low_np, remaining_pred)
                for di, local_ti in low_matched:
                    ti = remaining_idx[local_ti]
                    self._trackers[ti].update(low_np[di], low[di].score)

        # --- Step 3: create new tracks for unmatched HIGH-conf detections only ---
        for di in unmatched_high:
            self._trackers.append(
                _KFTrack(high_np[di], high[di].score, self._next_id)
            )
            self._next_id += 1

        # --- Step 5: emit confirmed tracks, prune stale ones ---
        result: dict[int, Track] = {}
        survivors: List[_KFTrack] = []
        for trk in self._trackers:
            if trk.time_since_update <= self.max_age:
                survivors.append(trk)
            if trk.hits >= self.min_hits and trk.time_since_update == 0:
                bbox = trk.get_bbox()
                tid = trk.id
                result[tid] = Track(track_id=tid, is_confirmed=True)
                result[tid].detections.append(
                    Detection(
                        bbox=BBox(
                            float(bbox[0]), float(bbox[1]),
                            float(bbox[2]), float(bbox[3]),
                        ),
                        score=trk.last_score,
                        frame_id=frame_id,
                    )
                )
        self._trackers = survivors
        return list(result.values())

    def reset(self) -> None:
        self._trackers.clear()
        self._frame_count = 0
        self._next_id = 1

    def _match(
        self, dets: np.ndarray, preds: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        from scipy.optimize import linear_sum_assignment

        if len(preds) == 0 or len(dets) == 0:
            return [], list(range(len(dets)))

        iou = _iou_batch(dets, preds)
        row_idx, col_idx = linear_sum_assignment(1.0 - iou)

        matched: List[Tuple[int, int]] = []
        matched_det_set: set = set()
        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= self.iou_threshold:
                matched.append((int(r), int(c)))
                matched_det_set.add(r)

        unmatched_dets = [i for i in range(len(dets)) if i not in matched_det_set]
        return matched, unmatched_dets
