from __future__ import annotations
from typing import List, Dict, Optional
from src.schema import Track, BoxResult
from src.geometry.base import BaseGeometryEstimator


class TrackAggregator:
    """
    Accumulates per-frame tracks, finalizes into ranked BoxResults.
    Discards ghost tracks shorter than min_frames.

    Origin filter (origin_filter dict, opt-in): classifies each track by WHERE
    and WHEN it first appeared, based on the physical constraint that a real
    box can only (a) already be on the belt at video start, or (b) enter from
    the conveyor entry edge at the bottom of the frame. It cannot materialize
    mid-scene mid-video.

    Empirical basis (scripts/diagnose_fragmentation.py on count-mismatch
    videos): a global speed threshold CANNOT separate false positives from
    real boxes — a confirmed FP in train_005 moved at 0.76 px/frame while a
    confirmed real box in train_000 moved at 0.75 px/frame. What does separate
    them is origin: every confirmed ghost appeared mid-scene (y 180-370) mid-
    video (f56-192), while every real box either existed at frame 0 or entered
    at the bottom entry line (y≈700+). The one exception class is frame-0
    static background FPs (train_004 id=5: 59 px total drift in 248 frames,
    speed 0.24), caught by a much lower static-speed cut.
    """

    def __init__(
        self,
        geometry_estimator: BaseGeometryEstimator,
        min_frames: int = 5,
        min_speed_px_per_frame: float = 0.0,
        origin_filter: Optional[dict] = None,
    ) -> None:
        self.geometry_estimator = geometry_estimator
        self.min_frames = min_frames
        # Legacy global speed filter (superseded by origin_filter; kept for
        # backward compat). 0.0 disables.
        self.min_speed_px_per_frame = min_speed_px_per_frame
        of = origin_filter or {}
        self.origin_enabled = bool(of.get("enabled", False))
        self.entry_y_frac = float(of.get("entry_y_frac", 0.8))
        self.start_frame_grace = int(of.get("start_frame_grace", 10))
        self.min_frames_entry = int(of.get("min_frames_entry", min_frames))
        self.static_speed = float(of.get("static_speed", 0.5))
        self.ghost_speed = float(of.get("ghost_speed", 0.9))
        self._frame_h: Optional[int] = None
        self._tracks: Dict[int, Track] = {}
        self._seen_frames: Dict[int, set] = {}

    def set_frame_shape(self, height: int, width: int) -> None:
        """Called once by VideoProcessor at the first frame; needed to place
        the entry line (entry_y_frac is relative to frame height)."""
        self._frame_h = height

    def ingest(self, tracks: List[Track]) -> None:
        for t in tracks:
            if t.track_id not in self._tracks:
                self._tracks[t.track_id] = Track(
                    track_id=t.track_id, is_confirmed=t.is_confirmed
                )
                self._seen_frames[t.track_id] = set()
            # Dedupe by frame_id: cumulative trackers (DummyTracker) re-send
            # the full detection history every frame; without this the list
            # grows quadratically and min_frames counts duplicates.
            seen = self._seen_frames[t.track_id]
            for det in t.detections:
                if det.frame_id not in seen:
                    seen.add(det.frame_id)
                    self._tracks[t.track_id].detections.append(det)
            self._tracks[t.track_id].is_confirmed |= t.is_confirmed

    @staticmethod
    def _track_speed(dets) -> float:
        f0, f1 = dets[0].frame_id, dets[-1].frame_id
        if f1 <= f0:
            return 0.0
        b0, b1 = dets[0].bbox, dets[-1].bbox
        dist = ((b1.cx - b0.cx) ** 2 + (b1.cy - b0.cy) ** 2) ** 0.5
        return dist / (f1 - f0)

    def _keep_track(self, track: Track) -> bool:
        dets = sorted(track.detections, key=lambda d: d.frame_id)

        if self.origin_enabled and self._frame_h:
            speed = self._track_speed(dets)
            start_frame = dets[0].frame_id
            start_cy = dets[0].bbox.cy
            if start_cy >= self.entry_y_frac * self._frame_h:
                # Entered at the conveyor entry line → real box. Perspective
                # makes far boxes slow, so no speed cut; relaxed min_frames
                # recovers real boxes that got occluded shortly after entry.
                return len(dets) >= self.min_frames_entry
            if start_frame <= self.start_frame_grace:
                # Already on the belt at video start (anywhere in frame).
                # Only a near-zero drift marks a static background FP.
                return len(dets) >= self.min_frames and speed >= self.static_speed
            # Materialized mid-scene mid-video → ghost unless clearly moving
            # like a real box (fast fragments are kept to avoid dropping a
            # box whose earlier fragment died before min_frames).
            return len(dets) >= self.min_frames and speed >= self.ghost_speed

        # Legacy path: length cut + optional global speed cut.
        if len(dets) < self.min_frames:
            return False
        if self.min_speed_px_per_frame > 0:
            return self._track_speed(dets) >= self.min_speed_px_per_frame
        return True

    def finalize(self) -> List[BoxResult]:
        results = []
        for box_id, track in enumerate(self._tracks.values()):
            if not self._keep_track(track):
                continue
            size = self.geometry_estimator.estimate(track)
            if size.volume_cm3 <= 0:
                continue
            results.append(BoxResult(box_id=box_id, size_cm=size))
        return results

    def reset(self) -> None:
        self._tracks.clear()
        self._seen_frames.clear()
