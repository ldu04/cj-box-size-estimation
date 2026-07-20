from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Detection:
    bbox: BBox
    score: float
    class_id: int = 0
    frame_id: int = 0
    # True unless the detector was configured with a low_conf_thresh tier
    # (below its normal conf_thresh) for ByteTrack-style two-stage
    # association — low-confidence detections may only extend an existing
    # track, never spawn a new one.
    is_high_conf: bool = True


@dataclass
class Track:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    is_confirmed: bool = False

    @property
    def last_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None


@dataclass
class CameraInfo:
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float


# size_cm uses competition naming: w (width), d (depth), h (height)
@dataclass
class SizeCm:
    w: float = 0.0
    d: float = 0.0
    h: float = 0.0

    @property
    def volume_cm3(self) -> float:
        return self.w * self.d * self.h


@dataclass
class BoxResult:
    box_id: Optional[int]
    size_cm: SizeCm


@dataclass
class VideoResult:
    video_id: str
    objects: List[BoxResult] = field(default_factory=list)


@dataclass
class CompetitionResult:
    videos: List[VideoResult] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        payload = {
            "videos": [
                {
                    "video_id": v.video_id,
                    "objects": [
                        {
                            # box_id is internal only — official result.json
                            # format contains size_cm exclusively
                            "size_cm": {
                                "w": round(b.size_cm.w, 3),
                                "d": round(b.size_cm.d, 3),
                                "h": round(b.size_cm.h, 3),
                            },
                        }
                        for b in v.objects
                    ],
                }
                for v in self.videos
            ]
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
