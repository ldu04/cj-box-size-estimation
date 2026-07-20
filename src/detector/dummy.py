import numpy as np
from typing import List
from src.schema import BBox, Detection
from .base import BaseDetector


class DummyDetector(BaseDetector):
    def load(self) -> None:
        pass  # ONNX loading stub — wired in Phase 2

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        bw, bh = w // 4, h // 4
        return [
            Detection(
                bbox=BBox(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                score=0.99,
                class_id=0,
            )
        ]
