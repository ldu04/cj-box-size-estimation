from __future__ import annotations
import cv2
import numpy as np
from typing import Iterator
from src.exceptions import VideoReadError


class VideoReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> VideoReader:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise VideoReadError(f"Cannot open video: {self.path}")
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            raise VideoReadError("VideoReader used outside context manager")
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0
