from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from src.exceptions import CalibrationError


@dataclass
class RailLines:
    left_line: tuple
    right_line: tuple
    rail_width_px: float


class RailDetector:
    def __init__(self, known_rail_width_cm: float = 60.0) -> None:
        self.known_rail_width_cm = known_rail_width_cm

    def detect(self, frame: np.ndarray) -> RailLines:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 80, minLineLength=100, maxLineGap=20
        )
        if lines is None:
            raise CalibrationError("No rail lines detected in frame")
        vertical = [l[0] for l in lines if self._is_vertical(l[0])]
        if len(vertical) < 2:
            raise CalibrationError(
                f"Need ≥2 vertical lines, got {len(vertical)}"
            )
        vertical.sort(key=lambda l: (l[0] + l[2]) / 2)
        left, right = vertical[0], vertical[-1]
        rail_width_px = abs((right[0] + right[2]) / 2 - (left[0] + left[2]) / 2)
        return RailLines(tuple(left), tuple(right), rail_width_px)

    @staticmethod
    def _is_vertical(line: np.ndarray, angle_tol: float = 30.0) -> bool:
        x1, y1, x2, y2 = line
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx + 1e-6))
        return angle > (90 - angle_tol)
