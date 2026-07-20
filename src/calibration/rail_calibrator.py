from __future__ import annotations
from typing import Optional, List
import numpy as np

from src.schema import CameraInfo
from src.exceptions import CalibrationError
from src.calibration.pixel_converter import PixelToCmConverter
from src.calibration.rail_detector import RailDetector
from .base import BaseCalibrator


class RailCalibrator(BaseCalibrator):
    """
    Calibrates pixel→cm scale by detecting conveyor rail edges over N frames.
    Uses Canny + HoughLinesP to find the two vertical rail lines,
    then maps the inter-rail pixel distance to the known physical width.

    Connection point (Phase 2):
        Replace or tune RailDetector parameters once real video analysis
        confirms the rail detection quality.
    """

    def __init__(
        self,
        known_rail_width_cm: float = 60.0,
        n_frames: int = 10,
    ) -> None:
        self.known_rail_width_cm = known_rail_width_cm
        self.n_frames = n_frames
        self._rail_detector = RailDetector(known_rail_width_cm)
        self._widths_px: List[float] = []

    def feed(self, frame: np.ndarray, camera_info: Optional[CameraInfo] = None) -> None:
        if self.is_ready:
            return
        try:
            rails = self._rail_detector.detect(frame)
            self._widths_px.append(rails.rail_width_px)
        except CalibrationError:
            pass  # skip frames where rail detection fails

    @property
    def is_ready(self) -> bool:
        return len(self._widths_px) >= self.n_frames

    def build_converter(self) -> PixelToCmConverter:
        if not self._widths_px:
            raise CalibrationError("RailCalibrator: no calibration frames collected")
        median_px = float(np.median(self._widths_px))
        return PixelToCmConverter.from_rail(median_px, self.known_rail_width_cm)

    def reset(self) -> None:
        self._widths_px.clear()
