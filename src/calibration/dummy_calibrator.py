from __future__ import annotations
from typing import Optional
import numpy as np

from src.schema import CameraInfo
from src.calibration.pixel_converter import PixelToCmConverter
from .base import BaseCalibrator


class DummyCalibrator(BaseCalibrator):
    """
    Always-ready calibrator that returns a fixed px/cm scale.
    Default calibrator for Phase 1 and any stage where a real
    calibration strategy has not yet been validated.
    """

    def __init__(self, px_per_cm: float = 10.0) -> None:
        self.px_per_cm = px_per_cm

    def feed(self, frame: np.ndarray, camera_info: Optional[CameraInfo] = None) -> None:
        pass  # no-op

    @property
    def is_ready(self) -> bool:
        return True

    def build_converter(self) -> PixelToCmConverter:
        return PixelToCmConverter(px_per_cm=self.px_per_cm)

    def reset(self) -> None:
        pass  # no state to clear
