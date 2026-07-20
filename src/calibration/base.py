from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from src.schema import CameraInfo
from src.calibration.pixel_converter import PixelToCmConverter


class BaseCalibrator(ABC):
    """Plugin interface for all calibration strategies."""

    @abstractmethod
    def feed(self, frame: np.ndarray, camera_info: Optional[CameraInfo] = None) -> None:
        """Consume one frame (and optional per-video camera metadata)."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True once enough frames have been accumulated to build a converter."""
        ...

    @abstractmethod
    def build_converter(self) -> PixelToCmConverter:
        """Return a calibrated PixelToCmConverter. Call only when is_ready."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state for the next video."""
        ...
