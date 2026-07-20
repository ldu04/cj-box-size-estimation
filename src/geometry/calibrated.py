from __future__ import annotations
import numpy as np

from src.schema import Track, SizeCm
from src.calibration.pixel_converter import PixelToCmConverter
from .base import BaseGeometryEstimator


class CalibratedGeometryEstimator(BaseGeometryEstimator):
    """
    Median-based estimator using a single pixel→cm converter.
    Simpler than MultiFrameGeometryEstimator (no tilt model).
    Suitable as an intermediate implementation before camera geometry
    is fully characterised.
    """

    def __init__(
        self, converter: PixelToCmConverter, depth_ratio: float = 0.6
    ) -> None:
        self.converter = converter
        self.depth_ratio = depth_ratio

    def set_converter(self, converter: PixelToCmConverter) -> None:
        self.converter = converter

    def estimate(self, track: Track) -> SizeCm:
        widths = [d.bbox.width for d in track.detections]
        heights = [d.bbox.height for d in track.detections]
        w = self.converter.px_to_cm(float(np.median(widths)))
        d = self.converter.px_to_cm(float(np.median(heights)))
        h = d * self.depth_ratio
        return SizeCm(w=w, d=d, h=h)
