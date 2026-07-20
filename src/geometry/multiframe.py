from __future__ import annotations
import numpy as np

from src.schema import Track, SizeCm
from src.calibration.pixel_converter import PixelToCmConverter
from src.aggregator.outlier_filter import OutlierFilter
from .base import BaseGeometryEstimator


class MultiFrameGeometryEstimator(BaseGeometryEstimator):
    """
    Top-view assumption: bbox width → package w, bbox height → package d.
    Physical height h is derived from the camera tilt angle.

    Connection point (Phase 2 / Phase 3):
        Call set_converter() with a calibrated PixelToCmConverter
        (from RailCalibrator or IntrinsicCalibrator) before finalize().
        This is done automatically by VideoProcessor once calibration is ready.
    """

    def __init__(
        self,
        converter: PixelToCmConverter,
        camera_tilt_deg: float = 30.0,
    ) -> None:
        self.converter = converter
        self.outlier_filter = OutlierFilter()
        self._sin = np.sin(np.radians(camera_tilt_deg))

    def set_converter(self, converter: PixelToCmConverter) -> None:
        self.converter = converter

    def estimate(self, track: Track) -> SizeCm:
        dets = self.outlier_filter.filter(track.detections)
        if not dets:
            return SizeCm()
        widths_cm = [self.converter.px_to_cm(det.bbox.width) for det in dets]
        depths_cm = [self.converter.px_to_cm(det.bbox.height) for det in dets]
        w = float(np.median(widths_cm))
        d = float(np.median(depths_cm))
        h = d * self._sin / max(1.0 - self._sin, 1e-6)
        return SizeCm(w=w, d=d, h=max(h, 1.0))
