"""
IntrinsicCalibrator — camera-intrinsic based pixel→cm scale estimator.

STATUS: Implemented but NOT the default. Activate only after estimating
assumed_distance_cm from train data analysis (train_label.json ground truth).

Problem:
    Converting focal length + sensor size to px/cm requires the physical
    distance from camera to the object plane (working distance).
    The competition provides focal_length_mm and sensor dimensions but NOT
    the camera mounting height. This must be estimated empirically:

        Option A: regress distance from train ground-truth box sizes vs.
                  detected pixel sizes.
        Option B: use a reference object of known size in the scene.
        Option C: assume a fixed distance and measure the error vs. train labels.

Formula (once distance_cm is known):
    focal_px  = focal_length_mm * image_width_px / sensor_width_mm
    px_per_cm = focal_px / (distance_cm * 10)   # 10 mm per cm

Activation:
    Set configs/default.yaml → calibration.type: intrinsic
    and set calibration.assumed_distance_cm to the calibrated value.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from src.schema import CameraInfo
from src.exceptions import CalibrationError
from src.calibration.pixel_converter import PixelToCmConverter
from .base import BaseCalibrator


class IntrinsicCalibrator(BaseCalibrator):
    """
    Derives px/cm from camera intrinsics + an assumed working distance.

    Args:
        assumed_distance_cm: Camera-to-conveyor plane distance in cm.
            Must be calibrated from train data before use.
            Default 100 cm is a placeholder — do NOT use without validation.
    """

    def __init__(self, assumed_distance_cm: float = 100.0) -> None:
        self.assumed_distance_cm = assumed_distance_cm
        self._camera_info: Optional[CameraInfo] = None
        self._image_width_px: Optional[int] = None

    def feed(self, frame: np.ndarray, camera_info: Optional[CameraInfo] = None) -> None:
        if self._image_width_px is None:
            self._image_width_px = frame.shape[1]
        if camera_info is not None and self._camera_info is None:
            self._camera_info = camera_info

    @property
    def is_ready(self) -> bool:
        return self._camera_info is not None and self._image_width_px is not None

    def build_converter(self) -> PixelToCmConverter:
        if not self.is_ready:
            raise CalibrationError(
                "IntrinsicCalibrator: camera_info not yet provided via feed()"
            )
        ci = self._camera_info
        focal_px = ci.focal_length_mm * self._image_width_px / ci.sensor_width_mm
        # distance in mm = assumed_distance_cm * 10
        px_per_cm = focal_px / (self.assumed_distance_cm * 10.0)
        return PixelToCmConverter(px_per_cm=px_per_cm)

    def reset(self) -> None:
        self._camera_info = None
        self._image_width_px = None
