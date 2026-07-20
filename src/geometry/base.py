from __future__ import annotations
from abc import ABC, abstractmethod
from src.schema import Track, SizeCm


class BaseGeometryEstimator(ABC):
    @abstractmethod
    def estimate(self, track: Track) -> SizeCm: ...

    def set_converter(self, converter) -> None:
        """
        Update the pixel→cm converter used by this estimator.
        Called by VideoProcessor once the calibrator becomes ready.
        Default: no-op (DummyGeometryEstimator ignores it).
        """
        pass

    def observe(self, frame, tracks, camera_info=None) -> None:
        """
        Per-frame hook called by VideoProcessor after tracking.
        Estimators needing pixel data (RegressorGeometryEstimator) cache
        crops here. Default: no-op.
        """
        pass

    def reset(self) -> None:
        """Clear per-video state. Called by VideoProcessor at video start."""
        pass
