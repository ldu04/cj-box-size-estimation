from src.schema import Track, SizeCm
from .base import BaseGeometryEstimator


class DummyGeometryEstimator(BaseGeometryEstimator):
    def estimate(self, track: Track) -> SizeCm:
        return SizeCm(w=30.0, d=20.0, h=15.0)
