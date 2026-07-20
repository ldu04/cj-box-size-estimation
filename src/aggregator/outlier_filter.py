import numpy as np
from typing import List
from src.schema import Detection


class OutlierFilter:
    """Removes detections whose bbox area deviates beyond sigma std from the mean."""

    def __init__(self, sigma: float = 2.5) -> None:
        self.sigma = sigma

    def filter(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) < 4:
            return detections
        areas = np.array([d.bbox.area for d in detections])
        mean, std = areas.mean(), areas.std()
        if std == 0:
            return detections
        return [d for d, a in zip(detections, areas) if abs(a - mean) <= self.sigma * std]
