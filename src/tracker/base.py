from abc import ABC, abstractmethod
from typing import List
from src.schema import Detection, Track


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection], frame_id: int) -> List[Track]: ...

    @abstractmethod
    def reset(self) -> None: ...
