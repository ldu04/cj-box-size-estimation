from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.schema import Detection


class BaseDetector(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]: ...

    def warmup(self, frame: np.ndarray) -> None:
        self.detect(frame)
