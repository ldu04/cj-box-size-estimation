from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PixelToCmConverter:
    px_per_cm: float

    @classmethod
    def from_rail(
        cls, rail_width_px: float, rail_width_cm: float
    ) -> PixelToCmConverter:
        return cls(px_per_cm=rail_width_px / rail_width_cm)

    def px_to_cm(self, pixels: float) -> float:
        return pixels / self.px_per_cm

    def cm_to_px(self, cm: float) -> float:
        return cm * self.px_per_cm
