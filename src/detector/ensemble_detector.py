from __future__ import annotations
import logging
from typing import List

import numpy as np

from src.schema import BBox, Detection
from .base import BaseDetector
from .onnx_detector import OnnxDetector
from .wbf import weighted_box_fusion

logger = logging.getLogger(__name__)


class EnsembleDetector(BaseDetector):
    """
    Runs several independently trained YOLO ONNX detectors per frame and
    fuses their outputs with Weighted Box Fusion (see wbf.py).

    Each sub-model keeps its own model_path/class_ids/conf_thresh/input_size
    (they may have different class heads, e.g. an older 3-class model and a
    newer 2-class fine-tune) — composed from independent OnnxDetector
    instances rather than duplicating pre/post-processing logic.

    config shape (configs/default.yaml detector.ensemble):
        - model_path: checkpoints/detector_a.onnx
          class_ids: [2]
          weight: 1.0
        - model_path: checkpoints/detector_b.onnx
          class_ids: [1]
          weight: 1.0
    """

    def __init__(
        self,
        models: list[dict],
        wbf_iou_thresh: float = 0.55,
        min_fused_score: float = 0.5,
        input_size: tuple = (1280, 1280),
        cpu_intra_op_threads: int | None = 4,
    ) -> None:
        if not models:
            raise ValueError("EnsembleDetector requires at least one model config")
        self._sub_detectors: List[OnnxDetector] = []
        self._weights: List[float] = []
        for m in models:
            self._sub_detectors.append(
                OnnxDetector(
                    model_path=m["model_path"],
                    conf_thresh=m.get("conf_thresh", 0.4),
                    nms_thresh=m.get("nms_thresh", 0.45),
                    input_size=tuple(m.get("input_size", input_size)),
                    class_ids=m.get("class_ids"),
                    cpu_intra_op_threads=cpu_intra_op_threads,
                )
            )
            self._weights.append(float(m.get("weight", 1.0)))
        self.wbf_iou_thresh = wbf_iou_thresh
        # WBF gives a box seen by only one model a fused score scaled by
        # ~1/n_models (see wbf.py) — without this floor, the ensemble
        # degenerates into a UNION of both models' outputs (each model's own
        # false positives all survive), which made over-detection videos
        # worse, not better. Requiring min_fused_score means a single-model
        # box only survives if that model was quite confident about it.
        self.min_fused_score = min_fused_score

    def load(self) -> None:
        for d in self._sub_detectors:
            d.load()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        per_model_boxes, per_model_scores = [], []
        for d in self._sub_detectors:
            dets = d.detect(frame)
            if dets:
                per_model_boxes.append(np.array(
                    [[dt.bbox.x1, dt.bbox.y1, dt.bbox.x2, dt.bbox.y2] for dt in dets]
                ))
                per_model_scores.append(np.array([dt.score for dt in dets]))
            else:
                per_model_boxes.append(np.zeros((0, 4)))
                per_model_scores.append(np.zeros((0,)))

        fused_boxes, fused_scores = weighted_box_fusion(
            per_model_boxes, per_model_scores, self._weights,
            iou_thr=self.wbf_iou_thresh,
        )
        if len(fused_scores):
            keep = fused_scores >= self.min_fused_score
            fused_boxes, fused_scores = fused_boxes[keep], fused_scores[keep]
        return [
            Detection(bbox=BBox(*[float(v) for v in b]), score=float(s), class_id=0)
            for b, s in zip(fused_boxes, fused_scores)
        ]
