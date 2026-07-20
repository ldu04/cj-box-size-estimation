from __future__ import annotations
import logging
import numpy as np
from typing import List

from src.schema import BBox, Detection
from src.exceptions import ModelLoadError, InferenceError
from .base import BaseDetector

logger = logging.getLogger(__name__)


class OnnxDetector(BaseDetector):
    """
    ONNX-based detector using CUDAExecutionProvider by default.

    Connection point (Phase 2):
        Override _postprocess() to match the actual YOLO model output format.
        Standard YOLO11 output is [batch, num_boxes, 85] or [batch, 4+nc, num_boxes].
        Implement NMS using torchvision.ops.nms (available in Docker) or numpy.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
        input_size: tuple = (640, 640),
        class_ids: list | None = None,
        cpu_intra_op_threads: int | None = None,
        low_conf_thresh: float | None = None,
        max_bbox_area_frac: float | None = None,
    ) -> None:
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.class_ids = set(class_ids) if class_ids else None
        # ByteTrack-style two-stage association support: when set (< conf_thresh),
        # detections scoring between low_conf_thresh and conf_thresh are still
        # returned (tagged is_high_conf=False) so the tracker can use them to
        # extend an existing track through a momentary confidence dip, without
        # ever letting them spawn a brand-new track. None disables the tier
        # (all returned detections are high-conf, current behavior).
        self.low_conf_thresh = low_conf_thresh
        # 2026-07-16 팀 병합: areafilter_patch.zip에서 이식(원 아이디어는 팀원 제공,
        # 옛 detector_a+conf0.4 스냅샷 기준이라 그대로 적용 대신 현재 코드에 이식).
        # 컨테이너 벽/터널 실루엣처럼 프레임의 큰 비율을 차지하는 오탐을 detector
        # 단계에서 조기 차단 — regressor의 MAX_PLAUSIBLE_DIM_CM 클램프(예측 cm 크기
        # 기준)와 같은 문제(train_013류)를 겨냥하지만 더 이른 단계(픽셀 bbox 면적
        # 기준)에서 잡는다는 점이 다름. None이면 비활성(기존 동작).
        self.max_bbox_area_frac = max_bbox_area_frac
        # Only matters on CPU fallback: multiple concurrently-loaded ORT
        # sessions (as in EnsembleDetector) each default to a full-core
        # thread pool, causing oversubscription/contention far worse than
        # 2x slowdown. On CUDAExecutionProvider this is a no-op (GPU does
        # the compute).
        self.cpu_intra_op_threads = cpu_intra_op_threads
        self._session = None

    def load(self) -> None:
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.cpu_intra_op_threads:
                opts.intra_op_num_threads = self.cpu_intra_op_threads

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=providers
            )

            # Log which provider was actually selected
            active = self._session.get_providers()
            if "CUDAExecutionProvider" in active:
                logger.info("OnnxDetector: using CUDAExecutionProvider")
            else:
                logger.warning(
                    "OnnxDetector: CUDAExecutionProvider not available, "
                    "falling back to %s", active[0]
                )
        except Exception as e:
            raise ModelLoadError(
                f"Cannot load ONNX model at {self.model_path}: {e}"
            ) from e

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        import cv2

        h, w = frame.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        # ultralytics models are trained on RGB with gray(114) letterbox padding
        resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (nw, nh))
        padded = np.full((*self.input_size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, (h, w)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list:
        """Greedy NMS on xyxy boxes. Returns kept indices (score-descending)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
            iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
            order = rest[iou <= iou_thresh]
        return keep

    def _postprocess(
        self, outputs: list, scale: float, orig_shape: tuple
    ) -> List[Detection]:
        """
        Decode ultralytics YOLOv8/YOLO11 ONNX output: [batch, 4+nc, num_boxes]
        with rows (cx, cy, w, h, cls0..clsN) in input_size coordinates.
        (This is what scripts/export_onnx.py produces — no objectness column.)
        """
        out = np.asarray(outputs[0])
        if out.ndim == 3:
            out = out[0]                       # (4+nc, N)
        out = out.T                            # → (N, 4+nc); layout is fixed
                                               # by scripts/export_onnx.py

        boxes_cxcywh = out[:, :4]
        cls_scores = out[:, 4:]
        scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)

        effective_thresh = (
            self.low_conf_thresh if self.low_conf_thresh is not None else self.conf_thresh
        )
        mask = scores >= effective_thresh
        if self.class_ids is not None:
            mask &= np.isin(class_ids, list(self.class_ids))
        if not mask.any():
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # cxcywh → xyxy, undo letterbox (top-left padded resize by `scale`)
        cx, cy, w, h = boxes_cxcywh.T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes /= scale
        oh, ow = orig_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        if self.max_bbox_area_frac is not None:
            box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            size_mask = box_area <= self.max_bbox_area_frac * (oh * ow)
            if not size_mask.any():
                return []
            boxes = boxes[size_mask]
            scores = scores[size_mask]
            class_ids = class_ids[size_mask]

        # class-wise NMS
        detections: List[Detection] = []
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            for k in self._nms(boxes[idx], scores[idx], self.nms_thresh):
                gi = idx[k]
                detections.append(
                    Detection(
                        bbox=BBox(*[float(v) for v in boxes[gi]]),
                        score=float(scores[gi]),
                        class_id=int(cid),
                        is_high_conf=bool(scores[gi] >= self.conf_thresh),
                    )
                )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._session is None:
            raise InferenceError("Model not loaded. Call load() first.")
        blob, scale, orig = self._preprocess(frame)
        try:
            inp_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {inp_name: blob})
        except Exception as e:
            raise InferenceError(f"ONNX inference failed: {e}") from e
        return self._postprocess(outputs, scale, orig)
