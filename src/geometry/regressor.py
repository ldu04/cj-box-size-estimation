from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.schema import Track, SizeCm, CameraInfo
from src.exceptions import ModelLoadError
from .base import BaseGeometryEstimator

logger = logging.getLogger(__name__)


class RegressorGeometryEstimator(BaseGeometryEstimator):
    """
    CNN Dimension Regressor (C_regressor_train) as a geometry estimator.

    Streaming design: observe() caches the top-N largest crops per track while
    frames flow through VideoProcessor (no second video pass), estimate() runs
    the CNN on the cached crops and takes the per-dimension median.

    Preprocessing MUST stay identical to C_regressor_train/dataset.py:
    BGR as-is (no RGB swap), ±pad_px crop padding, resize 128, /255.0 only,
    meta = [w/fw, h/fh, cy/fh, focal_mm/40.0] from the UNPADDED bbox.
    """

    IMG_SIZE = 128
    FOCAL_FRAMES = 5           # frames sampled per video for focal estimation
    FOCAL_INPUT = (320, 180)   # focalnet input size (w, h)
    # Plausibility clamp: train_label.json GT across all 100 train videos
    # (2754 w/d/h values) has max=70.9cm, p99=55.0cm. Any predicted dimension
    # beyond 100cm (>40% margin over the observed max) is not a real parcel —
    # almost certainly a detector false positive on background structure
    # (e.g. container wall) that the regressor then dutifully sizes up.
    # Root-caused 2026-07-15 on train_013/detector_b: a container-wall crop
    # was consistently (5/5 crops) predicted at ~140x165x44cm. Dropping such
    # tracks (volume=0 → filtered by TrackAggregator, same path as "no crops")
    # trades a wild size error for a missed-detection padding penalty, which
    # is almost always cheaper under the official scoring.
    MAX_PLAUSIBLE_DIM_CM = 100.0

    def __init__(
        self,
        model_path: str,
        focal_length_mm: float = 5.14,
        max_crops: int = 5,
        pad_px: int = 5,
        focal_model_path: str | None = None,
        crop_quality_filter: bool = False,
        multiview_model_path: str | None = None,
        use_geo_feature: bool = False,
        focal_stride: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelLoadError(f"Regressor checkpoint not found: {model_path}")
        self.focal_length_mm = focal_length_mm  # fallback when video has no CameraInfo
        self.max_crops = max_crops
        self.pad_px = pad_px
        self.focal_model_path = Path(focal_model_path) if focal_model_path else None
        if self.focal_model_path and not self.focal_model_path.is_file():
            raise ModelLoadError(f"Focal model not found: {focal_model_path}")
        # Prefer sharp, fully-in-frame crops over merely-large ones. Off by
        # default: bbox area alone was the original (validated) heuristic;
        # this is an untested refinement, kept opt-in until compared.
        self.crop_quality_filter = crop_quality_filter
        # Optional 2nd model: multi-view regressor (model_multiview.py) that
        # sees up to max_crops crops jointly in one forward pass and predicts
        # a single box size, instead of median-ing independent per-crop
        # predictions. Val box-level MAE: CNN alone 4.10cm, multiview alone
        # 4.27cm, averaged 3.94cm — the two architectures make partly
        # decorrelated errors, so simple cm-space averaging helps despite
        # multiview being individually weaker. Off by default (untested in
        # the official harness until validated).
        self.multiview_model_path = (
            Path(multiview_model_path) if multiview_model_path else None
        )
        if self.multiview_model_path and not self.multiview_model_path.is_file():
            raise ModelLoadError(f"Multiview model not found: {multiview_model_path}")
        # Track self-calibration feature (geo_feature.py): slope of
        # log(pixel width) vs normalized y-position, fit from the track's own
        # observed frames. Encodes this track's perspective/scale without any
        # rail detection or GT — needs a model trained with 6D metadata
        # (model_geofeat.py); requires the matching model_path to accept it.
        self.use_geo_feature = use_geo_feature
        # Focal sampling policy. None (default): legacy behavior — focalnet on
        # the FIRST 5 frames only. Set to N: focalnet on every N-th frame
        # across the WHOLE video (median at estimate() time). Measured on all
        # 100 train videos (scripts/analyze_focalnet_accuracy.py, 2026-07-12):
        # first-5 rel err 3.18% vs spread-15 2.14% — early frames are a biased
        # sample (belt often still empty/entering), spreading fixes it.
        self.focal_stride = focal_stride
        self._obs_frame_idx = 0
        self._model = None
        self._mv_model = None
        self._focal_session = None
        # track_id -> [(bbox_area, img_chw, meta3, blur_score, edge_cut)] —
        # focal column is appended at estimate() time, once the per-video
        # focal estimate has settled
        self._crops: Dict[int, List[Tuple[float, np.ndarray, np.ndarray, float, bool]]] = {}
        # track_id -> [(cy/fh, bbox_px_width), ...] — EVERY observed frame,
        # independent of the area-ranked crop cache above (the slope needs
        # the full trajectory, not just the top-N largest crops).
        self._trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self._focal_estimates: List[float] = []   # focalnet predictions (this video)
        self._true_focal: float | None = None     # from CameraInfo when provided

    def _load_model(self):
        """Load by extension: .onnx → onnxruntime (submission path, per the
        competition rule that all AI models must ship and load as ONNX);
        .pt → torch state_dict (local dev/tests only)."""
        if self._model is None:
            try:
                if self.model_path.suffix.lower() == ".onnx":
                    import onnxruntime as ort

                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    session = ort.InferenceSession(
                        str(self.model_path),
                        sess_options=opts,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self._model = ("onnx", session)
                else:
                    import torch

                    if self.use_geo_feature:
                        from C_regressor_train.model_geofeat import RegressorGeoFeat

                        model = RegressorGeoFeat()
                    else:
                        from C_regressor_train.model import Regressor

                        model = Regressor()
                    model.load_state_dict(
                        torch.load(self.model_path, map_location="cpu")
                    )
                    model.eval()
                    self._model = ("torch", model)
            except Exception as e:
                raise ModelLoadError(
                    f"Cannot load regressor at {self.model_path}: {e}"
                ) from e
            logger.info("RegressorGeometryEstimator: loaded %s", self.model_path)
        return self._model

    def _load_mv_model(self):
        """multiview_model_path 지정 시에만 로드 (지연 로딩, 옵트인)."""
        if self._mv_model is None and self.multiview_model_path is not None:
            try:
                if self.multiview_model_path.suffix.lower() == ".onnx":
                    import onnxruntime as ort

                    session = ort.InferenceSession(
                        str(self.multiview_model_path),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self._mv_model = ("onnx", session)
                else:
                    import torch
                    from C_regressor_train.model_multiview import MultiViewRegressor

                    model = MultiViewRegressor(max_views=self.max_crops)
                    model.load_state_dict(
                        torch.load(self.multiview_model_path, map_location="cpu")
                    )
                    model.eval()
                    self._mv_model = ("torch", model)
            except Exception as e:
                raise ModelLoadError(
                    f"Cannot load multiview regressor at {self.multiview_model_path}: {e}"
                ) from e
            logger.info("RegressorGeometryEstimator: loaded multiview %s",
                        self.multiview_model_path)
        return self._mv_model

    def reset(self) -> None:
        self._crops.clear()
        self._trajectories.clear()
        self._focal_estimates.clear()
        self._true_focal = None
        self._obs_frame_idx = 0

    def _load_focal_session(self):
        if self._focal_session is None:
            import onnxruntime as ort

            self._focal_session = ort.InferenceSession(
                str(self.focal_model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("RegressorGeometryEstimator: focal model %s", self.focal_model_path)
        return self._focal_session

    def _estimate_focal_from(self, frame: np.ndarray) -> None:
        """전체 프레임에서 focal 추정치를 수집 (영상당 FOCAL_FRAMES회)."""
        small = cv2.resize(frame, self.FOCAL_INPUT).astype(np.float32) / 255.0
        sess = self._load_focal_session()
        inp = sess.get_inputs()[0].name
        log_f = sess.run(None, {inp: small.transpose(2, 0, 1)[None]})[0]
        self._focal_estimates.append(float(np.exp(log_f[0, 0])))

    def _current_focal(self) -> float:
        if self._true_focal is not None:
            return self._true_focal          # CameraInfo가 주어진 경우 (train 진단)
        if self._focal_estimates:
            return float(np.median(self._focal_estimates))
        return self.focal_length_mm          # focalnet 없음/미수집 시 fallback

    def observe(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        camera_info: Optional[CameraInfo] = None,
    ) -> None:
        fh, fw = frame.shape[:2]
        if camera_info is not None:
            self._true_focal = camera_info.focal_length_mm
        elif self.focal_model_path is not None:
            if self.focal_stride:
                # spread sampling: every focal_stride-th frame, whole video
                if self._obs_frame_idx % self.focal_stride == 0:
                    self._estimate_focal_from(frame)
            elif len(self._focal_estimates) < self.FOCAL_FRAMES:
                self._estimate_focal_from(frame)
        self._obs_frame_idx += 1

        # When quality-filtering, keep a larger area-ranked candidate pool so
        # estimate() has room to swap out blurry/edge-cut crops for the next
        # best-area alternative; otherwise behave exactly as before.
        pool_size = max(self.max_crops * 3, 15) if self.crop_quality_filter else self.max_crops

        for t in tracks:
            det = t.last_detection
            if det is None:
                continue
            b = det.bbox
            if self.use_geo_feature:
                self._trajectories.setdefault(t.track_id, []).append(
                    (b.cy / fh, b.width)
                )
            x1 = max(0, int(b.x1) - self.pad_px)
            y1 = max(0, int(b.y1) - self.pad_px)
            x2 = min(fw, int(b.x2) + self.pad_px)
            y2 = min(fh, int(b.y2) + self.pad_px)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            img = cv2.resize(crop, (self.IMG_SIZE, self.IMG_SIZE)).astype(np.float32) / 255.0
            meta3 = np.array(
                [b.width / fw, b.height / fh, b.cy / fh],
                dtype=np.float32,
            )
            if self.crop_quality_filter:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                edge_cut = bool(b.x1 <= 1 or b.y1 <= 1 or b.x2 >= fw - 1 or b.y2 >= fh - 1)
            else:
                blur_score, edge_cut = 0.0, False
            cache = self._crops.setdefault(t.track_id, [])
            cache.append((b.area, img.transpose(2, 0, 1), meta3, blur_score, edge_cut))
            # keep only the pool_size largest crops (bigger box = closer/sharper)
            cache.sort(key=lambda c: c[0], reverse=True)
            del cache[pool_size:]

    def _select_crops(self, pool: List[tuple]) -> List[tuple]:
        if not self.crop_quality_filter or len(pool) <= self.max_crops:
            return pool[: self.max_crops]
        # Prefer fully-in-frame crops; only fall back to edge-cut ones if
        # there aren't enough clean candidates (never starve a track to 0).
        clean = [c for c in pool if not c[4]]
        candidates = clean if len(clean) >= self.max_crops else pool
        # Drop the blurriest third of the candidate pool, but never below
        # max_crops candidates to choose from.
        by_blur = sorted(candidates, key=lambda c: c[3], reverse=True)
        keep_n = max(self.max_crops, int(len(by_blur) * 0.67))
        sharp = by_blur[:keep_n]
        return sorted(sharp, key=lambda c: c[0], reverse=True)[: self.max_crops]

    def estimate(self, track: Track) -> SizeCm:
        pool = self._crops.get(track.track_id)
        if not pool:
            return SizeCm()  # volume 0 → dropped by TrackAggregator
        crops = self._select_crops(pool)

        kind, model = self._load_model()
        imgs = np.stack([c[1] for c in crops]).astype(np.float32)
        focal_col = np.full((len(crops), 1), self._current_focal() / 40.0, dtype=np.float32)
        metas = np.hstack(
            [np.stack([c[2] for c in crops]).astype(np.float32), focal_col]
        )

        if self.use_geo_feature:
            from C_regressor_train.geo_feature import compute_slope_feature

            traj = self._trajectories.get(track.track_id, [])
            ys = np.array([p[0] for p in traj], dtype=np.float64)
            widths = np.array([p[1] for p in traj], dtype=np.float64)
            slope, flag = compute_slope_feature(ys, widths)
            geo_col = np.full((len(crops), 2), (slope, flag), dtype=np.float32)
            metas = np.hstack([metas, geo_col])

        if kind == "onnx":
            inputs = model.get_inputs()
            pred_log = model.run(
                None, {inputs[0].name: imgs, inputs[1].name: metas}
            )[0]
        else:
            import torch

            with torch.no_grad():
                pred_log = model(
                    torch.from_numpy(imgs), torch.from_numpy(metas)
                ).numpy()

        # train.py fits log(cm) targets → exp() back to cm here
        pred_cm = np.median(np.exp(pred_log), axis=0)

        if self.multiview_model_path is not None:
            mv_cm = self._estimate_multiview(crops)
            pred_cm = (pred_cm + mv_cm) / 2.0

        # NaN 가드 포함: NaN은 >비교가 항상 False라 클램프를 통과해버리고,
        # json.dump가 NaN을 그대로 출력하면 채점기 JSON 파싱이 깨질 수 있음.
        if not np.all(np.isfinite(pred_cm)) or float(np.max(pred_cm)) > self.MAX_PLAUSIBLE_DIM_CM:
            logger.warning(
                "RegressorGeometryEstimator: implausible size %.1fx%.1fx%.1fcm "
                "dropped (track_id=%s, likely detector false positive)",
                pred_cm[0], pred_cm[1], pred_cm[2], track.track_id,
            )
            return SizeCm()  # volume 0 → dropped by TrackAggregator

        return SizeCm(w=float(pred_cm[0]), d=float(pred_cm[1]), h=float(pred_cm[2]))

    def _estimate_multiview(self, crops: List[tuple]) -> np.ndarray:
        """crops(최대 max_crops개, 면적 내림차순)를 하나의 다중 뷰 입력으로 묶어
        단일 예측을 낸다. max_crops보다 적으면 있는 크롭을 반복해서 채운다
        (학습 시 복원추출로 채운 것과 동일한 분포)."""
        k = self.max_crops
        n = len(crops)
        chosen = crops if n >= k else crops + [crops[i % n] for i in range(k - n)]
        chosen = chosen[:k]

        focal = self._current_focal() / 40.0
        imgs = np.stack([c[1] for c in chosen]).astype(np.float32)[None]  # (1,k,3,H,W)
        metas3 = np.stack([c[2] for c in chosen]).astype(np.float32)
        focal_col = np.full((k, 1), focal, dtype=np.float32)
        metas = np.hstack([metas3, focal_col])[None]  # (1,k,4)

        kind, model = self._load_mv_model()
        if kind == "onnx":
            inputs = model.get_inputs()
            pred_log = model.run(None, {inputs[0].name: imgs, inputs[1].name: metas})[0]
        else:
            import torch

            with torch.no_grad():
                pred_log = model(torch.from_numpy(imgs), torch.from_numpy(metas)).numpy()
        return np.exp(pred_log[0])
