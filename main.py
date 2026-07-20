#!/usr/bin/env python3
"""
Competition entrypoint.

Interface (official competition spec):
    python main.py --input {video_folder_path}

--input_dir is kept as a deprecated alias for local tooling.
result.json is written next to this file (spec: main.py 위치 디렉터리)
unless --output overrides it.
"""
import argparse
import json
import logging
from pathlib import Path

try:
    import yaml  # PyYAML may be absent in the eval image — fallback below
except ImportError:
    yaml = None

from src.detector.dummy import DummyDetector
from src.detector.onnx_detector import OnnxDetector
from src.detector.ensemble_detector import EnsembleDetector
from src.tracker.dummy import DummyTracker
from src.tracker.sort_tracker import SortTracker
from src.tracker.byte_tracker import ByteTrackWrapper
from src.calibration.dummy_calibrator import DummyCalibrator
from src.calibration.rail_calibrator import RailCalibrator
from src.calibration.intrinsic_calibrator import IntrinsicCalibrator
from src.geometry.dummy import DummyGeometryEstimator
from src.geometry.calibrated import CalibratedGeometryEstimator
from src.geometry.multiframe import MultiFrameGeometryEstimator
from src.geometry.regressor import RegressorGeometryEstimator
from src.calibration.pixel_converter import PixelToCmConverter
from src.pipeline import Pipeline
from src.validator.result_validator import ResultValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Mirrors configs/default.yaml — used if PyYAML is unavailable in the eval env.
# 2026-07-16 오후: 마지막 제출 앞두고 detector_a로 최종 원복 (default.yaml과 반드시 동기화 유지 —
# 이 파일이 default.yaml과 어긋나면 PyYAML 부재 시 다른 조합으로 조용히 벗어남, 2026-07-16 오전 실수 재발 금지)
FALLBACK_CONFIG = {
    "detector": {
        "type": "onnx", "model_path": "checkpoints/detector_a.onnx",
        "conf_thresh": 0.4, "nms_thresh": 0.45,
        "input_size": [1280, 1280], "class_ids": [2],
        "low_conf_thresh": 0.15,
    },
    "tracker": {"type": "sort", "max_age": 90, "min_hits": 1, "iou_threshold": 0.3},
    "calibration": {"type": "dummy", "px_per_cm": 10.0},
    "geometry": {
        "type": "regressor", "model_path": "checkpoints/regressor.onnx",
        "focal_model_path": "checkpoints/focalnet.onnx",
        "focal_length_mm": 8.767, "max_crops": 5,
    },
    "aggregator": {
        "min_frames": 45, "min_speed_px_per_frame": 0.0,
        "origin_filter": {
            "enabled": True, "entry_y_frac": 0.8, "start_frame_grace": 10,
            "min_frames_entry": 25, "static_speed": 0.5, "ghost_speed": 0.9,
        },
    },
}


def load_config(path: str) -> dict:
    if yaml is None:
        logger.warning("PyYAML unavailable — using built-in fallback config")
        return json.loads(json.dumps(FALLBACK_CONFIG))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path: str) -> str:
    """Relative model paths are resolved against main.py's directory so the
    pipeline works regardless of the evaluator's working directory."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = Path(__file__).resolve().parent / p
        if candidate.exists():
            return str(candidate)
    return str(p)


def build_detector(cfg: dict):
    det = cfg["detector"]
    det_type = det.get("type", "dummy")
    if det_type == "onnx":
        return OnnxDetector(
            model_path=_resolve(det["model_path"]),
            conf_thresh=det.get("conf_thresh", 0.5),
            nms_thresh=det.get("nms_thresh", 0.45),
            input_size=tuple(det.get("input_size", [640, 640])),
            class_ids=det.get("class_ids"),
            low_conf_thresh=det.get("low_conf_thresh"),
            max_bbox_area_frac=det.get("max_bbox_area_frac"),
        )
    if det_type == "ensemble":
        models = []
        for m in det["ensemble"]["models"]:
            m = dict(m)
            m["model_path"] = _resolve(m["model_path"])
            models.append(m)
        return EnsembleDetector(
            models=models,
            wbf_iou_thresh=det["ensemble"].get("wbf_iou_thresh", 0.55),
            min_fused_score=det["ensemble"].get("min_fused_score", 0.5),
            input_size=tuple(det.get("input_size", [1280, 1280])),
        )
    return DummyDetector()


def build_tracker(cfg: dict):
    trk = cfg["tracker"]
    trk_type = trk.get("type", "dummy")
    if trk_type == "sort":
        return SortTracker(
            max_age=trk.get("max_age", 30),
            min_hits=trk.get("min_hits", 1),
            iou_threshold=trk.get("iou_threshold", 0.3),
        )
    if trk_type == "bytetrack":
        return ByteTrackWrapper(
            track_thresh=trk.get("track_thresh", 0.5),
            track_buffer=trk.get("track_buffer", 30),
            match_thresh=trk.get("match_thresh", 0.8),
            frame_rate=trk.get("frame_rate", 30),
        )
    return DummyTracker()


def build_calibrator(cfg: dict):
    cal = cfg.get("calibration", {})
    cal_type = cal.get("type", "dummy")
    if cal_type == "rail":
        return RailCalibrator(
            known_rail_width_cm=cal.get("known_rail_width_cm", 60.0),
            n_frames=cal.get("calibration_frame_count", 10),
        )
    if cal_type == "intrinsic":
        # Not default: assumed_distance_cm must be calibrated from train data first.
        return IntrinsicCalibrator(
            assumed_distance_cm=cal.get("assumed_distance_cm", 100.0),
        )
    return DummyCalibrator(px_per_cm=cal.get("px_per_cm", 10.0))


def build_geometry(cfg: dict):
    geo = cfg["geometry"]
    gtype = geo.get("type", "dummy")
    # Placeholder converter — replaced at runtime by VideoProcessor.calibrator
    placeholder = PixelToCmConverter(px_per_cm=10.0)
    if gtype == "regressor":
        focal_model = geo.get("focal_model_path")
        mv_model = geo.get("multiview_model_path")
        return RegressorGeometryEstimator(
            model_path=_resolve(geo.get("model_path", "checkpoints/regressor.onnx")),
            focal_length_mm=geo.get("focal_length_mm", 5.14),
            max_crops=geo.get("max_crops", 5),
            focal_model_path=_resolve(focal_model) if focal_model else None,
            crop_quality_filter=geo.get("crop_quality_filter", False),
            multiview_model_path=_resolve(mv_model) if mv_model else None,
            use_geo_feature=geo.get("use_geo_feature", False),
            focal_stride=geo.get("focal_stride"),
        )
    if gtype == "multiframe":
        return MultiFrameGeometryEstimator(
            converter=placeholder,
            camera_tilt_deg=geo.get("camera_tilt_deg", 30.0),
        )
    if gtype == "calibrated":
        return CalibratedGeometryEstimator(
            converter=placeholder,
            depth_ratio=geo.get("depth_ratio", 0.6),
        )
    return DummyGeometryEstimator()


def main() -> None:
    parser = argparse.ArgumentParser(description="CJ Logistics inference pipeline")
    parser.add_argument(
        "--input", "--input_dir", dest="input_dir", required=True,
        help="Directory containing .mp4 test videos (official arg: --input)",
    )
    # CLI --output always wins; do not let config override this.
    # Default: next to main.py, per official spec.
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "result.json"),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    pipeline = Pipeline(
        detector=build_detector(cfg),
        tracker=build_tracker(cfg),
        calibrator=build_calibrator(cfg),
        geometry_estimator=build_geometry(cfg),
        output_path=args.output,          # CLI arg, not config
        min_track_frames=cfg.get("aggregator", {}).get("min_frames", 5),
        min_track_speed=cfg.get("aggregator", {}).get("min_speed_px_per_frame", 0.0),
        origin_filter=cfg.get("aggregator", {}).get("origin_filter"),
    )
    pipeline.run(args.input_dir)

    validator = ResultValidator()
    validator.validate(args.output)
    logger.info("[OK] result.json written and validated → %s", args.output)


if __name__ == "__main__":
    main()
