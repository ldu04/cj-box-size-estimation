from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from src.detector.base import BaseDetector
from src.tracker.base import BaseTracker
from src.calibration.base import BaseCalibrator
from src.geometry.base import BaseGeometryEstimator
from src.aggregator.track_aggregator import TrackAggregator
from src.io.video_reader import VideoReader
from src.schema import CameraInfo, BoxResult, VideoResult, CompetitionResult

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes a single video file → VideoResult.

    Per-frame loop:
        1. Feed frame to calibrator (streaming calibration)
        2. Once calibrator.is_ready: update geometry estimator's converter
        3. Detect + track → ingest into TrackAggregator
    After all frames:
        4. TrackAggregator.finalize() → BoxResult list
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        calibrator: BaseCalibrator,
        geometry_estimator: BaseGeometryEstimator,
        max_frames: Optional[int] = None,
        min_track_frames: int = 5,
        min_track_speed: float = 0.0,
        origin_filter: Optional[dict] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.calibrator = calibrator
        self.geometry = geometry_estimator
        self.max_frames = max_frames
        self.min_track_frames = min_track_frames
        self.min_track_speed = min_track_speed
        self.origin_filter = origin_filter

    def process(
        self,
        video_path: str,
        camera_info: Optional[CameraInfo] = None,
    ) -> VideoResult:
        video_id = Path(video_path).stem
        self.tracker.reset()
        self.calibrator.reset()
        self.geometry.reset()
        calibration_done = False

        aggregator = TrackAggregator(
            geometry_estimator=self.geometry,
            min_frames=self.min_track_frames,
            min_speed_px_per_frame=self.min_track_speed,
            origin_filter=self.origin_filter,
        )

        with VideoReader(video_path) as reader:
            for frame_id, frame in enumerate(reader):
                if self.max_frames is not None and frame_id >= self.max_frames:
                    break
                if frame_id == 0:
                    aggregator.set_frame_shape(frame.shape[0], frame.shape[1])

                # --- Streaming calibration ---
                if not calibration_done:
                    self.calibrator.feed(frame, camera_info)
                    if self.calibrator.is_ready:
                        converter = self.calibrator.build_converter()
                        self.geometry.set_converter(converter)
                        calibration_done = True
                        logger.debug("%s: calibration done at frame %d", video_id, frame_id)

                # --- Detection + tracking ---
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame_id)
                self.geometry.observe(frame, tracks, camera_info)
                aggregator.ingest(tracks)

        objects = aggregator.finalize()
        logger.info("%s: %d objects found", video_id, len(objects))
        return VideoResult(video_id=video_id, objects=objects)


class Pipeline:
    """Processes all .mp4 files in input_dir → CompetitionResult."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        calibrator: BaseCalibrator,
        geometry_estimator: BaseGeometryEstimator,
        output_path: str = "result.json",
        max_frames: Optional[int] = None,
        min_track_frames: int = 5,
        min_track_speed: float = 0.0,
        origin_filter: Optional[dict] = None,
    ) -> None:
        self.processor = VideoProcessor(
            detector=detector,
            tracker=tracker,
            calibrator=calibrator,
            geometry_estimator=geometry_estimator,
            max_frames=max_frames,
            min_track_frames=min_track_frames,
            min_track_speed=min_track_speed,
            origin_filter=origin_filter,
        )
        self.output_path = output_path

    def run(self, input_dir: str) -> CompetitionResult:
        self.processor.detector.load()
        video_files = sorted(Path(input_dir).glob("*.mp4"))

        result = CompetitionResult()
        for vf in video_files:
            try:
                vr = self.processor.process(str(vf))
            except Exception:
                # 한 영상의 실패(손상 파일 등)가 전체 제출을 무효화하면 안 됨 —
                # 그 영상만 빈 objects로 기록하고 계속 진행. 채점상 해당 영상은
                # 전부 (0,0,0) 패딩 페널티를 받지만, result.json 미생성으로
                # 전체 FAILED가 되는 것보다는 항상 낫다.
                logger.exception("%s: processing failed — emitting empty objects", vf.stem)
                vr = VideoResult(video_id=vf.stem, objects=[])
            result.videos.append(vr)

        result.to_json(self.output_path)
        logger.info("Wrote %d videos → %s", len(result.videos), self.output_path)
        return result
