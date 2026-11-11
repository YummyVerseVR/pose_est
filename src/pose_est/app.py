from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

from .analyzer import PostureAnalyzer
from .baseline import PostureBaseline, validate_baseline, rgbd_context
from .camera_source import WebcamFrameSource
from .rgbd_geometry import RGBDGeometry
from .config import PostureConfig
from .mediapipe_estimator import MediaPipePoseEstimator
from .models import Point3D, PoseFrame, PostureAssessment
from .overlay import OverlayRenderer


class CameraPostureApp:
    """Webcam application shell.

    on_result is the integration seam for a VR process, logger, socket
    publisher, or any other consumer. The UI is intentionally optional to the
    core analyzer.
    """

    def __init__(
        self,
        config: PostureConfig,
        *,
        model_path: str | Path | None = None,
        baseline_path: str | Path | None = None,
        font_path: str | None = None,
    ) -> None:
        self.config = config
        self.model_path = Path(model_path or config.model_path)
        self.baseline_path = Path(baseline_path or ("baseline_realsense.json" if config.camera_source == "realsense" and config.baseline_path == "baseline.json" else config.baseline_path))
        self.font_path = font_path

    def run(
        self,
        on_result: Callable[[PostureAssessment], None] | None = None,
    ) -> None:
        import cv2

        baseline = None
        baseline_notice = ""
        if self.baseline_path.is_file():
            try:
                baseline = PostureBaseline.load(self.baseline_path)
                if self.config.camera_source == "webcam":
                    validate_baseline(baseline, "mediapipe_world")
            except (ValueError, KeyError, TypeError) as exc:
                baseline_notice = f"基準を再登録してください: {exc}"
                baseline = None
        analyzer = PostureAnalyzer(self.config, baseline)
        renderer = OverlayRenderer(self.font_path)
        if self.config.camera_source == "realsense":
            from .realsense_source import RealSenseFrameSource
            source = RealSenseFrameSource(self.config)
        else:
            source = WebcamFrameSource(self.config)
        geometry = RGBDGeometry(self.config)
        context = None
        calibrating = False
        try:
            with ExitStack() as stack:
                # Register cleanup before acquisition, including partial startup failures.
                stack.callback(source.stop)
                source.start()
                log = None
                if self.config.research_log_path:
                    log = stack.enter_context(Path(self.config.research_log_path).open("a", encoding="utf-8"))
                estimator = stack.enter_context(MediaPipePoseEstimator(self.model_path))
                while True:
                    packet = source.read()
                    frame, timestamp_ms = packet.color_bgr, packet.timestamp_ms
                    if self.config.camera_source == "realsense":
                        actual_context = rgbd_context(self.config, packet)
                        if actual_context != context:
                            context = actual_context
                            calibrating = False
                            analyzer.cancel_calibration()
                            geometry.reset()
                            if baseline is not None:
                                try:
                                    validate_baseline(baseline, "realsense_depth", context)
                                    geometry.reference = dict(baseline.metadata["reference"])
                                except (ValueError, KeyError, TypeError) as exc:
                                    baseline_notice = f"基準を再登録してください: {exc}"
                                    baseline = None
                            analyzer.set_baseline(baseline)
                    pose = estimator.process(frame, timestamp_ms)
                    observation = None
                    if self.config.camera_source == "realsense":
                        observation = geometry.observe(pose, packet, calibrating=calibrating and not analyzer.paused)
                    assessment = analyzer.update(pose, timestamp_ms, observation=observation)
                    updated_baseline = analyzer.take_baseline_update()
                    if updated_baseline is not None:
                        if observation is not None:
                            updated_baseline = replace(updated_baseline, version=2, metadata={
                                "measurement_source": "realsense_depth", "context": context,
                                "reference": geometry.reference,
                            })
                            analyzer.set_baseline(updated_baseline)
                        updated_baseline.save(self.baseline_path)
                        baseline = updated_baseline
                        baseline_notice = ""
                        calibrating = False
                    if on_result is not None:
                        on_result(assessment)
                    if log is not None:
                        payload = assessment.to_dict()
                        payload["measurement_source"] = observation.measurement_source if observation else "mediapipe_world"
                        payload["unknown_reasons"] = dict(observation.unknown_reasons) if observation else {}
                        payload["valid_depth_ratio"] = observation.valid_depth_ratio if observation else None
                        # No images, depth arrays, sensor serials or landmark coordinates.
                        log.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
                        log.flush()
                    display_frame, display_pose = frame, pose
                    if self.config.mirror_image:
                        display_frame = cv2.flip(frame, 1)
                        display_pose = _mirror_pose_for_display(pose)
                    diagnostics = [f"Camera: {self.config.camera_source} / {self.config.camera_side}"]
                    if baseline_notice:
                        diagnostics.append(baseline_notice)
                    if observation is not None:
                        diagnostics.append(f"RGB-D valid: {observation.valid_depth_ratio:.0%}")
                        for key, rule in self.config.metrics.items():
                            if rule.enabled:
                                diagnostics.append(f"{key}: {observation.unknown_reasons.get(key, 'tracked')}")
                    if analyzer.calibration_sample_counts is not None:
                        diagnostics.append("Samples: " + ", ".join(f"{k}={n}" for k,n in analyzer.calibration_sample_counts.items()))
                    rendered = renderer.render(display_frame, display_pose, assessment, diagnostics=diagnostics)
                    cv2.imshow("Posture Monitor", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("c"), ord("C")):
                        analyzer.set_paused(False)
                        analyzer.set_baseline(None)
                        baseline = None
                        geometry.reset()
                        calibrating = True
                        analyzer.start_calibration(timestamp_ms)
                    if key in (ord("p"), ord("P")):
                        analyzer.set_paused(not analyzer.paused)
                        calibrating = False
        finally:
            cv2.destroyAllWindows()


def _mirror_pose_for_display(pose: PoseFrame | None) -> PoseFrame | None:
    """Mirror only image coordinates while preserving anatomical assessment data."""

    if pose is None:
        return None
    image_landmarks = tuple(
        Point3D(
            x=1.0 - point.x,
            y=point.y,
            z=point.z,
            visibility=point.visibility,
            presence=point.presence,
        )
        for point in pose.image_landmarks
    )
    return PoseFrame(
        image_landmarks=image_landmarks,
        world_landmarks=pose.world_landmarks,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam posture feedback")
    parser.add_argument(
        "--config",
        default=None,
        help="JSON config path. Defaults to built-in values.",
    )
    parser.add_argument("--model", default=None, help="MediaPipe .task model path")
    parser.add_argument("--baseline", default=None, help="Per-user baseline JSON path")
    parser.add_argument("--font", default=None, help="Japanese TrueType/OpenType font path")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--camera-source", choices=("webcam", "realsense"), default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = PostureConfig.load(args.config) if args.config else PostureConfig()
        if args.camera_source is not None:
            config = replace(config, camera_source=args.camera_source)
        if args.camera is not None:
            config = replace(config, camera_index=args.camera)
        app = CameraPostureApp(
            config,
            model_path=args.model,
            baseline_path=args.baseline,
            font_path=args.font,
        )
        app.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
