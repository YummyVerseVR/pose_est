from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys
import time

from .analyzer import PostureAnalyzer
from .baseline import PostureBaseline
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
        self.baseline_path = Path(baseline_path or config.baseline_path)
        self.font_path = font_path

    def run(
        self,
        on_result: Callable[[PostureAssessment], None] | None = None,
    ) -> None:
        import cv2

        baseline = None
        if self.baseline_path.is_file():
            baseline = PostureBaseline.load(self.baseline_path)
        analyzer = PostureAnalyzer(self.config, baseline)
        renderer = OverlayRenderer(self.font_path)

        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"camera {self.config.camera_index} could not be opened")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        try:
            with MediaPipePoseEstimator(self.model_path) as estimator:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("failed to read a frame from the camera")

                    timestamp_ms = time.monotonic_ns() // 1_000_000
                    pose = estimator.process(frame, timestamp_ms)
                    assessment = analyzer.update(pose, timestamp_ms)

                    updated_baseline = analyzer.take_baseline_update()
                    if updated_baseline is not None:
                        updated_baseline.save(self.baseline_path)

                    if on_result is not None:
                        on_result(assessment)

                    display_frame = frame
                    display_pose = pose
                    if self.config.mirror_image:
                        display_frame = cv2.flip(frame, 1)
                        display_pose = _mirror_pose_for_display(pose)

                    rendered = renderer.render(
                        display_frame, display_pose, assessment
                    )
                    cv2.imshow("Posture Monitor", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("c"), ord("C")):
                        analyzer.start_calibration(timestamp_ms)
        finally:
            capture.release()
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = PostureConfig.load(args.config) if args.config else PostureConfig()
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
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
