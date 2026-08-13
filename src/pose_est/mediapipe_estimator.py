from __future__ import annotations

from pathlib import Path

from .models import Point3D, PoseFrame


class MediaPipePoseEstimator:
    """Synchronous MediaPipe Pose Landmarker adapter for video frames."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {path}. "
                "Run 'python scripts/download_model.py' first."
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame_bgr, timestamp_ms: int) -> PoseFrame | None:
        import cv2

        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None

        return PoseFrame(
            image_landmarks=tuple(self._convert(item) for item in result.pose_landmarks[0]),
            world_landmarks=tuple(
                self._convert(item) for item in result.pose_world_landmarks[0]
            ),
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _convert(landmark) -> Point3D:
        return Point3D(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
            visibility=float(getattr(landmark, "visibility", 1.0) or 0.0),
            presence=float(getattr(landmark, "presence", 1.0) or 0.0),
        )

