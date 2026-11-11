"""Acquisition packets. Depth is aligned Z16; depth_scale converts to metres."""
from dataclasses import dataclass, field
import time
from typing import Any, Protocol


@dataclass(frozen=True)
class CameraPacket:
    color_bgr: Any
    timestamp_ms: int
    depth: Any = None
    depth_scale: float | None = None
    intrinsics: Any = None
    metadata: dict = field(default_factory=dict)
    deproject: Any = None


class FrameSource(Protocol):
    def start(self) -> None: ...
    def read(self) -> CameraPacket: ...
    def stop(self) -> None: ...


class MonotonicClock:
    def __init__(self):
        self.last = -1

    def now(self):
        self.last = max(self.last + 1, time.monotonic_ns() // 1_000_000)
        return self.last


class WebcamFrameSource:
    def __init__(self, config):
        self.config = config
        self.capture = None
        self.clock = MonotonicClock()

    def start(self):
        import cv2
        try:
            self.capture = cv2.VideoCapture(self.config.camera_index)
            if not self.capture.isOpened():
                raise RuntimeError(f"camera {self.config.camera_index} could not be opened")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        except BaseException:
            self.stop()
            raise

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("failed to read a frame from the camera")
        return CameraPacket(frame, self.clock.now(), metadata={"measurement_source": "mediapipe_world"})

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
