from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping


class AssessmentStatus(str, Enum):
    OK = "ok"
    ADVICE = "advice"
    UNKNOWN = "unknown"
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    PAUSED = "paused"


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class PoseFrame:
    """Landmarks for one frame.

    image_landmarks use normalized image coordinates. world_landmarks
    use MediaPipe's estimated world coordinates and are used for assessment.
    """

    image_landmarks: tuple[Point3D, ...]
    world_landmarks: tuple[Point3D, ...]


@dataclass(frozen=True)
class ObservedMetric:
    value: float
    confidence: float


@dataclass(frozen=True)
class PostureObservation:
    metrics: Mapping[str, ObservedMetric] = field(default_factory=dict)
    unknown_reasons: Mapping[str, str] = field(default_factory=dict)
    measurement_source: str = "mediapipe_world"
    valid_depth_ratio: float | None = None


@dataclass(frozen=True)
class MetricAssessment:
    key: str
    status: AssessmentStatus
    value: float | None
    baseline: float | None
    deviation: float | None
    lower_delta: float
    upper_delta: float
    confidence: float
    direction: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "value": self.value,
            "baseline": self.baseline,
            "deviation": self.deviation,
            "lower_delta": self.lower_delta,
            "upper_delta": self.upper_delta,
            "confidence": self.confidence,
            "direction": self.direction,
            "message": self.message,
        }


@dataclass(frozen=True)
class FootAssessment:
    side: str
    status: AssessmentStatus
    position_deviation_ratio: float | None
    motion_rms_ratio: float | None
    pitch_deviation_deg: float | None
    confidence: float
    reasons: tuple[str, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "status": self.status.value,
            "position_deviation_ratio": self.position_deviation_ratio,
            "motion_rms_ratio": self.motion_rms_ratio,
            "pitch_deviation_deg": self.pitch_deviation_deg,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "message": self.message,
        }


@dataclass(frozen=True)
class PostureAssessment:
    timestamp_ms: int
    overall_status: AssessmentStatus
    calibrated: bool
    metrics: Mapping[str, MetricAssessment] = field(default_factory=dict)
    feet: Mapping[str, FootAssessment] = field(default_factory=dict)
    messages: tuple[str, ...] = ()
    calibration_progress: float | None = None
    # A rate-limited, actionable event. `messages` remain UI/status text.
    feedback_messages: tuple[str, ...] = ()

    @property
    def needs_feedback(self) -> bool:
        return bool(self.feedback_messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "overall_status": self.overall_status.value,
            "calibrated": self.calibrated,
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "feet": {key: value.to_dict() for key, value in self.feet.items()},
            "messages": list(self.messages),
            "calibration_progress": self.calibration_progress,
            "feedback_messages": list(self.feedback_messages),
        }

    def to_json(self, *, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii)
