from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class MetricRule:
    enabled: bool
    lower_delta: float
    upper_delta: float
    warn_hold_seconds: float
    recover_hold_seconds: float
    message_below: str
    message_above: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricRule":
        return cls(
            enabled=bool(data.get("enabled", True)),
            lower_delta=float(data["lower_delta"]),
            upper_delta=float(data["upper_delta"]),
            warn_hold_seconds=float(data.get("warn_hold_seconds", 0.8)),
            recover_hold_seconds=float(data.get("recover_hold_seconds", 1.0)),
            message_below=str(data["message_below"]),
            message_above=str(data["message_above"]),
        )


@dataclass(frozen=True)
class FootRule:
    enabled: bool = True
    motion_window_seconds: float = 1.5
    min_window_samples: int = 12
    visibility_threshold: float = 0.35
    position_threshold_ratio: float = 0.20
    motion_threshold_ratio: float = 0.04
    pitch_threshold_deg: float = 35.0
    depth_weight: float = 0.25
    require_position_and_pitch: bool = True
    warn_hold_seconds: float = 1.0
    recover_hold_seconds: float = 1.0
    message_left: str = "左足の裏を足台に着けてください"
    message_right: str = "右足の裏を足台に着けてください"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FootRule":
        defaults = cls()
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            motion_window_seconds=float(
                data.get("motion_window_seconds", defaults.motion_window_seconds)
            ),
            min_window_samples=int(
                data.get("min_window_samples", defaults.min_window_samples)
            ),
            visibility_threshold=float(
                data.get("visibility_threshold", defaults.visibility_threshold)
            ),
            position_threshold_ratio=float(
                data.get("position_threshold_ratio", defaults.position_threshold_ratio)
            ),
            motion_threshold_ratio=float(
                data.get("motion_threshold_ratio", defaults.motion_threshold_ratio)
            ),
            pitch_threshold_deg=float(
                data.get("pitch_threshold_deg", defaults.pitch_threshold_deg)
            ),
            depth_weight=float(data.get("depth_weight", defaults.depth_weight)),
            require_position_and_pitch=bool(
                data.get(
                    "require_position_and_pitch",
                    defaults.require_position_and_pitch,
                )
            ),
            warn_hold_seconds=float(
                data.get("warn_hold_seconds", defaults.warn_hold_seconds)
            ),
            recover_hold_seconds=float(
                data.get("recover_hold_seconds", defaults.recover_hold_seconds)
            ),
            message_left=str(data.get("message_left", defaults.message_left)),
            message_right=str(data.get("message_right", defaults.message_right)),
        )


def _default_metrics() -> dict[str, MetricRule]:
    return {
        "head_pitch_deg": MetricRule(
            enabled=True,
            lower_delta=-15.0,
            upper_delta=8.0,
            warn_hold_seconds=0.7,
            recover_hold_seconds=0.8,
            message_below="顎を引きすぎています",
            message_above="もう少し顎を引いてください",
        ),
        "torso_forward_deg": MetricRule(
            enabled=True,
            lower_delta=-8.0,
            upper_delta=10.0,
            warn_hold_seconds=0.8,
            recover_hold_seconds=1.0,
            message_below="少し前へ戻ってください",
            message_above="背筋をもう少し起こしてください",
        ),
        "head_forward_ratio": MetricRule(
            enabled=True,
            lower_delta=-0.10,
            upper_delta=0.10,
            warn_hold_seconds=0.8,
            recover_hold_seconds=1.0,
            message_below="頭を少し前へ戻してください",
            message_above="頭を少し後ろへ戻してください",
        ),
    }


@dataclass(frozen=True)
class PostureConfig:
    model_path: str = "models/pose_landmarker_full.task"
    baseline_path: str = "baseline.json"
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    mirror_image: bool = True
    visibility_threshold: float = 0.55
    smoothing_alpha: float = 0.25
    calibration_seconds: float = 3.0
    min_calibration_samples: int = 30
    unknown_hold_seconds: float = 0.3
    metrics: Mapping[str, MetricRule] = field(default_factory=_default_metrics)
    feet: FootRule = field(default_factory=FootRule)

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be in [0, 1]")
        if self.calibration_seconds <= 0.0:
            raise ValueError("calibration_seconds must be positive")
        if self.min_calibration_samples <= 0:
            raise ValueError("min_calibration_samples must be positive")
        for key, rule in self.metrics.items():
            if rule.lower_delta >= rule.upper_delta:
                raise ValueError(f"{key}: lower_delta must be smaller than upper_delta")
        if self.feet.motion_window_seconds <= 0.0:
            raise ValueError("feet.motion_window_seconds must be positive")
        if self.feet.min_window_samples < 2:
            raise ValueError("feet.min_window_samples must be at least 2")
        if not 0.0 <= self.feet.visibility_threshold <= 1.0:
            raise ValueError("feet.visibility_threshold must be in [0, 1]")
        if self.feet.position_threshold_ratio <= 0.0:
            raise ValueError("feet.position_threshold_ratio must be positive")
        if self.feet.motion_threshold_ratio <= 0.0:
            raise ValueError("feet.motion_threshold_ratio must be positive")
        if self.feet.pitch_threshold_deg <= 0.0:
            raise ValueError("feet.pitch_threshold_deg must be positive")
        if not 0.0 <= self.feet.depth_weight <= 1.0:
            raise ValueError("feet.depth_weight must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PostureConfig":
        defaults = cls()
        metric_data = data.get("metrics", {})
        metrics = dict(defaults.metrics)
        for key, value in metric_data.items():
            metrics[str(key)] = MetricRule.from_dict(value)

        return cls(
            model_path=str(data.get("model_path", defaults.model_path)),
            baseline_path=str(data.get("baseline_path", defaults.baseline_path)),
            camera_index=int(data.get("camera_index", defaults.camera_index)),
            frame_width=int(data.get("frame_width", defaults.frame_width)),
            frame_height=int(data.get("frame_height", defaults.frame_height)),
            mirror_image=bool(data.get("mirror_image", defaults.mirror_image)),
            visibility_threshold=float(
                data.get("visibility_threshold", defaults.visibility_threshold)
            ),
            smoothing_alpha=float(data.get("smoothing_alpha", defaults.smoothing_alpha)),
            calibration_seconds=float(
                data.get("calibration_seconds", defaults.calibration_seconds)
            ),
            min_calibration_samples=int(
                data.get("min_calibration_samples", defaults.min_calibration_samples)
            ),
            unknown_hold_seconds=float(
                data.get("unknown_hold_seconds", defaults.unknown_hold_seconds)
            ),
            metrics=metrics,
            feet=FootRule.from_dict(data.get("feet", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PostureConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
