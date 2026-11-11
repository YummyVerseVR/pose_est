from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
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
    # Feet are often occluded by a table. Enable only if both feet are visible.
    enabled: bool = False
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
    message_left: str = "左足の位置を確認してください"
    message_right: str = "右足の位置を確認してください"

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
            warn_hold_seconds=2.5,
            recover_hold_seconds=1.0,
            message_below="頭の向きを無理なく基準姿勢へ戻してください",
            message_above="頭の向きを無理なく基準姿勢へ戻してください",
        ),
        "torso_forward_deg": MetricRule(
            enabled=True,
            lower_delta=-8.0,
            upper_delta=10.0,
            warn_hold_seconds=2.5,
            recover_hold_seconds=1.0,
            message_below="座り直して体幹を基準姿勢へ戻してください",
            message_above="座り直して体幹を基準姿勢へ戻してください",
        ),
        "head_forward_ratio": MetricRule(
            enabled=True,
            lower_delta=-0.10,
            upper_delta=0.10,
            warn_hold_seconds=2.5,
            recover_hold_seconds=1.0,
            message_below="頭と肩の位置を無理なく整えてください",
            message_above="頭と肩の位置を無理なく整えてください",
        ),
    }


@dataclass(frozen=True)
class PostureConfig:
    camera_source: str = "webcam"
    rgbd_width: int = 640
    rgbd_height: int = 480
    rgbd_fps: int = 30
    frame_timeout_ms: int = 3000
    sensor_serial: str = ""
    camera_side: str = "left"
    placement_id: str = "initial"
    camera_azimuth_deg: float = 75.0
    camera_distance_m: float = 1.3
    camera_height_m: float = 1.0
    support_profile: str = "general_meal_support"
    research_log_path: str | None = None
    depth_patch_size: int = 5
    depth_min_pixels: int = 9
    depth_min_m: float = 0.2
    depth_max_m: float = 3.0
    depth_min_valid_ratio: float = 0.6
    depth_max_mad_m: float = 0.02
    depth_max_spread_m: float = 0.06
    depth_min_inlier_ratio: float = 0.8
    depth_min_confidence: float = 0.5
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
    feedback_min_interval_seconds: float = 10.0
    feedback_repeat_seconds: float = 35.0
    metrics: Mapping[str, MetricRule] = field(default_factory=_default_metrics)
    feet: FootRule = field(default_factory=FootRule)

    def __post_init__(self) -> None:
        if self.camera_source not in {"webcam", "realsense"}:
            raise ValueError("camera_source must be webcam or realsense")
        if self.camera_side not in {"left", "right"}:
            raise ValueError("camera_side must be left or right (optical placement)")
        if self.support_profile not in {"general_meal_support", "clinician_defined"}:
            raise ValueError("unknown support_profile")
        if not self.placement_id:
            raise ValueError("placement_id must identify the fixed camera setup")
        for name in ("rgbd_width", "rgbd_height", "rgbd_fps", "frame_timeout_ms", "depth_min_pixels"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.depth_patch_size, int) or self.depth_patch_size < 1 or self.depth_patch_size % 2 != 1:
            raise ValueError("depth_patch_size must be a positive odd integer")
        if self.depth_min_pixels > self.depth_patch_size ** 2:
            raise ValueError("depth_min_pixels exceeds patch area")
        for name in ("depth_min_valid_ratio", "depth_min_inlier_ratio", "depth_min_confidence"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("depth_min_m", "depth_max_m", "depth_max_mad_m", "depth_max_spread_m", "camera_distance_m", "camera_height_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.camera_azimuth_deg) or not 0 <= self.camera_azimuth_deg <= 90:
            raise ValueError("camera_azimuth_deg must be in [0,90]")
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError("invalid depth range")
        if self.camera_source == "realsense" and self.feet.enabled:
            raise ValueError("RGB-D feet metrics are unsupported; disable feet")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be in [0, 1]")
        if self.calibration_seconds <= 0.0:
            raise ValueError("calibration_seconds must be positive")
        if self.min_calibration_samples <= 0:
            raise ValueError("min_calibration_samples must be positive")
        if self.unknown_hold_seconds < 0:
            raise ValueError("unknown_hold_seconds must be non-negative")
        if self.feedback_min_interval_seconds < 0:
            raise ValueError("feedback_min_interval_seconds must be non-negative")
        if self.feedback_repeat_seconds <= 0:
            raise ValueError("feedback_repeat_seconds must be positive")
        for key, rule in self.metrics.items():
            if rule.lower_delta >= rule.upper_delta:
                raise ValueError(f"{key}: lower_delta must be smaller than upper_delta")
            if rule.warn_hold_seconds < 0 or rule.recover_hold_seconds < 0:
                raise ValueError(f"{key}: hold times must be non-negative")
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
        if self.feet.warn_hold_seconds < 0 or self.feet.recover_hold_seconds < 0:
            raise ValueError("feet: hold times must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PostureConfig":
        defaults = cls()
        metric_data = data.get("metrics", {})
        metrics = dict(defaults.metrics)
        for key, value in metric_data.items():
            metrics[str(key)] = MetricRule.from_dict(value)

        return cls(
            **{key: data.get(key, getattr(defaults, key)) for key in ('camera_source', 'rgbd_width', 'rgbd_height', 'rgbd_fps', 'frame_timeout_ms', 'sensor_serial', 'camera_side', 'placement_id', 'camera_azimuth_deg', 'camera_distance_m', 'camera_height_m', 'support_profile', 'research_log_path', 'depth_patch_size', 'depth_min_pixels', 'depth_min_m', 'depth_max_m', 'depth_min_valid_ratio', 'depth_max_mad_m', 'depth_max_spread_m', 'depth_min_inlier_ratio', 'depth_min_confidence')},
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
            feedback_min_interval_seconds=float(
                data.get("feedback_min_interval_seconds", defaults.feedback_min_interval_seconds)
            ),
            feedback_repeat_seconds=float(
                data.get("feedback_repeat_seconds", defaults.feedback_repeat_seconds)
            ),
            metrics=metrics,
            feet=FootRule.from_dict(data.get("feet", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PostureConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
