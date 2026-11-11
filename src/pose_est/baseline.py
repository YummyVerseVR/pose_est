from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

from .models import PostureObservation


@dataclass(frozen=True)
class PostureBaseline:
    metrics: Mapping[str, float]
    created_at: str
    version: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def create(cls, metrics: Mapping[str, float]) -> "PostureBaseline":
        return cls(
            metrics=dict(metrics),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "PostureBaseline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("version", 0)) not in (1, 2):
            raise ValueError("unsupported baseline version")
        if not all(math.isfinite(float(v)) for v in data["metrics"].values()):
            raise ValueError("nonfinite baseline")
        return cls(
            metrics={str(key): float(value) for key, value in data["metrics"].items()},
            created_at=str(data["created_at"]),
            version=int(data["version"]),
            metadata=data.get("metadata", {}),
        )


class CalibrationSession:
    def __init__(
        self,
        started_at_ms: int,
        duration_seconds: float,
        min_samples: int,
        metric_keys: set[str],
    ) -> None:
        self.started_at_ms = started_at_ms
        self.duration_ms = int(duration_seconds * 1000.0)
        self.min_samples = min_samples
        self.metric_keys = metric_keys
        self.samples: dict[str, list[float]] = {key: [] for key in metric_keys}

    def add(self, observation: PostureObservation) -> None:
        for key in self.metric_keys:
            metric = observation.metrics.get(key)
            if metric is not None:
                self.samples[key].append(metric.value)

    def progress(self, timestamp_ms: int) -> float:
        time_progress = (timestamp_ms - self.started_at_ms) / max(self.duration_ms, 1)
        if not self.samples:
            return 0.0
        sample_progress = min(
            len(values) / self.min_samples for values in self.samples.values()
        )
        return max(0.0, min(1.0, time_progress, sample_progress))

    def ready(self, timestamp_ms: int) -> bool:
        enough_time = timestamp_ms - self.started_at_ms >= self.duration_ms
        enough_samples = all(
            len(values) >= self.min_samples for values in self.samples.values()
        )
        return enough_time and enough_samples

    def finish(self) -> PostureBaseline:
        if not all(len(values) >= self.min_samples for values in self.samples.values()):
            raise RuntimeError("calibration does not have enough valid samples")
        return PostureBaseline.create(
            {key: float(median(values)) for key, values in self.samples.items()}
        )



def validate_baseline(baseline, measurement_source, context=None):
    """Do not compare estimated world coordinates with metric sensor coordinates."""
    source = baseline.metadata.get("measurement_source", "mediapipe_world")
    if source != measurement_source:
        raise ValueError("measurement source changed; press C to recalibrate")
    if measurement_source == "realsense_depth":
        if baseline.version != 2 or baseline.metadata.get("context") != context:
            raise ValueError("sensor/setup/settings changed; press C to recalibrate")
        ref = baseline.metadata.get("reference", {})
        forward = ref.get("forward", [])
        indices = ref.get("indices", [])
        if (len(forward) != 3 or not all(math.isfinite(float(v)) for v in forward)
                or abs(sum(float(v)**2 for v in forward)-1) > 1e-5
                or abs(float(forward[1])) > 1e-6
                or indices not in ([7, 11, 23], [8, 12, 24])):
            raise ValueError("invalid RGB-D reference; press C to recalibrate")
        if "hip_origin" in ref and (len(ref["hip_origin"]) != 3 or
                not all(math.isfinite(float(v)) for v in ref["hip_origin"])):
            raise ValueError("invalid hip origin; press C to recalibrate")


def rgbd_context(config, packet):
    """Bind calibration to actual intrinsics, hardware, placement and quality policy."""
    from dataclasses import asdict
    settings = asdict(config)
    keys = [k for k in settings if k.startswith("depth_") or k in
            ("visibility_threshold", "support_profile", "metrics", "placement_id")]
    intr = packet.intrinsics
    return {"sensor": packet.metadata, "settings": {k: settings[k] for k in keys},
            "intrinsics": {k: getattr(intr, k) for k in ("width", "height", "fx", "fy", "ppx", "ppy")},
            "distortion": str(getattr(intr, "model", "none")),
            "coeffs": list(getattr(intr, "coeffs", [])), "depth_scale": packet.depth_scale}
