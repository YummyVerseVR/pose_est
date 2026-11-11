"""RGB-D posture proxies; independent of MediaPipe estimated world coordinates."""
import math
from .depth_geometry import sample_landmark
from .geometry import (_subtract, _horizontal, _normalize, _dot, _norm,
                       NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                       LEFT_HIP, RIGHT_HIP)
from .models import ObservedMetric, Point3D, PostureObservation


class RGBDGeometry:
    def __init__(self, config, reference=None):
        self.config = config
        self.reference = reference

    def reset(self):
        self.reference = None

    def observe(self, pose, packet, *, calibrating=False):
        keys = ("head_pitch_deg", "torso_forward_deg", "head_forward_ratio", "hip_position_delta")
        metrics, reasons = {}, {}
        if pose is None:
            return PostureObservation({}, {k: "no_pose" for k in keys}, "realsense_depth", 0.0)
        points = {i: sample_landmark(pose.image_landmarks[i], packet, self.config)
                  for i in (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
                  if i < len(pose.image_landmarks)}
        def point(i):
            sample = points.get(i)
            return sample.point if sample else None
        if self.reference is None and calibrating and point(NOSE) is not None:
            # Choose one anatomical side once, avoiding midpoint/side jumps during occlusion.
            enabled = {k for k, rule in self.config.metrics.items() if rule.enabled}
            needs_torso = bool(enabled & {"torso_forward_deg", "head_forward_ratio"})
            needs_hip = needs_torso or "hip_position_delta" in enabled
            candidates = []
            for e, s, h in ((LEFT_EAR, LEFT_SHOULDER, LEFT_HIP),
                             (RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP)):
                required = [e] + ([s] if needs_torso else []) + ([h] if needs_hip else [])
                if all(point(i) is not None for i in required):
                    candidates.append((min(points[i].confidence for i in required), e, s, h))
            if candidates:
                _, ear, shoulder, hip = max(candidates)
                forward = _normalize(_horizontal(_subtract(point(NOSE), point(ear))))
                if forward is not None:
                    self.reference = {"forward": [forward.x, forward.y, forward.z],
                                      "indices": [ear, shoulder, hip]}
        if self.reference is None:
            return PostureObservation({}, {k: "register_reference_with_C" for k in keys},
                                      "realsense_depth", 0.0)
        ear, shoulder, hip = self.reference["indices"]
        forward = Point3D(*self.reference["forward"])
        def emit(key, required, value):
            missing = [points[i].reason if i in points else "missing_landmark" for i in required if point(i) is None]
            if missing:
                reasons[key] = ",".join(dict.fromkeys(missing))
            elif value is not None and math.isfinite(value):
                metrics[key] = ObservedMetric(value, min(points[i].confidence for i in required))
            else:
                reasons[key] = "degenerate_geometry"
        head = None
        if point(NOSE) is not None and point(ear) is not None:
            face = _subtract(point(NOSE), point(ear))
            if _norm(_horizontal(face)) > 1e-6:
                head = math.degrees(math.atan2(-face.y, _norm(_horizontal(face))))
        emit("head_pitch_deg", [NOSE, ear], head)
        torso_angle = head_ratio = None
        if point(shoulder) is not None and point(hip) is not None:
            torso = _subtract(point(shoulder), point(hip))
            if _norm(torso) > 1e-6 and -torso.y > 1e-6:
                torso_angle = math.degrees(math.atan2(_dot(_horizontal(torso), forward), -torso.y))
                if point(ear) is not None:
                    head_ratio = _dot(_horizontal(_subtract(point(ear), point(shoulder))), forward) / _norm(torso)
        emit("torso_forward_deg", [shoulder, hip], torso_angle)
        emit("head_forward_ratio", [ear, shoulder, hip], head_ratio)
        # Experimental Euclidean displacement, not pelvic tilt/sliding detection.
        if calibrating and point(hip) is not None and "hip_origin" not in self.reference:
            p = point(hip)
            self.reference["hip_origin"] = [p.x, p.y, p.z]
        delta = None
        if point(hip) is not None and "hip_origin" in self.reference:
            delta = _norm(_subtract(point(hip), Point3D(*self.reference["hip_origin"])))
        emit("hip_position_delta", [hip], delta)
        used = [points[i].valid_ratio for i in (NOSE, ear, shoulder, hip) if i in points]
        return PostureObservation(metrics, reasons, "realsense_depth", sum(used)/4)
