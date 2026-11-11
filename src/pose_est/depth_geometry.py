"""Aligned depth -> camera XYZ metres (+X right, +Y down, +Z away).

Production uses the SDK deprojector (including its distortion model). The pinhole
fallback is for synthetic undistorted intrinsics only; world_landmarks are unused.
"""
from dataclasses import dataclass
import math
import numpy as np
from .models import Point3D


@dataclass(frozen=True)
class DepthPoint:
    point: Point3D | None
    confidence: float
    valid_ratio: float
    reason: str = ""


def sample_landmark(landmark, packet, config):
    def unknown(reason, ratio=0.0):
        return DepthPoint(None, 0.0, ratio, reason)
    values = (landmark.x, landmark.y, landmark.visibility, landmark.presence)
    if not all(math.isfinite(v) for v in values):
        return unknown("nonfinite_landmark")
    confidence = min(landmark.visibility, landmark.presence)
    if confidence < config.visibility_threshold:
        return unknown("landmark_confidence")
    if not (0 <= landmark.x < 1 and 0 <= landmark.y < 1):
        return unknown("outside_image")
    if packet.depth is None or packet.intrinsics is None:
        return unknown("missing_depth")
    h, w = packet.color_bgr.shape[:2]
    if packet.depth.shape != (h, w):
        return unknown("unaligned_depth")
    if packet.depth_scale is None or not math.isfinite(packet.depth_scale) or packet.depth_scale <= 0:
        return unknown("invalid_depth_scale")
    u, v = int(landmark.x * w), int(landmark.y * h)
    radius = config.depth_patch_size // 2
    patch = np.asarray(packet.depth[max(0, v-radius):min(h, v+radius+1),
                                    max(0, u-radius):min(w, u+radius+1)], dtype=float) * packet.depth_scale
    valid = patch[np.isfinite(patch) & (patch >= config.depth_min_m) & (patch <= config.depth_max_m)]
    ratio = valid.size / patch.size
    if valid.size < config.depth_min_pixels or ratio < config.depth_min_valid_ratio:
        return unknown("insufficient_depth", ratio)
    distance = float(np.median(valid))
    mad = float(np.median(np.abs(valid - distance)))
    # Reject mixed foreground/background patches even if their MAD is zero.
    inlier_ratio = float(np.mean(np.abs(valid - distance) <= config.depth_max_spread_m))
    if mad > config.depth_max_mad_m or inlier_ratio < config.depth_min_inlier_ratio:
        return unknown("depth_spread", ratio)
    confidence = min(1.0, confidence) * ratio * inlier_ratio * max(0.0, 1-mad/config.depth_max_mad_m)
    if confidence < config.depth_min_confidence:
        return unknown("depth_confidence", ratio)
    intr = packet.intrinsics
    if packet.deproject is not None:
        xyz = packet.deproject(intr, [u, v], distance)
    else:
        if intr.fx <= 0 or intr.fy <= 0:
            return unknown("invalid_intrinsics", ratio)
        xyz = [(u-intr.ppx)*distance/intr.fx, (v-intr.ppy)*distance/intr.fy, distance]
    if not all(math.isfinite(x) for x in xyz):
        return unknown("invalid_projection", ratio)
    return DepthPoint(Point3D(*xyz, confidence, confidence), confidence, ratio)
