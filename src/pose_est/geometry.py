from __future__ import annotations

import math
from typing import Sequence

from .models import ObservedMetric, Point3D, PoseFrame, PostureObservation


NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

_EPSILON = 1e-8


def _add(a: Point3D, b: Point3D) -> Point3D:
    return Point3D(a.x + b.x, a.y + b.y, a.z + b.z)


def _subtract(a: Point3D, b: Point3D) -> Point3D:
    return Point3D(a.x - b.x, a.y - b.y, a.z - b.z)


def _scale(point: Point3D, factor: float) -> Point3D:
    return Point3D(point.x * factor, point.y * factor, point.z * factor)


def _midpoint(a: Point3D, b: Point3D) -> Point3D:
    return _scale(_add(a, b), 0.5)


def _dot(a: Point3D, b: Point3D) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _norm(point: Point3D) -> float:
    return math.sqrt(_dot(point, point))


def _distance(a: Point3D, b: Point3D) -> float:
    return _norm(_subtract(a, b))


def _horizontal(point: Point3D) -> Point3D:
    return Point3D(point.x, 0.0, point.z)


def _normalize(point: Point3D) -> Point3D | None:
    length = _norm(point)
    if length < _EPSILON:
        return None
    return _scale(point, 1.0 / length)


def _visible(point: Point3D, threshold: float) -> bool:
    return min(point.visibility, point.presence) >= threshold


def _select_reference_points(
    landmarks: Sequence[Point3D], threshold: float
) -> tuple[Point3D, Point3D, Point3D, Point3D, float] | None:
    if len(landmarks) <= RIGHT_HIP:
        return None

    nose = landmarks[NOSE]
    if not _visible(nose, threshold):
        return None

    left = (
        landmarks[LEFT_EAR],
        landmarks[LEFT_SHOULDER],
        landmarks[LEFT_HIP],
    )
    right = (
        landmarks[RIGHT_EAR],
        landmarks[RIGHT_SHOULDER],
        landmarks[RIGHT_HIP],
    )

    if all(_visible(point, threshold) for point in (*left, *right)):
        ear = _midpoint(left[0], right[0])
        shoulder = _midpoint(left[1], right[1])
        hip = _midpoint(left[2], right[2])
        confidence = min(
            nose.visibility,
            nose.presence,
            *(min(point.visibility, point.presence) for point in (*left, *right)),
        )
        return nose, ear, shoulder, hip, confidence

    candidates = []
    for side in (left, right):
        confidence = min(
            nose.visibility,
            nose.presence,
            *(min(point.visibility, point.presence) for point in side),
        )
        if confidence >= threshold:
            candidates.append((confidence, side))

    if not candidates:
        return None

    confidence, (ear, shoulder, hip) = max(candidates, key=lambda item: item[0])
    return nose, ear, shoulder, hip, confidence


def foot_metric_keys(side: str) -> tuple[str, str, str, str]:
    if side not in {"left", "right"}:
        raise ValueError(f"unknown foot side: {side}")
    return (
        f"{side}_foot_x_ratio",
        f"{side}_foot_y_ratio",
        f"{side}_foot_z_ratio",
        f"{side}_foot_pitch_deg",
    )


def _compute_foot_metrics(
    landmarks: Sequence[Point3D], side: str, threshold: float
) -> dict[str, ObservedMetric]:
    indices = {
        "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX),
        "right": (
            RIGHT_HIP,
            RIGHT_KNEE,
            RIGHT_ANKLE,
            RIGHT_HEEL,
            RIGHT_FOOT_INDEX,
        ),
    }
    hip_index, knee_index, ankle_index, heel_index, toe_index = indices[side]
    if len(landmarks) <= toe_index:
        return {}

    hip = landmarks[hip_index]
    knee = landmarks[knee_index]
    ankle = landmarks[ankle_index]
    heel = landmarks[heel_index]
    toe = landmarks[toe_index]
    points = (hip, knee, ankle, heel, toe)
    if not all(_visible(point, threshold) for point in points):
        return {}

    leg_scale = _distance(hip, knee) + _distance(knee, ankle)
    if leg_scale < _EPSILON:
        return {}

    foot_center = _midpoint(heel, toe)
    local_position = _scale(_subtract(foot_center, hip), 1.0 / leg_scale)
    foot_vector = _subtract(toe, heel)
    horizontal_length = _norm(_horizontal(foot_vector))
    foot_pitch_deg = math.degrees(
        math.atan2(-foot_vector.y, max(horizontal_length, _EPSILON))
    )
    confidence = min(
        min(point.visibility, point.presence) for point in points
    )
    x_key, y_key, z_key, pitch_key = foot_metric_keys(side)
    metric = lambda value: ObservedMetric(value=value, confidence=confidence)
    return {
        x_key: metric(local_position.x),
        y_key: metric(local_position.y),
        z_key: metric(local_position.z),
        pitch_key: metric(foot_pitch_deg),
    }


def compute_posture_observation(
    frame: PoseFrame | None,
    visibility_threshold: float,
    foot_visibility_threshold: float | None = None,
) -> PostureObservation:
    """Compute camera-relative posture proxies from MediaPipe landmarks.

    The values are deliberately used as relative measures against a calibrated
    baseline. They must not be interpreted as medical spinal measurements.
    """

    if frame is None:
        return PostureObservation()

    metrics: dict[str, ObservedMetric] = {}
    selected = _select_reference_points(frame.world_landmarks, visibility_threshold)
    if selected is not None:
        nose, ear, shoulder, hip, confidence = selected
        face_vector = _subtract(nose, ear)
        face_horizontal = _horizontal(face_vector)
        face_forward = _normalize(face_horizontal)
        torso_vector = _subtract(shoulder, hip)
        torso_length = _norm(torso_vector)

        if face_forward is not None and torso_length >= _EPSILON:
            face_horizontal_length = _norm(face_horizontal)
            head_pitch_deg = math.degrees(
                math.atan2(-face_vector.y, max(face_horizontal_length, _EPSILON))
            )

            # Positive means that the shoulders have moved in the direction the
            # face points, relative to the hips. MediaPipe world y grows downwards.
            torso_forward_distance = _dot(_horizontal(torso_vector), face_forward)
            torso_vertical_rise = max(-torso_vector.y, _EPSILON)
            torso_forward_deg = math.degrees(
                math.atan2(torso_forward_distance, torso_vertical_rise)
            )

            head_from_shoulder = _subtract(ear, shoulder)
            head_forward_ratio = (
                _dot(_horizontal(head_from_shoulder), face_forward) / torso_length
            )

            metric = lambda value: ObservedMetric(
                value=value, confidence=confidence
            )
            metrics.update(
                {
                    "head_pitch_deg": metric(head_pitch_deg),
                    "torso_forward_deg": metric(torso_forward_deg),
                    "head_forward_ratio": metric(head_forward_ratio),
                }
            )

    foot_threshold = (
        visibility_threshold
        if foot_visibility_threshold is None
        else foot_visibility_threshold
    )
    for side in ("left", "right"):
        metrics.update(
            _compute_foot_metrics(
                frame.world_landmarks, side, foot_threshold
            )
        )
    return PostureObservation(metrics=metrics)
