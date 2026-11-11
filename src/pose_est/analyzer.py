from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .baseline import CalibrationSession, PostureBaseline
from .config import FootRule, MetricRule, PostureConfig
from .geometry import compute_posture_observation, foot_metric_keys
from .models import (
    AssessmentStatus,
    FootAssessment,
    MetricAssessment,
    PoseFrame,
    PostureAssessment,
    PostureObservation,
)


@dataclass
class _TemporalState:
    current: str = "unknown"
    candidate: str | None = None
    candidate_since_ms: int | None = None

    def update(
        self,
        raw_state: str,
        timestamp_ms: int,
        rule: MetricRule | FootRule,
        unknown_hold_seconds: float,
    ) -> str:
        if raw_state == self.current:
            self.candidate = None
            self.candidate_since_ms = None
            return self.current

        if raw_state != self.candidate:
            self.candidate = raw_state
            self.candidate_since_ms = timestamp_ms
            return self.current

        if raw_state == "ok":
            hold_ms = int(rule.recover_hold_seconds * 1000.0)
        elif raw_state == "unknown":
            hold_ms = int(unknown_hold_seconds * 1000.0)
        else:
            hold_ms = int(rule.warn_hold_seconds * 1000.0)

        assert self.candidate_since_ms is not None
        if timestamp_ms - self.candidate_since_ms >= hold_ms:
            self.current = raw_state
            self.candidate = None
            self.candidate_since_ms = None
        return self.current


class PostureAnalyzer:
    """Stateful posture assessment independent of camera/UI code."""

    def __init__(
        self,
        config: PostureConfig | None = None,
        baseline: PostureBaseline | None = None,
    ) -> None:
        self.config = config or PostureConfig()
        self.baseline = baseline
        self._calibration: CalibrationSession | None = None
        self._smoothed: dict[str, float] = {}
        self._states: dict[str, _TemporalState] = {
            key: _TemporalState() for key in self.config.metrics
        }
        self._foot_states = {side: _TemporalState() for side in ("left", "right")}
        self._foot_histories: dict[
            str, deque[tuple[int, tuple[float, float, float]]]
        ] = {side: deque() for side in ("left", "right")}
        self._baseline_updated = False
        self._paused = False
        self._last_feedback_ms: int | None = None
        self._last_feedback_key: str | None = None
        self._last_feedback_for_key_ms: int | None = None

    @property
    def calibration_sample_counts(self):
        if self._calibration is None:
            return None
        return {k: len(v) for k, v in self._calibration.samples.items()}

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Suspend feedback during a mouthful, conversation or care activity."""
        if paused == self._paused:
            return
        self._paused = paused
        if paused:
            self.cancel_calibration()
        # Do not carry stale posture states from before the pause.
        self._reset_temporal_state()
        self._last_feedback_key = None
        self._last_feedback_for_key_ms = None

    def set_baseline(self, baseline: PostureBaseline | None) -> None:
        self.baseline = baseline
        self._reset_temporal_state()
        self._last_feedback_key = None
        self._last_feedback_for_key_ms = None

    def start_calibration(self, timestamp_ms: int) -> None:
        enabled_keys = {
            key for key, rule in self.config.metrics.items() if rule.enabled
        }
        if self.config.feet.enabled:
            for side in ("left", "right"):
                enabled_keys.update(foot_metric_keys(side))
        self._reset_temporal_state()
        self._calibration = CalibrationSession(
            started_at_ms=timestamp_ms,
            duration_seconds=self.config.calibration_seconds,
            min_samples=self.config.min_calibration_samples,
            metric_keys=enabled_keys,
        )
        self._baseline_updated = False
        self._last_feedback_key = None
        self._last_feedback_for_key_ms = None

    def cancel_calibration(self) -> None:
        self._calibration = None

    def take_baseline_update(self) -> PostureBaseline | None:
        if not self._baseline_updated:
            return None
        self._baseline_updated = False
        return self.baseline

    def update(self, frame: PoseFrame | None, timestamp_ms: int, *, observation: PostureObservation | None = None) -> PostureAssessment:
        if self._paused:
            return PostureAssessment(
                timestamp_ms=timestamp_ms,
                overall_status=AssessmentStatus.PAUSED,
                calibrated=self.baseline is not None,
                messages=("通知を一時停止中です。Pキーで再開できます",),
            )

        if observation is None:
            observation = compute_posture_observation(
                frame, self.config.visibility_threshold, self.config.feet.visibility_threshold
            )

        if self._calibration is not None:
            self._calibration.add(observation)
            progress = self._calibration.progress(timestamp_ms)
            if self._calibration.ready(timestamp_ms):
                self.baseline = self._calibration.finish()
                self._calibration = None
                self._baseline_updated = True
                self._reset_temporal_state()
                return PostureAssessment(
                    timestamp_ms=timestamp_ms,
                    overall_status=AssessmentStatus.OK,
                    calibrated=True,
                    messages=("基準姿勢を登録しました",),
                    calibration_progress=1.0,
                )
            return PostureAssessment(
                timestamp_ms=timestamp_ms,
                overall_status=AssessmentStatus.CALIBRATING,
                calibrated=self.baseline is not None,
                messages=(self._calibration_message(),),
                calibration_progress=progress,
            )

        if self.baseline is None:
            return PostureAssessment(
                timestamp_ms=timestamp_ms,
                overall_status=AssessmentStatus.NOT_CALIBRATED,
                calibrated=False,
                messages=("Cキーを押して基準姿勢を登録してください",),
            )

        assessments: dict[str, MetricAssessment] = {}
        for key, rule in self.config.metrics.items():
            if not rule.enabled:
                continue
            observed = observation.metrics.get(key)
            baseline_value = self.baseline.metrics.get(key)
            if observed is None or baseline_value is None:
                self._smoothed.pop(key, None)
                self._states[key] = _TemporalState()
                assessments[key] = self._make_metric_assessment(
                    key=key,
                    # Never give advice based only on a previously visible frame.
                    stable_state="unknown",
                    value=None,
                    baseline=baseline_value,
                    deviation=None,
                    confidence=0.0,
                    rule=rule,
                )
                continue

            value = self._smooth(key, observed.value)
            deviation = value - baseline_value
            if deviation < rule.lower_delta:
                raw_state = "below"
            elif deviation > rule.upper_delta:
                raw_state = "above"
            else:
                raw_state = "ok"

            stable_state = self._states[key].update(
                raw_state, timestamp_ms, rule, self.config.unknown_hold_seconds
            )
            assessments[key] = self._make_metric_assessment(
                key=key,
                stable_state=stable_state,
                value=value,
                baseline=baseline_value,
                deviation=deviation,
                confidence=observed.confidence,
                rule=rule,
            )

        feet = self._assess_feet(observation, timestamp_ms)
        advice_messages = [
            assessment.message
            for assessment in assessments.values()
            if assessment.status is AssessmentStatus.ADVICE
            and assessment.message is not None
        ]
        advice_messages.extend(
            assessment.message
            for assessment in feet.values()
            if assessment.status is AssessmentStatus.ADVICE
            and assessment.message is not None
        )
        if advice_messages:
            overall = AssessmentStatus.ADVICE
            messages = self._deduplicate(advice_messages)
        else:
            metric_unknown = any(
                item.status is AssessmentStatus.UNKNOWN
                for item in assessments.values()
            )
            foot_unknown = any(
                item.status is AssessmentStatus.UNKNOWN for item in feet.values()
            )
            if metric_unknown or foot_unknown:
                overall = AssessmentStatus.UNKNOWN
                unknown_messages: list[str] = []
                if metric_unknown:
                    unknown_messages.append(
                        "顔・肩・腰が見える位置に座ってください"
                    )
                unknown_messages.extend(
                    item.message
                    for item in feet.values()
                    if item.status is AssessmentStatus.UNKNOWN
                    and item.message is not None
                )
                messages = self._deduplicate(unknown_messages)
            else:
                overall = AssessmentStatus.OK
                messages = ("基準姿勢を保てています",)

        feedback_messages = self._select_feedback(assessments, feet, timestamp_ms)

        return PostureAssessment(
            timestamp_ms=timestamp_ms,
            overall_status=overall,
            calibrated=True,
            metrics=assessments,
            feet=feet,
            messages=messages,
            feedback_messages=feedback_messages,
        )

    def _select_feedback(
        self,
        assessments: dict[str, MetricAssessment],
        feet: dict[str, FootAssessment],
        timestamp_ms: int,
    ) -> tuple[str, ...]:
        """Emit at most one low-frequency cue, independent of the UI status."""
        candidates: list[tuple[str, str]] = []
        # Prioritize sustained trunk/neck deviations over the optional feet proxy.
        for key in ("torso_forward_deg", "head_forward_ratio", "head_pitch_deg"):
            item = assessments.get(key)
            if item is not None and item.status is AssessmentStatus.ADVICE and item.message:
                candidates.append((key, item.message))
        for key, item in assessments.items():
            if key not in {"torso_forward_deg", "head_forward_ratio", "head_pitch_deg"}:
                if item.status is AssessmentStatus.ADVICE and item.message:
                    candidates.append((key, item.message))
        for side in ("left", "right"):
            item = feet.get(side)
            if item is not None and item.status is AssessmentStatus.ADVICE and item.message:
                candidates.append((f"foot_{side}", item.message))

        if not candidates:
            self._last_feedback_key = None
            self._last_feedback_for_key_ms = None
            return ()

        key, message = candidates[0]
        since_global = (
            float("inf") if self._last_feedback_ms is None
            else timestamp_ms - self._last_feedback_ms
        )
        since_key = (
            float("inf") if self._last_feedback_key != key
            or self._last_feedback_for_key_ms is None
            else timestamp_ms - self._last_feedback_for_key_ms
        )
        if (
            since_global < self.config.feedback_min_interval_seconds * 1000
            or since_key < self.config.feedback_repeat_seconds * 1000
        ):
            return ()
        self._last_feedback_ms = timestamp_ms
        self._last_feedback_key = key
        self._last_feedback_for_key_ms = timestamp_ms
        return (message,)

    def _assess_feet(
        self,
        observation: PostureObservation,
        timestamp_ms: int,
    ) -> dict[str, FootAssessment]:
        rule = self.config.feet
        if not rule.enabled:
            return {}

        results: dict[str, FootAssessment] = {}
        assert self.baseline is not None
        for side in ("left", "right"):
            keys = foot_metric_keys(side)
            baseline_values = [self.baseline.metrics.get(key) for key in keys]
            if any(value is None for value in baseline_values):
                self._foot_histories[side].clear()
                for key in keys:
                    self._smoothed.pop(key, None)
                results[side] = FootAssessment(
                    side=side,
                    status=AssessmentStatus.UNKNOWN,
                    position_deviation_ratio=None,
                    motion_rms_ratio=None,
                    pitch_deviation_deg=None,
                    confidence=0.0,
                    message="Cキーで足を含む基準姿勢を再登録してください",
                )
                continue

            observed_values = [observation.metrics.get(key) for key in keys]
            if any(value is None for value in observed_values):
                self._foot_histories[side].clear()
                for key in keys:
                    self._smoothed.pop(key, None)
                self._foot_states[side].update(
                    "unknown", timestamp_ms, rule, self.config.unknown_hold_seconds
                )
                results[side] = self._make_foot_assessment(
                    side=side,
                    stable_state="unknown",
                    position_deviation=None,
                    motion_rms=None,
                    pitch_deviation=None,
                    confidence=0.0,
                    reasons=(),
                    unknown_message="両足が画面に入る位置へ移動してください",
                )
                continue

            present = [value for value in observed_values if value is not None]
            current = [
                self._smooth(key, metric.value)
                for key, metric in zip(keys, present, strict=True)
            ]
            baseline = [float(value) for value in baseline_values if value is not None]
            position = (current[0], current[1], current[2])
            position_deviation = self._weighted_distance(
                position,
                (baseline[0], baseline[1], baseline[2]),
                rule.depth_weight,
            )
            pitch_deviation = abs(self._angle_delta(current[3], baseline[3]))
            confidence = min(metric.confidence for metric in present)

            history = self._foot_histories[side]
            history.append((timestamp_ms, position))
            window_ms = int(rule.motion_window_seconds * 1000.0)
            while history and timestamp_ms - history[0][0] > window_ms:
                history.popleft()
            motion_rms = self._motion_rms(
                history,
                window_ms,
                rule.min_window_samples,
                rule.depth_weight,
            )

            if motion_rms is None:
                raw_state = "unknown"
                reasons: tuple[str, ...] = ()
                unknown_message = "足の動きを観測中です"
            else:
                position_bad = position_deviation > rule.position_threshold_ratio
                motion_bad = motion_rms > rule.motion_threshold_ratio
                pitch_bad = pitch_deviation > rule.pitch_threshold_deg
                reason_list: list[str] = []
                if motion_bad:
                    reason_list.append("motion")
                if rule.require_position_and_pitch:
                    if position_bad and pitch_bad:
                        reason_list.extend(("position", "pitch"))
                else:
                    if position_bad:
                        reason_list.append("position")
                    if pitch_bad:
                        reason_list.append("pitch")
                reasons = tuple(reason_list)
                raw_state = "advice" if reasons else "ok"
                unknown_message = None

            stable_state = self._foot_states[side].update(
                raw_state, timestamp_ms, rule, self.config.unknown_hold_seconds
            )
            results[side] = self._make_foot_assessment(
                side=side,
                stable_state=stable_state,
                position_deviation=position_deviation,
                motion_rms=motion_rms,
                pitch_deviation=pitch_deviation,
                confidence=confidence,
                reasons=reasons,
                unknown_message=unknown_message,
            )
        return results

    def _smooth(self, key: str, value: float) -> float:
        previous = self._smoothed.get(key)
        if previous is None:
            smoothed = value
        else:
            alpha = self.config.smoothing_alpha
            smoothed = alpha * value + (1.0 - alpha) * previous
        self._smoothed[key] = smoothed
        return smoothed

    def _reset_temporal_state(self) -> None:
        self._smoothed.clear()
        self._states = {key: _TemporalState() for key in self.config.metrics}
        self._foot_states = {side: _TemporalState() for side in ("left", "right")}
        for history in self._foot_histories.values():
            history.clear()

    def _calibration_message(self) -> str:
        if self.config.feet.enabled:
            return "顔から足先まで映し、両足を置いた基準姿勢を保ってください"
        return "無理なく保てる食事時の基準姿勢を取ってください"

    @staticmethod
    def _angle_delta(value: float, baseline: float) -> float:
        return (value - baseline + 180.0) % 360.0 - 180.0

    @staticmethod
    def _motion_rms(
        history: deque[tuple[int, tuple[float, float, float]]],
        window_ms: int,
        min_samples: int,
        depth_weight: float,
    ) -> float | None:
        if len(history) < min_samples:
            return None
        if history[-1][0] - history[0][0] < int(window_ms * 0.6):
            return None

        means = tuple(
            sum(position[axis] for _, position in history) / len(history)
            for axis in range(3)
        )
        squared_distances = []
        for _, position in history:
            dx = position[0] - means[0]
            dy = position[1] - means[1]
            dz = position[2] - means[2]
            squared_distances.append(dx * dx + dy * dy + depth_weight * dz * dz)
        return math.sqrt(sum(squared_distances) / len(squared_distances))

    @staticmethod
    def _weighted_distance(
        value: tuple[float, float, float],
        baseline: tuple[float, float, float],
        depth_weight: float,
    ) -> float:
        dx = value[0] - baseline[0]
        dy = value[1] - baseline[1]
        dz = value[2] - baseline[2]
        return math.sqrt(dx * dx + dy * dy + depth_weight * dz * dz)

    def _make_foot_assessment(
        self,
        *,
        side: str,
        stable_state: str,
        position_deviation: float | None,
        motion_rms: float | None,
        pitch_deviation: float | None,
        confidence: float,
        reasons: tuple[str, ...],
        unknown_message: str | None,
    ) -> FootAssessment:
        if stable_state == "ok":
            status = AssessmentStatus.OK
            message = None
        elif stable_state == "advice":
            status = AssessmentStatus.ADVICE
            message = (
                self.config.feet.message_left
                if side == "left"
                else self.config.feet.message_right
            )
        else:
            status = AssessmentStatus.UNKNOWN
            message = unknown_message or "足を判定できません"
        return FootAssessment(
            side=side,
            status=status,
            position_deviation_ratio=position_deviation,
            motion_rms_ratio=motion_rms,
            pitch_deviation_deg=pitch_deviation,
            confidence=confidence,
            reasons=reasons,
            message=message,
        )

    @staticmethod
    def _deduplicate(messages: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(messages))

    @staticmethod
    def _make_metric_assessment(
        *,
        key: str,
        stable_state: str,
        value: float | None,
        baseline: float | None,
        deviation: float | None,
        confidence: float,
        rule: MetricRule,
    ) -> MetricAssessment:
        if stable_state == "unknown":
            status = AssessmentStatus.UNKNOWN
            message = None
            direction = None
        elif stable_state == "ok":
            status = AssessmentStatus.OK
            message = None
            direction = None
        elif stable_state == "below":
            status = AssessmentStatus.ADVICE
            message = rule.message_below
            direction = "below"
        else:
            status = AssessmentStatus.ADVICE
            message = rule.message_above
            direction = "above"

        return MetricAssessment(
            key=key,
            status=status,
            value=value,
            baseline=baseline,
            deviation=deviation,
            lower_delta=rule.lower_delta,
            upper_delta=rule.upper_delta,
            confidence=confidence,
            direction=direction,
            message=message,
        )
