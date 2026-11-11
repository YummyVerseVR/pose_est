"""Core behavior tests that do not require a webcam or MediaPipe installation."""

import unittest
from unittest.mock import patch

from pose_est.analyzer import PostureAnalyzer
from pose_est.baseline import PostureBaseline
from pose_est.config import FootRule, MetricRule, PostureConfig
from pose_est.geometry import compute_posture_observation
from pose_est.models import (
    AssessmentStatus,
    ObservedMetric,
    Point3D,
    PoseFrame,
    PostureObservation,
)


def torso_rule() -> MetricRule:
    return MetricRule(
        enabled=True,
        lower_delta=-8,
        upper_delta=10,
        warn_hold_seconds=2.5,
        recover_hold_seconds=1,
        message_below="below",
        message_above="above",
    )


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        config = PostureConfig(
            metrics={"torso_forward_deg": torso_rule()},
            feet=FootRule(enabled=False),
        )
        self.analyzer = PostureAnalyzer(
            config, baseline=PostureBaseline.create({"torso_forward_deg": 0})
        )

    def update(self, timestamp_ms: int, value: float | None):
        observations = (
            {} if value is None else {"torso_forward_deg": ObservedMetric(value, 1)}
        )
        with patch(
            "pose_est.analyzer.compute_posture_observation",
            return_value=PostureObservation(metrics=observations),
        ):
            return self.analyzer.update(None, timestamp_ms)

    def test_feedback_emitted_once_then_repeated_after_long_interval(self):
        first = self.update(0, 20)
        self.assertEqual(first.overall_status, AssessmentStatus.UNKNOWN)
        self.assertFalse(first.needs_feedback)
        warning = self.update(3000, 20)
        self.assertTrue(warning.needs_feedback)
        self.assertEqual(warning.feedback_messages, ("above",))
        self.assertEqual(warning.to_dict()["feedback_messages"], ["above"])
        self.assertFalse(self.update(5000, 20).needs_feedback)
        self.assertTrue(self.update(40000, 20).needs_feedback)

    def test_missing_landmarks_never_show_stale_advice(self):
        self.update(0, 20)
        self.update(3000, 20)
        missed = self.update(3100, None)
        self.assertEqual(missed.overall_status, AssessmentStatus.UNKNOWN)
        self.assertEqual(missed.metrics["torso_forward_deg"].status, AssessmentStatus.UNKNOWN)
        self.assertFalse(missed.needs_feedback)

    def test_pause_suppresses_analysis_and_resets_temporal_state(self):
        self.update(0, 20)
        self.update(3000, 20)
        self.analyzer.set_paused(True)
        paused = self.update(4000, 20)
        self.assertEqual(paused.overall_status, AssessmentStatus.PAUSED)
        self.assertFalse(paused.needs_feedback)
        self.analyzer.set_paused(False)
        restarted = self.update(5000, 20)
        self.assertEqual(restarted.overall_status, AssessmentStatus.UNKNOWN)
        self.assertFalse(restarted.needs_feedback)

    def test_camera_ready_default_does_not_require_feet(self):
        self.assertFalse(PostureConfig().feet.enabled)

    def test_invalid_notification_intervals_rejected(self):
        with self.assertRaises(ValueError):
            PostureConfig(feedback_repeat_seconds=0)


class PartialVisibilityTests(unittest.TestCase):
    def test_head_pitch_still_available_when_hips_hidden(self):
        hidden = Point3D(0, 0, 0, visibility=0, presence=0)
        landmarks = [hidden for _ in range(33)]
        landmarks[0] = Point3D(0, -0.4, -0.1)
        landmarks[7] = Point3D(-0.1, -0.2, 0)
        landmarks[8] = Point3D(0.1, -0.2, 0)
        frame = PoseFrame(tuple(landmarks), tuple(landmarks))
        observation = compute_posture_observation(frame, 0.55, 0.35)
        self.assertIn("head_pitch_deg", observation.metrics)
        self.assertNotIn("torso_forward_deg", observation.metrics)


if __name__ == "__main__":
    unittest.main()
