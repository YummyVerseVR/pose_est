"""Response/transient acceptance checks for the optional faster UX profile."""
from pathlib import Path
import unittest

from pose_est.analyzer import PostureAnalyzer
from pose_est.baseline import PostureBaseline
from pose_est.config import PostureConfig
from pose_est.models import AssessmentStatus, ObservedMetric, PostureObservation

ROOT = Path(__file__).resolve().parents[1]


def first_advice(config, excursion_ms):
    analyzer = PostureAnalyzer(config, PostureBaseline.create({k: 0.0 for k in config.metrics}))
    for timestamp in range(0, 7000, 33):
        torso = 20.0 if 2000 <= timestamp < 2000 + excursion_ms else 0.0
        observation = PostureObservation({
            key: ObservedMetric(torso if key == 'torso_forward_deg' else 0.0, 1.0)
            for key in config.metrics
        }, measurement_source='realsense_depth')
        result = analyzer.update(None, timestamp, observation=observation)
        if result.metrics['torso_forward_deg'].status is AssessmentStatus.ADVICE:
            return timestamp - 2000
    return None


class ResponsiveProfileTests(unittest.TestCase):
    def setUp(self):
        self.normal = PostureConfig.load(ROOT / 'config.realsense.example.json')
        self.fast = PostureConfig.load(ROOT / 'config.realsense-responsive.example.json')

    def test_sustained_deviation_gets_earlier_advice_without_notification_spam(self):
        normal_latency = first_advice(self.normal, 4000)
        fast_latency = first_advice(self.fast, 4000)
        self.assertIsNotNone(normal_latency)
        self.assertIsNotNone(fast_latency)
        self.assertLess(fast_latency, normal_latency)
        self.assertLess(fast_latency, 1800)
        self.assertEqual(self.fast.feedback_min_interval_seconds, self.normal.feedback_min_interval_seconds)
        self.assertEqual(self.fast.feedback_repeat_seconds, self.normal.feedback_repeat_seconds)

    def test_brief_reach_does_not_raise_advice(self):
        self.assertIsNone(first_advice(self.fast, 400))


if __name__ == '__main__':
    unittest.main()
