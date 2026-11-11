import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace

import numpy as np
from pose_est.app import CameraPostureApp, _mirror_pose_for_display, main
from pose_est.analyzer import PostureAnalyzer
from pose_est.baseline import PostureBaseline, validate_baseline, rgbd_context
from pose_est.camera_source import CameraPacket, MonotonicClock, WebcamFrameSource
from pose_est.config import PostureConfig
from pose_est.depth_geometry import sample_landmark
from pose_est.models import Point3D, PoseFrame, AssessmentStatus
from pose_est.realsense_source import RealSenseFrameSource
from pose_est.rgbd_geometry import RGBDGeometry


def packet():
    return CameraPacket(np.zeros((100,100,3), dtype=np.uint8), 0,
                        np.full((100,100), 1000, dtype=float), .001,
                        NS(width=100,height=100,fx=100,fy=100,ppx=50,ppy=50),
                        {"measurement_source":"realsense_depth","sensor_serial":"synthetic"})


def pose():
    hidden = Point3D(0,0,0,0,0)
    points = [hidden]*33
    # Anatomical left visible. Face +X, upright trunk; all depth=1 m.
    for i,x,y in [(0,.70,.20),(7,.60,.20),(11,.55,.40),(23,.55,.80)]:
        points[i] = Point3D(x,y,0)
    return PoseFrame(tuple(points), tuple([Point3D(999,999,999)]*33))


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.p = packet()
        self.c = PostureConfig()
        self.l = Point3D(.7,.2,0)

    def test_units_axes_and_sdk_deprojection(self):
        result = sample_landmark(self.l,self.p,self.c)
        self.assertAlmostEqual(result.point.x,.2)
        self.assertAlmostEqual(result.point.y,-.3)
        self.assertEqual(result.point.z,1)
        fn = MagicMock(return_value=[.2,-.3,1])
        result = sample_landmark(self.l,replace(self.p,deproject=fn),self.c)
        fn.assert_called_once_with(self.p.intrinsics,[70,20],1)
        self.assertEqual(result.point.z,1)

    def test_invalid_landmarks_and_depth(self):
        for landmark in [Point3D(-.1,.2,0), Point3D(1,.2,0), Point3D(float('nan'),.2,0),
                         Point3D(float('inf'),.2,0), Point3D(.5,.5,0,.1,1),
                         Point3D(.5,.5,0,1,float('nan'))]:
            with self.subTest(landmark=landmark):
                self.assertIsNone(sample_landmark(landmark,self.p,self.c).point)
        for value in [0,-1,float('inf'),float('nan'),999999]:
            self.p.depth[:] = value
            self.assertIsNone(sample_landmark(self.l,self.p,self.c).point)
        self.assertIsNone(sample_landmark(self.l,replace(self.p,depth=None),self.c).point)
        self.assertIsNone(sample_landmark(self.l,replace(self.p,depth_scale=-1),self.c).point)

    def test_median_outliers_low_valid_rate_and_mixed_background(self):
        self.p.depth[20,70]=2000
        self.assertEqual(sample_landmark(self.l,self.p,self.c).point.z,1)
        self.p.depth[18:23,68:73] = 0
        self.p.depth[20,70]=1000
        self.assertIsNone(sample_landmark(self.l,self.p,self.c).point)
        patch = self.p.depth[18:23,68:73]
        patch[:]=1000
        patch[:2]=2000  # 40% background; MAD alone would miss this
        self.assertEqual(sample_landmark(self.l,self.p,self.c).reason,'depth_spread')
        patch[:]=np.arange(25).reshape(5,5)*100+300
        self.assertIsNone(sample_landmark(self.l,self.p,self.c).point)

    def test_quality_settings_validation(self):
        for settings in [dict(depth_patch_size=4),dict(depth_min_m=4),dict(depth_min_confidence=float('nan')),
                         dict(depth_max_mad_m=0),dict(rgbd_fps=0)]:
            with self.assertRaises(ValueError): PostureConfig(**settings)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.c = PostureConfig(camera_source='realsense')
        self.g = RGBDGeometry(self.c)
        self.p,self.f = packet(),pose()
        self.initial=self.g.observe(self.f,self.p,calibrating=True)

    def test_depth_actually_changes_geometry_and_assessment(self):
        self.assertAlmostEqual(self.initial.metrics['torso_forward_deg'].value,0)
        self.p.depth[38:43,53:58]=1500
        changed=self.g.observe(self.f,self.p)
        self.assertNotEqual(changed.metrics['torso_forward_deg'].value,0)
        self.assertNotEqual(changed.metrics['head_forward_ratio'].value,self.initial.metrics['head_forward_ratio'].value)
        c=replace(self.c,smoothing_alpha=1,metrics={**self.c.metrics, "torso_forward_deg": replace(self.c.metrics["torso_forward_deg"],upper_delta=1)})
        analyzer=PostureAnalyzer(c,PostureBaseline.create({k:v.value for k,v in self.initial.metrics.items()}))
        analyzer.update(None,0,observation=changed)
        result=analyzer.update(None,3000,observation=changed)
        self.assertEqual(result.metrics['torso_forward_deg'].status,AssessmentStatus.ADVICE)
        self.assertTrue(result.needs_feedback)
        # After a missing-depth frame, reappearance must earn a new warning hold.
        self.p.depth[38:43,53:58]=0
        missing=self.g.observe(self.f,self.p)
        self.assertIn('head_pitch_deg',missing.metrics)
        self.assertNotIn('torso_forward_deg',missing.metrics)
        lost=analyzer.update(None,3100,observation=missing)
        self.assertFalse(lost.needs_feedback)
        self.p.depth[38:43,53:58]=1500
        resumed=analyzer.update(None,3200,observation=self.g.observe(self.f,self.p))
        self.assertEqual(resumed.metrics['torso_forward_deg'].status,AssessmentStatus.UNKNOWN)

    def test_face_turn_does_not_change_torso_and_forward_axis_is_fixed(self):
        points=list(self.f.image_landmarks)
        points[0]=Point3D(.6,.2,0)
        self.p.depth[18:23,58:63]=1100
        turned=self.g.observe(PoseFrame(tuple(points),()),self.p)
        self.assertEqual(turned.metrics['torso_forward_deg'],self.initial.metrics['torso_forward_deg'])
        self.assertEqual(self.g.reference['forward'],[1,0,0])
        # Nose occlusion does not stop torso/head-forward measurements.
        points[0]=Point3D(0,0,0,0,0)
        partial=self.g.observe(PoseFrame(tuple(points),()),self.p)
        self.assertIn('torso_forward_deg',partial.metrics)
        self.assertNotIn('head_pitch_deg',partial.metrics)

    def test_pitch_sign_translation_and_mirror_display_only(self):
        points=list(self.f.image_landmarks)
        points[0]=Point3D(.7,.15,0)
        upward=self.g.observe(PoseFrame(tuple(points),()),self.p)
        self.assertGreater(upward.metrics['head_pitch_deg'].value,0)
        shifted=tuple(Point3D(p.x+.05,p.y,p.z,p.visibility,p.presence) for p in self.f.image_landmarks)
        translation=self.g.observe(PoseFrame(shifted,()),self.p)
        self.assertAlmostEqual(translation.metrics['head_forward_ratio'].value,self.initial.metrics['head_forward_ratio'].value)
        mirrored=_mirror_pose_for_display(self.f)
        self.assertEqual(mirrored.world_landmarks,self.f.world_landmarks)
        self.assertAlmostEqual(mirrored.image_landmarks[0].x,.3)

    def test_selects_observable_chain_when_both_ears_visible(self):
        points=list(self.f.image_landmarks)
        points[8]=Point3D(.5,.2,0)
        g=RGBDGeometry(self.c)
        observation=g.observe(PoseFrame(tuple(points),()),self.p,calibrating=True)
        self.assertEqual(g.reference['indices'],[7,11,23])
        self.assertIn('torso_forward_deg',observation.metrics)

    def test_hidden_hip_cannot_finish_full_calibration(self):
        self.p.depth[78:83,53:58]=0
        observation=self.g.observe(self.f,self.p,calibrating=True)
        a=PostureAnalyzer(replace(self.c,calibration_seconds=.1,min_calibration_samples=1))
        a.start_calibration(0)
        a.update(None,0,observation=observation)
        result=a.update(None,200,observation=observation)
        self.assertEqual(result.overall_status,AssessmentStatus.CALIBRATING)
        self.assertEqual(a.calibration_sample_counts['torso_forward_deg'],0)
        self.assertIsNone(a.take_baseline_update())


class BaselineTests(unittest.TestCase):
    def test_old_webcam_round_trip_and_sensor_setup_binding(self):
        old=PostureBaseline.create({'head_pitch_deg':0})
        validate_baseline(old,'mediapipe_world')
        with self.assertRaises(ValueError): validate_baseline(old,'realsense_depth',{})
        c=PostureConfig(camera_source='realsense')
        p=packet(); g=RGBDGeometry(c);g.observe(pose(),p,calibrating=True)
        ctx=rgbd_context(c,p)
        new=replace(old,version=2,metadata={'measurement_source':'realsense_depth','context':ctx,'reference':g.reference})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'baseline.json'
            new.save(path)
            loaded=PostureBaseline.load(path)
        validate_baseline(loaded,'realsense_depth',ctx)
        for context in [rgbd_context(replace(c,placement_id='moved'),p),
                        rgbd_context(c,replace(p,metadata={'sensor_serial':'different'})),
                        rgbd_context(replace(c,depth_min_m=.3),p)]:
            with self.assertRaises(ValueError):validate_baseline(loaded,'realsense_depth',context)
        with self.assertRaises(ValueError):validate_baseline(loaded,'mediapipe_world')


class SourceTests(unittest.TestCase):
    def test_clock_strictly_increases(self):
        with patch('pose_est.camera_source.time.monotonic_ns',return_value=1000000):
            c=MonotonicClock();self.assertEqual([c.now() for _ in range(3)],[1,2,3])

    def test_webcam_release_and_no_sdk_import(self):
        cv=MagicMock();cv.VideoCapture.return_value.isOpened.return_value=True
        cv.VideoCapture.return_value.read.return_value=(True,packet().color_bgr)
        with patch.dict(sys.modules,{'cv2':cv,'pyrealsense2':None}):
            source=WebcamFrameSource(PostureConfig());source.start()
            self.assertIsNone(source.read().depth);source.stop();source.stop()
            cv.VideoCapture.return_value.release.assert_called_once()
            with self.assertRaisesRegex(RuntimeError,'requires pyrealsense2'):
                RealSenseFrameSource(PostureConfig()).start()

    def test_sdk_device_selection_frames_timeout_and_cleanup(self):
        sdk=MagicMock()
        device=MagicMock();device.get_info.return_value='synthetic'
        sdk.context.return_value.query_devices.return_value=[device]
        pipe=sdk.pipeline.return_value
        pipe.start.return_value.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value=.001
        frames=sdk.align.return_value.process.return_value
        frames.get_color_frame.return_value.get_data.return_value=np.zeros((100,100,3),dtype=np.uint8)
        frames.get_depth_frame.return_value.get_data.return_value=np.ones((100,100),dtype=np.uint16)
        with patch.dict(sys.modules,{'pyrealsense2':sdk}):
            s=RealSenseFrameSource(PostureConfig());s.start();p=s.read()
            self.assertEqual(p.depth_scale,.001)
            self.assertEqual(p.depth.shape,(100,100))
            sdk.align.assert_called_once_with(sdk.stream.color)
            pipe.wait_for_frames.assert_called_once_with(3000)
            pipe.wait_for_frames.side_effect=RuntimeError('disconnected')
            with self.assertRaisesRegex(RuntimeError,'USB/timeout'):s.read()
            s.stop();s.stop();pipe.stop.assert_called_once()
            pipe.wait_for_frames.side_effect=None
            frames.get_depth_frame.return_value=None
            s=RealSenseFrameSource(PostureConfig());s.start()
            with self.assertRaisesRegex(RuntimeError,'missing synchronized'):s.read()
            s.stop()
            sdk.context.return_value.query_devices.return_value=[device,device]
            with self.assertRaisesRegex(RuntimeError,'Multiple'):RealSenseFrameSource(PostureConfig()).start()
            sdk.context.return_value.query_devices.return_value=[]
            with self.assertRaisesRegex(RuntimeError,'not connected'):RealSenseFrameSource(PostureConfig()).start()
            sdk.context.return_value.query_devices.return_value=[device]
            pipe.start.side_effect=RuntimeError('start failed')
            pipe.stop.reset_mock()
            with self.assertRaisesRegex(RuntimeError,'start failed'):RealSenseFrameSource(PostureConfig()).start()
            pipe.stop.assert_called_once()

    def test_app_releases_on_acquisition_failure_and_ctrl_c(self):
        for exc in [RuntimeError('USB removed'),KeyboardInterrupt()]:
            source=MagicMock();source.read.side_effect=exc
            with tempfile.TemporaryDirectory() as folder, patch('cv2.destroyAllWindows') as destroy, \
                 patch('pose_est.app.WebcamFrameSource',return_value=source), \
                 patch('pose_est.app.OverlayRenderer'),patch('pose_est.app.MediaPipePoseEstimator'):
                app=CameraPostureApp(PostureConfig(),baseline_path=Path(folder)/'none.json')
                with self.assertRaises(type(exc)):app.run()
                source.stop.assert_called_once();destroy.assert_called_once()

    def test_app_rgbd_calibration_pause_log_and_cleanup(self):
        source=MagicMock()
        p=packet()
        source.read.side_effect=[replace(p,timestamp_ms=t) for t in (0,100,200,300)]
        with tempfile.TemporaryDirectory() as folder:
            baseline_path=Path(folder)/'calibration.json'
            log_path=Path(folder)/'log.jsonl'
            c=PostureConfig(camera_source='realsense',calibration_seconds=.001,
                            min_calibration_samples=1,research_log_path=str(log_path))
            with patch('pose_est.realsense_source.RealSenseFrameSource',return_value=source), \
                 patch('pose_est.app.OverlayRenderer'),patch('pose_est.app.MediaPipePoseEstimator') as estimator, \
                 patch('cv2.imshow'),patch('cv2.destroyAllWindows'), \
                 patch('cv2.waitKey',side_effect=[ord('c'),-1,ord('p'),ord('q')]):
                estimator.return_value.__enter__.return_value.process.return_value=pose()
                results=[]
                CameraPostureApp(c,baseline_path=baseline_path).run(results.append)
            saved=PostureBaseline.load(baseline_path)
            validate_baseline(saved,'realsense_depth',rgbd_context(c,p))
            self.assertEqual(results[-1].overall_status,AssessmentStatus.PAUSED)
            self.assertEqual(saved.version,2)
            self.assertEqual(set(saved.metrics),set(c.metrics))
            records=[json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(records),4)
            self.assertNotIn('synthetic',log_path.read_text())
            self.assertTrue(all(r['measurement_source']=='realsense_depth' for r in records))
            source.stop.assert_called_once()

    def test_cli_without_sdk_and_source_switch(self):
        script="import sys; sys.modules['pyrealsense2']=None; from pose_est.app import main; main(['--help'])"
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--camera-source',result.stdout)
        with patch('pose_est.app.CameraPostureApp') as app:
            self.assertEqual(main(['--camera-source','realsense']),0)
            self.assertEqual(app.call_args.args[0].camera_source,'realsense')


if __name__ == '__main__':unittest.main()
