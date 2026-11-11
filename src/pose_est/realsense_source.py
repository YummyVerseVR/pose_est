"""Optional SDK adapter: no pyrealsense2 import until start()."""
from .camera_source import CameraPacket, MonotonicClock


class RealSenseFrameSource:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.clock = MonotonicClock()

    def start(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("RealSense mode requires pyrealsense2: uv sync --extra realsense") from exc
        self.rs = rs
        devices = list(rs.context().query_devices())
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        serial = self.config.sensor_serial
        if not devices or (serial and serial not in serials):
            raise RuntimeError("RealSense device not connected (check USB 3.x / serial)")
        if len(devices) > 1 and not serial:
            raise RuntimeError("Multiple RealSense devices: set sensor_serial explicitly")
        serial = serial or serials[0]
        settings = rs.config()
        settings.enable_device(serial)
        w, h, fps = self.config.rgbd_width, self.config.rgbd_height, self.config.rgbd_fps
        settings.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        settings.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(settings)
            self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            self.metadata = {
                "measurement_source": "realsense_depth", "sensor_serial": serial,
                "resolution": [w, h], "fps": fps,
                "camera_side": self.config.camera_side,
                "placement_id": self.config.placement_id,
                "azimuth_deg": self.config.camera_azimuth_deg,
                "distance_m": self.config.camera_distance_m,
                "height_m": self.config.camera_height_m,
            }
        except BaseException:
            self.stop()
            raise

    def read(self):
        import numpy as np
        try:
            frames = self.align.process(self.pipeline.wait_for_frames(self.config.frame_timeout_ms))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("missing synchronized color/depth frame")
            intrinsics = depth.profile.as_video_stream_profile().intrinsics
            return CameraPacket(
                np.asanyarray(color.get_data()).copy(), self.clock.now(),
                np.asanyarray(depth.get_data()).copy(), self.scale, intrinsics,
                dict(self.metadata), self.rs.rs2_deproject_pixel_to_point,
            )
        except Exception as exc:
            raise RuntimeError(f"RealSense frame acquisition failed (USB/timeout): {exc}") from exc

    def stop(self):
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass  # start may have failed before a stream existed
