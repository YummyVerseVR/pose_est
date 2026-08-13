from __future__ import annotations

from pathlib import Path
import os

from .models import AssessmentStatus, PoseFrame, PostureAssessment


_METRIC_LABELS = {
    "head_pitch_deg": "頭部ピッチ",
    "torso_forward_deg": "体幹前傾",
    "head_forward_ratio": "頭部前方変位",
}

_CONNECTIONS = (
    (0, 7),
    (0, 8),
    (11, 12),
    (23, 24),
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)


def find_japanese_font(explicit_path: str | None = None) -> str | None:
    candidates = [
        explicit_path,
        os.environ.get("POSTURE_FONT_PATH"),
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


class OverlayRenderer:
    def __init__(self, font_path: str | None = None) -> None:
        from PIL import ImageFont

        selected = find_japanese_font(font_path)
        if selected is None:
            try:
                self._large_font = ImageFont.load_default(size=30)
                self._font = ImageFont.load_default(size=22)
                self._small_font = ImageFont.load_default(size=16)
            except TypeError:
                self._large_font = ImageFont.load_default()
                self._font = ImageFont.load_default()
                self._small_font = ImageFont.load_default()
        else:
            self._large_font = ImageFont.truetype(selected, 30)
            self._font = ImageFont.truetype(selected, 22)
            self._small_font = ImageFont.truetype(selected, 16)

    def render(
        self,
        frame_bgr,
        pose: PoseFrame | None,
        assessment: PostureAssessment,
    ):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw

        output = frame_bgr.copy()
        self._draw_pose(output, pose)
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image, "RGBA")

        panel_width = min(max(680, image.width // 2), image.width - 24)
        panel_height = min(380, image.height - 24)
        draw.rounded_rectangle(
            (12, 12, 12 + panel_width, 12 + panel_height),
            radius=14,
            fill=(20, 24, 31, 205),
        )

        color = {
            AssessmentStatus.OK: (75, 220, 130, 255),
            AssessmentStatus.ADVICE: (255, 190, 70, 255),
            AssessmentStatus.UNKNOWN: (210, 210, 210, 255),
            AssessmentStatus.NOT_CALIBRATED: (130, 190, 255, 255),
            AssessmentStatus.CALIBRATING: (130, 190, 255, 255),
        }[assessment.overall_status]

        status_label = {
            AssessmentStatus.OK: "OK",
            AssessmentStatus.ADVICE: "姿勢を調整してください",
            AssessmentStatus.UNKNOWN: "判定できません",
            AssessmentStatus.NOT_CALIBRATED: "未設定",
            AssessmentStatus.CALIBRATING: "基準姿勢を記録中",
        }[assessment.overall_status]
        draw.text((30, 25), status_label, font=self._large_font, fill=color)

        y = 70
        for message in assessment.messages[:3]:
            draw.text((30, y), f"・{message}", font=self._font, fill=(255, 255, 255, 255))
            y += 33

        if assessment.calibration_progress is not None:
            progress = max(0.0, min(1.0, assessment.calibration_progress))
            x0, y0, x1 = 30, y + 5, 30 + panel_width - 55
            draw.rounded_rectangle((x0, y0, x1, y0 + 16), 8, fill=(70, 75, 85, 255))
            draw.rounded_rectangle(
                (x0, y0, x0 + int((x1 - x0) * progress), y0 + 16),
                8,
                fill=(100, 180, 255, 255),
            )
            y += 32

        for key, metric in assessment.metrics.items():
            if metric.deviation is None:
                value = "--"
            elif key.endswith("_ratio"):
                value = f"{metric.deviation:+.3f}"
            else:
                value = f"{metric.deviation:+.1f}°"
            label = _METRIC_LABELS.get(key, key)
            draw.text(
                (30, y),
                f"{label}: 基準差 {value}",
                font=self._small_font,
                fill=(220, 225, 235, 255),
            )
            y += 24

        for side, foot in assessment.feet.items():
            side_label = "左足" if side == "left" else "右足"
            status = {
                AssessmentStatus.OK: "OK",
                AssessmentStatus.ADVICE: "調整",
                AssessmentStatus.UNKNOWN: "--",
            }.get(foot.status, "--")
            position = self._format_value(foot.position_deviation_ratio, 3)
            motion = self._format_value(foot.motion_rms_ratio, 3)
            pitch = self._format_value(foot.pitch_deviation_deg, 1, "°")
            draw.text(
                (30, y),
                f"{side_label}: {status}  位置差 {position}  揺れ {motion}  傾き {pitch}",
                font=self._small_font,
                fill=(220, 225, 235, 255),
            )
            y += 24

        draw.text(
            (30, 12 + panel_height - 28),
            "C: 基準姿勢を再登録   Q / Esc: 終了",
            font=self._small_font,
            fill=(180, 185, 195, 255),
        )

        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _format_value(
        value: float | None, digits: int, suffix: str = ""
    ) -> str:
        if value is None:
            return "--"
        return f"{value:.{digits}f}{suffix}"

    @staticmethod
    def _draw_pose(frame_bgr, pose: PoseFrame | None) -> None:
        if pose is None:
            return
        import cv2

        height, width = frame_bgr.shape[:2]
        points = pose.image_landmarks
        for start, end in _CONNECTIONS:
            if start >= len(points) or end >= len(points):
                continue
            a, b = points[start], points[end]
            if min(a.visibility, b.visibility) < 0.4:
                continue
            pa = (int(a.x * width), int(a.y * height))
            pb = (int(b.x * width), int(b.y * height))
            cv2.line(frame_bgr, pa, pb, (80, 220, 160), 2, cv2.LINE_AA)
            cv2.circle(frame_bgr, pa, 4, (40, 180, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, pb, 4, (40, 180, 255), -1, cv2.LINE_AA)
