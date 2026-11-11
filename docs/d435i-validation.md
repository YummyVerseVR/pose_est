# Implementation validation — 2026-11-12

Base: PR #1 / feat/meal-posture-feedback at
1ae763c1f642dc423379c1671a3893d2088170e3. Clean isolated checkout, no target AGENTS.md.

## Commands and outcomes

- Before fix: `uv run pose-est --help` failed with `ImportError: cannot import name main`.
- After entry point fix: `UV_CACHE_DIR=/private/tmp/pose-est-uv-cache uv run pose-est --help`
  succeeded, including `--camera-source {webcam,realsense}`. No pyrealsense2 installed.
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`: **22 passed**.
- `PYTHONPATH=src .venv/bin/python -m compileall -q src tests`: passed.
- `git diff --check`: passed.
- `python -m json.tool` and `PostureConfig.load` on both example configs: passed.
- Synthetic 640×480 overlay rendered and visually inspected; reduced panel height/opacity.
- PyPI JSON checked: current version 2.59.1.11053, CPython 3.12 Windows x64 wheel present.

The sandboxed host uv process could not access its default cache and later panicked in
macOS system-configuration. Dependency resolution / real CLI verification succeeded outside
that sandbox with a task-specific temporary cache. These are host restrictions, not Windows tests.

## Automated evidence

Synthetic aligned depth changes real XYZ and torso/head-forward metrics and passes through
PostureAnalyzer to sustained advice. Tests cover invalid/mixed/sparse depth, metric units,
positive pitch, fixed forward axis despite face turning, translation invariance, display-only
mirroring, independent metric loss, reset after occlusion, insufficient calibration,
old baseline rejection in RGB-D, changed sensor/setup/policy rejection, lazy SDK errors,
SDK align, bounded timeout/USB failure, multi-device checks, partial startup cleanup,
Ctrl+C, full mocked C→calibration save→P→Q, minimal JSONL logging and CLI without SDK.

## Independent read-only review

A separate reviewer inspected code/tests/docs and independently reran all 22 tests.
Two findings resolved: selection of an ear on a side with hidden shoulder/hip, and metre
hip displacement incorrectly formatted as degrees. Side-selection regression test added.
Reviewer reported no remaining findings. This is code review, not clinical or hardware approval.

## Pending and limits

All Windows 11/D435i Viewer, real RGB-D alignment, frame rate, USB unplug behavior,
75°/1.3m observability, food/arm/table occlusion, 30–60 second baseline and user notification
acceptance tests in realsense.md remain **not performed**. No camera or participant study ran.
No clinical efficacy or safety validation is implied. Fixed camera level and placement changes
must be checked manually; no IMU, automatic motion detection, swallowing recognition or foot
contact inference. A uniformly wrong background patch can still pass local depth quality checks.
