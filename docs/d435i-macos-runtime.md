# D435i macOS runtime checkpoint — 2026-11-12

Scope: local hardware bring-up following implementation commit 1be7e42. This is
partial engineering evidence, not posture accuracy, clinical safety or Windows validation.
No images/depth files were saved or transmitted by the diagnostic programs.

## Operator-reported hardware results

The user ran locally compiled SDK diagnostic programs in Terminal with sudo and supplied
text output. No personal identifiers or device serials are reproduced here.

Environment: macOS 15.1.1 arm64; Homebrew librealsense 2.58.3; D435i firmware 5.16.0.1;
SDK-reported USB type 3.2.

- First diagnostic: depth Z16 640×480×30 received frames; later color-only and combined
  tests timed out (5 seconds). Combined 15 fps also timed out.
- After reconnect, color-first diagnostic: BGR8 640×480×30 received 3 frames; subsequent
  YUYV30/RGB15 attempts timed out. This does not establish a format incompatibility.
- After reconnect, single-start RGB-D diagnostic: 5 paired color BGR8/depth Z16 frames
  at requested 640×480×30 were received and depth aligned to color using rs2::align.
  Aligned depth intrinsics fx=606.745, fy=605.395 were reported.

Both individual streams and combined acquisition/alignment have short-run evidence.
The repeat-start failure pattern suggests a stop/start or device ownership issue, but
root cause remains unconfirmed. Continuous frame rate, long-run stability, repeated startup,
pose inference on actual images, per-landmark depth fusion and calibration remain pending.

## Python environment prepared by agent

- Official SDK source tag v2.58.3, commit dfd6aa91250f5c31521d72d627865417989bb4e7,
  from https://github.com/realsenseai/librealsense/tree/v2.58.3.
- Python 3.12.9 native arm64; SDK wrapper sources compiled against installed
  librealsense2 and rsutils, pybind11 2.13.6 (SDK-pinned), nlohmann_json 3.12.0.
  Wrapper built in /private/tmp/d435i-python-wrapper and copied into the app virtualenv.
  Import reports pyrealsense2 2.58.3; no hardware opened during import verification.
- MediaPipe full model obtained by existing scripts/download_model.py. Model initialization
  and inference on an all-zero synthetic frame succeeded outside the restricted sandbox.
- Launcher: /private/tmp/run-d435i-posture.sh. Local config uses finite 15-second acquisition
  timeout, 30-second baseline and 300 samples per enabled metric, no research logging.
  Baseline remains local and Git-ignored. Start the app directly after reconnect, avoiding
  intermediate camera probes. C calibrates, P pauses, Q exits.

These paths are temporary local build artifacts, not a portable macOS SDK distribution.
During hardware bring-up no firmware or OS security settings were changed.

SDK sources: https://github.com/realsenseai/librealsense/blob/v2.58.3/wrappers/python/CMakeLists.txt
and https://github.com/realsenseai/librealsense/blob/v2.58.3/CMake/pybind11-download.cmake.in.

## Operator feedback and local response tuning

Operator reported that the posture app ran and requested quicker assessment. Actual
metric accuracy and end-to-end latency were not measured. Only the local Mac config
/private/tmp/d435i-posture-macos.json was tuned: warning hold 2.5→1.2 s, recovery hold
1.0→0.5 s, smoothing alpha 0.25→0.5, calibration 30→8 s and minimum samples 300→60.
Notification cooldowns (10/35 s), angular deviation thresholds and depth quality rules
were retained. Shorter persistence may increase unwanted advice during meal motions.
The previous local config is backed up as d435i-posture-macos.before-response-tuning.json.
Changing metric settings invalidates the sensor-bound baseline; C must register it again.
The time and sample requirements must both be met, so registration may exceed 8 s.

Deterministic synthetic observations at 33 ms intervals, sustained torso step 0→20°:
first advice after 2587 ms before tuning, 1267 ms after tuning; a 400 ms excursion
produced no advice. These are analyzer tests, not measured camera-to-display latencies.
All 22 repository tests passed after tuning. The active app reads config only at startup;
restart and hardware verification of the new settings remain pending.

The local response tuning is reproduced without machine-specific paths in
`config.realsense-responsive.example.json`. Reusable setup is documented in realsense.md.
