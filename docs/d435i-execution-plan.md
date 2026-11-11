# D435i execution plan (2026-11-12)

Scope: RGB-D acquisition, robust landmark deprojection, independent metric observation,
versioned sensor-bound calibration, UI/CLI, synthetic/mock tests and setup documentation.
User request: execute supplied pose_est_D435i_implementation_spec.md. Base PR #1 head:
1ae763c1f642dc423379c1671a3893d2088170e3; checkout initially clean; no AGENTS.md present.
No stimulation, clinical threshold validation, participant study or release is performed.
Implementation settings are experimental UX parameters. Existing webcam API stays compatible.

Verification: existing tests plus synthetic depth to assessment tests, mocked SDK lifecycle,
CLI entry point with SDK absent, compile and diff checks; independent read-only review.
Hardware and Windows runtime tests remain explicitly pending. Rollback: discard this checkout's
uncommitted changes; existing baseline v1 files and remote branch remain untouched until review.
