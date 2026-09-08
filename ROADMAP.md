# Roadmap

## Short-term — Stabilization

- [ ] **Robust normalization during MediaPipe dropout** — bounding-box-diagonal normalization (merged) is more stable than torso-length, but mid-hip/neck dropout during rapid movement still degrades reliability. Consider temporal interpolation of the bbox diagonal during tracking loss.
- [ ] **MAD baseline drift mitigation** — when rally density is high (>30s continuous hitting), the rolling median shifts toward active energy, raising thresholds and causing missed detections. Approaches: asymmetric update rates (slow to rise, fast to fall), or explicit rest-state detection via the motion detector.
- [ ] **Deduplicate `_mask_to_indices` / `_mask_to_segments`** — both methods exist in `base.py` with identical logic. Remove the unused duplicate.

## Medium-term — Detection quality

- [ ] **Racket head tracking** — whip ratio is currently derived from wrist velocity relative to shoulder. Tracking the racket head (object detection or dedicated keypoints) would dramatically improve stroke detection specificity and reduce false positives from non-stroke arm movements.
- [x] **Pose-based serve detection (V1)** — `ServeDetector` recognizes the toss → overhead-swing sequence from MediaPipe pose alone (toss arm rises first → opposite arm swings overhead within a temporal window). Emits handedness + toss/contact timestamps. Trophy/racket-drop are intentionally not scored (not visible in wrist trajectories).
- [ ] **Serve detection V2 (YOLO ball/racket/court)** — add a YOLO object/keypoint stream (CourtSide-Computer-Vision-v1 YOLOv11 nano covers ball + racket + court zones; RacketVision provides racket keypoints) to add contact-accuracy, ball-flight, and behind-baseline scoring terms. The `ServeDetector` FSM is stream-agnostic and accepts these as extra scoring without structural change.
- [ ] **Occlusion handling during swing phase** — wrist/elbow keypoints are frequently lost at the moment of ball contact (the most critical instant). Temporal smoothing or interpolation during dropout windows would improve reliability.
- [ ] **Systematic benchmark dataset** — the evaluation framework (`evaluate.py`) and annotation template are in place. Build a corpus of 10-20 hand-labeled videos spanning all three modes (rally, serve, feed) with varied camera positions, lighting, and player counts.
- [ ] **Multi-player support** — current single-player MediaPipe pose limits the tool to practice footage. Multi-pose tracking would enable match-play rally extraction.
- [ ] **Per-mode threshold tuning** — the feed processor re-instantiates `SelfAdaptiveThreshold` per segment rather than maintaining global state like the base processor. (Serve now uses `ServeDetector`, so it no longer shares this issue.)

## Long-term — Product maturity

- [ ] **Test suite** — ✅ 66 unit tests covering evaluation, features, models, config, and media I/O. Expand with integration tests on synthetic video clips.
- [ ] **CI/CD** — ✅ GitHub Actions workflow. Add pre-commit hooks for ruff + pytest.
- [ ] **Package installation** — ✅ `pyproject.toml` with entry points (`really-rally`, `really-rally-gui`).
- [ ] **Mode auto-detection** — the user must manually select rally/serve/feed mode. Camera setup heuristics (player position relative to baseline, repetitive short-duration motion patterns) could auto-select.
- [ ] **Real-time / streaming mode** — the current two-pass design requires a complete video file. A single-pass streaming mode with a short lookahead buffer would enable live practice feedback.
- [ ] **Export format options** — currently hardcoded to H.264 MP4. Add options for ProRes, HEVC, or raw frame sequences for editing workflows.
- [ ] **Configuration profiles** — move tunable parameters from `config.py` constants to a user-editable YAML/TOML file with named presets (indoor, outdoor, wide-angle, etc.).

## Known failure modes (to benchmark)

1. **D_torso normalization** — when MediaPipe loses neck/mid-hip landmarks, bounding-box diagonal collapses, inflating normalized velocities → false positives
2. **Wrist/elbow occlusion during swing** — the most informative frames are often missing → stroke detection gaps
3. **MAD baseline drift** — continuous hitting >30s shifts the median to active energy → missed strokes
4. **Coach/feeder false positives** — when the coach walks onto court to feed, their larger scale triggers detection; partially mitigated by player-scale calibration in feed mode
5. **Low-light / backlit footage** — MediaPipe pose confidence drops significantly, reducing valid pose frames → missed rallies
6. **Wide-angle / fisheye lenses** — player scale varies nonlinearly across the frame, breaking the fixed height thresholds
