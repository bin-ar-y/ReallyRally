# Roadmap

## Short-term — Stabilization

- [x] **Robust normalization during MediaPipe dropout** — short dropouts are now bridged (`POSE_DROPOUT_BRIDGE_STEPS`): the previous pose is held and displacement divided by the steps spanned, so the frame reacquired after a dropout still yields a per-step velocity instead of a spurious zero. Measured on broadcast footage: 9.1% of frames lose pose in 3 runs, 2 of which are now bridged.
- [x] **Z-score noise amplification** — *not* the drift originally hypothesised. Measurement showed raw signal and Z-score move in opposite directions while the adaptive threshold falls in step, so baseline drift was not causing misses. The real defect was the opposite: a scale-free Z-score amplified numerically negligible arm energy (~0.0007 vs ~0.10 for real swings) to z > 2 on quiet windows. Fixed via raw-space floors (`COARSE_MIN_ABS_ARM` / `COARSE_MIN_ABS_WHIP`); A/B across 6 clips cut candidate frames 6.0% with no change in exported segments.
- [x] **Deduplicate `_mask_to_indices` / `_mask_to_segments`** — duplicate removed.
- [x] **`transfer_metadata` was incomplete** — the method ended at a `# Replace original` comment, so metadata was never applied and an orphan `_meta.mp4` was left behind. Now applies via `os.replace` and cleans up on failure; verified the merged clip carries the source `creation_time`.
- [ ] **MediaPipe dropout interpolation** — bridging covers gaps ≤ `POSE_DROPOUT_BRIDGE_STEPS`; longer gaps (observed: one 41-step run) are still treated as a fresh start.

## Medium-term — Detection quality

- [x] **Coarse-to-fine detection** — the strict AND-gate (`z_arm AND z_whip AND hip_vel AND spatial_var`) discarded real strokes whenever any single signal was weak. Replaced with an additive `CoarseCandidateDetector` (one strong signal suffices) feeding a hysteresis `RallyStateMachine` (idle → candidate → active → cooldown) where starting a rally needs two strokes within `RALLY_START_GAP_SEC` but continuing tolerates `RALLY_CONTINUE_GAP_SEC`. Measured mean stroke-peak recall 40.8% → 54.2% over 6 clips (label-free proxy).
- [x] **Object-evidence architecture** — `ObjectDetector` contract, `BallTracker`, `EvidenceFusion` and a candidate-window-only Pass 3 are in place. Racket/ball confidence defaults to neutral when no backend is registered, so a missing model cannot reduce recall. **No detector is installed yet**, so the pose / +racket / +racket+ball ablation is still pending. Note a COCO-pretrained model covers both `tennis racket` and `sports ball`, so no custom training is needed.
- [ ] **Racket head tracking** — whip ratio is currently derived from wrist velocity relative to shoulder. Tracking the racket head would improve stroke specificity; the wiring now exists, only the backend is missing. The environment needed to judge whether it *helps* is now in place: label a few recordings once in the Debug Studio, then score `Pose` against `Pose + racket` on identical ground truth rather than arguing from architecture.
- [x] **Pose-based serve detection (V1)** — `ServeDetector` recognizes the toss → overhead-swing sequence from MediaPipe pose alone (toss arm rises first → opposite arm swings overhead within a temporal window). Emits handedness + toss/contact timestamps. Trophy/racket-drop are intentionally not scored (not visible in wrist trajectories).
- [ ] **Serve detection V2 (YOLO ball/racket/court)** — add a YOLO object/keypoint stream (CourtSide-Computer-Vision-v1 YOLOv11 nano covers ball + racket + court zones; RacketVision provides racket keypoints) to add contact-accuracy, ball-flight, and behind-baseline scoring terms. The `ServeDetector` FSM is stream-agnostic and accepts these as extra scoring without structural change.
- [ ] **Occlusion handling during swing phase** — wrist/elbow keypoints are frequently lost at the moment of ball contact (the most critical instant). Temporal smoothing or interpolation during dropout windows would improve reliability.
- [ ] **Systematic benchmark dataset** — tooling is in place (`benchmarks/label.py` for contact sheets + interactive/scripted labelling, `--status` to track coverage; the Debug Studio for in-app labelling with the video, telemetry and detector state on one timeline) and the runner reports recall/precision/F1 plus boundary timing error and runtime. **The corpus itself is still empty**: `benchmarks/cases/demo_serve_practice/` ships with no labels. Every detection number below the label-free proxies is blocked on hand-labelling 3-5 clips. The studio does not remove that work, but it makes it roughly a video-watch instead of a CSV exercise, and it writes labels in a format `run_benchmark.py` already consumes.
- [ ] **GUI responsiveness** — *done*: worker→UI updates moved to a thread-safe queue with a main-thread poller, the pose graph is cached process-wide, Start/Stop give immediate feedback, and ffmpeg work honours `stop_event` with timeouts.
- [ ] **Multi-player support** — current single-player MediaPipe pose limits the tool to practice footage. Multi-pose tracking would enable match-play rally extraction.
- [ ] **Per-mode threshold tuning** — the feed processor re-instantiates `SelfAdaptiveThreshold` per segment rather than maintaining global state like the base processor. (Serve now uses `ServeDetector`, so it no longer shares this issue.)

## Long-term — Product maturity

- [ ] **Test suite** — ✅ 480 unit tests. Beyond evaluation, features, models, config and media I/O, they cover the feature store, config snapshot, extracted detection (including exact parity against the pre-refactor implementation), replay-vs-live equivalence, annotation, and the studio's thread-safety invariant. Still thin on end-to-end tests over real footage — the synthetic clips exercise the plumbing, not the vision.
- [ ] **CI/CD** — ✅ GitHub Actions workflow. Add pre-commit hooks for ruff + pytest.
- [ ] **Package installation** — ✅ `pyproject.toml` with entry points (`really-rally`, `really-rally-gui`, `really-rally-studio`).
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
