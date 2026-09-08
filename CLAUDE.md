# CLAUDE.md

## Project

ReallyRally — local, offline tennis rally extraction from practice videos. Two-pass CV pipeline: fast pixel-diff motion scan → MediaPipe pose estimation → self-adaptive kinematic thresholding → FFmpeg export. macOS-first (Apple Silicon), cross-platform.

## Architecture

```
src/really_rally/
  cli.py              # argparse CLI (rally, serve, feed, evaluate subcommands)
  gui.py              # CustomTkinter GUI with drag-and-drop
  core/
    config.py         # All tuning parameters — no external config files
    features.py       # FeatureEngine, SelfAdaptiveThreshold, RollingZScoreNormalizer, MovementTracker
    models.py         # RallySegment + ServeEvent dataclasses
    serve_detector.py # ServeDetector — pose-based serve FSM + confidence scoring
    vision.py         # MotionDetector (pixel-diff), PoseExtractor (MediaPipe wrapper)
  processors/
    base.py           # TennisRallyProcessor — 2-pass pipeline, 3-state machine, export
    serve.py          # ServePracticeProcessor — drives ServeDetector, exports serve_events.csv
    feed.py           # FeedPracticeProcessor — pre-scan player scale, feeder rejection
  utils/
    media_io.py       # Video metadata extraction, filename generation
    telemetry.py      # Matplotlib 3/4-panel PNG plots
    evaluate.py       # Temporal IoU precision/recall/F1 eval framework
```

Three processing modes inherit from `TennisRallyProcessor`:
- **Rally**: 3-state temporal state machine (Idle → Potential → Active) with Z-scored adaptive thresholds
- **Serve**: Pose-based finite-state machine (`ServeDetector`) that recognizes the toss → overhead-swing sequence: it infers handedness (toss arm rises first while the hitting arm is still down), then requires an overhead contact (high wrist velocity + raised arm) on the opposite arm within a temporal window. Emits `serve_events.csv` with per-serve handedness, toss/contact timestamps, and component scores. No ball/racket/court stream yet.
- **Feed**: Pre-scan player scale calibration to reject coach/feeder near camera

## Conventions

- No comments unless the WHY is non-obvious
- All tunable parameters in `core/config.py` (UPPER_CASE constants), imported via `from really_rally.core.config import *`
- Processors override `analyze_segment_data()` for mode-specific detection logic
- FPS-dependent calculations use `effective_fps = self.fps / PROCESS_EVERY_N_FRAMES`
- Format standardization via `target_format` dict (`{width, height, fps}`) ensures clean concatenation of mixed-resolution clips

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# CLI
python run.py rally video.mp4 [--output dir] [--split] [--plot]
python run.py serve video.mp4
python run.py feed video.mp4
python run.py evaluate --gt annotations.json --pred rallies.csv

# GUI
python gui.py

# Tests
pytest tests/ -v

# Lint
ruff check src/
```

## Key Config Parameters

Rally detection uses a 3-state machine with Z-scored adaptive thresholds:
- **Z-score normalization** converts raw kinematic signals (arm_energy, whip_ratio) to unitless values via `RollingZScoreNormalizer` (causal sliding window)
- **Robust adaptive threshold** uses P25 (not median) as baseline + k * MAD, with asymmetric EMA smoothing (slow rise 0.05, fast fall 0.3)
- Stroke gate: `z_arm > t_arm AND z_whip > t_whip AND hip_vel > HIP_VELOCITY_STROKE_THRESHOLD AND spatial_variance >= SPATIAL_VARIANCE_STROKE_MIN`
- Player distance gate: `PLAYER_HEIGHT_MIN <= player_height <= PLAYER_HEIGHT_MAX`

All kinematic values are bbox-diagonal-normalized, so they are very small (arm_energy ~0-0.2, hip_vel ~0-0.05). The distance/velocity gates operate in this normalized space.

## Serve Detection (pose-only)

`ServeDetector` recognizes the serve from two pose signatures — the toss and the overhead swing — without ball/racket/court detection:

- **Toss**: one arm rises above the shoulder (ideally the head) and is *held* there (`SERVE_TOSS_MIN_SEC`). The toss arm is the one that rises *while the opposite arm is still down* — this disambiguates it from the hitting arm's trophy position, which rises only after the toss arm is already up.
- **Contact**: the opposite arm's wrist crosses `SERVE_CONTACT_WRIST_VEL` (bbox-normalized) while raised above the shoulder, within `SERVE_TOSS_TO_CONTACT_MIN_SEC`…`MAX_SEC` after toss release. Scoring uses the swing region's *peak* velocity and *peak* wrist height (not a single frame) because the velocity spike and the highest wrist position are a few frames apart.
- **Empirical finding**: the trophy (both-arms-up) and racket-drop phases are *not* reliably visible in wrist trajectories — the toss arm goes up while the hitting arm stays cocked low, then the hitting arm swings up as the toss arm drops. So toss + overhead-swing/contact carry the detection; trophy/racket-drop were dropped from the pose-only score.

Handedness, toss time, and contact time are emitted per serve in `serve_events.csv`. A YOLO ball/racket/court stream can be folded in later as extra scoring terms without changing the FSM.

## Gotchas

- **MediaPipe must be < 0.11.0** — the legacy Solutions API was removed in 0.11+
- **FFmpeg required** on PATH for video export, concat, and metadata transfer
- **First frame of concat** — `merge_clips` needs absolute paths in the concat list
- **VFR videos** — iPhone footage is often variable framerate; `target_format` standardization is critical
- **Adaptive threshold padding** — rest-gap zero-padding between motion segments keeps the MAD baseline from drifting upward during dense rallies
- **`_mask_to_indices` / `_mask_to_segments`** — both exist in base.py; `_mask_to_indices` is the current one, `_mask_to_segments` is a duplicate (legacy from refactor)
- **OpenCV seek + read** — `process_pixel_segment` uses `cap.set()` then `cap.read()` because some backends misbehave with `cap.retrieve()` after seeking
- **Bbox-normalized values** — all velocities and energies are divided by the pose bounding-box diagonal. This makes hip_vel very small (max ~0.05) and arm_energy very small (max ~0.2). Thresholds must be tuned in this normalized space.
- **player_height caps** — the upper bound was 0.28 (rejecting 97% of frames on real footage). Raised to 1.50 as a default; feed mode auto-calibrates via `estimate_player_scale()`.
