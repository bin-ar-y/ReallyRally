# Tennis Rally Extractor

A local, offline tool to automatically extract tennis rallies from practice videos. Designed for **macOS (Apple Silicon)** but compatible with other platforms.

## Prerequisites

- **Python 3.9–3.11** (3.11 recommended; the pinned MediaPipe release has no Python 3.12 wheel)
- **FFmpeg** (Must be installed and accessible in your terminal)

### 1. Install FFmpeg
On macOS with Homebrew:
```bash
brew install ffmpeg
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```
> [!IMPORTANT]
> This project requires **MediaPipe < 0.11.0** to use the legacy Solutions API.
> If you are not using the provided `requirements.txt`, ensure you install a compatible version:
> ```bash
> pip install "mediapipe<0.11.0"
> pip install customtkinter
> ```

### 3. (Recommended) Use a Virtual Environment
To avoid conflicts with other Python packages, run the script within a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (GUI)
The easiest way to use the tool is via the graphical interface.

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch GUI
python3 gui.py
```

### GUI Features
1.  **Focused Workspace**: A charcoal-and-green interface with a video queue, session settings, and a persistent progress area.
2.  **Select Videos**: Choose or drop one or multiple video files to process in batch. Files are checked in the background, with an “Adding videos” indicator.
3.  **Output Location**: Select where the results should be saved.
4.  **Start Processing**: Runs the analysis.
    - **Progress Bar**: Shows scanning, analysis, and export progress, with elapsed time and an activity indicator during preparation, plotting, and merging.
    - **Session Settings**: Locked during processing so the whole batch uses the options you started with.
    - **Stop Button**: Acknowledges cancellation immediately, stops the encoder, and keeps completed clips.
5.  **Robust Batch Processing**:
    - Automatically standardizes different video resolutions (e.g., mixing 4K and 1080p).
    - Ensures unique filenames for every clip to prevent overwrites.
    - merges all detected rallies into a single output file: `{Date}_Merged_Rallies.mp4`.
6.  **Finish at Your Pace**: Completion appears inside the app. Use **Open results** to open the output folder, process another session, or choose **Quit**. Nothing opens automatically.
7.  **Normal Quit**: The Quit button, window close button, and Cmd+Q on macOS (Ctrl+Q elsewhere) share the same exit flow. If work is running, choose whether to keep working or stop and quit; the app waits for cleanup before closing.

## Usage (Command Line)

Run the script on your video file:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Rally practice
python3 run.py rally /path/to/my_match.mp4

# Serve practice
python3 run.py serve /path/to/my_serves.mov

# Multiball / coach feeding
python3 run.py feed /path/to/feeding.mp4

# Options (any mode)
#   --output DIR   write somewhere other than ./<date>_<video>_<mode>/
#   --split        keep the individual clips as well as the merged video
#   --plot         write a telemetry PNG
```

If you installed the package with `pip install -e .`, the equivalent commands
are `really-rally rally ...` and `really-rally-gui`.

### Output

Results are grouped by mode and then by recording, so one folder holds
everything about one video:

```
out/
└── serve/                            # named after what was extracted
    ├── 022626_Merged_Serves.mp4      # whole-batch merge (multi-video runs only)
    ├── manifest.csv                  # every clip in the batch
    └── IMG_1165/
        ├── 092626_IMG_1165_serve.mp4 # this video's segments, merged
        ├── serve.csv                 # start/end time of each clip
        ├── features.csv              # full per-frame feature store
        ├── debug_metrics.csv         # 18-column telemetry summary
        ├── telemetry.png             # with --plot / the GUI checkbox (on by default)
        ├── serve_events.csv          # serve mode only: handedness, toss/contact times
        ├── labels.csv                # your ground truth        (written by the studio)
        ├── predictions.csv           # detector output          (written by the studio)
        ├── experiments.csv           # evaluated configurations (written by the studio)
        └── clips/                    # with --split: the individual clips
```

Mode folders are `rally/`, `serve/` and `feed/`. The folder is the unit the
Debug Studio opens — point it at `serve/IMG_1165/` and the video, telemetry,
labels and predictions all load together.

The three data files have different owners, and that separation is deliberate:
`features.csv` is written by extraction, `predictions.csv` by the detector, and
`labels.csv` by you. **Re-tuning the detector can never alter your labels**, so
a labelled recording stays a valid regression test.

At the end of a run the CLI prints an **output manifest** listing every file
written and its location.

### Where to look when detection is wrong

Open the **Debug + Label Studio** — it is the fastest way to see *why* a
decision was made, because it synchronises video, telemetry and detector state
on one timeline:

```bash
# Open a finished run — video, telemetry, labels and predictions together
really-rally studio --results out/serve/IMG_1165

really-rally studio                       # or: python run.py studio
really-rally studio --video IMG_1133.MOV  # just a video
```

Scrub to a suspicious moment, read the raw values next to the thresholds they
were compared against and the evidence that fired, mark what actually happened,
then tune a threshold and re-run detection **without re-analysing the video** —
about 20 ms for a replay, versus minutes for a full pass.

It reads `features.csv` from the recording's results folder, which the normal
extraction run writes, so run `rally` / `serve` / `feed` first.

See **[docs/STUDIO.md](docs/STUDIO.md)** for the keyboard map and workflow.

`<video>_telemetry.png` remains the static summary for reports, and
`<video>_debug_metrics.csv` carries the same data numerically.

## Configuration

All tunable parameters live in `src/really_rally/core/config.py` — there are no
external config files. The ones worth knowing:

```python
# Adaptive thresholding
ADAPTIVE_WINDOW_SEC = 45.0        # Sliding window for the resting baseline
ARM_ENERGY_K_MULTIPLIER = 2.5     # Sensitivity (P25 + k*MAD)
WHIP_RATIO_K_MULTIPLIER = 2.5
MIN_ENERGY_FLOOR = 0.25           # Absolute floor, in Z-score space

# Coarse candidate detection (additive, one strong signal is enough)
COARSE_CANDIDATE_THRESHOLD = 2.0
COARSE_MIN_ABS_ARM = 0.005        # Raw-space floor; stops Z-score noise amplification
COARSE_MIN_ABS_WHIP = 0.3

# Rally continuation — harder to START a rally than to CONTINUE one
RALLY_START_GAP_SEC = 7.0         # Two strokes within this window confirm a rally
RALLY_CONTINUE_GAP_SEC = 11.0     # An active rally tolerates a gap this long

# Serve
SERVE_LOCK_HANDEDNESS = True      # One serving hand per video
SERVE_FORCED_HANDEDNESS = None    # None = infer; or 'left' / 'right'

# Pose
MP_MODEL_COMPLEXITY = 1           # 0=Lite, 1=Full, 2=Heavy (see note below)
POSE_DROPOUT_BRIDGE_STEPS = 3     # Bridge short tracking dropouts
```

> **Note on `MP_MODEL_COMPLEXITY`:** the mediapipe wheel bundles only the
> `full` (1) model. Setting it to 0 or 2 makes MediaPipe download the lite/heavy
> weights from GitHub on first use, which fails behind a TLS-inspecting proxy.
> See Troubleshooting.

## How it Works

Detection is **coarse-to-fine**: cheap signals propose candidates generously,
and only candidates get expensive verification.

1. **Pre-motion scan** — downscaled pixel differencing skips empty scenes.
2. **Pose extraction** — MediaPipe tracks the player's skeleton on 360p frames.
3. **Feature analysis** — kinematic features (Arm Energy, Whip Ratio, hip
   velocity, spatial variance), normalized by the pose bounding-box diagonal so
   they are comparable across camera distances.
4. **Adaptive thresholding** — a rolling Z-score normalizes each signal against
   its own local baseline, then a robust P25 + k·MAD threshold adapts to the
   camera. Evidence must *also* clear an absolute raw-space floor, because a
   scale-free Z-score will otherwise amplify numerical noise into a "strong"
   signal on quiet stretches.
5. **Coarse candidate detection** — each signal above its threshold adds
   weighted evidence to a score. One strong signal is enough to open a
   candidate; there is no requirement for all of them to fire at once.
6. **Hysteresis state machine** — `Idle → Candidate → Active → Cooldown`.
   Confirming a rally takes two strokes within `RALLY_START_GAP_SEC`; continuing
   one tolerates `RALLY_CONTINUE_GAP_SEC`, so a single missed stroke does not
   cut the segment short.
7. **(Optional) object verification** — racket and ball detection over candidate
   windows only, fused with pose and temporal evidence. Ships disabled: with no
   detector registered this step is a no-op and behaviour is pose-only.
8. **Export** — segments are cut and merged with FFmpeg.

## Evaluation

The project ships an evaluation harness, but **`benchmarks/cases/` is empty** —
there is no hand-labelled ground truth yet, so published recall and precision
figures are label-free proxies rather than real measurements.

- **[docs/LABELLING.md](docs/LABELLING.md)** — how to label clips (about 5–10
  minutes each, five clips is enough to produce meaningful numbers).
- `python benchmarks/label.py --status` — see which cases still need labels.
- `python benchmarks/run_benchmark.py` — recall, precision, F1, boundary error
  and runtime.

## Troubleshooting

- **Rallies missed?**
  - Lower `COARSE_CANDIDATE_THRESHOLD`, or raise `RALLY_CONTINUE_GAP_SEC`.
  - Check whether the rally appears at all in the pre-motion scan; a very still
    player can be filtered before pose estimation ever runs.
- **Ball picking or walking included?**
  - Raise `COARSE_CANDIDATE_THRESHOLD` and/or `COARSE_MIN_ABS_ARM`.
- **Serve detection finds nothing?**
  - Run with `--plot` and check whether the wrist ever rises above the
    shoulder in the telemetry. If it does not, the camera angle is likely
    hiding the toss.
- **Telemetry plot missing?**
  - It is only written with `--plot` (CLI) or the "Generate Telemetry Plot"
    checkbox (GUI, on by default).
- **"MediaPipe could not load the pose model"?**
  - Only the `full` model ships inside the wheel; complexity 0 and 2 are
    downloaded on first use. Either set `MP_MODEL_COMPLEXITY = 1`, fix your
    certificate chain, or (trusted networks only) set
    `REALLY_RALLY_ALLOW_INSECURE_SSL=1`.
- **MediaPipe Import Error?**
  - Ensure you have installed `mediapipe<0.11.0`. Newer versions (0.11+) have
    changed the API structure.

