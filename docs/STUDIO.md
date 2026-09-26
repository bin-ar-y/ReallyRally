# Debug + Label Studio

An interactive workspace for answering the question a static telemetry plot
cannot: **what was the player actually doing at this instant, and why did the
detector decide what it decided?**

Video, raw features, adaptive thresholds, detector state, events and human
labels all share one coordinate — video time — so moving the playhead moves
everything together.

## Launching

```bash
# Empty workspace — open a folder from the toolbar
really-rally studio
python run.py studio            # same thing, no install needed

# Open a finished run: video, telemetry, labels and predictions together
really-rally studio --results out/serve/IMG_1165

# Or just a video
really-rally studio --video IMG_1133.MOV
```

**`--results` is the intended entry point** once a run has finished. The
pipeline writes one folder per recording, and pointing at it restores the whole
session, so a run is examined as a unit rather than as a video and a CSV opened
separately. If you point at a `serve/` folder holding exactly one recording it
resolves to that; if it holds several, you get asked to pick (the studio will
not silently choose for you).

There is also a direct console script, `really-rally-studio`. It is declared in
`pyproject.toml`, so it appears after `pip install -e .` has been re-run — an
environment installed before the studio existed will not have it yet. The two
commands above always work.

The studio pairs a video with its `*_features.csv` automatically when they sit
in the same folder. Generate it by running the normal pipeline first:

```bash
python run.py rally IMG_1133.MOV --output ./out
# writes IMG_1133_features.csv alongside the debug CSV and telemetry PNG
```

The studio needs that feature store. Without one it still plays the video, but
there is no telemetry to inspect and no replay, because replay reads the cached
features rather than the video.

## Keyboard

| Key | Action |
|---|---|
| `space` | play / pause |
| `←` / `→` | ∓ 1 frame |
| `shift` + `←` / `→` | ∓ 10 frames |
| `j` / `l` | ∓ 1 second |
| `shift` + `j` / `l` | ∓ 5 seconds |
| `↑` / `↓` | previous / next label |
| `[` | start a ground-truth interval at the playhead |
| `]` | close the interval |
| `s` | add a point label at the playhead |
| `backspace` | delete the label at the playhead |

Clicking anywhere on the telemetry seeks the video to that timestamp. Dragging
the playhead collapses to the newest request, so scrubbing stays responsive
instead of queueing hundreds of frames the user has already passed.

## The results folder

Everything about one recording lives in `<output>/<mode>/<video_stem>/`:

```
out/serve/IMG_1165/
    serve.mp4            merged clips        ← extraction
    serve.csv            clip manifest
    features.csv         full feature store
    debug_metrics.csv    telemetry summary
    telemetry.png        static plot
    labels.csv           your ground truth   ← you
    predictions.csv      detector output     ← detector
    experiments.csv      tuning history
    clips/               individual clips (--split)
```

The studio keeps these apart on purpose. Tuning a threshold rewrites only the
predictions. **Ground truth is never written by the detector**, so a labelled
recording stays a valid regression test no matter how the detector changes.
`experiments.csv` records every evaluated configuration, so a tuning result can
be reproduced rather than remembered.

Open the folder (`--results`, or the toolbar button) and all of it loads
together — no opening a video, a PNG and a CSV in three applications to work out
what happened.

## Labelling vocabulary

Deliberately small to start — the immediate goal is separating real play from
tennis-like movement without ball play, which is the confusion the detector
currently makes.

- **Intervals**: Rally · Shadow Practice · Idle / Non-tennis · Uncertain
- **Points**: Real Stroke · Serve · Shadow Swing · Uncertain Event

Labels are free text; the lists are suggestions. Extending to forehand /
backhand / volley needs no schema change.

Labels are written to `*_labels.csv` and can be exported to the
`annotations.json` format that `benchmarks/run_benchmark.py` and
`python run.py evaluate` already consume.

## Tuning: two tiers

The Config tab separates parameters by what changing them actually costs.

**Replayable** — thresholds, evidence weights and rally gaps. These change the
*decision*, not the measurement, so **Apply & Replay** recomputes detection from
the cached feature store in milliseconds. No video decode, no MediaPipe.

**Requires video re-analysis** — pose model, frame sampling, motion pre-filter.
These change how features are extracted, so they cannot be replayed from cached
data. The studio lists them but will not pretend to apply them; re-run the video
instead.

## The workflow this enables

1. Open a video and its feature store.
2. Label the rallies you can see — once per recording.
3. Press **Score now** to get precision / recall / F1 against your labels.
4. Use **Next disagreement** to jump to each missed or invented rally, with the
   video, telemetry and the detector's stated reason all on screen.
5. Change a threshold and **Apply & Replay**.
6. **Compare vs previous** to see what actually changed, including which
   labelled rallies were gained or lost.

Because ground truth is independent of the detector, the same labelled
recording can then be used to compare `Pose`, `Pose + racket`, `Pose + ball` and
fused detectors on identical ground truth — which is how you find out whether a
more expensive model actually fixes the problem.

## Why replay is trustworthy

`DetectorReplayEngine` does not re-implement detection. It calls the same
`core.detection` analyzers the live pipeline calls, over features read from
disk. Regression tests run the extracted analyzer against a frozen copy of the
pre-refactor implementation and require identical output, and verify that
replaying a real run reproduces its segments exactly.

## What is not replayable

Serve and feed detection need per-arm pose features and a video-derived player
scale respectively. The feature store now persists every `FeatureEngine` key, so
both replay; feed mode uses the default scale bounds unless the calibrated ones
are supplied, because those come from a video pre-scan.
