# Changelog

All notable changes to ReallyRally. Newest first.

The detection work below was driven by measurement rather than assumption.
Where a hypothesis was tested and **rejected**, it is recorded that way, so the
reasoning is not silently lost.

---

## Unreleased — output layout and export reliability

### Fixed — clip export timed out on long sessions

`export_clips` passed `-ss`/`-to` *after* `-i`, which makes them output options:
ffmpeg decoded the source **from frame 0** and discarded everything before the
clip. Export cost therefore grew with the clip's position in the recording.

Measured on a 146s serve clip, exporting a 2s segment:

| Clip starts at | Before | After |
|---|---|---|
| 5s | 1.31s | 0.87s |
| 40s | 4.75s | 0.92s |
| 90s | 9.33s | 0.86s |
| 140s | 13.94s | 0.83s |

Against the hardcoded 120s timeout, a serve late in a long session could not
finish — which is why `serve_032` exported and `serve_033` reported
"ffmpeg timed out". Six clips spread across `serveclips/IMG_0901.mov`
(including one at 142s) now export in 6.5s total instead of ~21s for the single
latest one, and all six come out at exactly 2.601s, so input seeking is
frame-accurate.

`-ss` now precedes `-i` and `-t <duration>` replaces `-to <end>`, which is
unambiguous where `-to` depends on where the seek landed. The timeout moved to
`EXPORT_TIMEOUT_SEC` (180s) and its message now says what the clip was and that
export continued, rather than leaving the user to guess.

### Changed — results are grouped per recording

Per-video artifacts used to land flat in the output directory, mixed with the
merged video and a clips folder, so a batch produced dozens of prefixed files
with no structure. Now:

```
<output>/<mode>/<video_stem>/     # rally | serve | feed
    serve.mp4, serve.csv, features.csv, debug_metrics.csv,
    telemetry.png, serve_events.csv, labels.csv, predictions.csv,
    experiments.csv, clips/
<output>/<mode>/Merged_*.mp4      # batch merge, multi-video runs only
<output>/<mode>/manifest.csv      # batch clip index
```

- The folder is named after the extraction (`rally` / `serve` / `feed`) rather
  than `rallies` / `serves` / `feeds`, and the per-file stem prefix is gone —
  each recording has its own folder, so it was redundant.
- The CLI default output is now the current directory, giving `./serve/IMG_1165/`
  instead of a date-stamped bundle that had to be opened to see what was in it.
- `studio/results_folder.py` resolves such a folder back into a session, and the
  studio gained **Open results folder…** / `--results`. Pointing at a folder with
  several recordings asks which one rather than picking arbitrarily; pointing at
  a `clips/` subfolder resolves up to the recording that owns it, since an
  individual clip does not share the recording's timeline.
- `docs/STUDIO.md` and the README document the layout.

No compatibility shim: the old flat paths are no longer written. `cli.py`,
`gui.py`, `benchmarks/run_benchmark.py` and `benchmarks/gate_ablation.py` were
updated, and the latter two now ask the processor for its paths rather than
reconstructing the layout, so they cannot drift again.

---

## Unreleased — Debug + Label Studio

An interactive workspace for synchronised debugging, ground-truth annotation and
interactive detector tuning. The guiding constraint: **ground truth must never
be written by the detector.**

### Added — feature store

- **`<video>_features.csv`** persists *every* `FeatureEngine` key per sampled
  frame, not the 18-column subset `debug_metrics.csv` keeps. This is what makes
  replay possible: the previous telemetry dump could not reproduce serve or feed
  detection at all, because the per-arm pose features were never written down.
- `Frame` is stored as an integer and is the source of truth for time. Motion
  segment boundaries are recovered from gaps in it, so the replay engine can
  rebuild per-segment detector state without a separate boundary column.

### Added — Debug + Label Studio (`studio/`)

- Time-synchronised workspace: dragging the playhead moves the video frame,
  telemetry cursor, current-frame inspector and label track together.
  Clicking telemetry seeks the video. Seeks are collapsed to the newest request,
  so scrubbing cannot queue a backlog of stale frame decodes.
- **Current-frame inspector** rendering values by role — raw, threshold, derived
  score, decision, state — so `0.023` is never shown without the `0.249` bar it
  failed to clear and the verdict that followed.
- **Interactive telemetry** built from whatever keys the store contains; panels
  are derived from the data, so a future racket/ball column appears with no UI
  change.
- **Annotation** with an interval + point vocabulary, autosaved to
  `*_labels.csv`, with import/export of the existing `annotations.json` format.
- **Config console** splitting parameters into replayable and
  requires-re-analysis, and refusing (loudly) to apply the latter from cache.
- **Evaluation** reusing `utils/evaluate.py`, plus a disagreement navigator that
  jumps to each missed and invented rally with the detector's stated reason.
- **Experiment history** in `*_experiments.csv`, with a comparison that reports
  which labelled rallies were gained and lost, not just the aggregate score.

### Added — launch surface

- `really-rally studio [--video V] [--features F]` — a new CLI subcommand;
  also reachable as `python run.py studio`.
- `really-rally-studio` — a direct console script. It is declared in
  `pyproject.toml`, so it only appears after `pip install -e .` is re-run; an
  environment installed before the studio existed will not have it, which is why
  the subcommand above is the documented entry point.
- `pillow` is now a declared dependency (see Fixed).

### Changed — detection extracted for replay

`TennisRallyProcessor.analyze_segment_data` no longer contains detection logic.
It is now a thin adapter over `core.detection.RallyAnalyzer`, which the studio's
replay engine calls with the same arguments. A re-implementation would have
drifted, so there is exactly one implementation and two callers.

- Cross-segment adaptive state moved from processor attributes into
  `AdaptiveState`, owned by an analyzer instance. The old class-level buffers
  made it possible to leak thresholds between videos.
- `CoarseCandidateDetector.score()`'s evidence list is no longer discarded; it
  is persisted as `Fired_Signals` and is what makes a decision explainable.
- `ServeAnalyzer` / `FeedAnalyzer` port the serve and feed paths the same way.
  Their deliberate quirks are preserved: serve mode reads thresholds *before*
  updating the buffers and does not pad rest gaps, unlike the rally path.
- Verified by parity test: the extracted analyzer is compared against a frozen
  copy of the pre-refactor implementation and must match exactly, including
  segments, thresholds, scores and states.

### Fixed

- **`generate_evaluation_plot` silently produced nothing from a CSV.**
  `telemetry.py` read lower-case in-memory keys while `cli.py` fed it
  `csv.DictReader` rows carrying `Title_Case` headers, so every row was skipped
  and the function returned after printing a warning. Both plot functions now
  route rows through `feature_store.normalize_row`.
- **A label deletion could destroy an interval.** Backspace deleted *every*
  label touching the playhead, so removing a stroke point that sat on a rally
  boundary also deleted the rally. It now removes one label, preferring points
  over containing intervals.
- **`pillow` was an undeclared dependency**, present only transitively via
  customtkinter while the studio imports it directly. A clean install would have
  failed on the first frame render; it is now listed in `pyproject.toml`.

### Notes

- 480 tests pass, up from 245. The new suites cover the feature store, config
  snapshot, decision signals, frame inspector, annotation, replay, config
  parsing, evaluation, and the studio's thread-safety invariant.
- The studio's threading contract is enforced the same way the extraction GUI's
  is: a source-lint test asserts `after()` appears only in the main-thread
  poller, so a worker thread cannot start touching Tk later.

---

## Unreleased — detection upgrade, output fixes, tooling

### Fixed — output files (data loss)

- **Batch runs overwrote each other's telemetry.** Every video wrote
  `debug_metrics.csv`, `movement_telemetry.png` and `serve_events.csv` into the
  shared output directory, so processing N videos kept only the last one's
  data. All per-video artifacts are now prefixed with the source video stem:
  `IMG_0830_debug_metrics.csv`, `IMG_0830_telemetry.png`, `IMG_0830_serve_events.csv`.
- **Clip manifests accumulated duplicate rows forever.** `export_clips` opened
  the CSV in append mode whenever it existed, and clip filenames carry a fresh
  uuid on every run — so re-running the same video appended an identical row
  every time and the CSV ended up describing clips that no longer existed.
  Verified: one video run twice produced 2 rows before, 1 row now. Manifests are
  written fresh by `write_clip_manifest()`; the GUI writes one per batch.
- **The telemetry plot was silently absent.** The GUI checkbox defaulted to
  unchecked, so an ordinary run produced no plot at all. It now defaults to on,
  and `process_video()` prints a manifest of every file written and where.
- **`transfer_metadata` never actually transferred anything.** The method ended
  at a `# Replace original` comment, so metadata was never applied and an orphan
  `_meta.mp4` was left behind on every run. Now applies via `os.replace`,
  cleans up on failure, and is verified: merged clips carry the source
  `creation_time`.

### Fixed — serve mode

- **Handedness could flip between serves.** A player serves with one hand for a
  whole session; per-serve inference sometimes read the opposite hand from a
  mis-associated toss/contact pair. The dominant hand is now chosen once per
  video by score-weighted total evidence and applied to every event.
  `SERVE_FORCED_HANDEDNESS` overrides it; `SERVE_LOCK_HANDEDNESS` disables it.
- **Serve telemetry columns were empty.** `debug_metrics.csv` reported
  `Smoothed_Arm = 0.0000` on every frame and a blank `State` column in serve
  mode, which reads as "no activity" even when detection works. Serve mode now
  writes the same columns as rally mode, with `State`/`Candidate` marking
  detected serve intervals.

### Changed — detection (coarse-to-fine)

- **The strict stroke gate was replaced by an additive coarse detector.** The
  old gate required arm energy AND whip ratio AND hip velocity AND spatial
  variance simultaneously, so a real stroke was discarded whenever any single
  signal was weak — a body-rotation groundstroke has `whip_ratio` near zero,
  and a planted-foot hit has near-zero hip velocity. `CoarseCandidateDetector`
  accumulates weighted evidence instead: one strong signal opens a candidate.
- **Hysteresis in the rally state machine.** `Idle → Candidate → Active →
  Cooldown`, where starting a rally requires two strokes within
  `RALLY_START_GAP_SEC` (7s) but continuing tolerates `RALLY_CONTINUE_GAP_SEC`
  (11s). A single missed stroke no longer truncates a segment.
  *Measured (label-free proxy, 6 clips): mean stroke-peak recall 40.8% → 54.2%.*
- **Raw-space floors stop Z-score noise amplification.** The rolling Z-score is
  scale-free, so a numerically negligible arm energy (~0.0007, versus ~0.10 for
  a real swing) divided by a small window std reached z > 2 and opened
  candidates on noise. Evidence now also requires `COARSE_MIN_ABS_ARM` /
  `COARSE_MIN_ABS_WHIP`. *A/B over 6 clips: candidate frames −6.0%, exported
  segment count unchanged on every clip.*
- **Short MediaPipe dropouts are bridged.** The pose is dropped exactly at ball
  contact, where the swing is fastest and most blurred, and resetting
  `prev_landmarks` there reported zero velocity at the most informative moment.
  Displacement is now measured across gaps up to `POSE_DROPOUT_BRIDGE_STEPS`
  and divided by the steps spanned. *Measured: 9.1% of frames lose pose on
  broadcast footage in 3 runs, 2 of which are now bridged.*

### Added — object evidence and fusion (architecture only, no model installed)

- `core/objects.py` — `ObjectDetector` contract, `NullObjectDetector`,
  `BallTracker`, and a registry. Racket/ball confidence defaults to **neutral,
  never absent**, so a missing or broken model cannot reduce recall.
- `core/fusion.py` — `EvidenceFusion` combining pose/racket/ball/temporal with
  configurable weights, encoding: strong pose + racket without ball is still
  likely a rally; ball alone is insufficient; "detector ran, found nothing"
  scores below "no detector" but never vetoes.
- `verify_candidate_windows()` — optional Pass 3 running object inference
  **only** inside candidate windows, downscaled. A no-op until a backend is
  registered, so the pose-only pipeline is unchanged today.

### Added — evaluation and tooling

- `benchmarks/label.py` — contact sheets to identify clip content without
  watching it end to end, `--init-case` scaffolding, interactive and scripted
  labelling, `--status` to track coverage.
- `benchmarks/gate_ablation.py` — replays telemetry to compare the old strict
  gate against the new coarse gate.
- Evaluation now reports **boundary timing error** (seconds) alongside P/R/F1 —
  a detector can find the right rallies but clip them badly — and **runtime as a
  multiple of video duration**.

### Fixed — GUI responsiveness

- **Worker thread updated Tk widgets directly** via `self.after(0, ...)`, which
  is not thread-safe. Updates and dialogs now go through a queue drained by a
  main-thread poller, with a source-level test asserting no `after()` call
  outside the poller.
- **The MediaPipe graph was rebuilt per video and per motion segment**,
  dominating startup and leaving the UI looking frozen before the first
  progress event. It is now cached process-wide.
- **Start/Stop gave no immediate feedback.** Start now flips every control and
  shows "Starting…" before any work begins; Stop acknowledges instantly,
  disables itself, and cancels cooperatively. Double-clicks can no longer launch
  a second worker.
- **ffmpeg calls were uninterruptible.** Export now honours `stop_event` between
  clips and the concat merge has a timeout, so Stop cannot hang. The blocking
  `subprocess.call(["open", ...])` is now `Popen`.

### Fixed — other

- Duplicate `_mask_to_segments` removed (`_mask_to_indices` is the live one).
- The unguarded `ssl._create_unverified_context` assignment — which silently
  disabled TLS verification process-wide — is now behind
  `REALLY_RALLY_ALLOW_INSECURE_SSL`.
- README contained a merge artefact (duplicate header and a `=======` marker).

### Investigated and rejected

- **"MAD baseline drift causes missed strokes" (the original hypothesis) was
  not supported.** Measurement showed the raw signal and the Z-score move in
  *opposite* directions during dense activity, while the adaptive threshold
  falls in step with the Z-score — so drift was not causing misses. The real
  defect was the inverse (noise amplification, fixed above).
- **Serve raise thresholds are not scale-normalized.** `SERVE_TOSS_ABOVE_SHOULDER_MARGIN`
  and the `/0.15` raise divisors use raw image-normalized units while the rest
  of the project is bbox-normalized, so they were suspected of breaking at
  different camera distances. Measured on `022626_Merged_Serves.MOV`, making
  them scale-relative changes the contact count only 3→3 and 7→8 — not the
  bottleneck, so left unchanged.
- **`MP_MODEL_COMPLEXITY` was not retuned.** Profiling showed MediaPipe pose is
  63% of runtime, which makes the complexity setting the obvious speed knob —
  but the mediapipe wheel bundles only the `full` (complexity 1) weights;
  lite/heavy download on first use and that download fails behind TLS
  inspection. The speed comparison could not be measured, so the default stands.

### Known limitations

- **No object detector is installed**, so the pose / +racket / +ball ablation is
  pending. A COCO-pretrained model covers both `tennis racket` and
  `sports ball`, so no custom training is needed.
- **No labelled ground truth yet.** `benchmarks/cases/demo_serve_practice/`
  ships with empty annotations, so every recall figure above is a label-free
  proxy. `benchmarks/label.py` exists to close this gap.
- **Serve detection returns 0 events on some footage.**
  `022626_Merged_Serves.MOV` yields 35 toss candidates but only 1 contact match:
  just 60 of 3071 frames have a fast wrist, and only 10 of those coincide with a
  raised arm. The toss side of the FSM is over-firing while the contact side is
  starved. Needs labelled data before retuning.
