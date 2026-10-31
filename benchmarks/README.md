# Benchmark Cases

Place benchmark cases in this directory with the following structure:

```
benchmarks/cases/
    my_test_case/
        video.mp4           # Input video file (.mp4 or .mov)
        annotations.json    # Ground-truth rally intervals
        mode.txt            # "rally", "serve", or "feed" (default: rally)
```

## annotations.json format

```json
[
    {"start": 12.5, "end": 25.0, "label": "rally", "notes": "Baseline crosscourt, ~6 strokes"},
    {"start": 40.0, "end": 55.3, "label": "rally", "notes": "Short rally with drop shot"}
]
```

- `start` / `end`: time in seconds
- `label`: optional tag (e.g. "rally", "serve", "feed")
- `notes`: optional description of the segment

## Running benchmarks

```bash
python benchmarks/run_benchmark.py --cases benchmarks/cases --output benchmarks/report.json
```

Reports, per case and aggregated at IoU=0.3: precision, **recall (the primary
metric — missed rallies are the main failure mode)**, F1, missed-rally count,
false-rally count, boundary timing error in seconds, and processing runtime as
a multiple of video duration.

## Labelling ground truth

Use `label.py` rather than hand-writing JSON. It shows what a clip contains
without making you watch it end to end, and it keeps the annotation schema
consistent with `utils/evaluate.py`.

```bash
# 1. See what is actually in a video (one strip of timestamped frames)
python benchmarks/label.py --video clip.mov --contact-sheet /tmp/sheet.png

# 2. Create the case skeleton (copies the video, writes mode.txt)
python benchmarks/label.py --init-case my_rally --video clip.mov --mode rally

# 3. Label it interactively
python benchmarks/label.py --video clip.mov --case my_rally
#    label> 12.5-25        add a segment
#    label> list           show what you have
#    label> undo           drop the last one
#    label> quit           save and exit

# Or script it, and check what still needs work
python benchmarks/label.py --video clip.mov --case my_rally --add 12.5-25
python benchmarks/label.py --status
```

Video files are gitignored; `annotations.json` and `mode.txt` are tracked, so
labelling work is shared even though the footage is not.

**Label conservatively.** These annotations are the only ground truth the
project has — every recall/precision number is measured against them, so a
carelessly placed boundary silently corrupts every later comparison.

## Adding a new case manually

1. Create a subdirectory under `cases/`
2. Add the video file named `video.mp4` (or any `.mp4`/`.mov`)
3. Create `annotations.json` with hand-labeled ground-truth intervals
4. Create `mode.txt` containing one of: `rally`, `serve`, `feed`

Tip: Use the `--plot` flag with the CLI to generate telemetry plots that help
identify correct segment boundaries when labeling ground truth.

## Other analysis tools

```bash
# Compare the strict stroke gate against the additive coarse gate
python benchmarks/gate_ablation.py --videos clip1.mov clip2.mov

# Synthetic threshold-drift simulation
python benchmarks/evaluate_threshold.py
```
