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

## Adding a new case

1. Create a subdirectory under `cases/`
2. Add the video file named `video.mp4` (or any `.mp4`/`.mov`)
3. Create `annotations.json` with hand-labeled ground-truth intervals
4. Create `mode.txt` containing one of: `rally`, `serve`, `feed`

Tip: Use the `--plot` flag with the CLI to generate telemetry plots that help
identify correct segment boundaries when labeling ground truth.
