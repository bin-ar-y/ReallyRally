#!/usr/bin/env python3
"""Stroke-gate ablation: strict AND-gate (pre-Phase C) vs additive coarse score.

Replays the per-frame telemetry written by ``debug_metrics.csv`` and counts
contiguous stroke *events* under two gates:

  strict  : z_arm > t_arm AND z_whip > t_whip AND hip_vel > HIP_MIN
            AND spatial_var >= SPATIAL_MIN          (the pre-Phase-C gate)
  coarse  : coarse_score >= COARSE_CANDIDATE_THRESHOLD  (the Phase C gate)

The strict gate is a conjunction, so a real stroke is discarded whenever any
single signal is weak (whip ratio collapses under body rotation; hip velocity
is near zero on a planted-foot hit). The coarse gate is additive, so one
strong signal suffices. The difference in event counts is the recall headroom
Phase C buys at the stroke level, before any temporal or object verification.

Usage:
    python benchmarks/gate_ablation.py --videos clip1.mov clip2.mov
    python benchmarks/gate_ablation.py --videos serveclips/*.mov --limit 5
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from typing import Dict, List

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from really_rally.core.config import (  # noqa: E402
    COARSE_CANDIDATE_THRESHOLD,
    HIP_VELOCITY_STROKE_THRESHOLD,
    PLAYER_HEIGHT_MAX,
    PLAYER_HEIGHT_MIN,
    PROCESS_EVERY_N_FRAMES,
    SPATIAL_VARIANCE_STROKE_MIN,
)
from really_rally.processors.base import TennisRallyProcessor  # noqa: E402


def count_events(mask: List[bool]) -> int:
    """Count contiguous True runs; a stroke event spans several candidate frames.

    Note: this under-counts the *coarse* gate. Because the coarse gate fires on
    a denser set of frames, adjacent strokes separated by a brief lull merge
    into a single run. Prefer :func:`stroke_peaks` for gate comparison.
    """
    return sum(1 for i, v in enumerate(mask) if v and (i == 0 or not mask[i - 1]))


def stroke_peaks(values: List[float], fps: float, min_separation_sec: float = 0.4,
                 floor_frac: float = 0.25) -> List[int]:
    """Locate stroke peaks as prominent local maxima of the activity signal.

    Used as a label-free proxy for "a stroke happened here": any well-separated
    local maximum above a fraction of the clip's peak activity. Gate recall is
    then measured per peak rather than per frame, which does not reward a gate
    merely for firing on more frames.
    """
    n = len(values)
    if n == 0:
        return []
    peak = max(values)
    if peak <= 0:
        return []
    floor = peak * floor_frac
    min_sep = max(1, int(min_separation_sec * fps))

    peaks: List[int] = []
    last = -min_sep
    for i in range(1, n - 1):
        v = values[i]
        if v < floor or v < values[i - 1] or v < values[i + 1]:
            continue
        if i - last < min_sep:
            # Same stroke: keep the stronger of the two peaks.
            if v > values[peaks[-1]]:
                peaks[-1] = i
                last = i
            continue
        peaks.append(i)
        last = i
    return peaks


def recall_around_peaks(mask: List[bool], peaks: List[int], fps: float,
                        tolerance_sec: float = 0.25) -> float:
    """Fraction of stroke peaks with a gate fire within +/- tolerance."""
    if not peaks:
        return float("nan")
    tol = max(1, int(tolerance_sec * fps))
    hit = 0
    for p in peaks:
        lo, hi = max(0, p - tol), min(len(mask), p + tol + 1)
        if any(mask[lo:hi]):
            hit += 1
    return hit / len(peaks)


def run_case(video: str, mode: str = "rally") -> Dict:
    """Run the pipeline on one video and return gate-ablation statistics."""
    out = tempfile.mkdtemp(prefix="rr_ablation_")
    t0 = time.time()
    proc = TennisRallyProcessor(video, out, keep_clips=False, generate_plot=False)
    clips = proc.process_video()
    elapsed = time.time() - t0

    metrics_path = os.path.join(out, "debug_metrics.csv")
    if not os.path.isfile(metrics_path):
        return {"video": os.path.basename(video), "error": "no debug_metrics.csv"}

    with open(metrics_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("Valid_Pose") == "1"]

    if not rows:
        return {"video": os.path.basename(video), "error": "no valid pose frames"}

    strict, coarse = [], []
    for r in rows:
        z_arm, t_arm = float(r["Z_Arm"]), float(r["T_Arm"])
        z_whip, t_whip = float(r["Z_Whip"]), float(r["T_Whip"])
        hip = float(r["Hip_Vel"])
        spatial = float(r["Spatial_Var"])
        # The pipeline's real gate is only reachable on a valid, in-range pose.
        # Applying the same distance gate here keeps `strict` a true subset of
        # `coarse` (which already encodes it via Coarse_Score).
        height = float(r["Player_Height"])
        in_range = PLAYER_HEIGHT_MIN <= height <= PLAYER_HEIGHT_MAX
        strict.append(
            in_range
            and z_arm > t_arm and z_whip > t_whip
            and hip > HIP_VELOCITY_STROKE_THRESHOLD
            and spatial >= SPATIAL_VARIANCE_STROKE_MIN
        )
        coarse.append(float(r["Coarse_Score"]) >= COARSE_CANDIDATE_THRESHOLD)

    n = len(rows)
    effective_fps = proc.fps / PROCESS_EVERY_N_FRAMES
    activity = [float(r["Smoothed_Arm"]) for r in rows]
    peaks = stroke_peaks(activity, effective_fps)

    return {
        "video": os.path.basename(video)[:44],
        "frames": n,
        "strict_frames": sum(strict),
        "coarse_frames": sum(coarse),
        "strict_events": count_events(strict),
        "coarse_events": count_events(coarse),
        "missed_by_strict": sum(1 for c, s in zip(coarse, strict) if c and not s),
        "peaks": len(peaks),
        "strict_peak_recall": recall_around_peaks(strict, peaks, effective_fps),
        "coarse_peak_recall": recall_around_peaks(coarse, peaks, effective_fps),
        "candidate_rate": sum(coarse) / n if n else 0.0,
        "segments": len(clips),
        "elapsed": elapsed,
        "duration": proc.total_frames / proc.fps if proc.fps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True, help="Video files to analyse")
    ap.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    args = ap.parse_args()

    videos = args.videos[: args.limit] if args.limit else args.videos
    results = []
    for v in videos:
        if not os.path.isfile(v):
            print(f"[skip] not found: {v}")
            continue
        print(f"\n=== {os.path.basename(v)} ===")
        res = run_case(v)
        results.append(res)
        if "error" in res:
            print(f"  error: {res['error']}")
            continue
        print(f"  duration {res['duration']:.1f}s  processed in {res['elapsed']:.1f}s"
              f"  ({res['elapsed']/res['duration']:.2f}x realtime)")
        print(f"  strict gate : {res['strict_frames']:4d}/{res['frames']} frames"
              f"  |  {res['strict_events']:3d} runs"
              f"  |  peak recall {res['strict_peak_recall']:.1%}")
        print(f"  coarse gate : {res['coarse_frames']:4d}/{res['frames']} frames"
              f"  |  {res['coarse_events']:3d} runs"
              f"  |  peak recall {res['coarse_peak_recall']:.1%}")
        print(f"  stroke peaks: {res['peaks']}   candidate rate {res['candidate_rate']:.1%}"
              f"   frames strict discarded: {res['missed_by_strict']}")
        print(f"  -> {res['segments']} rally segment(s) exported")

    ok = [r for r in results if "error" not in r]
    if ok:
        print("\n" + "=" * 78)
        print("  AGGREGATE")
        print("=" * 78)
        ts = sum(r["strict_events"] for r in ok)
        tc = sum(r["coarse_events"] for r in ok)
        td = sum(r["duration"] for r in ok)
        te = sum(r["elapsed"] for r in ok)
        print(f"  videos            : {len(ok)}")
        print(f"  total video time  : {td:.1f}s")
        print(f"  total process time: {te:.1f}s  ({te/td:.2f}x realtime)" if td else "")
        sr = [r["strict_peak_recall"] for r in ok if r["peaks"]]
        cr = [r["coarse_peak_recall"] for r in ok if r["peaks"]]
        print(f"  stroke peaks      : {sum(r['peaks'] for r in ok)}")
        print(f"  mean peak recall  : strict {sum(sr)/len(sr):.1%}  ->  coarse {sum(cr)/len(cr):.1%}"
              if sr else "  (no stroke peaks found)")
        print("=" * 78)


if __name__ == "__main__":
    main()
