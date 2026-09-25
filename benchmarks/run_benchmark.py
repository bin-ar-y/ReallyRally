#!/usr/bin/env python3
"""
Systematic Rally Detection Benchmark Runner.

Discovers annotated test cases, runs detection across modes, computes
Precision/Recall/F1 at multiple IoU thresholds, and produces a
structured JSON report.

Directory structure expected:
    benchmarks/cases/
        case_01_rally/
            video.mp4          # input video
            annotations.json   # ground-truth intervals
            mode.txt           # "rally", "serve", or "feed"
        case_02_serve/
            ...

Usage:
    python benchmarks/run_benchmark.py --cases benchmarks/cases --output report.json
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src is on path
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from really_rally.utils.evaluate import (
    load_ground_truth,
    load_predictions,
    compute_metrics,
)
from really_rally.processors.base import TennisRallyProcessor
from really_rally.processors.serve import ServePracticeProcessor
from really_rally.processors.feed import FeedPracticeProcessor


IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]


def discover_cases(cases_dir: str) -> List[dict]:
    """Find all benchmark cases under cases_dir.

    Each case is a subdirectory containing video.mp4 + annotations.json.
    """
    cases = []
    root = Path(cases_dir)
    if not root.is_dir():
        print(f"Error: cases directory not found: {cases_dir}")
        return cases

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        video_files = list(case_dir.glob("*.mp4")) + list(case_dir.glob("*.mov"))
        annotation_file = case_dir / "annotations.json"
        mode_file = case_dir / "mode.txt"

        if not video_files:
            print(f"  [skip] {case_dir.name}: no video file found")
            continue
        if not annotation_file.is_file():
            print(f"  [skip] {case_dir.name}: no annotations.json")
            continue

        mode = "rally"
        if mode_file.is_file():
            mode = mode_file.read_text().strip().lower()

        cases.append({
            "name": case_dir.name,
            "video_path": str(video_files[0]),
            "gt_path": str(annotation_file),
            "mode": mode,
        })

    return cases


def run_detection(case: dict, output_dir: str) -> tuple:
    """Run the appropriate processor on a video.

    Returns (csv_path_or_None, timing_dict). Timing is reported separately from
    detection quality so a change that improves recall at an unacceptable
    slowdown is visible rather than silently accepted.
    """
    video_path = case["video_path"]
    mode = case["mode"]

    processor_cls = {
        "rally": TennisRallyProcessor,
        "serve": ServePracticeProcessor,
        "feed": FeedPracticeProcessor,
    }.get(mode, TennisRallyProcessor)

    print(f"    Running {mode} detection on {os.path.basename(video_path)}...")
    t0 = time.time()

    try:
        processor = processor_cls(video_path, output_dir, keep_clips=False)
        fps = processor.fps or 0.0
        duration = (processor.total_frames / fps) if fps else 0.0
        processor.process_video()
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s ({duration:.1f}s video)")

        timing = {
            "seconds": round(elapsed, 2),
            "video_seconds": round(duration, 2),
            "realtime_factor": round(elapsed / duration, 3) if duration else None,
        }

        # Find the output CSV
        csv_name = {"rally": "rallies.csv", "serve": "serves.csv", "feed": "feeds.csv"}.get(mode, "rallies.csv")
        csv_path = os.path.join(output_dir, csv_name)
        if os.path.isfile(csv_path):
            return csv_path, timing
        print(f"    Warning: no {csv_name} produced (no segments detected)")
        return None, timing
    except Exception as e:
        print(f"    Error: {e}")
        return None, {}


def evaluate_case(case: dict, csv_path: str, timing: Optional[Dict] = None) -> Dict:
    """Compute metrics at all IoU thresholds for one case."""
    gt = load_ground_truth(case["gt_path"])
    pred = load_predictions(csv_path)

    results = {"name": case["name"], "mode": case["mode"], "num_gt": len(gt),
               "num_pred": len(pred), "timing": timing or {}, "thresholds": {}}

    for iou in IOU_THRESHOLDS:
        metrics = compute_metrics(gt, pred, iou)
        results["thresholds"][f"iou_{iou:.1f}"] = {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "mean_iou": round(metrics["mean_iou"], 4),
            "boundary_start_error_sec": round(metrics["boundary_start_error_sec"], 3),
            "boundary_end_error_sec": round(metrics["boundary_end_error_sec"], 3),
        }

    return results


def print_summary(all_results: List[Dict]):
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)

    for r in all_results:
        best = r["thresholds"]["iou_0.3"]
        print(f"\n  {r['name']}  [{r['mode']}]  (GT={r['num_gt']}, Pred={r['num_pred']})")
        print(f"    IoU=0.3 →  P={best['precision']:.3f}  R={best['recall']:.3f}  F1={best['f1']:.3f}  |  TP={best['tp']} FP={best['fp']} FN={best['fn']}")
        print(f"    Boundary err → start {best['boundary_start_error_sec']:.2f}s, "
              f"end {best['boundary_end_error_sec']:.2f}s")
        t = r.get("timing") or {}
        if t.get("realtime_factor") is not None:
            print(f"    Runtime → {t['seconds']:.1f}s for {t['video_seconds']:.1f}s video "
                  f"({t['realtime_factor']:.2f}x realtime)")

    # Aggregate
    scored = [r for r in all_results if r.get("thresholds")]
    if scored:
        avg_f1 = sum(r["thresholds"]["iou_0.3"]["f1"] for r in scored) / len(scored)
        avg_recall = sum(r["thresholds"]["iou_0.3"]["recall"] for r in scored) / len(scored)
        avg_prec = sum(r["thresholds"]["iou_0.3"]["precision"] for r in scored) / len(scored)
        total_fn = sum(r["thresholds"]["iou_0.3"]["fn"] for r in scored)
        total_fp = sum(r["thresholds"]["iou_0.3"]["fp"] for r in scored)
        print("\n  --- Aggregate (IoU=0.3) ---")
        print(f"  Cases: {len(scored)}")
        print(f"  Avg Precision: {avg_prec:.3f}")
        print(f"  Avg Recall   : {avg_recall:.3f}   (primary metric)")
        print(f"  Avg F1       : {avg_f1:.3f}")
        print(f"  Missed rallies (FN): {total_fn}    False rallies (FP): {total_fp}")

        rts = [r["timing"]["realtime_factor"] for r in scored
               if (r.get("timing") or {}).get("realtime_factor")]
        if rts:
            print(f"  Avg runtime  : {sum(rts)/len(rts):.2f}x realtime")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ReallyRally Benchmark Runner")
    parser.add_argument("--cases", default="benchmarks/cases", help="Path to benchmark cases directory")
    parser.add_argument("--output", default="benchmarks/report.json", help="Path for JSON report output")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary output directories")
    args = parser.parse_args()

    cases = discover_cases(args.cases)
    if not cases:
        print("No benchmark cases found.")
        print(f"Expected structure: {args.cases}/<case_name>/{{video.mp4, annotations.json, mode.txt}}")
        return

    print(f"\nFound {len(cases)} benchmark case(s):")
    for c in cases:
        print(f"  - {c['name']} [{c['mode']}]")

    all_results = []

    for case in cases:
        print(f"\n--- {case['name']} ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, timing = run_detection(case, tmpdir)

            if csv_path:
                result = evaluate_case(case, csv_path, timing)
                all_results.append(result)
            else:
                all_results.append({
                    "name": case["name"],
                    "mode": case["mode"],
                    "num_gt": len(load_ground_truth(case["gt_path"])),
                    "num_pred": 0,
                    "thresholds": {},
                    "timing": timing,
                    "error": "No predictions produced",
                })

    # Write report
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "iou_thresholds": IOU_THRESHOLDS,
        "num_cases": len(cases),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {args.output}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
