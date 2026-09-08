#!/usr/bin/env python3
"""
Threshold Drift Simulation & Evaluation.

Generates synthetic arm-energy signals mimicking real tennis practice
patterns (rest → rally → rest → rally ...) and compares the old
SelfAdaptiveThreshold (median+MAD) against the new RobustAdaptiveThreshold
(P25+MAD+EMA). Produces comparison plots showing:

  1. Raw signal + smoothed signal
  2. Z-scored signal + both thresholds over time
  3. Threshold drift (ratio of threshold to floor) over time
  4. Detection comparison: which strokes each threshold catches

Usage:
    python benchmarks/evaluate_threshold.py [--output benchmarks/threshold_comparison.png]
"""

import argparse
import os
import sys

import numpy as np

# Allow running from project root
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from really_rally.core.features import (
    SelfAdaptiveThreshold,
    RobustAdaptiveThreshold,
    RollingZScoreNormalizer,
)


# ---------------------------------------------------------------------------
# Synthetic signal generator
# ---------------------------------------------------------------------------

def generate_synthetic_signal(
    duration_sec: float = 300.0,
    fps: float = 30.0,
    rest_level: float = 0.05,
    stroke_peak: float = 3.0,
    stroke_width: float = 0.15,
    rally_stroke_interval: float = 2.0,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic arm-energy signal with rest/rally patterns.

    Pattern:
      - 0-30s:   rest (warm-up, walking)
      - 30-120s: dense rally (~2s stroke interval, high activity)
      - 120-150s: rest (water break)
      - 150-240s: another dense rally
      - 240-300s: rest (cool-down)

    Each stroke is a Gaussian pulse. Between strokes there's residual
    activity (footwork, racket preparation) at ~0.2x stroke peak.

    Returns:
        1D array of arm energy values, one per frame.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(duration_sec * fps)
    t = np.arange(n_frames) / fps

    # Baseline activity (always some noise from MediaPipe even at rest)
    signal = np.full(n_frames, rest_level) + rng.normal(0, noise_std, n_frames)

    # Define rally intervals: (start_sec, end_sec, stroke_interval)
    rallies = [
        (30.0, 120.0, rally_stroke_interval),
        (150.0, 240.0, rally_stroke_interval * 0.7),  # denser second rally
    ]

    for rally_start, rally_end, interval in rallies:
        # Elevated baseline during rallies (footwork, preparation)
        rally_mask = (t >= rally_start) & (t < rally_end)
        signal[rally_mask] += 0.15 + rng.normal(0, noise_std * 2, rally_mask.sum())

        # Stroke pulses
        stroke_time = rally_start
        while stroke_time < rally_end:
            # Gaussian pulse centered at stroke_time
            pulse = stroke_peak * np.exp(-0.5 * ((t - stroke_time) / stroke_width) ** 2)
            signal += pulse
            # Jitter the interval slightly
            stroke_time += interval + rng.uniform(-0.3, 0.3)

    return np.maximum(signal, 0.0)


def ground_truth_strokes(signal: np.ndarray, fps: float, threshold: float = 1.5) -> np.ndarray:
    """Identify ground-truth stroke frames from the synthetic signal.

    A stroke is any frame where the raw signal exceeds the threshold.
    We merge adjacent detections within 0.3s into single strokes.
    """
    is_stroke = signal > threshold
    # Merge within 0.3s
    merge_frames = int(0.3 * fps)
    merged = is_stroke.copy()
    for i in range(1, len(merged)):
        if merged[i]:
            # Look back
            for j in range(max(0, i - merge_frames), i):
                if merged[j]:
                    merged[j:i] = True
                    break
    return merged


# ---------------------------------------------------------------------------
# Pipeline simulation
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_signal: np.ndarray,
    fps: float,
    threshold_cls,
    window_sec: float = 30.0,
    k: float = 4.0,
    min_floor: float = 0.5,
) -> dict:
    """Run the full analysis pipeline on a synthetic signal.

    Simulates: smoothing → Z-normalizer → adaptive threshold → detections.

    Returns dict with all intermediate signals for plotting.
    """
    effective_fps = fps  # assume PROCESS_EVERY_N_FRAMES=1 for simulation
    n = len(raw_signal)

    # Smooth (0.5s boxcar)
    window = int(effective_fps * 0.5)
    if window < 1:
        window = 1
    kernel = np.ones(window) / window
    smoothed = np.convolve(raw_signal, kernel, mode="same")

    # Z-score normalization
    z_normalizer = RollingZScoreNormalizer(window_sec, effective_fps)
    z_scores = z_normalizer.normalize_array(smoothed)

    # Adaptive threshold
    adaptive = threshold_cls(window_sec, effective_fps, k, min_floor)
    thresholds = np.zeros(n)
    for i in range(n):
        adaptive.update(z_scores[i])
        thresholds[i] = adaptive.get_threshold()

    # Detections: Z > threshold
    detections = z_scores > thresholds

    return {
        "raw": raw_signal,
        "smoothed": smoothed,
        "z_scores": z_scores,
        "thresholds": thresholds,
        "detections": detections,
        "threshold_name": threshold_cls.__name__,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_detection_metrics(
    gt: np.ndarray, detections: np.ndarray, fps: float
) -> dict:
    """Compute stroke-level precision/recall against ground truth.

    A detected stroke is a contiguous run of detection=True frames.
    A match requires temporal overlap within 0.3s.
    """
    merge_frames = int(0.3 * fps)

    def _extract_events(mask):
        events = []
        in_event = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_event:
                start = i
                in_event = True
            elif not val and in_event:
                # Extend end by merge_frames/2 for matching tolerance
                events.append((start, i))
                in_event = False
        if in_event:
            events.append((start, len(mask) - 1))
        return events

    gt_events = _extract_events(gt)
    pred_events = _extract_events(detections)

    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    for gi, (gs, ge) in enumerate(gt_events):
        for pi, (ps, pe) in enumerate(pred_events):
            if pi in matched_pred:
                continue
            # Check overlap
            if ps <= ge + merge_frames and pe >= gs - merge_frames:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    fp = len(pred_events) - len(matched_pred)
    fn = len(gt_events) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "total_gt": len(gt_events), "total_pred": len(pred_events),
    }


def compute_drift_metrics(thresholds: np.ndarray, min_floor: float) -> dict:
    """Compute metrics quantifying threshold drift away from the floor.

    Lower drift = better. Values near min_floor indicate a healthy
    resting baseline.
    """
    n = len(thresholds)
    # Drift: how far above the floor
    drift = thresholds - min_floor
    # Average drift over the full signal
    mean_drift = float(np.mean(drift))
    # Peak drift (worst case)
    peak_drift = float(np.max(drift))
    # Drift during known rest periods (first 30s and last 60s)
    rest_mask = np.zeros(n, dtype=bool)
    rest_mask[: int(30 * 30)] = True  # first 30s
    rest_mask[-int(60 * 30) :] = True  # last 60s
    rest_drift = float(np.mean(drift[rest_mask])) if rest_mask.any() else 0.0
    # Standard deviation of threshold (stability)
    threshold_std = float(np.std(thresholds))

    return {
        "mean_drift": round(mean_drift, 4),
        "peak_drift": round(peak_drift, 4),
        "rest_drift": round(rest_drift, 4),
        "threshold_std": round(threshold_std, 4),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results_old: dict, results_new: dict, gt: np.ndarray, fps: float, output_path: str):
    """Generate a 4-panel comparison plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    t = np.arange(len(results_old["raw"])) / fps

    fig, axs = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    # Panel 1: Raw + Smoothed signal
    axs[0].plot(t, results_old["raw"], color="#cccccc", linewidth=0.5, alpha=0.7, label="Raw signal")
    axs[0].plot(t, results_old["smoothed"], color="#1f77b4", linewidth=1.5, label="Smoothed (0.5s)")
    axs[0].set_ylabel("Arm Energy", fontweight="bold")
    axs[0].set_title("Synthetic Arm Energy Signal", fontweight="bold")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Z-scores + thresholds overlay
    axs[1].plot(t, results_old["z_scores"], color="#aaaaaa", linewidth=0.8, alpha=0.6, label="Z-score")
    axs[1].plot(t, results_old["thresholds"], color="#d62728", linewidth=1.5, linestyle="--", label="Old (median+MAD)")
    axs[1].plot(t, results_new["thresholds"], color="#2ca02c", linewidth=1.5, label="New (P25+MAD+EMA)")
    axs[1].axhline(y=0.5, color="#888888", linestyle=":", linewidth=0.8, alpha=0.5, label="Min floor (0.5)")
    axs[1].set_ylabel("Z-score / Threshold", fontweight="bold")
    axs[1].set_title("Z-Scored Signal vs Adaptive Thresholds", fontweight="bold")
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # Highlight rally regions
    for ax in axs[:2]:
        ax.axvspan(30, 120, alpha=0.05, color="red")
        ax.axvspan(150, 240, alpha=0.05, color="red")
    rally_patch = mpatches.Patch(color="red", alpha=0.08, label="Dense rally periods")
    axs[1].legend(loc="upper right", fontsize=8)

    # Panel 3: Threshold drift (ratio to floor)
    drift_old = results_old["thresholds"] / 0.5  # floor=0.5
    drift_new = results_new["thresholds"] / 0.5
    axs[2].plot(t, drift_old, color="#d62728", linewidth=1.5, label="Old threshold / floor")
    axs[2].plot(t, drift_new, color="#2ca02c", linewidth=1.5, label="New threshold / floor")
    axs[2].axhline(y=1.0, color="#888888", linestyle=":", linewidth=0.8, alpha=0.5, label="Floor (=1.0x)")
    axs[2].set_ylabel("Threshold / Floor", fontweight="bold")
    axs[2].set_title("Threshold Drift (higher = more drifted from resting baseline)", fontweight="bold")
    axs[2].legend(loc="upper right", fontsize=8)
    axs[2].grid(True, alpha=0.3)
    axs[2].axvspan(30, 120, alpha=0.05, color="red")
    axs[2].axvspan(150, 240, alpha=0.05, color="red")
    # Annotate drift metrics
    m_old = compute_drift_metrics(results_old["thresholds"], 0.5)
    m_new = compute_drift_metrics(results_new["thresholds"], 0.5)
    axs[2].text(0.02, 0.95,
                f"Old: mean drift={m_old['mean_drift']:.3f}, peak={m_old['peak_drift']:.3f}\n"
                f"New: mean drift={m_new['mean_drift']:.3f}, peak={m_new['peak_drift']:.3f}",
                transform=axs[2].transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Panel 4: Detection comparison
    axs[3].fill_between(t, 0, gt.astype(float), color="green", alpha=0.3, step="mid", label="Ground truth strokes")
    axs[3].fill_between(t, -1, results_old["detections"].astype(float) - 1, color="#d62728", alpha=0.3, step="mid", label="Old detections")
    axs[3].fill_between(t, -2, results_new["detections"].astype(float) - 2, color="#2ca02c", alpha=0.3, step="mid", label="New detections")
    axs[3].set_ylabel("Detection", fontweight="bold")
    axs[3].set_xlabel("Time (s)", fontweight="bold")
    axs[3].set_title("Stroke Detection Comparison", fontweight="bold")
    axs[3].set_ylim(-2.5, 1.5)
    axs[3].set_yticks([-1.5, 0.5])
    axs[3].set_yticklabels(["Detected", "GT"])
    axs[3].legend(loc="upper right", fontsize=8)
    axs[3].grid(True, alpha=0.3)

    # Detection metrics annotation
    m_det_old = compute_detection_metrics(gt, results_old["detections"], fps)
    m_det_new = compute_detection_metrics(gt, results_new["detections"], fps)
    axs[3].text(0.02, 0.95,
                f"Old: P={m_det_old['precision']:.3f}  R={m_det_old['recall']:.3f}  F1={m_det_old['f1']:.3f}  "
                f"(TP={m_det_old['tp']} FP={m_det_old['fp']} FN={m_det_old['fn']})\n"
                f"New: P={m_det_new['precision']:.3f}  R={m_det_new['recall']:.3f}  F1={m_det_new['f1']:.3f}  "
                f"(TP={m_det_new['tp']} FP={m_det_new['fp']} FN={m_det_new['fn']})",
                transform=axs[3].transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Threshold drift simulation & evaluation")
    parser.add_argument("--output", default="benchmarks/threshold_comparison.png", help="Output plot path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print("Generating synthetic arm-energy signal (5 min, 30 fps)...")
    fps = 30.0
    signal = generate_synthetic_signal(duration_sec=300.0, fps=fps, seed=args.seed)
    gt = ground_truth_strokes(signal, fps)
    print(f"  Signal: {len(signal)} frames, {np.count_nonzero(gt)} ground-truth stroke frames")

    print("\nRunning pipeline with old threshold (median+MAD)...")
    results_old = run_pipeline(signal, fps, SelfAdaptiveThreshold)

    print("Running pipeline with new threshold (P25+MAD+EMA)...")
    results_new = run_pipeline(signal, fps, RobustAdaptiveThreshold)

    # Metrics
    print("\n--- Threshold Drift Metrics (lower = better) ---")
    m_drift_old = compute_drift_metrics(results_old["thresholds"], 0.5)
    m_drift_new = compute_drift_metrics(results_new["thresholds"], 0.5)
    for key in m_drift_old:
        improvement = ""
        if m_drift_old[key] > 0:
            pct = (m_drift_old[key] - m_drift_new[key]) / m_drift_old[key] * 100
            improvement = f"  ({pct:+.0f}%)"
        print(f"  {key:20s}:  old={m_drift_old[key]:.4f}  new={m_drift_new[key]:.4f}{improvement}")

    print("\n--- Stroke Detection Metrics ---")
    m_det_old = compute_detection_metrics(gt, results_old["detections"], fps)
    m_det_new = compute_detection_metrics(gt, results_new["detections"], fps)
    for key in ["precision", "recall", "f1", "tp", "fp", "fn"]:
        print(f"  {key:12s}:  old={m_det_old[key]}  new={m_det_new[key]}")

    # Summary
    print("\n--- Summary ---")
    drift_reduction = (m_drift_old["mean_drift"] - m_drift_new["mean_drift"]) / m_drift_old["mean_drift"] * 100
    f1_improvement = (m_det_new["f1"] - m_det_old["f1"]) * 100
    print(f"  Mean threshold drift reduced by {drift_reduction:.0f}%")
    print(f"  F1 score change: {f1_improvement:+.1f} percentage points")
    if f1_improvement > 0:
        print("  Result: RobustAdaptiveThreshold improves stroke detection by avoiding drift")
    elif abs(f1_improvement) < 1:
        print("  Result: Detection quality preserved, but drift significantly reduced")

    if not args.no_plot:
        plot_comparison(results_old, results_new, gt, fps, args.output)


if __name__ == "__main__":
    main()
