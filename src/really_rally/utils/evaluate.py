"""
Rally Detection Evaluation Module.

Compares predicted rally segments against ground-truth annotations
using temporal Intersection-over-Union (t-IoU) matching, and reports
Precision, Recall, and F1 score.

Usage:
    python -m really_rally.utils.evaluate \
        --gt annotations.json \
        --pred rallies.csv \
        [--iou 0.3]

Ground-truth JSON format:
    [
        {"start": 12.5, "end": 25.0},
        {"start": 40.0, "end": 55.3},
        ...
    ]
"""

import json
import csv
import argparse
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_ground_truth(json_path: str) -> List[Tuple[float, float]]:
    """Load ground-truth rally intervals from a JSON file.
    
    Expected format: list of objects with 'start' and 'end' keys (seconds).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    segments = []
    for entry in data:
        s = float(entry['start'])
        e = float(entry['end'])
        if e > s:
            segments.append((s, e))
    
    segments.sort(key=lambda x: x[0])
    return segments


def load_predictions(csv_path: str) -> List[Tuple[float, float]]:
    """Load predicted rally intervals from the rallies.csv export.
    
    Expects columns: Source Video, Clip, Start Time, End Time, Duration
    """
    segments = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return segments
        
        # Find column indices by header name
        header_lower = [h.strip().lower() for h in header]
        try:
            start_idx = header_lower.index('start time')
            end_idx = header_lower.index('end time')
        except ValueError:
            raise ValueError(
                f"CSV must contain 'Start Time' and 'End Time' columns. "
                f"Found: {header}"
            )
        
        for row in reader:
            if len(row) > max(start_idx, end_idx):
                s = float(row[start_idx])
                e = float(row[end_idx])
                if e > s:
                    segments.append((s, e))
    
    segments.sort(key=lambda x: x[0])
    return segments


# ---------------------------------------------------------------------------
# Temporal IoU
# ---------------------------------------------------------------------------

def temporal_iou(seg_a: Tuple[float, float], seg_b: Tuple[float, float]) -> float:
    """Compute temporal Intersection over Union between two time intervals.
    
    t-IoU = intersection_duration / union_duration
    
    Args:
        seg_a: (start, end) in seconds
        seg_b: (start, end) in seconds
    
    Returns:
        IoU value in [0.0, 1.0]
    """
    inter_start = max(seg_a[0], seg_b[0])
    inter_end = min(seg_a[1], seg_b[1])
    intersection = max(0.0, inter_end - inter_start)
    
    if intersection == 0.0:
        return 0.0
    
    union_start = min(seg_a[0], seg_b[0])
    union_end = max(seg_a[1], seg_b[1])
    union = union_end - union_start
    
    if union <= 0.0:
        return 0.0
    
    return intersection / union


# ---------------------------------------------------------------------------
# Segment Matching
# ---------------------------------------------------------------------------

def match_segments(
    gt_segments: List[Tuple[float, float]],
    pred_segments: List[Tuple[float, float]],
    iou_threshold: float = 0.3
) -> Dict:
    """Greedy bipartite matching of predictions to ground-truth by t-IoU.
    
    Each ground-truth segment is matched to at most one prediction (the one
    with the highest t-IoU above the threshold), and vice versa.
    
    Args:
        gt_segments:    List of (start, end) ground-truth intervals.
        pred_segments:  List of (start, end) predicted intervals.
        iou_threshold:  Minimum t-IoU to count as a match.
    
    Returns:
        Dictionary with keys:
            'matches':       List of (gt_idx, pred_idx, iou) tuples
            'unmatched_gt':  List of gt indices (false negatives)
            'unmatched_pred': List of pred indices (false positives)
    """
    # Build IoU matrix
    iou_pairs = []
    for gi, gt in enumerate(gt_segments):
        for pi, pred in enumerate(pred_segments):
            iou = temporal_iou(gt, pred)
            if iou >= iou_threshold:
                iou_pairs.append((iou, gi, pi))
    
    # Sort descending by IoU for greedy matching
    iou_pairs.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    for iou_val, gi, pi in iou_pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matches.append((gi, pi, iou_val))
            matched_gt.add(gi)
            matched_pred.add(pi)
    
    unmatched_gt = [i for i in range(len(gt_segments)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_segments)) if i not in matched_pred]
    
    return {
        'matches': matches,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred,
    }


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

def compute_metrics(
    gt_segments: List[Tuple[float, float]],
    pred_segments: List[Tuple[float, float]],
    iou_threshold: float = 0.3
) -> Dict:
    """Compute Precision, Recall, and F1 for rally detection.
    
    Precision = TP / (TP + FP)   — of all predicted rallies, how many are real?
    Recall    = TP / (TP + FN)   — of all real rallies, how many did we find?
    F1        = 2 * P * R / (P + R)
    
    A true positive is a predicted segment matched to a ground-truth segment
    with t-IoU >= iou_threshold.
    """
    result = match_segments(gt_segments, pred_segments, iou_threshold)
    
    tp = len(result['matches'])
    fp = len(result['unmatched_pred'])
    fn = len(result['unmatched_gt'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Compute mean IoU of matched pairs
    mean_iou = 0.0
    if result['matches']:
        mean_iou = sum(m[2] for m in result['matches']) / len(result['matches'])
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'mean_iou': mean_iou,
        'matches': [
            {
                'gt': gt_segments[gi],
                'pred': pred_segments[pi],
                'iou': iou_val,
            }
            for gi, pi, iou_val in result['matches']
        ],
        'false_negatives': [gt_segments[i] for i in result['unmatched_gt']],
        'false_positives': [pred_segments[i] for i in result['unmatched_pred']],
    }


# ---------------------------------------------------------------------------
# Pretty Printing
# ---------------------------------------------------------------------------

def print_report(metrics: Dict, iou_threshold: float):
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("  RALLY DETECTION EVALUATION REPORT")
    print("=" * 60)
    print(f"  IoU Threshold : {iou_threshold:.2f}")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print("-" * 60)
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1']:.4f}")
    print(f"  Mean IoU      : {metrics['mean_iou']:.4f}")
    print("-" * 60)
    
    if metrics['matches']:
        print("\n  Matched Rallies:")
        for i, m in enumerate(metrics['matches']):
            gt_s, gt_e = m['gt']
            pr_s, pr_e = m['pred']
            print(f"    [{i+1}] GT [{gt_s:.1f}s – {gt_e:.1f}s] <-> "
                  f"Pred [{pr_s:.1f}s – {pr_e:.1f}s]  "
                  f"IoU={m['iou']:.3f}")
    
    if metrics['false_negatives']:
        print("\n  False Negatives (missed rallies):")
        for seg in metrics['false_negatives']:
            print(f"    GT [{seg[0]:.1f}s – {seg[1]:.1f}s]")
    
    if metrics['false_positives']:
        print("\n  False Positives (phantom detections):")
        for seg in metrics['false_positives']:
            print(f"    Pred [{seg[0]:.1f}s – {seg[1]:.1f}s]")
    
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rally detection against ground-truth annotations."
    )
    parser.add_argument(
        "--gt", required=True,
        help="Path to ground-truth JSON file."
    )
    parser.add_argument(
        "--pred", required=True,
        help="Path to predicted rallies CSV (rallies.csv)."
    )
    parser.add_argument(
        "--iou", type=float, default=0.3,
        help="Minimum temporal IoU to count a match (default: 0.3)."
    )
    
    args = parser.parse_args()
    
    gt = load_ground_truth(args.gt)
    pred = load_predictions(args.pred)
    
    print(f"Loaded {len(gt)} ground-truth segments, {len(pred)} predictions.")
    
    metrics = compute_metrics(gt, pred, args.iou)
    print_report(metrics, args.iou)


if __name__ == "__main__":
    main()
