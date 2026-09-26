"""Scoring detections against ground truth, and navigating the disagreements.

Wraps :mod:`really_rally.utils.evaluate` rather than reimplementing it: the
studio and the CLI must report the same numbers for the same inputs, or a
tuning decision made in one would not hold in the other.

The disagreement navigator is the part that changes how detector work is done.
Rather than reading a precision number and guessing which rallies are missing,
you step through the actual missed and invented segments with the video,
telemetry and the detector's stated reason all on screen at that timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from really_rally.utils.evaluate import compute_metrics, load_ground_truth, match_segments

# One-sided gate for reporting: below this F1 the detector is not usable, so
# the studio should say so rather than presenting a number without judgement.
USABLE_F1 = 0.5


@dataclass
class Disagreement:
    """One place the detector and the annotator disagree."""
    kind: str          # "missed" | "false"
    start: float
    end: float
    detail: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class EvaluationEngine:
    """Compare predictions to labels and expose the disagreements."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gt_segments: List[Tuple[float, float]],
        pred_segments: List[Tuple[float, float]],
    ) -> dict:
        """Full metric dict, including matched pairs and unmatched segments."""
        return compute_metrics(gt_segments, pred_segments, self.iou_threshold)

    @staticmethod
    def load_labels(path: str) -> List[Tuple[float, float]]:
        return load_ground_truth(path)

    def disagreements(
        self,
        gt_segments: List[Tuple[float, float]],
        pred_segments: List[Tuple[float, float]],
    ) -> List[Disagreement]:
        """Every missed and invented segment, ordered along the timeline."""
        result = match_segments(gt_segments, pred_segments, self.iou_threshold)
        out: List[Disagreement] = []

        for index in result["unmatched_gt"]:
            start, end = gt_segments[index]
            out.append(Disagreement(
                "missed", start, end,
                f"labelled {end - start:.1f}s but nothing was detected here"))

        for index in result["unmatched_pred"]:
            start, end = pred_segments[index]
            out.append(Disagreement(
                "false", start, end,
                f"detected {end - start:.1f}s with no label covering it"))

        return sorted(out, key=lambda d: d.start)

    def next_disagreement(
        self,
        gt_segments: List[Tuple[float, float]],
        pred_segments: List[Tuple[float, float]],
        after_time: float,
        kind: Optional[str] = None,
    ) -> Optional[Disagreement]:
        """The next disagreement after ``after_time``, optionally of one kind."""
        candidates = [d for d in self.disagreements(gt_segments, pred_segments)
                      if d.start > after_time and (kind is None or d.kind == kind)]
        return candidates[0] if candidates else None

    def previous_disagreement(
        self,
        gt_segments: List[Tuple[float, float]],
        pred_segments: List[Tuple[float, float]],
        before_time: float,
        kind: Optional[str] = None,
    ) -> Optional[Disagreement]:
        candidates = [d for d in self.disagreements(gt_segments, pred_segments)
                      if d.start < before_time and (kind is None or d.kind == kind)]
        return candidates[-1] if candidates else None


def format_metrics(metrics: dict, iou_threshold: float) -> str:
    """One-line-ish summary for the status area."""
    if metrics["tp"] == 0 and metrics["fp"] == 0 and metrics["fn"] == 0:
        return "No labels and no detections — nothing to score."
    if metrics["tp"] == 0 and metrics["fn"] > 0 and metrics["fp"] == 0:
        return (f"P {metrics['precision']:.2f}  R {metrics['recall']:.2f}  "
                f"F1 {metrics['f1']:.2f}  (all {metrics['fn']} labelled interval(s) missed)")
    return (f"@IoU {iou_threshold:.2f} — P {metrics['precision']:.2f}  "
            f"R {metrics['recall']:.2f}  F1 {metrics['f1']:.2f}  "
            f"TP {metrics['tp']} / FP {metrics['fp']} / FN {metrics['fn']}  "
            f"mean IoU {metrics['mean_iou']:.2f}")


def verdict(metrics: dict) -> str:
    """A blunt read on whether the numbers are usable at all."""
    if metrics["tp"] + metrics["fn"] == 0:
        return "no ground truth"
    if metrics["f1"] >= USABLE_F1:
        return "usable"
    return "below usable threshold"
