"""Experiment history: every evaluated configuration, recorded and comparable.

Tuning by eye leaves no trace, so a change that seemed to help cannot be
checked later and a change that made things worse cannot be identified. Each
evaluated configuration is appended here with the full parameter snapshot and
the label set it was scored against, so any result can be reproduced and two
configurations can be compared on identical ground truth.

The comparison deliberately reports what *changed*, including which labelled
events were gained and lost — not just the aggregate score. F1 can hold steady
while the detector starts missing one rally and inventing another.
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from really_rally.core.config_snapshot import ConfigSnapshot

EXPERIMENT_COLUMNS = [
    "Experiment ID", "Timestamp", "Mode", "Video", "Label Set", "IoU Threshold",
    "Config JSON", "Precision", "Recall", "F1", "TP", "FP", "FN",
    "Mean IoU", "Boundary Start Err", "Boundary End Err",
    "Num GT", "Num Pred", "Notes",
]

# Fields shown as a comparison table; the config JSON is kept in the file but
# not rendered, since raw JSON is unreadable in a table.
_DISPLAY_FIELDS = [
    ("precision", "Precision", "{:.3f}"),
    ("recall", "Recall", "{:.3f}"),
    ("f1", "F1", "{:.3f}"),
    ("tp", "TP", "{}"),
    ("fp", "FP", "{}"),
    ("fn", "FN", "{}"),
    ("mean_iou", "Mean IoU", "{:.3f}"),
    ("boundary_start_error_sec", "Start err (s)", "{:.2f}"),
    ("boundary_end_error_sec", "End err (s)", "{:.2f}"),
]


@dataclass
class Experiment:
    experiment_id: str = ""
    timestamp: str = ""
    mode: str = "rally"
    video: str = ""
    label_set: str = ""
    iou_threshold: float = 0.3
    config_json: str = ""
    metrics: Dict = field(default_factory=dict)
    notes: str = ""

    @property
    def f1(self) -> float:
        return float(self.metrics.get("f1", 0.0))

    def metric(self, name: str, default=0.0):
        return self.metrics.get(name, default)


class ExperimentRepository:
    """Append-only history of evaluated configurations."""

    def __init__(self, experiments: Optional[List[Experiment]] = None,
                 path: Optional[str] = None):
        self.experiments: List[Experiment] = list(experiments or [])
        self.path = path

    def record(
        self,
        config: ConfigSnapshot,
        metrics: Dict,
        video: str = "",
        label_set: str = "",
        mode: str = "rally",
        notes: str = "",
    ) -> Experiment:
        experiment = Experiment(
            experiment_id=uuid.uuid4().hex[:8],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            mode=mode,
            video=os.path.basename(video),
            label_set=os.path.basename(label_set),
            iou_threshold=metrics.get("iou_threshold", 0.3),
            config_json=config.to_json() if config else "",
            metrics=dict(metrics),
            notes=notes,
        )
        self.experiments.append(experiment)
        self._autosave()
        return experiment

    def _autosave(self) -> None:
        if self.path:
            try:
                self.save()
            except OSError:
                pass

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError("No path given and no default path set.")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(EXPERIMENT_COLUMNS)
            for experiment in self.experiments:
                writer.writerow(self._row(experiment))
        self.path = target
        return target

    @staticmethod
    def _row(experiment: Experiment) -> List:
        metrics = experiment.metrics
        return [
            experiment.experiment_id, experiment.timestamp, experiment.mode,
            experiment.video, experiment.label_set, f"{experiment.iou_threshold:.2f}",
            experiment.config_json,
            f"{metrics.get('precision', 0.0):.4f}",
            f"{metrics.get('recall', 0.0):.4f}",
            f"{metrics.get('f1', 0.0):.4f}",
            metrics.get("tp", 0), metrics.get("fp", 0), metrics.get("fn", 0),
            f"{metrics.get('mean_iou', 0.0):.4f}",
            f"{metrics.get('boundary_start_error_sec', 0.0):.3f}",
            f"{metrics.get('boundary_end_error_sec', 0.0):.3f}",
            metrics.get("num_gt", 0), metrics.get("num_pred", 0),
            experiment.notes,
        ]

    @classmethod
    def from_csv(cls, path: str) -> "ExperimentRepository":
        if not os.path.isfile(path):
            return cls([], path)
        experiments: List[Experiment] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                experiments.append(Experiment(
                    experiment_id=row.get("Experiment ID", ""),
                    timestamp=row.get("Timestamp", ""),
                    mode=row.get("Mode", ""),
                    video=row.get("Video", ""),
                    label_set=row.get("Label Set", ""),
                    iou_threshold=_to_float(row.get("IoU Threshold"), 0.3),
                    config_json=row.get("Config JSON", ""),
                    metrics={
                        "precision": _to_float(row.get("Precision")),
                        "recall": _to_float(row.get("Recall")),
                        "f1": _to_float(row.get("F1")),
                        "tp": int(_to_float(row.get("TP"))),
                        "fp": int(_to_float(row.get("FP"))),
                        "fn": int(_to_float(row.get("FN"))),
                        "mean_iou": _to_float(row.get("Mean IoU")),
                        "boundary_start_error_sec": _to_float(row.get("Boundary Start Err")),
                        "boundary_end_error_sec": _to_float(row.get("Boundary End Err")),
                        "num_gt": int(_to_float(row.get("Num GT"))),
                        "num_pred": int(_to_float(row.get("Num Pred"))),
                    },
                    notes=row.get("Notes", ""),
                ))
        return cls(experiments, path)


def compare(baseline: Experiment, candidate: Experiment) -> List[tuple]:
    """Field-by-field comparison, with a direction on each row.

    ``direction`` is "better", "worse" or "same" so the caller can colour the
    row without re-deriving which way is good for that metric.
    """
    rows = []
    for key, label, fmt in _DISPLAY_FIELDS:
        before, after = baseline.metric(key), candidate.metric(key)
        rows.append((label, fmt.format(before), fmt.format(after),
                     _direction(key, before, after)))
    return rows


def _direction(key: str, before, after) -> str:
    if before == after:
        return "same"
    # Fewer false positives/negatives and smaller boundary errors are better;
    # everything else in the table is better when larger.
    higher_is_better = key not in ("fp", "fn", "boundary_start_error_sec",
                                   "boundary_end_error_sec")
    improved = after > before if higher_is_better else after < before
    return "better" if improved else "worse"


def summarize_delta(baseline: Experiment, candidate: Experiment) -> str:
    """Plain-language summary of the change in F1 and counts."""
    delta_f1 = candidate.f1 - baseline.f1
    parts = [f"F1 {baseline.f1:.3f} → {candidate.f1:.3f} ({delta_f1:+.3f})"]
    delta_tp = candidate.metric("tp") - baseline.metric("tp")
    delta_fp = candidate.metric("fp") - baseline.metric("fp")
    delta_fn = candidate.metric("fn") - baseline.metric("fn")
    if delta_tp:
        parts.append(f"{delta_tp:+d} matched")
    if delta_fp:
        parts.append(f"{delta_fp:+d} false positive(s)")
    if delta_fn:
        parts.append(f"{delta_fn:+d} missed")
    return "; ".join(parts)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
