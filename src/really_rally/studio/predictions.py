"""Detector output, written separately from both features and ground truth.

Three files, three owners:

- ``*_features.csv`` — extracted measurements (immutable once written)
- ``*_predictions.csv`` — what the detector currently decides
- ``*_labels.csv`` — what actually happened (human)

Keeping predictions apart from features means tuning a threshold rewrites only
the predictions. Keeping them apart from labels means a detector change can
never contaminate ground truth, which is what makes a labelled recording a
permanent regression test.

Each prediction carries a ``Reason`` so a decision can be explained rather than
just asserted — the difference between "no rally here" and "no rally here
because the distance gate rejected every frame".
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from really_rally.core.models import RallySegment, ServeEvent

PREDICTION_COLUMNS = [
    "Source Video", "Detector", "Event Type", "Start Time", "End Time", "Duration",
    "Score", "State", "Reason", "Details",
]


@dataclass
class Prediction:
    detector: str                      # rally | serve | feed
    event_type: str = "rally"
    start_time: float = 0.0
    end_time: float = 0.0
    score: float = 0.0
    state: str = ""
    reason: str = ""
    details: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def span(self):
        return (self.start_time, self.end_time)


def _bbox_reason(records: List[Dict]) -> str:
    """Summarise the evidence inside a prediction's frames.

    Answers "why was this a rally?" with the signals that actually fired, not a
    restatement of the score.
    """
    inside = [r for r in records if r.get("candidate")]
    if not inside:
        return "no candidate frames"
    tally: Dict[str, int] = {}
    for record in inside:
        for name in str(record.get("fired_signals", "")).split("|"):
            if name:
                tally[name] = tally.get(name, 0) + 1
    if not tally:
        return f"{len(inside)} candidate frame(s)"
    parts = ", ".join(f"{name}×{count}" for name, count in sorted(tally.items()))
    return f"{len(inside)} candidate frame(s): {parts}"


def predictions_from_segments(
    segments: List[RallySegment],
    detector: str = "rally",
    records: Optional[List[Dict]] = None,
) -> List[Prediction]:
    """Convert detected segments into explainable predictions."""
    out: List[Prediction] = []
    for segment in segments:
        window = [r for r in (records or [])
                  if segment.start_frame <= r.get("frame", 0) <= segment.end_frame]
        states = [r.get("state", "") for r in window if r.get("state")]
        out.append(Prediction(
            detector=detector,
            event_type="rally" if detector != "serve" else "serve",
            start_time=segment.start_time,
            end_time=segment.end_time,
            score=segment.score,
            state=states[-1] if states else "",
            reason=_bbox_reason(window),
            details={
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "frames": len(window),
                "peak_coarse_score": max(
                    (float(r.get("coarse_score", 0.0)) for r in window), default=0.0),
            },
        ))
    return out


def predictions_from_events(events: List[ServeEvent],
                            records: Optional[List[Dict]] = None) -> List[Prediction]:
    """Convert serve events into predictions with their score breakdown."""
    out: List[Prediction] = []
    for event in events:
        out.append(Prediction(
            detector="serve",
            event_type="serve",
            start_time=event.start_time,
            end_time=event.end_time,
            score=event.score,
            state="serve",
            reason=(f"{event.handedness}-handed serve; toss "
                    f"{event.toss_score:.2f}, contact {event.contact_score:.2f}, "
                    f"follow-through {event.follow_through_score:.2f}"),
            details={
                "handedness": event.handedness,
                "toss_arm": event.toss_arm,
                "toss_time": event.toss_time,
                "contact_time": event.contact_time,
                "toss_score": event.toss_score,
                "contact_score": event.contact_score,
                "follow_through_score": event.follow_through_score,
                "outcome": event.outcome,
            },
        ))
    return out


class PredictionRepository:
    """Reads and writes ``*_predictions.csv``."""

    def __init__(self, predictions: Optional[List[Prediction]] = None,
                 path: Optional[str] = None):
        self.predictions: List[Prediction] = list(predictions or [])
        self.path = path

    def replace(self, predictions: List[Prediction]) -> None:
        self.predictions = list(predictions)

    def spans(self) -> List[tuple]:
        """(start, end) pairs — the shape utils/evaluate.py consumes."""
        return sorted(p.span() for p in self.predictions)

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError("No path given and no default path set.")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(PREDICTION_COLUMNS)
            for prediction in sorted(self.predictions, key=lambda p: p.start_time):
                writer.writerow([
                    os.path.basename(target).replace("_predictions.csv", ""),
                    prediction.detector,
                    prediction.event_type,
                    f"{prediction.start_time:.2f}",
                    f"{prediction.end_time:.2f}",
                    f"{prediction.duration:.2f}",
                    f"{prediction.score:.3f}",
                    prediction.state,
                    prediction.reason,
                    json.dumps(prediction.details, sort_keys=True),
                ])
        self.path = target
        return target

    @classmethod
    def from_csv(cls, path: str) -> "PredictionRepository":
        if not os.path.isfile(path):
            return cls([], path)
        predictions: List[Prediction] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    details = json.loads(row.get("Details") or "{}")
                except json.JSONDecodeError:
                    details = {}
                predictions.append(Prediction(
                    detector=row.get("Detector", ""),
                    event_type=row.get("Event Type", ""),
                    start_time=float(row.get("Start Time") or 0.0),
                    end_time=float(row.get("End Time") or 0.0),
                    score=float(row.get("Score") or 0.0),
                    state=row.get("State", ""),
                    reason=row.get("Reason", ""),
                    details=details,
                ))
        return cls(predictions, path)
