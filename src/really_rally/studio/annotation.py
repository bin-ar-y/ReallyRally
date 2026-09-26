"""Ground-truth annotation, stored independently of detector output.

Two rules drive this module:

1. **Ground truth is never written by the detector.** Labels live in their own
   ``*_labels.csv``. Changing a threshold cannot touch them, so a labelled
   recording stays a valid regression test no matter how the detector evolves.
2. **A human label and a prediction are different things.** They are kept in
   different files and only ever compared, never merged.

Times are seconds, derived the same way the pipeline derives them
(``frame / fps``, see ``processors/base.py``). Frames are deliberately *not*
stored alongside: two representations of one instant drift apart, and the
comparison against detector events only works if both sides use one convention.

The vocabulary is free text with suggested values. Constraining it to an enum
now would mean a schema migration the first time a stroke needs classifying as
forehand or backhand.
"""

from __future__ import annotations

import csv
import json
import os
import time as _time
import uuid
import zlib
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

KIND_INTERVAL = "interval"
KIND_POINT = "point"

SOURCE_MANUAL = "manual"
SOURCE_IMPORTED = "imported"

# Starting vocabulary. The goal is to separate real play from tennis-like
# movement without ball play — the confusion the detector currently makes —
# not to classify every stroke type yet.
INTERVAL_VOCABULARY = ["Rally", "Shadow Practice", "Idle / Non-tennis", "Uncertain"]
POINT_VOCABULARY = ["Real Stroke", "Serve", "Shadow Swing", "Uncertain Event"]

LABEL_COLUMNS = ["label_id", "kind", "start", "end", "time", "label", "notes",
                 "source", "created_at"]

# Deterministic per-class colours: a stable crc32 rather than hash(), which is
# salted per process and would repaint labels differently on every launch.
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def color_for(label: str) -> str:
    return _PALETTE[zlib.crc32(label.encode("utf-8")) % len(_PALETTE)]


@dataclass
class Label:
    kind: str
    label: str
    start: float = 0.0
    end: float = 0.0
    time: float = 0.0
    notes: str = ""
    source: str = SOURCE_MANUAL
    label_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = ""

    @property
    def is_interval(self) -> bool:
        return self.kind == KIND_INTERVAL

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def span(self) -> Tuple[float, float]:
        """The label's extent in seconds, whichever kind it is."""
        return (self.start, self.end) if self.is_interval else (self.time, self.time)

    def contains(self, t: float, tolerance: float = 0.0) -> bool:
        start, end = self.span()
        return (start - tolerance) <= t <= (end + tolerance)


class AnnotationRepository:
    """Create, edit and persist human labels for one recording."""

    def __init__(self, labels: Optional[List[Label]] = None, path: Optional[str] = None):
        self.labels: List[Label] = list(labels or [])
        self.path = path

    # -- construction -----------------------------------------------------

    @classmethod
    def from_labels_csv(cls, path: str) -> "AnnotationRepository":
        if not os.path.isfile(path):
            return cls([], path)
        labels: List[Label] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                labels.append(Label(
                    kind=row.get("kind", KIND_INTERVAL) or KIND_INTERVAL,
                    label=row.get("label", "") or "",
                    start=_to_float(row.get("start")),
                    end=_to_float(row.get("end")),
                    time=_to_float(row.get("time")),
                    notes=row.get("notes", "") or "",
                    source=row.get("source", SOURCE_MANUAL) or SOURCE_MANUAL,
                    label_id=row.get("label_id", "") or uuid.uuid4().hex[:8],
                    created_at=row.get("created_at", "") or "",
                ))
        return cls(labels, path)

    @classmethod
    def from_annotations_json(cls, path: str) -> "AnnotationRepository":
        """Import the existing benchmark annotation format.

        Reuses ``benchmarks/label.py`` so the two tools cannot disagree about
        what a valid annotation file is.
        """
        from benchmarks.label import load_annotations

        labels = [
            Label(kind=KIND_INTERVAL, label=entry.get("label", "rally"),
                  start=float(entry["start"]), end=float(entry["end"]),
                  notes=entry.get("notes", ""), source=SOURCE_IMPORTED)
            for entry in load_annotations(path)
        ]
        return cls(labels)

    # -- persistence ------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError("No path given and no default path set.")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        ordered = sorted(self.labels, key=lambda entry: (entry.span()[0], entry.kind, entry.label))
        with open(target, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            for label in ordered:
                row = asdict(label)
                row["start"] = f"{label.start:.3f}"
                row["end"] = f"{label.end:.3f}"
                row["time"] = f"{label.time:.3f}"
                writer.writerow(row)
        self.path = target
        return target

    def to_annotations_json(self, path: str) -> str:
        """Export intervals in the benchmark format.

        Keeps ``benchmarks/run_benchmark.py`` and ``utils/evaluate.py`` working
        against studio-authored ground truth without changing either.
        """
        intervals = [
            {"start": round(entry.start, 3), "end": round(entry.end, 3),
             "label": entry.label, "notes": entry.notes}
            for entry in self.intervals()
        ]
        merged = _merge_intervals(intervals)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(merged, f, indent=2)
        return path

    # -- editing ----------------------------------------------------------

    def add_interval(self, start: float, end: float, label: str, notes: str = "") -> Label:
        if end < start:
            start, end = end, start
        entry = Label(kind=KIND_INTERVAL, label=label, start=start, end=end,
                      notes=notes, created_at=_now())
        self.labels.append(entry)
        self._autosave()
        return entry

    def add_point(self, time: float, label: str, notes: str = "") -> Label:
        entry = Label(kind=KIND_POINT, label=label, time=time,
                      notes=notes, created_at=_now())
        self.labels.append(entry)
        self._autosave()
        return entry

    def update(self, label_id: str, **changes) -> bool:
        for i, label in enumerate(self.labels):
            if label.label_id == label_id:
                self.labels[i] = replace(label, **changes)
                self._autosave()
                return True
        return False

    def remove(self, label_id: str) -> bool:
        before = len(self.labels)
        self.labels = [entry for entry in self.labels if entry.label_id != label_id]
        if len(self.labels) != before:
            self._autosave()
            return True
        return False

    def remove_at(self, t: float, tolerance: float = 0.25) -> int:
        """Delete the single label closest to ``t``; returns 1 if one went.

        One label per press, points before intervals. A point sits exactly at
        the playhead whereas an interval merely contains it, so deleting the
        interval when the user meant to remove a stroke marker would discard a
        whole rally's ground truth on one keystroke.
        """
        candidates = self.labels_at(t, tolerance)
        if not candidates:
            return 0
        best = min(candidates, key=lambda entry: (entry.is_interval, abs(entry.span()[0] - t)))
        return 1 if self.remove(best.label_id) else 0

    def clear(self) -> None:
        self.labels = []
        self._autosave()

    def _autosave(self) -> None:
        if self.path:
            try:
                self.save()
            except OSError:
                # Losing an autosave must never lose the in-memory edit; the
                # caller can retry with an explicit save.
                pass

    # -- queries ----------------------------------------------------------

    def intervals(self) -> List[Label]:
        return [entry for entry in self.labels if entry.is_interval]

    def points(self) -> List[Label]:
        return [entry for entry in self.labels if not entry.is_interval]

    def interval_tuples(self) -> List[Tuple[float, float]]:
        """(start, end) pairs for scoring — the shape utils/evaluate.py wants."""
        return sorted((entry.start, entry.end) for entry in self.intervals())

    def interval_spans(self) -> List[Tuple[float, float, str]]:
        return sorted((entry.start, entry.end, entry.label) for entry in self.intervals())

    def point_marks(self) -> List[Tuple[float, str]]:
        return sorted((entry.time, entry.label) for entry in self.points())

    def labels_at(self, t: float, tolerance: float = 0.25) -> List[Label]:
        return [entry for entry in self.labels if entry.contains(t, tolerance)]

    def next_label(self, t: float) -> Optional[Label]:
        later = [entry for entry in self.labels if entry.span()[0] > t]
        return min(later, key=lambda entry: entry.span()[0]) if later else None

    def previous_label(self, t: float) -> Optional[Label]:
        earlier = [entry for entry in self.labels if entry.span()[0] < t]
        return max(earlier, key=lambda entry: entry.span()[0]) if earlier else None

    def count(self) -> Dict[str, int]:
        return {
            "intervals": len(self.intervals()),
            "points": len(self.points()),
            "total": len(self.labels),
        }


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%S")


def _merge_intervals(intervals: List[dict]) -> List[dict]:
    """Merge overlapping intervals, keeping the first label's class."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda s: s["start"])
    merged = [dict(ordered[0])]
    for entry in ordered[1:]:
        last = merged[-1]
        if entry["start"] <= last["end"]:
            last["end"] = max(last["end"], entry["end"])
        else:
            merged.append(dict(entry))
    return merged


# ---------------------------------------------------------------------------
# Track rendering
# ---------------------------------------------------------------------------

class AnnotationTrack:
    """Draws labels onto a dedicated lane of the telemetry figure.

    Intervals become coloured spans, points become markers. The lane shares the
    time axis with the telemetry, so a label and the signal that missed it are
    always read at the same x position.
    """

    def __init__(self, view):
        self.view = view
        self.ax = view.label_lane
        self._artists: List = []
        self._pending_artist = None

    def redraw(self, repository: AnnotationRepository) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists = []
        if self.ax is None:
            return

        for start, end, label in repository.interval_spans():
            self._artists.append(
                self.ax.axvspan(start, end, color=color_for(label), alpha=0.45))
            if end > start:
                self._artists.append(
                    self.ax.text((start + end) / 2, 0.5, label, ha="center", va="center",
                                 fontsize=6, clip_on=True))

        for time_sec, label in repository.point_marks():
            self._artists.append(
                self.ax.axvline(time_sec, color=color_for(label), linewidth=1.4))
            self._artists.append(self.ax.annotate(
                label, xy=(time_sec, 1.0), xytext=(2, -2), textcoords="offset points",
                fontsize=6, rotation=45, va="top", clip_on=True))

        self.view.canvas.draw_idle()

    def show_pending_start(self, time_sec: Optional[float]) -> None:
        """Mark the in-progress interval start while the user is drawing it."""
        if self._pending_artist is not None:
            self._pending_artist.remove()
            self._pending_artist = None
        if time_sec is not None and self.ax is not None:
            self._pending_artist = self.ax.axvline(
                time_sec, color="#d62728", linestyle=":", linewidth=1.5)
        self.view.canvas.draw_idle()
