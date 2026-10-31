"""Telemetry loading and interactive plotting.

Split in two on purpose:

- :class:`TelemetryRepository` is plain data — it loads a feature store into
  numpy arrays and answers "what were the values at frame N". It imports no GUI
  toolkit, so it is unit-testable and reusable.
- :class:`TelemetryView` embeds a matplotlib figure in Tk. It renders whatever
  keys the repository has, grouped by role, so a new telemetry column appears
  without touching this file.

The plots are built once; seeking only moves the playhead artists. Redrawing the
whole figure per frame is what made the static-PNG workflow feel slow, and it is
avoidable here.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from really_rally.utils.feature_store import (
    FEATURE_ROLES,
    ROLE_RAW,
    ROLE_SCORE,
    ROLE_THRESHOLD,
)

# Above this many samples a series is decimated before plotting. Long clips at
# 30Hz sampling produce tens of thousands of points per key and the extra detail
# is invisible at screen resolution anyway.
MAX_PLOT_POINTS = 20000


class TelemetryRepository:
    """Frame-indexed access to a loaded feature store."""

    def __init__(self, records: List[Dict], fps: float, total_frames: Optional[int] = None):
        self.records = records
        self.fps = fps
        self.frames = np.array([int(r.get("frame", 0)) for r in records], dtype=int)
        if total_frames:
            self.total_frames = int(total_frames)
        elif len(self.frames):
            self.total_frames = int(self.frames.max()) + 1
        else:
            self.total_frames = 0
        self._series_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_features_csv(cls, path: str, fps: float, total_frames: Optional[int] = None):
        from really_rally.utils.feature_store import read_feature_store
        return cls(read_feature_store(path), fps, total_frames)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    @property
    def times(self) -> np.ndarray:
        return self.frames / self.fps if self.fps else np.zeros_like(self.frames, dtype=float)

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0

    # -- lookup -----------------------------------------------------------

    def index_for_frame(self, frame: int) -> Optional[int]:
        """Nearest sampled frame. Feature rows exist only every Nth frame.

        An exact tie resolves to the earlier sample, so scrubbing across the
        midpoint of two samples never flickers forward and back.
        """
        if self.is_empty:
            return None
        pos = int(np.searchsorted(self.frames, frame))
        if pos == 0:
            return 0
        if pos >= len(self.frames):
            return len(self.frames) - 1
        before, after = self.frames[pos - 1], self.frames[pos]
        return pos - 1 if (frame - before) <= (after - frame) else pos

    def record_at(self, frame: int) -> Optional[Dict]:
        idx = self.index_for_frame(frame)
        return None if idx is None else self.records[idx]

    def value_at(self, key: str, frame: int, default=0.0):
        record = self.record_at(frame)
        if record is None:
            return default
        return record.get(key, default)

    def state_at(self, frame: int) -> str:
        return self.value_at("state", frame, "")

    def is_candidate_at(self, frame: int) -> bool:
        return bool(self.value_at("candidate", frame, 0))

    # -- series -----------------------------------------------------------

    def series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """(times, values) for one key, cached.

        Text columns (``state``, ``fired_signals``) are not plottable; they come
        back as zeros rather than raising, so callers can ask about any key.
        """
        if key not in self._series_cache:
            values = np.array([_as_float(r.get(key)) for r in self.records], dtype=float)
            self._series_cache[key] = (self.times, values)
        return self._series_cache[key]

    def present_keys(self) -> List[str]:
        if self.is_empty:
            return []
        return [k for k in self._series_cache.keys()] or self._record_keys()

    def _record_keys(self) -> List[str]:
        keys: List[str] = []
        for record in self.records:
            for key in record:
                if key not in keys:
                    keys.append(key)
        return keys

    def has_signal(self, key: str) -> bool:
        """True when a key carries numeric data worth plotting."""
        if self.is_empty:
            return False
        if any(isinstance(r.get(key), str) for r in self.records):
            return False
        _, values = self.series(key)
        if values.size == 0:
            return False
        return bool(np.any(np.nan_to_num(values) != 0.0))

    def plottable_keys(self, role: Optional[str] = None) -> List[str]:
        keys = [k for k in self._record_keys() if self.has_signal(k)]
        if role is None:
            return keys
        return [k for k in keys if FEATURE_ROLES.get(k) == role]

    def candidate_spans(self) -> List[Tuple[float, float]]:
        """Time ranges where the detector flagged a candidate."""
        return _mask_to_spans(self.frames, self._bool_column("candidate"), self.fps)

    def state_spans(self) -> List[Tuple[float, float, str]]:
        """Contiguous (start, end, state_name) runs of the FSM state column."""
        if self.is_empty:
            return []
        spans: List[Tuple[float, float, str]] = []
        start = 0
        current = self.records[0].get("state", "")
        for i in range(1, len(self.records)):
            state = self.records[i].get("state", "")
            if state != current:
                spans.append((self.frames[start] / self.fps,
                              self.frames[i] / self.fps, current))
                start, current = i, state
        spans.append((self.frames[start] / self.fps,
                      self.frames[-1] / self.fps, current))
        return [s for s in spans if s[0] != s[1] or s[2]]

    def _bool_column(self, key: str) -> np.ndarray:
        return np.array([bool(r.get(key, 0)) for r in self.records], dtype=bool)


def _as_float(value, default: float = 0.0) -> float:
    """Coerce a stored value to float, tolerating text/blank/None cells."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mask_to_spans(frames: np.ndarray, mask: np.ndarray, fps: float) -> List[Tuple[float, float]]:
    """Contiguous True runs as (start_sec, end_sec) pairs."""
    spans: List[Tuple[float, float]] = []
    if len(mask) == 0:
        return spans
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((frames[start] / fps, frames[i - 1] / fps))
            start = None
    if start is not None:
        spans.append((frames[start] / fps, frames[-1] / fps))
    return spans


# ---------------------------------------------------------------------------
# Interactive view
# ---------------------------------------------------------------------------

class TelemetryView:
    """A matplotlib figure embedded in a Tk widget, synchronized to a playhead.

    Panels are derived from the data: one per raw signal (with its paired
    adaptive threshold), one for the derived scores, and one decision strip.
    No metric name is hard-coded, so new telemetry extends the view for free.
    """

    PANEL_HEIGHT_IN = 1.5
    STRIP_HEIGHT_IN = 0.8

    def __init__(self, parent, repository: TelemetryRepository,
                 on_seek: Optional[Callable[[int], None]] = None,
                 with_label_lane: bool = False):
        # Imported here so importing this module (e.g. for the repository) does
        # not require a display.
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.figure import Figure

        self.repository = repository
        self.on_seek = on_seek
        self._with_label_lane = with_label_lane
        self._lines: Dict[str, list] = {}
        self._playheads: List = []
        self._overlay_artists: List = []
        self._panel_keys: List[str] = []
        self.label_lane = None

        raw_keys = [k for k in repository.plottable_keys(ROLE_RAW) if self._is_panel_key(k)]
        self._panel_keys = raw_keys
        self._has_scores = bool(repository.plottable_keys(ROLE_SCORE))

        rows = (max(1, len(raw_keys)) + (1 if self._has_scores else 0)
                + 1 + (1 if with_label_lane else 0))  # decision strip (+ label lane)
        height = (rows * self.PANEL_HEIGHT_IN) + self.STRIP_HEIGHT_IN
        self.figure = Figure(figsize=(10, min(height, 16)), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)

        self._build()
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- layout -----------------------------------------------------------

    def _is_panel_key(self, key: str) -> bool:
        """One panel per signal, but not for the raw inputs of a smoothed one."""
        return not (key.startswith("smoothed_") and key.replace("smoothed_", "t_") in
                    self.repository.plottable_keys(ROLE_THRESHOLD))

    def _paired_threshold(self, key: str) -> Optional[str]:
        """smoothed_arm -> t_arm, when that threshold exists."""
        base = key.replace("smoothed_", "")
        candidate = f"t_{base}"
        return candidate if self.repository.has_signal(candidate) else None

    def _build(self) -> None:
        n_panels = (len(self._panel_keys) + (1 if self._has_scores else 0) + 1
                    + (1 if self._with_label_lane else 0))
        idx = 0

        for key in self._panel_keys:
            ax = self.figure.add_subplot(n_panels, 1, idx + 1)
            self._plot_signal_panel(ax, key)
            idx += 1

        if self._has_scores:
            ax = self.figure.add_subplot(n_panels, 1, idx + 1)
            self._plot_score_panel(ax)
            idx += 1

        self.ax_decision = self.figure.add_subplot(n_panels, 1, idx + 1)
        self._plot_decision_panel(self.ax_decision)
        idx += 1

        if self._with_label_lane:
            self.label_lane = self.figure.add_subplot(n_panels, 1, idx + 1)
            self.label_lane.set_ylim(0, 1)
            self.label_lane.set_yticks([])
            self.label_lane.set_ylabel("labels", fontsize=8, fontweight="bold")

        for ax in self.figure.axes[:-1]:
            ax.tick_params(labelbottom=False)
        self.figure.axes[-1].set_xlabel("Time (s)")

        self._draw_decision_overlays()
        self.figure.tight_layout()

    def _plot_signal_panel(self, ax, key: str) -> None:
        times, values = self.repository.series(key)
        line, = ax.plot(times, values, linewidth=1.2, label=key)
        self._lines[key] = [line]

        threshold_key = self._paired_threshold(key)
        if threshold_key:
            t_times, t_values = self.repository.series(threshold_key)
            t_line, = ax.plot(t_times, t_values, linestyle="--", linewidth=1.0,
                              color="#d62728", label=threshold_key)
            self._lines[threshold_key] = [t_line]

        ax.set_ylabel(key, fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)
        self._add_playhead(ax)

    def _plot_score_panel(self, ax) -> None:
        for key in self.repository.plottable_keys(ROLE_SCORE):
            times, values = self.repository.series(key)
            line, = ax.plot(times, values, linewidth=1.1, label=key)
            self._lines[key] = [line]
        for key in self.repository.plottable_keys(ROLE_THRESHOLD):
            times, values = self.repository.series(key)
            line, = ax.plot(times, values, linestyle="--", linewidth=1.0, label=key)
            self._lines.setdefault(key, []).append(line)
        ax.set_ylabel("score", fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)
        self._add_playhead(ax)

    def _plot_decision_panel(self, ax) -> None:
        """Where the detector said yes, and what state it was in."""
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("decision", fontsize=8, fontweight="bold")
        self._add_playhead(ax)

    def _add_playhead(self, ax) -> None:
        line = ax.axvline(0.0, color="#111111", linewidth=1.0, alpha=0.8, zorder=5)
        self._playheads.append(line)

    def _draw_decision_overlays(self) -> None:
        """Shade candidate spans and label state runs.

        Drawn once: these only change when predictions change, not when the
        playhead moves.
        """
        if self.repository.is_empty:
            return
        spans = self.repository.candidate_spans()
        for ax in self.figure.axes:
            for start, end in spans:
                artist = ax.axvspan(start, end, color="#2ca02c", alpha=0.12, zorder=0)
                self._overlay_artists.append(artist)

        for start, end, state in self.repository.state_spans():
            if not state or state == "idle":
                continue
            self.ax_decision.axvspan(start, end, color="#1f77b4", alpha=0.35, zorder=1)
            self.ax_decision.text((start + end) / 2, 0.5, state, ha="center", va="center",
                                  fontsize=6, clip_on=True)

    # -- interaction ------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or not self.on_seek:
            return
        self.on_seek(int(round(event.xdata * self.repository.fps)))

    def set_playhead(self, frame: int) -> None:
        """Move the cursor only — never rebuilds the plots."""
        t = frame / self.repository.fps if self.repository.fps else 0.0
        for line in self._playheads:
            line.set_xdata([t, t])
        self.canvas.draw_idle()

    def set_xlim(self, start_sec: float, end_sec: float) -> None:
        for ax in self.figure.axes:
            ax.set_xlim(start_sec, end_sec)
        self.canvas.draw_idle()

    def toggle_series(self, key: str, visible: bool) -> None:
        for line in self._lines.get(key, []):
            line.set_visible(visible)
        self.canvas.draw_idle()

    def available_series(self) -> List[str]:
        return sorted(self._lines.keys())

    def set_segment_overlays(self, segments, color: str = "#1f77b4") -> None:
        """Shade predicted/labelled segments. Replaces previous overlays."""
        for artist in self._overlay_artists:
            artist.remove()
        self._overlay_artists = []
        for start, end in segments:
            for ax in self.figure.axes:
                self._overlay_artists.append(
                    ax.axvspan(start, end, color=color, alpha=0.15, zorder=0))
        self.canvas.draw_idle()
