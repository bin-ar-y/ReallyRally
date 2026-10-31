"""Re-run detection against a cached feature store — no video, no MediaPipe.

This is what makes threshold tuning interactive. The expensive part of the
pipeline is pose estimation over video frames; once ``*_features.csv`` exists,
detection is a few numpy operations over a few thousand numbers and completes in
milliseconds.

Correctness rests on the replay path calling the *same* ``core.detection``
analyzers the live pipeline calls — not a re-implementation. The one thing the
replay has to reconstruct is segment boundaries, which it recovers from gaps in
the stored frame indices rather than a separate column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from really_rally.core.config_snapshot import ConfigSnapshot
from really_rally.core.detection import FeedAnalyzer, RallyAnalyzer, ServeAnalyzer
from really_rally.core.models import RallySegment, ServeEvent
from really_rally.utils.feature_store import read_feature_store, split_into_segments

MODE_RALLY = "rally"
MODE_SERVE = "serve"
MODE_FEED = "feed"


@dataclass
class ReplayResult:
    """Everything a replay produced, plus the configuration that produced it."""
    records: List[Dict] = field(default_factory=list)
    segments: List[RallySegment] = field(default_factory=list)
    events: List[ServeEvent] = field(default_factory=list)
    config: Optional[ConfigSnapshot] = None
    mode: str = MODE_RALLY
    segments_found: int = 0

    @property
    def event_tuples(self) -> List[tuple]:
        return [(s.start_time, s.end_time) for s in self.segments]


def infer_fps(records: List[Dict], default: float = 30.0) -> float:
    """Recover the frame rate from frame/time pairs.

    Only used when the caller cannot supply the video's real fps. The stored
    ``time`` is frame-derived, so consecutive rows give the rate exactly even on
    variable-framerate footage (where a single nominal fps would not).
    """
    for previous, current in zip(records, records[1:]):
        frame_delta = current["frame"] - previous["frame"]
        time_delta = current.get("time", 0.0) - previous.get("time", 0.0)
        if frame_delta > 0 and time_delta > 1e-9:
            return frame_delta / time_delta
    return default


class DetectorReplayEngine:
    """Drives the pure analyzers over a cached feature store."""

    def __init__(self, mode: str = MODE_RALLY):
        self.mode = mode

    def replay(
        self,
        features_path: str,
        fps: Optional[float] = None,
        total_frames: Optional[int] = None,
        overrides: Optional[Dict[str, float]] = None,
        process_every_n: Optional[int] = None,
    ) -> ReplayResult:
        """Recompute detection for one feature store.

        ``fps`` and ``total_frames`` describe the source video and are needed to
        reproduce the live run exactly, so the studio passes the probed values.
        They are inferred only as a fallback, since the store records the
        sampled frames rather than the whole timeline.
        """
        from really_rally.core.config import PROCESS_EVERY_N_FRAMES

        records = read_feature_store(features_path)
        if not records:
            return ReplayResult(config=ConfigSnapshot.from_module(
                fps=fps or 30.0, max_time=0.0), mode=self.mode)

        if fps is None:
            fps = infer_fps(records)
        stride = process_every_n or PROCESS_EVERY_N_FRAMES
        if total_frames is None:
            total_frames = records[-1]["frame"] + 1

        config = ConfigSnapshot.from_module(
            fps=fps, max_time=total_frames / fps, process_every_n=stride)
        if overrides:
            config = config.apply(overrides)

        segments_list = split_into_segments(records, stride)

        if self.mode == MODE_SERVE:
            analyzer = ServeAnalyzer(config)
            segments: List[RallySegment] = []
            events: List[ServeEvent] = []
            for segment in segments_list:
                new_segments, new_events = analyzer.analyze(segment)
                segments.extend(new_segments)
                events.extend(new_events)
        elif self.mode == MODE_FEED:
            analyzer = FeedAnalyzer(config, total_frames=total_frames)
            segments, events = [], []
            for segment in segments_list:
                segments.extend(analyzer.analyze(segment))
        else:
            analyzer = RallyAnalyzer(config)
            segments, events = [], []
            for segment in segments_list:
                segments.extend(analyzer.analyze(segment))

        return ReplayResult(
            records=records,
            segments=segments,
            events=events,
            config=config,
            mode=self.mode,
            segments_found=len(segments_list),
        )
