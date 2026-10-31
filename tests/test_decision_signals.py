"""Tests that detection records *why* it decided, not just what it decided.

CoarseCandidateDetector.score() has always returned the list of evidence that
fired; the pipeline discarded it. Keeping it is what lets the Debug Studio
answer "why was this a candidate?" — so it gets a regression guard.

Built via __new__ so the test needs no video and no MediaPipe graph.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from really_rally.core.config import PROCESS_EVERY_N_FRAMES  # noqa: E402
from really_rally.processors.base import TennisRallyProcessor  # noqa: E402

FPS = 30.0


def make_processor(total_frames=600):
    """A processor wired just enough for analyze_segment_data (no video).

    Detection itself lives in ``core.detection.RallyAnalyzer``; the processor
    only supplies the video context and collects telemetry.
    """
    p = TennisRallyProcessor.__new__(TennisRallyProcessor)
    p.fps = FPS
    p.total_frames = total_frames
    p._rally_analyzer = None
    p.all_frame_metrics = []
    return p


def stroke_frames(count=80):
    """Quiet baseline with a sharp burst every 20 samples — a stroke pattern."""
    frames = []
    for i in range(count):
        burst = (i % 20) in (8, 9)
        frames.append({
            "frame": i * PROCESS_EVERY_N_FRAMES,
            "timestamp": (i * PROCESS_EVERY_N_FRAMES) / FPS,
            "is_valid_pose": True,
            "energy": 0.02 if burst else 0.002,
            "arm_energy": 0.12 if burst else 0.004,
            "whip_ratio": 3.0 if burst else 0.5,
            "hip_vel": 0.02 if burst else 0.002,
            "spatial_variance": 0.05 if burst else 0.01,
            "player_height": 0.4,
        })
    return frames


class TestFiredSignals:
    def test_every_frame_records_its_fired_evidence(self):
        p = make_processor()
        frames = stroke_frames()
        p.analyze_segment_data(frames)
        assert all("fired_signals" in f for f in frames)

    def test_burst_frames_report_more_evidence_than_quiet_ones(self):
        """Compare channel counts, not exact sets.

        The 0.5s smoothing window deliberately spreads a stroke into adjacent
        samples, so an 'arm' firing just outside a burst index is correct
        behaviour rather than leakage.
        """
        p = make_processor()
        frames = stroke_frames()
        p.analyze_segment_data(frames)

        def channels(is_burst):
            picked = [f for i, f in enumerate(frames) if ((i % 20) in (8, 9)) == is_burst]
            return sum(len(f["fired_signals"].split("|")) if f["fired_signals"] else 0
                       for f in picked) / len(picked)

        assert channels(is_burst=True) > channels(is_burst=False)

    def test_signals_are_pipe_joined_names(self):
        p = make_processor()
        frames = stroke_frames()
        p.analyze_segment_data(frames)
        allowed = {"arm", "whip", "hip", "spatial"}
        for f in frames:
            assert set(f["fired_signals"].split("|")) - {""} <= allowed

    def test_a_strong_frame_reports_all_four_channels(self):
        p = make_processor()
        frames = stroke_frames()
        p.analyze_segment_data(frames)
        assert "arm|whip|hip|spatial" in Counter(f["fired_signals"] for f in frames)

    def test_evidence_suppressed_by_the_distance_gate_is_empty(self):
        """An implausible player scale must produce no evidence at all."""
        p = make_processor()
        frames = stroke_frames()
        for f in frames:
            f["player_height"] = 99.0  # outside PLAYER_HEIGHT_MIN..MAX
        p.analyze_segment_data(frames)
        assert all(f["fired_signals"] == "" for f in frames)
        assert all(f["coarse_score"] == 0.0 for f in frames)

    def test_signal_capture_does_not_change_detection_output(self):
        """The capture is observation only; segments must be unaffected."""
        p = make_processor()
        frames = stroke_frames()
        segments = p.analyze_segment_data(frames)
        assert len(segments) == 1
        assert segments[0].start_time == pytest.approx(0.0)
        assert segments[0].end_time > segments[0].start_time


class TestAdaptiveStateWiring:
    def test_state_lives_on_the_analyzer_not_the_processor(self):
        """Cross-segment state must be per-analyzer, so it cannot leak
        between videos the way class-level buffers invited."""
        p = make_processor()
        p.analyze_segment_data(stroke_frames())
        assert p._get_rally_analyzer().state is not None
        assert not hasattr(p, "arm_adaptive")

    def test_cross_segment_padding_is_applied(self):
        """The rest gap between segments must pad the threshold buffers.

        Without this the window fills with rally activity and the baseline
        drifts upward, which is the failure the padding exists to prevent.
        """
        p = make_processor()
        analyzer = p._get_rally_analyzer()
        p.analyze_segment_data(stroke_frames())

        far = stroke_frames()
        for f in far:
            f["frame"] += 5000  # a long rest gap
            f["timestamp"] = f["frame"] / FPS
        p.analyze_segment_data(far)

        # The buffer is capacity-capped, so assert it filled rather than that
        # the exact padding count is observable.
        assert analyzer.state.arm.buffer.maxlen == analyzer.state.arm.capacity
        assert len(analyzer.state.arm.buffer) == analyzer.state.arm.capacity

    def test_state_is_fresh_per_analyzer(self):
        """Two analyzers must not share thresholds."""
        a, b = make_processor(), make_processor()
        a.analyze_segment_data(stroke_frames())
        assert a._get_rally_analyzer().state is not b._get_rally_analyzer().state

    def test_last_segment_end_frame_advances(self):
        p = make_processor()
        frames = stroke_frames()
        p.analyze_segment_data(frames)
        assert p._get_rally_analyzer().state.last_segment_end_frame == frames[-1]["frame"]

    def test_all_frame_metrics_accumulate(self):
        p = make_processor()
        frames = stroke_frames()
        p.analyze_segment_data(frames)
        assert len(p.all_frame_metrics) == len(frames)
