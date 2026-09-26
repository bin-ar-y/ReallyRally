"""Tests for detector replay from a cached feature store.

The headline property is that replaying a feature store reproduces the live
run's output. If that fails, every tuning conclusion the studio reports is
built on sand — so it is verified against a real pipeline run over a synthetic
video, not against synthetic features alone.

The other property is speed: replay must not decode video or run pose
estimation, which is the entire reason the feature store exists.
"""

import os

import numpy as np
import pytest

from really_rally.core.config import PROCESS_EVERY_N_FRAMES
from really_rally.studio.replay import (
    MODE_FEED,
    MODE_SERVE,
    DetectorReplayEngine,
    infer_fps,
)
from really_rally.utils.feature_store import FEATURE_KEY_ORDER, write_feature_store

cv2 = pytest.importorskip("cv2")

FPS = 30.0


def stroke_records(count=80, fps=FPS, burst_period=20):
    """A feature store matching a plausible stroke pattern."""
    records = []
    for i in range(count):
        burst = (i % burst_period) in (8, 9)
        frame = i * PROCESS_EVERY_N_FRAMES
        record = {key: 0.0 for key in FEATURE_KEY_ORDER}
        record.update({
            "frame": frame,
            "time": frame / fps,
            "is_valid_pose": True,
            "energy": 0.02 if burst else 0.002,
            "arm_energy": 0.12 if burst else 0.004,
            "whip_ratio": 3.0 if burst else 0.5,
            "hip_vel": 0.02 if burst else 0.002,
            "spatial_variance": 0.05 if burst else 0.01,
            "player_height": 0.4,
        })
        records.append(record)
    return records


@pytest.fixture
def store_path(temp_dir):
    path = os.path.join(temp_dir, "v_features.csv")
    write_feature_store(stroke_records(), path, FPS)
    return path


class TestBasicReplay:
    def test_produces_segments(self, store_path):
        result = DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        assert result.segments
        assert all(s.end_time > s.start_time for s in result.segments)

    def test_returns_the_configuration_that_produced_it(self, store_path):
        result = DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        assert result.config is not None
        assert result.config.fps == FPS

    def test_populates_per_frame_telemetry(self, store_path):
        result = DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        for key in ("smoothed_arm", "t_arm", "z_arm", "coarse_score", "candidate",
                    "fired_signals", "state"):
            assert key in result.records[0]

    def test_empty_store_returns_empty(self, temp_dir):
        path = os.path.join(temp_dir, "empty_features.csv")
        write_feature_store([], path, FPS)
        result = DetectorReplayEngine().replay(path, fps=FPS, total_frames=100)
        assert result.segments == []

    def test_does_not_modify_the_stored_file(self, store_path):
        before = open(store_path, "rb").read()
        DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        assert open(store_path, "rb").read() == before


class TestConfigOverrides:
    def test_overrides_change_the_result(self, store_path):
        base = DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        strict = DetectorReplayEngine().replay(
            store_path, fps=FPS, total_frames=600,
            overrides={"COARSE_CANDIDATE_THRESHOLD": 99.0})
        assert base.segments
        assert strict.segments == []

    def test_override_is_recorded_in_the_result_config(self, store_path):
        result = DetectorReplayEngine().replay(
            store_path, fps=FPS, total_frames=600,
            overrides={"ARM_ENERGY_K_MULTIPLIER": 8.0})
        assert result.config.arm_energy_k_multiplier == 8.0

    def test_replay_is_deterministic(self, store_path):
        a = DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        b = DetectorReplayEngine().replay(store_path, fps=FPS, total_frames=600)
        assert a.event_tuples == b.event_tuples

    def test_a_reanalysis_override_is_refused(self, store_path):
        with pytest.raises(ValueError, match="re-analysis"):
            DetectorReplayEngine().replay(
                store_path, fps=FPS, total_frames=600,
                overrides={"PROCESS_EVERY_N_FRAMES": 1})


class TestFrameRateInference:
    def test_infers_fps_from_frame_time_pairs(self):
        records = stroke_records()
        assert infer_fps(records) == pytest.approx(FPS, rel=1e-6)

    def test_falls_back_when_undeterminable(self):
        assert infer_fps([{"frame": 0, "time": 0.0}], default=25.0) == 25.0

    def test_explicit_fps_wins(self, store_path):
        result = DetectorReplayEngine().replay(store_path, fps=60.0, total_frames=600)
        assert result.config.fps == 60.0


class TestSegmentReconstruction:
    def test_gap_in_frames_splits_into_two_segments(self, temp_dir):
        path = os.path.join(temp_dir, "v_features.csv")
        records = stroke_records(count=40) + stroke_records(count=40)
        for record in records[40:]:
            record["frame"] += 9000
            record["time"] = record["frame"] / FPS
        write_feature_store(records, path, FPS)

        result = DetectorReplayEngine().replay(path, fps=FPS, total_frames=12000)
        # One rally per motion segment: the FSM is rebuilt per segment, so a
        # rally cannot span the gap — matching the live pipeline.
        assert result.segments_found == 2
        assert len(result.segments) >= 2

    def test_missing_stride_is_supplied_by_the_caller(self, temp_dir):
        path = os.path.join(temp_dir, "v_features.csv")
        write_feature_store(stroke_records(count=20), path, FPS)
        result = DetectorReplayEngine().replay(
            path, fps=FPS, total_frames=600, process_every_n=PROCESS_EVERY_N_FRAMES)
        assert result.segments_found == 1


class TestServeAndFeedModes:
    def test_serve_mode_returns_events_list(self, store_path):
        result = DetectorReplayEngine(mode=MODE_SERVE).replay(
            store_path, fps=FPS, total_frames=600)
        assert isinstance(result.events, list)
        assert result.mode == MODE_SERVE

    def test_feed_mode_runs(self, store_path):
        result = DetectorReplayEngine(mode=MODE_FEED).replay(
            store_path, fps=FPS, total_frames=600)
        assert result.mode == MODE_FEED
        assert isinstance(result.segments, list)

    def test_unknown_mode_falls_back_to_rally(self, store_path):
        result = DetectorReplayEngine(mode="nonsense").replay(
            store_path, fps=FPS, total_frames=600)
        assert isinstance(result.segments, list)


class TestReplayMatchesLiveRun:
    """The guarantee the whole studio depends on."""

    @pytest.fixture
    def live_run(self, tmp_path):
        """Run the real pipeline and capture what detection produced."""
        from really_rally.processors.base import TennisRallyProcessor

        video = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (320, 240))
        rng = np.random.default_rng(3)
        for i in range(180):
            # Alternating motion/rest so Pass 1 finds more than one segment.
            if (i // 30) % 2 == 0:
                frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            else:
                frame = np.full((240, 320, 3), 30, np.uint8)
            writer.write(frame)
        writer.release()

        processor = TennisRallyProcessor(str(video), str(tmp_path), generate_plot=False)
        captured = {}
        processor.export_clips = lambda segs: captured.setdefault("segments", segs) or []
        processor.process_video()

        return {
            "processor": processor,
            "segments": captured.get("segments", []),
            # Results land in <output>/<mode>/<video_stem>/.
            "features": str(tmp_path / "rally" / "clip" / "features.csv"),
            "total_frames": processor.total_frames,
        }

    def test_feature_store_is_written(self, live_run):
        assert os.path.exists(live_run["features"])

    def test_replay_reproduces_the_live_segments(self, live_run):
        live = live_run["segments"]
        result = DetectorReplayEngine().replay(
            live_run["features"], fps=FPS, total_frames=live_run["total_frames"])

        assert len(result.segments) == len(live), (
            f"replay found {len(result.segments)} segments, live found {len(live)}")
        for replayed, original in zip(result.segments, live):
            assert replayed.start_time == pytest.approx(original.start_time, abs=1e-6)
            assert replayed.end_time == pytest.approx(original.end_time, abs=1e-6)
            assert replayed.start_frame == original.start_frame
            assert replayed.end_frame == original.end_frame

    def test_replay_needs_no_video_decode(self, live_run):
        """Replay must not open the video at all — that is the point."""
        opened = []
        real_capture = cv2.VideoCapture

        def spy(*args, **kwargs):
            opened.append(args)
            return real_capture(*args, **kwargs)

        cv2.VideoCapture = spy
        try:
            DetectorReplayEngine().replay(
                live_run["features"], fps=FPS, total_frames=live_run["total_frames"])
        finally:
            cv2.VideoCapture = real_capture
        assert opened == [], f"replay opened a video capture: {opened}"

    def test_replay_is_much_faster_than_the_full_pipeline(self, live_run):
        import time

        start = time.perf_counter()
        DetectorReplayEngine().replay(
            live_run["features"], fps=FPS, total_frames=live_run["total_frames"])
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"replay took {elapsed:.2f}s — should be interactive"
