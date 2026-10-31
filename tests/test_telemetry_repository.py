"""Tests for the studio's telemetry repository.

Seeking is the studio's core interaction, so the properties that matter are:
the frame->row lookup picks the nearest *sampled* frame (feature rows exist only
every Nth frame), spans for overlays are correct, and a key that carries no data
is not offered as a plottable series.
"""

import os

import numpy as np
import pytest

from really_rally.studio.telemetry import TelemetryRepository, _mask_to_spans
from really_rally.utils.feature_store import FEATURE_KEY_ORDER, write_feature_store

FPS = 30.0


def make_record(frame, **overrides):
    record = {key: 0.0 for key in FEATURE_KEY_ORDER}
    record.update({
        "frame": frame, "time": frame / FPS, "timestamp": frame / FPS,
        "is_valid_pose": 1, "player_height": 0.44, "state": "idle", "fired_signals": "",
    })
    record.update(overrides)
    return record


@pytest.fixture
def repo():
    records = [
        make_record(0, arm_energy=0.01, smoothed_arm=0.01, t_arm=0.25, z_arm=0.5),
        make_record(2, arm_energy=0.12, smoothed_arm=0.11, t_arm=0.25, z_arm=1.1,
                    candidate=1, state="candidate"),
        make_record(4, arm_energy=0.13, smoothed_arm=0.12, t_arm=0.25, z_arm=1.2,
                    candidate=1, state="active"),
        make_record(6, arm_energy=0.01, smoothed_arm=0.01, t_arm=0.25, z_arm=0.4,
                    state="active"),
    ]
    return TelemetryRepository(records, FPS, total_frames=300)


class TestLookup:
    def test_len_and_empty(self, repo):
        assert len(repo) == 4
        assert not repo.is_empty

    def test_empty_repository(self):
        empty = TelemetryRepository([], FPS)
        assert empty.is_empty
        assert empty.index_for_frame(10) is None
        assert empty.record_at(10) is None

    def test_exact_frame_hit(self, repo):
        assert repo.index_for_frame(4) == 2

    def test_nearest_sampled_frame(self, repo):
        """Rows exist every 2nd frame; frame 5 is nearer to 4 than to 6."""
        assert repo.frames[repo.index_for_frame(5)] == 4

    def test_exact_tie_resolves_to_the_earlier_sample(self, repo):
        """Frame 1 sits between samples 0 and 2; pick one deterministically."""
        assert repo.frames[repo.index_for_frame(1)] == 0
        assert repo.frames[repo.index_for_frame(3)] == 2

    def test_clamps_before_the_first_sample(self, repo):
        assert repo.index_for_frame(0) == 0

    def test_clamps_past_the_end(self, repo):
        assert repo.frames[repo.index_for_frame(99999)] == 6

    def test_value_at_returns_the_row_value(self, repo):
        assert repo.value_at("smoothed_arm", 2) == pytest.approx(0.11)

    def test_value_at_default_when_missing(self, repo):
        assert repo.value_at("not_a_key", 2, default=-1.0) == -1.0

    def test_is_candidate_at(self, repo):
        assert repo.is_candidate_at(2) is True
        assert repo.is_candidate_at(0) is False

    def test_state_at(self, repo):
        assert repo.state_at(4) == "active"

    def test_total_frames_inferred_when_not_given(self):
        repo = TelemetryRepository([make_record(0), make_record(2)], FPS)
        assert repo.total_frames == 3


class TestSeries:
    def test_series_returns_times_and_values(self, repo):
        times, values = repo.series("z_arm")
        assert list(times) == pytest.approx([0.0, 2 / FPS, 4 / FPS, 6 / FPS])
        assert list(values) == pytest.approx([0.5, 1.1, 1.2, 0.4])

    def test_series_is_cached(self, repo):
        assert repo.series("z_arm")[1] is repo.series("z_arm")[1]

    def test_missing_key_yields_zeros_not_an_error(self, repo):
        _, values = repo.series("racket_confidence")
        assert list(values) == [0.0, 0.0, 0.0, 0.0]

    def test_has_signal_detects_real_data(self, repo):
        assert repo.has_signal("z_arm") is True

    def test_has_signal_rejects_all_zero_column(self, repo):
        assert repo.has_signal("racket_confidence") is False

    def test_plottable_keys_excludes_empty_columns(self, repo):
        keys = repo.plottable_keys()
        assert "z_arm" in keys
        assert "racket_confidence" not in keys

    def test_plottable_keys_filters_by_role(self, repo):
        from really_rally.utils.feature_store import ROLE_THRESHOLD
        keys = repo.plottable_keys(ROLE_THRESHOLD)
        assert "t_arm" in keys
        assert "z_arm" not in keys

    def test_future_telemetry_columns_are_surfaced(self):
        """A racket/ball column needs no code change to become plottable."""
        records = [make_record(0, **{"racket_confidence": 0.8})]
        repo = TelemetryRepository(records, FPS)
        assert "racket_confidence" in repo.plottable_keys()


class TestSpans:
    def test_mask_to_spans_basic(self):
        frames = np.array([0, 2, 4, 6, 8])
        mask = np.array([False, True, True, False, True])
        spans = _mask_to_spans(frames, mask, FPS)
        assert spans == [(2 / FPS, 4 / FPS), (8 / FPS, 8 / FPS)]

    def test_mask_to_spans_empty(self):
        assert _mask_to_spans(np.array([]), np.array([], dtype=bool), FPS) == []

    def test_mask_to_spans_all_false(self):
        frames = np.array([0, 2])
        assert _mask_to_spans(frames, np.array([False, False]), FPS) == []

    def test_mask_to_spans_trailing_run_closes_at_end(self):
        frames = np.array([0, 2, 4])
        spans = _mask_to_spans(frames, np.array([False, True, True]), FPS)
        assert spans == [(2 / FPS, 4 / FPS)]

    def test_candidate_spans(self, repo):
        assert repo.candidate_spans() == [(2 / FPS, 4 / FPS)]

    def test_state_spans_group_contiguous_runs(self, repo):
        spans = repo.state_spans()
        assert [(s, e, name) for s, e, name in spans] == [
            (0.0, 2 / FPS, "idle"),
            (2 / FPS, 4 / FPS, "candidate"),
            (4 / FPS, 6 / FPS, "active"),
        ]


class TestFromFeaturesCsv:
    def test_loads_a_written_store(self, temp_dir):
        path = os.path.join(temp_dir, "v_features.csv")
        write_feature_store([make_record(0, arm_energy=0.05), make_record(2, arm_energy=0.06)],
                            path, FPS)
        repo = TelemetryRepository.from_features_csv(path, FPS, total_frames=100)
        assert len(repo) == 2
        assert repo.series("arm_energy")[1][0] == pytest.approx(0.05)
        assert repo.total_frames == 100

    def test_time_is_frame_derived(self, temp_dir):
        path = os.path.join(temp_dir, "v_features.csv")
        write_feature_store([make_record(60)], path, FPS)
        repo = TelemetryRepository.from_features_csv(path, FPS)
        assert repo.times[0] == pytest.approx(2.0)

    def test_duration_uses_video_length(self, temp_dir):
        path = os.path.join(temp_dir, "v_features.csv")
        write_feature_store([make_record(0)], path, FPS)
        repo = TelemetryRepository.from_features_csv(path, FPS, total_frames=300)
        assert repo.duration == pytest.approx(10.0)
