"""Tests for the full feature store.

This file is the artifact that makes replay possible, so the properties that
matter are: every feature key survives a round trip, segment boundaries are
recoverable from frame indices alone, and the header normalizer accepts both
in-memory dicts and CSV rows.
"""

import csv
import os

import pytest

from really_rally.core.config import PROCESS_EVERY_N_FRAMES
from really_rally.utils.feature_store import (
    FEATURE_HEADERS,
    FEATURE_KEY_ORDER,
    normalize_header,
    normalize_row,
    read_feature_store,
    split_into_segments,
    write_feature_store,
)

FPS = 30.0


def make_record(frame, **overrides):
    record = {key: 0.0 for key in FEATURE_KEY_ORDER}
    record.update({
        "frame": frame,
        "time": frame / FPS,
        "timestamp": frame / FPS,
        "is_valid_pose": 1,
        "arm_energy": 0.05,
        "whip_ratio": 1.5,
        "candidate": 0,
        "state": "idle",
        "fired_signals": "",
    })
    record.update(overrides)
    return record


class TestHeaderMapping:
    def test_every_key_maps_to_a_header_and_back(self):
        """The round trip is the contract readers depend on."""
        for key, header in zip(FEATURE_KEY_ORDER, FEATURE_HEADERS):
            assert normalize_header(header) == key, f"{header} -> {normalize_header(header)}"

    def test_headers_are_unique(self):
        assert len(set(FEATURE_HEADERS)) == len(FEATURE_HEADERS)

    def test_normalizes_debug_metrics_spellings(self):
        # The exact headers base.py writes today.
        for header, expected in [
            ("Frame", "frame"), ("Time(s)", "time"), ("Arm_Energy", "arm_energy"),
            ("Smoothed_Arm", "smoothed_arm"), ("T_Arm", "t_arm"), ("Z_Arm", "z_arm"),
            ("Hip_Vel", "hip_vel"), ("Spatial_Var", "spatial_variance"),
            ("Whip_Ratio", "whip_ratio"), ("Player_Height", "player_height"),
            ("Valid_Pose", "is_valid_pose"), ("Coarse_Score", "coarse_score"),
            ("Candidate", "candidate"), ("State", "state"),
        ]:
            assert normalize_header(header) == expected

    def test_is_idempotent(self):
        for header in FEATURE_HEADERS + ["Valid_Pose", "Spatial_Var", "Time(s)"]:
            once = normalize_header(header)
            assert normalize_header(once) == once

    def test_normalize_row_rekeys_a_csv_row(self):
        row = {"Frame": "12", "Time(s)": "0.40", "Valid_Pose": "1"}
        assert normalize_row(row) == {"frame": "12", "time": "0.40", "is_valid_pose": "1"}


class TestSplitIntoSegments:
    def test_contiguous_frames_are_one_segment(self):
        records = [make_record(f) for f in (0, 2, 4, 6)]
        assert len(split_into_segments(records)) == 1

    def test_a_frame_gap_starts_a_new_segment(self):
        records = [make_record(f) for f in (0, 2, 4, 100, 102)]
        segments = split_into_segments(records)
        assert len(segments) == 2
        assert [r["frame"] for r in segments[0]] == [0, 2, 4]
        assert [r["frame"] for r in segments[1]] == [100, 102]

    def test_empty_input(self):
        assert split_into_segments([]) == []

    def test_respects_a_custom_stride(self):
        records = [make_record(f) for f in (0, 5, 10)]
        assert len(split_into_segments(records, process_every_n=5)) == 1
        assert len(split_into_segments(records, process_every_n=2)) == 3


class TestRoundTrip:
    def test_preserves_every_feature_key(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        records = [
            make_record(0, arm_energy=0.0123, whip_ratio=2.5, state="candidate"),
            make_record(2, arm_energy=0.0987, whip_ratio=1.25, state="active",
                        candidate=1, fired_signals="arm|whip"),
        ]
        write_feature_store(records, path, FPS)
        loaded = read_feature_store(path)

        assert len(loaded) == 2
        assert loaded[0]["arm_energy"] == pytest.approx(0.0123)
        assert loaded[1]["whip_ratio"] == pytest.approx(1.25)
        assert loaded[1]["state"] == "active"
        assert loaded[1]["candidate"] == 1
        assert loaded[1]["fired_signals"] == "arm|whip"

    def test_frame_is_an_integer(self, temp_dir):
        """Frame is the time source of truth; rounding it would break seeking."""
        path = os.path.join(temp_dir, "f.csv")
        write_feature_store([make_record(402)], path, FPS)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["Frame"] == "402"
        assert "." not in row["Frame"]

    def test_time_is_frame_derived(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        write_feature_store([make_record(90)], path, FPS)
        loaded = read_feature_store(path)
        assert loaded[0]["time"] == pytest.approx(3.0)
        assert loaded[0]["timestamp"] == pytest.approx(3.0)

    def test_timestamp_is_populated_for_the_detector(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        write_feature_store([make_record(60)], path, FPS)
        assert read_feature_store(path)[0]["timestamp"] == pytest.approx(2.0)

    def test_segment_index_is_written_and_survives(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        records = [make_record(f) for f in (0, 2, 4, 500, 502)]
        write_feature_store(records, path, FPS)
        loaded = read_feature_store(path)
        assert [r["segment"] for r in loaded] == [0, 0, 0, 1, 1]

    def test_segments_recoverable_from_frames_alone(self, temp_dir):
        """Replay rebuilds per-segment state from frame gaps, not the column."""
        path = os.path.join(temp_dir, "f.csv")
        records = [make_record(f) for f in (0, 2, 4, 500, 502)]
        write_feature_store(records, path, FPS)
        segments = split_into_segments(read_feature_store(path))
        assert [[r["frame"] for r in s] for s in segments] == [[0, 2, 4], [500, 502]]

    def test_missing_optional_columns_default(self, temp_dir):
        """A store written by an older/newer schema must still load."""
        path = os.path.join(temp_dir, "f.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Time(s)", "Arm_Energy"])
            writer.writerow(["10", "0.33", "0.02"])
        loaded = read_feature_store(path)
        assert loaded[0]["arm_energy"] == pytest.approx(0.02)
        assert loaded[0]["whip_ratio"] == 0.0
        assert loaded[0]["state"] == ""

    def test_empty_strings_do_not_crash(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Arm_Energy", "Candidate", "State"])
            writer.writerow(["4", "", "", ""])
        loaded = read_feature_store(path)
        assert loaded[0]["arm_energy"] == 0.0
        assert loaded[0]["candidate"] == 0

    def test_writes_all_feature_headers(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        write_feature_store([make_record(0)], path, FPS)
        with open(path) as f:
            assert next(csv.reader(f)) == FEATURE_HEADERS

    def test_replaces_previous_contents(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        write_feature_store([make_record(f) for f in (0, 2, 4)], path, FPS)
        write_feature_store([make_record(0)], path, FPS)
        assert len(read_feature_store(path)) == 1

    def test_empty_input_writes_header_only(self, temp_dir):
        path = os.path.join(temp_dir, "f.csv")
        write_feature_store([], path, FPS)
        with open(path) as f:
            assert len(list(csv.reader(f))) == 1


class TestBackwardCompatibility:
    def test_reads_an_existing_debug_metrics_file(self, temp_dir):
        """The studio can open today's artifacts, not just future ones."""
        path = os.path.join(temp_dir, "old_debug_metrics.csv")
        header = ["Frame", "Time(s)", "Arm_Energy", "Smoothed_Arm", "T_Arm", "Z_Arm",
                  "Hip_Vel", "Spatial_Var", "Action_Score", "Whip_Ratio", "Smoothed_Whip",
                  "T_Whip", "Z_Whip", "Player_Height", "Valid_Pose", "Coarse_Score",
                  "Candidate", "State"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow([120, "4.00", "0.0300", "0.0280", "0.2500", "1.1000",
                             "0.0040", "0.0200", "0.0012", "1.6700", "1.5000",
                             "0.2500", "2.0000", "0.4410", 1, "2.5000", 1, "active"])
        loaded = read_feature_store(path)
        assert loaded[0]["frame"] == 120
        assert loaded[0]["arm_energy"] == pytest.approx(0.03)
        assert loaded[0]["spatial_variance"] == pytest.approx(0.02)
        assert loaded[0]["is_valid_pose"] == 1
        assert loaded[0]["coarse_score"] == pytest.approx(2.5)
        assert loaded[0]["state"] == "active"

    def test_default_stride_matches_the_pipeline(self):
        """split_into_segments defaults must agree with PROCESS_EVERY_N_FRAMES."""
        records = [make_record(0), make_record(PROCESS_EVERY_N_FRAMES)]
        assert len(split_into_segments(records)) == 1
