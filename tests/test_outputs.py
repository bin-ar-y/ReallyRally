"""Tests for output artifact naming and manifest writing.

Regression guard for two data-loss bugs: batch runs sharing one output
directory overwrote each video's telemetry with the last video's, and the clip
manifest was append-mode so re-running duplicated every row forever.
"""

import csv
import os

import numpy as np
import pytest

from really_rally.processors.base import TennisRallyProcessor

cv2 = pytest.importorskip("cv2")

FPS = 30.0


@pytest.fixture
def synth_video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    for i in range(30):
        writer.write(np.full((120, 160, 3), i * 8 % 255, dtype=np.uint8))
    writer.release()
    return str(path)


class TestClipManifest:
    def test_writes_header_and_rows(self, tmp_path):
        path = str(tmp_path / "m.csv")
        rows = [["vid", "vid_001.mp4", "1.00", "2.00", "1.00"]]
        TennisRallyProcessor.write_clip_manifest(path, rows)
        with open(path) as f:
            got = list(csv.reader(f))
        assert got[0] == ["Source Video", "Clip", "Start Time", "End Time", "Duration"]
        assert got[1] == rows[0]

    def test_replaces_instead_of_appending(self, tmp_path):
        """Re-running must not duplicate rows from the previous run."""
        path = str(tmp_path / "m.csv")
        first = [["vid", "vid_001_aaaa.mp4", "1.00", "2.00", "1.00"]]
        second = [["vid", "vid_001_bbbb.mp4", "1.00", "2.00", "1.00"]]

        TennisRallyProcessor.write_clip_manifest(path, first)
        TennisRallyProcessor.write_clip_manifest(path, second)

        with open(path) as f:
            got = list(csv.reader(f))
        assert len(got) == 2, f"expected header + 1 row, got {len(got)}"
        assert got[1][1] == "vid_001_bbbb.mp4"

    def test_empty_rows_still_writes_header(self, tmp_path):
        path = str(tmp_path / "m.csv")
        TennisRallyProcessor.write_clip_manifest(path, [])
        with open(path) as f:
            got = list(csv.reader(f))
        assert len(got) == 1

    def test_creates_missing_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "deeper" / "m.csv")
        TennisRallyProcessor.write_clip_manifest(path, [])
        assert os.path.isfile(path)


class TestPerVideoNaming:
    def test_artifact_names_include_the_video_stem(self, synth_video, tmp_path):
        proc = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=False)
        assert proc.video_stem == "clip"
        assert all(proc.video_stem in name for name in
                   ["clip_debug_metrics.csv", f"clip_{proc.output_csv_name}"])

    def test_two_videos_in_one_folder_keep_separate_artifacts(self, tmp_path):
        """The bug: both videos wrote debug_metrics.csv, so one was lost.

        Built via __new__ to avoid constructing a MediaPipe graph per processor;
        only the naming attributes matter here.
        """
        def processor_for(stem):
            proc = TennisRallyProcessor.__new__(TennisRallyProcessor)
            proc.video_stem = stem
            proc.output_dir = str(tmp_path)
            proc.output_csv_name = "rallies.csv"
            proc.generate_plot = True
            return proc

        a, b = processor_for("first"), processor_for("second")
        for proc in (a, b):
            for name in (f"{proc.video_stem}_debug_metrics.csv",
                         f"{proc.video_stem}_rallies.csv",
                         f"{proc.video_stem}_telemetry.png"):
                open(os.path.join(str(tmp_path), name), "w").close()

        assert set(a.artifacts()).isdisjoint(b.artifacts())
        assert len(a.artifacts()) == 3

    def test_artifacts_lists_only_existing_files(self, synth_video, tmp_path):
        proc = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=True)
        assert proc.artifacts() == []  # nothing written yet

        with open(os.path.join(str(tmp_path), "clip_debug_metrics.csv"), "w") as f:
            f.write("x")
        assert [os.path.basename(p) for p in proc.artifacts()] == ["clip_debug_metrics.csv"]

    def test_artifacts_omits_plot_when_not_requested(self, synth_video, tmp_path):
        for name in ("clip_debug_metrics.csv", "clip_telemetry.png"):
            open(os.path.join(str(tmp_path), name), "w").close()

        without = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=False)
        assert all("telemetry" not in p for p in without.artifacts())

        with_plot = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=True)
        assert any("telemetry" in p for p in with_plot.artifacts())
