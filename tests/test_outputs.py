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


class TestResultsLayout:
    """Results are grouped per recording so a folder is self-contained.

    The point is that one recording's video, telemetry and data files sit
    together and the folder can be handed straight to the Debug Studio.
    """

    def test_results_folder_is_named_by_mode(self, synth_video, tmp_path):
        proc = TennisRallyProcessor(synth_video, str(tmp_path))
        assert proc.output_folder_name == "rally"
        assert proc.results_dir == os.path.join(str(tmp_path), "rally")

    def test_per_video_folder_is_named_by_video_stem(self, synth_video, tmp_path):
        proc = TennisRallyProcessor(synth_video, str(tmp_path))
        assert proc.video_stem == "clip"
        assert proc.video_dir == os.path.join(str(tmp_path), "rally", "clip")

    def test_clips_live_under_the_video_folder(self, synth_video, tmp_path):
        proc = TennisRallyProcessor(synth_video, str(tmp_path))
        assert proc.clips_dir == os.path.join(str(tmp_path), "rally", "clip", "clips")

    def test_serve_and_feed_use_their_own_mode_names(self, synth_video, tmp_path):
        """The folder is named after the extraction, not a generic bundle."""
        from really_rally.processors.serve import ServePracticeProcessor
        from really_rally.processors.feed import FeedPracticeProcessor

        serve = ServePracticeProcessor(synth_video, str(tmp_path))
        feed = FeedPracticeProcessor(synth_video, str(tmp_path))
        assert serve.output_folder_name == "serve"
        assert feed.output_folder_name == "feed"
        assert serve.results_dir == os.path.join(str(tmp_path), "serve")
        assert feed.results_dir == os.path.join(str(tmp_path), "feed")

    def test_artifact_names_are_short_inside_the_video_folder(self, synth_video, tmp_path):
        """The stem prefix is redundant once each video has its own folder."""
        proc = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=True)
        os.makedirs(proc.video_dir, exist_ok=True)
        for name in ("features.csv", "debug_metrics.csv", "rally.csv", "telemetry.png"):
            open(os.path.join(proc.video_dir, name), "w").close()

        assert sorted(os.path.basename(p) for p in proc.artifacts()) == [
            "debug_metrics.csv", "features.csv", "rally.csv", "telemetry.png"]

    def test_two_videos_keep_separate_artifacts(self, tmp_path):
        """The original bug: both videos wrote the same filename, so one was lost.

        Built via __new__ to avoid constructing a MediaPipe graph per processor;
        only the naming attributes matter here.
        """
        def processor_for(stem):
            proc = TennisRallyProcessor.__new__(TennisRallyProcessor)
            proc.video_stem = stem
            proc.output_dir = str(tmp_path)
            proc.output_folder_name = "rally"
            proc.output_csv_name = "rally.csv"
            proc.generate_plot = True
            return proc

        a, b = processor_for("first"), processor_for("second")
        assert a.video_dir != b.video_dir

        for proc in (a, b):
            os.makedirs(proc.video_dir, exist_ok=True)
            for name in ("features.csv", "debug_metrics.csv", "rally.csv", "telemetry.png"):
                open(os.path.join(proc.video_dir, name), "w").close()

        assert set(a.artifacts()).isdisjoint(b.artifacts())
        assert len(a.artifacts()) == 4

    def test_artifacts_lists_only_existing_files(self, synth_video, tmp_path):
        proc = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=True)
        assert proc.artifacts() == []  # nothing written yet

        os.makedirs(proc.video_dir, exist_ok=True)
        open(os.path.join(proc.video_dir, "debug_metrics.csv"), "w").close()
        assert [os.path.basename(p) for p in proc.artifacts()] == ["debug_metrics.csv"]

    def test_artifacts_omits_plot_when_not_requested(self, synth_video, tmp_path):
        without = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=False)
        os.makedirs(without.video_dir, exist_ok=True)
        for name in ("debug_metrics.csv", "telemetry.png"):
            open(os.path.join(without.video_dir, name), "w").close()
        assert all("telemetry" not in p for p in without.artifacts())

        with_plot = TennisRallyProcessor(synth_video, str(tmp_path), generate_plot=True)
        assert any("telemetry" in p for p in with_plot.artifacts())
