"""Tests for results-folder discovery.

The folder is the unit of work for the studio: opening one should restore a
whole session. Getting this wrong means opening the wrong recording, or opening
a folder and silently having no telemetry — so discovery is tested directly
rather than only through the GUI.
"""

import os


from really_rally.studio.results_folder import (
    KNOWN_MODES,
    find_video,
    list_recordings,
    resolve,
)


def make_recording(root, mode, stem, *, video="serve.mp4", features=True,
                   labels=False, predictions=False, telemetry=False, clips=False):
    """Build a folder shaped like the pipeline's output."""
    folder = os.path.join(root, mode, stem)
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, video), "w").close()
    if features:
        open(os.path.join(folder, "features.csv"), "w").close()
    if labels:
        open(os.path.join(folder, "labels.csv"), "w").close()
    if predictions:
        open(os.path.join(folder, "predictions.csv"), "w").close()
    if telemetry:
        open(os.path.join(folder, "telemetry.png"), "w").close()
    if clips:
        os.makedirs(os.path.join(folder, "clips"), exist_ok=True)
        open(os.path.join(folder, "clips", "serve_001.mp4"), "w").close()
    return folder


class TestFindVideo:
    def test_finds_the_only_video(self, tmp_path):
        folder = make_recording(str(tmp_path), "serve", "IMG_1")
        assert os.path.basename(find_video(folder)) == "serve.mp4"

    def test_prefers_the_merged_clip_over_raw_footage(self, tmp_path):
        """A stale source copy must not shadow the merged result."""
        folder = make_recording(str(tmp_path), "serve", "IMG_1")
        open(os.path.join(folder, "a_original.MOV"), "w").close()
        assert os.path.basename(find_video(folder)) == "serve.mp4"

    def test_ignores_videos_inside_clips(self, tmp_path):
        """clip segments are not the timeline the telemetry describes."""
        folder = make_recording(str(tmp_path), "serve", "IMG_1", clips=True)
        os.remove(os.path.join(folder, "serve.mp4"))
        assert find_video(folder) is None

    def test_empty_folder_has_no_video(self, tmp_path):
        folder = os.path.join(str(tmp_path), "empty")
        os.makedirs(folder)
        assert find_video(folder) is None

    def test_missing_folder_is_not_an_error(self, tmp_path):
        assert find_video(os.path.join(str(tmp_path), "nope")) is None


class TestResolve:
    def test_reads_a_complete_recording(self, tmp_path):
        folder = make_recording(str(tmp_path), "serve", "IMG_1165",
                                labels=True, predictions=True, telemetry=True)
        found = resolve(folder)
        assert found.is_usable and found.can_replay
        assert found.mode == "serve"
        assert found.labels and found.predictions and found.telemetry_png

    def test_video_only_folder_is_usable_but_not_replayable(self, tmp_path):
        folder = make_recording(str(tmp_path), "rally", "IMG_1", features=False)
        found = resolve(folder)
        assert found.is_usable
        assert not found.can_replay

    def test_optional_files_are_reported_as_missing(self, tmp_path):
        folder = make_recording(str(tmp_path), "rally", "IMG_1")
        missing = " ".join(resolve(folder).missing())
        assert "labels.csv" in missing
        assert "features.csv" not in missing

    def test_missing_features_is_called_out(self, tmp_path):
        folder = make_recording(str(tmp_path), "rally", "IMG_1", features=False)
        assert "features.csv" in " ".join(resolve(folder).missing())

    def test_resolves_a_mode_folder_with_one_recording(self, tmp_path):
        """Pointing at serve/ should work when it holds exactly one recording."""
        make_recording(str(tmp_path), "serve", "IMG_1165", features=True)
        found = resolve(os.path.join(str(tmp_path), "serve"))
        assert found.is_usable
        assert found.mode == "serve"
        assert os.path.basename(found.path) == "IMG_1165"

    def test_ambiguous_mode_folder_yields_no_video(self, tmp_path):
        """Two recordings must not be silently guessed between."""
        make_recording(str(tmp_path), "serve", "IMG_1")
        make_recording(str(tmp_path), "serve", "IMG_2")
        assert not resolve(os.path.join(str(tmp_path), "serve")).is_usable

    def test_missing_folder_yields_nothing_usable(self, tmp_path):
        assert not resolve(os.path.join(str(tmp_path), "nope")).is_usable

    def test_mode_is_inferred_from_the_parent_folder(self, tmp_path):
        folder = make_recording(str(tmp_path), "feed", "IMG_1")
        assert resolve(folder).mode == "feed"

    def test_unknown_parent_is_not_reported_as_a_mode(self, tmp_path):
        folder = make_recording(str(tmp_path), "somewhere", "IMG_1")
        assert resolve(folder).mode is None

    def test_explicit_mode_wins(self, tmp_path):
        folder = make_recording(str(tmp_path), "somewhere", "IMG_1")
        assert resolve(folder, mode="serve").mode == "serve"


class TestListRecordings:
    def test_lists_every_recording_under_an_output_folder(self, tmp_path):
        make_recording(str(tmp_path), "rally", "A")
        make_recording(str(tmp_path), "serve", "B")
        make_recording(str(tmp_path), "serve", "C")
        assert len(list_recordings(str(tmp_path))) == 3

    def test_known_modes_cover_the_pipeline(self):
        assert set(KNOWN_MODES) == {"rally", "serve", "feed"}

    def test_empty_output_folder(self, tmp_path):
        assert list_recordings(str(tmp_path)) == []

    def test_missing_folder(self, tmp_path):
        assert list_recordings(os.path.join(str(tmp_path), "nope")) == []

    def test_pointing_at_one_recording_returns_just_it(self, tmp_path):
        folder = make_recording(str(tmp_path), "serve", "IMG_1")
        found = list_recordings(folder)
        assert len(found) == 1 and found[0].is_usable


class TestSummary:
    def test_summary_names_the_mode_and_contents(self, tmp_path):
        folder = make_recording(str(tmp_path), "serve", "IMG_1165", labels=True)
        text = resolve(folder).summary()
        assert "IMG_1165" in text and "serve" in text and "labels" in text

    def test_summary_says_video_only_when_bare(self, tmp_path):
        folder = make_recording(str(tmp_path), "serve", "IMG_1", features=False)
        assert "video only" in resolve(folder).summary()


class TestLayoutMatchesThePipeline:
    """Discovery must agree with where base.py actually writes things."""

    def test_matches_the_processors_own_paths(self, tmp_path):
        from really_rally.processors.base import TennisRallyProcessor

        proc = TennisRallyProcessor.__new__(TennisRallyProcessor)
        proc.video_stem = "IMG_1165"
        proc.output_dir = str(tmp_path)
        proc.output_folder_name = "serve"
        proc.output_csv_name = "serve.csv"

        os.makedirs(proc.video_dir, exist_ok=True)
        for name in ("serve.mp4", "features.csv", "telemetry.png"):
            open(os.path.join(proc.video_dir, name), "w").close()

        found = resolve(proc.video_dir)
        assert found.is_usable and found.can_replay
        assert found.telemetry_png
        assert found.mode == "serve"

    def test_processor_clips_dir_is_not_mistaken_for_a_recording(self, tmp_path):
        from really_rally.processors.base import TennisRallyProcessor

        proc = TennisRallyProcessor.__new__(TennisRallyProcessor)
        proc.video_stem = "IMG_1165"
        proc.output_dir = str(tmp_path)
        proc.output_folder_name = "serve"
        proc.output_csv_name = "serve.csv"

        os.makedirs(proc.clips_dir, exist_ok=True)
        open(os.path.join(proc.clips_dir, "IMG_1165_serve_001.mp4"), "w").close()
        assert not resolve(proc.clips_dir).is_usable
