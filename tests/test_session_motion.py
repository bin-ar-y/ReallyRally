"""GUI stroke selection must refer to original frames, not merged-clip positions."""

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from really_rally.core import reconstruction
from really_rally.core.detection import _stroke_peaks
from really_rally.core.session_motion import ReplayResult, video_replays, write_replay_index, read_replay_index
from really_rally.utils.subprocesses import ProcessingCancelled


def recording(folder, mode, count, *groups):
    return SimpleNamespace(
        video_path="court.mov",
        video_dir=folder,
        output_folder_name=mode,
        fps=30,
        total_frames=count,
        detected_segments=[SimpleNamespace(stroke_frames=g) for g in groups],
        serve_events=[],
    )


def test_peak_runs_keep_source_offsets_and_separate_strokes():
    rows = [{"frame": 600 + i * 2, "arm_energy": e} for i, e in enumerate([2, 4, 3, 0, 0, 1, 0, 0, 0, 0, 5, 4])]
    mask = [row["arm_energy"] > 0 for row in rows]
    assert _stroke_peaks(rows, mask, 30) == (602, 620)


def test_serve_contacts_keep_detected_side_and_failed_replay_does_not_abort(tmp_path, monkeypatch):
    processor = recording(tmp_path, "serve", 1800, (610,), (610, 910))
    processor.serve_events = [SimpleNamespace(contact_time=910 / 30, handedness="left")]
    calls = []

    def build(source, directory, frame, stroke, before, after, **options):
        calls.append((source, frame, stroke, options))
        if frame == 610:
            raise ValueError("Player not visible")
        return Path(directory) / "good"

    monkeypatch.setattr(reconstruction, "reconstruct", build)
    entries = list(video_replays(processor, "supported", lambda *_: None, Event()))
    assert [r.frame for r in entries] == [610, 910]
    assert entries[0].bundle is None and entries[0].error == "Player not visible"
    assert entries[1].bundle == tmp_path / "motion" / "good"
    assert calls[-1][0:3] == ("court.mov", 910, "serve")
    assert calls[-1][3]["anchor_kind"] == "serve_contact"
    assert calls[-1][3]["hitting_hand"] == "left"
    assert calls[-1][3]["portable_html"] is False


def test_fallback_is_a_complete_short_recording_and_never_an_invented_hit(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(reconstruction, "reconstruct", lambda *a, **kw: captured.append((a, kw)) or tmp_path)
    short = recording(tmp_path, "rally", 60)
    [entry] = video_replays(short, "unconstrained", lambda *_: None, None)
    args, options = captured[0]
    assert entry.frame == 30
    assert options["anchor_kind"] == "clip_midpoint" and args[3] == "practice"
    assert (round(args[4] * 30), round(args[5] * 30)) == (30, 29)
    long = recording(tmp_path, "rally", 600)
    assert list(video_replays(long, "unconstrained", lambda *_: None, None)) == []


def test_cancel_preserves_the_first_model_and_skips_the_next(tmp_path, monkeypatch):
    stop = Event()
    processor = recording(tmp_path, "feed", 1800, (500, 800))
    monkeypatch.setattr(reconstruction, "reconstruct", lambda *a, **kw: tmp_path)
    builds = video_replays(processor, "supported", lambda *_: None, stop)
    assert next(builds).frame == 500
    stop.set()
    with pytest.raises(ProcessingCancelled):
        next(builds)


def test_batch_index_handles_special_filenames_and_lists_unavailable_models(tmp_path):
    entries = [
        ReplayResult("<serve>.mov", 18, "Estimated serve contact", tmp_path / "A #1"),
        ReplayResult("warmup.mov", 10, "Full clip", error="No <pose>"),
    ]
    index = write_replay_index(tmp_path, entries)
    assert index.suffix == ".json"
    assert read_replay_index(index) == entries
    assert write_replay_index(tmp_path, entries) != index
