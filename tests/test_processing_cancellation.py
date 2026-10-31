"""Cancellation reaps encoders and preserves previously completed outputs."""

import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from really_rally.processors import base
from really_rally.utils import subprocesses
from really_rally.utils.subprocesses import ProcessingCancelled, run_media_command


def test_media_command_drains_stderr():
    result = run_media_command([sys.executable, "-c", "import sys; sys.stderr.write('x' * 200000)"], timeout=5)
    assert result.returncode == 0
    assert len(result.stderr) == 200000


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_and_timeout_reap_child(monkeypatch, cancel):
    children = []
    original = subprocess.Popen

    def record(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocesses.subprocess, "Popen", record)
    stop = threading.Event()
    timer = threading.Timer(0.15, stop.set)
    if cancel:
        timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(ProcessingCancelled if cancel else subprocess.TimeoutExpired):
            run_media_command([sys.executable, "-c", "import time; time.sleep(30)"],
                              stop_event=stop, timeout=5 if cancel else 0.15)
    finally:
        if cancel:
            timer.join()
    assert time.monotonic() - start < 4
    assert len(children) == 1 and children[0].poll() is not None


def test_cancelled_merge_preserves_previous_summary_and_removes_partial_files(tmp_path, monkeypatch):
    output = tmp_path / "summary.mp4"
    output.write_bytes(b"previous complete summary")
    processor = base.TennisRallyProcessor.__new__(base.TennisRallyProcessor)
    processor.stop_event = threading.Event()

    def cancel(command, **kwargs):
        from pathlib import Path
        Path(command[-1]).write_bytes(b"partial merge")
        raise ProcessingCancelled()

    monkeypatch.setattr(base, "run_media_command", cancel)
    with pytest.raises(ProcessingCancelled):
        processor.merge_clips(["clip.mp4"], output.name, dest_dir=str(tmp_path))
    assert output.read_bytes() == b"previous complete summary"
    assert list(tmp_path.iterdir()) == [output]


def test_cancelled_export_keeps_completed_clip_and_truthful_manifest(tmp_path, monkeypatch):
    processor = base.TennisRallyProcessor.__new__(base.TennisRallyProcessor)
    processor.output_dir = str(tmp_path)
    processor.output_folder_name = "rally"
    processor.output_csv_name = "rally.csv"
    processor.video_stem = "practice"
    processor.video_path = "practice.mov"
    processor.clip_prefix = "_rally_"
    processor.target_format = None
    processor.progress_callback = None
    processor.stop_event = threading.Event()
    calls = []

    def encode(command, **kwargs):
        from pathlib import Path
        path = Path(command[-1])
        path.write_bytes(b"video")
        calls.append(path)
        if len(calls) == 2:
            processor.stop_event.set()
            raise ProcessingCancelled()
        return subprocess.CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr(base, "run_media_command", encode)
    segment = SimpleNamespace(start_time=1, end_time=2, duration=1)
    clips = processor.export_clips([segment, segment])
    assert clips == [str(calls[0])]
    assert calls[0].exists() and not calls[1].exists()
    assert len(processor.exported_clip_rows) == 1


def test_telemetry_chart_can_render_in_worker_without_gui_backend(tmp_path):
    from really_rally.utils.telemetry import generate_telemetry_plot
    from matplotlib._pylab_helpers import Gcf
    existing_figures = dict(Gcf.figs)
    path = tmp_path / "chart.png"
    worker = threading.Thread(target=generate_telemetry_plot, args=(
        [{"frame": i, "is_valid_pose": True, "smoothed_arm": i / 10} for i in range(10)], 30, str(path)))
    worker.start()
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert path.is_file() and path.stat().st_size > 1000
    assert dict(Gcf.figs) == existing_figures
