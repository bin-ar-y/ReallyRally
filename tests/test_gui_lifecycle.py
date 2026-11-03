"""User-visible lifecycle regressions, without requiring a Tk display."""

from collections import deque
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from really_rally import gui


class Widget:
    def __init__(self):
        self.options = {"state": "normal"}
        self.value = 0
        self.running = False

    def configure(self, **options):
        self.options.update(options)

    def set(self, value):
        self.value = value

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


class Harness:
    """Real event handlers with display-free widgets and a controlled worker."""

    def __init__(self):
        self.video_files = ["/videos/practice.mov"]
        self.output_dir = "/results"
        self.is_processing = self.is_loading = False
        self._close_requested = self._destroyed = False
        self._worker = None
        self._pending_result = None
        self._pending_rows = deque()
        self._motion_index = None
        self.movement = Mock()
        self._result_path = None
        self._progress = 0
        self._started_at = None
        self._poll_id = None
        self.stop_event = threading.Event()
        self._ui_queue = queue.Queue()
        self.processing_mode = SimpleNamespace(get=lambda: gui.MODES[0])
        self.plot_var = SimpleNamespace(get=lambda: True)
        self.motion_var = SimpleNamespace(get=lambda: True)
        self.planted_var = SimpleNamespace(get=lambda: True)
        for name in ("btn_add_files", "btn_clear_files", "btn_select_output", "chk_plot",
                     "chk_motion", "chk_planted", "btn_open_replays", "btn_process", "btn_stop", "btn_open_results", "btn_quit", "lbl_elapsed",
                     "lbl_progress_text", "lbl_status", "progress_bar", "activity_bar"):
            setattr(self, name, Widget())
        self.mode_buttons = [Widget() for _ in gui.MODES]
        self.after = Mock(return_value="poll")
        self.after_cancel = Mock()
        self.destroy = Mock()


for method in ("_busy", "_set_controls", "_set_status", "log", "_notify", "_poll_ui_queue",
               "_handle_ui_message", "start_processing_thread", "_start_request", "stop_processing", "request_quit",
               "_destroy_window", "run_processing", "reset_ui", "_validate_files"):
    setattr(Harness, method, getattr(gui.RallyExtractorGUI, method))
Harness._serve_summary = staticmethod(gui.RallyExtractorGUI._serve_summary)


class HeldThread:
    def __init__(self, target=None, args=(), **kwargs):
        self.target, self.args = target, args
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(gui.threading, "Thread", HeldThread)
    return Harness()


def test_start_snapshots_options_and_acknowledges_before_worker_runs(app):
    app.start_processing_thread()
    request, = app._worker.args
    app.video_files.append("/videos/later.mp4")
    app.processing_mode.get = lambda: gui.MODES[2]
    assert request.videos == ("/videos/practice.mov",)
    assert request.mode == gui.MODES[0]
    assert request.generate_plot is True
    assert request.generate_motion is True and request.footwork == "supported"
    assert app.chk_motion.options["state"] == "disabled"
    assert app.activity_bar.running
    assert app.btn_stop.options["state"] == "normal"
    assert all(button.options["state"] == "disabled" for button in app.mode_buttons)
    assert app.chk_plot.options["state"] == "disabled"
    first_worker = app._worker
    app.start_processing_thread()
    assert app._worker is first_worker  # double click cannot start another run


@pytest.mark.parametrize("status", ["success", "error", "stopped"])
def test_result_is_inline_and_controls_restore_only_after_worker_exits(app, monkeypatch, status):
    monkeypatch.setattr(gui.messagebox, "showinfo", Mock(side_effect=AssertionError("No completion modal")))
    monkeypatch.setattr(gui.subprocess, "Popen", Mock(side_effect=AssertionError("No automatic Finder")))
    app.start_processing_thread()
    app._ui_queue.put(("finished", status, "Finished", "Review or quit.", "/results/rally"))
    app._poll_ui_queue()
    assert app.is_processing
    assert app.btn_process.options["state"] == "disabled"
    app._worker.alive = False
    app._poll_ui_queue()
    assert not app.is_processing
    assert not app.activity_bar.running
    assert not app.destroy.called
    assert app.btn_process.options["state"] == "normal"
    assert app.btn_open_results.options["state"] == "normal"
    assert app.lbl_progress_text.options["text"] == "Finished"
    if status == "success":
        assert app.progress_bar.value == 1


def test_stop_is_immediate_and_late_progress_cannot_overwrite_it(app):
    app.start_processing_thread()
    app.stop_processing()
    assert app.stop_event.is_set()
    assert app.btn_stop.options["state"] == "disabled"
    app._handle_ui_message(("log", "Still processing", 0.8))
    assert app.lbl_progress_text.options["text"] == "Stopping safely…"
    assert "cleanup" in app.lbl_status.options["text"]


def test_quit_can_be_cancelled_then_waits_for_worker_cleanup(app, monkeypatch):
    app.start_processing_thread()
    question = Mock(side_effect=[False, True])
    monkeypatch.setattr(gui.messagebox, "askyesno", question)
    app.request_quit()
    assert not app.stop_event.is_set()
    assert not app._close_requested
    app.request_quit()
    assert app.stop_event.is_set()
    assert app._close_requested
    app._poll_ui_queue()
    app.destroy.assert_not_called()
    app._worker.alive = False
    app._poll_ui_queue()
    app.destroy.assert_called_once()
    app.after_cancel.assert_called_with("poll")


def test_idle_quit_needs_no_dialog(app, monkeypatch):
    monkeypatch.setattr(gui.messagebox, "askyesno", Mock(side_effect=AssertionError("Idle quit is direct")))
    app.request_quit()
    app.destroy.assert_called_once()


def test_progress_flood_leaves_time_for_other_ui_events(app):
    for i in range(1000):
        app.log(f"Frame {i}", i / 1000)
    app._poll_ui_queue()
    assert app._ui_queue.qsize() == 900
    app.after.assert_called_once()
    assert app.lbl_status.options["text"] == "Frame 99"


def test_file_loading_filters_duplicates_and_unsupported_files_off_thread(app, tmp_path):
    video = tmp_path / "a practice {session}.MOV"
    video.touch()
    other = tmp_path / "notes.txt"
    other.touch()
    app._validate_files([str(video), str(video), str(other), str(tmp_path / "missing.mp4")], ())
    kind, accepted, note = app._ui_queue.get_nowait()
    assert kind == "files"
    assert accepted == [str(video)]
    assert note.startswith("2 ")


@pytest.mark.parametrize("mode", gui.MODES)
@pytest.mark.parametrize("motion", [True, False])
def test_worker_uses_snapshot_releases_video_and_never_reads_tk(app, monkeypatch, tmp_path, mode, motion):
    def no_tk():
        raise AssertionError("Worker accessed Tk")
    app.processing_mode.get = app.plot_var.get = app.motion_var.get = app.planted_var.get = no_tk
    built = []

    class Processor:
        write_clip_manifest = staticmethod(gui.TennisRallyProcessor.write_clip_manifest)

        def __init__(self, video, output, **options):
            built.append(self)
            self.options = options
            self.cap = SimpleNamespace(release=Mock())
            self.exported_clip_rows = []
            self.serve_events = []

        def process_video(self):
            self.options["progress_callback"](0.5, "Analyzing")
            return ["clip.mp4"]

        def merge_clips(self, clips, name, dest_dir):
            assert self.options["stop_event"] is app.stop_event
            return str(tmp_path / name)

        def transfer_metadata(self, source, target):
            pass

    for name in ("TennisRallyProcessor", "ServePracticeProcessor", "FeedPracticeProcessor"):
        monkeypatch.setattr(gui, name, Processor)
    monkeypatch.setattr(gui, "probe_video", lambda _: {"width": 640, "height": 480, "fps": 30})
    monkeypatch.setattr(gui, "get_video_date", lambda _: "103126")
    replay = tmp_path / "replay.html"
    from really_rally.core.session_motion import ReplayResult
    build = Mock(return_value=[ReplayResult("practice.mov", 12, "Full clip", replay)])
    monkeypatch.setattr(gui, "video_replays", build)
    monkeypatch.setattr(gui, "write_replay_index", Mock(return_value=replay))
    request = gui.ProcessingRequest(("practice.mov",), str(tmp_path), mode, False, motion, "unconstrained")
    app.run_processing(request)
    messages = list(app._ui_queue.queue)
    assert messages[-1][:2] == ("finished", "success")
    assert len(built) == 1  # no extra model/decoder for the batch merge
    assert built[0].options["generate_plot"] is False
    assert built[0].cap.release.called
    if motion:
        build.assert_called_once()
        assert build.call_args.args[:2] == (built[0], "unconstrained")
        assert ("motion_index", replay) in messages
    else:
        build.assert_not_called()
        assert not any(msg[0] == "motion_index" for msg in messages)
