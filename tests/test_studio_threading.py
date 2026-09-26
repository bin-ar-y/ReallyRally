"""Thread-safety guards for the Debug Studio.

Same invariant the extraction GUI is held to: Tk is touched only from the main
thread. Workers post to a queue and the ``after()`` poller drains it. A source
lint is used rather than a runtime test because the failure mode is a stray call
added later, not a logic error now.
"""

import queue
import re
from pathlib import Path

import pytest

from really_rally.studio.controller import StudioController

STUDIO_DIR = Path(__file__).resolve().parents[1] / "src" / "really_rally" / "studio"
APP_SOURCE = STUDIO_DIR / "app.py"


class _Stub:
    """Stands in for the app; the worker-facing methods only use the queue."""

    def __init__(self):
        self._ui_queue = queue.Queue()


class TestWorkerToUiQueue:
    def test_log_enqueues_without_touching_widgets(self):
        stub = _Stub()
        StudioController.log(stub, "Loading telemetry…", 0.3)
        assert stub._ui_queue.get_nowait() == ("log", "Loading telemetry…", 0.3)

    def test_log_progress_optional(self):
        stub = _Stub()
        StudioController.log(stub, "Ready.")
        assert stub._ui_queue.get_nowait() == ("log", "Ready.", None)

    @pytest.mark.parametrize("kind", ["info", "error", "warning"])
    def test_notify_enqueues_dialog(self, kind):
        stub = _Stub()
        StudioController._notify(stub, kind, "Title", "Body")
        assert stub._ui_queue.get_nowait() == (kind, "Title", "Body")

    def test_post_enqueues_arbitrary_payload(self):
        stub = _Stub()
        StudioController.post(stub, "dataset", "/v.mp4", {"fps": 30.0}, None)
        assert stub._ui_queue.get_nowait() == ("dataset", "/v.mp4", {"fps": 30.0}, None)

    def test_queue_preserves_order(self):
        stub = _Stub()
        StudioController.log(stub, "a")
        StudioController._notify(stub, "info", "t", "b")
        StudioController.log(stub, "c")
        drained = [stub._ui_queue.get_nowait() for _ in range(3)]
        assert [m[0] for m in drained] == ["log", "info", "log"]


class TestDrainMessages:
    def test_drains_everything_then_returns_empty(self):
        controller = StudioController()
        controller.log("a")
        controller.log("b")
        assert [m[0] for m in controller.drain_messages()] == ["log", "log"]
        assert controller.drain_messages() == []

    def test_does_not_raise_when_empty(self):
        assert StudioController().drain_messages() == []


class TestLatestFrameSlot:
    """Frames must not queue: the UI would fall behind the playhead."""

    def test_returns_none_before_any_frame(self):
        assert StudioController().take_frame() is None

    def test_take_returns_the_frame_once(self):
        controller = StudioController()
        controller._on_frame(5, "image")
        assert controller.take_frame() == (5, "image")
        assert controller.take_frame() is None

    def test_only_the_newest_frame_survives(self):
        controller = StudioController()
        for frame in range(20):
            controller._on_frame(frame, f"image{frame}")
        assert controller.take_frame() == (19, "image19")
        assert controller.take_frame() is None


class TestNoThreadUnsafeCalls:
    @pytest.fixture(scope="class")
    def source(self):
        return APP_SOURCE.read_text()

    def test_no_after_from_worker(self, source):
        # after() is only legitimate inside the main-thread queue poller.
        offenders = [
            (i, line.strip())
            for i, line in enumerate(source.splitlines(), 1)
            if ".after(" in line and "_poll_ui_queue" not in line
        ]
        assert offenders == [], f"after() called outside the poller: {offenders}"

    def test_no_blocking_subprocess_call(self, source):
        assert "subprocess.call(" not in source, "use Popen so the worker never blocks"

    def test_poller_reschedules_itself(self, source):
        poller = re.search(r"def _poll_ui_queue\(self\).*?:(.*?)\n    def ", source, re.S)
        assert poller, "_poll_ui_queue not found"
        assert "self.after(" in poller.group(1), "poller must reschedule itself"


class TestControllerHasNoTkDependency:
    """The controller must stay importable and testable without a display."""

    def test_controller_source_imports_no_gui_toolkit(self):
        source = (STUDIO_DIR / "controller.py").read_text()
        for banned in ("customtkinter", "tkinter", "matplotlib.backends"):
            assert banned not in source, f"controller must not depend on {banned}"

    def test_navigator_source_imports_no_gui_toolkit(self):
        source = (STUDIO_DIR / "video_navigator.py").read_text()
        assert "customtkinter" not in source
        assert "tkinter" not in source
