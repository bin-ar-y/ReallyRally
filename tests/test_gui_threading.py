"""Regression tests for GUI thread-safety and cancellation wiring.

Tk widgets may only be touched from the main thread, so the worker must
communicate exclusively through ``_ui_queue``. These tests exercise the
worker-facing methods on a stub object, so they need no display.
"""

import queue
import re
from pathlib import Path

import pytest

from really_rally.gui import RallyExtractorGUI

GUI_SOURCE = Path(__file__).resolve().parents[1] / "src" / "really_rally" / "gui.py"


class _Stub:
    """Stands in for the GUI; the worker-facing methods only use _ui_queue."""

    def __init__(self):
        self._ui_queue = queue.Queue()


class TestWorkerToUiQueue:
    def test_log_enqueues_without_touching_widgets(self):
        stub = _Stub()
        RallyExtractorGUI.log(stub, "Scanning Motion...", 0.25)
        assert stub._ui_queue.get_nowait() == ("log", "Scanning Motion...", 0.25)

    def test_log_progress_optional(self):
        stub = _Stub()
        RallyExtractorGUI.log(stub, "Done!")
        assert stub._ui_queue.get_nowait() == ("log", "Done!", None)

    @pytest.mark.parametrize("kind", ["info", "error", "warning"])
    def test_notify_enqueues_dialog(self, kind):
        stub = _Stub()
        RallyExtractorGUI._notify(stub, kind, "Title", "Body")
        assert stub._ui_queue.get_nowait() == (kind, "Title", "Body")

    def test_queue_preserves_order(self):
        stub = _Stub()
        RallyExtractorGUI.log(stub, "a")
        RallyExtractorGUI._notify(stub, "info", "t", "b")
        RallyExtractorGUI.log(stub, "c")
        assert [m[0] for m in [stub._ui_queue.get_nowait() for _ in range(3)]] == ["log", "info", "log"]


class TestNoThreadUnsafeCalls:
    """The worker path must never schedule Tk work or block on a subprocess."""

    @pytest.fixture(scope="class")
    def source(self):
        return GUI_SOURCE.read_text()

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
        poller = re.search(r"def _poll_ui_queue\(self\):(.*?)\n    def ", source, re.S)
        assert poller, "_poll_ui_queue not found"
        assert "self.after(" in poller.group(1), "poller must reschedule itself"
