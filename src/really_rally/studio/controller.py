"""Worker/queue plumbing for the studio.

Everything slow — decoding, CSV loading, and later detector replay — happens on
worker threads. Workers never touch a widget; they post to ``_ui_queue`` (or the
latest-frame slot), and the Tk main thread drains both from its ``after()``
poller. This mirrors the proven pattern in ``gui.py``.

Decoded frames do *not* go through the queue. A queue of frames would let the UI
fall further behind the playhead the faster the user scrubs, so frames go into a
single "latest" slot instead: whatever arrived most recently wins and anything
older is dropped without ever being rendered.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Callable, List, Optional, Tuple

from really_rally.studio.telemetry import TelemetryRepository
from really_rally.studio.video_navigator import VideoNavigator
from really_rally.utils.media_io import probe_video


class StudioController:
    """Owns the worker threads and the two worker->UI channels."""

    def __init__(self):
        self._ui_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()

        self._navigator: Optional[VideoNavigator] = None
        self._navigator_thread: Optional[threading.Thread] = None
        self._job_thread: Optional[threading.Thread] = None

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[Tuple[int, object]] = None
        self._has_new_frame = False

        self.video_path: Optional[str] = None
        self.video_meta: dict = {}
        self.repository: Optional[TelemetryRepository] = None
        self.current_frame = 0

    # -- worker -> UI channels (safe to call from any thread) -------------

    def log(self, message: str, progress: Optional[float] = None) -> None:
        self._ui_queue.put(("log", message, progress))

    def _notify(self, kind: str, title: str, message: str) -> None:
        self._ui_queue.put((kind, title, message))

    def post(self, kind: str, *payload) -> None:
        """Post an arbitrary message for the UI poller to handle."""
        self._ui_queue.put((kind,) + payload)

    def drain_messages(self) -> List[tuple]:
        """Take every pending message. Main thread only."""
        messages = []
        while True:
            try:
                messages.append(self._ui_queue.get_nowait())
            except queue.Empty:
                return messages

    def take_frame(self) -> Optional[Tuple[int, object]]:
        """Newest decoded frame, or None. Never blocks; never backlogs."""
        with self._frame_lock:
            if not self._has_new_frame:
                return None
            frame = self._latest_frame
            self._has_new_frame = False
        return frame

    # -- dataset loading --------------------------------------------------

    def open_dataset(self, video_path: str, features_path: Optional[str] = None) -> None:
        """Load the video and its telemetry on a worker thread."""
        if self._job_thread and self._job_thread.is_alive():
            self._notify("warning", "Busy", "A job is already running.")
            return
        self._job_thread = threading.Thread(
            target=self._load_dataset, args=(video_path, features_path), daemon=True)
        self._job_thread.start()

    def _load_dataset(self, video_path: str, features_path: Optional[str]) -> None:
        try:
            self.log("Probing video…", 0.0)
            meta = probe_video(video_path)

            repository = None
            if features_path and os.path.exists(features_path):
                self.log("Loading telemetry…", 0.3)
                repository = TelemetryRepository.from_features_csv(
                    features_path, meta["fps"], meta["frame_count"])
                if repository.is_empty:
                    repository = None
                    self.log("Telemetry file was empty.", 0.5)
            else:
                self.log("No telemetry file selected — video only.", 0.5)

            self._start_navigator(video_path, meta)
            self.post("dataset", video_path, meta, repository)
        except Exception as exc:  # surfaced in a dialog, not swallowed
            self._notify("error", "Load failed", str(exc))

    def _start_navigator(self, video_path: str, meta: dict) -> None:
        self._stop_navigator()
        self.stop_event.clear()
        navigator = VideoNavigator(
            video_path,
            meta["fps"],
            meta["frame_count"],
            on_frame=self._on_frame,
            stop_event=self.stop_event,
            on_error=lambda message: self._notify("error", "Decode error", message),
        )
        self._navigator = navigator
        self._navigator_thread = threading.Thread(target=navigator.run, daemon=True)
        self._navigator_thread.start()

    def _on_frame(self, frame: int, image) -> None:
        with self._frame_lock:
            self._latest_frame = (frame, image)
            self._has_new_frame = True

    def _stop_navigator(self) -> None:
        if self._navigator:
            self._navigator.stop()
            self._navigator = None
        # The navigator loop exit is what releases the capture; the thread is a
        # daemon so a slow decode cannot block shutdown.

    # -- navigation -------------------------------------------------------

    def request_seek(self, frame: int) -> None:
        self.current_frame = frame
        if self._navigator:
            self._navigator.request_frame(frame)

    def step_frames(self, delta: int) -> None:
        """Move relative to the last decoded frame (frame stepping)."""
        base = self.current_frame
        if self._navigator and self._navigator.current_frame is not None:
            base = self._navigator.current_frame
        self.request_seek(max(0, base + delta))

    def step_seconds(self, delta: float) -> None:
        fps = self.video_meta.get("fps") or 30.0
        self.step_frames(int(round(delta * fps)))

    def set_playing(self, playing: bool) -> None:
        if self._navigator:
            self._navigator.set_playing(playing)

    def set_speed(self, speed: float) -> None:
        if self._navigator:
            self._navigator.set_speed(speed)

    # -- shutdown ---------------------------------------------------------

    def shutdown(self) -> None:
        self.stop_event.set()
        if self._navigator:
            self._navigator.stop()

    # -- helpers for later phases ----------------------------------------

    def run_job(self, target: Callable, *args, **kwargs) -> None:
        """Run a background job, posting ('job-done', result) or an error."""
        if self._job_thread and self._job_thread.is_alive():
            self._notify("warning", "Busy", "A job is already running.")
            return
        self.log("Building 3D reconstruction…")
        def _runner():
            try:
                result = target(*args, **kwargs)
                self.post("job-done", result)
            except Exception as exc:
                self._notify("error", "Job failed", str(exc))
        self._job_thread = threading.Thread(target=_runner, daemon=True)
        self._job_thread.start()
