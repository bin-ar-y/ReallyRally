"""Video frame source running on a worker thread.

Dragging the playhead emits a seek on every mouse-move. Serving each one would
queue hundreds of decodes the user has already moved past, and the UI would lag
further behind the longer they dragged. So requests are collapsed into a single
"latest frame" slot: the worker always decodes the newest request and stale ones
are never read at all.

Decoding happens here, never in a Tk callback. Frames are handed back through
``on_frame``, which the controller wires to a queue.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2


class VideoNavigator:
    """Seeks and decodes frames, one worker thread, latest request wins."""

    def __init__(
        self,
        video_path: str,
        fps: float,
        total_frames: int,
        on_frame: Callable[[int, object], None],
        stop_event: Optional[threading.Event] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.video_path = video_path
        self.fps = fps
        self.total_frames = total_frames
        self._on_frame = on_frame
        self._on_error = on_error
        self._stop = stop_event or threading.Event()

        self._lock = threading.Lock()
        self._pending: Optional[int] = None
        self._wake = threading.Event()

        self._playing = False
        self._speed = 1.0
        self._current: Optional[int] = None

    # -- worker-facing API (thread-safe) ---------------------------------

    def request_frame(self, frame: int) -> None:
        """Ask for a frame. Overwrites any unserved request."""
        frame = max(0, int(frame))
        if self.total_frames:
            frame = min(frame, self.total_frames - 1)
        with self._lock:
            self._pending = frame
        self._wake.set()

    def request_time(self, time_sec: float) -> None:
        self.request_frame(int(round(time_sec * self.fps)))

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = bool(playing)
        self._wake.set()

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def current_frame(self) -> Optional[int]:
        """Last frame successfully decoded, or None before the first."""
        return self._current

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self._speed = max(0.05, float(speed))

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # -- internals --------------------------------------------------------

    def _take_pending(self) -> Optional[int]:
        with self._lock:
            frame = self._pending
            self._pending = None
            self._wake.clear()
        return frame

    def _snapshot(self):
        with self._lock:
            return self._playing, self._speed

    def _read(self, cap, frame: int):
        """Decode one frame. Sequential reads skip the seek entirely.

        ``cap.set()`` then ``cap.read()`` is the reliable seek pattern (some
        backends misbehave with ``retrieve()`` after seeking), but during
        playback the next frame is usually the current one plus one, and
        avoiding a redundant seek keeps playback smooth on long clips.
        """
        if self._current is None or frame != self._current + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        return cap.read()

    def run(self) -> None:
        """Blocking worker loop. Returns when stop_event is set."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._fail(f"Could not open video: {self.video_path}")
            return

        try:
            while not self._stop.is_set():
                frame = self._take_pending()
                playing, speed = self._snapshot()

                if frame is None:
                    if not playing:
                        # Idle: sleep until a request arrives, but stay
                        # interruptible by stop().
                        self._wake.wait(0.05)
                        continue
                    frame = 0 if self._current is None else self._current + 1

                if self.total_frames and frame >= self.total_frames:
                    if playing:
                        self.set_playing(False)
                        continue
                    frame = self.total_frames - 1

                started = time.monotonic()
                ok, image = self._read(cap, frame)
                if not ok:
                    # A single unreadable frame (corrupt GOP, seek past EOF)
                    # should not kill the thread; report and stop playback.
                    self._fail(f"Could not decode frame {frame}.")
                    if playing:
                        self.set_playing(False)
                    continue

                self._current = frame
                self._on_frame(frame, image)

                if playing:
                    interval = 1.0 / max(1e-3, self.fps * speed)
                    self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
        finally:
            cap.release()

    def _fail(self, message: str) -> None:
        if self._on_error:
            self._on_error(message)
