"""Cancellable media commands, with no orphaned encoders on GUI shutdown."""

import subprocess
import time


class ProcessingCancelled(Exception):
    """The user requested cancellation of a media operation."""


def check_cancelled(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise ProcessingCancelled()


def run_media_command(command, *, stop_event=None, timeout):
    """Like subprocess.run, but observe cancellation while draining stderr.

    communicate() avoids filling FFmpeg's stderr pipe. On cancellation or a
    timeout, reap the child before returning control to the caller for cleanup.
    """
    if stop_event is not None and stop_event.is_set():
        raise ProcessingCancelled()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise ProcessingCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                _, stderr = process.communicate(timeout=min(0.2, remaining))
                return subprocess.CompletedProcess(command, process.returncode, stderr=stderr)
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
