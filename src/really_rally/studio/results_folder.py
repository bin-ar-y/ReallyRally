"""Locating the artifacts of one recording inside a results folder.

The pipeline writes everything about a recording into
``<output>/<mode>/<video_stem>/``. This module is what turns that folder back
into a session: which video, which feature store, and which of the optional
studio files (labels, predictions, experiments) are present.

Kept separate from the app so the discovery rules are testable without a
display, and so the studio and any future CLI share one definition of "what is
in a results folder".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# The mode folders the pipeline produces, in the order they should be probed
# when the caller points at a parent directory.
KNOWN_MODES = ("rally", "serve", "feed")

# Preferred video filenames inside a per-recording folder, best first.
_PREFERRED_VIDEO_STEMS = ("rally", "serve", "feeds", "feed", "output")

# Subfolders of a recording that hold derived media rather than the recording
# itself. Pointing at one is an easy mistake, and treating an individual clip as
# the recording would show a video whose timeline does not match the telemetry.
RESERVED_SUBFOLDERS = ("clips",)


@dataclass
class ResultsFolder:
    """What one results folder contains."""
    path: str
    video: Optional[str] = None
    features: Optional[str] = None
    debug_metrics: Optional[str] = None
    telemetry_png: Optional[str] = None
    labels: Optional[str] = None
    predictions: Optional[str] = None
    experiments: Optional[str] = None
    mode: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        """A video is the minimum; telemetry is what makes replay possible."""
        return self.video is not None

    @property
    def can_replay(self) -> bool:
        return self.video is not None and self.features is not None

    def missing(self) -> List[str]:
        """What a full session would have but this folder lacks."""
        absent = []
        if self.video is None:
            absent.append("a video")
        if self.features is None:
            absent.append("features.csv (re-run extraction with this version)")
        if self.labels is None:
            absent.append("labels.csv (nothing labelled yet)")
        return absent

    def summary(self) -> str:
        parts = [os.path.basename(self.path)]
        if self.mode:
            parts.append(f"{self.mode} mode")
        found = [name for name in ("features", "labels", "predictions", "experiments")
                 if getattr(self, name)]
        parts.append(", ".join(found) if found else "video only")
        return " · ".join(parts)


def _first_existing(*paths: Optional[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def find_video(folder: str) -> Optional[str]:
    """The recording's video, preferring the merged clip over raw footage.

    A `clips/` subfolder holds the individual segments, which are not what the
    studio should open — the merged video is the one that lines up with the
    single telemetry timeline.
    """
    entries = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    videos = [e for e in entries
              if e.lower().endswith(VIDEO_SUFFIXES)
              and os.path.isfile(os.path.join(folder, e))]
    if not videos:
        return None

    for stem in _PREFERRED_VIDEO_STEMS:
        for name in videos:
            if os.path.splitext(name)[0].lower() == stem:
                return os.path.join(folder, name)
    return os.path.join(folder, videos[0])


def resolve(folder: str, mode: Optional[str] = None) -> ResultsFolder:
    """Describe the results folder at ``folder``.

    Accepts either a per-recording folder or a mode/parent folder, in which case
    the single recording inside it is used. Pointing at a parent with several
    recordings is ambiguous, so the caller gets the folder back with no video
    rather than an arbitrary pick.
    """
    folder = os.path.abspath(folder)
    result = ResultsFolder(path=folder, mode=mode)

    if not os.path.isdir(folder):
        return result

    # A derived-media subfolder is never itself a recording: its videos are
    # segments whose timeline does not match the telemetry. Resolve onto the
    # recording that owns it, and refuse rather than fall through if there isn't
    # one — opening a single clip as if it were the recording would be silent
    # nonsense.
    if os.path.basename(folder) in RESERVED_SUBFOLDERS:
        parent = os.path.dirname(folder)
        if find_video(parent):
            return resolve(parent, mode=mode)
        return result

    video = find_video(folder)
    if video is None:
        # Not a recording folder — maybe a mode or output folder above one.
        candidates = _recording_subfolders(folder)
        if len(candidates) == 1:
            nested = resolve(candidates[0], mode=mode or os.path.basename(folder))
            nested.path = candidates[0]
            return nested
        return result

    result.video = video
    if mode is None:
        parent = os.path.basename(os.path.dirname(folder))
        result.mode = parent if parent in KNOWN_MODES else None
    result.features = _first_existing(os.path.join(folder, "features.csv"))
    result.debug_metrics = _first_existing(os.path.join(folder, "debug_metrics.csv"))
    result.telemetry_png = _first_existing(os.path.join(folder, "telemetry.png"))
    result.labels = _first_existing(os.path.join(folder, "labels.csv"))
    result.predictions = _first_existing(os.path.join(folder, "predictions.csv"))
    result.experiments = _first_existing(os.path.join(folder, "experiments.csv"))
    return result


def _recording_subfolders(folder: str) -> List[str]:
    """Immediate subfolders that look like a recording's results."""
    out = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isdir(path) and find_video(path):
            out.append(path)
    return out


def list_recordings(parent: str) -> List[ResultsFolder]:
    """Every recording under a mode or output folder."""
    parent = os.path.abspath(parent)
    if not os.path.isdir(parent):
        return []
    if find_video(parent):
        return [resolve(parent)]

    found = []
    for mode in KNOWN_MODES:
        mode_dir = os.path.join(parent, mode)
        if os.path.isdir(mode_dir):
            for sub in _recording_subfolders(mode_dir):
                found.append(resolve(sub, mode=mode))
    if found:
        return found
    return [resolve(sub) for sub in _recording_subfolders(parent)]
