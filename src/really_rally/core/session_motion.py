"""Build optional 3D replays from an extraction run's original source frames."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from really_rally.utils.media_io import time_to_frame
from really_rally.utils.subprocesses import check_cancelled, ProcessingCancelled


@dataclass(frozen=True)
class ReplayResult:
    source: str
    frame: int
    label: str
    bundle: Optional[Path] = None
    error: str = ""


def video_replays(processor, footwork, progress, stop_event):
    """Yield each completed bundle immediately so Stop retains earlier models."""
    from really_rally.core.reconstruction import reconstruct

    frames = sorted(
        {int(f) for s in processor.detected_segments for f in s.stroke_frames if 0 <= f < processor.total_frames}
    )
    mode = processor.output_folder_name
    stroke, anchor, label = {
        "serve": ("serve", "serve_contact", "Estimated serve contact"),
        "feed": ("feed stroke", "motion_peak", "Multiball motion peak"),
        "rally": ("rally stroke", "motion_peak", "Rally motion peak"),
    }[mode]
    hands = {}
    if mode == "serve":
        hands = {
            time_to_frame(ev.contact_time, processor.fps, processor.total_frames): ev.handedness
            for ev in processor.serve_events
        }
    before, after = 2.0, 0.8
    if not frames and processor.fps > 0 and 0 < processor.total_frames <= 10 * processor.fps:
        # A prepared stroke clip can be shorter than the session detector's
        # confirmation window. Show all its frames without inventing contact.
        mid = processor.total_frames // 2
        frames = [mid]
        before = mid / processor.fps
        after = max(1, processor.total_frames - mid - 1) / processor.fps
        stroke, anchor, label = "practice", "clip_midpoint", "Full clip"
        hands.clear()
    output = Path(processor.video_dir) / "motion"
    for index, frame in enumerate(frames):
        check_cancelled(stop_event)
        progress(index / len(frames), f"Building 3D replay {index + 1}/{len(frames)} · frame {frame}")
        try:
            bundle = reconstruct(
                processor.video_path,
                output,
                frame,
                stroke,
                before,
                after,
                footwork=footwork,
                anchor_kind=anchor,
                hitting_hand=hands.get(frame, "unknown"),
                stop_event=stop_event,
                portable_html=False,
            )
        except ProcessingCancelled:
            raise
        except Exception as exc:
            # A missed player or failed optional model must not lose the clips
            # or prevent another stroke in the same session from being reviewed.
            yield ReplayResult(processor.video_path, frame, label, error=str(exc))
        else:
            yield ReplayResult(processor.video_path, frame, label, bundle=bundle)


def write_replay_index(folder, entries):
    """A run-specific index avoids stale links on retry or after cancellation."""
    folder = Path(folder).resolve()
    rows = [
        dict(
            source=e.source,
            frame=e.frame,
            label=e.label,
            error=e.error,
            bundle=os.path.relpath(e.bundle, folder) if e.bundle else None,
        )
        for e in entries
    ]
    content = json.dumps(dict(version=1, strokes=rows), ensure_ascii=False, indent=2)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=folder, prefix="movement-session-", suffix=".tmp", mode="w", encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        index = temporary.with_suffix(".json")
        temporary.replace(index)
        return index
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_replay_index(path):
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "impact_frame" in doc and (path.parent / "pose.npz").is_file():
        return [ReplayResult(doc["source"], doc["impact_frame"], doc["stroke"], path.parent)]
    if doc.get("version") != 1 or not isinstance(doc.get("strokes"), list):
        raise ValueError("Choose a movement session JSON file or a stroke’s metadata.json.")
    return [
        ReplayResult(
            row["source"],
            row["frame"],
            row["label"],
            path.parent / row["bundle"] if row["bundle"] else None,
            row.get("error", ""),
        )
        for row in doc["strokes"]
    ]
