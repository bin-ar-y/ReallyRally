#!/usr/bin/env python3
"""Ground-truth labelling helper for rally/serve/feed detection.

The benchmark runner (`run_benchmark.py`) already knows how to score
predictions against annotations; what it lacked was a fast way to *produce*
those annotations. This tool covers that gap:

    # See what is in a video without watching it all
    python benchmarks/label.py --video clip.mov --contact-sheet /tmp/sheet.png

    # Create a benchmark case from a video
    python benchmarks/label.py --init-case my_rally --video clip.mov --mode rally

    # Label interactively (prints timings as you go)
    python benchmarks/label.py --video clip.mov --case my_rally

    # Or script it
    python benchmarks/label.py --video clip.mov --case my_rally --add 12.5-25.0 --add 40-55.3

    # Which cases still need labels?
    python benchmarks/label.py --status

Annotation format (matches benchmarks/README.md and utils/evaluate.py):

    [{"start": 12.5, "end": 25.0, "label": "rally", "notes": ""}, ...]
"""

import argparse
import json
import os
import re
import shutil
import sys
from typing import List, Tuple

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
ANNOTATIONS_NAME = "annotations.json"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*[-:,\s]\s*(\d+(?:\.\d+)?)\s*\Z")


def parse_range(text: str) -> Tuple[float, float]:
    """Parse 'START-END' (seconds) into a (start, end) tuple.

    Accepts '12.5-25', '12.5:25', '12.5 25'. Deliberately strict: malformed or
    ambiguous input is rejected rather than reinterpreted, because these values
    become ground truth and a silently misparsed label would corrupt every
    measurement taken against it.
    """
    match = _RANGE_RE.fullmatch(text)
    if not match:
        raise ValueError(f"expected START-END in seconds, got {text!r}")
    start, end = float(match.group(1)), float(match.group(2))
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    return start, end


def merge_overlapping(segments: List[dict]) -> List[dict]:
    """Sort by start and merge overlapping/adjacent intervals."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s["start"])
    merged = [dict(ordered[0])]
    for seg in ordered[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"]:
            last["end"] = max(last["end"], seg["end"])
            if seg.get("notes") and seg["notes"] not in last.get("notes", ""):
                last["notes"] = (last.get("notes", "") + " " + seg["notes"]).strip()
        else:
            merged.append(dict(seg))
    return merged


def load_annotations(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [
        {"start": float(e["start"]), "end": float(e["end"]),
         "label": e.get("label", "rally"), "notes": e.get("notes", "")}
        for e in data
    ]


def save_annotations(path: str, segments: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(segments, f, indent=2)


def video_metadata(path: str) -> dict:
    """Read duration/fps/dimensions. Kept tiny so labelling needs no deps."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "path": path,
        "fps": fps,
        "frames": frames,
        "duration": (frames / fps) if fps else 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def contact_sheet(video: str, output: str, count: int = 8, width: int = 260) -> str:
    """Write a horizontal strip of evenly spaced frames with timestamps.

    Lets a human identify what a clip contains (rally vs serve vs not tennis)
    in one glance instead of watching it end to end.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tiles = []
    for i in range(count):
        idx = int(total * (i + 0.5) / count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = width / float(frame.shape[1])
        frame = cv2.resize(frame, (width, max(1, int(frame.shape[0] * scale))))
        cv2.putText(frame, f"{idx / fps:.1f}s", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(frame)
    cap.release()

    if not tiles:
        raise ValueError(f"could not read frames from {video}")

    h = max(t.shape[0] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    cv2.imwrite(output, np.hstack(padded))
    return output


# ---------------------------------------------------------------------------
# Case management
# ---------------------------------------------------------------------------

def case_dir(name: str) -> str:
    return os.path.join(CASES_DIR, name)


def annotations_path(case: str) -> str:
    return os.path.join(case_dir(case), ANNOTATIONS_NAME)


def init_case(name: str, video: str, mode: str = "rally", move: bool = False) -> str:
    """Create benchmarks/cases/<name>/ with the video, mode.txt and a stub."""
    target = case_dir(name)
    os.makedirs(target, exist_ok=True)

    ext = os.path.splitext(video)[1].lower() or ".mp4"
    dest = os.path.join(target, f"video{ext}")
    if os.path.abspath(video) != os.path.abspath(dest):
        if move:
            shutil.move(video, dest)
        else:
            shutil.copy2(video, dest)

    with open(os.path.join(target, "mode.txt"), "w") as f:
        f.write(mode + "\n")

    annot = os.path.join(target, ANNOTATIONS_NAME)
    if not os.path.isfile(annot):
        save_annotations(annot, [])
    return target


def report_status() -> int:
    """Print which cases exist and which still need labels."""
    if not os.path.isdir(CASES_DIR):
        print(f"No cases directory at {CASES_DIR}")
        print("Create one with: python benchmarks/label.py --init-case NAME --video CLIP.mov")
        return 0

    cases = sorted(d for d in os.listdir(CASES_DIR) if os.path.isdir(case_dir(d)))
    if not cases:
        print(f"No cases in {CASES_DIR}")
        return 0

    print(f"\n{'case':28s} {'mode':7s} {'video':8s} {'labels':>7s}  status")
    print("-" * 70)
    unlabelled = 0
    for name in cases:
        d = case_dir(name)
        mode_file = os.path.join(d, "mode.txt")
        mode = open(mode_file).read().strip() if os.path.isfile(mode_file) else "rally"
        has_video = any(
            f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
            for f in os.listdir(d)
        )
        segs = load_annotations(os.path.join(d, ANNOTATIONS_NAME))
        if not segs:
            unlabelled += 1
        status = "OK" if segs else ("NEEDS LABELS" if has_video else "NEEDS VIDEO")
        print(f"{name:28s} {mode:7s} {str(has_video):8s} {len(segs):7d}  {status}")

    print(f"\n{len(cases)} case(s), {unlabelled} still needing labels.")
    return unlabelled


# ---------------------------------------------------------------------------
# Interactive labelling
# ---------------------------------------------------------------------------

def interactive_label(video: str, annot_path: str, mode: str) -> List[dict]:
    meta = video_metadata(video)
    segments = load_annotations(annot_path)

    print(f"\nVideo : {os.path.basename(video)}")
    print(f"        {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps "
          f"| {meta['duration']:.1f}s | mode={mode}")
    print(f"Labels: {annot_path} ({len(segments)} existing)")
    print("\nCommands:")
    print("  <start>-<end>   add a segment in seconds, e.g. 12.5-25")
    print("  list            show current segments")
    print("  undo            remove the last segment")
    print("  sheet [path]    write a contact sheet to check content")
    print("  save            write annotations.json")
    print("  quit            save and exit\n")

    while True:
        try:
            raw = input("label> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        if raw in ("quit", "q", "exit"):
            break
        if raw in ("list", "ls"):
            if not segments:
                print("  (no segments yet)")
            for i, s in enumerate(segments):
                print(f"  [{i}] {s['start']:.2f}s - {s['end']:.2f}s "
                      f"({s['end'] - s['start']:.2f}s) {s.get('notes', '')}")
            continue
        if raw in ("undo", "u"):
            if segments:
                print(f"  removed {segments.pop()}")
            continue
        if raw.startswith("sheet"):
            parts = raw.split(maxsplit=1)
            out = parts[1] if len(parts) > 1 else "/tmp/really_rally_sheet.png"
            try:
                print(f"  wrote {contact_sheet(video, out)}")
            except Exception as exc:
                print(f"  failed: {exc}")
            continue
        if raw in ("save", "s"):
            save_annotations(annot_path, merge_overlapping(segments))
            segments = load_annotations(annot_path)
            print(f"  saved {len(segments)} segment(s) to {annot_path}")
            continue

        try:
            start, end = parse_range(raw)
        except ValueError as exc:
            print(f"  {exc}")
            continue

        if meta["duration"] and start > meta["duration"]:
            print(f"  warning: start {start:.1f}s is past the end of the "
                  f"{meta['duration']:.1f}s video")
        segments.append({"start": start, "end": end, "label": mode, "notes": ""})
        print(f"  added {start:.2f}-{end:.2f}s  (total {len(segments)})")

    segments = merge_overlapping(segments)
    save_annotations(annot_path, segments)
    print(f"Saved {len(segments)} segment(s) to {annot_path}")
    return segments


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="Video to label or inspect")
    ap.add_argument("--case", help="Benchmark case name (defaults to the video stem)")
    ap.add_argument("--mode", default="rally", choices=["rally", "serve", "feed"])
    ap.add_argument("--init-case", metavar="NAME", help="Create benchmarks/cases/NAME")
    ap.add_argument("--move", action="store_true", help="Move (not copy) the video into the case")
    ap.add_argument("--add", action="append", default=[], metavar="START-END",
                    help="Add a segment non-interactively (repeatable)")
    ap.add_argument("--contact-sheet", metavar="PATH", help="Write a frame strip and exit")
    ap.add_argument("--frames", type=int, default=8, help="Frames in the contact sheet")
    ap.add_argument("--status", action="store_true", help="List cases and their label state")
    args = ap.parse_args()

    if args.status:
        report_status()
        return 0

    if args.init_case:
        if not args.video:
            ap.error("--init-case requires --video")
        target = init_case(args.init_case, args.video, args.mode, move=args.move)
        print(f"Created case at {target}")
        if not args.contact_sheet and not args.add:
            return 0
        args.case = args.init_case

    if not args.video:
        ap.error("--video is required (or use --status)")

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}")
        return 1

    if args.contact_sheet:
        out = contact_sheet(args.video, args.contact_sheet, count=args.frames)
        print(f"Wrote contact sheet to {out}")
        if not args.add and not args.case:
            return 0

    case = args.case or os.path.splitext(os.path.basename(args.video))[0]
    annot = annotations_path(case)

    if args.add:
        segments = load_annotations(annot)
        for spec in args.add:
            try:
                start, end = parse_range(spec)
            except ValueError as exc:
                print(f"Error: {exc}")
                return 1
            segments.append({"start": start, "end": end, "label": args.mode, "notes": ""})
        segments = merge_overlapping(segments)
        os.makedirs(os.path.dirname(annot), exist_ok=True)
        save_annotations(annot, segments)
        print(f"Saved {len(segments)} segment(s) to {annot}")
        return 0

    if sys.stdin.isatty():
        interactive_label(args.video, annot, args.mode)
    else:
        print("Not a terminal; nothing to do. Use --add START-END or --status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
