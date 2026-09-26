import json
import subprocess
import os
from datetime import datetime

def get_video_date(video_path: str) -> str:
    """
    Returns date string in MMDDYY format from metadata (creation_time) or file mtime.
    """
    date_str = None
    # 1. Try ffprobe for creation_time
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "format_tags=creation_time",
            video_path
        ]
        # Run with timeout to avoid hanging
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            tags = data.get('format', {}).get('tags', {})
            c_time = tags.get('creation_time') # e.g. "2024-05-10T15:30:00.000000Z"
            if c_time:
                # Parse ISO string. 
                # Handle potential "Z" or offset.
                c_time = c_time.replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(c_time)
                    date_str = dt.strftime("%m%d%y")
                except ValueError:
                    # Fallback if format is unexpected
                    pass
    except Exception as e:
        print(f"Warning: Could not extract creation_time via ffprobe: {e}")

    # 2. Fallback to file modification time
    if not date_str:
        try:
            timestamp = os.path.getmtime(video_path)
            dt = datetime.fromtimestamp(timestamp)
            date_str = dt.strftime("%m%d%y")
        except Exception as e:
            print(f"Warning: Could not get file mtime: {e}")
            date_str = datetime.now().strftime("%m%d%y") # Ultimate fallback

    return date_str

def probe_video(video_path: str) -> dict:
    """Read basic video metadata with OpenCV.

    Single place for the fps/frame-count probe, so the studio and the CLI agree
    on the numbers the frame<->time conversion depends on.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    # A container can report fps=0 or nan on malformed headers; a zero would
    # make every time conversion a division by zero downstream.
    if not fps or fps != fps:
        fps = 30.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps else 0.0,
    }


def frame_to_time(frame: int, fps: float) -> float:
    """Frame index -> seconds. Frame index is the source of truth."""
    return frame / fps if fps else 0.0


def time_to_frame(time_sec: float, fps: float, total_frames: int = None) -> int:
    """Seconds -> nearest frame index, clamped to the video when known."""
    frame = int(round(time_sec * fps)) if fps else 0
    if frame < 0:
        return 0
    if total_frames is not None and frame > total_frames - 1:
        return max(0, total_frames - 1)
    return frame


def generate_rally_name(video_path: str) -> str:
    """
    Generates name in format: {Date(MMDDYY)}_{OriginalName}_rally
    """
    date_str = get_video_date(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{date_str}_{video_stem}_rally"

