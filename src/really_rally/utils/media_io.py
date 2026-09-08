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

def generate_rally_name(video_path: str) -> str:
    """
    Generates name in format: {Date(MMDDYY)}_{OriginalName}_rally
    """
    date_str = get_video_date(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{date_str}_{video_stem}_rally"

