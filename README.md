# RealRally
A local, offline tool to automatically extract tennis rallies from practice videos. Designed for macOS (Apple Silicon) but compatible with oth     er platforms.
=======
# Tennis Rally Extractor

A local, offline tool to automatically extract tennis rallies from practice videos. Designed for **macOS (Apple Silicon)** but compatible with other platforms.

## Prerequisites

- **Python 3.8+**
- **FFmpeg** (Must be installed and accessible in your terminal)

### 1. Install FFmpeg
On macOS with Homebrew:
```bash
brew install ffmpeg
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```
> [!IMPORTANT]
> This project requires **MediaPipe < 0.11.0** to use the legacy Solutions API.
> If you are not using the provided `requirements.txt`, ensure you install a compatible version:
> ```bash
> pip install "mediapipe<0.11.0"
> pip install customtkinter
> ```

### 3. (Recommended) Use a Virtual Environment
To avoid conflicts with other Python packages, run the script within a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (GUI)
The easiest way to use the tool is via the graphical interface.

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch GUI
python3 gui.py
```

### GUI Features
1.  **Modern Dark UI**: A sleek, dark-themed interface built with `CustomTkinter`. 
2.  **Select Videos**: Choose one or multiple video files to process in batch.
3.  **Output Location**: Select where the results should be saved.
4.  **Start Processing**: Runs the analysis.
    - **Progress Bar**: Shows real-time scanning and analysis progress for each video.
    - **Stop Button**: Safely cancels the operation at any time.
5.  **Robust Batch Processing**:
    - Automatically standardizes different video resolutions (e.g., mixing 4K and 1080p).
    - Ensures unique filenames for every clip to prevent overwrites.
    - merges all detected rallies into a single output file: `{Date}_Merged_Rallies.mp4`.

## Usage (Command Line)

Run the script on your video file:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Default: Creates output folder named after video (e.g., ./my_match/)
python3 extract_rallies.py /path/to/my_match.mp4


# Custom Output Location:
python3 extract_rallies.py /path/to/your/video.mp4 --output /path/to/output_folder

# Keep Individual Clips (Default is Merged Video Only):
python3 extract_rallies.py /path/to/your/video.mp4 --split
```

### Output
The tool will always create:
- `output_folder/{VideoName}_rallies.mp4`: A single video file combining all extracted rallies (e.g., `my_match_rallies.mp4`).
- `output_folder/rallies.csv`: CSV file identifying start/end times.

If you use `--split`, it will ALSO keep:
- `output_folder/rallies/`: Folder containing individual `.mp4` clips.

## Configuration
You can adjust the sensitivity and constraints by editing the **CONFIGURATION** section at the top of `extract_rallies.py`.

**Current Tuned Values (Swing-to-Run Ratio Logic):**
```python
MIN_RALLY_DURATION_SEC = 2.0     # Minimum duration to keep
GAP_MERGE_THRESHOLD_SEC = 5.0    # Merge gaps < 5s
SWING_RATIO_THRESHOLD = 15.0     # Minimum "Swing Quality" (Arm Vel / Hip Vel)
PROCESS_EVERY_N_FRAMES = 3       # Sample every 3 frames (~20fps) to catch fast swings
```

## How it Works
1.  **Pre-Motion Detection**: Checks pixel differences first. Skips "Empty Scenes" instantly.
2.  **Pose Extraction**: Uses Google MediaPipe to track the player's skeleton.
3.  **Feature Analysis**: Computes `Swing-to-Run Ratio`.
    - **Rally**: High Arm Velocity + Low Hip Velocity (Planted feet = High Ratio).
    - **Ball Picking**: High Arm Velocity + High Hip Velocity (Running = Low Ratio).
4.  **Filtering**: Rejects segments with `Max Swing Ratio < 15.0`.
5.  **Export**: Saves clips using FFmpeg.

## Troubleshooting
- **Rallies missed?** 
    - Lower `SWING_RATIO_THRESHOLD` (e.g., to 10.0 or 8.0).
- **Ball picking included?** 
    - Raise `SWING_RATIO_THRESHOLD` (e.g., to 20.0).
- **MediaPipe Import Error?**
    - Ensure you have installed `mediapipe<0.11.0`. Newer versions (0.11+) have changed the API structure.
    - The script includes a check for this and will suggest fixes if it fails.

