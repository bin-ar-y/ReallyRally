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
- `output_folder/debug_metrics.csv`: Raw telemetry data for each frame.

If you enable the **Telemetry Plot** option (`--plot` in CLI or checkbox in GUI):
- `output_folder/movement_telemetry.png`: A graph showing Arm Energy, Whip Ratio, and Player Scale timelines for visual analysis of the extraction decisions.

If you use `--split`, it will ALSO keep:
- `output_folder/rallies/`: Folder containing individual `.mp4` clips.

## Configuration
You can adjust the sensitivity and constraints by editing the **CONFIGURATION** section at `src/really_rally/core/config.py`.

**Current Tuned Values (Self-Adaptive Thresholding):**
```python
ADAPTIVE_WINDOW_SEC = 30.0      # Sliding window for resting baseline
ARM_ENERGY_K_MULTIPLIER = 4.0   # Sensitivity multiplier for MAD
WHIP_RATIO_K_MULTIPLIER = 4.0   # Sensitivity multiplier for MAD
RALLY_MAX_STROKE_GAP_SEC = 7.0  # Max time between strokes (t_max)
```

## How it Works
1.  **Pre-Motion Detection**: Checks pixel differences first. Skips "Empty Scenes" instantly.
2.  **Pose Extraction**: Uses Google MediaPipe to track the player's skeleton.
3.  **Feature Analysis**: Extracts kinematic features (Arm Energy, Whip Ratio) normalized by Torso Length to account for perspective shifts.
4.  **Self-Adaptive Thresholding**: Maintains a 30-second rolling buffer of resting states using statistical Median and Median Absolute Deviation (MAD) to adjust action thresholds dynamically based on the current camera layout.
5.  **State Machine Logic**: Detects rallies using a 3-State Temporal Logic gate (Idle -> Potential Rally -> Active Rally). Rallies are kept alive if consecutive strokes are detected within a 7.0-second timeout.
6.  **Export**: Saves segments and clips using FFmpeg.

## Troubleshooting
- **Rallies missed?** 
    - Decrease `ARM_ENERGY_K_MULTIPLIER` (e.g., to 3.0) or increase `RALLY_MAX_STROKE_GAP_SEC`.
- **Ball picking included?** 
    - Increase `ARM_ENERGY_K_MULTIPLIER` (e.g., to 5.0) to make the threshold more strict against noise.
- **MediaPipe Import Error?**
    - Ensure you have installed `mediapipe<0.11.0`. Newer versions (0.11+) have changed the API structure.
    - The script includes a check for this and will suggest fixes if it fails.

