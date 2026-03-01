MIN_RALLY_DURATION_SEC = 2.0    # Debug: Lowered to 2s to catch shorter segments
GAP_MERGE_THRESHOLD_SEC = 5.0   # Merge gaps < 5s
BOUNDARY_PADDING_SEC = 0.5      # Add this much padding to start/end of clips

# --- Two-Pass Optimization Settings ---
MIN_MOTION_SEGMENT_SEC = 3.0    # Discard motion segments shorter than this (Pass 1 Filter)
MOTION_SCAN_SKIP_FRAMES = 5     # Skip frames during Pass 1 for speed (e.g. 5 = 12fps check)

# --- Pose Extraction Settings ---
MP_MODEL_COMPLEXITY = 1         # 0=Lite, 1=Full, 2=Heavy. 1 is good balance for M1.
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF = 0.5

# --- Feature Extraction Settings ---
WINDOW_SIZE_FRAMES = 15         # Smoothing window size (e.g., 0.25s @ 60fps)
# Lowering this to 3 to catch fast tennis swings (Swing-to-Run Ratio)
PROCESS_EVERY_N_FRAMES = 2      # Sample rate (approx 30fps effective for smoother tracking)
# --- Action Classifier Thresholds (Heuristic) ---
MOTION_ENERGY_THRESHOLD = 0.001 
LATERAL_VELOCITY_THRESHOLD = 0.0005
VERTICAL_VELOCITY_THRESHOLD = 0.001
STATIONARY_ENERGY_THRESHOLD = 0.001
ARM_ENERGY_THRESHOLD = 1.2      # Loose threshold for arm swinging
WHIP_RATIO_THRESHOLD = 1.2      # Minimum Wrist/Elbow speed ratio to verify a swing vs a run

# --- Pre-Motion Detection ---
PIXEL_DIFF_THRESHOLD = 20       # Intensity change (0-255) to consider a pixel "changed"
PIXEL_CHANGE_THRESHOLD = 0.0005 # Fraction of pixels changed to consider "Motion"

# --- Stationary Feeder Rejection ---
MIN_COURT_COVERAGE = 0.03       # Min spatial variance (std dev) to be considered "Moving"
MOVEMENT_WINDOW_SEC = 3.0       # Time window to track movement history
