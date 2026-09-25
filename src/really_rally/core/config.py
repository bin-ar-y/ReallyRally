MIN_RALLY_DURATION_SEC = 2.0    # Minimum rally duration to keep
GAP_MERGE_THRESHOLD_SEC = 5.0   # Merge motion gaps < 5s
BOUNDARY_PADDING_SEC = 0.5      # Padding at start/end of clips

# --- Two-Pass Optimization Settings ---
MIN_MOTION_SEGMENT_SEC = 2.0    # Discard motion segments shorter than this (Pass 1 Filter)
MOTION_SCAN_SKIP_FRAMES = 5     # Skip frames during Pass 1 for speed (e.g. 5 = ~12fps check)

# --- Pose Extraction Settings ---
MP_MODEL_COMPLEXITY = 1         # 0=Lite, 1=Full, 2=Heavy. 1 is good balance for M1.
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF = 0.5

# --- Feature Extraction Settings ---
WINDOW_SIZE_FRAMES = 15         # Smoothing window size (e.g., 0.25s @ 60fps)
PROCESS_EVERY_N_FRAMES = 2      # Sample every Nth frame (~30fps effective for smoother tracking)

# --- Heuristic Thresholds (legacy, kept for reference) ---
MOTION_ENERGY_THRESHOLD = 0.001
LATERAL_VELOCITY_THRESHOLD = 0.0005
VERTICAL_VELOCITY_THRESHOLD = 0.001
STATIONARY_ENERGY_THRESHOLD = 0.001

# --- Static Action Classifier Thresholds (for Serve/Feed mode) ---
ARM_ENERGY_THRESHOLD = 1.2
WHIP_RATIO_THRESHOLD = 1.2

# --- Self-Adaptive Thresholding Settings ---
ADAPTIVE_WINDOW_SEC = 45.0      # Sliding window for resting baseline (longer = stabler)
ARM_ENERGY_K_MULTIPLIER = 2.5   # Sensitivity multiplier (P25 + k*MAD). 2.5 ≈ ~2σ in Z-space
WHIP_RATIO_K_MULTIPLIER = 2.5   # Sensitivity multiplier for whip ratio
# Rally state machine: harder to START than to CONTINUE. Two candidate strokes
# must arrive within START_GAP to confirm a rally, but once active a gap up to
# CONTINUE_GAP keeps it alive (a single missed stroke never truncates it).
RALLY_START_GAP_SEC = 7.0
RALLY_CONTINUE_GAP_SEC = 11.0
MIN_ENERGY_FLOOR = 0.25         # Absolute minimum threshold for arm energy (Z-score space)
MIN_WHIP_FLOOR = 0.25           # Absolute minimum threshold for whip ratio (Z-score space)

# --- Player Distance Gate ---
# Bbox-diagonal-normalized nose-to-ankle height. Rejects poses that are
# too small (far away / irrelevant) or implausibly large (pose errors).
PLAYER_HEIGHT_MIN = 0.05        # Minimum fraction of frame height
PLAYER_HEIGHT_MAX = 1.50        # Maximum fraction of frame height

# --- Stroke Detection Gates ---
HIP_VELOCITY_STROKE_THRESHOLD = 0.005  # Minimum hip velocity for a stroke (bbox-normalized)
SPATIAL_VARIANCE_STROKE_MIN = 0.01     # Minimum spatial variance for stroke (was MIN_COURT_COVERAGE)

# --- Coarse Candidate Detection (high-recall additive score) ---
# Replaces the strict AND-gate: each signal that clears its own threshold adds
# its weight to an evidence score. A single strong signal (arm OR whip) can
# fire a candidate alone; weaker signals accumulate. This is the recall-first
# stage — object/temporal verification happens later.
COARSE_W_ARM = 2.0
COARSE_W_WHIP = 2.0
COARSE_W_HIP = 1.0
COARSE_W_SPATIAL = 0.5
COARSE_CANDIDATE_THRESHOLD = 2.0

# --- Pre-Motion Detection ---
PIXEL_DIFF_THRESHOLD = 20       # Intensity change (0-255) to consider a pixel "changed"
PIXEL_CHANGE_THRESHOLD = 0.0005 # Fraction of pixels changed to consider "Motion"

# --- Stationary Feeder Rejection ---
MIN_COURT_COVERAGE = 0.03       # Min spatial variance (std dev) to be considered "Moving"
MOVEMENT_WINDOW_SEC = 3.0       # Time window to track movement history

# --- Serve Detection (pose-based finite-state machine) ---
# All positions are MediaPipe-normalized (y increases downward, so a wrist with
# y < shoulder_y is raised above the shoulder; y < nose_y is above the head).
# Velocities are bbox-diagonal-normalized per sample step (see FeatureEngine).
SERVE_TOSS_ABOVE_SHOULDER_MARGIN = 0.02   # wrist_y must be this much below shoulder_y to count as raised
SERVE_TOSS_MIN_SEC = 0.12                  # min duration the toss arm must stay raised
SERVE_TOSS_TO_CONTACT_MIN_SEC = 0.10       # min gap between toss start and contact
SERVE_TOSS_TO_CONTACT_MAX_SEC = 2.0        # max gap between toss start and contact
SERVE_CONTACT_WRIST_VEL = 0.15             # min hit-wrist velocity (bbox-norm) for contact
SERVE_CONTACT_ELBOW_ANGLE = 110.0          # soft min elbow extension (deg) at contact
SERVE_FOLLOW_THROUGH_SEC = 1.0             # follow-through window after contact

# Weighted scoring (renormalized for pose-only: location & ball-flight dropped
# because they need court/ball streams not yet available).
SERVE_WEIGHT_TOSS = 0.4
SERVE_WEIGHT_CONTACT = 0.4
SERVE_WEIGHT_FOLLOW_THROUGH = 0.2

# Decision gates
SERVE_TOSS_MIN_SCORE = 0.5
SERVE_CONTACT_MIN_SCORE = 0.5
SERVE_MIN_SCORE = 0.55

# --- Feed Mode ---
FEED_HIP_VELOCITY_MIN = 0.005   # Min avg hip velocity for feed segments
