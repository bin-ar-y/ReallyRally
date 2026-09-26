"""The full per-frame feature store — the artifact that makes replay possible.

``*_debug_metrics.csv`` is an 18-column *interpretation* dump: it keeps only the
signals the current detector happens to read, only inside motion segments, and
only every ``PROCESS_EVERY_N_FRAMES``. That is enough to look at, but not enough
to re-run detection: the serve detector's per-arm features and the feed
detector's ``energy`` are simply not there.

``*_features.csv`` persists every :class:`FeatureEngine` key plus the columns the
processor injects, in a fixed order. Detection can then be replayed from disk
without MediaPipe, and new telemetry columns (racket, ball, fused scores) slot in
by appending to :data:`FEATURE_KEY_ORDER` — no schema migration.

``Frame`` is written as an integer and is the source of truth for time: within a
motion segment samples are exactly ``process_every_n`` frames apart, so a larger
jump marks a segment boundary. That is what lets the replay engine rebuild the
per-segment state (including the inter-segment rest-gap padding) exactly.
"""

from __future__ import annotations

import csv
from typing import Dict, List

from really_rally.core.config import PROCESS_EVERY_N_FRAMES

# Canonical serialization order. Lower-case keys throughout, mirroring
# FeatureEngine.compute_features (core/features.py) plus the columns
# analyze_segment_data injects (smoothed_*, t_*, z_*, coarse_score, candidate,
# fired_signals, state). Grouped for readability; order is what matters.
FEATURE_KEY_ORDER: List[str] = [
    # identity / coordinates
    "frame", "time", "segment",
    # raw motion features
    "energy", "lat_v", "vert_v", "action_score", "hip_vel", "whip_ratio",
    "arm_energy", "player_height", "is_valid_pose", "spatial_variance",
    # per-arm pose features (serve detection's inputs)
    "nose_y", "l_shoulder_y", "r_shoulder_y",
    "l_wrist_x", "l_wrist_y", "r_wrist_x", "r_wrist_y",
    "l_elbow_x", "l_elbow_y", "r_elbow_x", "r_elbow_y",
    "l_wrist_vel", "r_wrist_vel", "l_elbow_vel", "r_elbow_vel",
    "l_wrist_vy", "r_wrist_vy", "l_elbow_angle", "r_elbow_angle",
    "hip_center_x", "hip_center_y",
    # detection-stage columns (recomputed on replay, persisted for inspection)
    "smoothed_arm", "smoothed_whip", "t_arm", "t_whip", "z_arm", "z_whip",
    "coarse_score", "candidate", "fired_signals", "state",
]

# Header spellings that don't follow the mechanical title-casing below.
_HEADER_OVERRIDES = {
    "time": "Time(s)",
    "is_valid_pose": "Valid_Pose",
    "spatial_variance": "Spatial_Variance",
}

# Non-mechanical header -> key mappings seen in the wild (debug_metrics.csv
# abbreviates two of these differently).
_HEADER_TO_KEY = {
    "time(s)": "time",
    "valid_pose": "is_valid_pose",
    "spatial_var": "spatial_variance",
    "spatial_variance": "spatial_variance",
}

# Keys whose values are discrete/boolean rather than continuous measurements.
_INT_KEYS = ("is_valid_pose", "candidate")
_STR_KEYS = ("state", "fired_signals")

# ---------------------------------------------------------------------------
# Display roles
# ---------------------------------------------------------------------------
# A number on its own does not tell you whether it is a measurement, a bar it
# was compared against, or the outcome of that comparison. The inspector groups
# values by role so the interpretation is never ambiguous. New features (racket,
# ball, fused scores) are displayed by adding a key and a role here — nothing in
# the UI hard-codes a metric name.

ROLE_CONTEXT = "context"
ROLE_RAW = "raw"
ROLE_THRESHOLD = "threshold"
ROLE_SCORE = "score"
ROLE_DECISION = "decision"
ROLE_STATE = "state"
ROLE_POSE = "pose"

FEATURE_ROLES: Dict[str, str] = {}
for _key in ("frame", "time", "segment"):
    FEATURE_ROLES[_key] = ROLE_CONTEXT
for _key in ("energy", "lat_v", "vert_v", "action_score", "hip_vel", "whip_ratio",
             "arm_energy", "player_height", "spatial_variance",
             "smoothed_arm", "smoothed_whip"):
    FEATURE_ROLES[_key] = ROLE_RAW
for _key in ("t_arm", "t_whip"):
    FEATURE_ROLES[_key] = ROLE_THRESHOLD
for _key in ("z_arm", "z_whip", "coarse_score"):
    FEATURE_ROLES[_key] = ROLE_SCORE
for _key in ("is_valid_pose", "candidate", "fired_signals"):
    FEATURE_ROLES[_key] = ROLE_DECISION
FEATURE_ROLES["state"] = ROLE_STATE

# Anything not listed above is pose geometry (the per-arm serve inputs and the
# remaining landmark coordinates), which is the sensible default for a new
# positional feature.
for _key in FEATURE_KEY_ORDER:
    FEATURE_ROLES.setdefault(_key, ROLE_POSE)


def header_for(key: str) -> str:
    """Column name written for a lower-case feature key."""
    if key in _HEADER_OVERRIDES:
        return _HEADER_OVERRIDES[key]
    return "_".join(part.capitalize() for part in key.split("_"))


FEATURE_HEADERS: List[str] = [header_for(k) for k in FEATURE_KEY_ORDER]


def normalize_header(header: str) -> str:
    """Map any known column spelling to its canonical lower-case key.

    Idempotent, so it accepts both in-memory feature dicts (already lower-case)
    and CSV rows read with :class:`csv.DictReader` (Title_Case).
    """
    s = header.strip().lower().replace("(s)", "")
    return _HEADER_TO_KEY.get(s, s)


def normalize_row(row) -> Dict[str, str]:
    """Re-key one CSV-ish row (or in-memory dict) to canonical feature keys."""
    return {normalize_header(k): v for k, v in row.items()}


# ---------------------------------------------------------------------------
# Segment reconstruction
# ---------------------------------------------------------------------------

def split_into_segments(records: List[Dict], process_every_n: int = PROCESS_EVERY_N_FRAMES) -> List[List[Dict]]:
    """Group records into motion segments by frame-index gaps.

    Inside a segment pose runs every ``process_every_n`` frames, so a step
    larger than that is a genuine discontinuity — the gap between two motion
    segments. Recovering this from ``frame`` is what lets a replay reproduce the
    live path's per-segment state without a separate boundary column.
    """
    segments: List[List[Dict]] = []
    current: List[Dict] = []
    prev_frame = None
    for record in records:
        frame = int(record.get("frame", 0))
        if prev_frame is not None and frame - prev_frame > process_every_n:
            if current:
                segments.append(current)
            current = []
        current.append(record)
        prev_frame = frame
    if current:
        segments.append(current)
    return segments


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize(record: Dict, key: str, fps: float, segment_idx: int):
    if key == "frame":
        return int(record.get("frame", 0))
    if key == "time":
        # Frame-derived, never re-estimated from a single nominal fps.
        return f"{int(record.get('frame', 0)) / fps:.2f}"
    if key == "segment":
        return segment_idx
    if key in _STR_KEYS:
        return record.get(key, "")
    if key in _INT_KEYS:
        return int(record.get(key, 0))
    return f"{float(record.get(key, 0.0)):.6f}"


def write_feature_store(records: List[Dict], path: str, fps: float,
                        process_every_n: int = PROCESS_EVERY_N_FRAMES) -> None:
    """Write the full feature store. Replaces any previous contents."""
    segments = split_into_segments(records, process_every_n)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_HEADERS)
        for segment_idx, segment in enumerate(segments):
            for record in segment:
                writer.writerow([_serialize(record, key, fps, segment_idx) for key in FEATURE_KEY_ORDER])


def read_feature_store(path: str) -> List[Dict]:
    """Load a feature store into per-frame dicts with canonical keys.

    ``timestamp`` is populated from the stored ``time`` column because the
    detection code keys off it; it stays frame-derived rather than recomputed.
    """
    records: List[Dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            norm = normalize_row(row)
            record: Dict = {}
            for key in FEATURE_KEY_ORDER:
                raw = norm.get(key, "")
                missing = raw is None or raw == ""
                if key == "frame":
                    record["frame"] = 0 if missing else int(float(raw))
                elif key == "time":
                    value = 0.0 if missing else float(raw)
                    record["time"] = value
                    record["timestamp"] = value
                elif key == "segment":
                    record["segment"] = 0 if missing else int(float(raw))
                elif key in _STR_KEYS:
                    record[key] = "" if missing else raw
                elif key in _INT_KEYS:
                    record[key] = 0 if missing else int(float(raw))
                else:
                    record[key] = 0.0 if missing else float(raw)
            records.append(record)
    return records
