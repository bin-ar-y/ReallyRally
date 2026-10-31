"""Snapshot of the detector's tunable configuration, split by replayability.

``config.py`` holds every constant flat. The Debug Studio needs a finer
distinction: which parameters can be re-tuned from cached features (replayable)
and which change feature extraction itself and therefore require re-running the
video. This module makes that split explicit and gives the replay engine one
object to carry, so the live path and the replay path read the *same* values.

A snapshot is JSON-serializable so ``*_experiments.csv`` can record exactly
which parameters produced a result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List

import really_rally.core.config as cfg

# Context fields carried alongside the tunables. They are read-only in the
# sense that ``apply`` refuses them: ``process_every_n`` changes sampling
# (re-analysis), and fps/max_time describe the video, not the detector.
CONTEXT_FIELDS = ("process_every_n", "fps", "max_time")

# Parameters that change feature extraction (pose model, frame sampling, motion
# pre-filter, object inference) or whose inputs are not persisted. Tuning any of
# these invalidates cached features, so the studio shows them as
# "requires video re-analysis" rather than offering a live replay.
REANALYSIS_KEYS = frozenset([
    # Pass 1 motion pre-filter
    "GAP_MERGE_THRESHOLD_SEC", "MIN_MOTION_SEGMENT_SEC", "MOTION_SCAN_SKIP_FRAMES",
    "PIXEL_DIFF_THRESHOLD", "PIXEL_CHANGE_THRESHOLD",
    # Pose model
    "MP_MODEL_COMPLEXITY", "MP_MIN_DETECTION_CONF", "MP_MIN_TRACKING_CONF",
    # Feature extraction
    "PROCESS_EVERY_N_FRAMES", "POSE_DROPOUT_BRIDGE_STEPS", "MOVEMENT_WINDOW_SEC",
    # Object verification (detections are not cached)
    "RACKET_HAND_MAX_DIST_NORM", "BALL_TRACK_GATE_NORM", "BALL_TRACK_MAX_GAP_FRAMES",
    "BALL_TRACK_MIN_OBSERVATIONS", "OBJECT_INFERENCE_LONG_SIDE", "OBJECT_WINDOW_PAD_SEC",
    "FUSION_W_RACKET", "FUSION_W_BALL",
    # Serve detection (per-arm pose features are not cached)
    "SERVE_TOSS_ABOVE_SHOULDER_MARGIN", "SERVE_TOSS_MIN_SEC",
    "SERVE_TOSS_TO_CONTACT_MIN_SEC", "SERVE_TOSS_TO_CONTACT_MAX_SEC",
    "SERVE_CONTACT_WRIST_VEL", "SERVE_CONTACT_ELBOW_ANGLE",
    "SERVE_FOLLOW_THROUGH_SEC", "SERVE_WEIGHT_TOSS", "SERVE_WEIGHT_CONTACT",
    "SERVE_WEIGHT_FOLLOW_THROUGH", "SERVE_TOSS_MIN_SCORE",
    "SERVE_CONTACT_MIN_SCORE", "SERVE_MIN_SCORE",
    "SERVE_LOCK_HANDEDNESS", "SERVE_FORCED_HANDEDNESS",
    # Feed mode
    "FEED_HIP_VELOCITY_MIN",
])


@dataclass(frozen=True)
class ConfigSnapshot:
    """One complete, immutable detector configuration.

    Field names are the lower-case form of their ``config.py`` constants, so
    ``ARM_ENERGY_K_MULTIPLIER`` lives here as ``arm_energy_k_multiplier``.
    """

    adaptive_window_sec: float = cfg.ADAPTIVE_WINDOW_SEC
    arm_energy_k_multiplier: float = cfg.ARM_ENERGY_K_MULTIPLIER
    whip_ratio_k_multiplier: float = cfg.WHIP_RATIO_K_MULTIPLIER
    min_energy_floor: float = cfg.MIN_ENERGY_FLOOR
    min_whip_floor: float = cfg.MIN_WHIP_FLOOR
    player_height_min: float = cfg.PLAYER_HEIGHT_MIN
    player_height_max: float = cfg.PLAYER_HEIGHT_MAX
    hip_velocity_stroke_threshold: float = cfg.HIP_VELOCITY_STROKE_THRESHOLD
    spatial_variance_stroke_min: float = cfg.SPATIAL_VARIANCE_STROKE_MIN
    coarse_w_arm: float = cfg.COARSE_W_ARM
    coarse_w_whip: float = cfg.COARSE_W_WHIP
    coarse_w_hip: float = cfg.COARSE_W_HIP
    coarse_w_spatial: float = cfg.COARSE_W_SPATIAL
    coarse_candidate_threshold: float = cfg.COARSE_CANDIDATE_THRESHOLD
    coarse_min_abs_arm: float = cfg.COARSE_MIN_ABS_ARM
    coarse_min_abs_whip: float = cfg.COARSE_MIN_ABS_WHIP
    rally_start_gap_sec: float = cfg.RALLY_START_GAP_SEC
    rally_continue_gap_sec: float = cfg.RALLY_CONTINUE_GAP_SEC
    min_rally_duration_sec: float = cfg.MIN_RALLY_DURATION_SEC
    boundary_padding_sec: float = cfg.BOUNDARY_PADDING_SEC
    fusion_w_pose: float = cfg.FUSION_W_POSE
    fusion_w_temporal: float = cfg.FUSION_W_TEMPORAL
    fusion_neutral: float = cfg.FUSION_NEUTRAL
    fusion_min_confidence: float = cfg.FUSION_MIN_CONFIDENCE

    process_every_n: int = cfg.PROCESS_EVERY_N_FRAMES
    fps: float = 0.0
    max_time: float = 0.0

    # -- construction ----------------------------------------------------

    @classmethod
    def replayable_field_names(cls) -> List[str]:
        return [f.name for f in fields(cls) if f.name not in CONTEXT_FIELDS]

    @classmethod
    def from_module(cls, fps: float, max_time: float, process_every_n: int = None) -> "ConfigSnapshot":
        """Read the current values out of ``config.py``."""
        values = {name: getattr(cfg, name.upper()) for name in cls.replayable_field_names()}
        values["process_every_n"] = cfg.PROCESS_EVERY_N_FRAMES if process_every_n is None else process_every_n
        values["fps"] = fps
        values["max_time"] = max_time
        return cls(**values)

    # -- derivation ------------------------------------------------------

    @property
    def effective_fps(self) -> float:
        """Sampling rate actually seen by the detector (video fps / stride)."""
        return self.fps / self.process_every_n

    # -- editing ---------------------------------------------------------

    def apply(self, overrides: Dict[str, float]) -> "ConfigSnapshot":
        """Return a new snapshot with ``overrides`` applied.

        Keys are the ``config.py`` constant names. A key that requires
        re-analysis is rejected loudly rather than silently ignored: replaying a
        pose-model change from cached features would produce an answer that
        looks valid and is not.
        """
        changes: Dict[str, float] = {}
        for key, value in overrides.items():
            upper = key.upper()
            if upper not in REPLAYABLE_KEYS:
                reason = ("requires video re-analysis" if upper in REANALYSIS_KEYS
                          else "is not a known detector parameter")
                raise ValueError(f"Cannot apply {key!r}: it {reason}.")
            changes[upper.lower()] = value
        return replace(self, **changes)

    # -- serialization ---------------------------------------------------

    def classify(self) -> Dict[str, List[str]]:
        """The two parameter tiers, for the config editor to render."""
        return {
            "replayable": sorted(REPLAYABLE_KEYS),
            "requires_reanalysis": sorted(REANALYSIS_KEYS),
        }

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["replayable_keys"] = sorted(REPLAYABLE_KEYS)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


# Derived from the dataclass so the two can never drift.
REPLAYABLE_KEYS = frozenset(name.upper() for name in ConfigSnapshot.replayable_field_names())
