"""Pure detection: feature dicts in, events out. No video, no GUI.

This module holds the detection logic that used to live inline in
``TennisRallyProcessor.analyze_segment_data``. It was extracted for one reason:
the Debug Studio must be able to re-run detection against a cached feature store
without decoding the video again, and that is only trustworthy if the live path
and the replay path execute *the same code*. A re-implementation would drift.

Consequences of the extraction:

- Every tunable arrives via :class:`~really_rally.core.config_snapshot.ConfigSnapshot`
  instead of being read from ``config.py`` at call time. The detector classes bake
  module defaults into their signatures at import, so each one must be constructed
  with explicit arguments — a missed argument silently reverts to the default.
- Cross-segment state (the adaptive threshold buffers and the last frame seen)
  lives in :class:`AdaptiveState`, owned by the analyzer instance. One analyzer per
  video: sharing one across videos would bleed thresholds between recordings, which
  is the bug the old class-level buffers invited.
- ``CoarseCandidateDetector.score()``'s evidence list is kept on each frame as
  ``fired_signals``, so a decision can be explained rather than just reported.

Behavior is byte-for-byte the previous implementation, including two things that
look like inconsistencies but are deliberate ports: serve mode reads the
thresholds *before* updating the buffers while rally mode updates first, and serve
mode does not pad the buffers across rest gaps. Changing either would change
existing output, so both are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from really_rally.core.candidate import CoarseCandidateDetector, RallyStateMachine
from really_rally.core.config import (
    GAP_MERGE_THRESHOLD_SEC,
    MIN_COURT_COVERAGE,
    MIN_RALLY_DURATION_SEC,
    MOTION_ENERGY_THRESHOLD,
)
from really_rally.core.config_snapshot import ConfigSnapshot
from really_rally.core.features import (
    RobustAdaptiveThreshold,
    RollingZScoreNormalizer,
    SelfAdaptiveThreshold,
)
from really_rally.core.models import RallySegment, ServeEvent

# Outlier caps applied before smoothing. Genuine values sit far below these, so
# they only clip numerical blow-ups.
ARM_OUTLIER_CAP = 4.0
WHIP_OUTLIER_CAP = 25.0
# Only values above this count toward the data-driven cap, so a quiet clip does
# not compute a "typical" value out of noise.
OUTLIER_ACTIVE_FLOOR = 0.1
MIN_OUTLIER_SAMPLES = 5


@dataclass
class AnalysisResult:
    """What a detection pass produced."""
    segments: List[RallySegment] = field(default_factory=list)
    events: List[ServeEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.segments) + len(self.events)


def mitigate_outliers(arr: np.ndarray, absolute_max: float) -> np.ndarray:
    """Clip extreme spikes so smoothing is not dominated by a single sample."""
    if len(arr) == 0:
        return arr
    capped = np.clip(arr, 0, absolute_max)
    active = capped[capped > OUTLIER_ACTIVE_FLOOR]
    if len(active) > MIN_OUTLIER_SAMPLES:
        median = np.median(active)
        mad = np.median(np.abs(active - median))
        if mad > 0.01:
            dynamic_cap = max(median + 6 * mad, 1.0)
            capped = np.clip(capped, 0, dynamic_cap)
    return capped


def smooth(values: np.ndarray, effective_fps: float) -> np.ndarray:
    """Half-second moving average — the window the detector thresholds."""
    window = int(effective_fps * 0.5)
    if window < 1:
        window = 1
    return np.convolve(values, np.ones(window) / window, mode="same")


def mask_to_indices(mask) -> List[Tuple[int, int]]:
    """Contiguous True runs as ``(start, end_exclusive)`` index pairs.

    End-exclusive to match ``TennisRallyProcessor._mask_to_indices``, whose
    callers slice with ``[start:end]`` and index the last member at ``end - 1``.
    """
    padded = np.concatenate(([0], mask, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


@dataclass
class AdaptiveState:
    """Cross-segment detector state for one video.

    The adaptive thresholds persist across motion segments so the baseline
    reflects the whole recording, and the rest gap between segments is padded
    with zeros so a long break reads as inactivity rather than being skipped.
    """

    arm: RobustAdaptiveThreshold
    whip: RobustAdaptiveThreshold
    last_segment_end_frame: int = 0

    @classmethod
    def from_config(cls, config: ConfigSnapshot) -> "AdaptiveState":
        effective_fps = config.effective_fps
        return cls(
            arm=RobustAdaptiveThreshold(config.adaptive_window_sec, effective_fps,
                                        config.arm_energy_k_multiplier,
                                        config.min_energy_floor),
            whip=RobustAdaptiveThreshold(config.adaptive_window_sec, effective_fps,
                                         config.whip_ratio_k_multiplier,
                                         config.min_whip_floor),
        )

    def pad_rest_gap(self, first_frame: int, process_every_n: int) -> int:
        """Zero-pad the buffers for the gap before this segment."""
        gap_frames = max(0, first_frame - self.last_segment_end_frame)
        gap_samples = int(gap_frames / process_every_n)
        pad_count = min(gap_samples, self.arm.capacity)
        if pad_count > 0:
            self.arm.extend_padding(0.0, pad_count)
            self.whip.extend_padding(0.0, pad_count)
        return pad_count


class RallyAnalyzer:
    """Coarse candidate detection + hysteresis FSM over one video's segments."""

    def __init__(self, config: ConfigSnapshot):
        self.config = config
        self.state = AdaptiveState.from_config(config)

    def analyze(self, frame_data: List[Dict]) -> List[RallySegment]:
        """Enrich ``frame_data`` in place and return the rallies it contains."""
        if not frame_data:
            return []

        config = self.config
        effective_fps = config.effective_fps

        self.state.pad_rest_gap(frame_data[0].get("frame", 0), config.process_every_n)
        arm_adaptive = self.state.arm
        whip_adaptive = self.state.whip

        energies = np.array([d.get("energy", 0) for d in frame_data])
        arm_energies = mitigate_outliers(
            np.array([d.get("arm_energy", 0) for d in frame_data]), ARM_OUTLIER_CAP)
        hip_vels = np.array([d.get("hip_vel", 0) for d in frame_data])
        whips = mitigate_outliers(
            np.array([d.get("whip_ratio", 0) for d in frame_data]), WHIP_OUTLIER_CAP)

        smoothed_arm = smooth(arm_energies, effective_fps)
        smoothed_whip = smooth(whips, effective_fps)
        # Computed for parity with the original pipeline; not used by the rally
        # path, which thresholds smoothed arm/whip only.
        _smoothed_energy = smooth(energies, effective_fps)

        z_arm = RollingZScoreNormalizer(
            config.adaptive_window_sec, effective_fps).normalize_array(smoothed_arm)
        z_whip = RollingZScoreNormalizer(
            config.adaptive_window_sec, effective_fps).normalize_array(smoothed_whip)

        coarse = CoarseCandidateDetector(
            w_arm=config.coarse_w_arm, w_whip=config.coarse_w_whip,
            w_hip=config.coarse_w_hip, w_spatial=config.coarse_w_spatial,
            candidate_threshold=config.coarse_candidate_threshold,
            hip_min=config.hip_velocity_stroke_threshold,
            spatial_min=config.spatial_variance_stroke_min,
            min_abs_arm=config.coarse_min_abs_arm,
            min_abs_whip=config.coarse_min_abs_whip,
        )
        fsm = RallyStateMachine(
            fps=config.fps, max_time=config.max_time,
            start_gap_sec=config.rally_start_gap_sec,
            continue_gap_sec=config.rally_continue_gap_sec,
            min_duration_sec=config.min_rally_duration_sec,
            boundary_padding_sec=config.boundary_padding_sec,
        )

        final_segments: List[RallySegment] = []
        for i, record in enumerate(frame_data):
            record["smoothed_arm"] = smoothed_arm[i]
            record["smoothed_whip"] = smoothed_whip[i]

            arm_adaptive.update(z_arm[i])
            whip_adaptive.update(z_whip[i])
            t_arm = arm_adaptive.get_threshold()
            t_whip = whip_adaptive.get_threshold()

            record["t_arm"] = t_arm
            record["t_whip"] = t_whip
            record["z_arm"] = z_arm[i]
            record["z_whip"] = z_whip[i]

            scale = record.get("player_height", 0)
            valid_distance = config.player_height_min <= scale <= config.player_height_max

            score, fired = coarse.score(
                valid_pose=record.get("is_valid_pose", False),
                valid_distance=valid_distance,
                z_arm=z_arm[i], t_arm=t_arm,
                z_whip=z_whip[i], t_whip=t_whip,
                hip_vel=hip_vels[i],
                spatial_var=record.get("spatial_variance", 1.0),
                raw_arm=smoothed_arm[i],
                raw_whip=smoothed_whip[i],
            )
            is_candidate = coarse.is_candidate(score)

            record["coarse_score"] = score
            record["candidate"] = int(is_candidate)
            record["fired_signals"] = "|".join(fired)

            segment = fsm.step(is_candidate, record.get("frame", 0),
                               record.get("timestamp", 0.0), score)
            record["state"] = fsm.state_name
            if segment is not None:
                final_segments.append(segment)

        trailing = fsm.finalize()
        if trailing is not None:
            final_segments.append(trailing)

        self.state.last_segment_end_frame = frame_data[-1].get("frame", 0)
        return final_segments


class ServeAnalyzer:
    """Pose-based serve detection over one video's segments.

    Ports ``ServePracticeProcessor.analyze_segment_data``. Threshold ordering
    deliberately differs from the rally path: thresholds are read before the
    buffers are updated, and the rest gap is not padded. Preserving that is what
    keeps replay output identical to a live run.
    """

    def __init__(self, config: ConfigSnapshot, lock_handedness: bool = True,
                 forced_handedness: Optional[str] = None):
        self.config = config
        self.lock_handedness = lock_handedness
        self.forced_handedness = forced_handedness
        self.state = AdaptiveState.from_config(config)

    def analyze(self, frame_data: List[Dict]) -> Tuple[List[RallySegment], List[ServeEvent]]:
        from really_rally.core.serve_detector import ServeDetector

        if not frame_data:
            return [], []

        effective_fps = self.config.effective_fps
        smoothed_arm = smooth(np.array([d.get("arm_energy", 0.0) for d in frame_data]),
                              effective_fps)
        smoothed_whip = smooth(np.array([d.get("whip_ratio", 0.0) for d in frame_data]),
                               effective_fps)

        detector = ServeDetector(effective_fps, lock_handedness=self.lock_handedness,
                                 forced_handedness=self.forced_handedness)
        events = detector.detect(frame_data)

        serve_hit = np.zeros(len(frame_data), dtype=bool)
        for event in events:
            serve_hit[event.start_idx:event.end_idx + 1] = True

        for i, record in enumerate(frame_data):
            record["smoothed_arm"] = smoothed_arm[i]
            record["smoothed_whip"] = smoothed_whip[i]
            record["t_arm"] = self.state.arm.get_threshold()
            record["t_whip"] = self.state.whip.get_threshold()
            record["z_arm"] = 0.0
            record["z_whip"] = 0.0
            record["coarse_score"] = 0.0
            record["fired_signals"] = ""
            record["candidate"] = int(serve_hit[i])
            record["state"] = "serve" if serve_hit[i] else "idle"
            self.state.arm.update(smoothed_arm[i])
            self.state.whip.update(smoothed_whip[i])

        segments = [
            RallySegment(
                start_frame=int(event.start_time * self.config.fps),
                end_frame=int(event.end_time * self.config.fps),
                start_time=event.start_time,
                end_time=event.end_time,
                score=event.score,
            )
            for event in events
        ]
        return segments, events


class FeedAnalyzer:
    """Multiball feed detection.

    Feed mode calibrates its accepted player scale from the video itself
    (``FeedPracticeProcessor.estimate_player_scale``), which a cached feature
    store cannot reproduce. The calibrated bounds are therefore an explicit
    input: replay uses the defaults unless the caller supplies the ones the
    original run measured.
    """

    def __init__(self, config: ConfigSnapshot, min_player_height: float = None,
                 max_player_height: float = None, total_frames: int = 0):
        from really_rally.core.config import PLAYER_HEIGHT_MAX, PLAYER_HEIGHT_MIN

        self.config = config
        self.min_player_height = (PLAYER_HEIGHT_MIN if min_player_height is None
                                  else min_player_height)
        self.max_player_height = (PLAYER_HEIGHT_MAX if max_player_height is None
                                  else max_player_height)
        self.total_frames = total_frames

    def analyze(self, frame_data: List[Dict]) -> List[RallySegment]:
        if not frame_data:
            return []

        config = self.config
        n = len(frame_data)
        effective_fps = config.effective_fps

        smoothed_energy = smooth(np.array([d.get("energy", 0) for d in frame_data]),
                                 effective_fps)
        smoothed_arm = smooth(np.array([d.get("arm_energy", 0) for d in frame_data]),
                              effective_fps)
        smoothed_whip = smooth(np.array([d.get("whip_ratio", 0) for d in frame_data]),
                               effective_fps)

        # Feed mode uses the simpler self-adaptive threshold, rebuilt per
        # segment rather than persisted across the video.
        arm_adaptive = SelfAdaptiveThreshold(
            config.adaptive_window_sec, effective_fps,
            config.arm_energy_k_multiplier, config.min_energy_floor)
        whip_adaptive = SelfAdaptiveThreshold(
            config.adaptive_window_sec, effective_fps,
            config.whip_ratio_k_multiplier, config.min_whip_floor)

        t_arm_arr = np.zeros(n)
        t_whip_arr = np.zeros(n)
        for i, record in enumerate(frame_data):
            record["smoothed_arm"] = smoothed_arm[i]
            record["smoothed_whip"] = smoothed_whip[i]
            arm_adaptive.update(smoothed_arm[i])
            whip_adaptive.update(smoothed_whip[i])
            t_arm_arr[i] = arm_adaptive.get_threshold()
            t_whip_arr[i] = whip_adaptive.get_threshold()
            record["t_arm"] = t_arm_arr[i]
            record["t_whip"] = t_whip_arr[i]
            record.setdefault("candidate", 0)
            record.setdefault("state", "")

        valid_pose = np.array([d.get("is_valid_pose", False) for d in frame_data], dtype=bool)
        sub_gap_frames = int(GAP_MERGE_THRESHOLD_SEC * effective_fps)
        raw_segments = mask_to_indices(valid_pose.astype(int))
        if not raw_segments:
            return []

        merged: List[Tuple[int, int]] = []
        curr_start, curr_end = raw_segments[0]
        for next_start, next_end in raw_segments[1:]:
            if next_start - curr_end < sub_gap_frames:
                curr_end = next_end
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

        heights = np.array([d.get("player_height", 0) for d in frame_data])
        hip_vels = np.array([d.get("hip_vel", 0) for d in frame_data])
        min_frames = int(MIN_RALLY_DURATION_SEC * effective_fps)

        final_segments: List[RallySegment] = []
        for start_idx, end_idx in merged:
            if (end_idx - start_idx) < min_frames:
                continue

            segment_arms = smoothed_arm[start_idx:end_idx]
            segment_whips = smoothed_whip[start_idx:end_idx]
            segment_heights = heights[start_idx:end_idx]
            segment_t_arm = t_arm_arr[start_idx:end_idx]
            segment_t_whip = t_whip_arr[start_idx:end_idx]

            valid_baseline = (
                (segment_arms > segment_t_arm)
                & (segment_whips > segment_t_whip)
                & (segment_heights > self.min_player_height)
                & (segment_heights <= self.max_player_height)
            )
            valid_volley = (
                (segment_arms > np.maximum(1.5, segment_t_arm * 1.2))
                & (segment_whips > np.maximum(1.2, segment_t_whip))
                & (segment_heights > self.max_player_height)
            )
            if not (np.any(valid_baseline) or np.any(valid_volley)):
                continue

            if np.mean(hip_vels[start_idx:end_idx]) < config.feed_hip_velocity_min:
                continue

            if np.max(smoothed_energy[start_idx:end_idx]) < MOTION_ENERGY_THRESHOLD * 2:
                continue

            variances = [d.get("spatial_variance", 0) for d in frame_data[start_idx:end_idx]]
            if (np.max(variances) if variances else 0) < MIN_COURT_COVERAGE:
                continue

            t_start = max(0.0, frame_data[start_idx]["timestamp"] - config.boundary_padding_sec)
            t_end = min(self.total_frames / config.fps if config.fps else t_start,
                        frame_data[end_idx - 1]["timestamp"] + config.boundary_padding_sec)
            final_segments.append(RallySegment(
                start_frame=int(t_start * config.fps),
                end_frame=int(t_end * config.fps),
                start_time=t_start,
                end_time=t_end,
                score=float(np.max(segment_arms)),
            ))

        return final_segments
