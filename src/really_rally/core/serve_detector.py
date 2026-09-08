"""Pose-based tennis serve detection.

A finite-state machine + confidence scorer that recognizes a conventional
overhead serve from a 2D pose time series (MediaPipe 33-keypoint skeleton),
without ball/racket/court detection.

The discriminative serve signature in pose space is:

    toss arm rises above the shoulder (and head) and is held there
        -> the *other* arm swings overhead with a velocity spike
        -> that arm extends and comes down through the follow-through.

Groundstrokes and overhead smashes lack the sustained self-toss; a serve return
lacks both the toss and the overhead swing. The sequence and timing therefore
carry more information than any single frame.

The class is deliberately stream-agnostic: it consumes the per-frame feature
dicts produced by `FeatureEngine.compute_features`, so a ball/racket/court
stream can later be folded in as extra scoring terms without changing the FSM.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from really_rally.core.config import (
    BOUNDARY_PADDING_SEC,
    SERVE_CONTACT_ELBOW_ANGLE,
    SERVE_CONTACT_MIN_SCORE,
    SERVE_CONTACT_WRIST_VEL,
    SERVE_FOLLOW_THROUGH_SEC,
    SERVE_MIN_SCORE,
    SERVE_TOSS_ABOVE_SHOULDER_MARGIN,
    SERVE_TOSS_MIN_SCORE,
    SERVE_TOSS_MIN_SEC,
    SERVE_TOSS_TO_CONTACT_MAX_SEC,
    SERVE_TOSS_TO_CONTACT_MIN_SEC,
    SERVE_WEIGHT_CONTACT,
    SERVE_WEIGHT_FOLLOW_THROUGH,
    SERVE_WEIGHT_TOSS,
)
from really_rally.core.models import ServeEvent

# Per-arm feature keys emitted by FeatureEngine.
_ARM_FIELDS = {
    "left": {
        "wrist_y": "l_wrist_y",
        "wrist_x": "l_wrist_x",
        "shoulder_y": "l_shoulder_y",
        "wrist_vel": "l_wrist_vel",
        "wrist_vy": "l_wrist_vy",
        "elbow_angle": "l_elbow_angle",
    },
    "right": {
        "wrist_y": "r_wrist_y",
        "wrist_x": "r_wrist_x",
        "shoulder_y": "r_shoulder_y",
        "wrist_vel": "r_wrist_vel",
        "wrist_vy": "r_wrist_vy",
        "elbow_angle": "r_elbow_angle",
    },
}
_OTHER_ARM = {"left": "right", "right": "left"}


def _runs(mask: np.ndarray) -> List[tuple]:
    """Return (start, end) inclusive index pairs of contiguous True runs."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    window = max(1, min(window, len(arr)))
    if window == 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


class ServeDetector:
    """Detect serve events from a downsampled pose-feature time series."""

    def __init__(self, effective_fps: float):
        # effective_fps accounts for PROCESS_EVERY_N_FRAMES downsampling.
        self.effective_fps = max(effective_fps, 1e-3)
        self.toss_min_frames = max(1, int(SERVE_TOSS_MIN_SEC * self.effective_fps))
        self.contact_max_frames = max(
            self.toss_min_frames,
            int(SERVE_TOSS_TO_CONTACT_MAX_SEC * self.effective_fps),
        )
        self.follow_through_frames = max(1, int(SERVE_FOLLOW_THROUGH_SEC * self.effective_fps))
        self.pad_frames = int(BOUNDARY_PADDING_SEC * self.effective_fps)

    def detect(self, frame_data: List[Dict]) -> List[ServeEvent]:
        n = len(frame_data)
        if n == 0:
            return []

        valid = np.array([bool(d.get("is_valid_pose", False)) for d in frame_data])
        if valid.sum() < self.toss_min_frames:
            return []

        timestamps = np.array([d.get("timestamp", 0.0) for d in frame_data])
        nose_y = np.array([d.get("nose_y", 0.0) for d in frame_data])

        arms = {}
        for name, keys in _ARM_FIELDS.items():
            arms[name] = {
                "wrist_y": np.array([d.get(keys["wrist_y"], 0.0) for d in frame_data]),
                "shoulder_y": np.array([d.get(keys["shoulder_y"], 0.0) for d in frame_data]),
                "wrist_vel": np.array([d.get(keys["wrist_vel"], 0.0) for d in frame_data]),
                "elbow_angle": np.array([d.get(keys["elbow_angle"], 180.0) for d in frame_data]),
            }

        # Raised-arm masks (only on valid poses).
        raised = {}
        above_head = {}
        for name in arms:
            arm = arms[name]
            with np.errstate(invalid="ignore"):
                raised[name] = (
                    valid
                    & (arm["wrist_y"] < arm["shoulder_y"] - SERVE_TOSS_ABOVE_SHOULDER_MARGIN)
                )
                above_head[name] = valid & (arm["wrist_y"] < nose_y)

        # Smooth wrist velocity to robustly locate the swing, but score on the
        # raw velocity of the true spike (smoothing otherwise smears the peak).
        vel_smooth = {name: _smooth(arms[name]["wrist_vel"], 3) for name in arms}

        toss_runs = self._detect_toss_runs(raised, timestamps)
        contacts = self._detect_contacts(vel_smooth, raised, arms)

        serves: List[ServeEvent] = []
        used_contacts = set()

        for toss_arm, run in toss_runs:
            hit_arm = _OTHER_ARM[toss_arm]
            toss_score, toss_idx = self._score_toss(toss_arm, run, arms, above_head, timestamps)
            if toss_score < SERVE_TOSS_MIN_SCORE:
                continue

            contact = self._find_contact(hit_arm, toss_idx, contacts, used_contacts)
            if contact is None:
                continue

            contact_idx = contact["idx"]
            contact_score = self._score_contact(contact)
            if contact_score < SERVE_CONTACT_MIN_SCORE:
                continue

            follow_through_score = self._score_follow_through(hit_arm, contact_idx, arms)

            score = (
                SERVE_WEIGHT_TOSS * toss_score
                + SERVE_WEIGHT_CONTACT * contact_score
                + SERVE_WEIGHT_FOLLOW_THROUGH * follow_through_score
            )
            if score < SERVE_MIN_SCORE:
                continue

            start_idx = max(0, run[0] - self.pad_frames)
            end_idx = min(n - 1, contact_idx + self.follow_through_frames + self.pad_frames)
            if end_idx <= start_idx:
                continue

            used_contacts.add(contact_idx)
            serves.append(
                ServeEvent(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=float(timestamps[start_idx]),
                    end_time=float(timestamps[end_idx]),
                    score=float(score),
                    handedness="right" if hit_arm == "right" else "left",
                    toss_arm=toss_arm,
                    toss_idx=int(toss_idx),
                    contact_idx=int(contact_idx),
                    toss_time=float(timestamps[toss_idx]),
                    contact_time=float(timestamps[contact_idx]),
                    toss_score=float(toss_score),
                    contact_score=float(contact_score),
                    follow_through_score=float(follow_through_score),
                )
            )

        # Deduplicate overlapping serves, keeping the higher-scoring one.
        return self._dedupe(serves)

    # ------------------------------------------------------------------ toss
    def _detect_toss_runs(self, raised, timestamps):
        """Return [(arm, (start, end)), ...] for every sustained raised-arm run.

        The toss arm rises while the opposite arm is still down. A raised-arm
        run that begins when the other arm is already up is the hitting arm's
        trophy/loading position, not a toss, and is discarded to avoid
        double-counting the same serve with inverted handedness.
        """
        runs = []
        for name in raised:
            other = _OTHER_ARM[name]
            for s, e in _runs(raised[name]):
                duration = timestamps[e] - timestamps[s]
                if duration < SERVE_TOSS_MIN_SEC:
                    continue
                if raised[other][s]:
                    continue
                runs.append((name, (s, e)))
        return runs

    def _score_toss(self, arm, run, arms, above_head, timestamps):
        """Score a raised-arm run as a toss; return (score, highest-point index).

        A convincing toss is sustained, reaches well above the shoulder (ideally
        the head), and the highest point marks ball release.
        """
        s, e = run
        wrist_y = arms[arm]["wrist_y"]
        shoulder_y = arms[arm]["shoulder_y"]

        duration = timestamps[e] - timestamps[s]
        duration_score = min(1.0, duration / 0.4)

        peak_raise = float(np.max(shoulder_y[s : e + 1] - wrist_y[s : e + 1]))
        height_score = min(1.0, max(0.0, peak_raise) / 0.15)

        head_reached = bool(np.any(above_head[arm][s : e + 1]))
        head_score = 1.0 if head_reached else 0.0

        release_idx = s + int(np.argmin(wrist_y[s : e + 1]))

        score = 0.5 * duration_score + 0.3 * height_score + 0.2 * head_score
        return float(min(1.0, score)), release_idx

    # ---------------------------------------------------------------- contact
    def _detect_contacts(self, vel_smooth, raised, arms):
        """Return {arm: [contact_dict, ...]} for overhead swing regions.

        A swing is located by its smoothed velocity crossing the threshold on a
        raised arm. Each region yields its peak raw velocity (swing speed), its
        peak wrist raise above the shoulder (overhead height), and the frame of
        peak velocity as the contact index. Scoring region maxima rather than a
        single frame keeps the estimate stable when the velocity spike and the
        highest wrist position are a few frames apart.
        """
        contacts = {name: [] for name in vel_smooth}
        for name in vel_smooth:
            mask = (vel_smooth[name] > SERVE_CONTACT_WRIST_VEL) & raised[name]
            raw = arms[name]["wrist_vel"]
            wy = arms[name]["wrist_y"]
            sy = arms[name]["shoulder_y"]
            ea = arms[name]["elbow_angle"]
            for s, e in _runs(mask):
                idx = s + int(np.argmax(raw[s : e + 1]))
                contacts[name].append({
                    "idx": idx,
                    "peak_vel": float(np.max(raw[s : e + 1])),
                    "peak_raise": float(np.max(sy[s : e + 1] - wy[s : e + 1])),
                    "elbow": float(ea[idx]),
                })
        return contacts

    def _find_contact(self, hit_arm, release_idx, contacts, used):
        """Pick the strongest swing on the hit arm after the toss release."""
        lo = release_idx + int(SERVE_TOSS_TO_CONTACT_MIN_SEC * self.effective_fps)
        hi = release_idx + self.contact_max_frames
        best = None
        best_vel = -1.0
        for c in contacts.get(hit_arm, []):
            if c["idx"] in used or c["idx"] < lo or c["idx"] > hi:
                continue
            if c["peak_vel"] > best_vel:
                best_vel = c["peak_vel"]
                best = c
        return best

    def _score_contact(self, contact):
        vel_factor = min(1.0, contact["peak_vel"] / (2.0 * SERVE_CONTACT_WRIST_VEL))
        height_factor = min(1.0, max(0.0, contact["peak_raise"]) / 0.15)
        elbow_factor = min(
            1.0,
            max(0.0, (contact["elbow"] - SERVE_CONTACT_ELBOW_ANGLE) / (170.0 - SERVE_CONTACT_ELBOW_ANGLE)),
        )
        return float(0.5 * vel_factor + 0.3 * height_factor + 0.2 * elbow_factor)

    # ---------------------------------------------------------- follow-through
    def _score_follow_through(self, hit_arm, contact_idx, arms):
        wrist_y = arms[hit_arm]["wrist_y"]
        lo = contact_idx + 1
        hi = min(len(wrist_y), contact_idx + self.follow_through_frames)
        if lo >= hi:
            return 0.0
        # The arm should drop (y increases) after contact.
        drop = float(np.max(wrist_y[lo:hi]) - wrist_y[contact_idx])
        return min(1.0, max(0.0, drop) / 0.15)

    # ------------------------------------------------------------------ utils
    def _dedupe(self, serves: List[ServeEvent]) -> List[ServeEvent]:
        """Drop lower-scoring serves that overlap a higher-scoring one."""
        serves = sorted(serves, key=lambda s: s.score, reverse=True)
        kept: List[ServeEvent] = []
        for s in serves:
            if any(s.start_idx < k.end_idx and s.end_idx > k.start_idx for k in kept):
                continue
            kept.append(s)
        return sorted(kept, key=lambda s: s.start_idx)
