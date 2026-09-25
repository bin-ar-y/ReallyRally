"""High-recall coarse candidate detection and the hysteresis rally state machine.

Two components for Phase C of the detection upgrade:

- :class:`CoarseCandidateDetector` turns one frame's feature dict plus its
  adaptive thresholds into an additive *candidate* evidence score. It replaces
  the old strict AND-gate that demanded arm energy AND whip ratio AND hip
  velocity AND spatial variance all fire at once — which discarded real
  strokes whose whip ratio collapsed under body rotation or whose hip velocity
  was near zero on a planted-foot hit. With the additive score a single strong
  signal (fast arm, or a pronounced whip) fires a candidate by itself, while
  weaker multi-signal evidence still accumulates. It is deliberately
  recall-first: object/temporal verification happens in later phases.

- :class:`RallyStateMachine` adds hysteresis so a rally is harder to START
  than to CONTINUE. Two candidates within ``start_gap`` confirm a rally; once
  active, a gap up to ``continue_gap`` keeps it alive through a Cooldown
  state, so a single missed stroke never truncates the segment.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from really_rally.core.config import (
    BOUNDARY_PADDING_SEC,
    COARSE_CANDIDATE_THRESHOLD,
    COARSE_W_ARM,
    COARSE_W_HIP,
    COARSE_W_SPATIAL,
    COARSE_W_WHIP,
    HIP_VELOCITY_STROKE_THRESHOLD,
    MIN_RALLY_DURATION_SEC,
    RALLY_CONTINUE_GAP_SEC,
    RALLY_START_GAP_SEC,
    SPATIAL_VARIANCE_STROKE_MIN,
)
from really_rally.core.models import RallySegment


class CoarseCandidateDetector:
    """Additive evidence scorer: which signals fire, and does that clear the bar."""

    def __init__(
        self,
        w_arm: float = COARSE_W_ARM,
        w_whip: float = COARSE_W_WHIP,
        w_hip: float = COARSE_W_HIP,
        w_spatial: float = COARSE_W_SPATIAL,
        candidate_threshold: float = COARSE_CANDIDATE_THRESHOLD,
        hip_min: float = HIP_VELOCITY_STROKE_THRESHOLD,
        spatial_min: float = SPATIAL_VARIANCE_STROKE_MIN,
    ):
        self.w_arm = w_arm
        self.w_whip = w_whip
        self.w_hip = w_hip
        self.w_spatial = w_spatial
        self.candidate_threshold = candidate_threshold
        self.hip_min = hip_min
        self.spatial_min = spatial_min

    def score(
        self,
        valid_pose: bool,
        valid_distance: bool,
        z_arm: float,
        t_arm: float,
        z_whip: float,
        t_whip: float,
        hip_vel: float,
        spatial_var: float,
    ) -> Tuple[float, List[str]]:
        """Return (evidence_score, fired_signals) for one frame."""
        if not (valid_pose and valid_distance):
            return 0.0, []

        score = 0.0
        fired: List[str] = []
        if z_arm > t_arm:
            score += self.w_arm
            fired.append("arm")
        if z_whip > t_whip:
            score += self.w_whip
            fired.append("whip")
        if hip_vel > self.hip_min:
            score += self.w_hip
            fired.append("hip")
        if spatial_var >= self.spatial_min:
            score += self.w_spatial
            fired.append("spatial")
        return score, fired

    def is_candidate(self, score: float) -> bool:
        return score >= self.candidate_threshold


class RallyStateMachine:
    """Hysteresis FSM: IDLE -> CANDIDATE -> ACTIVE -> COOLDOWN -> IDLE.

    Confirmation requires two candidate strokes more than
    ``double_trigger_guard`` apart within ``start_gap``. Once active, a gap
    longer than ``start_gap`` (but shorter than ``continue_gap``) enters
    Cooldown; a new candidate resumes, and only a gap longer than
    ``continue_gap`` ends the rally. This makes a rally harder to start than
    to continue and tolerates a single missed stroke.
    """

    IDLE = 0
    CANDIDATE = 1
    ACTIVE = 2
    COOLDOWN = 3

    def __init__(
        self,
        fps: float,
        max_time: float,
        start_gap_sec: float = RALLY_START_GAP_SEC,
        continue_gap_sec: float = RALLY_CONTINUE_GAP_SEC,
        min_duration_sec: float = MIN_RALLY_DURATION_SEC,
        boundary_padding_sec: float = BOUNDARY_PADDING_SEC,
        double_trigger_guard_sec: float = 0.5,
    ):
        self.fps = fps
        self.max_time = max_time
        self.start_gap = start_gap_sec
        self.continue_gap = continue_gap_sec
        self.min_duration = min_duration_sec
        self.pad = boundary_padding_sec
        self.double_trigger_guard = double_trigger_guard_sec
        self._reset()

    def _reset(self):
        self.state = self.IDLE
        self.start_time = 0.0
        self.start_frame = 0
        self.last_stroke_time = 0.0
        self.peak_score = 0.0

    @property
    def state_name(self) -> str:
        return {self.IDLE: "idle", self.CANDIDATE: "candidate",
                self.ACTIVE: "active", self.COOLDOWN: "cooldown"}[self.state]

    def step(self, candidate: bool, frame: int, time: float, score: float) -> Optional[RallySegment]:
        """Advance one frame; return a RallySegment iff a rally just ended."""
        segment: Optional[RallySegment] = None

        if self.state == self.IDLE:
            if candidate:
                self.state = self.CANDIDATE
                self._open(time, frame, score)

        elif self.state == self.CANDIDATE:
            if candidate:
                if (time - self.last_stroke_time) > self.double_trigger_guard:
                    self.state = self.ACTIVE
                    self._stroke(time, score)
            elif (time - self.last_stroke_time) > self.start_gap:
                self.state = self.IDLE

        elif self.state == self.ACTIVE:
            if candidate:
                self._stroke(time, score)
            elif (time - self.last_stroke_time) > self.continue_gap:
                # Gap jumped past both thresholds in one step (sparse sampling
                # across a segment boundary) — close without a cooldown detour.
                segment = self._close()
                self.state = self.IDLE
            elif (time - self.last_stroke_time) > self.start_gap:
                self.state = self.COOLDOWN

        elif self.state == self.COOLDOWN:
            if candidate:
                self.state = self.ACTIVE
                self._stroke(time, score)
            elif (time - self.last_stroke_time) > self.continue_gap:
                segment = self._close()
                self.state = self.IDLE

        return segment

    def finalize(self) -> Optional[RallySegment]:
        """Emit a trailing rally if the video ends while a rally is in progress."""
        if self.state in (self.ACTIVE, self.COOLDOWN):
            segment = self._close()
            self.state = self.IDLE
            return segment
        return None

    def _open(self, time: float, frame: int, score: float):
        self.start_time = max(0.0, time - self.pad)
        self.start_frame = max(0, frame - int(self.pad * self.fps))
        self.last_stroke_time = time
        self.peak_score = score

    def _stroke(self, time: float, score: float):
        self.last_stroke_time = time
        self.peak_score = max(self.peak_score, score)

    def _close(self) -> Optional[RallySegment]:
        end_time = min(self.last_stroke_time + self.pad, self.max_time)
        duration = end_time - self.start_time
        if duration < self.min_duration:
            return None
        return RallySegment(
            start_frame=self.start_frame,
            end_frame=int(end_time * self.fps),
            start_time=self.start_time,
            end_time=end_time,
            score=self.peak_score,
        )
