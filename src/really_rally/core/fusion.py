"""Evidence fusion: combine pose, racket, ball and temporal evidence.

Replaces binary dependence on any single detector with a weighted confidence.
The scoring rules encode the reasoning agreed for this project:

    strong pose + racket + ball            -> very high confidence
    strong pose + racket, ball missing     -> likely a rally (ball is hard)
    strong pose, no ball over a long window-> possible shadow practice
    movement without stroke structure      -> not a rally
    ball alone, no pose/stroke evidence    -> insufficient

The neutral value (0.5) is load-bearing: it is what an unavailable detector
contributes. A detector that never ran must not look like a detector that ran
and found nothing, or the pipeline would reject genuine rallies whenever a
model failed to load. Weights and thresholds are configurable so they can be
tuned against labelled data rather than hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from really_rally.core.config import (
    FUSION_MIN_CONFIDENCE,
    FUSION_NEUTRAL,
    FUSION_W_BALL,
    FUSION_W_POSE,
    FUSION_W_RACKET,
    FUSION_W_TEMPORAL,
)


@dataclass
class RallyEvidence:
    """Per-window evidence. Every score is in [0, 1] with 0.5 = no information."""

    pose_score: float = FUSION_NEUTRAL
    racket_score: Optional[float] = None
    ball_score: Optional[float] = None
    temporal_score: float = FUSION_NEUTRAL


class EvidenceFusion:
    """Weighted fusion of independent evidence channels."""

    def __init__(
        self,
        w_pose: float = FUSION_W_POSE,
        w_racket: float = FUSION_W_RACKET,
        w_ball: float = FUSION_W_BALL,
        w_temporal: float = FUSION_W_TEMPORAL,
        neutral: float = FUSION_NEUTRAL,
        min_confidence: float = FUSION_MIN_CONFIDENCE,
    ):
        self.w_pose = w_pose
        self.w_racket = w_racket
        self.w_ball = w_ball
        self.w_temporal = w_temporal
        self.neutral = neutral
        self.min_confidence = min_confidence

    def fuse(self, evidence: RallyEvidence) -> float:
        """Return overall rally confidence in [0, 1].

        A ``None`` channel means "no detector available" and contributes the
        neutral score at its weight; an explicit low score means the detector
        ran and found nothing, which does count against the rally.
        """
        racket = self.neutral if evidence.racket_score is None else evidence.racket_score
        ball = self.neutral if evidence.ball_score is None else evidence.ball_score

        total_w = self.w_pose + self.w_racket + self.w_ball + self.w_temporal
        if total_w <= 0:
            return self.neutral

        score = (
            self.w_pose * evidence.pose_score
            + self.w_racket * racket
            + self.w_ball * ball
            + self.w_temporal * evidence.temporal_score
        ) / total_w
        return max(0.0, min(1.0, score))

    def accept(self, evidence: RallyEvidence) -> bool:
        return self.fuse(evidence) >= self.min_confidence

    # ------------------------------------------------------------------
    # Channel scorers
    # ------------------------------------------------------------------

    def score_racket(self, racket_frames: int, frames_checked: int,
                     near_hand_frames: int) -> Optional[float]:
        """Score racket evidence for a candidate window.

        Returns ``None`` when no detector ran (no information). A detector that
        ran and found nothing across a window returns a low-but-not-zero score:
        a missed box must not destroy a genuine rally, so the floor stays well
        above zero.
        """
        if frames_checked <= 0:
            return None
        # Never let "no racket seen" collapse to a hard veto.
        presence = min(1.0, racket_frames / max(1, frames_checked * 0.3))
        hand_consistency = (near_hand_frames / racket_frames) if racket_frames else 0.0
        return 0.15 + 0.85 * (0.6 * presence + 0.4 * hand_consistency)

    def score_ball(self, trajectory_confidence: float, observations: int,
                   frames_checked: int) -> Optional[float]:
        """Score ball evidence. Trajectory confidence dominates.

        Single-frame detections carry little weight; a tracked trajectory is
        what makes ball evidence meaningful. Ball never gates on its own.

        Returns ``None`` when no detector ran. A detector that ran and saw no
        ball returns a low-but-nonzero score, matching `score_racket`: ball
        detection is intrinsically hard, so absence must not be a hard veto.
        """
        if frames_checked <= 0:
            return None
        if observations <= 0:
            return 0.15  # detector ran, saw nothing: weak negative signal
        return max(0.15, min(1.0, trajectory_confidence))
