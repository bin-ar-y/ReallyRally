"""Object evidence: racket and ball detection interfaces + ball tracking.

This module defines the *contract* for object evidence without committing to a
detector implementation. The pipeline ships with :class:`NullObjectDetector`,
which reports no information rather than "absent", so a missing or broken model
can never turn a genuine rally into a rejection.

Design rules (from the detection upgrade plan):

- Object evidence is *supporting*, never an absolute gate. Racket detection is
  more reliable than ball detection (larger, persistent, rarely occluded);
  ball detection is intrinsically hard (a tennis ball is small, fast, motion
  blurred, intermittently occluded, and sometimes a few pixels wide).
- A missed detection and a confident absence are different things. Detectors
  return :class:`FrameObjects` where ``available=False`` means "no information"
  and must be scored neutrally.
- Detectors run only inside candidate windows, never over a whole video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from really_rally.core.config import (
    BALL_TRACK_GATE_NORM,
    BALL_TRACK_MAX_GAP_FRAMES,
    BALL_TRACK_MIN_OBSERVATIONS,
    RACKET_HAND_MAX_DIST_NORM,
)

BBox = Tuple[float, float, float, float]  # normalized (x1, y1, x2, y2)

RACKET_LABEL = "tennis racket"
BALL_LABEL = "sports ball"


@dataclass
class Detection:
    """One detected object in one frame. Boxes are normalized to [0, 1]."""

    label: str
    confidence: float
    bbox: BBox

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FrameObjects:
    """Object detections for a single frame.

    ``available`` distinguishes "the detector ran and found nothing" from "no
    detector ran". Only the former is evidence of absence.
    """

    frame_idx: int
    rackets: List[Detection] = field(default_factory=list)
    balls: List[Detection] = field(default_factory=list)
    available: bool = True

    @property
    def best_racket(self) -> Optional[Detection]:
        return max(self.rackets, key=lambda d: d.confidence, default=None)

    @property
    def best_ball(self) -> Optional[Detection]:
        return max(self.balls, key=lambda d: d.confidence, default=None)


class ObjectDetector:
    """Base class. Subclasses implement :meth:`detect`."""

    #: False for the null detector, so fusion can treat output as uninformative.
    provides_evidence: bool = False

    def detect(self, frame_bgr, frame_idx: int) -> FrameObjects:  # pragma: no cover
        raise NotImplementedError


class NullObjectDetector(ObjectDetector):
    """Default backend: reports no information at all.

    Every frame is returned with ``available=False``, which downstream fusion
    maps to a neutral score. This is what keeps the pose-only pipeline's
    behaviour unchanged until a real model is configured.
    """

    provides_evidence = False

    def detect(self, frame_bgr, frame_idx: int) -> FrameObjects:
        return FrameObjects(frame_idx=frame_idx, available=False)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_FACTORY = None


def register_object_detector(factory) -> None:
    """Install the callable that builds the active object detector.

    Call this once at start-up (or from a plugin module) to enable real object
    verification. The factory takes no arguments and returns an
    :class:`ObjectDetector`. Until it is called, the pipeline uses the null
    backend and produces identical results to the pose-only version.
    """
    global _FACTORY
    _FACTORY = factory


def get_object_detector() -> ObjectDetector:
    """Build the registered detector, or the null backend if none is set."""
    if _FACTORY is None:
        return NullObjectDetector()
    return _FACTORY()


class BallTracker:
    """Short-range tracker turning per-frame ball proposals into a trajectory.

    Frame-by-frame ball detections are noisy and frequently missing. A ball
    that appears in several nearby frames on a plausible path is far stronger
    evidence than any single detection, so tracking — not a bigger model — is
    the preferred way to raise ball confidence.
    """

    def __init__(
        self,
        gate: float = BALL_TRACK_GATE_NORM,
        max_gap_frames: int = BALL_TRACK_MAX_GAP_FRAMES,
        min_observations: int = BALL_TRACK_MIN_OBSERVATIONS,
    ):
        self.gate = gate
        self.max_gap_frames = max_gap_frames
        self.min_observations = min_observations
        self.tracks: List[dict] = []

    def update(self, frame_idx: int, detections: Sequence[Detection]) -> None:
        """Associate this frame's ball detections with existing tracks."""
        for det in detections:
            cx, cy = det.center
            best, best_dist = None, self.gate
            for track in self.tracks:
                if frame_idx - track["last_frame"] > self.max_gap_frames:
                    continue
                lx, ly = track["positions"][-1]
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = track, dist
            if best is not None:
                best["positions"].append((cx, cy))
                best["last_frame"] = frame_idx
                best["confidences"].append(det.confidence)
            else:
                self.tracks.append({
                    "positions": [(cx, cy)],
                    "confidences": [det.confidence],
                    "start_frame": frame_idx,
                    "last_frame": frame_idx,
                })

        # Drop tracks that have gone stale so they cannot absorb later detections.
        self.tracks = [t for t in self.tracks if frame_idx - t["last_frame"] <= self.max_gap_frames]

    def confidence(self, frame_idx: int) -> float:
        """Trajectory confidence in [0, 1] at the current frame.

        A track only counts once it has ``min_observations`` points, so an
        isolated false positive cannot masquerade as ball flight. Longer,
        smoother, higher-confidence tracks score higher.
        """
        live = [t for t in self.tracks if frame_idx - t["last_frame"] <= self.max_gap_frames]
        if not live:
            return 0.0

        best = 0.0
        for t in live:
            n = len(t["positions"])
            if n < self.min_observations:
                continue
            length_score = min(1.0, n / (2.0 * self.min_observations))
            mean_conf = sum(t["confidences"]) / n
            best = max(best, length_score * mean_conf)
        return best

    def reset(self) -> None:
        self.tracks = []


def racket_near_hand(racket: Detection, wrist_xy: Optional[Tuple[float, float]],
                     max_dist: float = RACKET_HAND_MAX_DIST_NORM) -> bool:
    """True if a racket box plausibly belongs to the given wrist.

    Uses the nearest point of the box rather than its centre, because a racket
    box is elongated and the hand sits at one end of it.
    """
    if wrist_xy is None:
        return False
    wx, wy = wrist_xy
    x1, y1, x2, y2 = racket.bbox
    nx = min(max(wx, x1), x2)
    ny = min(max(wy, y1), y2)
    return ((wx - nx) ** 2 + (wy - ny) ** 2) ** 0.5 <= max_dist
