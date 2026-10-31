"""Tests for the pose-based serve detector and per-arm feature extraction."""

import pytest

from really_rally.core.models import ServeEvent
from really_rally.core.serve_detector import ServeDetector, _runs
from really_rally.core.features import FeatureEngine, _joint_angle

from .helpers import MockLandmarkList, make_stationary_landmarks


def _frame(t, l_wy, r_wy, l_wv=0.02, r_wv=0.02, l_ea=160.0, r_ea=160.0, valid=True):
    return {
        "is_valid_pose": valid,
        "timestamp": t,
        "nose_y": 0.30,
        "l_shoulder_y": 0.45,
        "r_shoulder_y": 0.45,
        "l_wrist_x": 0.40,
        "r_wrist_x": 0.60,
        "l_wrist_y": l_wy,
        "r_wrist_y": r_wy,
        "l_wrist_vel": l_wv,
        "r_wrist_vel": r_wv,
        "l_wrist_vy": 0.0,
        "r_wrist_vy": 0.0,
        "l_elbow_angle": l_ea,
        "r_elbow_angle": r_ea,
    }


def _make_righty_serve(dt=1 / 30):
    """Synthetic right-handed serve: left arm tosses, right arm swings overhead."""
    frames = []
    t = 0.0

    def add(n, l_wy, r_wy, l_wv=0.02, r_wv=0.02, l_ea=160.0, r_ea=160.0):
        nonlocal t
        for _ in range(n):
            frames.append(_frame(t, l_wy, r_wy, l_wv, r_wv, l_ea, r_ea))
            t += dt

    add(6, 0.60, 0.60)            # idle, both arms down
    add(5, 0.30, 0.60)            # toss arm rises to head height
    add(10, 0.18, 0.60)           # toss arm held above the head
    add(3, 0.45, 0.35, r_wv=0.10)  # hit arm rising
    add(1, 0.50, 0.20, r_wv=0.50, r_ea=170.0)  # contact: overhead velocity peak
    add(4, 0.60, 0.45, r_wv=0.05)  # hit arm dropping
    add(6, 0.62, 0.62)            # follow-through, both arms low
    return frames


def _make_forehand(dt=1 / 30):
    """Forehand: no toss, one arm swings at waist height."""
    frames = []
    t = 0.0
    for _ in range(6):
        frames.append(_frame(t, 0.60, 0.60))
        t += dt
    # Right arm swings horizontally at waist height (never above shoulder).
    for _ in range(3):
        frames.append(_frame(t, 0.60, 0.62, r_wv=0.40))
        t += dt
    for _ in range(6):
        frames.append(_frame(t, 0.60, 0.60))
        t += dt
    return frames


# ---------------------------------------------------------------------------
# ServeDetector
# ---------------------------------------------------------------------------

class TestServeDetector:
    def test_detects_righty_serve(self):
        frames = _make_righty_serve()
        serves = ServeDetector(effective_fps=30).detect(frames)
        assert len(serves) == 1
        s = serves[0]
        assert s.handedness == "right"
        assert s.toss_arm == "left"
        assert s.toss_time < s.contact_time
        assert s.score > 0.0
        assert s.contact_score >= 0.5
        assert s.toss_score >= 0.5

    def test_detects_lefty_serve(self):
        # Mirror the righty serve: right arm tosses, left arm swings.
        frames = _make_righty_serve()
        for f in frames:
            f["l_wrist_y"], f["r_wrist_y"] = f["r_wrist_y"], f["l_wrist_y"]
            f["l_wrist_vel"], f["r_wrist_vel"] = f["r_wrist_vel"], f["l_wrist_vel"]
            f["l_elbow_angle"], f["r_elbow_angle"] = f["r_elbow_angle"], f["l_elbow_angle"]
        serves = ServeDetector(effective_fps=30).detect(frames)
        assert len(serves) == 1
        assert serves[0].handedness == "left"
        assert serves[0].toss_arm == "right"

    def test_forehand_not_detected(self):
        serves = ServeDetector(effective_fps=30).detect(_make_forehand())
        assert serves == []

    def test_empty_input(self):
        assert ServeDetector(effective_fps=30).detect([]) == []

    def test_all_invalid_pose(self):
        frames = [_frame(i / 30, 0.6, 0.6, valid=False) for i in range(30)]
        assert ServeDetector(effective_fps=30).detect(frames) == []

    def test_no_false_positive_on_stationary(self):
        frames = [_frame(i / 30, 0.60, 0.60) for i in range(60)]
        assert ServeDetector(effective_fps=30).detect(frames) == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestRuns:
    def test_runs(self):
        import numpy as np

        assert _runs(np.array([0, 0, 1, 1, 0, 1, 0])) == [(2, 3), (5, 5)]


# ---------------------------------------------------------------------------
# FeatureEngine per-arm fields
# ---------------------------------------------------------------------------

class TestPerArmFeatures:
    def test_per_arm_positions_populated(self):
        fe = FeatureEngine(fps=30)
        lms = MockLandmarkList(make_stationary_landmarks())
        feats = fe.compute_features(lms, 0.0)
        assert feats["l_wrist_y"] == pytest.approx(0.45, abs=0.01)
        assert feats["r_wrist_y"] == pytest.approx(0.45, abs=0.01)
        assert feats["l_shoulder_y"] == pytest.approx(0.25, abs=0.01)
        assert feats["nose_y"] == pytest.approx(0.10, abs=0.01)

    def test_elbow_angle_extends_on_straight_arm(self):
        # Straight arm (shoulder->elbow->wrist collinear) => angle near 180.
        fe = FeatureEngine(fps=30)
        lms = MockLandmarkList(make_stationary_landmarks())
        feats = fe.compute_features(lms, 0.0)
        # left: shoulder (0.44,0.25) -> elbow (0.42,0.35) -> wrist (0.38,0.45)
        assert feats["l_elbow_angle"] > 150.0
        assert feats["r_elbow_angle"] > 150.0

    def test_joint_angle_right_angle(self):
        import numpy as np

        a = np.array([0.0, 0.0])
        b = np.array([0.0, 1.0])
        c = np.array([1.0, 1.0])
        assert _joint_angle(a, b, c) == pytest.approx(90.0, abs=0.01)


class TestHandednessLocking:
    """A player serves with one hand for a whole session.

    Per-serve inference can flip when a toss/contact pair is mis-associated,
    which is physically implausible and makes serve_events.csv inconsistent.
    """

    @staticmethod
    def _event(hand, score, t):
        return ServeEvent(
            start_idx=0, end_idx=10, start_time=t, end_time=t + 2.0, score=score,
            handedness=hand, toss_arm="left" if hand == "right" else "right",
            toss_idx=1, contact_idx=5,
        )

    def test_dominant_hand_applied_to_all_events(self):
        d = ServeDetector(30.0, lock_handedness=True, forced_handedness=None)
        serves = [self._event("right", 0.9, 1.0),
                  self._event("right", 0.8, 5.0),
                  self._event("left", 0.6, 9.0)]
        out = d._lock_handedness(serves)
        assert {s.handedness for s in out} == {"right"}
        assert {s.toss_arm for s in out} == {"left"}

    def test_explicit_override_wins(self):
        d = ServeDetector(30.0, lock_handedness=True, forced_handedness="left")
        out = d._lock_handedness([self._event("right", 0.9, 1.0),
                                  self._event("left", 0.6, 5.0)])
        assert all(s.handedness == "left" and s.toss_arm == "right" for s in out)

    def test_locking_disabled_preserves_inference(self):
        d = ServeDetector(30.0, lock_handedness=False)
        out = d._lock_handedness([self._event("right", 0.9, 1.0),
                                  self._event("left", 0.6, 5.0)])
        assert [s.handedness for s in out] == ["right", "left"]

    def test_uses_total_evidence_not_event_count(self):
        """Three marginal left detections must not outvote two confident rights.

        Raw counting would pick 'left' here (3 > 2); weighting by score picks
        the hand with more accumulated evidence.
        """
        d = ServeDetector(30.0, lock_handedness=True, forced_handedness=None)
        out = d._lock_handedness([
            self._event("left", 0.55, 1.0),
            self._event("left", 0.55, 3.0),
            self._event("left", 0.55, 5.0),   # count 3, total 1.65
            self._event("right", 0.99, 9.0),
            self._event("right", 0.99, 12.0),  # count 2, total 1.98
        ])
        assert all(s.handedness == "right" for s in out)

    def test_empty_input_is_safe(self):
        assert ServeDetector(30.0)._lock_handedness([]) == []

    def test_single_event_is_kept(self):
        d = ServeDetector(30.0, lock_handedness=True, forced_handedness=None)
        out = d._lock_handedness([self._event("left", 0.7, 1.0)])
        assert len(out) == 1 and out[0].handedness == "left"
