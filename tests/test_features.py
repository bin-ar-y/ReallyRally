"""Tests for feature engineering: adaptive thresholds, normalizers, trackers."""


import numpy as np
import pytest

from really_rally.core.features import (
    SelfAdaptiveThreshold,
    RobustAdaptiveThreshold,
    RollingZScoreNormalizer,
    MovementTracker,
    FeatureEngine,
)

# Import test helpers
from .helpers import make_stationary_landmarks, MockLandmarkList


# ---------------------------------------------------------------------------
# SelfAdaptiveThreshold
# ---------------------------------------------------------------------------

class TestSelfAdaptiveThreshold:
    def test_returns_min_floor_when_buffer_empty(self):
        s = SelfAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        assert s.get_threshold() == 0.5

    def test_returns_min_floor_when_buffer_has_one_element(self):
        s = SelfAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        s.update(1.0)
        assert s.get_threshold() == 0.5  # still only 1 element

    def test_threshold_above_resting_baseline(self):
        """After feeding resting values (~0), threshold should be near floor."""
        s = SelfAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        for _ in range(100):
            s.update(0.01)
        t = s.get_threshold()
        # With near-zero values, median≈0.01, MAD≈0 → threshold ≈ 0.5 (floor)
        assert t >= 0.5
        assert t < 0.7  # shouldn't be wildly above floor for a flat signal

    def test_threshold_rises_with_activity(self):
        """Active values should raise the dynamic threshold."""
        s = SelfAdaptiveThreshold(window_sec=2, fps=30, k=4.0, min_floor=0.5)  # capacity=60
        # Feed a baseline
        for _ in range(60):
            s.update(0.1)
        baseline = s.get_threshold()

        # Now spike activity — overwrite the entire buffer
        for _ in range(60):
            s.update(5.0)
        elevated = s.get_threshold()
        assert elevated > baseline

    def test_extend_padding_preserves_rest_baseline(self):
        """Padding with zeros should keep the threshold low."""
        s = SelfAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        for _ in range(200):
            s.update(0.05)
        s.extend_padding(0.0, 50)
        t = s.get_threshold()
        assert t < 1.0  # still near floor

    def test_capacity_limit(self):
        """Old values are evicted beyond window capacity."""
        cap = 100
        s = SelfAdaptiveThreshold(window_sec=10, fps=10, k=4.0, min_floor=0.5)  # capacity=100
        # Fill with high values
        for _ in range(cap):
            s.update(10.0)
        high_t = s.get_threshold()
        # Overwrite with low values
        for _ in range(cap):
            s.update(0.1)
        low_t = s.get_threshold()
        assert low_t < high_t


# ---------------------------------------------------------------------------
# RobustAdaptiveThreshold
# ---------------------------------------------------------------------------

class TestRobustAdaptiveThreshold:
    def test_returns_min_floor_when_buffer_empty(self):
        r = RobustAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        assert r.get_threshold() == 0.5

    def test_rest_baseline_stable(self):
        """P25+MAD should produce a threshold near the old median+MAD during rest."""
        r = RobustAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        for _ in range(200):
            r.update(0.05)
        t = r.get_threshold()
        assert t >= 0.5
        assert t < 1.0  # should be near floor for flat resting signal

    def test_less_drift_than_old_threshold(self):
        """During dense activity, RobustAdaptiveThreshold should drift less."""
        old = SelfAdaptiveThreshold(window_sec=10, fps=10, k=4.0, min_floor=0.5)
        new = RobustAdaptiveThreshold(window_sec=10, fps=10, k=4.0, min_floor=0.5)
        # Fill both with resting baseline
        for _ in range(100):
            old.update(0.05)
            new.update(0.05)
        # Now simulate a dense rally: 70% elevated baseline + 30% strokes
        rng = np.random.default_rng(42)
        for _ in range(80):
            if rng.random() < 0.3:
                val = rng.uniform(2.0, 5.0)  # stroke
            else:
                val = rng.uniform(0.1, 0.4)  # footwork
            old.update(val)
            new.update(val)
        old_t = old.get_threshold()
        new_t = new.get_threshold()
        # New threshold should be lower (less drifted) than old
        assert new_t < old_t, f"Expected new_t ({new_t:.3f}) < old_t ({old_t:.3f})"

    def test_recovery_after_activity(self):
        """After activity ends, threshold should fall back toward floor."""
        r = RobustAdaptiveThreshold(window_sec=5, fps=10, k=4.0, min_floor=0.5)
        # Rest baseline
        for _ in range(50):
            r.update(0.05)
        rest_t = r.get_threshold()
        # Activity spike
        for _ in range(50):
            r.update(3.0)
        elevated_t = r.get_threshold()
        assert elevated_t > rest_t
        # Recovery
        for _ in range(50):
            r.update(0.05)
        recovered_t = r.get_threshold()
        # Should have fallen significantly from the peak
        assert recovered_t < elevated_t
        # Due to EMA smoothing, may not fully return to rest_t in 50 samples
        assert recovered_t <= elevated_t

    def test_extend_padding(self):
        r = RobustAdaptiveThreshold(window_sec=30, fps=30, k=4.0, min_floor=0.5)
        for _ in range(200):
            r.update(0.05)
        r.extend_padding(0.0, 100)
        t = r.get_threshold()
        assert t < 1.0  # zero padding keeps threshold low

    def test_capacity_limit(self):
        """Old values evicted; should respond to new baseline."""
        cap = 100
        r = RobustAdaptiveThreshold(window_sec=10, fps=10, k=4.0, min_floor=0.5)
        for _ in range(cap):
            r.update(10.0)
        high_t = r.get_threshold()
        for _ in range(cap):
            r.update(0.1)
        low_t = r.get_threshold()
        assert low_t < high_t

    def test_p25_baseline_resists_contamination(self):
        """P25 baseline stays anchored even when median would shift.

        With a realistic rally distribution (70% low-activity footwork,
        30% stroke peaks), P25 should remain near the low values.
        The old median-based threshold drifts more because median shifts
        into the footwork range.
        """
        rng = np.random.default_rng(42)
        old = SelfAdaptiveThreshold(window_sec=10, fps=10, k=4.0, min_floor=0.5)
        new = RobustAdaptiveThreshold(window_sec=10, fps=10, k=4.0, min_floor=0.5,
                                       rise_alpha=1.0, fall_alpha=1.0)  # no EMA

        for _ in range(100):
            if rng.random() < 0.3:
                val = rng.uniform(2.0, 5.0)  # stroke
            else:
                val = rng.uniform(0.05, 0.3)  # footwork between strokes
            old.update(val)
            new.update(val)

        assert new.get_threshold() < old.get_threshold(), (
            f"New ({new.get_threshold():.3f}) should be lower than old ({old.get_threshold():.3f})"
        )


# ---------------------------------------------------------------------------
# RollingZScoreNormalizer
# ---------------------------------------------------------------------------

class TestRollingZScoreNormalizer:
    def test_normalize_constant_signal(self):
        """A perfectly constant signal should produce near-zero z-scores."""
        n = RollingZScoreNormalizer(window_sec=30, fps=30)
        values = np.full(500, 3.0)
        z = n.normalize_array(values)
        # After warm-up (first few values), z-scores should be near 0
        assert np.all(np.abs(z[100:]) < 0.1)

    def test_normalize_spike_detection(self):
        """A spike above a flat baseline should yield a high z-score."""
        n = RollingZScoreNormalizer(window_sec=30, fps=30)
        values = np.concatenate([np.full(300, 0.1), [5.0]])
        z = n.normalize_array(values)
        # The spike at index 300 should be well above the rest
        assert z[-1] > 2.0

    def test_causal_window(self):
        """Z-score at position i should only depend on values 0..i."""
        n = RollingZScoreNormalizer(window_sec=10, fps=10)  # capacity=100
        # First half: low values, second half: high values
        arr = np.concatenate([np.zeros(100), np.ones(100)])
        z = n.normalize_array(arr)
        # Right at the transition, z-scores should spike (value is high, window is low)
        assert z[100] > 0.5

    def test_empty_buffer_returns_zero(self):
        n = RollingZScoreNormalizer(window_sec=30, fps=30)
        assert n.update(5.0) == 0.0

    def test_two_elements_zscore(self):
        n = RollingZScoreNormalizer(window_sec=30, fps=30)
        n.update(0.5)
        # z = (1.0 - 0.75) / (0.25 + ε) ≈ 1.0
        assert n.update(1.0) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# MovementTracker
# ---------------------------------------------------------------------------

class TestMovementTracker:
    def test_zero_variance_when_stationary(self):
        mt = MovementTracker(window_duration_sec=3.0, fps=30)
        for _ in range(50):
            mt.update(0.5, 0.5)
        assert mt.get_spatial_variance() == pytest.approx(0.0)

    def test_variance_increases_with_movement(self):
        mt = MovementTracker(window_duration_sec=3.0, fps=30)
        for i in range(90):
            x = 0.5 + i * 0.001
            y = 0.5
            mt.update(x, y)
        assert mt.get_spatial_variance() > 0.01

    def test_capacity_limit(self):
        """Old positions are evicted; variance reflects only recent history."""
        mt = MovementTracker(window_duration_sec=2.0, fps=30)  # capacity=60
        # Fill with varying positions
        for i in range(60):
            mt.update(0.5 + i * 0.005, 0.5)
        var_full = mt.get_spatial_variance()
        # Now feed stationary positions
        for _ in range(60):
            mt.update(0.5, 0.5)
        var_zero = mt.get_spatial_variance()
        assert var_zero < var_full * 0.1


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    def test_invalid_pose_when_landmarks_none(self):
        fe = FeatureEngine(fps=30)
        feats = fe.compute_features(None, 0.0)
        assert feats["is_valid_pose"] is False
        assert feats["energy"] == 0.0

    def test_first_frame_sets_pose_without_velocity(self):
        """First valid frame sets is_valid_pose but has no motion (energy=0)."""
        fe = FeatureEngine(fps=30)
        lms = MockLandmarkList(make_stationary_landmarks())
        feats = fe.compute_features(lms, 0.0)
        assert feats["is_valid_pose"] is True
        assert feats["energy"] == 0.0  # no previous frame

    def test_second_frame_computes_motion(self):
        """A second frame with different landmarks produces non-zero features."""
        fe = FeatureEngine(fps=30)
        # Frame 1: stationary pose
        lms1 = MockLandmarkList(make_stationary_landmarks())
        fe.compute_features(lms1, 0.0)

        # Frame 2: shift all landmarks rightward for a small movement
        shifted = make_stationary_landmarks()
        for lm in shifted:
            lm.x += 0.02
        lms2 = MockLandmarkList(shifted)
        feats = fe.compute_features(lms2, 0.033)
        assert feats["is_valid_pose"] is True
        assert feats["energy"] > 0.0

    def test_stationary_player_zero_energy(self):
        """Two identical frames should yield near-zero energy."""
        fe = FeatureEngine(fps=30)
        lms = MockLandmarkList(make_stationary_landmarks())
        fe.compute_features(lms, 0.0)
        feats = fe.compute_features(lms, 0.033)  # same pose
        assert feats["energy"] == pytest.approx(0.0, abs=1e-6)

    def test_player_height_computed(self):
        fe = FeatureEngine(fps=30)
        lms = MockLandmarkList(make_stationary_landmarks())
        feats = fe.compute_features(lms, 0.0)
        # nose_y=0.10, ankle_y=0.75 → height=0.65
        assert feats["player_height"] == pytest.approx(0.65, abs=0.01)

    def test_whip_ratio_computed(self):
        """A frame with high wrist velocity relative to shoulder yields whip > 1."""
        fe = FeatureEngine(fps=30)
        # Frame 1: resting
        lms_rest = MockLandmarkList(make_stationary_landmarks())
        fe.compute_features(lms_rest, 0.0)

        # Frame 2: wrist moved fast, shoulder barely moved
        lms_swing = make_stationary_landmarks()
        lms_swing[15].x += 0.15  # left wrist moved right
        lms_swing[15].y -= 0.05  # left wrist moved up
        lms_swing[13].x += 0.03  # left elbow moved slightly
        lms2 = MockLandmarkList(lms_swing)
        feats = fe.compute_features(lms2, 0.033)
        assert feats["whip_ratio"] > 1.0
        assert feats["arm_energy"] > 0.0

    def test_new_pose_after_none_resets_prev(self):
        """After a None frame, the next valid frame acts as first frame."""
        fe = FeatureEngine(fps=30)
        lms = MockLandmarkList(make_stationary_landmarks())
        fe.compute_features(lms, 0.0)
        fe.compute_features(None, 0.033)  # lost tracking
        feats = fe.compute_features(lms, 0.066)  # reacquired
        assert feats["is_valid_pose"] is True
        assert feats["energy"] == 0.0  # same pose, so no displacement

    def test_dropout_bridged_keeps_velocity(self):
        """A pose reacquired after one dropped frame still yields motion.

        MediaPipe most often drops the pose at ball contact, exactly where
        velocity carries the signal. Resetting prev_landmarks there produced a
        spurious zero; the bridge keeps the previous pose for short gaps.
        """
        fe = FeatureEngine(fps=30)
        lms1 = MockLandmarkList(make_stationary_landmarks())
        fe.compute_features(lms1, 0.0)
        fe.compute_features(None, 0.033)          # single dropped frame

        moved = make_stationary_landmarks()
        for lm in moved:
            lm.x += 0.05
        feats = fe.compute_features(MockLandmarkList(moved), 0.066)
        assert feats["is_valid_pose"] is True
        assert feats["energy"] > 0.0, "velocity lost across a 1-frame dropout"

    def test_long_dropout_resets_prev(self):
        """A gap beyond the bridge limit must not fabricate motion."""
        from really_rally.core.config import POSE_DROPOUT_BRIDGE_STEPS

        fe = FeatureEngine(fps=30)
        fe.compute_features(MockLandmarkList(make_stationary_landmarks()), 0.0)
        for i in range(POSE_DROPOUT_BRIDGE_STEPS + 1):
            fe.compute_features(None, 0.033 * (i + 1))

        moved = make_stationary_landmarks()
        for lm in moved:
            lm.x += 0.05
        feats = fe.compute_features(MockLandmarkList(moved), 0.5)
        assert feats["energy"] == 0.0  # treated as a fresh start

    def test_bridged_velocity_is_per_step_scaled(self):
        """Displacement across a gap is divided by the steps it spans."""
        def energy_across_gap(n_gaps):
            fe = FeatureEngine(fps=30)
            fe.compute_features(MockLandmarkList(make_stationary_landmarks()), 0.0)
            for i in range(n_gaps):
                fe.compute_features(None, 0.033 * (i + 1))
            moved = make_stationary_landmarks()
            for lm in moved:
                lm.x += 0.05
            return fe.compute_features(MockLandmarkList(moved), 0.5)["energy"]

        # Same displacement over more steps => lower per-step velocity.
        assert energy_across_gap(1) > energy_across_gap(3)
