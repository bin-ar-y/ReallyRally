"""Tests for the extracted detection layer.

The refactor moved detection out of ``TennisRallyProcessor.analyze_segment_data``
into ``core.detection.RallyAnalyzer`` so the Debug Studio can replay it from a
cached feature store. That is only safe if the two behave identically, so the
central test here runs the new analyzer against a frozen copy of the *original*
inline implementation and requires identical output.

The frozen copy is deliberately verbose and unrefactored: it is a historical
record of what shipped, and it should not be "cleaned up".
"""

import numpy as np
import pytest

from really_rally.core.candidate import CoarseCandidateDetector, RallyStateMachine
from really_rally.core.config import (
    ADAPTIVE_WINDOW_SEC,
    ARM_ENERGY_K_MULTIPLIER,
    COARSE_W_ARM,
    COARSE_W_HIP,
    COARSE_W_SPATIAL,
    COARSE_W_WHIP,
    MIN_ENERGY_FLOOR,
    MIN_WHIP_FLOOR,
    PLAYER_HEIGHT_MAX,
    PLAYER_HEIGHT_MIN,
    PROCESS_EVERY_N_FRAMES,
    WHIP_RATIO_K_MULTIPLIER,
)
from really_rally.core.config_snapshot import ConfigSnapshot
from really_rally.core.detection import (
    AdaptiveState,
    FeedAnalyzer,
    RallyAnalyzer,
    ServeAnalyzer,
    mask_to_indices,
    mitigate_outliers,
    smooth,
)
from really_rally.core.features import RobustAdaptiveThreshold, RollingZScoreNormalizer

FPS = 30.0
TOTAL_FRAMES = 600
EFFECTIVE_FPS = FPS / PROCESS_EVERY_N_FRAMES


# ---------------------------------------------------------------------------
# A frozen copy of the pre-refactor implementation (base.py, verbatim logic).
# Do not modernize: its value is being exactly what used to run.
# ---------------------------------------------------------------------------

class _LegacyState:
    def __init__(self):
        self.arm_adaptive = RobustAdaptiveThreshold(
            ADAPTIVE_WINDOW_SEC, EFFECTIVE_FPS, ARM_ENERGY_K_MULTIPLIER, MIN_ENERGY_FLOOR)
        self.whip_adaptive = RobustAdaptiveThreshold(
            ADAPTIVE_WINDOW_SEC, EFFECTIVE_FPS, WHIP_RATIO_K_MULTIPLIER, MIN_WHIP_FLOOR)
        self.last_segment_end_frame = 0


def legacy_analyze_segment_data(frame_data, state):
    n = len(frame_data)
    if n == 0:
        return []

    first_frame = frame_data[0].get('frame', 0)
    gap_frames = max(0, first_frame - state.last_segment_end_frame)
    gap_samples = int(gap_frames / PROCESS_EVERY_N_FRAMES)
    pad_count = min(gap_samples, state.arm_adaptive.capacity)
    if pad_count > 0:
        state.arm_adaptive.extend_padding(0.0, pad_count)
        state.whip_adaptive.extend_padding(0.0, pad_count)

    arm_adaptive = state.arm_adaptive
    whip_adaptive = state.whip_adaptive

    energies = np.array([d.get('energy', 0) for d in frame_data])
    arm_energies = np.array([d.get('arm_energy', 0) for d in frame_data])
    hip_vels = np.array([d.get('hip_vel', 0) for d in frame_data])
    whips = np.array([d.get('whip_ratio', 0) for d in frame_data])

    def mitigate_outliers(arr, absolute_max):
        if len(arr) == 0:
            return arr
        arr_capped = np.clip(arr, 0, absolute_max)
        active_vals = arr_capped[arr_capped > 0.1]
        if len(active_vals) > 5:
            median = np.median(active_vals)
            mad = np.median(np.abs(active_vals - median))
            if mad > 0.01:
                dynamic_cap = median + 6 * mad
                dynamic_cap = max(dynamic_cap, 1.0)
                arr_capped = np.clip(arr_capped, 0, dynamic_cap)
        return arr_capped

    arm_energies = mitigate_outliers(arm_energies, absolute_max=4.0)
    whips = mitigate_outliers(whips, absolute_max=25.0)

    window = int(EFFECTIVE_FPS * 0.5)
    if window < 1:
        window = 1
    kernel = np.ones(window) / window

    smoothed_energy = np.convolve(energies, kernel, mode='same')
    smoothed_arm = np.convolve(arm_energies, kernel, mode='same')
    smoothed_whip = np.convolve(whips, kernel, mode='same')

    z_normalizer_arm = RollingZScoreNormalizer(ADAPTIVE_WINDOW_SEC, EFFECTIVE_FPS)
    z_normalizer_whip = RollingZScoreNormalizer(ADAPTIVE_WINDOW_SEC, EFFECTIVE_FPS)
    z_arm = z_normalizer_arm.normalize_array(smoothed_arm)
    z_whip = z_normalizer_whip.normalize_array(smoothed_whip)

    final_segments = []
    coarse = CoarseCandidateDetector()
    fsm = RallyStateMachine(fps=FPS, max_time=TOTAL_FRAMES / FPS)

    for i, d in enumerate(frame_data):
        d['smoothed_arm'] = smoothed_arm[i]
        d['smoothed_whip'] = smoothed_whip[i]

        arm_adaptive.update(z_arm[i])
        whip_adaptive.update(z_whip[i])

        t_arm = arm_adaptive.get_threshold()
        t_whip = whip_adaptive.get_threshold()

        d['t_arm'] = t_arm
        d['t_whip'] = t_whip
        d['z_arm'] = z_arm[i]
        d['z_whip'] = z_whip[i]

        scale = d.get('player_height', 0)
        valid_distance = (scale >= PLAYER_HEIGHT_MIN) and (scale <= PLAYER_HEIGHT_MAX)

        score, fired = coarse.score(
            valid_pose=d.get('is_valid_pose', False),
            valid_distance=valid_distance,
            z_arm=z_arm[i], t_arm=t_arm,
            z_whip=z_whip[i], t_whip=t_whip,
            hip_vel=hip_vels[i],
            spatial_var=d.get('spatial_variance', 1.0),
            raw_arm=smoothed_arm[i],
            raw_whip=smoothed_whip[i],
        )
        is_candidate = coarse.is_candidate(score)

        d['coarse_score'] = score
        d['candidate'] = int(is_candidate)

        segment = fsm.step(is_candidate, d.get('frame', 0), d.get('timestamp', 0.0), score)
        d['state'] = fsm.state_name
        if segment is not None:
            final_segments.append(segment)

    trailing = fsm.finalize()
    if trailing is not None:
        final_segments.append(trailing)

    if frame_data:
        state.last_segment_end_frame = frame_data[-1].get('frame', 0)

    return final_segments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_frames(count=80, offset=0, burst_period=20, burst_len=2, jitter=0, seed=0):
    """A plausible stroke pattern: quiet baseline with periodic sharp bursts."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(count):
        burst = (i % burst_period) in tuple(range(8, 8 + burst_len))
        noise = lambda: rng.normal(0, jitter) if jitter else 0.0  # noqa: E731
        frame = (i * PROCESS_EVERY_N_FRAMES) + offset
        frames.append({
            "frame": frame,
            "timestamp": frame / FPS,
            "is_valid_pose": True,
            "energy": (0.02 if burst else 0.002) + noise(),
            "arm_energy": (0.12 if burst else 0.004) + noise(),
            "whip_ratio": (3.0 if burst else 0.5) + noise(),
            "hip_vel": (0.02 if burst else 0.002) + noise(),
            "spatial_variance": (0.05 if burst else 0.01) + noise(),
            "player_height": 0.4,
        })
    return frames


def config():
    return ConfigSnapshot.from_module(fps=FPS, max_time=TOTAL_FRAMES / FPS)


# fired_signals is excluded here: the legacy implementation computed the
# evidence list but discarded it, so there is nothing to compare against. It is
# checked separately in TestParityWithLegacyImplementation.
COMPARED_KEYS = ["smoothed_arm", "smoothed_whip", "t_arm", "t_whip", "z_arm", "z_whip",
                 "coarse_score", "candidate", "state"]


def assert_matches_legacy(frames):
    """Run both implementations over the same input and require equality."""
    legacy_frames = [dict(f) for f in frames]
    new_frames = [dict(f) for f in frames]

    legacy_state = _LegacyState()
    legacy_segments = legacy_analyze_segment_data(legacy_frames, legacy_state)

    analyzer = RallyAnalyzer(config())
    new_segments = analyzer.analyze(new_frames)

    assert len(new_segments) == len(legacy_segments), (
        f"segment count differs: {len(new_segments)} vs {len(legacy_segments)}")
    for new, old in zip(new_segments, legacy_segments):
        assert (new.start_frame, new.end_frame) == (old.start_frame, old.end_frame)
        assert new.start_time == pytest.approx(old.start_time)
        assert new.end_time == pytest.approx(old.end_time)
        assert new.score == pytest.approx(old.score)

    assert len(new_frames) == len(legacy_frames)
    for i, (new, old) in enumerate(zip(new_frames, legacy_frames)):
        for key in COMPARED_KEYS:
            new_value, old_value = new[key], old[key]
            if isinstance(old_value, float):
                assert new_value == pytest.approx(old_value), \
                    f"frame {i} key {key}: {new_value!r} vs {old_value!r}"
            else:
                assert new_value == old_value, \
                    f"frame {i} key {key}: {new_value!r} vs {old_value!r}"

    return new_segments, legacy_segments


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

class TestParityWithLegacyImplementation:
    def test_identical_on_a_clean_stroke_pattern(self):
        assert_matches_legacy(make_frames())

    def test_identical_with_noisy_features(self):
        assert_matches_legacy(make_frames(jitter=0.003, seed=7))

    def test_identical_with_a_longer_clip(self):
        assert_matches_legacy(make_frames(count=300, burst_period=37))

    def test_identical_when_nothing_moves(self):
        frames = make_frames(count=40)
        for frame in frames:
            frame.update({"arm_energy": 0.0, "whip_ratio": 0.0, "energy": 0.0,
                          "hip_vel": 0.0, "spatial_variance": 0.0})
        segments, legacy = assert_matches_legacy(frames)
        assert segments == [] and legacy == []

    def test_identical_across_a_rest_gap(self):
        """The rest-gap padding path is the subtlest part of the port."""
        first = make_frames(count=60)
        second = make_frames(count=60, offset=9000)
        legacy_state = _LegacyState()
        analyzer = RallyAnalyzer(config())

        for frames in (first, second):
            legacy = legacy_analyze_segment_data([dict(f) for f in frames], legacy_state)
            new = analyzer.analyze([dict(f) for f in frames])
            assert len(new) == len(legacy)

    def test_identical_for_an_out_of_range_player_scale(self):
        frames = make_frames()
        for frame in frames:
            frame["player_height"] = 99.0
        segments, legacy = assert_matches_legacy(frames)
        assert segments == [] and legacy == []

    def test_fired_signals_explain_the_coarse_score(self):
        """The evidence list must account for the score exactly.

        ``coarse_score`` is produced by CoarseCandidateDetector and
        ``fired_signals`` is the list the refactor now captures from the same
        call, so summing the weights of the fired signals is an independent
        check that the capture is complete and correctly joined.
        """
        frames = make_frames()
        RallyAnalyzer(config()).analyze(frames)

        weights = {"arm": COARSE_W_ARM, "whip": COARSE_W_WHIP,
                   "hip": COARSE_W_HIP, "spatial": COARSE_W_SPATIAL}
        assert any(f["fired_signals"] for f in frames), "fixture should fire evidence"
        for i, frame in enumerate(frames):
            names = [n for n in frame["fired_signals"].split("|") if n]
            assert sum(weights[n] for n in names) == pytest.approx(frame["coarse_score"]), \
                f"frame {i}: {names} does not explain score {frame['coarse_score']}"

    def test_fired_signals_are_empty_below_the_threshold(self):
        frames = make_frames(count=40)
        for frame in frames:
            frame.update({"arm_energy": 0.0, "whip_ratio": 0.0, "hip_vel": 0.0,
                          "spatial_variance": 0.0})
        RallyAnalyzer(config()).analyze(frames)
        assert all(f["fired_signals"] == "" for f in frames)


class TestAdaptiveState:
    def test_from_config_uses_the_snapshot_values(self):
        snapshot = config().apply({"ARM_ENERGY_K_MULTIPLIER": 9.0})
        state = AdaptiveState.from_config(snapshot)
        assert state.arm.k == 9.0
        assert state.arm.min_floor == snapshot.min_energy_floor

    def test_pad_rest_gap_returns_the_pad_count(self):
        state = AdaptiveState.from_config(config())
        state.last_segment_end_frame = 0
        assert state.pad_rest_gap(100, PROCESS_EVERY_N_FRAMES) == 50

    def test_pad_rest_gap_is_zero_without_a_gap(self):
        state = AdaptiveState.from_config(config())
        state.last_segment_end_frame = 100
        assert state.pad_rest_gap(100, PROCESS_EVERY_N_FRAMES) == 0

    def test_pad_rest_gap_is_capped_at_capacity(self):
        state = AdaptiveState.from_config(config())
        state.last_segment_end_frame = 0
        assert state.pad_rest_gap(10 ** 9, 1) == state.arm.capacity


class TestConfigIsHonoured:
    """A tuned snapshot must actually change detection — otherwise the
    ConfigEditor would appear to work while doing nothing."""

    def test_a_high_candidate_threshold_suppresses_candidates(self):
        frames = make_frames()
        analyzer = RallyAnalyzer(config().apply({"COARSE_CANDIDATE_THRESHOLD": 99.0}))
        analyzer.analyze(frames)
        assert all(f["candidate"] == 0 for f in frames)

    def test_a_low_threshold_produces_candidates(self):
        frames = make_frames()
        analyzer = RallyAnalyzer(config().apply({"COARSE_CANDIDATE_THRESHOLD": 0.0}))
        segments = analyzer.analyze(frames)
        assert any(f["candidate"] for f in frames)
        assert segments

    def test_overriding_k_multipliers_changes_the_thresholds(self):
        base = make_frames()
        tuned = make_frames()
        RallyAnalyzer(config()).analyze(base)
        RallyAnalyzer(config().apply({"ARM_ENERGY_K_MULTIPLIER": 20.0})).analyze(tuned)
        assert max(f["t_arm"] for f in tuned) != max(f["t_arm"] for f in base)

    def test_replay_is_deterministic(self):
        first = make_frames()
        second = make_frames()
        a = RallyAnalyzer(config())
        b = RallyAnalyzer(config())
        a.analyze(first)
        b.analyze(second)
        assert [f["coarse_score"] for f in first] == [f["coarse_score"] for f in second]


class TestHelpers:
    def test_smooth_preserves_length(self):
        assert len(smooth(np.arange(10, dtype=float), 15.0)) == 10

    def test_smooth_window_is_never_below_one(self):
        assert len(smooth(np.ones(5), 0.1)) == 5

    def test_mitigate_outliers_clips_extremes(self):
        arr = np.array([0.0, 0.1, 5.0])
        assert mitigate_outliers(arr, absolute_max=1.0).max() <= 1.0

    def test_mitigate_outliers_handles_empty(self):
        assert len(mitigate_outliers(np.array([]), 1.0)) == 0

    def test_mask_to_indices_finds_end_exclusive_runs(self):
        assert mask_to_indices(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]

    def test_mask_to_indices_returns_plain_ints(self):
        """Callers index lists with these; numpy scalars leak into CSV output."""
        assert all(type(v) is int for pair in mask_to_indices(np.array([1, 1])) for v in pair)

    def test_mask_to_indices_empty(self):
        assert mask_to_indices(np.array([0, 0, 0])) == []


class TestServeAnalyzer:
    def test_runs_over_feature_dicts(self):
        frames = make_frames(count=40)
        for frame in frames:
            frame.update({"nose_y": 0.1, "l_shoulder_y": 0.2, "r_shoulder_y": 0.2,
                          "l_wrist_y": 0.3, "r_wrist_y": 0.3, "l_wrist_vel": 0.0,
                          "r_wrist_vel": 0.0, "l_elbow_angle": 180.0, "r_elbow_angle": 180.0})
        segments, events = ServeAnalyzer(config()).analyze(frames)
        assert isinstance(segments, list) and isinstance(events, list)

    def test_populates_the_shared_telemetry_columns(self):
        frames = make_frames(count=20)
        ServeAnalyzer(config()).analyze(frames)
        for frame in frames:
            assert "smoothed_arm" in frame and "state" in frame
            assert frame["candidate"] in (0, 1)

    def test_empty_input_is_safe(self):
        assert ServeAnalyzer(config()).analyze([]) == ([], [])


class TestFeedAnalyzer:
    def test_runs_over_feature_dicts(self):
        frames = make_frames(count=60)
        assert isinstance(FeedAnalyzer(config(), total_frames=TOTAL_FRAMES).analyze(frames), list)

    def test_scale_bounds_are_explicit_inputs(self):
        frames = make_frames(count=60)
        analyzer = FeedAnalyzer(config(), min_player_height=0.0, max_player_height=0.05,
                                total_frames=TOTAL_FRAMES)
        assert analyzer.analyze(frames) == []

    def test_empty_input_is_safe(self):
        assert FeedAnalyzer(config()).analyze([]) == []
