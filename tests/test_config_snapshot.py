"""Tests for the detector configuration snapshot.

The snapshot exists so the live detection path and the replay path read the
same parameter values. The classification it carries gates a user-facing
promise: a "replayable" parameter really can be re-tuned from cached features,
and a "requires re-analysis" one really cannot.
"""

import json

import pytest

import really_rally.core.config as cfg
from really_rally.core.config_snapshot import (
    CONTEXT_FIELDS,
    REANALYSIS_KEYS,
    REPLAYABLE_KEYS,
    ConfigSnapshot,
)


class TestFromModule:
    def test_reads_current_config_values(self):
        snap = ConfigSnapshot.from_module(fps=60.0, max_time=120.0)
        assert snap.arm_energy_k_multiplier == cfg.ARM_ENERGY_K_MULTIPLIER
        assert snap.whip_ratio_k_multiplier == cfg.WHIP_RATIO_K_MULTIPLIER
        assert snap.adaptive_window_sec == cfg.ADAPTIVE_WINDOW_SEC
        assert snap.rally_start_gap_sec == cfg.RALLY_START_GAP_SEC
        assert snap.coarse_candidate_threshold == cfg.COARSE_CANDIDATE_THRESHOLD

    def test_carries_video_context(self):
        snap = ConfigSnapshot.from_module(fps=60.0, max_time=120.0)
        assert snap.fps == 60.0
        assert snap.max_time == 120.0
        assert snap.process_every_n == cfg.PROCESS_EVERY_N_FRAMES

    def test_effective_fps_divides_by_sampling_stride(self):
        assert ConfigSnapshot.from_module(fps=60.0, max_time=1.0).effective_fps == 30.0

    def test_every_replayable_key_is_a_real_config_constant(self):
        """A typo here would silently snapshot a stale default."""
        for key in REPLAYABLE_KEYS:
            assert hasattr(cfg, key), f"{key} is not defined in config.py"


class TestClassification:
    def test_tiers_are_disjoint(self):
        assert not (REPLAYABLE_KEYS & REANALYSIS_KEYS)

    def test_classify_lists_both_tiers(self):
        tiers = ConfigSnapshot.from_module(fps=30.0, max_time=1.0).classify()
        assert "arm_energy_k_multiplier" not in tiers["replayable"]  # upper-case names
        assert "ARM_ENERGY_K_MULTIPLIER" in tiers["replayable"]
        assert "MP_MODEL_COMPLEXITY" in tiers["requires_reanalysis"]
        assert "PROCESS_EVERY_N_FRAMES" in tiers["requires_reanalysis"]

    def test_context_fields_are_not_exposed_as_tunable(self):
        for name in CONTEXT_FIELDS:
            assert name.upper() not in REPLAYABLE_KEYS

    def test_pose_and_motion_params_require_reanalysis(self):
        # The two that motivated the whole distinction.
        assert "MP_MODEL_COMPLEXITY" in REANALYSIS_KEYS
        assert "PROCESS_EVERY_N_FRAMES" in REANALYSIS_KEYS

    def test_threshold_params_are_replayable(self):
        for key in ("ARM_ENERGY_K_MULTIPLIER", "WHIP_RATIO_K_MULTIPLIER",
                    "RALLY_START_GAP_SEC", "RALLY_CONTINUE_GAP_SEC",
                    "COARSE_CANDIDATE_THRESHOLD", "ADAPTIVE_WINDOW_SEC"):
            assert key in REPLAYABLE_KEYS, f"{key} should be replayable"


class TestApply:
    def test_applies_a_replayable_override(self):
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        tuned = snap.apply({"ARM_ENERGY_K_MULTIPLIER": 4.0})
        assert tuned.arm_energy_k_multiplier == 4.0
        assert tuned.whip_ratio_k_multiplier == snap.whip_ratio_k_multiplier

    def test_accepts_lower_case_keys(self):
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        assert snap.apply({"arm_energy_k_multiplier": 3.5}).arm_energy_k_multiplier == 3.5

    def test_does_not_mutate_the_original(self):
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        snap.apply({"ARM_ENERGY_K_MULTIPLIER": 99.0})
        assert snap.arm_energy_k_multiplier == cfg.ARM_ENERGY_K_MULTIPLIER

    def test_rejects_a_parameter_requiring_reanalysis(self):
        """Silently accepting this would report a replay that never happened."""
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        with pytest.raises(ValueError, match="re-analysis"):
            snap.apply({"MP_MODEL_COMPLEXITY": 2})

    def test_rejects_an_unknown_parameter(self):
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        with pytest.raises(ValueError, match="not a known detector parameter"):
            snap.apply({"NOT_A_REAL_SETTING": 1})

    def test_multiple_overrides_at_once(self):
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        tuned = snap.apply({
            "ARM_ENERGY_K_MULTIPLIER": 3.0,
            "WHIP_RATIO_K_MULTIPLIER": 3.5,
            "RALLY_START_GAP_SEC": 8.0,
        })
        assert (tuned.arm_energy_k_multiplier, tuned.whip_ratio_k_multiplier,
                tuned.rally_start_gap_sec) == (3.0, 3.5, 8.0)


class TestSerialization:
    def test_to_dict_includes_context_and_replayable_keys(self):
        d = ConfigSnapshot.from_module(fps=30.0, max_time=1.0).to_dict()
        assert d["fps"] == 30.0
        assert d["arm_energy_k_multiplier"] == cfg.ARM_ENERGY_K_MULTIPLIER
        assert "ARM_ENERGY_K_MULTIPLIER" in d["replayable_keys"]

    def test_to_json_round_trips(self):
        snap = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        restored = json.loads(snap.to_json())
        assert restored["max_time"] == 1.0
        assert restored["adaptive_window_sec"] == snap.adaptive_window_sec

    def test_to_json_is_deterministic(self):
        a = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        b = ConfigSnapshot.from_module(fps=30.0, max_time=1.0)
        assert a.to_json() == b.to_json()
