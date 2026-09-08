"""Validation tests for the configuration module."""


# Import via the package to ensure no import errors after the split
from really_rally.core import config


class TestConfigValues:
    def test_all_thresholds_positive(self):
        """All time and threshold constants must be positive."""
        for name in dir(config):
            if name.isupper() and not name.startswith("_"):
                val = getattr(config, name)
                if isinstance(val, (int, float)):
                    assert val > 0, f"{name} = {val} is not positive"

    def test_min_rally_shorter_than_max_gap(self):
        """MIN_RALLY_DURATION_SEC should be <= GAP_MERGE_THRESHOLD_SEC."""
        assert config.MIN_RALLY_DURATION_SEC <= config.GAP_MERGE_THRESHOLD_SEC

    def test_boundary_padding_reasonable(self):
        """Padding should be less than min rally duration."""
        assert config.BOUNDARY_PADDING_SEC < config.MIN_RALLY_DURATION_SEC

    def test_mp_model_complexity_valid(self):
        """MediaPipe model complexity must be 0, 1, or 2."""
        assert config.MP_MODEL_COMPLEXITY in (0, 1, 2)

    def test_mp_confidence_in_range(self):
        """Detection/tracking confidence must be in (0, 1]."""
        assert 0 < config.MP_MIN_DETECTION_CONF <= 1.0
        assert 0 < config.MP_MIN_TRACKING_CONF <= 1.0

    def test_process_every_n_frames_positive_int(self):
        assert isinstance(config.PROCESS_EVERY_N_FRAMES, int)
        assert config.PROCESS_EVERY_N_FRAMES >= 1

    def test_k_multipliers_positive(self):
        assert config.ARM_ENERGY_K_MULTIPLIER > 0
        assert config.WHIP_RATIO_K_MULTIPLIER > 0

    def test_floor_values_positive(self):
        assert config.MIN_ENERGY_FLOOR > 0
        assert config.MIN_WHIP_FLOOR > 0

    def test_pixel_thresholds_reasonable(self):
        """Pixel threshold values should be in valid ranges."""
        assert 0 < config.PIXEL_DIFF_THRESHOLD < 255
        assert 0.0 < config.PIXEL_CHANGE_THRESHOLD < 1.0

    def test_motion_scan_skip_frames_positive(self):
        assert config.MOTION_SCAN_SKIP_FRAMES >= 1


class TestConfigImports:
    """Verify that the config module is importable and has expected attributes."""

    def test_required_keys_exist(self):
        required = [
            "MIN_RALLY_DURATION_SEC",
            "GAP_MERGE_THRESHOLD_SEC",
            "ADAPTIVE_WINDOW_SEC",
            "ARM_ENERGY_K_MULTIPLIER",
            "WHIP_RATIO_K_MULTIPLIER",
            "RALLY_MAX_STROKE_GAP_SEC",
            "MIN_ENERGY_FLOOR",
            "MIN_WHIP_FLOOR",
            "MIN_COURT_COVERAGE",
            "MOVEMENT_WINDOW_SEC",
        ]
        for key in required:
            assert hasattr(config, key), f"Missing config key: {key}"
