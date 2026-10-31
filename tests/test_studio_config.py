"""Tests for the configuration editor's pure logic.

The panel is a thin wrapper; what needs guarding is the classification it
displays and the parsing it performs. Two failure modes matter: an edit that is
silently dropped, and a re-analysis parameter that appears applyable when it is
not — both produce a confident result that is wrong.
"""

import pytest

from really_rally.core.config import ARM_ENERGY_K_MULTIPLIER
from really_rally.core.config_snapshot import REANALYSIS_KEYS, ConfigSnapshot
from really_rally.studio.config_editor import (
    REANALYSIS_LABELS,
    REPLAYABLE_PARAMETERS,
    parse_override_values,
)


@pytest.fixture
def baseline():
    return ConfigSnapshot.from_module(fps=30.0, max_time=10.0)


class TestParameterSurface:
    def test_every_advertised_parameter_is_replayable(self, baseline):
        """A parameter offered in the panel must be applicable."""
        for spec in REPLAYABLE_PARAMETERS:
            baseline.apply({spec.name: 1.0})

    def test_every_advertised_parameter_exists(self, baseline):
        for spec in REPLAYABLE_PARAMETERS:
            assert hasattr(baseline, spec.name.lower()), spec.name

    def test_no_duplicate_parameters(self):
        names = [spec.name for spec in REPLAYABLE_PARAMETERS]
        assert len(names) == len(set(names))

    def test_reanalysis_labels_are_real_constants(self):
        assert set(REANALYSIS_LABELS) <= REANALYSIS_KEYS

    def test_parameters_have_human_labels(self):
        for spec in REPLAYABLE_PARAMETERS:
            assert spec.label and spec.label != spec.name


class TestParsing:
    def test_unchanged_values_produce_no_overrides(self, baseline):
        values = {spec.name: f"{getattr(baseline, spec.name.lower()):g}"
                  for spec in REPLAYABLE_PARAMETERS}
        overrides, errors = parse_override_values(values, baseline)
        assert overrides == {} and errors == []

    def test_changed_value_becomes_an_override(self, baseline):
        values = {"ARM_ENERGY_K_MULTIPLIER": "4.0"}
        overrides, errors = parse_override_values(values, baseline)
        assert overrides == {"ARM_ENERGY_K_MULTIPLIER": 4.0}
        assert errors == []

    def test_blank_fields_are_ignored(self, baseline):
        overrides, errors = parse_override_values(
            {"ARM_ENERGY_K_MULTIPLIER": "", "WHIP_RATIO_K_MULTIPLIER": "  "}, baseline)
        assert overrides == {} and errors == []

    def test_non_numeric_is_reported_not_guessed(self, baseline):
        overrides, errors = parse_override_values(
            {"ARM_ENERGY_K_MULTIPLIER": "fast"}, baseline)
        assert overrides == {}
        assert len(errors) == 1 and "not a number" in errors[0]

    def test_one_typo_does_not_discard_the_other_edits(self, baseline):
        """The good edit must still apply; the bad one is reported."""
        overrides, errors = parse_override_values(
            {"ARM_ENERGY_K_MULTIPLIER": "4.0", "WHIP_RATIO_K_MULTIPLIER": "oops"}, baseline)
        assert overrides == {"ARM_ENERGY_K_MULTIPLIER": 4.0}
        assert len(errors) == 1

    def test_unknown_parameter_is_reported(self, baseline):
        overrides, errors = parse_override_values({"NOT_A_SETTING": "1"}, baseline)
        assert overrides == {}
        assert "unknown parameter" in errors[0]

    def test_integer_text_is_accepted(self, baseline):
        overrides, _ = parse_override_values({"ARM_ENERGY_K_MULTIPLIER": "3"}, baseline)
        assert overrides == {"ARM_ENERGY_K_MULTIPLIER": 3.0}

    def test_negative_and_zero_values_are_accepted(self, baseline):
        overrides, errors = parse_override_values(
            {"ARM_ENERGY_K_MULTIPLIER": "0", "RALLY_START_GAP_SEC": "-1"}, baseline)
        assert overrides == {"ARM_ENERGY_K_MULTIPLIER": 0.0, "RALLY_START_GAP_SEC": -1.0}
        assert errors == []

    def test_equivalent_numeric_text_is_not_a_change(self, baseline):
        """'2.50' and '2.5' are the same value; neither should count as an edit."""
        overrides, _ = parse_override_values(
            {"ARM_ENERGY_K_MULTIPLIER": f"{ARM_ENERGY_K_MULTIPLIER:.4f}"}, baseline)
        assert overrides == {}

    def test_multiple_edits_apply_together(self, baseline):
        overrides, errors = parse_override_values({
            "ARM_ENERGY_K_MULTIPLIER": "3.0",
            "RALLY_START_GAP_SEC": "8.0",
        }, baseline)
        assert errors == []
        assert overrides == {"ARM_ENERGY_K_MULTIPLIER": 3.0, "RALLY_START_GAP_SEC": 8.0}

    def test_overrides_apply_cleanly_to_the_snapshot(self, baseline):
        overrides, errors = parse_override_values(
            {"ARM_ENERGY_K_MULTIPLIER": "9.0", "RALLY_START_GAP_SEC": "1.0"}, baseline)
        assert errors == []
        tuned = baseline.apply(overrides)
        assert tuned.arm_energy_k_multiplier == 9.0
        assert tuned.rally_start_gap_sec == 1.0
