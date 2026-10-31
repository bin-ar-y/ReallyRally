"""Tests for the current-frame inspector.

The inspector's whole job is to make a number interpretable, so the properties
worth locking are: raw / threshold / score / decision / state stay labelled
distinctly, the trigger verdict matches what the detector actually did, and a
new telemetry column can never make it crash.
"""

from really_rally.core.config import PLAYER_HEIGHT_MAX
from really_rally.studio.frame_inspector import build_inspector
from really_rally.utils.feature_store import (
    ROLE_DECISION,
    ROLE_RAW,
    ROLE_SCORE,
    ROLE_STATE,
    ROLE_THRESHOLD,
)

FRAME = {
    "frame": 12100, "time": 403.42, "segment": 3, "state": "candidate",
    "arm_energy": 0.023, "smoothed_arm": 0.028, "t_arm": 0.249, "z_arm": 1.10,
    "whip_ratio": 1.67, "smoothed_whip": 1.5, "t_whip": 0.25, "z_whip": 2.00,
    "energy": 0.02, "hip_vel": 0.004, "spatial_variance": 0.02,
    "action_score": 0.001, "player_height": 0.441,
    "is_valid_pose": 1, "candidate": 1, "coarse_score": 2.5,
    "fired_signals": "whip|hip", "nose_y": 0.1, "l_wrist_y": 0.3,
}


def find(sections, title):
    for section in sections:
        if section.title == title:
            return section
    return None


def labels(sections, title):
    section = find(sections, title)
    return {} if section is None else {r.label: r for r in section.rows}


class TestStructure:
    def test_no_record_reports_a_status_instead_of_crashing(self):
        sections = build_inspector(None)
        assert sections[0].rows[0].value == "no telemetry loaded"

    def test_empty_record_does_not_crash(self):
        assert build_inspector({})

    def test_context_section_has_frame_time_and_segment(self):
        section = find(build_inspector(FRAME), "Frame")
        assert {r.label for r in section.rows} >= {"Frame", "Time", "Segment"}

    def test_time_is_formatted_with_units(self):
        assert find(build_inspector(FRAME), "Frame").rows[1].value == "403.42s"

    def test_signal_section_is_named_from_the_threshold_key(self):
        assert find(build_inspector(FRAME), "Arm Energy") is not None
        assert find(build_inspector(FRAME), "Whip Ratio") is not None


class TestValueInterpretation:
    def test_raw_threshold_and_score_are_distinguishable(self):
        rows = labels(build_inspector(FRAME), "Arm Energy")
        assert rows["Raw"].role == ROLE_RAW
        assert rows["Threshold"].role == ROLE_THRESHOLD
        assert rows["Z-score"].role == ROLE_SCORE

    def test_raw_and_smoothed_are_both_shown(self):
        rows = labels(build_inspector(FRAME), "Arm Energy")
        assert rows["Raw"].value == "0.0230"
        assert rows["Smoothed"].value == "0.0280"

    def test_trigger_true_when_z_exceeds_threshold(self):
        rows = labels(build_inspector(FRAME), "Arm Energy")
        assert rows["Trigger"].value == "TRUE"          # 1.10 > 0.249
        assert rows["Trigger"].role == ROLE_DECISION

    def test_trigger_false_when_z_is_below_threshold(self):
        frame = dict(FRAME, z_arm=0.10)
        assert labels(build_inspector(frame), "Arm Energy")["Trigger"].value == "FALSE"

    def test_decision_section_reports_candidate_and_gate(self):
        rows = labels(build_inspector(FRAME), "Decision")
        assert rows["Candidate"].value == "TRUE"
        assert rows["Pose valid"].value == "TRUE"
        assert rows["Distance gate"].value.startswith("TRUE")

    def test_distance_gate_flags_an_out_of_range_scale(self):
        frame = dict(FRAME, player_height=PLAYER_HEIGHT_MAX + 0.5)
        row = labels(build_inspector(frame), "Decision")["Distance gate"]
        assert row.value.startswith("FALSE")
        assert row.emphasis is True

    def test_fired_evidence_is_shown_verbatim(self):
        rows = labels(build_inspector(FRAME), "Decision")
        assert rows["Fired evidence"].value == "whip|hip"

    def test_no_evidence_reads_none_not_empty(self):
        frame = dict(FRAME, fired_signals="")
        assert labels(build_inspector(frame), "Decision")["Fired evidence"].value == "none"

    def test_state_section_reports_the_fsm_state(self):
        rows = labels(build_inspector(FRAME), "State")
        assert rows["Detector state"].value == "candidate"
        assert rows["Detector state"].role == ROLE_STATE

    def test_coarse_score_is_under_derived_score(self):
        assert labels(build_inspector(FRAME), "Derived score")["Coarse score"].value == "2.5000"

    def test_pose_geometry_is_present_and_generic(self):
        rows = labels(build_inspector(FRAME), "Pose geometry")
        assert "nose_y" in rows and "l_wrist_y" in rows


class TestExtensibility:
    def test_unknown_numeric_column_is_displayed_not_dropped(self):
        """Future racket/ball telemetry must appear without a code change."""
        frame = dict(FRAME, racket_confidence=0.87)
        all_rows = {r.label for s in build_inspector(frame) for r in s.rows}
        assert "racket_confidence" in all_rows

    def test_unknown_threshold_key_gets_its_own_section(self):
        """A new t_<name> column is grouped generically."""
        frame = dict(FRAME, t_racket=0.5, z_racket=0.9)
        rows = labels(build_inspector(frame), "Racket")
        assert rows["Threshold"].value == "0.5000"
        assert rows["Trigger"].value == "TRUE"

    def test_text_columns_do_not_leak_into_pose_geometry(self):
        frame = dict(FRAME, state="active", fired_signals="arm")
        rows = labels(build_inspector(frame), "Pose geometry")
        assert "state" not in rows and "fired_signals" not in rows

    def test_missing_optional_sections_are_simply_absent(self):
        sections = build_inspector({"frame": 1, "time": 0.0})
        assert find(sections, "Decision") is None
        assert find(sections, "State") is None
