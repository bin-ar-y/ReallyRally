"""Tests for scoring and the disagreement navigator.

The navigator is the workflow the studio exists to enable: step through the
actual misses and false positives instead of guessing from a number. Its
correctness (never skipping a disagreement, never reporting one that is not
there) is what makes that workflow trustworthy.
"""

import os

import pytest

from really_rally.studio.evaluation import (
    Disagreement,
    EvaluationEngine,
    format_metrics,
    verdict,
)
from really_rally.studio.experiments import (
    ExperimentRepository,
    compare,
    summarize_delta,
)
from really_rally.studio.predictions import (
    PredictionRepository,
    predictions_from_segments,
)
from really_rally.core.models import RallySegment
from really_rally.core.config_snapshot import ConfigSnapshot

GT = [(10.0, 20.0), (40.0, 55.0), (70.0, 80.0)]


class TestScoring:
    def test_perfect_match(self):
        metrics = EvaluationEngine().evaluate(GT, list(GT))
        assert metrics["f1"] == pytest.approx(1.0)
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (3, 0, 0)

    def test_all_missed(self):
        metrics = EvaluationEngine().evaluate(GT, [])
        assert metrics["recall"] == 0.0
        assert metrics["fn"] == 3

    def test_all_false(self):
        metrics = EvaluationEngine().evaluate([], [(1.0, 2.0)])
        assert metrics["precision"] == 0.0
        assert metrics["fp"] == 1

    def test_delegates_to_the_shared_scorer(self):
        """The studio and the CLI must agree on the same inputs."""
        from really_rally.utils.evaluate import compute_metrics

        expected = compute_metrics(GT, [(10.0, 20.0)], 0.3)
        assert EvaluationEngine().evaluate(GT, [(10.0, 20.0)]) == expected

    def test_iou_threshold_is_configurable(self):
        loose = EvaluationEngine(iou_threshold=0.1).evaluate([(0.0, 10.0)], [(2.0, 12.0)])
        strict = EvaluationEngine(iou_threshold=0.9).evaluate([(0.0, 10.0)], [(2.0, 12.0)])
        assert loose["tp"] == 1 and strict["tp"] == 0


class TestDisagreements:
    def test_reports_a_missed_interval(self):
        found = EvaluationEngine().disagreements(GT, [(10.0, 20.0)])
        assert len(found) == 2
        assert all(d.kind == "missed" for d in found)
        assert [d.start for d in found] == [40.0, 70.0]

    def test_reports_a_false_positive(self):
        found = EvaluationEngine().disagreements([], [(5.0, 9.0)])
        assert len(found) == 1
        assert found[0].kind == "false"
        assert found[0].start == 5.0

    def test_none_when_everything_matches(self):
        assert EvaluationEngine().disagreements(GT, list(GT)) == []

    def test_ordered_along_the_timeline(self):
        pred = [(2.0, 3.0), (10.0, 20.0)]
        found = EvaluationEngine().disagreements(GT, pred)
        assert [d.start for d in found] == sorted(d.start for d in found)

    def test_details_explain_the_disagreement(self):
        found = EvaluationEngine().disagreements(GT, [])
        assert "nothing was detected" in found[0].detail

    def test_duration_of_a_disagreement(self):
        assert Disagreement("missed", 10.0, 15.0).duration == pytest.approx(5.0)


class TestNavigator:
    def test_next_skips_to_the_following_disagreement(self):
        engine = EvaluationEngine()
        found = engine.next_disagreement(GT, [], after_time=15.0)
        assert found.start == 40.0

    def test_next_from_the_start_gives_the_first(self):
        assert EvaluationEngine().next_disagreement(GT, [], 0.0).start == 10.0

    def test_next_past_the_end_is_none(self):
        assert EvaluationEngine().next_disagreement(GT, [], 1000.0) is None

    def test_previous_finds_the_earlier_one(self):
        assert EvaluationEngine().previous_disagreement(GT, [], 60.0).start == 40.0

    def test_previous_before_the_start_is_none(self):
        assert EvaluationEngine().previous_disagreement(GT, [], 0.0) is None

    def test_kind_filter_selects_only_misses(self):
        pred = [(5.0, 9.0)]
        found = EvaluationEngine().next_disagreement(GT, pred, 0.0, kind="missed")
        assert found.kind == "missed"

    def test_kind_filter_selects_only_false_positives(self):
        pred = [(5.0, 9.0)]
        found = EvaluationEngine().next_disagreement(GT, pred, 0.0, kind="false")
        assert found.kind == "false" and found.start == 5.0

    def test_kind_filter_returns_none_when_absent(self):
        assert EvaluationEngine().next_disagreement(GT, list(GT), 0.0, kind="false") is None

    def test_navigation_is_repeatable(self):
        """Stepping forward twice must advance, not stick."""
        engine = EvaluationEngine()
        first = engine.next_disagreement(GT, [], 0.0)
        second = engine.next_disagreement(GT, [], first.start)
        assert second.start > first.start


class TestReporting:
    def test_metrics_line_names_the_threshold(self):
        metrics = EvaluationEngine().evaluate(GT, list(GT))
        assert "@IoU 0.30" in format_metrics(metrics, 0.3)

    def test_no_data_is_stated_plainly(self):
        metrics = EvaluationEngine().evaluate([], [])
        assert "nothing to score" in format_metrics(metrics, 0.3)

    def test_all_missed_is_called_out(self):
        text = format_metrics(EvaluationEngine().evaluate(GT, []), 0.3)
        assert "missed" in text

    def test_verdict_flags_an_unusable_detector(self):
        assert verdict(EvaluationEngine().evaluate(GT, [])) == "below usable threshold"

    def test_verdict_accepts_a_good_detector(self):
        assert verdict(EvaluationEngine().evaluate(GT, list(GT))) == "usable"

    def test_verdict_without_labels(self):
        assert verdict(EvaluationEngine().evaluate([], [])) == "no ground truth"


class TestPredictions:
    def make_segments(self):
        return [
            RallySegment(start_frame=300, end_frame=600, start_time=10.0,
                         end_time=20.0, score=4.0),
            RallySegment(start_frame=1200, end_frame=1650, start_time=40.0,
                         end_time=55.0, score=6.0),
        ]

    def test_converts_segments_to_predictions(self):
        found = predictions_from_segments(self.make_segments())
        assert [p.span() for p in found] == [(10.0, 20.0), (40.0, 55.0)]

    def test_spans_feed_the_scorer(self):
        repo = PredictionRepository(predictions_from_segments(self.make_segments()))
        metrics = EvaluationEngine().evaluate(GT, repo.spans())
        assert metrics["tp"] == 2 and metrics["fn"] == 1

    def test_reason_names_the_evidence_that_fired(self):
        records = [
            {"frame": 300, "candidate": 1, "fired_signals": "arm|whip",
             "state": "candidate", "coarse_score": 4.0},
            {"frame": 302, "candidate": 1, "fired_signals": "arm",
             "state": "active", "coarse_score": 4.0},
        ]
        found = predictions_from_segments(self.make_segments()[:1], "rally", records)
        assert "arm" in found[0].reason and "whip" in found[0].reason

    def test_reason_is_honest_when_nothing_fired(self):
        found = predictions_from_segments(self.make_segments()[:1], "rally", [])
        assert found[0].reason

    def test_state_is_taken_from_the_last_candidate_frame(self):
        records = [
            {"frame": 300, "candidate": 1, "state": "candidate", "fired_signals": "arm"},
            {"frame": 600, "candidate": 1, "state": "active", "fired_signals": "arm"},
        ]
        found = predictions_from_segments(self.make_segments()[:1], "rally", records)
        assert found[0].state == "active"

    def test_round_trip_through_csv(self, temp_dir):
        path = os.path.join(temp_dir, "v_predictions.csv")
        repo = PredictionRepository(predictions_from_segments(self.make_segments()), path)
        repo.save()

        restored = PredictionRepository.from_csv(path)
        assert len(restored.predictions) == 2
        assert restored.predictions[0].span() == (10.0, 20.0)
        assert restored.predictions[0].detector == "rally"

    def test_details_survive_the_round_trip(self, temp_dir):
        path = os.path.join(temp_dir, "v_predictions.csv")
        PredictionRepository(predictions_from_segments(self.make_segments()), path).save()
        restored = PredictionRepository.from_csv(path)
        assert restored.predictions[0].details["start_frame"] == 300

    def test_header_is_stable(self, temp_dir):
        import csv
        from really_rally.studio.predictions import PREDICTION_COLUMNS

        path = os.path.join(temp_dir, "v_predictions.csv")
        PredictionRepository([], path).save()
        with open(path) as f:
            assert next(csv.reader(f)) == PREDICTION_COLUMNS

    def test_empty_predictions_score_as_all_missed(self):
        assert EvaluationEngine().evaluate(GT, PredictionRepository().spans())["fn"] == 3


class TestExperimentComparison:
    def make_experiment(self, f1, tp, fp, fn, iou=0.5):
        from really_rally.studio.experiments import Experiment

        return Experiment(metrics={"f1": f1, "precision": 0.6, "recall": 0.7,
                                   "tp": tp, "fp": fp, "fn": fn, "mean_iou": iou,
                                   "boundary_start_error_sec": 1.0,
                                   "boundary_end_error_sec": 1.0})

    def test_records_an_experiment_with_its_config(self, temp_dir):
        path = os.path.join(temp_dir, "v_experiments.csv")
        repo = ExperimentRepository(path=path)
        config = ConfigSnapshot.from_module(fps=30.0, max_time=10.0)
        repo.record(config, {"f1": 0.8, "tp": 2, "fp": 0, "fn": 1})
        assert len(repo.experiments) == 1
        assert repo.experiments[0].config_json

    def test_history_persists(self, temp_dir):
        path = os.path.join(temp_dir, "v_experiments.csv")
        config = ConfigSnapshot.from_module(fps=30.0, max_time=10.0)
        ExperimentRepository(path=path).record(config, {"f1": 0.8, "tp": 2})
        restored = ExperimentRepository.from_csv(path)
        assert restored.experiments[0].f1 == pytest.approx(0.8)

    def test_config_json_survives_the_round_trip(self, temp_dir):
        path = os.path.join(temp_dir, "v_experiments.csv")
        config = ConfigSnapshot.from_module(fps=30.0, max_time=10.0)
        ExperimentRepository(path=path).record(config, {"f1": 0.5})
        restored = ExperimentRepository.from_csv(path)
        assert restored.experiments[0].config_json == config.to_json()

    def test_comparison_directions(self):
        rows = dict((label, direction) for label, _, _, direction in
                    compare(self.make_experiment(0.5, 2, 1, 1),
                            self.make_experiment(0.8, 3, 0, 0)))
        assert rows["F1"] == "better"
        assert rows["FP"] == "better"      # fewer false positives
        assert rows["FN"] == "better"

    def test_comparison_flags_a_regression(self):
        rows = dict((label, direction) for label, _, _, direction in
                    compare(self.make_experiment(0.8, 3, 0, 0),
                            self.make_experiment(0.5, 2, 4, 3)))
        assert rows["F1"] == "worse"
        assert rows["FP"] == "worse"

    def test_comparison_detects_no_change(self):
        rows = compare(self.make_experiment(0.5, 2, 1, 1),
                       self.make_experiment(0.5, 2, 1, 1))
        assert all(direction == "same" for _, _, _, direction in rows)

    def test_summary_states_the_direction_of_change(self):
        text = summarize_delta(self.make_experiment(0.5, 2, 1, 1),
                               self.make_experiment(0.8, 3, 0, 0))
        assert "+0.300" in text and "matched" in text

    def test_summary_reports_lost_matches(self):
        text = summarize_delta(self.make_experiment(0.8, 3, 0, 0),
                               self.make_experiment(0.5, 1, 0, 2))
        assert "missed" in text
