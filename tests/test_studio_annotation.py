"""Tests for the studio annotation repository.

The load-bearing property is separation: human ground truth must survive
detector re-tuning untouched, and must stay readable by the existing benchmark
tooling. Everything else here guards against silently corrupting labels, which
would invalidate every measurement taken against them.
"""

import csv
import json
import os

import pytest

from really_rally.studio.annotation import (
    INTERVAL_VOCABULARY,
    KIND_INTERVAL,
    KIND_POINT,
    POINT_VOCABULARY,
    SOURCE_IMPORTED,
    AnnotationRepository,
    Label,
    color_for,
)


@pytest.fixture
def repo():
    return AnnotationRepository()


class TestCreate:
    def test_add_interval(self, repo):
        entry = repo.add_interval(12.5, 25.0, "Rally", notes="crosscourt")
        assert entry.kind == KIND_INTERVAL
        assert entry.duration == pytest.approx(12.5)
        assert repo.count() == {"intervals": 1, "points": 0, "total": 1}

    def test_add_point(self, repo):
        entry = repo.add_point(403.42, "Real Stroke")
        assert entry.kind == KIND_POINT
        assert entry.time == pytest.approx(403.42)
        assert entry.duration == 0.0

    def test_interval_arguments_may_be_reversed(self, repo):
        entry = repo.add_interval(25.0, 12.5, "Rally")
        assert (entry.start, entry.end) == (12.5, 25.0)

    def test_ids_are_unique(self, repo):
        ids = {repo.add_point(i, "Real Stroke").label_id for i in range(20)}
        assert len(ids) == 20

    def test_created_at_is_recorded(self, repo):
        assert repo.add_point(1.0, "Serve").created_at

    def test_suggested_vocabulary_is_available(self, repo):
        for name in INTERVAL_VOCABULARY:
            repo.add_interval(0.0, 1.0, name)
        for name in POINT_VOCABULARY:
            repo.add_point(0.0, name)
        assert repo.count()["total"] == len(INTERVAL_VOCABULARY) + len(POINT_VOCABULARY)

    def test_arbitrary_labels_are_allowed(self, repo):
        """Future classes (forehand/backhand) must not need a schema change."""
        assert repo.add_point(1.0, "Backhand Volley").label == "Backhand Volley"


class TestEdit:
    def test_update_changes_a_field(self, repo):
        entry = repo.add_point(1.0, "Real Stroke")
        assert repo.update(entry.label_id, label="Shadow Swing") is True
        assert repo.labels[0].label == "Shadow Swing"

    def test_update_unknown_id_returns_false(self, repo):
        assert repo.update("nope", label="x") is False

    def test_remove(self, repo):
        entry = repo.add_point(1.0, "Serve")
        assert repo.remove(entry.label_id) is True
        assert repo.count()["total"] == 0
        assert repo.remove(entry.label_id) is False

    def test_remove_at_clears_labels_under_the_playhead(self, repo):
        repo.add_point(10.0, "Serve")
        repo.add_point(50.0, "Serve")
        assert repo.remove_at(10.1, tolerance=0.25) == 1
        assert repo.count()["points"] == 1

    def test_remove_at_does_not_touch_distant_labels(self, repo):
        repo.add_point(10.0, "Serve")
        assert repo.remove_at(40.0, tolerance=0.25) == 0

    def test_remove_at_can_delete_an_interval_spanning_the_point(self, repo):
        repo.add_interval(5.0, 20.0, "Rally")
        assert repo.remove_at(12.0) == 1
        assert repo.count()["intervals"] == 0

    def test_remove_at_deletes_only_one_label_per_press(self, repo):
        """Two labels at one instant must not both vanish on one keystroke."""
        repo.add_point(10.0, "Real Stroke")
        repo.add_point(10.1, "Serve")
        assert repo.remove_at(10.05, tolerance=0.25) == 1
        assert repo.count()["points"] == 1

    def test_remove_at_prefers_a_point_over_a_containing_interval(self, repo):
        """The interval boundary case: deleting the rally would be data loss."""
        repo.add_interval(10.0, 20.0, "Rally")
        repo.add_point(20.0, "Real Stroke")
        assert repo.remove_at(20.0) == 1
        assert repo.count() == {"intervals": 1, "points": 0, "total": 1}
        assert repo.interval_tuples() == [(10.0, 20.0)]

    def test_remove_at_falls_back_to_a_containing_interval(self, repo):
        repo.add_interval(10.0, 20.0, "Rally")
        assert repo.remove_at(15.0) == 1
        assert repo.count()["intervals"] == 0

    def test_remove_at_picks_the_nearest_point(self, repo):
        repo.add_point(10.0, "Real Stroke")
        repo.add_point(10.4, "Serve")
        repo.remove_at(10.4, tolerance=0.5)
        assert repo.point_marks() == [(10.0, "Real Stroke")]

    def test_clear(self, repo):
        repo.add_interval(0.0, 5.0, "Rally")
        repo.clear()
        assert repo.count()["total"] == 0

    def test_editing_never_renumbers_other_labels(self, repo):
        a = repo.add_point(1.0, "Serve")
        b = repo.add_point(2.0, "Serve")
        repo.remove(a.label_id)
        assert b.label_id in [entry.label_id for entry in repo.labels]


class TestQueries:
    def test_intervals_and_points_are_separated(self, repo):
        repo.add_interval(0.0, 5.0, "Rally")
        repo.add_point(2.0, "Real Stroke")
        assert len(repo.intervals()) == 1
        assert len(repo.points()) == 1

    def test_interval_tuples_are_sorted(self, repo):
        repo.add_interval(30.0, 40.0, "Rally")
        repo.add_interval(5.0, 10.0, "Rally")
        assert repo.interval_tuples() == [(5.0, 10.0), (30.0, 40.0)]

    def test_interval_spans_carry_the_class(self, repo):
        repo.add_interval(1.0, 2.0, "Shadow Practice")
        assert repo.interval_spans() == [(1.0, 2.0, "Shadow Practice")]

    def test_point_marks_carry_the_class(self, repo):
        repo.add_point(3.0, "Serve")
        assert repo.point_marks() == [(3.0, "Serve")]

    def test_labels_at_finds_a_covering_interval(self, repo):
        repo.add_interval(10.0, 20.0, "Rally")
        assert len(repo.labels_at(15.0)) == 1
        assert repo.labels_at(25.0) == []

    def test_labels_at_uses_tolerance_for_points(self, repo):
        repo.add_point(10.0, "Serve")
        assert len(repo.labels_at(10.2, tolerance=0.25)) == 1
        assert repo.labels_at(10.5, tolerance=0.25) == []

    def test_next_and_previous_label(self, repo):
        repo.add_point(10.0, "Serve")
        repo.add_point(30.0, "Serve")
        assert repo.next_label(15.0).time == 30.0
        assert repo.previous_label(15.0).time == 10.0

    def test_next_at_the_end_is_none(self, repo):
        repo.add_point(10.0, "Serve")
        assert repo.next_label(20.0) is None

    def test_previous_at_the_start_is_none(self, repo):
        repo.add_point(10.0, "Serve")
        assert repo.previous_label(5.0) is None


class TestPersistence:
    def test_round_trip_through_labels_csv(self, temp_dir):
        path = os.path.join(temp_dir, "v_labels.csv")
        repo = AnnotationRepository(path=path)
        repo.add_interval(12.5, 25.0, "Rally", notes="crosscourt")
        repo.add_point(403.42, "Real Stroke", notes="late swing")

        restored = AnnotationRepository.from_labels_csv(path)
        assert restored.count() == {"intervals": 1, "points": 1, "total": 2}
        interval = restored.intervals()[0]
        assert (interval.start, interval.end) == pytest.approx((12.5, 25.0))
        assert interval.label == "Rally"
        assert interval.notes == "crosscourt"
        point = restored.points()[0]
        assert point.time == pytest.approx(403.42)

    def test_header_is_stable(self, temp_dir):
        from really_rally.studio.annotation import LABEL_COLUMNS
        path = os.path.join(temp_dir, "v_labels.csv")
        AnnotationRepository(path=path).add_point(1.0, "Serve")
        with open(path) as f:
            assert next(csv.reader(f)) == LABEL_COLUMNS

    def test_autosave_persists_edits(self, temp_dir):
        """Annotations must not be lost if the app closes without a save."""
        path = os.path.join(temp_dir, "v_labels.csv")
        repo = AnnotationRepository(path=path)
        repo.add_point(5.0, "Serve")
        assert os.path.exists(path)
        assert AnnotationRepository.from_labels_csv(path).count()["points"] == 1

    def test_autosave_keeps_edits_when_the_path_is_unwritable(self):
        repo = AnnotationRepository(path="/nonexistent-root-dir/x/../y_labels.csv")
        entry = repo.add_point(1.0, "Serve")   # must not raise
        assert entry in repo.labels

    def test_save_without_a_path_is_an_error(self):
        with pytest.raises(ValueError):
            AnnotationRepository().save()

    def test_repository_without_a_path_keeps_edits_in_memory(self):
        repo = AnnotationRepository()
        repo.add_interval(0.0, 1.0, "Rally")
        assert repo.count()["intervals"] == 1

    def test_missing_file_loads_empty(self, temp_dir):
        repo = AnnotationRepository.from_labels_csv(os.path.join(temp_dir, "nope.csv"))
        assert repo.count()["total"] == 0

    def test_sorted_output_is_stable(self, temp_dir):
        path = os.path.join(temp_dir, "v_labels.csv")
        repo = AnnotationRepository(path=path)
        repo.add_point(30.0, "Serve")
        repo.add_point(10.0, "Serve")
        restored = AnnotationRepository.from_labels_csv(path)
        assert [p.time for p in restored.points()] == [10.0, 30.0]


class TestAnnotationsJsonCompat:
    def test_imports_the_benchmark_format(self, gt_json_file):
        repo = AnnotationRepository.from_annotations_json(gt_json_file)
        assert repo.count()["intervals"] == 3
        assert repo.interval_tuples() == [(10.0, 20.0), (30.0, 45.0), (60.0, 75.0)]
        assert repo.intervals()[0].label == "rally"
        assert repo.intervals()[0].source == SOURCE_IMPORTED

    def test_missing_file_imports_empty(self, temp_dir):
        repo = AnnotationRepository.from_annotations_json(os.path.join(temp_dir, "no.json"))
        assert repo.count()["total"] == 0

    def test_exports_a_format_evaluate_py_can_read(self, temp_dir):
        """Round-trips through the scoring tool rather than trusting the shape."""
        from really_rally.utils.evaluate import compute_metrics, load_ground_truth

        path = os.path.join(temp_dir, "annotations.json")
        repo = AnnotationRepository()
        repo.add_interval(10.0, 20.0, "Rally")
        repo.add_interval(30.0, 45.0, "Rally")
        repo.to_annotations_json(path)

        gt = load_ground_truth(path)
        assert gt == [(10.0, 20.0), (30.0, 45.0)]
        metrics = compute_metrics(gt, [(10.0, 20.0), (30.0, 45.0)])
        assert metrics["f1"] == pytest.approx(1.0)

    def test_export_merges_overlapping_intervals(self, temp_dir):
        path = os.path.join(temp_dir, "annotations.json")
        repo = AnnotationRepository()
        repo.add_interval(10.0, 20.0, "Rally")
        repo.add_interval(15.0, 30.0, "Rally")
        repo.to_annotations_json(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["start"] == 10.0 and data[0]["end"] == 30.0

    def test_export_excludes_points(self, temp_dir):
        """evaluate.py matches intervals; a point would be a zero-length seg."""
        path = os.path.join(temp_dir, "annotations.json")
        repo = AnnotationRepository()
        repo.add_interval(10.0, 20.0, "Rally")
        repo.add_point(15.0, "Real Stroke")
        repo.to_annotations_json(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1 and data[0]["start"] == 10.0

    def test_export_empty_writes_an_empty_list(self, temp_dir):
        path = os.path.join(temp_dir, "annotations.json")
        AnnotationRepository().to_annotations_json(path)
        with open(path) as f:
            assert json.load(f) == []


class TestGroundTruthIsolation:
    """The core guarantee: retuning the detector cannot touch labels."""

    def test_labels_file_is_independent_of_predictions(self, temp_dir):
        labels_path = os.path.join(temp_dir, "v_labels.csv")
        predictions_path = os.path.join(temp_dir, "v_predictions.csv")

        repo = AnnotationRepository(path=labels_path)
        repo.add_interval(10.0, 20.0, "Rally")

        with open(predictions_path, "w") as f:
            f.write("Start Time,End Time\n99.0,100.0\n")

        reloaded = AnnotationRepository.from_labels_csv(labels_path)
        assert reloaded.interval_tuples() == [(10.0, 20.0)]

    def test_editing_labels_does_not_touch_a_predictions_file(self, temp_dir):
        predictions_path = os.path.join(temp_dir, "v_predictions.csv")
        original = "Start Time,End Time\n1.0,2.0\n"
        with open(predictions_path, "w") as f:
            f.write(original)

        repo = AnnotationRepository(path=os.path.join(temp_dir, "v_labels.csv"))
        repo.add_point(1.5, "Real Stroke")
        repo.remove_at(1.5)

        with open(predictions_path) as f:
            assert f.read() == original


class TestLabelHelpers:
    def test_span_of_an_interval(self):
        assert Label(kind=KIND_INTERVAL, label="Rally", start=1.0, end=2.0).span() == (1.0, 2.0)

    def test_span_of_a_point_is_zero_width(self):
        assert Label(kind=KIND_POINT, label="Serve", time=7.0).span() == (7.0, 7.0)

    def test_contains_respects_tolerance(self):
        label = Label(kind=KIND_POINT, label="Serve", time=10.0)
        assert label.contains(10.2, tolerance=0.25)
        assert not label.contains(10.9, tolerance=0.25)

    def test_color_is_deterministic_across_calls(self):
        assert color_for("Rally") == color_for("Rally")

    def test_color_differs_between_classes(self):
        assert color_for("Rally") != color_for("Shadow Practice")
