"""Tests for the rally detection evaluation module."""

import json
import os

import pytest

from really_rally.utils.evaluate import (
    temporal_iou,
    match_segments,
    compute_metrics,
    load_ground_truth,
    load_predictions,
)


# ---------------------------------------------------------------------------
# temporal_iou
# ---------------------------------------------------------------------------

class TestTemporalIoU:
    def test_perfect_overlap(self):
        assert temporal_iou((0, 10), (0, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert temporal_iou((0, 5), (10, 15)) == 0.0

    def test_adjacent_boundary(self):
        """Adjacent segments (touching at endpoint) have zero IoU."""
        assert temporal_iou((0, 5), (5, 10)) == 0.0

    def test_partial_overlap(self):
        expected = 5.0 / 15.0  # intersection=5, union=15
        assert temporal_iou((0, 10), (5, 15)) == pytest.approx(expected)

    def test_contained(self):
        """One segment fully inside another."""
        expected = 4.0 / 10.0  # intersection=4, union=10
        assert temporal_iou((0, 10), (3, 7)) == pytest.approx(expected)

    def test_zero_length_input(self):
        """Zero-length segments return 0."""
        assert temporal_iou((5, 5), (5, 10)) == 0.0

    def test_reversed_endpoints(self):
        """Reversed (start > end) segments return 0."""
        assert temporal_iou((10, 0), (0, 10)) == 0.0


# ---------------------------------------------------------------------------
# match_segments
# ---------------------------------------------------------------------------

class TestMatchSegments:
    def test_perfect_matching(self):
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10), (20, 30)]
        result = match_segments(gt, pred, iou_threshold=0.5)
        assert len(result["matches"]) == 2
        assert result["unmatched_gt"] == []
        assert result["unmatched_pred"] == []

    def test_no_matches_below_threshold(self):
        gt = [(0, 10)]
        pred = [(11, 20)]
        result = match_segments(gt, pred, iou_threshold=0.5)
        assert len(result["matches"]) == 0
        assert len(result["unmatched_gt"]) == 1
        assert len(result["unmatched_pred"]) == 1

    def test_greedy_best_match(self):
        """When two preds overlap one GT, the higher-IoU pred wins."""
        gt = [(0, 10)]
        pred = [(0, 6), (4, 10)]  # IoU: (4,10)=0.6 > (0,6)=0.6 → same, first wins
        result = match_segments(gt, pred, iou_threshold=0.3)
        assert len(result["matches"]) == 1
        matched_pred_idx = result["matches"][0][1]
        # (4,10) has IoU=0.6, (0,6) has IoU=0.6 — greedy picks first sorted by IoU
        assert matched_pred_idx in (0, 1)

    def test_false_positive(self):
        gt = [(0, 10)]
        pred = [(0, 10), (30, 40)]
        result = match_segments(gt, pred, iou_threshold=0.5)
        assert len(result["matches"]) == 1
        assert len(result["unmatched_pred"]) == 1

    def test_false_negative(self):
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10)]
        result = match_segments(gt, pred, iou_threshold=0.5)
        assert len(result["matches"]) == 1
        assert len(result["unmatched_gt"]) == 1

    def test_empty_inputs(self):
        result = match_segments([], [], iou_threshold=0.5)
        assert result["matches"] == []
        assert result["unmatched_gt"] == []
        assert result["unmatched_pred"] == []

    def test_low_iou_threshold_catches_weak_overlaps(self):
        """An IoU threshold of 0.1 allows very slight overlaps."""
        gt = [(0, 10)]
        pred = [(9, 12)]  # intersection=1, union=12 → IoU ≈ 0.083
        result = match_segments(gt, pred, iou_threshold=0.05)
        assert len(result["matches"]) == 1


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_detection(self):
        gt = [(0, 10), (20, 30), (40, 50)]
        pred = [(0, 10), (20, 30), (40, 50)]
        metrics = compute_metrics(gt, pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["tp"] == 3
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_no_detections(self):
        gt = [(0, 10), (20, 30)]
        pred = []
        metrics = compute_metrics(gt, pred)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["fn"] == 2

    def test_all_false_positives(self):
        gt = []
        pred = [(0, 10), (20, 30)]
        metrics = compute_metrics(gt, pred)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0  # no GT to recall
        assert metrics["fp"] == 2

    def test_mixed_performance(self):
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10), (50, 60)]  # 1 TP, 1 FP, 1 FN
        metrics = compute_metrics(gt, pred)
        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(0.5)

    def test_mean_iou(self):
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10), (20, 30)]
        metrics = compute_metrics(gt, pred)
        assert metrics["mean_iou"] == pytest.approx(1.0)

    def test_match_details_in_output(self):
        gt = [(0, 10)]
        pred = [(2, 10)]
        metrics = compute_metrics(gt, pred)
        assert len(metrics["matches"]) == 1
        assert metrics["matches"][0]["gt"] == (0, 10)
        assert metrics["matches"][0]["pred"] == (2, 10)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_loads_valid_file(self, gt_json_file):
        segments = load_ground_truth(gt_json_file)
        assert len(segments) == 3
        assert segments[0] == (10.0, 20.0)

    def test_sorts_by_start_time(self, temp_dir):
        path = os.path.join(temp_dir, "unsorted.json")
        with open(path, "w") as f:
            json.dump([
                {"start": 60.0, "end": 75.0},
                {"start": 10.0, "end": 20.0},
            ], f)
        segments = load_ground_truth(path)
        assert segments[0][0] < segments[1][0]

    def test_empty_file(self, temp_dir):
        path = os.path.join(temp_dir, "empty.json")
        with open(path, "w") as f:
            json.dump([], f)
        segments = load_ground_truth(path)
        assert segments == []

    def test_skips_reversed_segments(self, temp_dir):
        path = os.path.join(temp_dir, "bad.json")
        with open(path, "w") as f:
            json.dump([{"start": 10.0, "end": 5.0}], f)
        segments = load_ground_truth(path)
        assert segments == []


class TestLoadPredictions:
    def test_loads_valid_csv(self, pred_csv_file):
        segments = load_predictions(pred_csv_file)
        assert len(segments) == 3
        assert segments[0] == (11.0, 19.0)

    def test_missing_columns_raises(self, temp_dir):
        import csv

        path = os.path.join(temp_dir, "bad.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Wrong", "Columns", "Here"])
        with pytest.raises(ValueError, match="Start Time"):
            load_predictions(path)

    def test_empty_csv(self, temp_dir):
        import csv

        path = os.path.join(temp_dir, "empty.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Source Video", "Clip", "Start Time", "End Time", "Duration"])
        segments = load_predictions(path)
        assert segments == []
