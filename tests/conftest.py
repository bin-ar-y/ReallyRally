"""Shared test fixtures for ReallyRally tests."""

import os
import sys
import json
import tempfile

import pytest

# Ensure the src directory is on sys.path for local test runs
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def gt_json_file(temp_dir):
    """Create a temporary ground-truth JSON file."""
    data = [
        {"start": 10.0, "end": 20.0, "label": "rally", "notes": "test 1"},
        {"start": 30.0, "end": 45.0, "label": "rally", "notes": "test 2"},
        {"start": 60.0, "end": 75.0, "label": "rally", "notes": "test 3"},
    ]
    path = os.path.join(temp_dir, "gt.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def pred_csv_file(temp_dir):
    """Create a temporary predictions CSV file."""
    import csv

    path = os.path.join(temp_dir, "pred.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source Video", "Clip", "Start Time", "End Time", "Duration"])
        writer.writerow(["test.mp4", "clip_001.mp4", "11.0", "19.0", "8.0"])
        writer.writerow(["test.mp4", "clip_002.mp4", "32.0", "43.0", "11.0"])
        writer.writerow(["test.mp4", "clip_003.mp4", "80.0", "85.0", "5.0"])
    return path
