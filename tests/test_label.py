"""Tests for the benchmark labelling helpers."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks")))

from label import (  # noqa: E402
    contact_sheet,
    init_case,
    load_annotations,
    merge_overlapping,
    parse_range,
    save_annotations,
    video_metadata,
)

cv2 = pytest.importorskip("cv2")
import numpy as np  # noqa: E402


class TestParseRange:
    @pytest.mark.parametrize("text,expected", [
        ("12.5-25", (12.5, 25.0)),
        ("12.5:25", (12.5, 25.0)),
        ("40-55.3", (40.0, 55.3)),
        ("0-1", (0.0, 1.0)),
        ("12.5 25", (12.5, 25.0)),
        ("12,25", (12.0, 25.0)),
    ])
    def test_accepts_common_separators(self, text, expected):
        assert parse_range(text) == expected

    @pytest.mark.parametrize("bad", ["25-12", "abc", "12", "12-12", "-5-10", "1-2-3"])
    def test_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_range(bad)


class TestMergeOverlapping:
    def test_empty(self):
        assert merge_overlapping([]) == []

    def test_sorts_by_start(self):
        out = merge_overlapping([
            {"start": 30, "end": 40, "label": "rally", "notes": ""},
            {"start": 5, "end": 10, "label": "rally", "notes": ""},
        ])
        assert [s["start"] for s in out] == [5, 30]

    def test_merges_overlapping(self):
        out = merge_overlapping([
            {"start": 0, "end": 10, "label": "rally", "notes": ""},
            {"start": 5, "end": 20, "label": "rally", "notes": ""},
        ])
        assert len(out) == 1
        assert out[0]["start"] == 0 and out[0]["end"] == 20

    def test_keeps_disjoint(self):
        out = merge_overlapping([
            {"start": 0, "end": 2, "label": "rally", "notes": ""},
            {"start": 10, "end": 12, "label": "rally", "notes": ""},
        ])
        assert len(out) == 2

    def test_does_not_mutate_input(self):
        segs = [{"start": 0, "end": 10, "label": "rally", "notes": ""},
                {"start": 5, "end": 20, "label": "rally", "notes": ""}]
        merge_overlapping(segs)
        assert segs[0]["end"] == 10


class TestAnnotationIO:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "annotations.json")
        segs = [{"start": 1.0, "end": 2.0, "label": "rally", "notes": "n"}]
        save_annotations(path, segs)
        assert load_annotations(path) == segs

    def test_missing_file_is_empty(self, tmp_path):
        assert load_annotations(str(tmp_path / "nope.json")) == []

    def test_tolerates_missing_optional_fields(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text(json.dumps([{"start": 1, "end": 2}]))
        got = load_annotations(str(path))
        assert got[0]["label"] == "rally" and got[0]["notes"] == ""


@pytest.fixture
def synth_video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for i in range(90):  # 3 seconds
        writer.write(np.full((120, 160, 3), i % 255, dtype=np.uint8))
    writer.release()
    return str(path)


class TestContactSheet:
    def test_writes_a_strip(self, synth_video, tmp_path):
        out = str(tmp_path / "sheet.png")
        assert contact_sheet(synth_video, out, count=4) == out
        assert os.path.isfile(out)
        assert cv2.imread(out) is not None

    def test_missing_video_raises(self, tmp_path):
        with pytest.raises(ValueError):
            contact_sheet(str(tmp_path / "nope.mp4"), str(tmp_path / "s.png"))


class TestVideoMetadata:
    def test_reads_dimensions(self, synth_video):
        meta = video_metadata(synth_video)
        assert meta["width"] == 160 and meta["height"] == 120
        assert meta["fps"] == pytest.approx(30.0, abs=0.5)
        assert meta["duration"] == pytest.approx(3.0, abs=0.2)


class TestInitCase:
    def test_creates_case_layout(self, tmp_path, monkeypatch, synth_video):
        import label

        monkeypatch.setattr(label, "CASES_DIR", str(tmp_path / "cases"))
        target = init_case("my_case", synth_video, mode="serve")
        assert os.path.isdir(target)
        assert open(os.path.join(target, "mode.txt")).read().strip() == "serve"
        assert os.path.isfile(os.path.join(target, "annotations.json"))
        assert any(f.startswith("video") for f in os.listdir(target))

    def test_does_not_clobber_existing_labels(self, tmp_path, monkeypatch, synth_video):
        import label

        monkeypatch.setattr(label, "CASES_DIR", str(tmp_path / "cases"))
        target = init_case("c", synth_video)
        save_annotations(os.path.join(target, "annotations.json"),
                         [{"start": 1.0, "end": 2.0, "label": "rally", "notes": ""}])
        init_case("c", synth_video)
        assert len(load_annotations(os.path.join(target, "annotations.json"))) == 1
