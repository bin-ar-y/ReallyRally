"""Tests for the RallySegment data model."""

import pytest

from really_rally.core.models import RallySegment


class TestRallySegment:
    def test_creation_defaults(self):
        seg = RallySegment(
            start_frame=0,
            end_frame=100,
            start_time=0.0,
            end_time=1.667,
        )
        assert seg.start_frame == 0
        assert seg.end_frame == 100
        assert seg.score == 0.0

    def test_duration(self):
        seg = RallySegment(
            start_frame=0,
            end_frame=180,
            start_time=0.0,
            end_time=3.0,
            score=0.85,
        )
        assert seg.duration == pytest.approx(3.0)

    def test_custom_score(self):
        seg = RallySegment(
            start_frame=50,
            end_frame=200,
            start_time=0.833,
            end_time=3.333,
            score=2.45,
        )
        assert seg.score == pytest.approx(2.45)
        assert seg.duration == pytest.approx(2.5)

    def test_equality(self):
        a = RallySegment(0, 100, 0.0, 1.0, 0.5)
        b = RallySegment(0, 100, 0.0, 1.0, 0.5)
        assert a == b

    def test_inequality(self):
        a = RallySegment(0, 100, 0.0, 1.0, 0.5)
        b = RallySegment(0, 100, 0.0, 1.0, 0.7)
        assert a != b
