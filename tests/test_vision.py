"""Tests for the pose extractor lifecycle and model-download failure handling."""

import pytest

from really_rally.core import vision


@pytest.fixture(autouse=True)
def _reset_cache():
    """The extractor is process-global; never leak state between tests."""
    original = vision._CACHED_POSE_EXTRACTOR
    yield
    vision._CACHED_POSE_EXTRACTOR = original


class TestPoseExtractorCaching:
    def test_extractor_is_shared_across_calls(self, monkeypatch):
        built = []

        class FakePose:
            def __init__(self):
                built.append(1)

        monkeypatch.setattr(vision, "PoseExtractor", FakePose)
        vision._CACHED_POSE_EXTRACTOR = None

        first = vision.get_pose_extractor()
        second = vision.get_pose_extractor()
        assert first is second
        assert len(built) == 1, "graph must be built once, not per caller"

    def test_failure_is_reported_with_guidance_not_a_raw_traceback(self, monkeypatch):
        """Model download / SSL failures are common and otherwise unreadable."""

        class Boom:
            def __init__(self):
                raise OSError("certificate verify failed")

        monkeypatch.setattr(vision, "PoseExtractor", Boom)
        vision._CACHED_POSE_EXTRACTOR = None

        with pytest.raises(RuntimeError) as excinfo:
            vision.get_pose_extractor()

        message = str(excinfo.value)
        assert "MP_MODEL_COMPLEXITY" in message
        assert "REALLY_RALLY_ALLOW_INSECURE_SSL" in message
        # The original cause must survive so the real error is not lost.
        assert "certificate verify failed" in message

    def test_failed_build_is_not_cached(self, monkeypatch):
        """A failure must not poison the cache for a later retry."""
        calls = {"n": 0}

        class Flaky:
            def __init__(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("transient")

        monkeypatch.setattr(vision, "PoseExtractor", Flaky)
        vision._CACHED_POSE_EXTRACTOR = None

        with pytest.raises(RuntimeError):
            vision.get_pose_extractor()
        assert vision._CACHED_POSE_EXTRACTOR is None
        # A second attempt gets through.
        assert vision.get_pose_extractor() is not None
