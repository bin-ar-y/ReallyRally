"""Integration tests for Pass 3: object verification inside candidate windows.

These exercise the real wiring — video seeking, window padding, per-frame
downscaling, tracking and fusion — using a synthetic video and a fake detector,
so the architecture is proven end-to-end without any inference dependency.
"""

import numpy as np
import pytest

from really_rally.core import objects
from really_rally.core.models import RallySegment
from really_rally.core.objects import (
    BALL_LABEL,
    RACKET_LABEL,
    Detection,
    FrameObjects,
    ObjectDetector,
)
from really_rally.processors.base import TennisRallyProcessor

cv2 = pytest.importorskip("cv2")

FPS = 30.0
N_FRAMES = 60


@pytest.fixture
def synth_video(tmp_path):
    """A small real video file, so the processor can open and seek it."""
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for i in range(N_FRAMES):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + i, 100), (60 + i, 180), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture(autouse=True)
def _reset_registry():
    """The detector registry is process-global; never leak between tests."""
    yield
    objects._FACTORY = None


class FakeDetector(ObjectDetector):
    """Reports a racket and a ball, or nothing, per configuration."""

    provides_evidence = True

    def __init__(self, with_racket=True, with_ball=True, available=True):
        self.with_racket = with_racket
        self.with_ball = with_ball
        self.available = available
        self.calls = 0

    def detect(self, frame_bgr, frame_idx):
        self.calls += 1
        if not self.available:
            return FrameObjects(frame_idx=frame_idx, available=False)
        rackets, balls = [], []
        if self.with_racket:
            # A racket box near the wrist position the metrics report (0.5, 0.5).
            rackets.append(Detection(RACKET_LABEL, 0.9, (0.45, 0.45, 0.55, 0.55)))
        if self.with_ball:
            # A plausible rightward trajectory.
            x = 0.20 + (frame_idx % 20) * 0.01
            balls.append(Detection(BALL_LABEL, 0.8, (x - 0.01, 0.2, x + 0.01, 0.22)))
        return FrameObjects(frame_idx=frame_idx, rackets=rackets, balls=balls)


def _make_processor(video, tmp_path):
    proc = TennisRallyProcessor(video, str(tmp_path), keep_clips=False)
    # Populate telemetry the way Pass 2 would, so window scoring has inputs.
    proc.all_frame_metrics = [
        {
            "frame": f,
            "coarse_score": 4.0,
            "candidate": 1 if f % 10 < 3 else 0,
            "l_wrist_vel": 0.9,
            "r_wrist_vel": 0.4,
            "l_wrist_x": 0.5,
            "l_wrist_y": 0.5,
            "r_wrist_x": 0.5,
            "r_wrist_y": 0.5,
        }
        for f in range(N_FRAMES)
    ]
    return proc


def _segment():
    return RallySegment(start_frame=10, end_frame=40, start_time=10 / FPS, end_time=40 / FPS, score=3.0)


class TestPass3DisabledByDefault:
    def test_null_detector_skips_verification(self, synth_video, tmp_path):
        proc = _make_processor(synth_video, tmp_path)
        assert proc.object_detector.provides_evidence is False
        segs = proc.verify_candidate_windows([_segment()])
        assert len(segs) == 1
        assert segs[0].evidence is None  # untouched


class TestPass3WithDetector:
    def test_racket_and_ball_evidence_is_recorded(self, synth_video, tmp_path):
        objects.register_object_detector(lambda: FakeDetector())
        proc = _make_processor(synth_video, tmp_path)
        segs = proc.verify_candidate_windows([_segment()])
        assert len(segs) == 1
        ev = segs[0].evidence
        assert ev is not None
        assert ev.racket_score is not None and ev.racket_score > 0.5
        assert ev.ball_score is not None
        assert segs[0].score > 0.0

    def test_detector_runs_only_inside_the_window(self, synth_video, tmp_path):
        """Inference must be confined to the padded candidate window."""
        det = FakeDetector()
        objects.register_object_detector(lambda: det)
        proc = _make_processor(synth_video, tmp_path)

        # Window is frames 10-40 padded by 1s (30 frames) -> clamped to 0-59.
        proc.verify_candidate_windows([_segment()])
        # Every 2nd frame over 60 frames, and never more than the whole video.
        assert 0 < det.calls <= N_FRAMES // 2 + 1

    def test_inference_scales_with_window_not_video_length(self, synth_video, tmp_path):
        """Cost must track candidate-window size, not the length of the video."""
        import really_rally.processors.base as base

        det = FakeDetector()
        objects.register_object_detector(lambda: det)
        proc = _make_processor(synth_video, tmp_path)

        # No padding: the window is exactly the segment, so inference is
        # confined to it even though the video is much longer.
        base.OBJECT_WINDOW_PAD_SEC = 0.0
        try:
            seg = RallySegment(start_frame=10, end_frame=20, start_time=10 / FPS, end_time=20 / FPS)
            proc.verify_candidate_windows([seg])
            # 11 frames, every 2nd one processed.
            assert det.calls <= 6, f"ran {det.calls} inferences for an 11-frame window"
        finally:
            base.OBJECT_WINDOW_PAD_SEC = 1.0

    def test_padding_widens_the_window(self, synth_video, tmp_path):
        import really_rally.processors.base as base

        det = FakeDetector()
        objects.register_object_detector(lambda: det)
        proc = _make_processor(synth_video, tmp_path)
        base.OBJECT_WINDOW_PAD_SEC = 0.0
        try:
            seg = RallySegment(start_frame=20, end_frame=30, start_time=20 / FPS, end_time=30 / FPS)
            proc.verify_candidate_windows([seg])
            unpadded = det.calls

            det.calls = 0
            base.OBJECT_WINDOW_PAD_SEC = 0.5  # 15 frames each side
            proc.verify_candidate_windows([seg])
            padded = det.calls
        finally:
            base.OBJECT_WINDOW_PAD_SEC = 1.0

        assert padded > unpadded

    def test_missing_ball_does_not_reject_a_racket_rally(self, synth_video, tmp_path):
        """Ball detection is hard; its absence must not destroy a real rally."""
        objects.register_object_detector(lambda: FakeDetector(with_ball=False))
        proc = _make_processor(synth_video, tmp_path)
        segs = proc.verify_candidate_windows([_segment()])
        assert len(segs) == 1

    def test_no_racket_and_no_ball_can_be_rejected(self, synth_video, tmp_path):
        """Strong pose but the detector saw nothing -> possible shadow swing."""
        objects.register_object_detector(
            lambda: FakeDetector(with_racket=False, with_ball=False)
        )
        proc = _make_processor(synth_video, tmp_path)
        # Weak pose/temporal evidence so the fused score lands below the bar.
        for d in proc.all_frame_metrics:
            d["coarse_score"] = 0.5
            d["candidate"] = 0
        segs = proc.verify_candidate_windows([_segment()])
        assert segs == []

    def test_unavailable_detector_is_neutral_not_absent(self, synth_video, tmp_path):
        """A detector that fails to run must not count as evidence of absence."""
        objects.register_object_detector(lambda: FakeDetector(available=False))
        proc = _make_processor(synth_video, tmp_path)
        segs = proc.verify_candidate_windows([_segment()])
        assert len(segs) == 1
        ev = segs[0].evidence
        assert ev.racket_score is None and ev.ball_score is None

    def test_stop_event_halts_verification(self, synth_video, tmp_path):
        import threading

        objects.register_object_detector(lambda: FakeDetector())
        proc = _make_processor(synth_video, tmp_path)
        stop = threading.Event()
        stop.set()
        proc.stop_event = stop
        segs = proc.verify_candidate_windows([_segment()])
        assert segs == []


class TestRegistry:
    def test_default_is_null(self):
        assert isinstance(objects.get_object_detector(), objects.NullObjectDetector)

    def test_registration_takes_effect(self):
        objects.register_object_detector(lambda: FakeDetector())
        assert isinstance(objects.get_object_detector(), FakeDetector)
