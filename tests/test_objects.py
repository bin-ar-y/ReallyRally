"""Tests for object evidence: detector contract, ball tracking, racket-hand proximity."""


from really_rally.core.objects import (
    BALL_LABEL,
    RACKET_LABEL,
    BallTracker,
    Detection,
    FrameObjects,
    NullObjectDetector,
    racket_near_hand,
)


def racket(cx, cy, size=0.1, conf=0.9):
    return Detection(RACKET_LABEL, conf, (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))


def ball(cx, cy, conf=0.8):
    return Detection(BALL_LABEL, conf, (cx - 0.01, cy - 0.01, cx + 0.01, cy + 0.01))


class TestNullObjectDetector:
    def test_reports_unavailable_not_absent(self):
        """The null backend must not be mistaken for 'detector ran, found none'."""
        fo = NullObjectDetector().detect(None, 7)
        assert fo.available is False
        assert fo.racket is None if hasattr(fo, "racket") else fo.best_racket is None
        assert fo.best_ball is None

    def test_does_not_provide_evidence(self):
        assert NullObjectDetector.provides_evidence is False


class TestFrameObjects:
    def test_best_racket_picks_highest_confidence(self):
        fo = FrameObjects(0, rackets=[racket(0.2, 0.2, conf=0.4), racket(0.5, 0.5, conf=0.95)])
        assert fo.best_racket.confidence == 0.95

    def test_best_ball_none_when_empty(self):
        assert FrameObjects(0).best_ball is None


class TestRacketNearHand:
    def test_wrist_inside_box_is_near(self):
        assert racket_near_hand(racket(0.5, 0.5, size=0.2), (0.5, 0.5))

    def test_uses_nearest_box_point_not_centre(self):
        """An elongated racket box has its hand at one end, not the centre."""
        # Box spanning x 0.30-0.70, y 0.48-0.52; wrist at the far left end.
        det = Detection(RACKET_LABEL, 0.9, (0.30, 0.48, 0.70, 0.52))
        assert racket_near_hand(det, (0.31, 0.50), max_dist=0.05)

    def test_distant_wrist_is_not_near(self):
        assert not racket_near_hand(racket(0.1, 0.1), (0.9, 0.9), max_dist=0.05)

    def test_missing_wrist_is_not_near(self):
        assert not racket_near_hand(racket(0.5, 0.5), None)


class TestBallTracker:
    def test_isolated_detection_is_not_a_trajectory(self):
        """One sighting must not count as ball flight."""
        t = BallTracker(min_observations=3)
        t.update(0, [ball(0.5, 0.5)])
        assert t.confidence(0) == 0.0

    def test_consistent_track_scores(self):
        t = BallTracker(min_observations=3)
        for i in range(6):
            t.update(i, [ball(0.30 + i * 0.02, 0.50)])
        assert t.confidence(5) > 0.5

    def test_disconnected_sightings_form_separate_tracks(self):
        """Detections far apart in space must not be joined into one path."""
        t = BallTracker(min_observations=3, gate=0.05)
        for i in range(4):
            t.update(i, [ball(0.10, 0.10)])
        assert len(t.tracks) == 1
        for i in range(4, 8):
            t.update(i, [ball(0.90, 0.90)])
        # A single track teleporting 0.8 across the frame would be a false
        # trajectory; the gate must keep the two clusters apart.
        assert len(t.tracks) == 2
        assert all(len(tr["positions"]) == 4 for tr in t.tracks)

    def test_stale_tracks_do_not_absorb_later_detections(self):
        t = BallTracker(min_observations=2, max_gap_frames=2)
        t.update(0, [ball(0.50, 0.50)])
        t.update(1, [ball(0.51, 0.50)])
        before = len(t.tracks)
        # A long gap: the old track must be dropped, not resumed.
        t.update(50, [ball(0.90, 0.90)])
        t.update(51, [ball(0.91, 0.90)])
        assert len(t.tracks) == before  # old dropped, one new started

    def test_reset_clears_tracks(self):
        t = BallTracker(min_observations=2)
        t.update(0, [ball(0.5, 0.5)])
        t.reset()
        assert t.tracks == []

    def test_track_across_gap_within_tolerance(self):
        """A brief occlusion should not break the trajectory."""
        t = BallTracker(min_observations=3, max_gap_frames=5)
        t.update(0, [ball(0.30, 0.50)])
        t.update(1, [ball(0.32, 0.50)])
        t.update(2, [ball(0.34, 0.50)])
        t.update(5, [ball(0.40, 0.50)])  # reappears after 2 missing frames
        assert len(t.tracks) == 1
        assert len(t.tracks[0]["positions"]) == 4
