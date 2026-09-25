"""Tests for evidence fusion across pose / racket / ball / temporal channels."""

from really_rally.core.fusion import EvidenceFusion, RallyEvidence


def _f():
    return EvidenceFusion()


class TestFusionContract:
    def test_no_detectors_available_is_neutral(self):
        """Every unavailable channel contributes neutral, not zero."""
        f = _f()
        score = f.fuse(RallyEvidence(pose_score=0.5, racket_score=None,
                                     ball_score=None, temporal_score=0.5))
        assert score == 0.5

    def test_absent_racket_scores_lower_than_unknown_racket(self):
        """'Detector ran, found nothing' must count against; 'no detector' must not."""
        f = _f()
        unknown = f.fuse(RallyEvidence(pose_score=0.9, racket_score=None, ball_score=None))
        absent = f.fuse(RallyEvidence(pose_score=0.9, racket_score=0.15, ball_score=0.15))
        assert absent < unknown

    def test_full_agreement_is_high(self):
        f = _f()
        assert f.fuse(RallyEvidence(pose_score=1.0, racket_score=1.0,
                                    ball_score=1.0, temporal_score=1.0)) == 1.0

    def test_all_zero_is_zero(self):
        f = _f()
        assert f.fuse(RallyEvidence(pose_score=0.0, racket_score=0.0,
                                    ball_score=0.0, temporal_score=0.0)) == 0.0

    def test_score_bounded(self):
        f = _f()
        for p in (0.0, 0.5, 1.0):
            s = f.fuse(RallyEvidence(pose_score=p, racket_score=p, ball_score=p, temporal_score=p))
            assert 0.0 <= s <= 1.0


class TestFusionReasoning:
    """The reasoning table agreed for this project, as executable checks."""

    def test_strong_pose_racket_ball_is_very_high(self):
        f = _f()
        assert f.fuse(RallyEvidence(0.9, 0.9, 0.9, 0.8)) > 0.85

    def test_strong_pose_and_racket_with_ball_missing_is_likely(self):
        """Ball detection is intrinsically hard; its absence must not veto."""
        f = _f()
        likely = f.fuse(RallyEvidence(pose_score=0.9, racket_score=0.85, ball_score=None))
        assert likely > 0.7

    def test_strong_pose_no_ball_across_window_is_only_possible(self):
        """Long ball absence lowers confidence but stays above a hard reject."""
        f = _f()
        shadow = f.fuse(RallyEvidence(pose_score=0.9, racket_score=0.15, ball_score=0.15))
        definite = f.fuse(RallyEvidence(pose_score=0.9, racket_score=0.9, ball_score=0.9))
        assert shadow < definite

    def test_ball_alone_is_insufficient(self):
        """Ball evidence with no pose/stroke structure must not confirm a rally."""
        f = _f()
        assert f.fuse(RallyEvidence(pose_score=0.0, racket_score=None,
                                    ball_score=1.0, temporal_score=0.0)) < 0.5

    def test_movement_without_ball_never_hard_rejects(self):
        """Even the worst detector outcome leaves a real rally above zero."""
        f = _f()
        assert f.fuse(RallyEvidence(pose_score=1.0, racket_score=0.0,
                                    ball_score=0.0, temporal_score=0.0)) > 0.0


class TestChannelScorers:
    def test_racket_no_detector_run_returns_none(self):
        assert _f().score_racket(0, 0, 0) is None

    def test_racket_absent_does_not_collapse_to_zero(self):
        """A missed box must not destroy a genuine rally."""
        s = _f().score_racket(racket_frames=0, frames_checked=100, near_hand_frames=0)
        assert s is not None and s >= 0.15

    def test_racket_persistently_present_scores_high(self):
        s = _f().score_racket(racket_frames=40, frames_checked=100, near_hand_frames=40)
        assert s > 0.8

    def test_racket_present_but_never_near_hand_scores_lower(self):
        near = _f().score_racket(40, 100, 40)
        far = _f().score_racket(40, 100, 0)
        assert far < near

    def test_ball_no_detector_run_returns_none(self):
        """Unavailable must stay neutral, never a low score."""
        assert _f().score_ball(trajectory_confidence=0.0, observations=0, frames_checked=0) is None

    def test_ball_no_observations_returns_weak_signal(self):
        s = _f().score_ball(trajectory_confidence=0.0, observations=0, frames_checked=100)
        assert s == 0.15

    def test_ball_trajectory_dominates_single_detection(self):
        f = _f()
        tracked = f.score_ball(trajectory_confidence=0.9, observations=10, frames_checked=100)
        single = f.score_ball(trajectory_confidence=0.1, observations=1, frames_checked=100)
        assert tracked > single


class TestAcceptance:
    def test_accept_uses_threshold(self):
        f = EvidenceFusion(min_confidence=0.6)
        assert f.accept(RallyEvidence(0.9, 0.9, 0.9, 0.9))
        assert not f.accept(RallyEvidence(0.1, 0.1, 0.1, 0.1))

    def test_pose_only_rally_still_accepted_by_default(self):
        """Regression guard: object evidence must not disable the current mode."""
        f = _f()
        pose_only = RallyEvidence(pose_score=0.9, racket_score=None,
                                  ball_score=None, temporal_score=0.5)
        assert f.accept(pose_only)
