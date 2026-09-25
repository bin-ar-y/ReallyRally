"""Tests for coarse candidate scoring and the hysteresis rally state machine."""


from really_rally.core.candidate import CoarseCandidateDetector, RallyStateMachine

FPS = 30.0
MAX_TIME = 1000.0


def _detector(**kw):
    return CoarseCandidateDetector(**kw)


class TestCoarseCandidateDetector:
    def test_no_evidence_when_pose_invalid(self):
        d = _detector()
        score, fired = d.score(
            valid_pose=False, valid_distance=True,
            z_arm=99.0, t_arm=1.0, z_whip=99.0, t_whip=1.0,
            hip_vel=1.0, spatial_var=1.0,
        )
        assert score == 0.0
        assert fired == []

    def test_no_evidence_when_player_out_of_range(self):
        d = _detector()
        score, fired = d.score(
            valid_pose=True, valid_distance=False,
            z_arm=99.0, t_arm=1.0, z_whip=99.0, t_whip=1.0,
            hip_vel=1.0, spatial_var=1.0,
        )
        assert score == 0.0
        assert fired == []

    def test_single_strong_signal_can_fire_alone(self):
        """A high arm energy alone must clear the bar — no AND-gate."""
        d = _detector()
        score, fired = d.score(
            valid_pose=True, valid_distance=True,
            z_arm=5.0, t_arm=1.0,
            z_whip=0.0, t_whip=1.0,     # whip fails
            hip_vel=0.0, spatial_var=0.0,  # hip + spatial fail
        )
        assert fired == ["arm"]
        assert d.is_candidate(score)

    def test_whip_alone_can_fire(self):
        """Body-rotation strokes have whip ≈ 0 contribution; arm alone suffices."""
        d = _detector()
        score, fired = d.score(
            valid_pose=True, valid_distance=True,
            z_arm=0.0, t_arm=1.0,
            z_whip=5.0, t_whip=1.0,
            hip_vel=0.0, spatial_var=0.0,
        )
        assert fired == ["whip"]
        assert d.is_candidate(score)

    def test_weak_signals_accumulate(self):
        """No single signal clears its own weight bar, but together they do."""
        d = _detector(w_hip=1.0, w_spatial=1.0)
        score, fired = d.score(
            valid_pose=True, valid_distance=True,
            z_arm=0.0, t_arm=1.0,
            z_whip=0.0, t_whip=1.0,
            hip_vel=1.0, spatial_var=1.0,
        )
        assert fired == ["hip", "spatial"]
        assert score == 2.0
        assert d.is_candidate(score)

    def test_single_weak_signal_does_not_fire(self):
        """Hip movement alone (walking) must not open a candidate."""
        d = _detector()
        score, _ = d.score(
            valid_pose=True, valid_distance=True,
            z_arm=0.0, t_arm=1.0,
            z_whip=0.0, t_whip=1.0,
            hip_vel=1.0, spatial_var=0.0,
        )
        assert score < d.candidate_threshold
        assert not d.is_candidate(score)

    def test_all_signals_max_out(self):
        d = _detector()
        score, fired = d.score(
            valid_pose=True, valid_distance=True,
            z_arm=5.0, t_arm=1.0, z_whip=5.0, t_whip=1.0,
            hip_vel=1.0, spatial_var=1.0,
        )
        assert set(fired) == {"arm", "whip", "hip", "spatial"}
        assert score == 5.5


class TestRallyStateMachine:
    def _sm(self, **kw):
        return RallyStateMachine(fps=FPS, max_time=MAX_TIME, **kw)

    def test_starts_idle(self):
        assert self._sm().state == RallyStateMachine.IDLE

    def test_single_candidate_does_not_confirm(self):
        sm = self._sm()
        sm.step(True, frame=0, time=0.0, score=3.0)
        assert sm.state == RallyStateMachine.CANDIDATE
        assert sm.finalize() is None  # never confirmed

    def test_two_candidates_confirm_rally(self):
        sm = self._sm()
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=30, time=1.0, score=3.0)
        assert sm.state == RallyStateMachine.ACTIVE

    def test_finalize_returns_trailing_rally(self):
        sm = self._sm()
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=120, time=4.0, score=3.0)
        seg = sm.finalize()
        assert seg is not None
        assert seg.end_time > seg.start_time

    def test_candidate_times_out_to_idle(self):
        sm = self._sm(start_gap_sec=7.0)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(False, frame=300, time=10.0, score=0.0)
        assert sm.state == RallyStateMachine.IDLE

    def test_active_enters_cooldown_then_ends(self):
        """A single missed stroke must not end the rally immediately."""
        sm = self._sm(start_gap_sec=7.0, continue_gap_sec=11.0)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=30, time=1.0, score=3.0)
        assert sm.state == RallyStateMachine.ACTIVE

        # 8s gap: beyond start_gap, inside continue_gap -> still alive
        seg = sm.step(False, frame=270, time=9.0, score=0.0)
        assert seg is None
        assert sm.state == RallyStateMachine.COOLDOWN

        # A new candidate in cooldown resumes the rally
        sm.step(True, frame=300, time=10.0, score=3.0)
        assert sm.state == RallyStateMachine.ACTIVE

    def test_cooldown_times_out_and_emits_segment(self):
        sm = self._sm(start_gap_sec=7.0, continue_gap_sec=11.0)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=120, time=4.0, score=3.0)
        seg = sm.step(False, frame=750, time=25.0, score=0.0)
        assert seg is not None
        assert seg.start_time == 0.0
        assert sm.state == RallyStateMachine.IDLE

    def test_continuation_gap_longer_than_start_gap(self):
        """Core hysteresis invariant, exercised end-to-end."""
        sm = self._sm(start_gap_sec=2.0, continue_gap_sec=20.0)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=30, time=1.0, score=3.0)
        # 3s gap exceeds start_gap but is well inside continue_gap
        seg = sm.step(False, frame=120, time=4.0, score=0.0)
        assert seg is None
        assert sm.state == RallyStateMachine.COOLDOWN

    def test_short_rally_discarded(self):
        """Rallies below min duration are not emitted."""
        sm = self._sm(min_duration_sec=10.0, start_gap_sec=7.0, continue_gap_sec=11.0)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=30, time=1.0, score=3.0)
        seg = sm.step(False, frame=400, time=13.0, score=0.0)
        assert seg is None

    def test_peak_score_tracked(self):
        sm = self._sm(start_gap_sec=7.0, continue_gap_sec=11.0)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=120, time=4.0, score=5.0)
        seg = sm.step(False, frame=750, time=25.0, score=0.0)
        assert seg is not None
        assert seg.score == 5.0

    def test_double_trigger_guard_ignores_same_stroke(self):
        """Two candidate frames 0.1s apart are one stroke, not two."""
        sm = self._sm(double_trigger_guard_sec=0.5)
        sm.step(True, frame=0, time=0.0, score=3.0)
        sm.step(True, frame=3, time=0.1, score=3.0)
        assert sm.state == RallyStateMachine.CANDIDATE

    def test_state_name_reported(self):
        sm = self._sm()
        assert sm.state_name == "idle"
        sm.step(True, frame=0, time=0.0, score=3.0)
        assert sm.state_name == "candidate"
