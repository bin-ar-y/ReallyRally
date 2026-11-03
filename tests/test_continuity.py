"""Observation provenance, anatomical constraints and occlusion boundaries."""

import numpy as np

from really_rally.core.continuity import FORWARD, MISSING, REVERSE, TEMPORAL, complete_pose


def swing():
    t = np.arange(26) * 0.02
    p = np.full((len(t), 33, 3), np.nan)
    p[:, [23, 24, 11, 12]] = [[0.12, 0, 0], [-0.12, 0, 0], [0.18, -0.5, 0], [-0.18, -0.5, 0]]
    for i, a in enumerate(t * 2):
        p[i, 14] = p[i, 12] + [0.02, 0.24 * np.cos(a), 0.24 * np.sin(a)]
        p[i, 16] = p[i, 14] + [-0.12, 0.18 * np.cos(a), 0.18 * np.sin(a)]
    xy = p.copy()
    xy[..., :2] = (p[..., :2] * 200 + 150) / [300, 300]
    lengths = {(12, 14): np.linalg.norm(p[0, 14] - p[0, 12]), (14, 16): np.linalg.norm(p[0, 16] - p[0, 14])}
    return t, p, xy, lengths


def solve(p, xy, lengths, t, reverse=None):
    vis = np.isfinite(p[..., 0]).astype(float)
    return complete_pose(p, p, lengths, xy, vis, t, [300, 300], reverse)


def test_hidden_arm_recovers_in_torso_coordinates_without_changing_observations():
    t, true, xy, lengths = swing()
    p = true.copy()
    p[7:17, [14, 16]] = np.nan
    out, state = solve(p, xy, lengths, t)
    np.testing.assert_equal(state[7:17, [14, 16]], TEMPORAL)
    np.testing.assert_equal(out[state == FORWARD], p[state == FORWARD])
    assert np.max(np.linalg.norm(out[7:17, 16] - true[7:17, 16], axis=1)) < 0.008
    for (a, b), length in lengths.items():
        np.testing.assert_allclose(np.linalg.norm(out[:, b] - out[:, a], axis=1), length, atol=0.001)
    assert np.isnan(p[7:17, 16]).all()


def test_no_bridge_over_long_gap_cut_or_clip_edge():
    t, p, xy, lengths = swing()
    p[1:-1, 14] = np.nan
    p[:3, 16] = np.nan
    out, state = solve(p, xy, lengths, t)
    assert (state[1:-1, 14] == MISSING).all() and np.isnan(out[:3, 16]).all()
    t, p, xy, lengths = swing()
    p[10:16, 16] = np.nan
    t[13:] += 0.3
    out, _ = solve(p, xy, lengths, t)
    assert np.isnan(out[10:16, 16]).all()


def test_reverse_estimate_requires_same_player_and_temporal_support():
    t, true, xy, lengths = swing()
    p = true.copy()
    p[5:20, 16] = np.nan
    vis = np.isfinite(true[..., 0]).astype(float)
    reverse = (true, true, xy, vis)
    _, state = solve(p, xy, lengths, t, reverse)
    assert (state[5:20, 16] == REVERSE).all()
    other = xy.copy()
    other[..., 0] += 0.5
    _, state = solve(p, xy, lengths, t, (true, true, other, vis))
    assert (state[5:20, 16] == TEMPORAL).all()


def test_observed_wrist_constrains_hidden_elbow():
    t, true, xy, lengths = swing()
    p = true.copy()
    p[9:15, 14] = np.nan
    out, state = solve(p, xy, lengths, t)
    np.testing.assert_equal(out[:, 16], true[:, 16])
    np.testing.assert_equal(state[9:15, 14], TEMPORAL)
    for (a, b), length in lengths.items():
        np.testing.assert_allclose(np.linalg.norm(out[9:15, b] - out[9:15, a], axis=1), length, atol=1e-6)


def test_image_excursion_onto_torso_does_not_split_a_slow_wrist_path():
    from really_rally.core.arm_flow import _reject_excursions

    times = np.arange(15) / 60
    wrist = np.stack((times * 60, np.full(len(times), 150)), axis=1)[:, None]
    wrist[5:8, 0, 1] -= 55
    _reject_excursions(wrist, times, np.full(len(times), 80), np.r_[False, np.ones(14, bool)])
    assert np.isnan(wrist[5:8]).all()
    assert np.isfinite(wrist[[4, 8]]).all()


def test_optical_cues_follow_visible_pixels_across_a_detector_gap():
    import cv2
    from really_rally.core.arm_flow import arm_flow

    t, pose, xy, _ = swing()
    rng = np.random.default_rng(4)
    texture = rng.integers(0, 255, (17, 17), dtype=np.uint8)
    frames = []
    image = xy.copy()
    wrist = np.stack((110 + np.arange(len(t)), np.full(len(t), 125)), axis=1)
    image[:, 16, :2] = wrist / [300, 300]
    vis = np.isfinite(pose[..., 0]).astype(float)
    vis[10:15, 16] = 0
    image[10:15, 16, :2] = np.nan
    for x, y in wrist:
        gray = np.full((300, 300), 50, np.uint8)
        gray[y - 8 : y + 9, x - 8 : x + 9] = texture
        frames.append(cv2.imencode(".jpg", gray)[1])
    hands = np.full((len(t), 2, 21, 3), np.nan)
    flow = arm_flow(frames, [0, 0, 300, 300], [300, 300], t, (image, vis), (image, vis), hands)
    np.testing.assert_allclose(flow[10:15, 3], wrist[10:15], atol=1)


def test_clip_edge_prediction_needs_weak_image_agreement_and_has_a_time_limit():
    from really_rally.core.continuity import BOUNDARY

    t, true, xy, lengths = swing()
    camera = (np.full(len(t), 200.0), np.full((len(t), 2), 150.0))
    for n, shift, expected in [(4, 0.0, BOUNDARY), (4, 0.4, MISSING), (12, 0.0, MISSING)]:
        p = true.copy()
        p[-n:, 16] = np.nan
        weak = np.isfinite(true[..., 0]).astype(float)
        weak[-n:, 16] = 0.25
        other = xy.copy()
        other[-n:, 16, 0] += shift
        reverse = (p, p, other, weak)
        _, state = complete_pose(p, p, lengths, xy, weak, t, [300, 300], reverse, camera=camera)
        np.testing.assert_equal(state[-n:, 16], expected)
