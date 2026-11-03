"""Known geometry exercises the angle conventions and gaps used by the tennis replay."""
import numpy as np
import pytest

from really_rally.core.stroke_analysis import fit_body, measurements
from really_rally.core.reconstruction import _wrist_observations


def _pose():
    pose = np.full((33, 3), np.nan)
    pose[[23, 24]] = [[-.15, 0, 0], [.15, 0, 0]]
    pose[[11, 12]] = [[-.2, .5, 0], [.2, .5, 0]]
    pose[[14, 16]] = [[.45, .5, 0], [.45, .75, 0]]
    return pose


def _measure(points, times):
    frames = np.arange(len(points))
    pixels = np.zeros((len(points), 33, 2))
    return measurements(points, points, {}, frames, np.array(times), 0, pixels)


def test_elbow_direction_flexion_signed_twist_and_rigid_transform():
    pose = _pose()
    neutral = _measure(pose[None], [0])[0]
    assert neutral["right_elbow_flexion_deg"] == pytest.approx(90)
    assert neutral["right_elbow_elevation_deg"] == pytest.approx(0)
    assert neutral["right_elbow_azimuth_deg"] == pytest.approx(0)
    theta = np.radians(30)
    yaw = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    pose[[11, 12, 14, 16]] = pose[[11, 12, 14, 16]] @ yaw.T
    roll = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    measured = _measure((pose @ roll.T + [3, 2, -1])[None], [0])[0]
    assert measured["trunk_separation_deg"] == pytest.approx(30)
    assert measured["right_elbow_azimuth_deg"] == pytest.approx(30)


def test_wrist_velocity_uses_source_time_and_does_not_bridge_an_occlusion():
    poses = np.repeat(_pose()[None], 5, axis=0)
    poses[:, 16, 0] += [0, .04, .08, .12, .16]
    poses[2, 16] = np.nan
    rows = _measure(poses, [0, .02, .04, .06, .09])
    assert rows[1]["right_wrist_speed_m_s"] == pytest.approx(2)
    assert rows[2]["right_wrist_x_m"] is None and rows[3]["right_wrist_speed_m_s"] is None
    assert rows[4]["right_wrist_speed_m_s"] == pytest.approx(.04/.03)


def test_fit_keeps_missing_wrist_and_reduces_screen_and_length_error():
    poses = np.repeat(_pose()[None], 10, axis=0)
    pixels = poses[..., :2] * 200 + [250, 180]
    poses[-1, 16, 0] += .07
    poses[0, 16] = np.nan
    fit, scale, offset, typical = fit_body(poses, pixels, np.isfinite(poses).all(axis=2))
    assert np.isnan(fit[0, 16]).all()
    assert np.linalg.norm(fit[-1, 16, :2] * scale[-1] + offset[-1] - pixels[-1, 16]) < 14
    assert abs(np.linalg.norm(fit[-1, 16]-fit[-1, 14])-typical[14, 16]) < abs(np.linalg.norm(poses[-1, 16]-poses[-1, 14])-.25)


def test_crossed_wrists_cannot_share_a_hand_detection():
    image = np.zeros((2, 33, 3))
    image[:, [23, 24], 1] = .4
    image[:, 15, :2] = [.4, .5]
    image[0, 16, :2] = [.401, .5]
    image[1, 16, :2] = [.8, .5]
    hands = np.full((2, 2, 21, 3), np.nan)
    hands[:, 0] = [.4, .5, 0]
    _, accepted = _wrist_observations(image, hands, np.ones((2, 33)), [100, 200])
    np.testing.assert_array_equal(accepted[:, 0], [False, True])


def test_review_knee_flexion_and_unusable_links():
    pose = _pose()
    pose[[25, 27]] = [[-.15, -.3, 0], [-.15, -.3, .3]]
    frames = np.repeat(pose[None], 2, axis=0)
    frames[1, 27] = np.nan
    rows = _measure(frames, [0, .02])
    assert rows[0]["left_knee_flexion_deg"] == pytest.approx(90)
    assert rows[0]["left_knee_quality"] == "usable"
    assert rows[1]["left_knee_flexion_deg"] is None
    assert rows[1]["left_knee_quality"] == "missing_landmarks"
    assert rows[0]["pelvis_heading_deg"] == pytest.approx(0)
    # A camera-vertical segment has no identifiable horizontal heading.
    pose[24] = pose[23] + [0, .3, 0]
    assert _measure(pose[None], [0])[0]["pelvis_heading_deg"] is None


def test_shoe_fits_projected_foot_direction_without_stretching_the_triangle():
    pose = _pose()
    pose[[25, 26, 27, 28]] = [[-.15, -.4, 0], [.15, -.4, 0], [-.15, -.8, 0], [.15, -.8, 0]]
    pose[[29, 31]] = [[-.18, -.84, -.02], [-.06, -.85, .14]]
    poses = np.repeat(pose[None], 6, axis=0)
    pixels = poses[..., :2] * 150 + [200, 175]
    ends = [29, 31]
    theta = .24
    turn = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pixels[:, ends] = (pixels[:, ends] - pixels[:, 27, None]) @ turn.T + pixels[:, 27, None]
    fit, scale, offset, _ = fit_body(poses, pixels, np.isfinite(poses).all(axis=2))
    def err(p):
        return np.linalg.norm(p[:, ends, :2]*scale[:, None, None] + offset[:, None] - pixels[:, ends])
    assert err(fit) < .25 * err(poses)
    for a, b in ((27, 29), (27, 31), (29, 31)):
        np.testing.assert_allclose(np.linalg.norm(fit[:, a] - fit[:, b], axis=1),
                                   np.linalg.norm(poses[:, a] - poses[:, b], axis=1))
