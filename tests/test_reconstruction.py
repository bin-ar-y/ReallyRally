"""Coordinate and missing-data contracts of the portable animated model."""
import json
import struct

import numpy as np

from really_rally.core.floor import floor_track, sole_heights
from really_rally.core.mannequin import parts
from really_rally.core.motion_model import write_model
from really_rally.core.reconstruction import stabilize


def test_an_invisible_wrist_does_not_pull_neighboring_estimates():
    positions = np.ones((3, 33, 3))
    visibility = np.ones((3, 33))
    positions[1, 16] = 999
    visibility[1, 16] = .1
    filtered = stabilize(positions, visibility)
    np.testing.assert_equal(filtered[[0, 2], 16], 1)
    assert np.isnan(filtered[1, 16]).all()
    assert positions[1, 16, 0] == 999


def test_glb_tracks_recover_each_observed_bone_and_hide_missing_frames(tmp_path):
    points = np.full((3, 33, 3), np.nan)
    points[:2, 11] = [0, 0, 0]
    points[:2, 13] = [[1, 0, 0], [0, -1, 0]]
    path = tmp_path / "player.glb"
    write_model(path, points, np.array([0., .02, .045]))
    glb = path.read_bytes()
    magic, version, total, size = struct.unpack_from("<4sIII", glb)
    assert (magic, version, total) == (b"glTF", 2, len(glb))
    doc = json.loads(glb[20:20 + size])
    binary = glb[28 + size:]

    def values(accessor):
        entry = doc["accessors"][accessor]
        view = doc["bufferViews"][entry["bufferView"]]
        return np.frombuffer(binary[view["byteOffset"]:][:view["byteLength"]], dtype="<f4").reshape(entry["count"], -1)

    animation = doc["animations"][0]
    channels = {c["target"]["path"]: animation["samplers"][c["sampler"]] for c in animation["channels"]
                if doc["nodes"][c["target"]["node"]]["name"] == "Left upper arm"}
    np.testing.assert_allclose(values(channels["scale"]["input"]).ravel(), [0, .02, .045])
    center = values(channels["translation"]["output"])
    scale = values(channels["scale"]["output"])
    rotation = values(channels["rotation"]["output"])
    for i in (0, 1):
        # Rotate local Y by the exported quaternion and recover both endpoints.
        xyz, w = rotation[i, :3], rotation[i, 3]
        axis = np.array([0., 1., 0.])
        axis += 2 * np.cross(xyz, np.cross(xyz, axis) + w * axis)
        np.testing.assert_allclose(center[i] - axis * scale[i, 1], points[i, 11], atol=1e-6)
        np.testing.assert_allclose(center[i] + axis * scale[i, 1], points[i, 13], atol=1e-6)
    assert all(sampler["interpolation"] == "LINEAR" for sampler in channels.values())
    child = next(i for i, node in enumerate(doc["nodes"]) if node["name"] == "Left upper arm")
    parent = next(i for i, node in enumerate(doc["nodes"]) if child in node.get("children", []))
    gate = next(animation["samplers"][c["sampler"]] for c in animation["channels"] if c["target"]["node"] == parent)
    assert gate["interpolation"] == "STEP"
    np.testing.assert_equal(values(gate["output"])[:, 0], [1, 1, 0])
    np.testing.assert_equal(center[2], center[1])


def test_rib_cage_follows_shoulders_independently_of_hips_and_missing_face():
    pose = np.full((2, 33, 3), np.nan)
    pose[:, [23, 24]] = [[.1, 0, 0], [-.1, 0, 0]]
    pose[:, [11, 12]] = [[0, .5, -.16], [0, .5, .16]]
    pose[:, [7, 8, 0]] = [[.07, .7, 0], [-.07, .7, 0], [0, .7, .1]]
    pose[1, 0] = np.nan
    tracks = {name: transform for _, name, transform in parts(pose)}
    for name, expected in [("Rib cage", [0, 0, -1]), ("Pelvis", [1, 0, 0])]:
        _, rotations, scale = tracks[name]
        xyz, w = rotations[0, :3], rotations[0, 3]
        lateral = np.array([1., 0., 0.])
        lateral += 2 * np.cross(xyz, np.cross(xyz, lateral) + w * lateral)
        np.testing.assert_allclose(lateral, expected, atol=1e-6)
        assert (scale > 0).all()
    head_scale = tracks["Head"][2]
    assert (head_scale[0] > 0).all()
    np.testing.assert_equal(head_scale[1], 0)


def test_floor_uses_mesh_sole_preserves_lift_and_refuses_a_lone_raised_shoe():
    pose = np.full((4, 33, 3), np.nan)
    for side in (0, 1):
        pose[:, [27+side, 29+side, 31+side]] = [
            [side*.18, -.60, 0], [side*.18, -.68, -.03], [side*.18, -.70, .14]]
    pose[1:3, [28, 30, 32], 1] += .20
    pixels = pose[..., :2]*[150, -150] + [150, 150]
    pose[2, [27, 29, 31]] = np.nan
    pose[3] = np.nan
    soles = sole_heights(pose)
    height, contact = floor_track(pose, pixels, [400, 400], np.arange(4)*.02)
    assert contact[1] == 0
    np.testing.assert_allclose(height[1] + soles[1], [0, .20], atol=1e-6)
    np.testing.assert_equal(contact[2:], -1)
    np.testing.assert_equal(height[2:], height[1])
    # No contact inference when feet are outside the recording.
    _, clipped = floor_track(pose, pixels, [50, 50], np.arange(4)*.02)
    np.testing.assert_equal(clipped, -1)


def test_small_swing_across_reference_axis_does_not_flip_the_limb():
    pose = np.full((3, 33, 3), np.nan)
    pose[:, 11] = 0
    for i, x in enumerate([.84, .85, .86]):
        pose[i, 13] = [x, np.sqrt(1-x*x), 0]
    _, quats, _ = next(t for _, name, t in parts(pose) if name == "Left upper arm")
    dots = abs((quats[:-1]*quats[1:]).sum(axis=1))
    assert (np.degrees(2*np.arccos(np.clip(dots, 0, 1))) < 2).all()


def test_temporal_smoothing_does_not_blend_across_a_recording_gap():
    positions = np.zeros((4, 33, 3))
    positions[2:] = 1
    times = np.array([0, .02, .6, .62])
    filtered = stabilize(positions, np.ones((4, 33)), times)
    np.testing.assert_equal(filtered, positions)
