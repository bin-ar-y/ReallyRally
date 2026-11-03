"""Native preview stays aligned to source frames and keeps estimate provenance."""

import csv
import json
from zipfile import ZipFile

import cv2
import numpy as np

from really_rally.core.motion_model import write_model
from really_rally.studio.motion_data import MotionData


def test_native_clip_uses_its_own_images_glb_and_timestamped_metrics(tmp_path):
    frames = np.array([123, 124, 125])
    times = np.array([1.4, 1.44, 1.5])
    points = np.full((3, 33, 3), np.nan)
    points[:, [11, 12, 23, 24]] = [[0.2, -0.45, 0], [-0.2, -0.45, 0], [0.12, 0, 0], [-0.12, 0, 0]]
    points[:, 14] = [[-0.25, -0.22, 0], [-0.31, -0.30, 0], [-0.33, -0.4, 0.06]]
    points[:, 16] = [[-0.32, -0.1, 0], [-0.5, -0.32, 0.02], [-0.48, -0.55, 0.09]]
    status = np.isfinite(points[..., 0]).astype(int)
    status[1, 16] = 3
    meta = dict(impact_frame=124, source="/missing/original.mov", source_size=[100, 80])
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    np.savez(
        tmp_path / "pose.npz",
        frame=frames,
        timestamp=times,
        trajectory_world=points,
        trajectory_status=status,
        pixels_per_meter=np.full(3, 80),
        pixel_offset=np.tile([50, 40], (3, 1)),
    )
    write_model(tmp_path / "player.glb", points * [1, -1, -1], times - times[0])
    for name, values in [("kinematics", ["10", "", "40"]), ("trajectory", ["12", "26", "43"])]:
        with (tmp_path / f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "right_elbow_flexion_deg"])
            writer.writerows(zip(frames, values))
    with ZipFile(tmp_path / "frames.zip", "w") as z:
        for i, n in enumerate(frames):
            img = np.full((80, 100, 3), 40 + i * 40, np.uint8)
            z.writestr(f"{n}.jpg", cv2.imencode(".jpg", img)[1].tobytes())
    data = MotionData(tmp_path)
    assert data.impact == 1
    vals, estimated, unit = data.series("Right", "Elbow flexion")
    np.testing.assert_equal(vals, [10, 26, 40])
    np.testing.assert_equal(estimated, [False, True, False])
    assert unit == "°"
    index, source, model = data.draw(data.times[1], (270, 230), 0, 0, "Right", False)
    assert index == 1 and source[0, 0, 0] == 80
    # An in-between display time still shows the same camera frame but animates the model.
    index, source2, between = data.draw(0.065, (270, 230), 0, 0, "Right", False)
    assert index == 1 and np.array_equal(source, source2) and not np.array_equal(between, model)
    _, _, side = data.draw(0.065, (270, 230), 1.4, 0, "Right", False)
    assert not np.array_equal(between, side)
