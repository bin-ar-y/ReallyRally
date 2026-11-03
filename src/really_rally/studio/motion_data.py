"""Desktop replay data: source frames, the exported mannequin and measured swing data."""

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from really_rally.studio.body_renderer import BodyRenderer

METRICS = {
    "Elbow flexion": ("elbow_flexion_deg", "°"),
    "Arm elevation": ("elbow_elevation_deg", "°"),
    "Arm azimuth": ("elbow_azimuth_deg", "°"),
    "Trunk separation": ("trunk_separation_deg", "°"),
    "Wrist speed": ("wrist_speed_m_s", " m/s"),
    "Knee flexion": ("knee_flexion_deg", "°"),
}
SOURCES = (
    "Unavailable",
    "Forward fit",
    "Reverse fit",
    "Temporal estimate",
    "Image-guided estimate",
    "Boundary prediction",
)


def discover(folder):
    folder = Path(folder)
    paths = [folder] if (folder / "pose.npz").is_file() else [p.parent for p in sorted(folder.rglob("pose.npz"))]
    return [p for p in paths if (p / "metadata.json").is_file() and (p / "player.glb").is_file()]


def _csv_rows(path, frames):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["frame"]) for row in rows] != list(frames):
        raise ValueError(f"Frame numbers in {path.name} do not match the pose archive.")
    return rows


def _source_images(folder, meta, frames):
    if (folder / "frames.zip").is_file():
        with ZipFile(folder / "frames.zip") as archive:
            return [archive.read(f"{n}.jpg") for n in frames]
    # Early desktop bundles kept their source frames only in the original video.
    source = Path(meta["source"])
    if not source.is_file():
        return []
    cap = cv2.VideoCapture(str(source))
    images = []
    try:
        for i, n in enumerate(frames):
            if not i or n != frames[i - 1] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
            ok, image = cap.read()
            if not ok:
                return []
            h, w = image.shape[:2]
            ratio = min(1, 640 / max(w, h))
            if ratio < 1:
                image = cv2.resize(
                    image, (max(1, round(w * ratio)), max(1, round(h * ratio))), interpolation=cv2.INTER_AREA
                )
            ok, jpeg = cv2.imencode(".jpg", image)
            if not ok:
                return []
            images.append(jpeg.tobytes())
    finally:
        cap.release()
    return images


class MotionData:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
        with np.load(self.folder / "pose.npz", allow_pickle=False) as pose:
            self.frames = pose["frame"]
            self.times = pose["timestamp"]
            self.times = self.times - self.times[0]
            if not np.isfinite(self.times).all() or (np.diff(self.times) <= 0).any():
                raise ValueError("The saved clip has invalid timestamps.")
            self.points = pose["trajectory_world" if "trajectory_world" in pose else "fitted_world"] * [1, -1, -1]
            if self.points.shape != (len(self.frames), 33, 3) or len(self.times) != len(self.frames):
                raise ValueError("The pose archive does not match the saved clip.")
            self.status = (
                pose["trajectory_status"]
                if "trajectory_status" in pose
                else np.isfinite(self.points).all(axis=2).astype(int)
            )
            self.root = pose.get("display_root_y", np.zeros(len(self.frames)))
            self.support = pose.get("support_foot", np.full(len(self.frames), -1))
            self.image_points = (
                self.points[..., :2] * [1, -1] * pose["pixels_per_meter"][:, None, None] + pose["pixel_offset"][:, None]
            )
        self.impact = list(self.frames).index(self.meta["impact_frame"])
        self.source = _source_images(self.folder, self.meta, self.frames)
        self.measured = _csv_rows(self.folder / "kinematics.csv", self.frames)
        path = self.folder / "trajectory.csv"
        self.curves = _csv_rows(path, self.frames) if path.is_file() else self.measured
        self.body = BodyRenderer(self.folder / "player.glb")
        actual = self.root[self.support >= 0]
        self.center_y = float(np.median(actual)) if actual.size else 0.0
        self.points[:, :, 1] += self.root[:, None]
        lengths = np.linalg.norm(self.points - [0, self.center_y, 0], axis=2)
        self.extent = max(1.02, np.max(lengths[np.isfinite(lengths)], initial=0))
        self.gap_limit = min(0.1, 2.5 * np.median(np.diff(self.times))) if len(self.times) > 1 else 0
        self._series = {}

    def series(self, side, name):
        key = (side, name)
        if key not in self._series:
            field, unit = METRICS[name]
            left = side == "Left"
            arm = [11, 13, 15] if left else [12, 14, 16]
            ids = {
                "Trunk separation": [11, 12, 23, 24],
                "Knee flexion": [23, 25, 27] if left else [24, 26, 28],
                "Arm azimuth": [*arm[:2], 11, 12, 23, 24],
                "Arm elevation": [*arm[:2], 11, 12, 23, 24],
                "Wrist speed": [*arm, 23, 24],
            }.get(name, arm)
            column = field if name == "Trunk separation" else side.lower() + "_" + field
            status = self.status[:, ids]
            estimated = (status > 1).any(axis=1)
            missing = (status == 0).any(axis=1)
            if name == "Wrist speed":
                estimated[1:] |= estimated[:-1].copy()
                missing[1:] |= missing[:-1].copy()
            values = np.full(len(self.frames), np.nan)
            for i in np.flatnonzero(~missing):
                val = (self.curves if estimated[i] else self.measured)[i].get(column)
                if val:
                    values[i] = float(val)
            self._series[key] = values, estimated, unit
        return self._series[key]

    def draw(self, seconds, size, yaw, pitch, side, trace, trail=True):
        i = max(0, self.times.searchsorted(seconds, side="right") - 1)
        j = min(i + 1, len(self.frames) - 1)
        gap = self.times[j] - self.times[i]
        mix = min(1, (seconds - self.times[i]) / gap) if gap > 0 else 0.0
        ids = [11, 13, 15] if side == "Left" else [12, 14, 16]
        lines = []
        if trace:
            for a, b in zip(ids[:-1], ids[1:]):
                ends = self.points[i, [a, b]].copy()
                if not np.isfinite(ends).all():
                    continue
                if mix and np.isfinite(self.points[j, [a, b]]).all() and gap <= self.gap_limit:
                    ends += mix * (self.points[j, [a, b]] - ends)
                lines.append((*ends, np.any(self.status[i, [a, b]] > 1), False))
        if trail:
            begin = self.times.searchsorted(seconds - 0.25)
            for k in range(begin + 1, i + 1):
                ids_wrist = ids + [23, 24]
                rows = self.status[k - 1 : k + 1, ids_wrist]
                positions = self.points[k - 1 : k + 1, ids[-1]]
                if (
                    (rows == 0).any()
                    or not np.isfinite(positions).all()
                    or self.times[k] - self.times[k - 1] > self.gap_limit
                ):
                    continue
                usable = all(
                    (self.curves if (self.status[n, ids_wrist] > 1).any() else self.measured)[n].get(
                        side.lower() + "_wrist_x_m"
                    )
                    for n in (k - 1, k)
                )
                if usable:
                    lines.append((*positions, (rows > 1).any(), True))
        body = self.body.render(seconds, size, yaw, pitch, self.center_y, self.extent, self.support[i] >= 0, lines)
        image = None
        if self.source:
            image = cv2.imdecode(np.frombuffer(self.source[i], np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if trace:
                    ratio = np.array([image.shape[1], image.shape[0]]) / self.meta["source_size"]
                    for a, b in zip(ids[:-1], ids[1:]):
                        points = self.image_points[i, [a, b]] * ratio
                        if np.isfinite(points).all():
                            cv2.line(image, *points.astype(int), (244, 176, 78), 2, cv2.LINE_AA)
        return i, image, body
