"""Orthographic, depth-tested rendering of ReallyRally's animated GLB in a Tk image."""

import json
import struct

import cv2
import numpy as np


def _rotation(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class BodyRenderer:
    """Read the meshes, TRS animation channels and visibility gates emitted by write_model."""

    def __init__(self, file):
        raw = file.read_bytes()
        magic, version, total, size, kind = struct.unpack_from("<4sIII4s", raw)
        if (magic, version, total, kind) != (b"glTF", 2, len(raw), b"JSON"):
            raise ValueError("Invalid animated model")
        doc = json.loads(raw[20 : 20 + size])
        binary = memoryview(raw)[28 + size :]

        def values(index):
            a = doc["accessors"][index]
            view = doc["bufferViews"][a["bufferView"]]
            dtype = {5123: "<u2", 5126: "<f4"}[a["componentType"]]
            offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
            return np.frombuffer(binary[offset : offset + view["byteLength"]], dtype=dtype).reshape(a["count"], -1)

        meshes = []
        for mesh in doc["meshes"]:
            prim = mesh["primitives"][0]
            meshes.append(
                (
                    values(prim["attributes"]["POSITION"]),
                    values(prim["attributes"]["NORMAL"]),
                    values(prim["indices"]).reshape(-1, 3),
                )
            )
        anim = doc["animations"][0]
        self.times = values(anim["samplers"][0]["input"]).ravel()
        tracks = [{} for _ in doc["nodes"]]
        for channel in anim["channels"]:
            sampler = anim["samplers"][channel["sampler"]]
            target = channel["target"]
            tracks[target["node"]][target["path"]] = (values(sampler["output"]), sampler["interpolation"])
        gates = {
            child: tracks[i]["scale"][0][:, 0]
            for i, node in enumerate(doc["nodes"])
            for child in node.get("children", [])
        }
        self.parts = [
            (meshes[node["mesh"]], tracks[i], gates.get(i)) for i, node in enumerate(doc["nodes"]) if "mesh" in node
        ]

    def render(self, timestamp, size, yaw, pitch, center_y, extent, floor, trace=()):
        w, h = size
        # Y is up and positive camera depth is nearer the reviewer.
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        camera = np.array([[cy, 0, sy], [sy * sp, cp, -cy * sp], [-sy * cp, sp, cy * cp]])
        scale = 0.46 * min(w, h) / extent

        def project(points):
            coords = (points - [0, center_y, 0]) @ camera.T
            coords[:, 0] = w / 2 + scale * coords[:, 0]
            coords[:, 1] = h / 2 - scale * coords[:, 1]
            return coords

        image = np.empty((h, w, 3), np.uint8)
        image[:] = np.linspace([32, 58, 77], [14, 35, 49], h)[:, None]
        if floor:
            for step in np.linspace(-1, 1, 9):
                for line in ([[step, 0, -1], [step, 0, 1]], [[-1, 0, step], [1, 0, step]]):
                    a, b = project(np.array(line))[:, :2].astype(int)
                    cv2.line(image, a, b, (48, 75, 87), 1, cv2.LINE_AA)
        t = np.float32(timestamp)
        lo = max(0, self.times.searchsorted(t, side="right") - 1)
        hi = min(lo + 1, len(self.times) - 1)
        mix = np.clip((t - self.times[lo]) / (self.times[hi] - self.times[lo]), 0, 1) if hi > lo else 0
        points, light, faces = [], [], []
        start = 0
        for (v, n, f), tracks, gate in self.parts:
            if gate is not None and not gate[lo]:
                continue

            def sample(prop, tracks=tracks):
                arr, mode = tracks[prop]
                a = arr[lo]
                if not mix or mode == "STEP":
                    return a
                b = arr[hi]
                if prop == "rotation":
                    dot = np.dot(a, b)
                    angle = np.arccos(min(1, abs(dot)))
                    b = -b if dot < 0 else b
                    if angle > 0.001:
                        return (np.sin((1 - mix) * angle) * a + np.sin(mix * angle) * b) / np.sin(angle)
                return (1 - mix) * a + mix * b

            s = sample("scale")
            if np.min(s) <= 0:
                continue
            rot = _rotation(sample("rotation"))
            points.append(project((v * s) @ rot.T + sample("translation")))
            normals = (n / s) @ rot.T @ camera.T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
            light.append(
                np.clip(
                    0.16
                    + 0.72 * np.maximum(0, normals @ [-0.44, 0.72, 0.54])
                    + 0.23 * np.maximum(0, normals @ [-0.25, 0.41, 0.875]) ** 18,
                    0,
                    1,
                )
            )
            faces.append(f.astype(int) + start)
            start += len(v)
        if points:
            _raster(image, np.concatenate(points), np.concatenate(light), np.concatenate(faces))
        for a, b, estimated, trail in trace:
            a, b = project(np.array([a, b]))[:, :2]
            line = b - a
            color = (246, 213, 113) if trail else (249, 179, 86)
            pieces = max(1, int(np.linalg.norm(line) / 4)) if estimated else 1
            for k in range(0, pieces, 2):
                cv2.line(
                    image,
                    tuple((a + k / pieces * line).astype(int)),
                    tuple((a + (k + 1) / pieces * line).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if not trail:
                cv2.circle(image, tuple(b.astype(int)), 3, (255, 191, 103), -1, cv2.LINE_AA)
        return image


def _raster(image, vertices, shade, faces):
    """Evaluate barycentric coordinates for triangle bounding boxes in NumPy batches."""
    height, width = image.shape[:2]
    depth = np.full(width * height, -np.inf)
    pixels = image.reshape(-1, 3)
    for batch in np.array_split(faces, max(1, (len(faces) + 799) // 800)):
        triangles = vertices[batch]
        a, b, c = triangles.transpose(1, 0, 2)
        u, v = b - a, c - a
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        low = np.maximum(0, np.ceil(triangles[:, :, :2].min(axis=1) - 0.5)).astype(int)
        high = np.minimum([width - 1, height - 1], np.floor(triangles[:, :, :2].max(axis=1) - 0.5)).astype(int)
        span = high - low + 1
        keep = (det < -1e-7) & (span > 0).all(axis=1)
        if not keep.any():
            continue
        count = span[keep].prod(axis=1)
        idx = np.repeat(np.flatnonzero(keep), count)
        offsets = np.arange(len(idx)) - np.repeat(np.cumsum(count) - count, count)
        x = low[idx, 0] + offsets % span[idx, 0]
        y = low[idx, 1] + offsets // span[idx, 0]
        dx, dy = x + 0.5 - a[idx, 0], y + 0.5 - a[idx, 1]
        wb = (dx * v[idx, 1] - dy * v[idx, 0]) / det[idx]
        wc = (dy * u[idx, 0] - dx * u[idx, 1]) / det[idx]
        wa = 1 - wb - wc
        inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
        k = idx[inside]
        dst = (y * width + x)[inside]
        z = a[k, 2] + wb[inside] * u[k, 2] + wc[inside] * v[k, 2]
        np.maximum.at(depth, dst, z)
        front = z == depth[dst]
        weights = np.stack((wa[inside], wb[inside], wc[inside]), axis=1)[front]
        intensity = (shade[batch[k[front]]] * weights).sum(axis=1)
        pixels[dst[front]] = [100, 125, 147] + intensity[:, None] * [155, 130, 105]
