"""Portable glTF 2.0 animation of the observed body segments, in estimated meters."""
import json
import struct

import numpy as np

from really_rally.core.mannequin import parts, surface

# Name, proximal landmark, distal landmark, display radius (meters).
LIMBS = [
    ("Shoulders", 11, 12, 0.012), ("Pelvis", 23, 24, 0.012),
    ("Left torso", 11, 23, 0.012), ("Right torso", 12, 24, 0.012),
    ("Left upper arm", 11, 13, 0.012), ("Right upper arm", 12, 14, 0.012),
    ("Left forearm", 13, 15, 0.009), ("Right forearm", 14, 16, 0.009),
    ("Left hand", 15, 19, 0.008), ("Right hand", 16, 20, 0.008),
    ("Left thigh", 23, 25, 0.012), ("Right thigh", 24, 26, 0.012),
    ("Left shin", 25, 27, 0.012), ("Right shin", 26, 28, 0.012),
    ("Left foot", 29, 31, 0.011), ("Right foot", 30, 32, 0.011),
    ("Left heel", 27, 29, 0.007), ("Right heel", 28, 30, 0.007),
    ("Left thumb", 15, 21, 0.006), ("Right thumb", 16, 22, 0.006),
    ("Left pinky", 15, 17, 0.006), ("Right pinky", 16, 18, 0.006),
    ("Left face", 7, 0, 0.008), ("Right face", 8, 0, 0.008),
    ("Head", 7, 8, 0.009),
]


def _clock(times):
    # Hold the last sample over a recording discontinuity in every GLB viewer.
    times = np.asarray(times, dtype=np.float32)
    limit = min(.10, 2.5 * np.median(np.diff(times))) if len(times) > 1 else .10
    clock, source = [], []
    for i, t in enumerate(times):
        if i and t - times[i-1] > limit:
            clock.append(np.nextafter(t, np.float32(0)))
            source.append(i-1)
        clock.append(t)
        source.append(i)
    return clock, source


def write_model(path, points, times, root_y=None):
    binary = bytearray()
    views, accessors = [], []

    def add(values, kind, indices=False, bounds=False):
        array = np.asarray(values, dtype="<u2" if indices else "<f4")
        binary.extend(b"\0" * (-len(binary) % 4))
        views.append(dict(buffer=0, byteOffset=len(binary), byteLength=array.nbytes))
        binary.extend(array.tobytes())
        entry = dict(bufferView=len(views) - 1, componentType=5123 if indices else 5126,
                     count=len(array), type=kind)
        if bounds:
            entry["min"] = np.atleast_1d(array.min(axis=0)).tolist()
            entry["max"] = np.atleast_1d(array.max(axis=0)).tolist()
        accessors.append(entry)
        return len(accessors) - 1

    keys, samples = _clock(times)
    clock = add(keys, "SCALAR", bounds=True)
    materials = [dict(name="Pearl ceramic", pbrMetallicRoughness=dict(
        baseColorFactor=[.84, .89, .94, 1], metallicFactor=.06, roughnessFactor=.39))]
    meshes, mesh_ids = [], {}
    nodes, samplers, channels, roots = [], [], [], []

    def track(node_id, prop, kind, values, interpolation):
        nodes[node_id][prop] = values[0].tolist()
        channels.append(dict(sampler=len(samplers), target=dict(node=node_id, path=prop)))
        samplers.append(dict(input=clock, output=add(values[samples], kind), interpolation=interpolation))
    for shape, name, transforms in parts(points):
        if shape not in mesh_ids:
            vertices, normals, faces = surface(shape)
            primitive = dict(attributes=dict(POSITION=add(vertices, "VEC3", bounds=True),
                                             NORMAL=add(normals, "VEC3")),
                             indices=add(faces, "SCALAR", indices=True), material=0)
            mesh_ids[shape] = len(meshes)
            meshes.append(dict(name=shape, primitives=[primitive]))
        child = len(nodes)
        nodes.append(dict(name=name, mesh=mesh_ids[shape]))
        pos, rotation, scale = transforms
        visible = (scale > 0).all(axis=1)
        if root_y is not None:
            pos[:, 1] += root_y
        if not visible.all():
            # A STEP gate preserves real observation gaps. Continuous child
            # transforms never shrink a limb or fly to the origin on dropout.
            gate = len(nodes)
            nodes.append(dict(name=f"{name} visibility", children=[child]))
            track(gate, "scale", "VEC3", np.repeat(visible[:, None], 3, axis=1).astype(float), "STEP")
            roots.append(gate)
            known = np.flatnonzero(visible)
            if len(known):
                previous = known[np.maximum(0, known.searchsorted(np.arange(len(pos)), side="right") - 1)]
                pos, rotation, scale = pos[previous], rotation[previous], scale[previous]
        else:
            roots.append(child)
        # Adjacent quaternions use the same hemisphere, including across half turns.
        flips = np.where((rotation[1:] * rotation[:-1]).sum(axis=1) < 0, -1, 1)
        rotation = rotation * np.r_[1, np.cumprod(flips)][:, None]
        track(child, "translation", "VEC3", pos, "LINEAR")
        track(child, "rotation", "VEC4", rotation, "LINEAR")
        track(child, "scale", "VEC3", scale, "LINEAR")
    doc = dict(asset=dict(version="2.0", generator="ReallyRally"), scene=0,
               scenes=[dict(nodes=roots)], nodes=nodes, meshes=meshes, materials=materials,
               animations=[dict(name="Stroke", channels=channels, samplers=samplers)],
               buffers=[dict(byteLength=len(binary))], bufferViews=views, accessors=accessors)
    header = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")
    header += b" " * (-len(header) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, 28 + len(header) + len(binary)))
        for kind, data in ((b"JSON", header), (b"BIN\0", binary)):
            stream.write(struct.pack("<I4s", len(data), kind))
            stream.write(data)
