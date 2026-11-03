"""Pose a neutral, articulated human surface from the visible fitted landmarks."""
import numpy as np


# Stations along each anatomical part: axial position, half-width, half-depth,
# and forward offset. Limb stations extend across joint centers for a rounded hinge.
PROFILES = {
    "chest": [(0, 0, 0, 0), (.05, .63, .63, 0), (.19, .72, .78, 0),
              (.48, .91, .99, .04), (.7, 1, 1, .08), (.84, .99, .88, .06),
              (.91, .86, .68, .03), (.97, .52, .51, 0), (1, 0, 0, 0)],
    "pelvis": [(0, 0, 0, .05), (.1, .54, .54, .02), (.28, .86, .9, -.06),
               (.47, 1, 1, -.12), (.68, .96, .94, -.08), (.86, .87, .81, 0),
               (.97, .74, .67, .02), (1, 0, 0, 0)],
    "head": [(0, 0, 0, .22), (.06, .47, .50, .27), (.19, .74, .78, .18),
             (.35, .88, .90, .11), (.54, .99, .99, .03), (.69, 1, 1, -.03),
             (.83, .91, .92, -.05), (.94, .65, .69, -.04), (.985, .32, .35, -.03), (1, 0, 0, -.03)],
    "neck": [(-.10, 0, 0, 0), (0, 1, 1, 0), (.35, .84, .88, 0),
             (.75, .77, .86, 0), (1.06, .84, .9, .06), (1.14, 0, 0, 0)],
    "shoulder": [(-.12, 0, 0, 0), (-.01, .68, .70, 0), (.19, .90, .91, 0),
                 (.48, .99, 1, .02), (.72, 1, .92, .02), (.91, .75, .68, 0),
                 (1.10, 0, 0, 0)],
    "ankle": [(-.24, 0, 0, 0), (-.04, .73, .66, 0), (.23, .82, .78, 0),
              (.56, .89, .93, 0), (.80, 1, 1, 0), (1.22, 0, 0, 0)],
    "finger": [(-.13, 0, 0, 0), (.01, .85, .90, 0), (.35, 1, 1, 0),
               (.68, .88, .85, 0), (.89, .73, .63, 0), (1, 0, 0, 0)],
    "upper_arm": [(-.17, 0, 0, 0), (-.1, .74, .70, 0), (.03, 1.05, 1.05, 0),
                  (.22, 1.02, 1, 0), (.43, .83, .90, .03), (.57, .79, .89, .03),
                  (.82, .63, .65, 0), (.97, .61, .62, 0), (1.10, 0, 0, 0)],
    "forearm": [(-.11, 0, 0, 0), (.01, .85, .83, 0), (.17, 1, .99, .04),
                (.36, .92, .94, .03), (.66, .70, .72, 0), (.90, .52, .52, 0),
                (1.01, .54, .50, 0), (1.09, 0, 0, 0)],
    "thigh": [(-.16, 0, 0, 0), (-.08, .81, .83, 0), (.07, 1, 1, -.02),
              (.27, .98, 1.04, .02), (.52, .87, .92, .07), (.74, .69, .72, .07),
              (.93, .55, .59, .06), (1.02, .51, .52, .04), (1.11, 0, 0, 0)],
    "shin": [(-.11, 0, 0, 0), (-.03, .69, .74, .08), (.08, .83, .90, .05),
             (.23, .97, 1, -.04), (.39, 1, .96, -.13), (.60, .78, .77, -.12),
             (.83, .51, .54, -.04), (.98, .45, .43, 0), (1.08, 0, 0, 0)],
    "foot": [(-.21, 0, 0, .1), (-.12, .67, .68, .16), (.08, .85, 1, .23),
             (.34, .78, .85, .14), (.62, .94, .62, -.05), (.84, 1, .50, -.14),
             (1.01, .81, .43, -.18), (1.16, 0, 0, -.18)],
    "hand": [(-.12, 0, 0, 0), (.01, .57, .72, 0), (.22, .87, .98, 0),
             (.52, 1, 1, .02), (.74, .96, .78, .01), (.99, .8, .57, -.04),
             (1.13, .52, .38, -.12), (1.20, 0, 0, -.15)],
    "thumb": [(-.1, 0, 0, 0), (.08, .83, .85, 0), (.33, 1, 1, 0),
              (.78, .76, .8, 0), (1, .58, .63, 0), (1.15, 0, 0, 0)],
}


def surface(kind, sides=32):
    """Build a closed mesh with outward winding and smooth area-weighted normals."""
    profile = PROFILES[kind]
    verts = [(0, 2 * profile[0][0] - 1, profile[0][3])]
    faces = []
    angles = np.arange(sides) * (2 * np.pi / sides)
    # Extra stations around the rounded skull shape localize the brow and nose.
    stations = np.array(profile[1:-1])
    if kind == "head":
        heights = np.unique(np.r_[stations[:, 0], [.28, .39, .44, .49, .59]])
        stations = np.column_stack([heights, *(np.interp(heights, stations[:, 0], col) for col in stations[:, 1:].T)])
    for t, width, depth, offset in stations:
        for angle in angles:
            x, z = np.cos(angle), np.sin(angle)
            front = max(0, z)
            relief = 0
            if kind == "head":
                # A subdued bridge, brow and ears keep the face direction readable.
                relief = (.26 * np.exp(-((t - .40) / .11) ** 2) * front ** 38
                          + .055 * np.exp(-((t - .56) / .07) ** 2) * front ** 3)
                x *= 1 + .08 * np.exp(-((t - .45) / .11) ** 2) * abs(x) ** 10
            elif kind == "chest":
                relief = .09 * np.exp(-((t - .69) / .18) ** 2) * front * abs(x) * 2
            verts.append((width * x, 2 * t - 1, offset + depth * z + relief))
    top = len(verts)
    verts.append((0, 2 * profile[-1][0] - 1, profile[-1][3]))
    for s in range(sides):
        nxt = (s + 1) % sides
        faces.append((0, 1 + s, 1 + nxt))
        for i in range(len(stations) - 1):
            a, b = 1 + i * sides + s, 1 + i * sides + nxt
            faces.extend(((a, a + sides, b), (b, a + sides, b + sides)))
        faces.append((top, top - sides + nxt, top - sides + s))
    vertices, indices = np.array(verts), np.array(faces)
    a, b, c = vertices[indices].transpose(1, 0, 2)
    face_normals = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for corner in indices.T:
        np.add.at(normals, corner, face_normals)
    normals = _unit(normals)
    if kind == "hand":
        # The fingers are a neutral surface detail rigidly attached to the palm.
        # Only wrist, index and pinky establish this part's motion in 3D.
        vertices[:, 1] = (vertices[:, 1] + 1) * .65 - 1
        normals = _unit(normals / [1, .65, 1])
        finger, finger_normals, finger_indices = surface("finger", sides=12)
        faces = [indices.ravel()]
        for lateral, length in [(-.69, .90), (-.235, 1), (.235, 1.02), (.69, .90)]:
            scaling = np.array([.235, .48 * length, .45])
            offset = [lateral, .70 * length, -.11]
            faces.append(finger_indices + len(vertices))
            vertices = np.vstack((vertices, finger * scaling + offset))
            normals = np.vstack((normals, _unit(finger_normals / scaling)))
        return vertices, normals, np.concatenate(faces)
    return vertices, normals, indices.ravel()


def _unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm > 1e-8, norm, np.nan)


def _quaternions(x, y, z):
    """Use the largest quaternion component for stability at half turns."""
    xx, xy, xz = x.T
    yx, yy, yz = y.T
    zx, zy, zz = z.T
    candidates = np.stack([
        np.column_stack((1 + xx - yy - zz, xy + yx, xz + zx, yz - zy)),
        np.column_stack((xy + yx, 1 - xx + yy - zz, yz + zy, zx - xz)),
        np.column_stack((xz + zx, yz + zy, 1 - xx - yy + zz, xy - yx)),
        np.column_stack((yz - zy, zx - xz, xy - yx, 1 + xx + yy + zz)),
    ], axis=1)
    choice = candidates.diagonal(axis1=1, axis2=2).argmax(axis=1)
    return _unit(candidates[np.arange(len(y)), choice])


def _carry_lateral(directions):
    """Transport the limb's neutral roll. Switching reference axes mid-swing
    can otherwise rotate the surface abruptly around a perfectly steady bone.
    """
    axes = np.full_like(directions, np.nan)
    last_y = last_x = None
    for i, y in enumerate(directions):
        if not np.isfinite(y).all():
            last_y = last_x = None
            continue
        if last_y is None:
            x = np.eye(3)[np.argmin(abs(y))]
        else:
            # Shortest rotation from last_y to y, applied to last_x. For an
            # exact half turn, last_x itself is a valid perpendicular axis.
            dot = np.dot(last_y, y)
            x = last_x if dot < -.99999 else last_x - (last_y + y) * (np.dot(last_x, y) / (1 + dot))
        x = x - np.dot(x, y) * y
        axes[i] = x / np.linalg.norm(x)
        last_x, last_y = axes[i], y
    return axes


def _pose(start, end, width, depth, lateral=None):
    y = _unit(end - start)
    if lateral is None:
        # These rotationally near-symmetric limbs do not measure axial twist.
        lateral = _carry_lateral(y)
    z = _unit(np.cross(lateral, y))
    x = np.cross(y, z)
    rot = _quaternions(x, y, z)
    pos = (start + end) / 2
    scale = np.column_stack(np.broadcast_arrays(width, np.linalg.norm(end - start, axis=1) / 2, depth))
    valid = np.isfinite(np.column_stack((pos, rot, scale))).all(axis=1) & (scale > 0).all(axis=1)
    return np.where(valid[:, None], pos, 0), np.where(valid[:, None], rot, [0, 0, 0, 1]), np.where(valid[:, None], scale, 0)


def parts(points):
    """Yield each shell's mesh type, name and TRS tracks, using Y-up fitted points."""
    p = points.transpose(1, 0, 2)
    hip, shoulder = (p[23] + p[24]) / 2, (p[11] + p[12]) / 2
    spine = shoulder - hip
    lateral, hip_line = p[11] - p[12], p[23] - p[24]
    shoulder_width = np.linalg.norm(lateral, axis=1)
    typical = shoulder_width[np.isfinite(shoulder_width) & (shoulder_width > .08)]
    unit = np.clip(np.median(typical) if typical.size else .34, .22, .52)
    hip_width = np.linalg.norm(hip_line, axis=1)
    yield "pelvis", "Pelvis", _pose(hip - .21 * spine, hip + .41 * spine, hip_width * .70, unit * .335, hip_line)
    yield "chest", "Rib cage", _pose(hip + .20 * spine, hip + 1.10 * spine, shoulder_width * .52, unit * .36, lateral)

    ear = (p[7] + p[8]) / 2
    ear_line = _unit(p[7] - p[8])
    head_up = _unit(np.cross(p[0] - ear, ear_line))
    face = np.cross(ear_line, head_up)
    center = ear + unit * (.045 * head_up + .035 * face)
    jaw, crown = center - unit * .31 * head_up, center + unit * .39 * head_up
    yield "head", "Head", _pose(jaw, crown, unit * .245, unit * .29, ear_line)
    yield "neck", "Neck", _pose(hip + 1.00 * spine, jaw + unit * .13 * head_up, unit * .185, unit * .175, lateral)

    for side, name in enumerate(("Left", "Right")):
        # Bridge the sternum to the actual moving shoulder. The shoulder line can
        # tilt independently of the trunk when the serving arm rises overhead.
        socket = p[11 + side]
        inner = shoulder + .19 * (socket - shoulder) + .10 * spine
        outer = socket + .13 * (p[13 + side] - socket)
        yield "shoulder", f"{name} shoulder", _pose(inner, outer, unit * .165, unit * .178)
        for kind, a, b, w, d in (("upper_arm", 11, 13, .183, .183), ("forearm", 13, 15, .142, .139),
                                  ("thigh", 23, 25, .245, .268), ("shin", 25, 27, .169, .177)):
            yield kind, f"{name} {kind.replace('_', ' ')}", _pose(p[a + side], p[b + side], unit * w, unit * d)
        heel, toe, ankle = p[29 + side], p[31 + side], p[27 + side]
        shoe_width = np.cross(toe - heel, ankle - heel)
        instep = heel + .28 * (toe - heel)
        yield "ankle", f"{name} ankle", _pose(ankle, instep, unit * .080, unit * .083)
        yield "foot", f"{name} foot", _pose(heel, toe, unit * .134, unit * .108, shoe_width)
        wrist, index, pinky, thumb = (p[j + side] for j in (15, 19, 17, 21))
        tip = (index + pinky) / 2
        yield "hand", f"{name} hand", _pose(wrist, tip, unit * .095, unit * .058, index - pinky)
        yield "thumb", f"{name} thumb", _pose(wrist + .28 * (tip - wrist), thumb, unit * .034, unit * .032)
