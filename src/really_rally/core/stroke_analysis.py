"""Single-view body fitting and tennis measurements. Raw inputs stay immutable."""
import csv

import numpy as np


# Only these links have approximately constant anatomical lengths during a stroke.
LINKS = ((11, 12), (23, 24), (11, 23), (12, 24), (11, 13), (12, 14),
         (13, 15), (14, 16), (23, 25), (24, 26), (25, 27), (26, 28))


def _length(frames, a, b):
    return np.linalg.norm(frames[:, a] - frames[:, b], axis=1)


def _camera(world, pixels, visibility):
    """Robust weak-perspective scale and pixel translation from trunk/legs only."""
    ids = [11, 12, 23, 24, 25, 26, 27, 28]
    src, dst = world[:, ids, :2], pixels[:, ids]
    finite = np.isfinite(src).all(axis=2) & np.isfinite(dst).all(axis=2)
    src = np.where(finite[..., None], src, 0)
    dst = np.where(finite[..., None], dst, 0)
    base = np.where(finite, visibility[:, ids], 0)
    weights = base.copy()
    for _ in range(3):
        weight = weights[..., None]
        total = np.maximum(weight.sum(axis=1), 1e-8)
        center = (src * weight).sum(axis=1) / total
        origin = (dst * weight).sum(axis=1) / total
        u, v = src - center[:, None], dst - origin[:, None]
        denom = (weight * u * u).sum(axis=(1, 2))
        scale = (weight * u * v).sum(axis=(1, 2)) / np.maximum(denom, 1e-8)
        offset = origin - center * scale[:, None]
        error = np.linalg.norm(src * scale[:, None, None] + offset[:, None] - dst, axis=2)
        weights = base * np.minimum(1, 4 / np.maximum(error, 1e-8))
    good = (finite.sum(axis=1) >= 4) & (denom > .01) & (scale > 1)
    return np.where(good, scale, np.nan), np.where(good[:, None], offset, np.nan)


def fit_body(world, pixels, visibility):
    """Balance observed pixels, the depth prior, and within-clip segment lengths."""
    scale, offset = _camera(world, pixels, visibility)
    targets = world.copy()
    screen = (pixels - offset[:, None]) / scale[:, None, None]
    # Bound the influence of the approximate camera, particularly at oblique angles.
    delta = screen - world[..., :2]
    norm = np.linalg.norm(delta, axis=2, keepdims=True)
    delta *= np.minimum(1, .09 / np.maximum(norm, 1e-8))
    targets[..., :2] += np.nan_to_num(delta) * visibility[..., None] * .75
    lengths = {}
    for a, b in LINKS:
        values = _length(world, a, b)
        sample = values[np.isfinite(values) & (values > .02)]
        if len(sample) >= 5:
            lengths[a, b] = float(np.median(sample))
    fitted = targets.copy()
    for _ in range(14):
        step = .20 * (targets - fitted)
        for (a, b), typical in lengths.items():
            vector = fitted[:, b] - fitted[:, a]
            size = np.linalg.norm(vector, axis=1)
            error = (size - typical) / np.maximum(size, 1e-8)
            correction = .22 * np.nan_to_num(error[:, None] * vector)
            step[:, a] += correction
            step[:, b] -= correction
        fitted += step
        change = fitted - world
        distance = np.linalg.norm(change, axis=2, keepdims=True)
        fitted = world + change * np.minimum(1., .09 / np.maximum(distance, 1e-8))
    # Move the small pose hand/foot landmarks with their fitted parent, without
    # inventing observations when the parent or endpoint was missing.
    for parent, children in ((15, [17, 19, 21]), (16, [18, 20, 22]),
                             (27, [29, 31]), (28, [30, 32])):
        fitted[:, children] = world[:, children] + (fitted[:, parent] - world[:, parent])[:, None]
    # Heel and toe initially follow the ankle. Rotate that triangle in the
    # image plane toward its own detections without changing its three lengths.
    for ankle, ends in ((27, [29, 31]), (28, [30, 32])):
        current = fitted[:, ends, :2] - fitted[:, ankle, None, :2]
        target = screen[:, ends] - fitted[:, ankle, None, :2]
        observed = np.isfinite(current).all(axis=2) & np.isfinite(target).all(axis=2)
        weight = np.where(observed, visibility[:, ends], 0)
        u, v = np.nan_to_num(current), np.nan_to_num(target)
        cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        dot = (u * v).sum(axis=2)
        angle = .8 * np.arctan2((weight * cross).sum(axis=1), (weight * dot).sum(axis=1))
        angle = np.clip(angle, -.32, .32)[:, None]
        cosine, sine = np.cos(angle), np.sin(angle)
        turned = np.stack((cosine * current[..., 0] - sine * current[..., 1],
                           sine * current[..., 0] + cosine * current[..., 1]), axis=2)
        fitted[:, ends, :2] = fitted[:, ankle, None, :2] + turned
    # Keep the hip origin while preserving the fitted projection.
    pelvis = np.mean(fitted[:, [23, 24]], axis=1)
    pelvis = np.where(np.isfinite(pelvis), pelvis, 0)
    fitted -= pelvis[:, None]
    offset += pelvis[:, :2] * scale[:, None]
    return fitted, scale, offset, lengths


def _unit(vector, minimum=.015):
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(norm) and norm > minimum else np.full(3, np.nan)


def _project_plane(unit, normal):
    return unit - normal * np.dot(unit, normal)


def measurements(display, original, reference, frames, times, impact, pixels):
    """Signed separation about the hip-to-shoulder axis. Both arms share this basis."""
    rows = []
    typical_dt = np.median(np.diff(times)) if len(times) > 1 else 0
    plausible = {}
    for link in LINKS:
        length = _length(original, *link)
        consistent = np.isfinite(length)
        if link in reference:
            consistent &= np.abs(length / reference[link] - 1) <= .30
        plausible[link] = consistent
    for i, points in enumerate(display):
        hip = (points[23] + points[24]) / 2
        up = _unit((points[11] + points[12]) / 2 - hip)
        right = _unit(_project_plane(points[24] - points[23], up))
        forward = np.cross(up, right)
        shoulders = _unit(_project_plane(points[12] - points[11], up))
        twist = np.degrees(np.arctan2(np.dot(up, np.cross(right, shoulders)), np.dot(right, shoulders)))

        torso_ok = plausible[11, 12][i] and plausible[23, 24][i]
        row = dict(frame=int(frames[i]), timestamp_s=float(times[i]), from_impact_s=float(times[i] - times[impact]),
                   trunk_separation_deg=twist if torso_ok else np.nan)
        # Headings about camera Y. The review requires a fixed, level view before
        # interpreting these as turns about court vertical.
        for name, a, b in (("pelvis", 23, 24), ("shoulder", 11, 12)):
            across = points[b] - points[a]
            row[f"{name}_heading_deg"] = (np.degrees(np.arctan2(-across[2], across[0]))
                                          if plausible[a, b][i] and np.hypot(across[0], across[2]) > .05 else np.nan)
        for side, h, k, a in (("left", 23, 25, 27), ("right", 24, 26, 28)):
            observed = np.isfinite(points[[h, k, a]]).all()
            valid = observed and plausible[h, k][i] and plausible[k, a][i]
            thigh = _unit(points[k] - points[h])
            shin = _unit(points[a] - points[k])
            angle = np.degrees(np.arccos(np.clip(thigh @ shin, -1, 1)))
            row[f"{side}_knee_flexion_deg"] = angle if valid else np.nan
            row[f"{side}_knee_quality"] = ("missing_landmarks" if not observed else
                                            "usable" if valid and np.isfinite(angle) else "segment_geometry")
        for side, s, e, w, sign in (("left", 11, 13, 15, -1), ("right", 12, 14, 16, 1)):
            upper = _unit(points[e] - points[s])
            forearm = _unit(points[w] - points[e])
            upper_ok, fore_ok = plausible[s, e][i], plausible[e, w][i]
            direction_ok = upper_ok and torso_ok and np.isfinite(right).all()
            # Signed azimuth from the arm's outward side toward anatomical forward.
            lateral, anterior = sign * np.dot(upper, right), np.dot(upper, forward)
            azimuth = np.degrees(np.arctan2(anterior, lateral))
            row[f"{side}_elbow_azimuth_deg"] = azimuth if direction_ok and np.hypot(lateral, anterior) > .1 else np.nan
            row[f"{side}_elbow_elevation_deg"] = np.degrees(np.arcsin(np.clip(np.dot(upper, up), -1, 1))) if direction_ok else np.nan
            row[f"{side}_elbow_flexion_deg"] = np.degrees(np.arccos(np.clip(upper @ forearm, -1, 1))) if upper_ok and fore_ok else np.nan
            observed = np.isfinite(points[[s, e, w, 23, 24]]).all()
            row[f"{side}_wrist_quality"] = ("missing_landmarks" if not observed else
                                            "segment_length_outlier" if not (upper_ok and fore_ok) else "usable")
            wrist = points[w] - hip if upper_ok and fore_ok else np.full(3, np.nan)
            keys = [f"{side}_wrist_{axis}_m" for axis in "xyz"]
            row.update(zip(keys, wrist))
            speed = np.nan
            if i and .25 * typical_dt <= times[i] - times[i-1] <= 2.5 * typical_dt and frames[i] == frames[i-1]+1:
                speed = np.linalg.norm(wrist - np.asarray([rows[-1][key] for key in keys], dtype=float)) / (times[i] - times[i-1])
            row[f"{side}_wrist_speed_m_s"] = speed
            row[f"{side}_wrist_u_px"], row[f"{side}_wrist_v_px"] = pixels[i, w]
        rows.append({k: (v if isinstance(v, str) or np.isfinite(v) else None) for k, v in row.items()})
    return rows


def write_measurements(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
