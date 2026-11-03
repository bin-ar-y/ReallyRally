"""Offline joint trajectories. Keep camera detections and reconstructed poses separate."""

import numpy as np

MISSING, FORWARD, REVERSE, TEMPORAL, IMAGE_GUIDED, BOUNDARY = range(6)
STATUS = ["unavailable", "forward_fit", "reverse_fit", "temporal_estimate", "image_guided", "boundary_prediction"]
MAX_BRIDGE = 0.45  # seconds between bracketing anchors, not a number of frames
TORSO = [11, 12, 23, 24]
# Parents precede children. Small pose landmarks ride with the associated wrist/ankle.
TREE = [(23, None), (24, None), (11, 23), (12, 24), (0, 11)]
TREE += [(j, 0) for j in range(1, 11)]
for side in (0, 1):
    TREE.extend(
        (j + side, p + side)
        for j, p in [(13, 11), (15, 13), (17, 15), (19, 15), (21, 15), (25, 23), (27, 25), (29, 27), (31, 27)]
    )


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-7, n, np.nan)


def _torso(points):
    hip = points[:, [23, 24]].mean(axis=1)
    up = _unit(points[:, [11, 12]].mean(axis=1) - hip)
    front = _unit(np.cross(points[:, 11] - points[:, 12], up))
    return hip, np.stack((np.cross(up, front), up, front), axis=-1)


def _arm_swaps(image, size, connected):
    """Flag brief bilateral label flips. Near a natural crossing both assignments
    cost about the same; a switch pays a penalty and has to persist to win.
    """
    xy = image[..., :2] * size
    torso = xy[:, TORSO]
    center = torso.mean(axis=1)
    span = np.linalg.norm(torso[:, :2].mean(axis=1) - torso[:, 2:].mean(axis=1), axis=1)
    arm = (xy[:, [13, 15, 14, 16]] - center[:, None]) / np.maximum(span[:, None, None], 1)
    arm = arm.reshape(-1, 2, 2, 2)
    swapped = np.zeros(len(arm), dtype=bool)
    valid = np.isfinite(arm).all(axis=(1, 2, 3)) & (span > 8)
    ids = np.flatnonzero(valid)
    for run in np.split(ids, np.flatnonzero((np.diff(ids) != 1) | ~connected[ids[1:]]) + 1):
        if len(run) < 3:
            continue
        cost = np.array([0.0, 0.45])
        back = np.zeros((len(run), 2), dtype=int)
        for k, (a, b) in enumerate(zip(run[:-1], run[1:]), 1):
            same = np.linalg.norm(arm[a] - arm[b], axis=-1).mean()
            cross = np.linalg.norm(arm[a] - arm[b, ::-1], axis=-1).mean() + 0.24
            options = cost[:, None] + [[same, cross], [cross, same]]
            back[k] = options.argmin(axis=0)
            cost = options.min(axis=0) + [0, 0.025]
        state = int(cost.argmin())
        for k in reversed(range(len(run))):
            swapped[run[k]] = bool(state)
            state = back[k, state]
    return swapped


def _plausible(fitted, raw, lengths, image, size, connected):
    ok = np.isfinite(fitted).all(axis=2)
    for child, parent in TREE:
        length = lengths.get((parent, child))
        if length is None:
            continue
        distance = np.linalg.norm(raw[:, child] - raw[:, parent], axis=1)
        # An independently tracked wrist can still constrain a missing elbow.
        ok[:, child] &= ~np.isfinite(distance) | (abs(distance / length - 1) <= 0.30)
    ok[_arm_swaps(image, size, connected), 13:23] = False
    return ok


def _clip_vector(vector, radius):
    return vector * min(1.0, radius / max(np.linalg.norm(vector), 1e-9))


def _bridge(track, quality, j, parent, times, connected, basis, body_present):
    anchors = np.flatnonzero(quality[:, j] > 0)
    origin = np.zeros_like(track[:, j]) if parent is None else track[:, parent]
    for a, b in zip(anchors[:-1], anchors[1:]):
        carried_hand = (
            17 <= j <= 22
            and np.isfinite(origin[a : b + 1]).all()
            and (quality[a : b + 1, parent] == IMAGE_GUIDED).any()
        )
        limit = 1.25 if carried_hand else MAX_BRIDGE
        if b == a + 1 or times[b] - times[a] > limit:
            continue
        window = slice(a, b + 1)
        if not (connected[a + 1 : b + 1].all() and body_present[window].all() and np.isfinite(origin[window]).all()):
            continue
        local = parent is not None and np.isfinite(basis[window]).all()

        def offset(i, local=local):
            v = track[i, j] - origin[i]
            return v @ basis[i] if local else v

        left, right = offset(a), offset(b)
        delta = right - left
        span = times[b] - times[a]
        v0, v1 = delta, delta
        if a and connected[a] and quality[a - 1, j] in (FORWARD, REVERSE):
            candidate = (left - offset(a - 1)) * span / (times[a] - times[a - 1])
            if np.isfinite(candidate).all():
                v0 = candidate
        if b + 1 < len(times) and connected[b + 1] and quality[b + 1, j] in (FORWARD, REVERSE):
            candidate = (offset(b + 1) - right) * span / (times[b + 1] - times[b])
            if np.isfinite(candidate).all():
                v1 = candidate
        length0, length1 = np.linalg.norm(left), np.linalg.norm(right)
        cap = 2 * np.linalg.norm(delta) + 0.2 * min(length0, length1) + 0.01
        v0, v1 = _clip_vector(v0, cap), _clip_vector(v1, cap)
        s = ((times[a + 1 : b] - times[a]) / span)[:, None]
        curve = (
            (1 - 3 * s * s + 2 * s**3) * left
            + (3 * s * s - 2 * s**3) * right
            + s * (1 - s) ** 2 * v0
            - s * s * (1 - s) * v1
        )
        if parent is not None:
            # Preserve anatomical length through the bend. An antipodal direction
            # without a bend-plane cue is ambiguous; do not collapse the limb.
            if (np.linalg.norm(curve, axis=1) < 0.12 * min(length0, length1)).any():
                continue
            curve = _unit(curve) * ((1 - s) * length0 + s * length1)
        if local:
            curve = np.einsum("nij,nj->ni", basis[a + 1 : b], curve)
        path = origin[a + 1 : b] + curve
        if np.isfinite(path).all():
            track[a + 1 : b, j] = path
            quality[a + 1 : b, j] = TEMPORAL


def _clip_edges(track, quality, j, parent, times, basis, connected, cues, scale, offset, span):
    """A short, one-sided prediction is allowed only when both trackers still
    put this joint near the predicted pixels on EVERY frame up to the clip edge."""
    ids = np.flatnonzero(quality[:, j])
    if len(ids) < 3:
        return
    for anchor, step in ((ids[0], -1), (ids[-1], 1)):
        edge = np.arange(anchor + step, -1 if step < 0 else len(times), step)
        if not len(edge) or abs(times[edge[-1]] - times[anchor]) > 0.22:
            continue
        history = anchor - step * np.arange(3)
        if (
            min(history) < 0
            or max(history) >= len(times)
            or not np.isin(quality[history, j], [FORWARD, REVERSE, IMAGE_GUIDED]).all()
        ):
            continue
        lo, hi = min(min(history), min(edge)), max(max(history), max(edge))
        if not connected[lo + 1 : hi + 1].all() or not np.isfinite(cues[edge]).all():
            continue
        vectors = track[history, j] - track[history, parent]
        local = np.einsum("ni,nij->nj", vectors, basis[history])
        velocity = np.median(np.diff(local, axis=0) / np.diff(times[history])[:, None], axis=0)
        length = np.linalg.norm(local[0])
        if not np.isfinite(length) or length < 0.005:
            continue
        # Damping avoids continuing a high swing velocity indefinitely.
        seconds = abs(times[edge] - times[anchor])
        change = _clip_vector(step * velocity * 0.065, 0.30 * length)
        direction = _unit(local[0] + (1 - np.exp(-seconds / 0.065))[:, None] * change)
        path = track[edge, parent] + length * np.einsum("nij,nj->ni", basis[edge], direction)
        screen = path[:, :2] * scale[edge, None] + offset[edge]
        if (np.linalg.norm(screen - cues[edge], axis=1) <= 0.35 * span[edge]).all():
            track[edge, j], quality[edge, j] = path, BOUNDARY


def _bend(track, quality, joint, parent, end, lengths):
    """Use the observed distal joint and the predicted bend plane to place the
    hidden elbow/knee on the intersection of the two segment-length spheres."""
    l1, l2 = lengths.get((parent, joint)), lengths.get((joint, end))
    if l1 is None or l2 is None:
        return
    for i in np.flatnonzero(
        (quality[:, joint] >= REVERSE) & np.isin(quality[:, end], [FORWARD, REVERSE, IMAGE_GUIDED])
    ):
        root = track[i, parent]
        axis = track[i, end] - root
        d = np.linalg.norm(axis)
        if not np.isfinite(d) or d < 0.005:
            continue
        axis /= d
        reachable = np.clip(d, abs(l1 - l2) + 0.001, l1 + l2 - 0.001)
        if abs(reachable - d) > 1e-7:
            track[i, end] = root + reachable * axis
            quality[i, end] = TEMPORAL
            d = reachable
        off = track[i, joint] - root
        pole = _unit(off - axis * np.dot(axis, off))
        if np.isfinite(pole).all():
            advance = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
            track[i, joint] = root + advance * axis + np.sqrt(max(0, l1 * l1 - advance**2)) * pole


def _image_anchors(track, quality, j, parent, pixels, scale, offset, lengths, times, connected):
    """Use agreed source pixels to anchor a hidden arm segment. Depth comes
    from bracketing 3D directions; a 2D track cannot establish it by itself.
    """
    length = lengths.get((parent, j))
    if length is None:
        return
    anchors = np.flatnonzero((quality[:, j] > 0) & (quality[:, parent] > 0))
    for i in np.flatnonzero((quality[:, j] != FORWARD) & np.isfinite(pixels).all(axis=1)):
        if not np.isfinite(scale[i]) or not np.isfinite(track[i, parent]).all():
            continue
        if quality[i, j] == REVERSE:
            prior = track[i, j] - track[i, parent]
        else:
            k = np.searchsorted(anchors, i)
            if not 0 < k < len(anchors):
                continue
            a, b = anchors[k - 1 : k + 1]
            if (
                times[b] - times[a] > 1.4
                or max(times[i] - times[a], times[b] - times[i]) > 1.0
                or not connected[a + 1 : b + 1].all()
            ):
                continue
            f = (times[i] - times[a]) / (times[b] - times[a])
            prior = (1 - f) * (track[a, j] - track[a, parent]) + f * (track[b, j] - track[b, parent])
        if np.linalg.norm(prior) < 0.01:
            continue
        origin = track[i, parent]
        xy = (pixels[i] - offset[i]) / scale[i] - origin[:2]
        if np.linalg.norm(xy) > 1.20 * length:
            # If the elbow is hidden, a visible wrist can constrain BOTH links.
            # Its image radius need only be reachable from the shoulder.
            if j not in (15, 16) or quality[i, parent] <= FORWARD:
                continue
            shoulder = parent - 2
            reach = lengths.get((shoulder, parent), 0) + length
            root = track[i, shoulder]
            planar = (pixels[i] - offset[i]) / scale[i] - root[:2]
            if not np.isfinite(root).all() or np.linalg.norm(planar) >= reach * 0.985:
                continue
            depth = np.sqrt((reach * 0.985) ** 2 - np.dot(planar, planar))
            track[i, j, :2] = root[:2] + planar
            track[i, j, 2] = root[2] + np.clip((origin + prior - root)[2], -depth, depth)
            quality[i, j] = IMAGE_GUIDED
            continue
        fraction = _clip_vector(xy / length, 0.995)
        direction = _unit(prior)
        z = np.sqrt(1 - np.dot(fraction, fraction)) * np.sign(direction[2] or 1.0)
        goal = np.r_[fraction, z]
        track[i, j] = origin + length * _unit(0.85 * goal + 0.15 * direction)
        quality[i, j] = IMAGE_GUIDED


def _image_corridor(flow, times, connected):
    # Use short bracketing image tracks to reject a side switch between cues.
    # These interpolants are never promoted to optical observations.
    guide = flow.copy()
    for j in range(flow.shape[1]):
        ids = np.flatnonzero(np.isfinite(flow[:, j]).all(axis=1))
        for a, b in zip(ids[:-1], ids[1:]):
            if b - a > 1 and times[b] - times[a] <= 0.30 and connected[a + 1 : b + 1].all():
                fraction = ((times[a + 1 : b] - times[a]) / (times[b] - times[a]))[:, None]
                guide[a + 1 : b, j] = (1 - fraction) * flow[a, j] + fraction * flow[b, j]
    return guide


def _settle(track, quality, j, parent, times, connected):
    """Regularize only reconstructed link directions, favoring their image xy
    over ambiguous depth. Reliable source samples stay fixed."""
    vec = track[:, j] - track[:, parent]
    lengths = np.linalg.norm(vec, axis=1)
    direction = _unit(vec)
    use = (quality[1:-1, j] >= REVERSE) & connected[1:-1] & connected[2:]
    use &= np.isfinite(direction[:-2] + direction[1:-1] + direction[2:]).all(axis=1)
    ids = np.flatnonzero(use) + 1
    if not len(ids):
        return
    frac = ((times[ids] - times[ids - 1]) / (times[ids + 1] - times[ids - 1]))[:, None]
    strength = np.where((quality[ids, j] == IMAGE_GUIDED)[:, None], [0.35, 0.35, 0.65], [0.55, 0.55, 0.65])
    initial = direction[ids].copy()
    for _ in range(5):
        target = (1 - frac) * direction[ids - 1] + frac * direction[ids + 1]
        direction[ids] = _unit(initial * (1 - strength) + target * strength)
    track[ids, j] = track[ids, parent] + direction[ids] * lengths[ids, None]


def complete_pose(fitted, filtered, lengths, image, visibility, times, size, reverse=None, flow=None, camera=None):
    """Return a fitted trajectory and per-joint provenance in detector coordinates.
    All inputs remain intact. Neither weak detections nor estimates become forward evidence.
    """
    dt = np.diff(times)
    connected = np.r_[False, (dt > 0) & (dt <= min(0.10, 2.5 * np.median(dt)))] if len(dt) else np.array([False])
    xy = image[:, TORSO, :2] * size
    center = xy.mean(axis=1)
    span = np.linalg.norm(xy[:, :2].mean(axis=1) - xy[:, 2:].mean(axis=1), axis=1)
    # Break trajectories when the original crop abruptly changes subject or scale.
    connected[1:] &= np.linalg.norm(np.diff(center, axis=0), axis=1) <= 0.65 * np.maximum(span[:-1], 8)
    connected[1:] &= (span[1:] > 0.6 * span[:-1]) & (span[1:] < 1.65 * span[:-1])
    ok = _plausible(fitted, filtered, lengths, image, size, connected)
    swapped = _arm_swaps(image, size, connected)
    if flow is not None:
        corridor = _image_corridor(flow, times, connected)
        mismatch = np.linalg.norm(image[:, 13:17, :2] * size - flow, axis=2)
        ok[:, 13:17] &= ~(mismatch > 0.30 * span[:, None])
    track = np.where(ok[..., None], fitted, np.nan)
    quality = ok.astype(np.uint8)
    hip, basis = _torso(fitted)
    body_present = np.isfinite(fitted[:, TORSO]).all(axis=2).sum(axis=1) >= 2
    edge_cues = None
    if reverse is not None:
        alt, raw, alt_image, alt_vis = reverse
        alt_hip, alt_basis = _torso(alt)
        offsets = np.einsum("nji,nkj->nki", alt_basis, alt - alt_hip[:, None])
        registered = hip[:, None] + np.einsum("nij,nkj->nki", basis, offsets)
        torso_error = np.linalg.norm(alt_image[:, TORSO, :2] * size - xy, axis=2)
        aligned = (
            (torso_error <= 0.25 * span[:, None]) & (alt_vis[:, TORSO] >= 0.6) & (visibility[:, TORSO] >= 0.5)
        ).sum(axis=1) >= 3
        # The two estimates must agree on the torso's facing, not just its projection.
        aligned &= (basis[:, :, 2] * alt_basis[:, :, 2]).sum(axis=1) > 0.6
        dist = np.linalg.norm((alt_image[..., :2] - image[..., :2]) * size, axis=2)
        agrees = aligned[:, None] & (dist <= 0.16 * span[:, None]) & (visibility >= 0.18) & (alt_vis >= 0.035)
        edge_cues = np.where(agrees[..., None], (alt_image[..., :2] + image[..., :2]) * size / 2, np.nan)
        edge_cues[swapped, 13:23] = np.nan
        candidate = _plausible(alt, raw, lengths, alt_image, size, connected)
        candidate &= (alt_vis >= 0.65) & aligned[:, None] & np.isfinite(registered).all(axis=2)
        if flow is not None:
            delta = np.linalg.norm(alt_image[:, 13:17, :2] * size - flow, axis=2)
            candidate[:, 13:17] &= ~(delta > 0.28 * span[:, None])
        for j in range(13, 23):
            other = j + (1 if j % 2 else -1)
            near_other = np.linalg.norm((alt_image[:, j, :2] - image[:, other, :2]) * size, axis=1)
            near_own = np.linalg.norm((alt_image[:, j, :2] - image[:, j, :2]) * size, axis=1)
            candidate[:, j] &= ~(swapped & (near_own < near_other))
            candidate[:, j] &= ~(
                (visibility[:, other] > 0.6) & (near_other < 0.18 * span) & (near_other + 0.10 * span < near_own)
            )
        # A single reverse-frame spike cannot become an anchor for an entire gap.
        consistent = np.zeros_like(candidate)
        adjacent = connected[1:, None] & candidate[1:] & candidate[:-1]
        adjacent &= np.linalg.norm(np.diff(registered, axis=0), axis=2) <= (0.04 + 12 * dt[:, None])
        consistent[1:] |= adjacent
        consistent[:-1] |= adjacent
        use = ~ok & consistent
        track[use], quality[use] = registered[use], REVERSE

    for j, parent in TREE:
        if j in (15, 16) and (parent, j) in lengths:
            # Re-evaluate distal anchors against the reconstructed elbow. An
            # apparently confident wrist on the tossing arm can be unreachable.
            span3 = np.linalg.norm(track[:, j] - track[:, parent], axis=1)
            suspect = (quality[:, parent] >= REVERSE) & (abs(span3 / lengths[parent, j] - 1) > 0.27)
            if flow is not None:
                drift = np.linalg.norm(image[:, j, :2] * size - corridor[:, j - 13], axis=1)
                suspect |= (quality[:, parent] >= REVERSE) & (quality[:, j] == FORWARD) & (drift > 0.32 * span)
            track[suspect, j], quality[suspect, j] = np.nan, MISSING
        if 13 <= j <= 16 and flow is not None and camera is not None:
            _image_anchors(track, quality, j, parent, flow[:, j - 13], *camera, lengths, times, connected)
        if 17 <= j <= 22:
            # Sparse hand tips depend on their wrist track in the detector.
            unreliable = (quality[:, parent] != quality[:, j]) & (quality[:, j] > MISSING)
            track[unreliable, j], quality[unreliable, j] = np.nan, MISSING
        _bridge(track, quality, j, parent, times, connected, basis, body_present)
        if parent is not None and edge_cues is not None and camera is not None:
            _clip_edges(track, quality, j, parent, times, basis, connected, edge_cues[:, j], *camera, span)
        if j in (13, 14, 15, 16):
            _settle(track, quality, j, parent, times, connected)
        if j in (15, 16):
            _bend(track, quality, j - 2, j - 4, j, lengths)
            length = lengths.get((parent, j))
            if length is not None:
                delta = track[:, j] - track[:, parent]
                norm = np.linalg.norm(delta, axis=1)
                repair = (quality[:, j] > MISSING) & ((quality[:, j] > FORWARD) | (quality[:, parent] > FORWARD))
                repair &= (norm > 0.005) & (abs(norm / length - 1) > 0.15)
                track[repair, j] = track[repair, parent] + length * _unit(delta[repair])
                quality[repair & (quality[:, j] == FORWARD), j] = TEMPORAL
        if j in (25, 26):
            _bend(track, quality, j, parent, j + 2, lengths)
    return track, quality
