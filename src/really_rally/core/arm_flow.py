"""Image-plane evidence for elbows and wrists between agreeing detector anchors."""

import cv2
import numpy as np

from really_rally.core.continuity import _arm_swaps
from really_rally.utils.subprocesses import check_cancelled

ARM = [13, 14, 15, 16]


def _reject_excursions(points, times, span, connected):
    """A patch can briefly stick to the torso then return to the wrist. Drop
    islands bounded by two large jumps that reverse each other."""
    for j in range(points.shape[1]):
        ids = np.flatnonzero(np.isfinite(points[:, j]).all(axis=1))
        for k in range(1, len(ids) - 1):
            a, first = ids[k - 1 : k + 1]
            distance = np.linalg.norm(points[first, j] - points[a, j])
            if distance < span[a] * max(0.30, 14 * (times[first] - times[a])):
                continue
            for m in range(k + 1, len(ids)):
                b, last = ids[m], ids[m - 1]
                if times[b] - times[a] > 0.15 or not connected[a + 1 : b + 1].all():
                    break
                returning = np.linalg.norm(points[b, j] - points[last, j])
                baseline = np.linalg.norm(points[b, j] - points[a, j])
                if returning > span[b] * max(0.30, 14 * (times[b] - times[last])) and baseline < 0.55 * min(
                    distance, returning
                ):
                    points[ids[k:m], j] = np.nan
                    break


def arm_flow(encoded, box, size, times, forward, reverse, hands, *, stop_event=None):
    """Return source pixels from agreeing patch tracks across the stroke. Discard tracks across cuts, poor texture, large patch errors or a lost
    forward/backward round trip. These pixels are cues, never depth observations.
    """
    image, vis = forward
    ri, rv = reverse
    obs, alt = image[..., :2] * size, ri[..., :2] * size
    torso = obs[:, [11, 12, 23, 24]]
    span = np.linalg.norm(torso[:, :2].mean(1) - torso[:, 2:].mean(1), axis=1)
    dt = np.diff(times)
    adjacent = np.r_[False, (dt > 0) & (dt <= min(0.10, 2.5 * np.median(dt)))] if len(dt) else np.array([False])
    adjacent[1:] &= np.linalg.norm(np.diff(torso.mean(1), axis=0), axis=1) < 0.65 * span[:-1]
    identity = [_arm_swaps(packet, size, adjacent) for packet in (image, ri)]
    same_player = (
        (np.linalg.norm(torso - alt[:, [11, 12, 23, 24]], axis=2) < 0.25 * span[:, None])
        & (rv[:, [11, 12, 23, 24]] >= 0.5)
    ).sum(1) >= 3
    agree = np.linalg.norm(obs[:, ARM] - alt[:, ARM], axis=2) < 0.14 * span[:, None]
    seeds = []
    for n, (coords, conf) in enumerate(((obs, vis), (alt, rv))):
        ok = (conf[:, ARM] >= 0.50) | (agree & (conf[:, ARM] >= 0.30))
        ok &= ~identity[n][:, None]
        if n:
            ok &= same_player[:, None] & ~(agree & identity[0][:, None])
        for k, j in enumerate(ARM):
            other = j + (1 if j % 2 else -1)
            # When both hands genuinely meet, the reliable original pose stays
            # available. Such an overlap cannot establish a new hand identity.
            near_other = np.linalg.norm(coords[:, j] - obs[:, other], axis=1)
            ok[:, k] &= ~((vis[:, other] >= 0.6) & (near_other < 0.30 * span))
        proposals = np.where(ok[..., None], coords[:, ARM], np.nan)
        seeds.append(proposals)
    hand_seeds = np.full_like(seeds[0], np.nan)
    for side in (0, 1):
        hand = hands[:, side, 0, :2] * size
        own = np.linalg.norm(hand - obs[:, 15 + side], axis=1)
        other = np.linalg.norm(hand - obs[:, 16 - side], axis=1)
        ok = (own < 0.16 * span) & (other > own + 0.09 * span) & ~identity[0]
        hand_seeds[ok, 2 + side] = hand[ok]
    seeds.append(hand_seeds)
    pixels = np.full_like(seeds[0], np.nan)
    if not all(item is not None for item in encoded):
        return pixels
    h, w = cv2.imdecode(encoded[0], cv2.IMREAD_GRAYSCALE).shape
    xy0 = np.asarray(box[:2])
    ratio = np.array([w, h]) / (np.asarray(box[2:]) - xy0)
    packets = (np.array(seeds) - xy0) * ratio
    config = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))

    def sweep(order):
        result = [[None] * 4 for _ in times]
        pts, born, joint = np.empty((0, 2)), np.empty(0, int), np.empty(0, int)
        previous = last_gray = None
        for i in order:
            check_cancelled(stop_event)
            gray = cv2.imdecode(encoded[i], cv2.IMREAD_GRAYSCALE)
            if previous is not None and len(pts):
                keep = (abs(times[i] - times[born]) <= 0.40) & adjacent[max(i, previous)]
                pts, born, joint = pts[keep], born[keep], joint[keep]
                if len(pts):
                    corners = pts.astype(np.float32).reshape(-1, 1, 2)
                    nxt, found, error = cv2.calcOpticalFlowPyrLK(last_gray, gray, corners, None, **config)
                    back, returned, _ = cv2.calcOpticalFlowPyrLK(gray, last_gray, nxt, None, **config)
                    tol = np.clip(0.021 * span[i] * ratio.mean(), 1.0, 3.0)
                    good = (found[:, 0] != 0) & (returned[:, 0] != 0) & (error[:, 0] < 25)
                    good &= np.linalg.norm((corners - back)[:, 0], axis=1) <= tol
                    good &= ((nxt[:, 0] >= 0) & (nxt[:, 0] < [w, h])).all(axis=1)
                    pts, born, joint = nxt[good, 0], born[good], joint[good]
                    for j in range(4):
                        selected = joint == j
                        result[i][j] = (pts[selected] / ratio + xy0, born[selected])
            # Several seeds at a time let later agreement discard one wrong label.
            for batch in packets[:, i]:
                ids = np.flatnonzero((np.isfinite(batch) & (batch >= 0) & (batch < [w, h])).all(axis=1))
                pts = np.vstack((pts, batch[ids]))
                born = np.r_[born, np.full(len(ids), i, int)]
                joint = np.r_[joint, ids]
            previous, last_gray = i, gray
        return result

    before, after = sweep(range(len(times))), sweep(reversed(range(len(times))))
    candidates = [[[] for _ in times] for _ in ARM]
    for i in range(len(times)):
        check_cancelled(stop_event)
        radius = np.clip(0.10 * span[i], 3.0, 12.0)
        for j in range(4):
            if before[i][j] is None or after[i][j] is None:
                continue
            past, future = before[i][j], after[i][j]
            points = np.vstack((past[0], future[0]))
            if not len(points):
                continue
            dist = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            close = dist <= radius
            # Each seed frame votes once, even if both detectors or a hand saw it.
            votes = []
            for neighbors in close:
                a = len(np.unique(past[1][neighbors[: len(past[0])]]))
                b = len(np.unique(future[1][neighbors[len(past[0]) :]]))
                # One-sided flow is a weaker hypothesis near a confirmed patch.
                score = 0.0
                if min(a, b) >= 2:
                    score = 2 * a * b / (a + b)
                elif max(a, b) >= 5:
                    score = min(3.0, max(a, b) * 0.28)
                votes.append(score)
            scores = np.asarray(votes)
            for _ in range(5):
                best = int(np.argmax(scores))
                if scores[best] <= 0:
                    break
                score = scores[best]
                proposed = np.median(points[close[best]], axis=0)
                scores[dist[best] < 1.5 * radius] = 0
                other = ARM[j] + (1 if ARM[j] % 2 else -1)
                if vis[i, other] >= 0.6 and np.linalg.norm(proposed - obs[i, other]) < 0.25 * span[i]:
                    continue
                neighbors = close[best]
                dual = (
                    len(np.unique(past[1][neighbors[: len(past[0])]])) >= 2
                    and len(np.unique(future[1][neighbors[len(past[0]) :]])) >= 2
                )
                candidates[j][i].append((proposed, np.log1p(score), dual))
    for j, frames in enumerate(candidates):
        ids = np.array([i for i, row in enumerate(frames) if row], dtype=int)
        cuts = [
            k
            for k, (a, b) in enumerate(zip(ids[:-1], ids[1:]), 1)
            if times[b] - times[a] > 0.15 or not adjacent[a + 1 : b + 1].all()
        ]
        for run in np.split(ids, cuts):
            if not len(run):
                continue
            history = []
            old = previous = None
            for k, i in enumerate(run):
                points, score, _ = zip(*frames[i])
                points, score = np.stack(points), np.array(score)
                if k:
                    length = max(span[i] * (times[i] - times[run[k - 1]]) * 5, 2)
                    travel = np.linalg.norm(old[:, None] - points, axis=2) / length
                    options = previous[:, None] - np.minimum(travel**2, 25)
                    history.append(options.argmax(axis=0))
                    score += options.max(axis=0)
                old, previous = points, score
            choice = int(previous.argmax())
            confirmed = np.zeros(len(run), bool)
            for k in reversed(range(len(run))):
                i = run[k]
                pixels[i, j], _, confirmed[k] = frames[i][choice]
                if k:
                    choice = history[k - 1][choice]
            # A one-sided patch has a short lifetime beyond two-sided consensus.
            good = times[run[confirmed]]
            for i in run[~confirmed]:
                if not len(good) or np.min(abs(good - times[i])) > 0.12:
                    pixels[i, j] = np.nan
    _reject_excursions(pixels, times, span, adjacent)
    # An independently localized, unambiguous hand also supplies wrist pixels
    # during body-tracker dropout. It supplies no body-world depth.
    direct = np.isfinite(hand_seeds).all(axis=2)
    pixels[direct] = hand_seeds[direct]
    return pixels
