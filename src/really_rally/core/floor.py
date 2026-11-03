"""Vertical placement for a stroke known to maintain court contact."""
import numpy as np

from really_rally.core.mannequin import parts, surface


def sole_heights(points):
    """Lowest posed mesh vertex per shoe, in the original hip coordinates."""
    vertices, _, _ = surface("foot")
    soles = np.full((len(points), 2), np.nan)
    for shape, name, (position, rotation, scale) in parts(points):
        if shape != "foot":
            continue
        x, y, z, w = rotation.T
        vertical = np.column_stack((2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)))
        low = position[:, 1] + ((vertical * scale) @ vertices.T).min(axis=1)
        soles[:, int(name == "Right foot")] = np.where(scale[:, 1] > 0, low, np.nan)
    return soles


def floor_track(points, pixels, size, times):
    """Place the lowest observed shoe on Y=0. A higher visible foot during
    occlusion is not evidence of support. Unresolved frames carry a height for
    display continuity, never a contact label.
    """
    soles = sole_heights(points)
    image_bottom = np.full_like(soles, np.nan)
    for side in (0, 1):
        obs = pixels[:, [27 + side, 29 + side, 31 + side]]
        inside = (np.isfinite(obs) & (obs >= 0) & (obs < size)).all(axis=(1, 2))
        soles[~inside, side] = np.nan
        image_bottom[:, side] = np.where(np.isfinite(soles[:, side]), obs[:, 1:, 1].max(axis=1), np.nan)

    both = np.isfinite(soles).all(axis=1)
    support = np.full(len(points), -1, dtype=np.int8)
    height = np.full(len(points), np.nan)
    for i, row in enumerate(soles):
        if not np.isfinite(row).any():
            continue
        side = int(np.nanargmin(row))
        if not both[i]:
            # Look for this same shoe near a recent or upcoming known support
            # position. Comparing each foot to itself avoids depth bias.
            near = both & (abs(times - times[i]) <= .35) & (soles[:, side] <= soles[:, 1-side] + .025)
            baseline = image_bottom[near, side]
            if not baseline.size or image_bottom[i, side] < np.median(baseline) - .016 * size[1]:
                continue
        height[i], support[i] = -row[side], side

    valid = np.flatnonzero(support >= 0)
    if not len(valid):
        return np.zeros(len(times)), support
    # Hold through long unknown gaps. Blend only across short tracking losses.
    for i in np.flatnonzero(support < 0):
        j = np.searchsorted(valid, i)
        left, right = valid[max(0, j-1)], valid[min(len(valid)-1, j)]
        if left < i < right and times[right] - times[left] <= .12:
            height[i] = np.interp(times[i], times[[left, right]], height[[left, right]])
        else:
            height[i] = height[left]
    return height, support
