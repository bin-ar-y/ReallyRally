"""Shared test helpers — mock landmarks, temp files, etc."""

from typing import List, Tuple


class MockLandmark:
    """Minimal mock of a MediaPipe NormalizedLandmark."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class MockLandmarkList:
    """Minimal mock of a MediaPipe NormalizedLandmarkList."""

    def __init__(self, landmarks: List[MockLandmark]):
        self.landmark = landmarks


def make_landmarks(coords: List[Tuple[float, float]]) -> List[MockLandmark]:
    """Create a list of 33 mocked landmarks.

    Fills unspecified indices with (0.5, 0.5).
    """
    lmks = [MockLandmark(0.5, 0.5) for _ in range(33)]
    for i, (x, y) in enumerate(coords):
        if i < 33:
            lmks[i] = MockLandmark(x, y)
    return lmks


def make_stationary_landmarks() -> List[MockLandmark]:
    """Return a default set of 33 landmarks for a stationary standing pose."""
    return make_landmarks([
        (0.50, 0.10),  # 0: nose
        (0.48, 0.12),  # 1: left eye inner
        (0.48, 0.11),  # 2: left eye
        (0.47, 0.12),  # 3: left eye outer
        (0.52, 0.12),  # 4: right eye inner
        (0.52, 0.11),  # 5: right eye
        (0.53, 0.12),  # 6: right eye outer
        (0.48, 0.14),  # 7: left ear
        (0.52, 0.14),  # 8: right ear
        (0.50, 0.18),  # 9: mouth left
        (0.50, 0.17),  # 10: mouth right
        (0.44, 0.25),  # 11: left shoulder
        (0.56, 0.25),  # 12: right shoulder
        (0.42, 0.35),  # 13: left elbow
        (0.58, 0.35),  # 14: right elbow
        (0.38, 0.45),  # 15: left wrist
        (0.62, 0.45),  # 16: right wrist
        (0.45, 0.45),  # 17: left pinky
        (0.55, 0.45),  # 18: right pinky
        (0.46, 0.44),  # 19: left index
        (0.54, 0.44),  # 20: right index
        (0.47, 0.43),  # 21: left thumb
        (0.53, 0.43),  # 22: right thumb
        (0.48, 0.55),  # 23: left hip
        (0.52, 0.55),  # 24: right hip
        (0.47, 0.65),  # 25: left knee
        (0.53, 0.65),  # 26: right knee
        (0.46, 0.75),  # 27: left ankle
        (0.54, 0.75),  # 28: right ankle
        (0.48, 0.80),  # 29: left heel
        (0.52, 0.80),  # 30: right heel
        (0.47, 0.82),  # 31: left foot index
        (0.53, 0.82),  # 32: right foot index
    ])
