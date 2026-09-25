import cv2
import mediapipe as mp
from typing import Optional
from really_rally.core.config import *

class MotionDetector:
    """
    Lightweight pixel-based motion detector to skip empty scenes
    before running expensive Pose Estimation.
    """
    def __init__(self, width, height, diff_threshold=PIXEL_DIFF_THRESHOLD, change_threshold=PIXEL_CHANGE_THRESHOLD):
        self.width = width
        self.height = height
        self.prev_gray = None
        self.diff_threshold = diff_threshold
        self.change_threshold = change_threshold
        
        # Downscale for speed
        self.scale = 0.1 # 10% size
        self.w_small = int(width * self.scale)
        self.h_small = int(height * self.scale)
    
    def has_motion(self, frame_bgr) -> bool:
        # Resize
        small = cv2.resize(frame_bgr, (self.w_small, self.h_small), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return True # Assume motion on first frame
            
        # Absdiff
        delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Count changed pixels
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = self.w_small * self.h_small
        fraction = changed_pixels / total_pixels
        
        self.prev_gray = gray
        return fraction > self.change_threshold

class PoseExtractor:
    """Wrapper around MediaPipe Pose."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF
        )

    def process_frame(self, frame_rgb):
        return self.pose.process(frame_rgb)

    def get_dominant_player(self, landmarks_list, prev_center=None) -> Optional[any]:
        return landmarks_list

    def close(self):
        self.pose.close()


_CACHED_POSE_EXTRACTOR: Optional[PoseExtractor] = None

# MediaPipe ships only the "full" (complexity=1) weights inside the wheel. The
# lite (0) and heavy (2) variants are fetched from GitHub on first use, so
# those settings need one-time network access.
_MODEL_DOWNLOAD_HINT = """\
MediaPipe could not load the pose model for MP_MODEL_COMPLEXITY={complexity}.

Only the 'full' model (MP_MODEL_COMPLEXITY = 1) ships inside the mediapipe
wheel. Complexity 0 (lite) and 2 (heavy) are downloaded from GitHub on first
use, so they need one-time network access. The underlying failure was:

    {cause}

Remedies, in order of preference:
  1. Use the bundled model: set MP_MODEL_COMPLEXITY = 1 in core/config.py.
  2. If you are behind a corporate proxy or a TLS-inspecting network, fix the
     certificate chain so Python can verify it, then retry.
  3. Last resort, in a trusted environment only: set the environment variable
     REALLY_RALLY_ALLOW_INSECURE_SSL=1 to skip certificate verification for the
     download. This disables TLS verification process-wide and is not safe on
     untrusted networks.
"""


def get_pose_extractor() -> PoseExtractor:
    """Return a process-wide shared PoseExtractor, building it on first use.

    Constructing a MediaPipe graph costs a few hundred milliseconds. The
    pipeline previously built one per processor and again per motion segment,
    which dominated startup and made the GUI look frozen before the first
    progress event. The graph is stateless between calls to ``process()`` for
    our purposes (tracking state carries at most one segment's worth of stale
    landmarks, and segments are separated by multi-second gaps anyway).

    Raises RuntimeError with actionable guidance if the model weights are
    missing and cannot be downloaded; a raw SSL traceback here is otherwise
    very hard to interpret from the GUI.
    """
    global _CACHED_POSE_EXTRACTOR
    if _CACHED_POSE_EXTRACTOR is None:
        try:
            _CACHED_POSE_EXTRACTOR = PoseExtractor()
        except Exception as exc:  # download/SSL/decoder failures surface here
            raise RuntimeError(
                _MODEL_DOWNLOAD_HINT.format(
                    complexity=MP_MODEL_COMPLEXITY,
                    cause=f"{type(exc).__name__}: {exc}",
                )
            ) from exc
    return _CACHED_POSE_EXTRACTOR

