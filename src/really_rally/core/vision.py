import cv2
import numpy as np
import mediapipe as mp
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

