import os
import cv2
import numpy as np
from typing import List, Dict, Optional

from really_rally.processors.base import TennisRallyProcessor
from really_rally.core.models import RallySegment
from really_rally.utils.media_io import get_video_date
from really_rally.core.config import PLAYER_HEIGHT_MAX, PLAYER_HEIGHT_MIN
from really_rally.core.detection import FeedAnalyzer

def generate_feed_name(video_path: str) -> str:
    """
    Generates name in format: {Date(MMDDYY)}_{OriginalName}_feed
    """
    date_str = get_video_date(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{date_str}_{video_stem}_feed"


class FeedPracticeProcessor(TennisRallyProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_folder_name = "feed"
        self.output_csv_name = "feed.csv"
        self.clip_prefix = "_feed_"
        
        # Adaptive scale boundaries (Specific for feeding mode to ignore coach)
        self.min_player_height = PLAYER_HEIGHT_MIN
        self.max_player_height = PLAYER_HEIGHT_MAX
        self._feed_analyzer: Optional[FeedAnalyzer] = None

    def get_output_filename(self) -> str:
        """
        Returns output filename based on video input: {Date(MMDDYY)}_{VideoStem}_feed.mp4
        """
        base_name = generate_feed_name(self.video_path)
        return f"{base_name}.mp4"

    def estimate_player_scale(self, num_samples=40):
        """
        Samples the video to find the median player height for this specific
        camera setup, allowing the system to self-adapt and reject when the
        player gets too close to the camera (e.g. the coach feeding the balls).
        """
        print("--- Pre-Scan (Feed Mode): Estimating Adaptive Player Scale Geometry ---")
        
        if self.total_frames <= 0:
            return
            
        step = max(1, self.total_frames // num_samples)
        heights = []
        
        for i in range(num_samples):
            frame_idx = i * step
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret: continue
            
            # Resize
            h, w = frame.shape[:2]
            target_h = 360
            scale = target_h / float(h)
            w_new = int(w * scale)
            proc_frame = cv2.resize(frame, (w_new, target_h), interpolation=cv2.INTER_LINEAR)
            
            results = self.pose_extractor.process_frame(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                nose_y = lms[0].y
                ankle_y = (lms[27].y + lms[28].y) / 2.0
                height = abs(ankle_y - nose_y)
                heights.append(height)
                
            if self.progress_callback:
                self.progress_callback(0.05 * (i / num_samples), "Estimating Camera Scale...")
                
        if heights:
            med_h = np.median(heights)
            self.min_player_height = max(0.02, med_h * 0.15)
            self.max_player_height = min(0.95, med_h * 1.6)
            print("Camera Baseline Calibrated:")
            print(f"  Median Height: {med_h:.3f}")
            print(f"  Valid Stroke Range: {self.min_player_height:.3f} to {self.max_player_height:.3f}")
        else:
            print("Warning: Could not detect player scale during initialization. Using default bounds.")
            self.max_player_height = PLAYER_HEIGHT_MAX
            
        # Rewind to the beginning for Pass 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, _ = self.cap.read()
        if not ret:
            # If rewind fails (common on some backends), reopen the stream
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def scan_motion_segments(self) -> List[tuple]:
        """
        Override scan_motion_segments to ensure we calculate the player scale
        before proceeding to the motion scan.
        """
        self.estimate_player_scale()
        # Call the parent's scan_motion_segments, which will then do the frame skipping logic
        return super().scan_motion_segments()
        
    def _get_feed_analyzer(self) -> FeedAnalyzer:
        if self._feed_analyzer is None:
            self._feed_analyzer = FeedAnalyzer(
                self.config_snapshot(),
                # The calibrated scale bounds come from a video pre-scan, so they
                # are passed in rather than read from config.
                min_player_height=self.min_player_height,
                max_player_height=self.max_player_height,
                total_frames=self.total_frames,
            )
        return self._feed_analyzer

    def analyze_segment_data(self, frame_data: List[Dict]) -> List[RallySegment]:
        """Detect feeds, delegating to ``core.detection.FeedAnalyzer``.

        The rigid player-height filter that rejects the coach/feeder is part of
        the analyzer, so replay applies the same gate.
        """
        if not frame_data:
            return []
        segments = self._get_feed_analyzer().analyze(frame_data)
        self.all_frame_metrics.extend(frame_data)
        return segments
