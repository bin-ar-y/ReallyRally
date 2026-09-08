import os
import cv2
import numpy as np
from typing import List, Dict

from really_rally.processors.base import TennisRallyProcessor
from really_rally.core.models import RallySegment
from really_rally.utils.media_io import get_video_date
from really_rally.core.config import (
    MIN_RALLY_DURATION_SEC,
    GAP_MERGE_THRESHOLD_SEC,
    MOTION_ENERGY_THRESHOLD,
    PROCESS_EVERY_N_FRAMES,
    BOUNDARY_PADDING_SEC,
    MIN_COURT_COVERAGE,
    ADAPTIVE_WINDOW_SEC,
    ARM_ENERGY_K_MULTIPLIER,
    MIN_ENERGY_FLOOR,
    WHIP_RATIO_K_MULTIPLIER,
    MIN_WHIP_FLOOR,
    PLAYER_HEIGHT_MIN,
    PLAYER_HEIGHT_MAX,
    FEED_HIP_VELOCITY_MIN
)

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
        self.output_folder_name = "feeds"
        self.output_csv_name = "feeds.csv"
        self.clip_prefix = "_feed_"
        
        # Adaptive scale boundaries (Specific for feeding mode to ignore coach)
        self.min_player_height = PLAYER_HEIGHT_MIN
        self.max_player_height = PLAYER_HEIGHT_MAX 

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
        
    def analyze_segment_data(self, frame_data: List[Dict]) -> List[RallySegment]:
        """
        Apply heuristics to segment. Keep the rigid player_height filter to ignore feeders.
        """
        n = len(frame_data)
        if n == 0: return []
        
        effective_fps = self.fps / PROCESS_EVERY_N_FRAMES
        
        energies = np.array([d.get('energy', 0) for d in frame_data])
        arm_energies = np.array([d.get('arm_energy', 0) for d in frame_data])
        hip_vels = np.array([d.get('hip_vel', 0) for d in frame_data])
        whips = np.array([d.get('whip_ratio', 0) for d in frame_data])
        heights = np.array([d.get('player_height', 0) for d in frame_data])
        
        # Smooth
        window = int(effective_fps * 0.5)
        if window < 1: window = 1
        kernel = np.ones(window)/window
        
        smoothed_energy = np.convolve(energies, kernel, mode='same')
        smoothed_arm = np.convolve(arm_energies, kernel, mode='same')
        smoothed_whip = np.convolve(whips, kernel, mode='same')
        
        # Initialize Adaptive Thresholds
        from really_rally.core.features import SelfAdaptiveThreshold
        arm_adaptive = SelfAdaptiveThreshold(ADAPTIVE_WINDOW_SEC, effective_fps, ARM_ENERGY_K_MULTIPLIER, MIN_ENERGY_FLOOR)
        whip_adaptive = SelfAdaptiveThreshold(ADAPTIVE_WINDOW_SEC, effective_fps, WHIP_RATIO_K_MULTIPLIER, MIN_WHIP_FLOOR)
        
        # Pre-calculate threshold arrays for the segment
        t_arm_arr = np.zeros(n)
        t_whip_arr = np.zeros(n)
        
        # Inject smoothed values for telemetry dump and update adaptive thresholds
        for i, d in enumerate(frame_data):
            d['smoothed_arm'] = smoothed_arm[i]
            d['smoothed_whip'] = smoothed_whip[i]
            
            arm_adaptive.update(smoothed_arm[i])
            whip_adaptive.update(smoothed_whip[i])
            
            t_arm_arr[i] = arm_adaptive.get_threshold()
            t_whip_arr[i] = whip_adaptive.get_threshold()
            
            d['t_arm'] = t_arm_arr[i]
            d['t_whip'] = t_whip_arr[i]
        
        self.all_frame_metrics.extend(frame_data)
        
        valid_pose = np.array([d.get('is_valid_pose', False) for d in frame_data], dtype=bool)
        
        # Segment primarily by Valid Pose (MediaPipe tracking)
        is_rally = valid_pose
        
        # Sub-segments
        sub_gap_frames = int(GAP_MERGE_THRESHOLD_SEC * effective_fps)
        segments = self._mask_to_indices(is_rally.astype(int))
        
        # Merge
        merged_segments = []
        if not segments: return []
        
        curr_start, curr_end = segments[0]
        for i in range(1, len(segments)):
            next_start, next_end = segments[i]
            if next_start - curr_end < sub_gap_frames:
                curr_end = next_end
            else:
                merged_segments.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_segments.append((curr_start, curr_end))
        
        final_segments = []
        min_frames = int(MIN_RALLY_DURATION_SEC * effective_fps)
        
        for start_idx, end_idx in merged_segments:
            if (end_idx - start_idx) < min_frames:
                continue
                
            # Gating Logic (Simultaneous double-check: Arm Energy, Wrist Whip, and Baseline Camera Scale)
            segment_arms = smoothed_arm[start_idx:end_idx]
            peak_arm_score = np.max(segment_arms)
            segment_whips = smoothed_whip[start_idx:end_idx]
            segment_heights = heights[start_idx:end_idx]
            segment_t_arm = t_arm_arr[start_idx:end_idx]
            segment_t_whip = t_whip_arr[start_idx:end_idx]
            max_height = np.max(segment_heights)
            
            # STRICT FILTER: Discard the motion if the person bounding box is huge (e.g. Coach feeding)
            valid_baseline_strokes = (
                (segment_arms > segment_t_arm) &
                (segment_whips > segment_t_whip) &
                (segment_heights > self.min_player_height) & 
                (segment_heights <= self.max_player_height)
            )
            
            VOLLEY_ARM_THRESHOLD = np.maximum(1.5, segment_t_arm * 1.2)
            VOLLEY_WHIP_THRESHOLD = np.maximum(1.2, segment_t_whip)
            
            valid_volley_strokes = (
                (segment_arms > VOLLEY_ARM_THRESHOLD) &
                (segment_whips > VOLLEY_WHIP_THRESHOLD) &
                (segment_heights > self.max_player_height)
            )
            
            segment_energies = smoothed_energy[start_idx:end_idx]
            peak_energy = np.max(segment_energies)
            
            segment_hips = hip_vels[start_idx:end_idx]
            avg_hip_vel = np.mean(segment_hips)

            if not (np.any(valid_baseline_strokes) or np.any(valid_volley_strokes)):
                max_whip = np.max(segment_whips) if len(segment_whips) > 0 else 0
                if max_height > self.max_player_height:
                    print(f"  [Feed Reject] Scale={max_height:.2f} > {self.max_player_height:.2f} but weak strokes (Peak Arm: {peak_arm_score:.2f}, Peak Whip: {max_whip:.2f}). Walking to camera/Coach feeding.")
                else:
                    print(f"  [Feed Reject] No frame passed double-check/scale. (Peak Arm: {peak_arm_score:.2f}, Peak Whip: {max_whip:.2f})")
                continue
            
            if avg_hip_vel < FEED_HIP_VELOCITY_MIN:
                print(f"  [Feed Reject] AvgHip={avg_hip_vel:.3f} < {FEED_HIP_VELOCITY_MIN} (Ball Picking/Standing)")
                continue

            if peak_energy < MOTION_ENERGY_THRESHOLD * 2:
                 print(f"  [Feed Reject] Energy={peak_energy:.5f} < {MOTION_ENERGY_THRESHOLD * 2}")
                 continue

            # Feeder/Coach Rejection Logic (Spatial Variance)
            segment_variances = [d.get('spatial_variance', 0) for d in frame_data[start_idx:end_idx]]
            max_variance = np.max(segment_variances) if segment_variances else 0
            
            if max_variance < MIN_COURT_COVERAGE:
                 print(f"  [Feed Reject] Stationary Feeder. Variance={max_variance:.3f} < {MIN_COURT_COVERAGE}")
                 continue
                 
            # Map back to original timestamps
            start_feat = frame_data[start_idx]
            end_feat = frame_data[end_idx-1]
            
            t_start = max(0, start_feat['timestamp'] - BOUNDARY_PADDING_SEC)
            t_end = min(self.total_frames / self.fps, end_feat['timestamp'] + BOUNDARY_PADDING_SEC)
            
            segment = RallySegment(
                start_frame=int(t_start * self.fps),
                end_frame=int(t_end * self.fps),
                start_time=t_start,
                end_time=t_end,
                score=peak_arm_score
            )
            final_segments.append(segment)
            
        return final_segments



