import os
import argparse
import sys
import numpy as np
from typing import List, Dict

from extract_rallies import (
    TennisRallyProcessor,
    RallySegment,
    get_video_date,
    MIN_RALLY_DURATION_SEC,
    GAP_MERGE_THRESHOLD_SEC,
    ARM_ENERGY_THRESHOLD,
    WHIP_RATIO_THRESHOLD,
    MOTION_ENERGY_THRESHOLD,
    PROCESS_EVERY_N_FRAMES,
    BOUNDARY_PADDING_SEC
)

def generate_serve_name(video_path: str) -> str:
    """
    Generates name in format: {Date(MMDDYY)}_{OriginalName}_serve
    """
    date_str = get_video_date(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{date_str}_{video_stem}_serve"


class ServePracticeProcessor(TennisRallyProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_folder_name = "serves"
        self.output_csv_name = "serves.csv"
        self.clip_prefix = "_serve_"

    def get_output_filename(self) -> str:
        """
        Returns output filename based on video input: {Date(MMDDYY)}_{VideoStem}_serve.mp4
        """
        base_name = generate_serve_name(self.video_path)
        return f"{base_name}.mp4"

    def analyze_segment_data(self, frame_data: List[Dict]) -> List[RallySegment]:
        """
        Apply heuristics specific for Serving Practice.
        Serves have lower hip_vel and court coverage but high arm energy and wrist whip.
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
        
        # Inject smoothed values for telemetry dump
        for i, d in enumerate(frame_data):
            d['smoothed_arm'] = smoothed_arm[i]
            d['smoothed_whip'] = smoothed_whip[i]
        
        self.all_frame_metrics.extend(frame_data)
        
        valid_pose = np.array([d.get('is_valid_pose', False) for d in frame_data], dtype=bool)
        
        # Segment primarily by Valid Pose (MediaPipe tracking)
        is_serve_segment = valid_pose
        
        # Sub-segments
        sub_gap_frames = int(GAP_MERGE_THRESHOLD_SEC * effective_fps)
        segments = self._mask_to_indices(is_serve_segment.astype(int))
        
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
            
            # For serve, arm energy is high, whip is high. The player is usually at baseline.
            MAX_SERVE_PLAYER_SCALE = 0.28
            max_height = np.max(segment_heights)
            
            # Valid Baseline Serve
            valid_baseline_strokes = (
                (segment_arms > ARM_ENERGY_THRESHOLD) &
                (segment_whips > WHIP_RATIO_THRESHOLD) &
                (segment_heights <= MAX_SERVE_PLAYER_SCALE)
            )
            
            # Valid Close Serve (Scale > 0.28)
            VOLLEY_ARM_THRESHOLD = max(1.5, ARM_ENERGY_THRESHOLD * 1.2)
            VOLLEY_WHIP_THRESHOLD = max(1.2, WHIP_RATIO_THRESHOLD)
            
            valid_volley_strokes = (
                (segment_arms > VOLLEY_ARM_THRESHOLD) &
                (segment_whips > VOLLEY_WHIP_THRESHOLD) &
                (segment_heights > MAX_SERVE_PLAYER_SCALE)
            )
            
            segment_energies = smoothed_energy[start_idx:end_idx]
            peak_energy = np.max(segment_energies)
            
            if not (np.any(valid_baseline_strokes) or np.any(valid_volley_strokes)):
                max_whip = np.max(segment_whips) if len(segment_whips) > 0 else 0
                if max_height > MAX_SERVE_PLAYER_SCALE:
                    print(f"  [Serve Reject] Scale={max_height:.2f} > {MAX_SERVE_PLAYER_SCALE} but weak strokes (Peak Arm: {peak_arm_score:.2f}, Peak Whip: {max_whip:.2f}). Walking to camera.")
                else:
                    print(f"  [Serve Reject] No frame passed double-check. (Peak Arm: {peak_arm_score:.2f}, Peak Whip: {max_whip:.2f})")
                continue
            
            if peak_energy < MOTION_ENERGY_THRESHOLD * 2:
                 print(f"  [Serve Reject] Energy={peak_energy:.5f} < {MOTION_ENERGY_THRESHOLD * 2} (Dur={end_idx-start_idx})")
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

def main():
    parser = argparse.ArgumentParser(description="Extract tennis serves from video.")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", help="Path to output directory (default: current directory)", default=".")
    parser.add_argument("--split", action="store_true", help="Keep individual serve clips instead of just the merged video")
    parser.add_argument("--plot", action="store_true", help="Generate movement telemetry plot")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video not found at {args.video_path}")
        sys.exit(1)
        
    if args.output == ".":
        base_name = generate_serve_name(args.video_path)
        args.output = os.path.join(".", base_name)
        
    os.makedirs(args.output, exist_ok=True)
    
    processor = ServePracticeProcessor(args.video_path, args.output, keep_clips=args.split, generate_plot=args.plot)
    processor.run()

if __name__ == "__main__":
    main()
