#!/usr/bin/env python3
"""
Debug Classifier for Tennis Rallies
Focus: Distinguishing "Ball Picking" from "Rally" using Swing-to-Run Ratio.
Hypothesis: Rallies involve high arm velocity while feet/hips are relatively planted.
            Ball Picking involves high arm velocity correlated with high running velocity.
"""

import sys
import cv2
import numpy as np
import ssl

# Robust MediaPipe Import
try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        import mediapipe.python.solutions as mp_solutions
        mp.solutions = mp_solutions
except (ImportError, AttributeError) as e:
    print("\n" + "="*80)
    print("ERROR: MediaPipe Solutions API not found.")
    print("This script uses the legacy MediaPipe Solutions API (mp.solutions.pose).")
    print("Your current environment seems to have a version of MediaPipe that doesn't support this.")
    print("\nSUGGESTED FIX:")
    print("1. Use the provided virtual environment: source .venv/bin/activate && python3 debug_classifier.py ...")
    print("2. Or install the correct version: pip install 'mediapipe<0.11.0'")
    print("="*80 + "\n")
    sys.exit(1)

# Fix for SSL certificate verify failed on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Constants
MP_MODEL_COMPLEXITY = 1
SWING_RATIO_THRESHOLD = 3.0      # Ratio of Arm/Hip velocity
PROCESS_EVERY_N_FRAMES = 2       # Sample frequently (every 2 frames ~ 33ms at 60fps, 66ms at 30fps)

class DebugPoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process(self, frame_rgb):
        return self.pose.process(frame_rgb)

def analyze_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extractor = DebugPoseExtractor()
    
    prev_landmarks = None
    
    max_ratio = 0.0
    frames_above_threshold = 0
    
    # Store energy profile
    ratio_profile = []

    frame_idx = 0
    
    print(f"\nAnalyzing: {video_path} ({total_frames} frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            # Resize for speed
            h, w = frame.shape[:2]
            target_h = 360
            scale = target_h / h
            w_new = int(w * scale)
            frame_resized = cv2.resize(frame, (w_new, target_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                 cv2.imwrite("debug_classifier_frame.jpg", frame_resized)
                 print(f"  [Debug] Saved debug_classifier_frame.jpg ({w_new}x{target_h})")
            
            results = extractor.process(frame_rgb)
            
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                curr_coords = np.array([[lm.x, lm.y] for lm in lms])
                
                if prev_landmarks is not None:
                     # Calculate Shoulder Width (11-12) for normalization
                    shoulder_width = np.linalg.norm(curr_coords[11] - curr_coords[12])
                    if shoulder_width < 0.01: shoulder_width = 0.01 
                    
                    # Calculate velocity per joint
                    diff = curr_coords - prev_landmarks
                    velocities = np.linalg.norm(diff, axis=1)
                    velocities = velocities / shoulder_width
                    
                    # Hip Velocity (for "Running" detection)
                    # 23: Left Hip, 24: Right Hip
                    hip_vel = (velocities[23] + velocities[24]) / 2.0
                    
                    # Arm Velocity
                    # 13: Left Elbow, 14: Right Elbow
                    # 15: Left Wrist, 16: Right Wrist
                    left_arm_energy = (velocities[13] + velocities[15]) / 2.0
                    right_arm_energy = (velocities[14] + velocities[16]) / 2.0
                    arm_energy = max(left_arm_energy, right_arm_energy)
                    
                    # Ratio: Arm Swing relative to Body Movement
                    # Epsilon to avoid noise when standing still (0.02 is arbitrary but small)
                    swing_ratio = arm_energy / (hip_vel + 0.02)
                    
                    ratio_profile.append(swing_ratio)
                    
                    if swing_ratio > max_ratio:
                        max_ratio = swing_ratio
                    
                    if swing_ratio > SWING_RATIO_THRESHOLD:
                        frames_above_threshold += 1
                        
                prev_landmarks = curr_coords
            
            if frame_idx % 100 == 0:
                print(f"\rProgress: {frame_idx}/{total_frames}", end="")

        frame_idx += 1
    
    cap.release()
    print("\nDone.")
    
    # Analyze Peaks (Swing Detection)
    peaks = 0
    if len(ratio_profile) > 2:
        for i in range(1, len(ratio_profile)-1):
            curr = ratio_profile[i]
            prev = ratio_profile[i-1]
            next_val = ratio_profile[i+1]
            
            if curr > SWING_RATIO_THRESHOLD and curr > prev and curr > next_val:
                peaks += 1

    print(f"--- Results for {video_path} ---")
    print(f"Max Swing Ratio: {max_ratio:.6f}")
    print(f"Frames > Threshold ({SWING_RATIO_THRESHOLD}): {frames_above_threshold}")
    print(f"Detected Swings (Peaks): {peaks}")
    
    classification = "UNKNOWN"
    if peaks >= 1:
        classification = "RALLY (Clean Swing Detected)"
    else:
        classification = "BALL PICKING / WALKING"
        
    print(f"Classification: {classification}")
    print("-" * 30)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 debug_classifier.py <video1> <video2> ...")
        sys.exit(1)
        
    for video_path in sys.argv[1:]:
        analyze_clip(video_path)

if __name__ == "__main__":
    main()
