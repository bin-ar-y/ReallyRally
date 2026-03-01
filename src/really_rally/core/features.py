import numpy as np
import collections
from typing import Dict
from really_rally.core.config import *

class MovementTracker:
    """
    Tracks the spatial history of a subject to reject stationary feeders.
    Calculates variance of position over a time window.
    """
    def __init__(self, window_duration_sec: float, fps: float):
        self.capacity = int(window_duration_sec * fps)
        self.history = collections.deque(maxlen=self.capacity)
        
    def update(self, x: float, y: float):
        self.history.append((x, y))
        
    def get_spatial_variance(self) -> float:
        """
        Returns a metric of how much ground has been covered.
        Using Standard Deviation of (x, y).
        """
        if len(self.history) < 2:
            return 0.0
            
        points = np.array(self.history)
        # Standard deviation in X and Y
        std_x = np.std(points[:, 0])
        std_y = np.std(points[:, 1])
        
        # Combined metric (Euclidean-ish or Sum)
        # Sqrt(std_x^2 + std_y^2) gives "average distance from mean center"
        variance = np.sqrt(std_x**2 + std_y**2)
        return variance
    
    def draw_debug_path(self, frame, color=(0, 255, 0)):
        """Draws the movement path on the frame."""
        if len(self.history) < 2:
            return
            
        h, w = frame.shape[:2]
        
        # Convert normalized [(x,y)] to pixel [(px, py)]
        pts = []
        for nx, ny in self.history:
            px = int(nx * w)
            py = int(ny * h)
            pts.append([px, py])
            
        pts_np = np.array(pts, np.int32)
        pts_np = pts_np.reshape((-1, 1, 2))
        
        cv2.polylines(frame, [pts_np], False, color, 2)
        # Draw current head
        cv2.circle(frame, tuple(pts[-1]), 4, color, -1)

class FeatureEngine:
    """Computes biomechanical features from pose landmarks."""
    def __init__(self, fps):
        self.fps = fps
        self.prev_landmarks = None
        
        # Buffers for smoothing
        self.energy_buffer = []
    
    def compute_features(self, landmarks, frame_timestamp: float) -> Dict[str, float]:
        """
        Computes features for the current frame based on previous frame.
        Landmarks: MediaPipe NormalizedLandmarkList (or None)
        """
        features = {
            'timestamp': frame_timestamp,
            'is_valid_pose': False,
            'energy': 0.0,
            'lat_v': 0.0,
            'vert_v': 0.0,
            'action_score': 0.0,
            'hip_vel': 0.0,
            'whip_ratio': 0.0,
            'arm_energy': 0.0,
            'player_height': 0.0
        }

        if landmarks is None:
            self.prev_landmarks = None
            return features
        
        # Extract (x, y) for all 33 landmarks
        lms = landmarks.landmark
        curr_coords = np.array([[lm.x, lm.y] for lm in lms])
        features['is_valid_pose'] = True
        
        # Center of mass approximation (Mid-hip)
        # 23: Left Hip, 24: Right Hip
        hip_center = (curr_coords[23] + curr_coords[24]) / 2.0
        features['hip_center_x'] = hip_center[0]
        features['hip_center_y'] = hip_center[1]
        
        # Player Normalized Height (Nose to Ankle)
        nose_y = curr_coords[0][1]
        ankle_y = (curr_coords[27][1] + curr_coords[28][1]) / 2.0
        features['player_height'] = abs(ankle_y - nose_y)

        if self.prev_landmarks is not None:
            # Calculate Shoulder Width (11-12) for normalization
            shoulder_width = np.linalg.norm(curr_coords[11] - curr_coords[12])
            if shoulder_width < 0.01: shoulder_width = 0.01 
            
            # Calculate velocities (Euclidean distance per joint)
            diff = curr_coords - self.prev_landmarks
            velocities = np.linalg.norm(diff, axis=1) # Shape (33,)
            
            # Global Motion Energy (Raw)
            features['energy'] = np.mean(velocities)
            
            # Normalize Velocities by Shoulder Width for Swing Analysis
            norm_velocities = velocities / shoulder_width
            
            # Swing Analysis
            # Hip Velocity (Running and Footwork)
            hip_vel = (norm_velocities[23] + norm_velocities[24]) / 2.0
            features['hip_vel'] = hip_vel
            
            # Arm Velocity (Elbow + Wrist)
            left_arm = (norm_velocities[13] + norm_velocities[15]) / 2.0
            right_arm = (norm_velocities[14] + norm_velocities[16]) / 2.0
            arm_energy = max(left_arm, right_arm)
            
            # Whip Ratio (Wrist vs Elbow speed)
            max_wrist = max(norm_velocities[15], norm_velocities[16])
            max_elbow = max(norm_velocities[13], norm_velocities[14])
            features['whip_ratio'] = max_wrist / (max_elbow + 0.01)
            features['arm_energy'] = arm_energy
            
            # True Action Score
            # High Arm + High Hip (swinging and moving) -> Rally
            # High Arm + Low Hip (standing still and reaching down) -> Picking
            features['action_score'] = arm_energy * (hip_vel + 0.1)
            
            # Hip Velocity for Lateral/Vertical decomposition (Raw coords)
            hip_diff = hip_center - ((self.prev_landmarks[23] + self.prev_landmarks[24]) / 2.0)
            features['lat_v'] = abs(hip_diff[0])
            features['vert_v'] = abs(hip_diff[1])
            
        self.prev_landmarks = curr_coords
        return features

