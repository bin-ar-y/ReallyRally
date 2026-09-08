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

def _joint_angle(a, b, c):
    """Angle (degrees) at vertex `b` formed by the three points a-b-c.

    Points are in MediaPipe-normalized coordinates. Non-uniform axis scaling
    distorts absolute angles, but collinearity (a straight arm) is preserved,
    which is the only property serve detection relies on.
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 180.0
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


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
            'player_height': 0.0,
            # Per-arm serve features (MediaPipe-normalized coordinates).
            'nose_y': 0.0,
            'l_shoulder_y': 0.0, 'r_shoulder_y': 0.0,
            'l_wrist_x': 0.0, 'l_wrist_y': 0.0,
            'r_wrist_x': 0.0, 'r_wrist_y': 0.0,
            'l_elbow_x': 0.0, 'l_elbow_y': 0.0,
            'r_elbow_x': 0.0, 'r_elbow_y': 0.0,
            'l_wrist_vel': 0.0, 'r_wrist_vel': 0.0,
            'l_elbow_vel': 0.0, 'r_elbow_vel': 0.0,
            'l_wrist_vy': 0.0, 'r_wrist_vy': 0.0,
            'l_elbow_angle': 180.0, 'r_elbow_angle': 180.0,
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

        # Per-arm serve features. Image y increases downward, so a wrist with
        # y < shoulder_y is raised above the shoulder.
        features['nose_y'] = nose_y
        features['l_shoulder_y'] = curr_coords[11][1]
        features['r_shoulder_y'] = curr_coords[12][1]
        features['l_wrist_x'] = curr_coords[15][0]
        features['l_wrist_y'] = curr_coords[15][1]
        features['r_wrist_x'] = curr_coords[16][0]
        features['r_wrist_y'] = curr_coords[16][1]
        features['l_elbow_x'] = curr_coords[13][0]
        features['l_elbow_y'] = curr_coords[13][1]
        features['r_elbow_x'] = curr_coords[14][0]
        features['r_elbow_y'] = curr_coords[14][1]

        # Elbow extension (shoulder -> elbow -> wrist). 180 = fully straight.
        features['l_elbow_angle'] = _joint_angle(curr_coords[11], curr_coords[13], curr_coords[15])
        features['r_elbow_angle'] = _joint_angle(curr_coords[12], curr_coords[14], curr_coords[16])

        if self.prev_landmarks is not None:
            # Normalize using pose bounding-box diagonal for scale invariance.
            # This is more stable than torso length because it uses all 33
            # keypoints, making it resilient to single-joint jitter.
            bbox_min = curr_coords.min(axis=0)
            bbox_max = curr_coords.max(axis=0)
            bbox_diag = np.linalg.norm(bbox_max - bbox_min)
            if bbox_diag < 0.01: bbox_diag = 0.01
            
            # Calculate velocities (Euclidean distance per joint)
            diff = curr_coords - self.prev_landmarks
            velocities = np.linalg.norm(diff, axis=1) # Shape (33,)
            
            # Global Motion Energy (Raw)
            features['energy'] = np.mean(velocities)
            
            # Normalize Velocities by Bounding-Box Diagonal
            norm_velocities = velocities / bbox_diag
            
            # Swing Analysis
            # Hip Velocity (Running and Footwork)
            hip_vel = (norm_velocities[23] + norm_velocities[24]) / 2.0
            features['hip_vel'] = hip_vel
            
            # E_a(t) = (v_wrist / D_torso)^2 + (v_elbow / D_torso)^2
            left_arm_energy = norm_velocities[15]**2 + norm_velocities[13]**2
            right_arm_energy = norm_velocities[16]**2 + norm_velocities[14]**2
            arm_energy = max(left_arm_energy, right_arm_energy)
            
            # W_r(t) = v_wrist / (v_shoulder + epsilon)
            # Find which arm had higher energy to use for whip ratio
            if left_arm_energy > right_arm_energy:
                whip_ratio = norm_velocities[15] / (norm_velocities[11] + 0.01)
            else:
                whip_ratio = norm_velocities[16] / (norm_velocities[12] + 0.01)
            
            features['whip_ratio'] = whip_ratio
            features['arm_energy'] = arm_energy

            # Per-arm velocities (bbox-diagonal-normalized) and signed vertical
            # wrist velocity (negative = upward in image space).
            features['l_wrist_vel'] = norm_velocities[15]
            features['r_wrist_vel'] = norm_velocities[16]
            features['l_elbow_vel'] = norm_velocities[13]
            features['r_elbow_vel'] = norm_velocities[14]
            features['l_wrist_vy'] = diff[15][1] / bbox_diag
            features['r_wrist_vy'] = diff[16][1] / bbox_diag

            # True Action Score
            features['action_score'] = arm_energy * (hip_vel + 0.1)
            
            # Hip Velocity for Lateral/Vertical decomposition (Raw coords)
            hip_diff = hip_center - ((self.prev_landmarks[23] + self.prev_landmarks[24]) / 2.0)
            features['lat_v'] = abs(hip_diff[0])
            features['vert_v'] = abs(hip_diff[1])
            
        self.prev_landmarks = curr_coords
        return features

class SelfAdaptiveThreshold:
    """
    Maintains a rolling buffer to compute moving Median and MAD 
    for calculating dynamic thresholds.
    """
    def __init__(self, window_sec: float, fps: float, k: float, min_floor: float):
        self.capacity = int(window_sec * fps)
        self.buffer = collections.deque(maxlen=self.capacity)
        self.k = k
        self.min_floor = min_floor
        
    def update(self, value: float):
        self.buffer.append(value)
        
    def extend_padding(self, value: float, count: int):
        """Pad the buffer with zeros during non-motion gaps to preserve time."""
        for _ in range(count):
            self.buffer.append(value)
            
    def get_threshold(self) -> float:
        if len(self.buffer) < 2:
            return self.min_floor
            
        arr = np.array(self.buffer)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        
        dynamic_t = median + (self.k * mad)
        return max(dynamic_t, self.min_floor)


class RobustAdaptiveThreshold:
    """Adaptive threshold resistant to baseline drift during dense activity.

    Uses P25 (25th percentile) as the resting-baseline estimate instead
    of the median. During continuous rallies the median shifts upward
    because >50% of frames may be active, but P25 stays near the true
    resting level as long as at least 25% of frames are low-activity
    (inter-stroke lulls, footwork resets between shots).

    Spread is measured via MAD on the full buffer (same as the original
    SelfAdaptiveThreshold) to maintain familiar scaling.

    Additionally applies asymmetric exponential smoothing:
      - Slow rise (alpha=0.05): prevents threshold from climbing during
        short activity bursts.
      - Fast fall (alpha=0.3): recovers quickly once activity drops.

    threshold = P25 + k * MAD   (floored at min_floor, then EMA-smoothed)
    """
    def __init__(self, window_sec: float, fps: float, k: float, min_floor: float,
                 rise_alpha: float = 0.05, fall_alpha: float = 0.3):
        self.capacity = int(window_sec * fps)
        self.buffer = collections.deque(maxlen=self.capacity)
        self.k = k
        self.min_floor = min_floor
        self.rise_alpha = rise_alpha
        self.fall_alpha = fall_alpha
        self._smoothed = min_floor

    def update(self, value: float):
        self.buffer.append(value)

    def extend_padding(self, value: float, count: int):
        """Pad the buffer with zeros during non-motion gaps to preserve time."""
        for _ in range(count):
            self.buffer.append(value)

    def get_threshold(self) -> float:
        if len(self.buffer) < 10:
            return self.min_floor

        arr = np.array(self.buffer)
        # P25 as baseline: drifts only when >75% of buffer is elevated
        p25 = np.percentile(arr, 25)
        # MAD for spread: same scaling semantics as SelfAdaptiveThreshold
        mad = np.median(np.abs(arr - np.median(arr)))
        raw_t = p25 + (self.k * mad)
        raw_t = max(raw_t, self.min_floor)

        # Asymmetric smoothing: slow to rise, fast to fall
        if raw_t > self._smoothed:
            self._smoothed += self.rise_alpha * (raw_t - self._smoothed)
        else:
            self._smoothed += self.fall_alpha * (raw_t - self._smoothed)

        return self._smoothed


class RollingZScoreNormalizer:
    """Rolling Z-score normalization for source-agnostic signal comparison.
    
    Maintains a sliding window of raw values and outputs:
        z = (x - μ_window) / (σ_window + ε)
    
    This produces a unitless signal where "above average activity" is
    always z > 0, regardless of the absolute scale of the input, making
    thresholds transferable across diverse video sources.
    """
    def __init__(self, window_sec: float, fps: float, epsilon: float = 1e-6):
        self.capacity = max(int(window_sec * fps), 2)
        self.buffer = collections.deque(maxlen=self.capacity)
        self.epsilon = epsilon
    
    def update(self, value: float) -> float:
        """Push a new value and return the Z-scored version."""
        self.buffer.append(value)
        if len(self.buffer) < 2:
            return 0.0
        arr = np.array(self.buffer)
        mu = np.mean(arr)
        sigma = np.std(arr)
        return (value - mu) / (sigma + self.epsilon)
    
    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize an entire array using a causal rolling window.
        
        Each output[i] is the Z-score of arr[i] using only arr[max(0,i-capacity):i+1].
        """
        out = np.zeros_like(arr, dtype=float)
        buf = collections.deque(maxlen=self.capacity)
        for i, val in enumerate(arr):
            buf.append(val)
            if len(buf) < 2:
                out[i] = 0.0
            else:
                a = np.array(buf)
                mu = np.mean(a)
                sigma = np.std(a)
                out[i] = (val - mu) / (sigma + self.epsilon)
        return out

