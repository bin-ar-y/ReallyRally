import numpy as np
from extract_rallies import MovementTracker
import collections

def test_movement_tracker():
    print("Testing MovementTracker...")
    
    # 1. Stationary Case
    tracker = MovementTracker(window_duration_sec=3.0, fps=10)
    for _ in range(30):
        tracker.update(0.5, 0.5) # Dead center, no motion
        
    var = tracker.get_spatial_variance()
    print(f"Stationary Variance: {var:.4f}")
    assert var < 0.01, "Variance should be near 0 for stationary points"

    # 2. Moving Case (Linear motion)
    tracker = MovementTracker(window_duration_sec=3.0, fps=10)
    for i in range(30):
        # Move from 0.4 to 0.6
        pos = 0.4 + (0.2 * i/30)
        tracker.update(pos, 0.5)
        
    var = tracker.get_spatial_variance()
    print(f"Moving Variance: {var:.4f}")
    assert var > 0.03, "Variance should be > 0.03 for movement"
    
    print("MovementTracker Test Passed!")

if __name__ == "__main__":
    test_movement_tracker()
