
import os
import sys
import time
from datetime import datetime

# Import the module
import extract_rallies

def test_naming():
    # Create a dummy video file
    video_path = "dummy_video.mp4"
    with open(video_path, "w") as f:
        f.write("dummy content")
    
    # Set mtime to known value if current time is variable?
    # Actually let's just use current time and verify fuzzy match
    
    try:
        # Get expected date
        # Fallback uses mtime
        timestamp = os.path.getmtime(video_path)
        dt = datetime.fromtimestamp(timestamp)
        expected_date = dt.strftime("%m%d%y")
        
        # Call function
        name = extract_rallies.generate_rally_name(video_path)
        print(f"Generated Name: {name}")
        
        expected_name = f"{expected_date}_dummy_video_rally"
        
        if name == expected_name:
            print("SUCCESS: Naming matches expected format.")
        else:
            print(f"FAILURE: Expected {expected_name}, got {name}")
            
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    test_naming()
