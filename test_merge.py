import os
import glob
from extract_rallies import TennisRallyProcessor

def test_merge():
    # Setup
    video_path = "Testvd/IMG_0037.MOV"
    output_dir = "results/IMG_0037"
    
    # Initialize processor (dummy init, we just want utility methods)
    processor = TennisRallyProcessor(video_path, output_dir)
    
    # Get existing clips
    rallies_dir = os.path.join(output_dir, "rallies")
    clip_paths = sorted(glob.glob(os.path.join(rallies_dir, "rally_*.mp4")))
    
    if not clip_paths:
        print("No clips found to merge!")
        return

    print(f"Found {len(clip_paths)} clips.")
    
    # Test Merge
    merged_name = "test_merged_output.mp4"
    print(f"Merging into {merged_name}...")
    merged_path = processor.merge_clips(clip_paths, merged_name)
    
    if merged_path and os.path.exists(merged_path):
        print(f"Merge successful: {merged_path}")
        
        # Test Metadata Transfer
        print("Transferring metadata...")
        processor.transfer_metadata(video_path, merged_path)
        print("Metadata transfer complete.")
    else:
        print("Merge failed.")

if __name__ == "__main__":
    test_merge()
