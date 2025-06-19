import cv2
import os
import sys
from tqdm import tqdm

def check_video(video_path):
    """
    Check if a video file is valid and not corrupted
    Returns: (is_valid, error_message, frame_count, fps, resolution)
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Could not open video file", 0, 0, (0, 0)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count <= 0:
            return False, "Video has no frames", 0, fps, (width, height)
        
        # Try to read all frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                return False, f"Invalid frame at index {frame_idx}", frame_idx, fps, (width, height)
            
            # Check frame dimensions
            if frame.shape[0] != height or frame.shape[1] != width:
                return False, f"Frame {frame_idx} has incorrect dimensions", frame_idx, fps, (width, height)
        
        # Check if we read all frames
        if frame_idx != frame_count:
            return False, f"Frame count mismatch: expected {frame_count}, got {frame_idx}", frame_idx, fps, (width, height)
        
        cap.release()
        return True, "Video is valid", frame_count, fps, (width, height)
        
    except Exception as e:
        return False, f"Error: {str(e)}", 0, 0, (0, 0)

if __name__ == "__main__":
    video_path = '/home/csgrad/susimmuk/acmdeepfake/data/extracted_frames/silent_videos/random5zmu2z1dqe/real/0_4/0_4_magnified.mp4'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        sys.exit(1)
    
    print(f"Checking video: {video_path}")
    print("-" * 80)
    
    is_valid, message, frame_count, fps, (width, height) = check_video(video_path)
    
    if is_valid:
        print(f"✓ Valid video:")
        print(f"  - Frames: {frame_count}")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - Resolution: {width}x{height}")
    else:
        print(f"✗ Corrupted video: {message}") 