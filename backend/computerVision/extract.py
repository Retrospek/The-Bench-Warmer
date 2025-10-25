import cv2
import os

def extract_frames(video_path, output_dir="frames", frame_interval=1, max_frames=None):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to extract (None = all frames)
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n=== Video Info ===")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Frame interval: Every {frame_interval} frame(s)")
    print(f"Estimated output frames: {total_frames // frame_interval}")
    print(f"\nExtracting frames...\n")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:  # No more frames
            break
        
        # Check if we should save this frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            # Progress update every 10 frames
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames... ({frame_filename})")
            
            saved_count += 1
            
            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                print(f"\nReached maximum frame limit: {max_frames}")
                break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Extraction Complete ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Configuration
    video_path = r"backend\computerVision\tidal_demo.mp4"
    output_dir = "frames"
    
    # Extract all frames
    extract_frames(video_path, output_dir, frame_interval=1)
    
    # OR extract every 5th frame (faster, less storage)
    # extract_frames(video_path, output_dir, frame_interval=5)
    
    # OR extract first 100 frames only
    # extract_frames(video_path, output_dir, frame_interval=1, max_frames=100)