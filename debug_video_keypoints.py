"""Debug script to test video keypoint extraction"""
import cv2
from ultralytics import YOLO
import os

video_path = r"D:\Yolov8 TD - Main\assets\Bài thể dục lớp 5 với hoa.mp4"
model_path = r"D:\Yolov8 TD - Main\models\yolov8s-pose.pt"

print(f"Testing video: {video_path}")
print(f"Video exists: {os.path.exists(video_path)}")

if not os.path.exists(video_path):
    print("ERROR: Video file not found!")
    exit(1)

# Check video properties
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Cannot open video file!")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0

print(f"\n=== Video Properties ===")
print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Resolution: {width}x{height}")
print(f"Duration: {duration:.2f} seconds")

# Test keypoint extraction
print(f"\n=== Testing Keypoint Extraction ===")
print(f"Loading model: {model_path}")
model = YOLO(model_path)

keypoints_list = []  # Store all extracted keypoints
frames_processed = 0

print(f"Processing ALL {frame_count} frames (this may take a minute)...")

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, verbose=False)
    frames_processed += 1
    
    if results[0].keypoints and results[0].keypoints.data.shape[0] > 0:
        # Store keypoint data (matching reference processor logic)
        keypoints_list.append(results[0].keypoints.data[0].cpu().numpy())
        if frames_processed <= 10:  # Show details for first 10 frames
            num_persons = results[0].keypoints.data.shape[0]
            print(f"  Frame {frames_processed-1}: ✓ Found {num_persons} person(s)")
    else:
        keypoints_list.append(None)  # Mark missing frame
        if frames_processed <= 10:
            print(f"  Frame {frames_processed-1}: ✗ No keypoints detected")

if frames_processed > 10:
    print(f"  ... (processed {frames_processed - 10} more frames)")

keypoints_found = len([kp for kp in keypoints_list if kp is not None])

cap.release()

print(f"\n=== Results ===")
print(f"Frames processed: {frames_processed}")
print(f"Frames with keypoints: {keypoints_found}")
print(f"Success rate: {keypoints_found/frames_processed*100:.1f}%")

if keypoints_found < 30:
    print(f"\n⚠️ WARNING: Only {keypoints_found} frames with keypoints detected!")
    print(f"The processor requires at least 30 frames to continue.")
    print(f"Possible issues:")
    print(f"  1. Video quality is too poor")
    print(f"  2. Person is not visible or too small in video")
    print(f"  3. Lighting conditions are bad")
    print(f"  4. Video codec issue")
else:
    print(f"\n✓ Video should process successfully ({keypoints_found} frames)")
