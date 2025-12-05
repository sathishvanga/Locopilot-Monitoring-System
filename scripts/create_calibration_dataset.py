"""
Create calibration dataset for INT8 quantization

Extracts representative frames from sample videos to ensure accurate
quantization with minimal accuracy loss.

Usage:
    1. Place 3-5 representative videos in sample_videos/
    2. Run: python scripts/create_calibration_dataset.py
    3. Generates calibration_data/ with 100-200 frames
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List


def extract_calibration_frames(video_path: str, num_frames: int = 20) -> List[np.ndarray]:
    """
    Extract evenly spaced frames from a video for calibration.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract

    Returns:
        List of extracted frames (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames

    # Extract frames at evenly spaced indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    print(f"  Extracted {len(frames)} frames from {Path(video_path).name}")
    return frames


def save_calibration_dataset(
    sample_videos_dir: str = 'sample_videos',
    output_dir: str = 'calibration_data',
    frames_per_video: int = 20,
    max_videos: int = 5
):
    """
    Create calibration dataset from sample videos.

    Args:
        sample_videos_dir: Directory containing sample videos
        output_dir: Output directory for calibration images
        frames_per_video: Frames to extract per video
        max_videos: Maximum number of videos to process
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sample_videos_dir, exist_ok=True)

    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(Path(sample_videos_dir).glob(f'*{ext}'))

    if not video_paths:
        print(f"⚠️  No videos found in {sample_videos_dir}/")
        print(f"   Please add 3-5 representative videos to {sample_videos_dir}/")
        return False

    print(f"\nFound {len(video_paths)} video(s) in {sample_videos_dir}/")

    # Process videos
    all_frames = []
    for video_path in video_paths[:max_videos]:
        print(f"\nProcessing: {video_path.name}")
        frames = extract_calibration_frames(str(video_path), frames_per_video)
        all_frames.extend(frames)

    if not all_frames:
        print("❌ No frames extracted. Check video files.")
        return False

    # Save frames as JPEG
    print(f"\nSaving {len(all_frames)} calibration frames to {output_dir}/")
    for idx, frame in enumerate(all_frames):
        output_path = os.path.join(output_dir, f'calib_{idx:04d}.jpg')
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n✅ Successfully saved {len(all_frames)} calibration frames")
    print(f"   Location: {output_dir}/")
    print(f"   Next step: Run scripts/quantize_to_int8.py")
    return True


if __name__ == "__main__":
    success = save_calibration_dataset()
    if not success:
        print("\n📋 To use INT8 quantization:")
        print("   1. Add 3-5 representative videos to sample_videos/")
        print("   2. Run this script again")
        print("   3. Then run scripts/quantize_to_int8.py")
