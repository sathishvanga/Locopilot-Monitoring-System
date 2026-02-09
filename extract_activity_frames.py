#!/usr/bin/env python3
"""
Extract frames from a video for each detected activity into per-activity subfolders.

Usage:
    python extract_activity_frames.py <video_path> <activities_json> <output_dir> [--fps 1]

Creates:
    output_dir/
        writing_1/          (activity 1)
            frame_000000_0m00s.jpg
            frame_000025_0m01s.jpg
            ...
        cell_phone_2/       (activity 2)
            ...
        mind_diversion_3/   (activity 3)
            ...
"""

import json
import os
import sys

import cv2


def extract_frames(video_path, activities_json_path, output_dir, extract_fps=1.0):
    """Extract frames for each activity into subfolders."""

    # Load activities
    with open(activities_json_path, 'r') as f:
        activities = json.load(f)

    if not activities:
        print("No activities found in JSON.")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    print(f"Video: {video_path}")
    print(f"  FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration:.1f}s")
    print(f"  Extracting at {extract_fps} fps per activity")
    print(f"  Output: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # Process each activity
    for idx, activity in enumerate(activities):
        # Get activity info
        obj_type = activity.get('objectType', 'unknown').replace(' ', '_')
        start_sec = float(activity.get('activityStartTime', 0))
        end_sec = float(activity.get('activityEndTime', 0))
        performing_role = activity.get('performingRole', 'Unknown')

        # Create subfolder: activity_type_index
        folder_name = f"{obj_type}_{idx + 1}_{start_sec:.0f}s-{end_sec:.0f}s_{performing_role}"
        activity_dir = os.path.join(output_dir, folder_name)
        os.makedirs(activity_dir, exist_ok=True)

        # Calculate frame extraction interval
        frame_interval = video_fps / extract_fps  # frames between extractions

        # Extract frames
        current_sec = start_sec
        frame_count = 0

        while current_sec <= end_sec:
            frame_idx = int(current_sec * video_fps)
            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                current_sec += 1.0 / extract_fps
                continue

            # Format timestamp as minutes:seconds
            mins = int(current_sec) // 60
            secs = int(current_sec) % 60

            filename = f"frame_{frame_idx:08d}_{mins:02d}m{secs:02d}s.jpg"
            filepath = os.path.join(activity_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            frame_count += 1
            current_sec += 1.0 / extract_fps

        print(f"  [{idx + 1}/{len(activities)}] {obj_type} ({start_sec:.0f}s - {end_sec:.0f}s) "
              f"-> {frame_count} frames in {folder_name}/")

    cap.release()
    print(f"\nDone! Extracted frames for {len(activities)} activities into {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python extract_activity_frames.py <video_path> <activities_json> <output_dir> [--fps N]")
        sys.exit(1)

    video_path = sys.argv[1]
    activities_json = sys.argv[2]
    output_dir = sys.argv[3]

    fps = 1.0
    if '--fps' in sys.argv:
        fps_idx = sys.argv.index('--fps')
        if fps_idx + 1 < len(sys.argv):
            fps = float(sys.argv[fps_idx + 1])

    extract_frames(video_path, activities_json, output_dir, fps)
