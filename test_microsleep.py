"""Local test: run microsleep clip through the monitor with debug frames."""

from locopilot_monitor import LocopilotActivityMonitor

video_path = "/Users/satishvanga/Documents/all_activities_25m_30m.mp4"

monitor = LocopilotActivityMonitor(
    video_path,
    output_dir="locopilot_evidence",
    save_annotated_frames=True,
    frame_save_interval=1,   # Save every sampled frame
    sample_fps=0.5           # 1 frame every 2 seconds
)

monitor.process_video()
