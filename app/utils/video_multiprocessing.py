"""
Multiprocessing utilities for video processing

This module implements a shared process pool architecture for parallel video processing
with worker initialization, work partitioning, and progress tracking.
"""

import os
import json
import cv2
import time
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, asdict
import multiprocessing as mp

from .multiprocessing_config import MultiprocessingConfig
from .logger import get_logger
from .config import get_settings


logger = get_logger(__name__)
settings = get_settings()


# Global variables for worker processes (initialized once per worker)
_worker_models = None
_worker_config = None


@dataclass
class FrameRange:
    """Represents a range of frames to process"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    expected_sampled_frames: int
    range_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ProcessingState:
    """Tracks processing progress and state"""
    total_expected_frames: int
    processed_frames: int
    completed_ranges: List[int]
    failed_ranges: List[int]
    start_time: float
    last_update_time: float
    done: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """Create from dictionary"""
        return cls(**data)
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_expected_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_expected_frames) * 100.0


def worker_initializer(config: MultiprocessingConfig):
    """
    Initialize worker process with models and configurations
    
    This function runs once per worker process at startup to:
    - Set thread counts for Torch and OpenCV
    - Disable OpenCV OpenCL
    - Preload heavy models (YOLO, MediaPipe)
    - Set environment variables
    
    Args:
        config: Multiprocessing configuration
    """
    global _worker_models, _worker_config
    
    try:
        # Set environment variables
        config.set_worker_env_vars()
        
        # Set PyTorch thread count
        torch.set_num_threads(config.torch_threads)
        
        # Disable OpenCV OpenCL
        if config.disable_opencv_opencl:
            cv2.ocl.setUseOpenCL(False)
        
        # Set OpenCV thread count
        cv2.setNumThreads(config.opencv_threads)
        
        logger.info(f"Worker {os.getpid()} initialized: torch_threads={config.torch_threads}, "
                   f"opencv_threads={config.opencv_threads}, opencl_disabled={config.disable_opencv_opencl}")
        
        # Preload models if enabled
        if config.preload_models:
            from ultralytics import YOLO
            import mediapipe as mp
            
            logger.info(f"Worker {os.getpid()} loading YOLO model: {config.yolo_model_path}")
            yolo_model = YOLO(config.yolo_model_path)
            
            logger.info(f"Worker {os.getpid()} initializing MediaPipe")
            mp_pose = mp.solutions.pose
            mp_face_mesh = mp.solutions.face_mesh
            
            pose = mp_pose.Pose(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            
            _worker_models = {
                'yolo': yolo_model,
                'pose': pose,
                'face_mesh': face_mesh,
                'mp_pose': mp_pose,
                'mp_face_mesh': mp_face_mesh
            }
            
            logger.info(f"Worker {os.getpid()} models loaded successfully")
        
        _worker_config = config
        
    except Exception as e:
        logger.error(f"Worker {os.getpid()} initialization failed: {e}", exc_info=True)
        raise


def calculate_frame_ranges(
    video_path: str,
    sample_fps: float,
    chunk_duration: float,
    min_chunk_duration: float = 2.0
) -> Tuple[List[FrameRange], int, float]:
    """
    Calculate frame ranges for parallel processing
    
    Splits video into fixed-duration chunks for balanced CPU and IO load.
    
    Args:
        video_path: Path to video file
        sample_fps: Sampling rate (frames per second)
        chunk_duration: Target duration per chunk in seconds
        min_chunk_duration: Minimum chunk duration in seconds
        
    Returns:
        Tuple of (frame_ranges, total_frames, native_fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / native_fps
    cap.release()
    
    # Calculate sampling step
    step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))
    
    # Calculate frame ranges based on chunk duration
    frame_ranges = []
    range_id = 0
    
    current_time = 0.0
    while current_time < video_duration:
        # Calculate chunk end time
        chunk_end_time = min(current_time + chunk_duration, video_duration)
        actual_chunk_duration = chunk_end_time - current_time
        
        # Skip chunks that are too short (except the last one)
        if actual_chunk_duration < min_chunk_duration and chunk_end_time < video_duration:
            current_time = chunk_end_time
            continue
        
        # Calculate frame indices
        start_frame = int(current_time * native_fps)
        end_frame = int(chunk_end_time * native_fps)
        
        # Calculate expected sampled frames in this range
        frames_in_range = end_frame - start_frame
        expected_samples = max(1, frames_in_range // step)
        
        frame_range = FrameRange(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=current_time,
            end_time=chunk_end_time,
            expected_sampled_frames=expected_samples,
            range_id=range_id
        )
        
        frame_ranges.append(frame_range)
        range_id += 1
        current_time = chunk_end_time
    
    logger.info(f"Video split into {len(frame_ranges)} chunks "
               f"(~{chunk_duration}s each, {sample_fps} FPS sampling)")
    
    return frame_ranges, total_frames, native_fps


def process_frame_range(
    video_path: str,
    frame_range: FrameRange,
    config_dict: Dict[str, Any],
    sample_fps: float,
    trip_id: str,
    crew_name: str,
    crew_id: str,
    crew_role: int,
    output_dir: str,
    run_dir: str = None,
    save_clips: bool = True
) -> Dict[str, Any]:
    """
    Process a specific frame range (worker task function)
    
    This function runs in a worker process and processes frames within
    the assigned range independently. Clips and images can be saved if requested.
    
    Args:
        video_path: Path to video file
        frame_range: Frame range to process
        config_dict: Pipeline configuration dictionary
        sample_fps: Sampling rate
        trip_id: Trip identifier
        crew_name: Crew member name
        crew_id: Crew member ID
        crew_role: Crew role
        output_dir: Output directory (base directory)
        run_dir: Run directory for saving clips (if None, no clips saved)
        save_clips: Whether to save clips and images (default: True)
        
    Returns:
        Dictionary with detected activities and metadata
    """
    try:
        from locopilot_monitor import LocopilotActivityMonitor
        
        worker_id = os.getpid()
        logger.info(f"Worker {worker_id} processing range {frame_range.range_id}: "
                   f"frames {frame_range.start_frame}-{frame_range.end_frame} "
                   f"({frame_range.start_time:.2f}s - {frame_range.end_time:.2f}s) "
                   f"[save_clips={save_clips}]")
        
        # Determine whether annotated frames should be persisted
        save_frames = save_clips and settings.save_annotated_frames

        # Create a monitor instance with or without run directory
        if run_dir and save_clips:
            # Use provided run directory for saving clips
            monitor = LocopilotActivityMonitor(
                video_path=video_path,
                output_dir=output_dir,
                save_annotated_frames=save_frames,
                frame_save_interval=settings.frame_save_interval,
                sample_fps=sample_fps,
                run_dir=run_dir,  # Use shared run directory
                create_run_dir=False  # Don't create new directory
            )
        else:
            # No run directory - activities in memory only
            monitor = LocopilotActivityMonitor(
                video_path=video_path,
                output_dir=output_dir,
                save_annotated_frames=False,
                frame_save_interval=settings.frame_save_interval,
                sample_fps=sample_fps,
                run_dir=None,
                create_run_dir=False
            )
        
        # Set trip and crew information
        monitor.trip_id = trip_id
        monitor.crew_name = crew_name
        monitor.crew_id = crew_id
        monitor.crew_role = crew_role
        
        # Process the assigned frame range with optional clip saving
        activities = monitor.process_video_range(
            start_frame=frame_range.start_frame,
            end_frame=frame_range.end_frame,
            save_clips=save_clips
        )
        
        logger.info(f"Worker {worker_id} completed range {frame_range.range_id}: "
                   f"{len(activities)} activities detected")
        
        return {
            'success': True,
            'range_id': frame_range.range_id,
            'activities': activities,
            'processed_frames': frame_range.expected_sampled_frames,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed processing range {frame_range.range_id}: {e}",
                    exc_info=True)
        return {
            'success': False,
            'range_id': frame_range.range_id,
            'activities': [],
            'processed_frames': 0,
            'error': str(e)
        }


class VideoMultiprocessingOrchestrator:
    """
    Orchestrates parallel video processing with shared process pool
    
    This class manages:
    - Shared process pool lifecycle
    - Task submission and result collection
    - Progress tracking and persistence
    - Result aggregation
    """
    
    def __init__(
        self,
        config: Optional[MultiprocessingConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize orchestrator
        
        Args:
            config: Multiprocessing configuration (uses default if None)
            output_dir: Output directory for results and state
        """
        self.config = config or MultiprocessingConfig()
        self.output_dir = output_dir or "locopilot_evidence"
        self.pool: Optional[ProcessPoolExecutor] = None
        self.state: Optional[ProcessingState] = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def initialize_pool(self):
        """Initialize shared process pool"""
        if self.pool is not None:
            logger.warning("Process pool already initialized")
            return
        
        num_workers = self.config.get_num_workers()
        
        logger.info(f"Initializing process pool with {num_workers} workers "
                   f"(method={self.config.start_method})")
        
        # Set multiprocessing start method
        try:
            mp.set_start_method(self.config.start_method, force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        # Create process pool with initializer
        self.pool = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=worker_initializer,
            initargs=(self.config,)
        )
        
        logger.info("Process pool initialized successfully")
    
    def shutdown_pool(self, wait: bool = True):
        """Shutdown process pool"""
        if self.pool is not None:
            logger.info("Shutting down process pool")
            self.pool.shutdown(wait=wait)
            self.pool = None
    
    def save_state(self, run_dir: str):
        """Save processing state to disk"""
        if not self.config.enable_result_persistence or self.state is None:
            return
        
        state_path = os.path.join(run_dir, self.config.state_file_name)
        try:
            with open(state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"State saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self, run_dir: str) -> Optional[ProcessingState]:
        """Load processing state from disk"""
        if not self.config.enable_result_persistence:
            return None
        
        state_path = os.path.join(run_dir, self.config.state_file_name)
        if not os.path.exists(state_path):
            return None
        
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
            state = ProcessingState.from_dict(data)
            logger.info(f"State loaded from {state_path}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def process_video_parallel(
        self,
        video_path: str,
        trip_id: str,
        crew_name: str,
        crew_id: str,
        crew_role: int,
        sample_fps: float,
        run_dir: str,
        save_clips: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process video in parallel using multiple workers
        
        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            sample_fps: Sampling rate
            run_dir: Run directory for output
            save_clips: Whether to save video clips and images (default: True)
            
        Returns:
            List of detected activities (merged from all ranges)
        """
        start_time = time.time()
        
        # Initialize pool if needed
        if self.pool is None:
            self.initialize_pool()
        
        # Calculate frame ranges
        logger.info("Calculating frame ranges for parallel processing")
        frame_ranges, total_frames, native_fps = calculate_frame_ranges(
            video_path=video_path,
            sample_fps=sample_fps,
            chunk_duration=self.config.chunk_duration_seconds,
            min_chunk_duration=self.config.min_chunk_duration_seconds
        )
        
        # Calculate total expected sampled frames
        total_expected_frames = sum(fr.expected_sampled_frames for fr in frame_ranges)
        
        # Initialize processing state
        self.state = ProcessingState(
            total_expected_frames=total_expected_frames,
            processed_frames=0,
            completed_ranges=[],
            failed_ranges=[],
            start_time=start_time,
            last_update_time=start_time,
            done=False
        )
        
        logger.info(f"Submitting {len(frame_ranges)} tasks to process pool "
                   f"(expected {total_expected_frames} sampled frames, save_clips={save_clips})")
        
        # Submit tasks to pool
        futures: Dict[Future, FrameRange] = {}
        config_dict = {}  # Pipeline configuration
        
        for frame_range in frame_ranges:
            future = self.pool.submit(
                process_frame_range,
                video_path=video_path,
                frame_range=frame_range,
                config_dict=config_dict,
                sample_fps=sample_fps,
                trip_id=trip_id,
                crew_name=crew_name,
                crew_id=crew_id,
                crew_role=crew_role,
                output_dir=self.output_dir,
                run_dir=run_dir,  # Pass run_dir for clip saving
                save_clips=save_clips
            )
            futures[future] = frame_range
        
        # Collect results as they complete
        all_activities = []
        
        for future in as_completed(futures):
            frame_range = futures[future]
            
            try:
                result = future.result()
                
                if result['success']:
                    # Append activities from this range
                    all_activities.extend(result['activities'])
                    
                    # Update state
                    self.state.processed_frames += result['processed_frames']
                    self.state.completed_ranges.append(result['range_id'])
                    
                    logger.info(f"Range {result['range_id']} completed: "
                               f"{len(result['activities'])} activities, "
                               f"{self.state.get_progress_percentage():.1f}% done")
                else:
                    # Record failure
                    self.state.failed_ranges.append(result['range_id'])
                    logger.error(f"Range {result['range_id']} failed: {result['error']}")
                
                # Update state timestamp and save
                self.state.last_update_time = time.time()
                if self.config.enable_result_persistence:
                    self.save_state(run_dir)
                
            except Exception as e:
                logger.error(f"Failed to get result for range {frame_range.range_id}: {e}",
                           exc_info=True)
                self.state.failed_ranges.append(frame_range.range_id)
        
        # Mark as done
        self.state.done = True
        self.state.last_update_time = time.time()
        
        if self.config.enable_result_persistence:
            self.save_state(run_dir)
        
        # Sort activities by start time
        all_activities.sort(key=lambda x: float(x.get('activityStartTime', 0)))
        
        processing_time = time.time() - start_time
        
        logger.info(f"Parallel processing completed in {processing_time:.2f}s: "
                   f"{len(all_activities)} activities detected, "
                   f"{len(self.state.completed_ranges)} ranges completed, "
                   f"{len(self.state.failed_ranges)} ranges failed")
        
        # Ensure clips directory exists
        clips_dir = os.path.join(run_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        
        # Save merged activities to main run directory
        if len(all_activities) > 0:
            activities_json_path = os.path.join(run_dir, "activities.json")
            try:
                import json
                with open(activities_json_path, 'w') as f:
                    json.dump(all_activities, f, indent=2)
                logger.info(f"Saved {len(all_activities)} activities to {activities_json_path}")
            except Exception as e:
                logger.error(f"Failed to save activities.json: {e}")
        
        # Log clip generation status
        if save_clips:
            # Count generated clips
            clip_count = sum(1 for a in all_activities if a.get('activityClip') and 
                           os.path.exists(a.get('activityClip', '')))
            logger.info(f"Generated {clip_count} video clips and images in {clips_dir}")
        else:
            logger.info("Note: Clip generation was disabled (save_clips=False)")
        
        return all_activities

