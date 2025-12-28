"""
Voting Verification Service for Activity Detection

This service implements a two-stage detection system:
- Stage 1: Quick detection on sampled frames (existing logic)
- Stage 2: When detection triggers, verify with multiple native frames using voting

The voting mechanism significantly reduces false positives by requiring
a configurable percentage of frames to confirm the detection.
"""

import cv2
import logging
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from logging.handlers import TimedRotatingFileHandler

from ..utils.config import get_settings


def _setup_voting_logger():
    """Setup a dedicated logger for voting verification with file output."""
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("VotingVerification")
    logger.setLevel(logging.DEBUG)  # Enable debug level for detailed logs

    if not logger.handlers:
        # File handler for all voting logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "voting_verification.log")
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Also add to main log file
        main_log_handler = logging.FileHandler(
            os.path.join(log_dir, "LocopilotMonitoring.log")
        )
        main_log_handler.setLevel(logging.INFO)
        main_log_handler.setFormatter(formatter)
        logger.addHandler(main_log_handler)

    return logger


logger = _setup_voting_logger()


class VotingVerificationService:
    """
    Service for multi-frame voting verification of activity detections.

    When an activity is detected in Stage 1 (sampled frame), this service
    extracts consecutive native frames and runs detection on each frame.
    The final decision is based on voting - activity is confirmed only if
    a configurable percentage of frames detect the activity.
    """

    def __init__(self, yolo_model=None, yolo_pose_model=None):
        """
        Initialize the voting verification service.

        Args:
            yolo_model: Pre-loaded YOLO model for object detection
            yolo_pose_model: Pre-loaded YOLO-Pose model for pose estimation
        """
        self.settings = get_settings()
        self.yolo_model = yolo_model
        self.yolo_pose_model = yolo_pose_model

        # Activity thresholds (matching locopilot_monitor.py)
        self.cell_phone_margin = 100  # Hand-to-phone proximity
        self.book_hand_margin = 180   # Hand-to-book proximity
        self.person_book_margin = 250 # Person-to-book region
        self.backpack_wrist_margin = 40  # Wrist inside backpack
        self.yaw_threshold = 45.0     # Mind diversion yaw
        self.pitch_threshold = 15.0   # Mind diversion pitch

        logger.info("="*60)
        logger.info("VotingVerificationService initialized")
        logger.info(f"  voting_enabled: {self.settings.voting_enabled}")
        logger.info(f"  voting_num_frames: {self.settings.voting_num_frames}")
        logger.info(f"  voting_frame_spread_ms: {self.settings.voting_frame_spread_ms}")
        logger.info(f"  Thresholds:")
        logger.info(f"    cell_phone: {self.settings.voting_threshold_cell_phone}")
        logger.info(f"    writing: {self.settings.voting_threshold_writing}")
        logger.info(f"    packing_bags: {self.settings.voting_threshold_packing_bags}")
        logger.info(f"    lp_hand_gesture: {self.settings.voting_threshold_lp_hand_gesture}")
        logger.info(f"    alp_hand_gesture: {self.settings.voting_threshold_alp_hand_gesture}")
        logger.info(f"    mind_diversion: {self.settings.voting_threshold_mind_diversion}")
        logger.info(f"    group_detected: {self.settings.voting_threshold_group_detected}")
        logger.info(f"  Debug Settings:")
        logger.info(f"    save_debug_frames: {self.settings.voting_save_debug_frames}")
        logger.info(f"    debug_frames_dir: {self.settings.voting_debug_frames_dir}")
        logger.info("="*60)

        # Initialize debug frames directory
        self.debug_frames_base_dir = self.settings.voting_debug_frames_dir

    def verify_activity(
        self,
        video_path: str,
        timestamp_sec: float,
        activity_type: str,
        person_bbox: List[float],
        trigger_frame: np.ndarray = None,
        trigger_detections: Dict = None,
        trigger_poses: Any = None,
        frame_shape: Tuple[int, int, int] = None
    ) -> Tuple[bool, Dict]:
        """
        Verify an activity detection using multi-frame voting.

        Args:
            video_path: Path to the source video file
            timestamp_sec: Timestamp where Stage 1 triggered (in seconds)
            activity_type: Type of activity to verify (e.g., 'cell_phone', 'writing')
            person_bbox: Bounding box of the person [x1, y1, x2, y2]
            trigger_frame: The frame that triggered Stage 1 (for reference)
            trigger_detections: YOLO detections from trigger frame
            trigger_poses: Pose landmarks from trigger frame
            frame_shape: Shape of frame (height, width, channels)

        Returns:
            Tuple of (is_confirmed, vote_details)
        """
        logger.info("")
        logger.info("="*70)
        logger.info(f"[VOTING] STARTING VERIFICATION for {activity_type.upper()}")
        logger.info(f"[VOTING] Video: {os.path.basename(video_path)}")
        logger.info(f"[VOTING] Timestamp: {timestamp_sec:.2f}s")
        logger.info(f"[VOTING] Person bbox: {person_bbox}")
        logger.info("="*70)

        # Check if voting is enabled
        if not self.settings.voting_enabled:
            logger.info(f"[VOTING] Voting DISABLED - bypassing verification")
            return True, {'method': 'bypass', 'reason': 'voting_disabled'}

        # Get threshold for this activity type
        threshold_attr = f'voting_threshold_{activity_type}'
        threshold = getattr(self.settings, threshold_attr, 0.5)
        num_frames = self.settings.voting_num_frames

        logger.debug(f"[VOTING] {activity_type}: threshold={threshold}, num_frames={num_frames}")

        # Extract native frames
        frames = self._extract_native_frames(video_path, timestamp_sec, num_frames, activity_type)

        if len(frames) < num_frames // 2:
            logger.warning(f"[VOTING] {activity_type}: Only extracted {len(frames)}/{num_frames} frames - BYPASSING")
            return True, {
                'method': 'bypass',
                'reason': 'insufficient_frames',
                'frames_extracted': len(frames),
                'frames_required': num_frames
            }

        # Run batch detection on all frames
        logger.info(f"[VOTING] {activity_type}: Running batch detection on {len(frames)} frames...")
        batch_detections = self._batch_detect_objects(frames, activity_type)
        batch_poses = self._batch_detect_poses(frames, activity_type)

        # Vote on each frame
        votes = []
        confidences = []
        frame_details = []

        logger.info(f"[VOTING] {activity_type}: Starting per-frame verification...")
        logger.info("-"*60)

        for i, (frame, detections, poses) in enumerate(zip(frames, batch_detections, batch_poses)):
            detected, confidence, details = self._verify_frame(
                frame, detections, poses, activity_type, person_bbox, i, len(frames)
            )
            votes.append(detected)
            confidences.append(confidence)
            frame_details.append(details)

        # Calculate voting result
        positive_votes = sum(votes)
        vote_ratio = positive_votes / len(votes) if votes else 0
        is_confirmed = vote_ratio >= threshold

        # Log final summary
        vote_breakdown = ['Y' if v else 'N' for v in votes]

        logger.info("-"*60)
        logger.info(f"[VOTING] ========== {activity_type.upper()} VOTING SUMMARY ==========")
        logger.info(f"[VOTING] Timestamp: {timestamp_sec:.2f}s")
        logger.info(f"[VOTING] Person bbox: {person_bbox}")
        logger.info(f"[VOTING] Frames analyzed: {len(votes)}")
        logger.info(f"[VOTING] Positive votes: {positive_votes}")
        logger.info(f"[VOTING] Vote breakdown: {vote_breakdown}")
        logger.info(f"[VOTING] Vote ratio: {vote_ratio*100:.1f}%")
        logger.info(f"[VOTING] Threshold: {threshold*100:.0f}%")
        logger.info(f"[VOTING] DECISION: {'CONFIRMED' if is_confirmed else 'REJECTED'}")
        logger.info(f"[VOTING] ================================================")
        logger.info("")

        # Save debug frames if enabled
        if self.settings.voting_save_debug_frames:
            self._save_debug_frames(
                frames=frames,
                batch_detections=batch_detections,
                batch_poses=batch_poses,
                votes=votes,
                activity_type=activity_type,
                timestamp_sec=timestamp_sec,
                person_bbox=person_bbox,
                is_confirmed=is_confirmed,
                vote_ratio=vote_ratio,
                threshold=threshold,
                video_path=video_path
            )

        return is_confirmed, {
            'method': 'voting',
            'activity_type': activity_type,
            'timestamp_sec': timestamp_sec,
            'frames_analyzed': len(votes),
            'positive_votes': positive_votes,
            'vote_breakdown': vote_breakdown,
            'vote_ratio': vote_ratio,
            'threshold': threshold,
            'is_confirmed': is_confirmed,
            'frame_details': frame_details,
            'confidences': confidences
        }

    def _extract_native_frames(
        self,
        video_path: str,
        timestamp_sec: float,
        num_frames: int,
        activity_type: str
    ) -> List[np.ndarray]:
        """
        Extract consecutive native frames centered around a timestamp.

        Args:
            video_path: Path to video file
            timestamp_sec: Center timestamp in seconds
            num_frames: Number of frames to extract
            activity_type: Activity type for logging

        Returns:
            List of BGR frames
        """
        frames = []

        logger.debug(f"[VOTING] {activity_type} @ {timestamp_sec:.2f}s: Extracting {num_frames} native frames")

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"[VOTING] {activity_type}: Failed to open video: {video_path}")
                return frames

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame window centered around timestamp
            center_frame = int(timestamp_sec * fps)
            start_frame = max(0, center_frame - num_frames // 2)

            # Ensure we don't go past the end
            if start_frame + num_frames > total_frames:
                start_frame = max(0, total_frames - num_frames)

            logger.debug(f"[VOTING] {activity_type}: Video FPS={fps:.1f}, total_frames={total_frames}")
            logger.debug(f"[VOTING] {activity_type}: center_frame={center_frame}, start_frame={start_frame}")

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Extract frames
            for i in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    logger.warning(f"[VOTING] {activity_type}: Failed to read frame {start_frame + i}")
                    break

            cap.release()

            logger.debug(f"[VOTING] {activity_type}: Successfully extracted {len(frames)}/{num_frames} frames")

        except Exception as e:
            logger.error(f"[VOTING] {activity_type}: Error extracting frames: {e}")

        return frames

    def _batch_detect_objects(self, frames: List[np.ndarray], activity_type: str) -> List[Dict]:
        """
        Run YOLO object detection on multiple frames in batch.

        Args:
            frames: List of BGR frames
            activity_type: Activity type for logging

        Returns:
            List of detection dictionaries, one per frame
        """
        if not frames or self.yolo_model is None:
            logger.warning(f"[VOTING] {activity_type}: No frames or YOLO model not available")
            return [{'person': [], 'cell_phone': [], 'book': [], 'backpack': [], 'bottle': []} for _ in frames]

        all_detections = []

        try:
            # YOLO supports batch inference
            logger.debug(f"[VOTING] {activity_type}: Running batch YOLO inference on {len(frames)} frames...")
            results = self.yolo_model(frames, verbose=False, imgsz=self.settings.yolo_imgsz)

            for i, r in enumerate(results):
                detections = {
                    'person': [],
                    'cell_phone': [],
                    'book': [],
                    'backpack': [],
                    'bottle': []
                }

                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        class_name = self.yolo_model.names[cls]

                        if class_name == 'person' and conf > 0.5:
                            detections['person'].append(xyxy)
                        elif class_name == 'cell phone' and conf > 0.3:
                            detections['cell_phone'].append({'bbox': xyxy, 'confidence': conf})
                        elif class_name == 'book' and conf > 0.4:
                            detections['book'].append({'bbox': xyxy, 'confidence': conf})
                        elif class_name in ['backpack', 'handbag', 'suitcase'] and conf > 0.75:
                            detections['backpack'].append({'bbox': xyxy, 'confidence': conf})
                        elif class_name == 'bottle' and conf > 0.4:
                            detections['bottle'].append({'bbox': xyxy, 'confidence': conf})

                all_detections.append(detections)

                logger.debug(f"[VOTING] {activity_type}: Frame {i+1} - "
                           f"persons={len(detections['person'])}, "
                           f"phones={len(detections['cell_phone'])}, "
                           f"books={len(detections['book'])}, "
                           f"bags={len(detections['backpack'])}, "
                           f"bottles={len(detections['bottle'])}")

        except Exception as e:
            logger.error(f"[VOTING] {activity_type}: Batch detection error: {e}")
            all_detections = [{'person': [], 'cell_phone': [], 'book': [], 'backpack': [], 'bottle': []} for _ in frames]

        return all_detections

    def _batch_detect_poses(self, frames: List[np.ndarray], activity_type: str) -> List[Dict]:
        """
        Run YOLO-Pose detection on multiple frames in batch.

        Args:
            frames: List of BGR frames
            activity_type: Activity type for logging

        Returns:
            List of pose dictionaries, one per frame
        """
        if not frames or self.yolo_pose_model is None:
            logger.warning(f"[VOTING] {activity_type}: No frames or YOLO-Pose model not available")
            return [{'keypoints': [], 'boxes': []} for _ in frames]

        all_poses = []

        try:
            logger.debug(f"[VOTING] {activity_type}: Running batch YOLO-Pose inference on {len(frames)} frames...")

            # Handle YoloPoseAdapter vs raw YOLO model
            # YoloPoseAdapter has .model attribute with the raw YOLO model
            if hasattr(self.yolo_pose_model, 'model'):
                # It's a YoloPoseAdapter, use the underlying model for batch inference
                pose_model = self.yolo_pose_model.model
            else:
                # It's a raw YOLO model
                pose_model = self.yolo_pose_model

            results = pose_model(
                frames,
                verbose=False,
                imgsz=self.settings.yolo_imgsz,
                conf=self.settings.yolo_pose_confidence
            )

            for i, r in enumerate(results):
                poses = {'keypoints': [], 'boxes': []}

                if r.keypoints is not None and len(r.keypoints) > 0:
                    for j, kp in enumerate(r.keypoints):
                        keypoints_data = kp.data.cpu().numpy()
                        if len(keypoints_data) > 0:
                            poses['keypoints'].append(keypoints_data[0])  # [17, 3] array
                            if r.boxes is not None and j < len(r.boxes):
                                poses['boxes'].append(r.boxes[j].xyxy[0].cpu().numpy().tolist())

                all_poses.append(poses)
                logger.debug(f"[VOTING] {activity_type}: Frame {i+1} - poses_detected={len(poses['keypoints'])}")

        except Exception as e:
            logger.error(f"[VOTING] {activity_type}: Batch pose detection error: {e}")
            all_poses = [{'keypoints': [], 'boxes': []} for _ in frames]

        return all_poses

    def _verify_frame(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        activity_type: str,
        person_bbox: List[float],
        frame_idx: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """
        Verify activity in a single frame.

        Args:
            frame: BGR frame
            detections: YOLO detections for this frame
            poses: Pose data for this frame
            activity_type: Type of activity to verify
            person_bbox: Original person bounding box
            frame_idx: Index of this frame
            total_frames: Total number of frames

        Returns:
            Tuple of (detected, confidence, details)
        """
        frame_num = frame_idx + 1

        if activity_type == 'cell_phone':
            return self._verify_cell_phone(frame, detections, poses, person_bbox, frame_num, total_frames)
        elif activity_type == 'writing':
            return self._verify_writing(frame, detections, poses, person_bbox, frame_num, total_frames)
        elif activity_type == 'packing_bags':
            return self._verify_packing(frame, detections, poses, person_bbox, frame_num, total_frames)
        elif activity_type in ['lp_hand_gesture', 'alp_hand_gesture']:
            return self._verify_hand_gesture(frame, detections, poses, person_bbox, activity_type, frame_num, total_frames)
        elif activity_type == 'mind_diversion':
            return self._verify_mind_diversion(frame, detections, poses, person_bbox, frame_num, total_frames)
        elif activity_type == 'group_detected':
            return self._verify_group(frame, detections, frame_num, total_frames)
        else:
            logger.warning(f"[VOTING] Unknown activity type: {activity_type}")
            return True, 1.0, {'reason': 'unknown_activity'}

    def _verify_cell_phone(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        person_bbox: List[float],
        frame_num: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """Verify cell phone detection in a single frame."""
        h, w = frame.shape[:2]
        phones = detections.get('cell_phone', [])

        logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}/{total_frames}: "
                    f"phones_detected={len(phones)}, persons={len(detections.get('person', []))}")

        if not phones:
            logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}: No phones detected - VOTE=NO")
            return False, 0.0, {'reason': 'no_phone_detected'}

        # Get hand coordinates from poses
        left_hand_coords = None
        right_hand_coords = None

        if poses.get('keypoints') and len(poses['keypoints']) > 0:
            # Find best matching pose for this person
            best_pose = self._find_matching_pose(poses, person_bbox)
            if best_pose is not None:
                # YOLO keypoints: 9=left_wrist, 10=right_wrist
                if best_pose[9][2] > 0.3:  # visibility check
                    left_hand_coords = (int(best_pose[9][0]), int(best_pose[9][1]))
                if best_pose[10][2] > 0.3:
                    right_hand_coords = (int(best_pose[10][0]), int(best_pose[10][1]))

        logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}: "
                    f"left_hand={left_hand_coords}, right_hand={right_hand_coords}")

        # Check each phone
        for phone_data in phones:
            phone_bbox = phone_data['bbox']
            conf = phone_data['confidence']

            logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}: "
                        f"phone_bbox={[int(x) for x in phone_bbox]}, confidence={conf:.3f}")

            # Check if phone is in person's region
            phone_in_region = self._bbox_overlap_with_margin(phone_bbox, person_bbox, self.cell_phone_margin)

            if phone_in_region:
                # Check if hand is near phone
                left_near = self._check_hand_near_object(left_hand_coords, phone_bbox, self.cell_phone_margin) if left_hand_coords else False
                right_near = self._check_hand_near_object(right_hand_coords, phone_bbox, self.cell_phone_margin) if right_hand_coords else False
                hand_near = left_near or right_near

                logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}: "
                            f"phone_in_region={phone_in_region}, hand_near_phone={hand_near}")

                if hand_near:
                    logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}: VOTE=YES")
                    return True, conf, {
                        'phone_bbox': phone_bbox,
                        'confidence': conf,
                        'hand_near': True
                    }

        logger.debug(f"[VOTING:CELL_PHONE] Frame {frame_num}: VOTE=NO (no hand near phone)")
        return False, 0.0, {'reason': 'no_hand_near_phone'}

    def _verify_writing(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        person_bbox: List[float],
        frame_num: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """Verify writing detection in a single frame."""
        h, w = frame.shape[:2]
        books = detections.get('book', [])

        detected_by_book = False
        detected_by_wrist = False

        logger.debug(f"[VOTING:WRITING] Frame {frame_num}/{total_frames}: books_detected={len(books)}")

        # Get hand coordinates
        left_hand_coords = None
        right_hand_coords = None
        head_down = False
        wrist_dist = float('inf')

        if poses.get('keypoints') and len(poses['keypoints']) > 0:
            best_pose = self._find_matching_pose(poses, person_bbox)
            if best_pose is not None:
                try:
                    # Get wrist positions (YOLO keypoints: 9=left_wrist, 10=right_wrist)
                    left_wrist = best_pose[9]
                    right_wrist = best_pose[10]

                    logger.debug(f"[VOTING:WRITING] Frame {frame_num}: "
                                f"raw left_wrist={left_wrist}, right_wrist={right_wrist}")

                    # Extract coordinates with proper checks
                    if len(left_wrist) >= 3 and left_wrist[2] > 0.3:
                        left_hand_coords = (int(left_wrist[0]), int(left_wrist[1]))
                    if len(right_wrist) >= 3 and right_wrist[2] > 0.3:
                        right_hand_coords = (int(right_wrist[0]), int(right_wrist[1]))

                    # Check head position (nose vs shoulders)
                    nose = best_pose[0]
                    left_shoulder = best_pose[5]
                    right_shoulder = best_pose[6]

                    if len(nose) >= 3 and len(left_shoulder) >= 3 and len(right_shoulder) >= 3:
                        if nose[2] > 0.3 and left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                            head_down = nose[1] > shoulder_y - 30  # Head level or below shoulders

                    # Calculate wrist proximity for writing heuristic
                    if left_hand_coords and right_hand_coords:
                        wrist_dist = abs(left_hand_coords[0] - right_hand_coords[0])

                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"[VOTING:WRITING] Frame {frame_num}: Error extracting keypoints: {e}")
            else:
                logger.debug(f"[VOTING:WRITING] Frame {frame_num}: No matching pose found")
        else:
            logger.debug(f"[VOTING:WRITING] Frame {frame_num}: No keypoints available in poses")

        logger.debug(f"[VOTING:WRITING] Frame {frame_num}: "
                    f"left_hand={left_hand_coords}, right_hand={right_hand_coords}, "
                    f"head_down={head_down}, wrist_dist={wrist_dist:.1f}px")

        # Method 1: Book detection
        for book_data in books:
            book_bbox = book_data['bbox']

            book_in_region = self._bbox_overlap_with_margin(book_bbox, person_bbox, self.person_book_margin)

            if book_in_region:
                left_near = self._check_hand_near_object(left_hand_coords, book_bbox, self.book_hand_margin) if left_hand_coords else False
                right_near = self._check_hand_near_object(right_hand_coords, book_bbox, self.book_hand_margin) if right_hand_coords else False

                logger.debug(f"[VOTING:WRITING] Frame {frame_num}: "
                            f"book_bbox={[int(x) for x in book_bbox]}, "
                            f"book_in_region={book_in_region}, hand_near_book={left_near or right_near}")

                if left_near or right_near:
                    detected_by_book = True
                    break

        # Method 2: Wrist proximity heuristic (hands close together + head down)
        if not detected_by_book and wrist_dist < 300 and head_down:
            detected_by_wrist = True

        detected = detected_by_book or detected_by_wrist
        method = 'book' if detected_by_book else ('wrist' if detected_by_wrist else 'none')

        logger.debug(f"[VOTING:WRITING] Frame {frame_num}: method={method}, VOTE={'YES' if detected else 'NO'}")

        return detected, 1.0 if detected else 0.0, {
            'method': method,
            'by_book': detected_by_book,
            'by_wrist': detected_by_wrist,
            'wrist_dist': wrist_dist,
            'head_down': head_down
        }

    def _verify_packing(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        person_bbox: List[float],
        frame_num: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """Verify packing bags detection in a single frame."""
        backpacks = detections.get('backpack', [])

        logger.debug(f"[VOTING:PACKING] Frame {frame_num}/{total_frames}: backpacks_detected={len(backpacks)}")

        if not backpacks:
            logger.debug(f"[VOTING:PACKING] Frame {frame_num}: No backpacks detected - VOTE=NO")
            return False, 0.0, {'reason': 'no_backpack_detected'}

        # Get wrist coordinates
        left_wrist = None
        right_wrist = None

        if poses.get('keypoints') and len(poses['keypoints']) > 0:
            best_pose = self._find_matching_pose(poses, person_bbox)
            if best_pose is not None:
                if best_pose[9][2] > 0.3:
                    left_wrist = (int(best_pose[9][0]), int(best_pose[9][1]))
                if best_pose[10][2] > 0.3:
                    right_wrist = (int(best_pose[10][0]), int(best_pose[10][1]))

        logger.debug(f"[VOTING:PACKING] Frame {frame_num}: "
                    f"left_wrist={left_wrist}, right_wrist={right_wrist}")

        for bag_data in backpacks:
            bag_bbox = bag_data['bbox']
            conf = bag_data['confidence']

            logger.debug(f"[VOTING:PACKING] Frame {frame_num}: "
                        f"backpack_bbox={[int(x) for x in bag_bbox]}, confidence={conf:.3f}")

            # Check if wrist is inside backpack bbox
            left_inside = self._point_inside_bbox(left_wrist, bag_bbox, self.backpack_wrist_margin) if left_wrist else False
            right_inside = self._point_inside_bbox(right_wrist, bag_bbox, self.backpack_wrist_margin) if right_wrist else False

            logger.debug(f"[VOTING:PACKING] Frame {frame_num}: "
                        f"left_wrist_inside={left_inside}, right_wrist_inside={right_inside}")

            if left_inside or right_inside:
                logger.debug(f"[VOTING:PACKING] Frame {frame_num}: VOTE=YES")
                return True, conf, {
                    'backpack_bbox': bag_bbox,
                    'confidence': conf,
                    'wrist_inside': True
                }

        logger.debug(f"[VOTING:PACKING] Frame {frame_num}: VOTE=NO (wrist not inside backpack)")
        return False, 0.0, {'reason': 'wrist_not_inside_backpack'}

    def _verify_hand_gesture(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        person_bbox: List[float],
        gesture_type: str,
        frame_num: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """Verify hand gesture detection in a single frame."""
        role = 'LP' if gesture_type == 'lp_hand_gesture' else 'ALP'

        logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}/{total_frames}: checking pose...")

        if not poses.get('keypoints') or len(poses['keypoints']) == 0:
            logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}: No pose detected - VOTE=NO")
            return False, 0.0, {'reason': 'no_pose_detected'}

        best_pose = self._find_matching_pose(poses, person_bbox)
        if best_pose is None:
            logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}: No matching pose - VOTE=NO")
            return False, 0.0, {'reason': 'no_matching_pose'}

        # Check both hands for raised gesture
        # YOLO keypoints: 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
        left_shoulder = best_pose[5]
        right_shoulder = best_pose[6]
        left_wrist = best_pose[9]
        right_wrist = best_pose[10]

        gesture_detected = False
        visibility = 0.0

        # Check left hand
        if left_wrist[2] > 0.3 and left_shoulder[2] > 0.4:
            wrist_y = left_wrist[1]
            shoulder_y = left_shoulder[1]
            hand_raised = wrist_y < shoulder_y  # Wrist above shoulder
            visibility = left_wrist[2]

            logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}: "
                        f"left_wrist_y={wrist_y:.0f}, left_shoulder_y={shoulder_y:.0f}, "
                        f"hand_raised={hand_raised}, visibility={visibility:.2f}")

            if hand_raised:
                gesture_detected = True

        # Check right hand
        if not gesture_detected and right_wrist[2] > 0.3 and right_shoulder[2] > 0.4:
            wrist_y = right_wrist[1]
            shoulder_y = right_shoulder[1]
            hand_raised = wrist_y < shoulder_y
            visibility = right_wrist[2]

            logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}: "
                        f"right_wrist_y={wrist_y:.0f}, right_shoulder_y={shoulder_y:.0f}, "
                        f"hand_raised={hand_raised}, visibility={visibility:.2f}")

            if hand_raised:
                gesture_detected = True

        # Check for suppression (hand near backpack = not a gesture)
        suppressed = False
        backpacks = detections.get('backpack', [])
        for bag_data in backpacks:
            bag_bbox = bag_data['bbox']
            left_near = self._check_hand_near_object(
                (int(left_wrist[0]), int(left_wrist[1])) if left_wrist[2] > 0.3 else None,
                bag_bbox, 250
            )
            right_near = self._check_hand_near_object(
                (int(right_wrist[0]), int(right_wrist[1])) if right_wrist[2] > 0.3 else None,
                bag_bbox, 250
            )
            if left_near or right_near:
                suppressed = True
                break

        logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}: suppressed_by_work={suppressed}")

        if suppressed:
            gesture_detected = False

        logger.debug(f"[VOTING:GESTURE:{role}] Frame {frame_num}: VOTE={'YES' if gesture_detected else 'NO'}")

        return gesture_detected, visibility, {
            'hand_raised': gesture_detected,
            'visibility': visibility,
            'suppressed': suppressed
        }

    def _verify_mind_diversion(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        person_bbox: List[float],
        frame_num: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """Verify mind diversion detection in a single frame.

        Mind diversion = head turned sideways (not looking at road ahead)
        BUT suppressed if person is holding/interacting with objects (writing, drinking, etc.)
        """
        logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}/{total_frames}: checking head pose...")

        if not poses.get('keypoints') or len(poses['keypoints']) == 0:
            logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: No pose detected - VOTE=NO")
            return False, 0.0, {'reason': 'no_pose_detected'}

        best_pose = self._find_matching_pose(poses, person_bbox)
        if best_pose is None:
            logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: No matching pose - VOTE=NO")
            return False, 0.0, {'reason': 'no_matching_pose'}

        # YOLO keypoints: 0=nose, 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
        nose = best_pose[0]
        left_shoulder = best_pose[5]
        right_shoulder = best_pose[6]
        left_wrist = best_pose[9]
        right_wrist = best_pose[10]

        # Check visibility of key points
        if nose[2] < 0.3 or left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
            logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: Low visibility "
                        f"(nose={nose[2]:.2f}, l_shoulder={left_shoulder[2]:.2f}, "
                        f"r_shoulder={right_shoulder[2]:.2f}) - VOTE=NO")
            return False, 0.0, {'reason': 'low_visibility'}

        # Calculate reference points
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        if shoulder_width < 10:  # Invalid pose (shoulders too close)
            logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: Invalid pose (shoulder_width={shoulder_width:.1f}) - VOTE=NO")
            return False, 0.0, {'reason': 'invalid_pose'}

        # YAW: Check if head is turned sideways
        # Nose offset from shoulder center as a ratio of shoulder width
        nose_offset_x = abs(nose[0] - shoulder_center_x)
        yaw_ratio = nose_offset_x / shoulder_width

        # YAW threshold: 0.5 = nose is half a shoulder width away from center (~45° turn)
        looking_sideways = yaw_ratio > 0.5

        logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: "
                    f"nose_offset_x={nose_offset_x:.1f}, shoulder_width={shoulder_width:.1f}, "
                    f"yaw_ratio={yaw_ratio:.2f}, looking_sideways={looking_sideways}")

        if not looking_sideways:
            logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: Not looking sideways - VOTE=NO")
            return False, 0.0, {
                'yaw_ratio': yaw_ratio,
                'looking_sideways': False,
                'suppressed': False,
                'looking_away': False
            }

        # SUPPRESSION: Check if person is holding/interacting with objects
        # If hands are near any detected objects, suppress mind diversion
        suppressed = False
        suppression_reason = None

        # Get wrist coordinates
        left_wrist_coords = None
        right_wrist_coords = None
        if left_wrist[2] > 0.3:
            left_wrist_coords = (int(left_wrist[0]), int(left_wrist[1]))
        if right_wrist[2] > 0.3:
            right_wrist_coords = (int(right_wrist[0]), int(right_wrist[1]))

        # Check proximity to phones
        for phone_data in detections.get('cell_phone', []):
            phone_bbox = phone_data['bbox']
            if self._check_hand_near_object(left_wrist_coords, phone_bbox, 150) or \
               self._check_hand_near_object(right_wrist_coords, phone_bbox, 150):
                suppressed = True
                suppression_reason = 'holding_phone'
                break

        # Check proximity to books (writing)
        if not suppressed:
            for book_data in detections.get('book', []):
                book_bbox = book_data['bbox']
                if self._check_hand_near_object(left_wrist_coords, book_bbox, 200) or \
                   self._check_hand_near_object(right_wrist_coords, book_bbox, 200):
                    suppressed = True
                    suppression_reason = 'writing'
                    break

        # Check proximity to backpacks/bags
        if not suppressed:
            for bag_data in detections.get('backpack', []):
                bag_bbox = bag_data['bbox']
                if self._check_hand_near_object(left_wrist_coords, bag_bbox, 150) or \
                   self._check_hand_near_object(right_wrist_coords, bag_bbox, 150):
                    suppressed = True
                    suppression_reason = 'handling_bag'
                    break

        # Check proximity to bottles (YOLO class 'bottle' = 39)
        if not suppressed:
            for bottle_data in detections.get('bottle', []):
                bottle_bbox = bottle_data['bbox']
                if self._check_hand_near_object(left_wrist_coords, bottle_bbox, 100) or \
                   self._check_hand_near_object(right_wrist_coords, bottle_bbox, 100):
                    suppressed = True
                    suppression_reason = 'holding_bottle'
                    break

        logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: "
                    f"suppressed={suppressed}, reason={suppression_reason}")

        # Mind diversion = looking sideways AND NOT holding anything
        looking_away = looking_sideways and not suppressed

        logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: "
                    f"looking_away={looking_away} (sideways={looking_sideways} AND NOT suppressed={not suppressed})")
        logger.debug(f"[VOTING:MIND_DIV] Frame {frame_num}: VOTE={'YES' if looking_away else 'NO'}")

        return looking_away, 1.0 if looking_away else 0.0, {
            'yaw_ratio': yaw_ratio,
            'looking_sideways': looking_sideways,
            'suppressed': suppressed,
            'suppression_reason': suppression_reason,
            'looking_away': looking_away
        }

    def _verify_group(
        self,
        frame: np.ndarray,
        detections: Dict,
        frame_num: int,
        total_frames: int
    ) -> Tuple[bool, float, Dict]:
        """Verify group detection (>2 people) in a single frame."""
        persons = detections.get('person', [])

        logger.debug(f"[VOTING:GROUP] Frame {frame_num}/{total_frames}: persons_detected={len(persons)}")

        # Deduplicate overlapping person detections
        dedup_persons = self._deduplicate_boxes(persons, iou_threshold=0.3)
        dedup_count = len(dedup_persons)

        logger.debug(f"[VOTING:GROUP] Frame {frame_num}: "
                    f"person_bboxes={[[int(x) for x in bbox] for bbox in persons[:3]]}...")
        logger.debug(f"[VOTING:GROUP] Frame {frame_num}: deduplicated_count={dedup_count}")

        is_group = dedup_count > 2

        logger.debug(f"[VOTING:GROUP] Frame {frame_num}: VOTE={'YES' if is_group else 'NO'}")

        return is_group, 1.0 if is_group else 0.0, {
            'raw_count': len(persons),
            'dedup_count': dedup_count,
            'is_group': is_group
        }

    # ==================== Helper Methods ====================

    def _find_matching_pose(self, poses: Dict, person_bbox: List[float]) -> Optional[np.ndarray]:
        """Find the pose that best matches the person bounding box."""
        if not poses.get('keypoints') or not poses.get('boxes'):
            # If no boxes, return first pose
            if poses.get('keypoints') and len(poses['keypoints']) > 0:
                return poses['keypoints'][0]
            return None

        best_iou = 0
        best_pose = None

        for i, (kp, box) in enumerate(zip(poses['keypoints'], poses['boxes'])):
            iou = self._calculate_iou(box, person_bbox)
            if iou > best_iou:
                best_iou = iou
                best_pose = kp

        # Return best match if IoU > 0.3, otherwise first pose
        if best_iou > 0.3:
            return best_pose
        elif poses['keypoints']:
            return poses['keypoints'][0]
        return None

    def _bbox_overlap_with_margin(
        self,
        bbox1: List[float],
        bbox2: List[float],
        margin: int
    ) -> bool:
        """Check if two bounding boxes overlap with margin."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Expand bbox2 by margin
        x1_2 -= margin
        y1_2 -= margin
        x2_2 += margin
        y2_2 += margin

        # Check overlap
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

    def _check_hand_near_object(
        self,
        hand_coords: Optional[Tuple[int, int]],
        obj_bbox: List[float],
        margin: int
    ) -> bool:
        """Check if hand coordinates are near an object bounding box."""
        if hand_coords is None:
            return False

        hx, hy = hand_coords
        x1, y1, x2, y2 = obj_bbox

        # Expand bbox by margin
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin

        return x1 <= hx <= x2 and y1 <= hy <= y2

    def _point_inside_bbox(
        self,
        point: Optional[Tuple[int, int]],
        bbox: List[float],
        margin: int = 0
    ) -> bool:
        """Check if a point is inside a bounding box."""
        if point is None:
            return False

        px, py = point
        x1, y1, x2, y2 = bbox

        # Expand bbox by margin
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin

        return x1 <= px <= x2 and y1 <= py <= y2

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def _deduplicate_boxes(self, boxes: List[List[float]], iou_threshold: float = 0.3) -> List[List[float]]:
        """Remove overlapping bounding boxes using NMS-like approach."""
        if not boxes:
            return []

        # Sort by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

        keep = []
        for box in sorted_boxes:
            should_keep = True
            for kept_box in keep:
                if self._calculate_iou(box, kept_box) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(box)

        return keep

    # ==================== Debug Frame Saving Methods ====================

    def _save_debug_frames(
        self,
        frames: List[np.ndarray],
        batch_detections: List[Dict],
        batch_poses: List[Dict],
        votes: List[bool],
        activity_type: str,
        timestamp_sec: float,
        person_bbox: List[float],
        is_confirmed: bool,
        vote_ratio: float,
        threshold: float,
        video_path: str
    ) -> None:
        """
        Save annotated debug frames for troubleshooting.

        Creates a subfolder for each voting session with annotated frames showing:
        - Detection bounding boxes (phones, books, backpacks, persons)
        - Pose keypoints
        - Vote result (YES/NO) for each frame
        - Overall voting summary

        Args:
            frames: List of extracted frames
            batch_detections: Detection results for each frame
            batch_poses: Pose results for each frame
            votes: Vote results for each frame
            activity_type: Type of activity being verified
            timestamp_sec: Timestamp in video
            person_bbox: Bounding box of person
            is_confirmed: Final voting decision
            vote_ratio: Percentage of positive votes
            threshold: Required threshold
            video_path: Path to source video
        """
        try:
            # Create directory structure: voting_debug_frames/<video_name>/<activity>_<timestamp>_<result>/
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            result_str = "CONFIRMED" if is_confirmed else "REJECTED"
            timestamp_str = f"{timestamp_sec:.2f}s".replace(".", "_")
            session_time = datetime.now().strftime("%H%M%S")

            session_dir = os.path.join(
                self.debug_frames_base_dir,
                video_name,
                f"{activity_type}_{timestamp_str}_{result_str}_{session_time}"
            )

            os.makedirs(session_dir, exist_ok=True)

            logger.info(f"[VOTING DEBUG] Saving {len(frames)} annotated frames to: {session_dir}")

            # Save each frame with annotations
            for i, (frame, detections, poses, vote) in enumerate(zip(frames, batch_detections, batch_poses, votes)):
                annotated_frame = self._annotate_frame(
                    frame=frame.copy(),
                    detections=detections,
                    poses=poses,
                    vote=vote,
                    frame_idx=i,
                    total_frames=len(frames),
                    activity_type=activity_type,
                    person_bbox=person_bbox
                )

                # Save frame
                frame_filename = f"frame_{i+1:02d}_{'YES' if vote else 'NO'}.jpg"
                frame_path = os.path.join(session_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)

            # Save summary image (first frame with overall results)
            summary_frame = self._create_summary_frame(
                frames[0].copy() if frames else np.zeros((480, 640, 3), dtype=np.uint8),
                activity_type=activity_type,
                timestamp_sec=timestamp_sec,
                votes=votes,
                vote_ratio=vote_ratio,
                threshold=threshold,
                is_confirmed=is_confirmed,
                person_bbox=person_bbox
            )
            summary_path = os.path.join(session_dir, f"_SUMMARY_{result_str}.jpg")
            cv2.imwrite(summary_path, summary_frame)

            logger.info(f"[VOTING DEBUG] Saved {len(frames) + 1} frames to {session_dir}")

        except Exception as e:
            logger.error(f"[VOTING DEBUG] Error saving debug frames: {e}")

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: Dict,
        poses: Dict,
        vote: bool,
        frame_idx: int,
        total_frames: int,
        activity_type: str,
        person_bbox: List[float]
    ) -> np.ndarray:
        """
        Annotate a frame with detection boxes, poses, and vote result.

        Args:
            frame: Frame to annotate
            detections: YOLO detections for this frame
            poses: Pose data for this frame
            vote: Whether this frame voted YES
            frame_idx: Index of this frame
            total_frames: Total number of frames
            activity_type: Type of activity
            person_bbox: Original person bounding box

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]

        # Colors
        COLOR_PERSON = (0, 255, 0)      # Green
        COLOR_PHONE = (0, 0, 255)       # Red
        COLOR_BOOK = (255, 165, 0)      # Orange
        COLOR_BACKPACK = (128, 0, 128)  # Purple
        COLOR_POSE = (255, 255, 0)      # Cyan
        COLOR_YES = (0, 255, 0)         # Green
        COLOR_NO = (0, 0, 255)          # Red
        COLOR_BBOX = (255, 0, 255)      # Magenta for person bbox

        # Draw person bounding box (from trigger)
        if person_bbox:
            x1, y1, x2, y2 = [int(v) for v in person_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
            cv2.putText(frame, "Person ROI", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 2)

        # Draw detected persons
        for i, person_box in enumerate(detections.get('person', [])):
            x1, y1, x2, y2 = [int(v) for v in person_box[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            cv2.putText(frame, f"Person {i+1}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PERSON, 1)

        # Draw cell phones
        for phone_data in detections.get('cell_phone', []):
            bbox = phone_data['bbox'] if isinstance(phone_data, dict) else phone_data
            conf = phone_data.get('confidence', 0) if isinstance(phone_data, dict) else 0
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PHONE, 3)
            cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PHONE, 2)

        # Draw books
        for book_data in detections.get('book', []):
            bbox = book_data['bbox'] if isinstance(book_data, dict) else book_data
            conf = book_data.get('confidence', 0) if isinstance(book_data, dict) else 0
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOOK, 2)
            cv2.putText(frame, f"BOOK {conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOOK, 2)

        # Draw backpacks
        for bag_data in detections.get('backpack', []):
            bbox = bag_data['bbox'] if isinstance(bag_data, dict) else bag_data
            conf = bag_data.get('confidence', 0) if isinstance(bag_data, dict) else 0
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BACKPACK, 2)
            cv2.putText(frame, f"BAG {conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BACKPACK, 2)

        # Draw pose keypoints
        if poses.get('keypoints'):
            for keypoints in poses['keypoints']:
                self._draw_pose_keypoints(frame, keypoints, COLOR_POSE)

        # Draw vote result banner at top
        vote_color = COLOR_YES if vote else COLOR_NO
        vote_text = "VOTE: YES" if vote else "VOTE: NO"

        # Draw banner background
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        # Draw frame info
        info_text = f"Frame {frame_idx + 1}/{total_frames} | {activity_type.upper()} | {vote_text}"
        cv2.putText(frame, info_text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, vote_color, 2)

        return frame

    def _draw_pose_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw pose keypoints on frame."""
        # YOLO keypoint connections for skeleton
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > 0.3:  # visibility check
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 4, color, -1)

                # Highlight wrists (9, 10) with larger circles
                if i in [9, 10]:
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)

        # Draw skeleton connections
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                if len(start_kp) >= 3 and len(end_kp) >= 3:
                    if start_kp[2] > 0.3 and end_kp[2] > 0.3:
                        start_pt = (int(start_kp[0]), int(start_kp[1]))
                        end_pt = (int(end_kp[0]), int(end_kp[1]))
                        cv2.line(frame, start_pt, end_pt, color, 2)

    def _create_summary_frame(
        self,
        frame: np.ndarray,
        activity_type: str,
        timestamp_sec: float,
        votes: List[bool],
        vote_ratio: float,
        threshold: float,
        is_confirmed: bool,
        person_bbox: List[float]
    ) -> np.ndarray:
        """
        Create a summary frame showing overall voting results.

        Args:
            frame: Base frame to annotate
            activity_type: Type of activity
            timestamp_sec: Timestamp in video
            votes: List of vote results
            vote_ratio: Percentage of positive votes
            threshold: Required threshold
            is_confirmed: Final decision
            person_bbox: Person bounding box

        Returns:
            Summary frame with voting results overlay
        """
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Colors
        result_color = (0, 255, 0) if is_confirmed else (0, 0, 255)
        result_text = "CONFIRMED" if is_confirmed else "REJECTED"

        # Draw title
        cv2.putText(frame, f"VOTING SUMMARY: {activity_type.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw result
        cv2.putText(frame, f"RESULT: {result_text}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)

        # Draw statistics
        positive_votes = sum(votes)
        cv2.putText(frame, f"Timestamp: {timestamp_sec:.2f}s", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Votes: {positive_votes}/{len(votes)} ({vote_ratio*100:.1f}%)", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Threshold: {threshold*100:.0f}%", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw vote breakdown
        vote_str = " ".join(['Y' if v else 'N' for v in votes])
        cv2.putText(frame, f"Breakdown: [{vote_str}]", (10, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw person bbox
        if person_bbox:
            x1, y1, x2, y2 = [int(v) for v in person_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return frame
