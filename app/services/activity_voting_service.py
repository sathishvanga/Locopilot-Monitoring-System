"""
Activity Voting Service - Reduces false positives through frame-by-frame voting.

An activity must be detected in N out of M frames to be confirmed. This prevents
single-frame false positives and provides temporal smoothing for activity detection.

FIX 2.1: Implements voting-based activity validation.
"""

from collections import deque
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ActivityVotingService:
    """
    Voting-based activity validation to reduce false positives.

    An activity must be detected in N (votes_required) out of M (window_frames)
    consecutive frames to be confirmed as a valid detection.

    This provides:
    1. Temporal smoothing - single-frame detections are filtered out
    2. Confidence aggregation - multiple detections increase confidence
    3. Hysteresis - prevents rapid on/off flickering of activity states
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the voting service with optional configuration.

        Args:
            config: Optional dictionary of voting configurations per activity type.
                   Format: {'activity_type': {'votes_required': N, 'window_frames': M}}
        """
        # Default voting configuration per activity type
        # Higher votes_required = stricter detection (fewer false positives)
        # Larger window_frames = more temporal smoothing
        self.voting_config = {
            'writing': {'votes_required': 2, 'window_frames': 4},      # 2/4 frames
            'mind_diversion': {'votes_required': 3, 'window_frames': 5},  # 3/5 frames - stricter
            'packing_bags': {'votes_required': 2, 'window_frames': 4},  # 2/4 frames
            'group_detected': {'votes_required': 4, 'window_frames': 6},  # 4/6 frames - very strict
            'lp_hand_gesture': {'votes_required': 1, 'window_frames': 2},  # 1/2 frames - instant with minimal smoothing
            'alp_hand_gesture': {'votes_required': 1, 'window_frames': 2},  # 1/2 frames - instant with minimal smoothing
            'cell_phone': {'votes_required': 2, 'window_frames': 3},   # 2/3 frames - responsive
            'microsleep': {'votes_required': 2, 'window_frames': 3},   # 2/3 frames - responsive (safety)
            'sleep': {'votes_required': 3, 'window_frames': 4},        # 3/4 frames
            'no_person_detected': {'votes_required': 3, 'window_frames': 5},  # 3/5 frames
        }

        # Override defaults with provided config
        if config:
            for activity_type, settings in config.items():
                if activity_type in self.voting_config:
                    self.voting_config[activity_type].update(settings)
                else:
                    self.voting_config[activity_type] = settings

        # Per-person voting history
        # Format: {person_idx: {activity_type: deque([True/False, ...])}}
        self.vote_history: Dict[int, Dict[str, deque]] = {}

        # Global voting history (for activities not tied to specific person)
        # Format: {activity_type: deque([True/False, ...])}
        self.global_vote_history: Dict[str, deque] = {}

        logger.info(f"Activity voting service initialized with config: {self.voting_config}")

    def vote(
        self,
        activity_type: str,
        detected: bool,
        person_idx: Optional[int] = None
    ) -> bool:
        """
        Cast a vote for an activity detection and return whether it's confirmed.

        This method:
        1. Records the detection result (True/False) in a sliding window
        2. Counts votes (True detections) in the window
        3. Returns True only if votes >= votes_required

        Args:
            activity_type: Type of activity ('writing', 'mind_diversion', etc.)
            detected: Whether the activity was detected in current frame
            person_idx: Optional person index for per-person voting

        Returns:
            bool: True if activity is confirmed (enough votes), False otherwise
        """
        # Get configuration for this activity type
        config = self.voting_config.get(
            activity_type,
            {'votes_required': 2, 'window_frames': 4}  # Default
        )
        votes_required = config['votes_required']
        window_frames = config['window_frames']

        # Get or create vote history deque
        if person_idx is not None:
            # Per-person voting
            if person_idx not in self.vote_history:
                self.vote_history[person_idx] = {}

            if activity_type not in self.vote_history[person_idx]:
                self.vote_history[person_idx][activity_type] = deque(maxlen=window_frames)

            history = self.vote_history[person_idx][activity_type]
        else:
            # Global voting (for activities like group_detected)
            if activity_type not in self.global_vote_history:
                self.global_vote_history[activity_type] = deque(maxlen=window_frames)

            history = self.global_vote_history[activity_type]

        # Cast vote
        history.append(detected)

        # Count votes
        votes = sum(history)
        confirmed = votes >= votes_required

        # Debug logging for significant events
        if confirmed and not detected:
            # Activity confirmed even though current frame didn't detect it
            logger.debug(
                f"Activity '{activity_type}' CONFIRMED via voting "
                f"(votes={votes}/{votes_required}, window={len(history)}/{window_frames})"
            )
        elif detected and not confirmed:
            # Detection not yet confirmed (building up votes)
            logger.debug(
                f"Activity '{activity_type}' detected but not confirmed yet "
                f"(votes={votes}/{votes_required})"
            )

        return confirmed

    def get_vote_count(
        self,
        activity_type: str,
        person_idx: Optional[int] = None
    ) -> tuple:
        """
        Get current vote count for an activity.

        Args:
            activity_type: Type of activity
            person_idx: Optional person index

        Returns:
            tuple: (current_votes, votes_required, window_size)
        """
        config = self.voting_config.get(
            activity_type,
            {'votes_required': 2, 'window_frames': 4}
        )

        if person_idx is not None:
            if person_idx in self.vote_history and activity_type in self.vote_history[person_idx]:
                history = self.vote_history[person_idx][activity_type]
                return (sum(history), config['votes_required'], len(history))
        else:
            if activity_type in self.global_vote_history:
                history = self.global_vote_history[activity_type]
                return (sum(history), config['votes_required'], len(history))

        return (0, config['votes_required'], 0)

    def reset_votes(
        self,
        activity_type: Optional[str] = None,
        person_idx: Optional[int] = None
    ):
        """
        Reset vote history for an activity or all activities.

        Args:
            activity_type: Optional activity type to reset. If None, resets all.
            person_idx: Optional person index. If None, affects global history.
        """
        if person_idx is not None:
            if person_idx in self.vote_history:
                if activity_type:
                    if activity_type in self.vote_history[person_idx]:
                        self.vote_history[person_idx][activity_type].clear()
                else:
                    self.vote_history[person_idx].clear()
        else:
            if activity_type:
                if activity_type in self.global_vote_history:
                    self.global_vote_history[activity_type].clear()
            else:
                self.global_vote_history.clear()

    def reset_all(self):
        """Reset all voting history (both per-person and global)."""
        self.vote_history.clear()
        self.global_vote_history.clear()
        logger.debug("All voting history reset")

    def update_config(self, activity_type: str, votes_required: int, window_frames: int):
        """
        Update voting configuration for an activity type.

        Args:
            activity_type: Activity type to configure
            votes_required: Minimum votes needed to confirm
            window_frames: Size of sliding window
        """
        self.voting_config[activity_type] = {
            'votes_required': votes_required,
            'window_frames': window_frames
        }
        logger.info(f"Updated voting config for '{activity_type}': {votes_required}/{window_frames}")


# Singleton instance
_voting_service: Optional[ActivityVotingService] = None


def get_activity_voting_service(config: Optional[Dict[str, Any]] = None) -> ActivityVotingService:
    """
    Get singleton instance of the activity voting service.

    Args:
        config: Optional configuration to apply on first initialization

    Returns:
        ActivityVotingService: Singleton instance
    """
    global _voting_service
    if _voting_service is None:
        _voting_service = ActivityVotingService(config)
    return _voting_service
