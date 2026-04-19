"""Concrete frame-pipeline stages (task 0002).

Each stage encapsulates a contiguous section of the original
``LocopilotActivityMonitor._process_frames_core`` function, extracted
verbatim so behavior remains byte-identical to the pre-refactor
implementation. Stages should only be reordered or edited once a snapshot
test is in place (see task 0006).
"""

from .face_mesh_stage import FaceMeshStage
from .object_detect_stage import ObjectDetectStage
from .person_dedup_stage import PersonDedupStage
from .role_identify_stage import RoleIdentifyStage
from .group_detect_stage import GroupDetectStage
from .train_motion_detect_stage import TrainMotionDetectStage
from .per_person_activities_stage import PerPersonActivitiesStage
from .state_machine_gate_stage import StateMachineGateStage
from .sleep_writing_override_stage import SleepWritingOverrideStage
from .gesture_coordination_stage import GestureCoordinationStage
from .train_motion_suppress_stage import TrainMotionSuppressStage
from .no_person_schedule_suppress_stage import NoPersonScheduleSuppressStage
from .temporal_filter_stage import TemporalFilterStage
from .evidence_stage import EvidenceStage

__all__ = [
    "FaceMeshStage",
    "ObjectDetectStage",
    "PersonDedupStage",
    "RoleIdentifyStage",
    "GroupDetectStage",
    "TrainMotionDetectStage",
    "PerPersonActivitiesStage",
    "StateMachineGateStage",
    "SleepWritingOverrideStage",
    "GestureCoordinationStage",
    "TrainMotionSuppressStage",
    "NoPersonScheduleSuppressStage",
    "TemporalFilterStage",
    "EvidenceStage",
]
