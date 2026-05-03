"""Pipeline-2 verifier: VLM-based false-positive filter for activity events.

Sends each confirmed activity's keyframe (``activityImage``) plus an
activity-specific prompt to a vLLM OpenAI-compatible endpoint and parses the
JSON verdict. Designed to fail-open: if the endpoint is down or slow, the
Pipeline-1 verdict passes through unchanged — VLM downtime must never silently
swallow real violations.

Two modes (controlled by ``VLM_SHADOW_MODE``):

- **Shadow** (default): every verified activity gets a ``vlm_review`` field
  attached to ``activities.json`` with verdict + reasoning + latency, but
  nothing is dropped. Used to compare VLM judgement against ground truth
  before enabling enforcement.
- **Enforcement**: activities with ``verdict=FALSE_POSITIVE`` and
  ``confidence >= VLM_DROP_THRESHOLD`` are filtered out of the activities list
  passed downstream to S3 upload + external API.

Prompt design follows the patterns validated in the Phase-0 spike (2026-04-26),
which scored ~85-90% precision on FP detection across the writing/eating
confounder archetypes (brake handle, radio handset, idle-lap, static book).
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.config import get_settings


logger = logging.getLogger(__name__)


_PROMPT_WRITING = """You are a railway safety auditor reviewing CCTV footage from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. Up to 2 persons visible:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

A classical CV pipeline flagged this frame as "WRITING IN LOG BOOK". The pipeline
does NOT specify which person — either LP or ALP could be the writer. Your job
is to VERIFY or REFUTE this classification by examining the image directly.

The image may have person/object bounding boxes drawn on it by the pipeline (green
person box, amber book box, white skeleton). Do NOT trust those overlays — verify
the activity from the underlying scene. A box labelled "book 0.96" may be drawn on
something that is not actually a book; verify yourself.

True writing requires ALL of these to be visibly true for the SAME person:
  (a) An OPEN book / notepad / paper is visible on the desk or in their lap
      (closed books and folded papers do NOT count)
  (b) Their hand is in PHYSICAL CONTACT with the open page — a fingertip or pen
      tip is visibly touching paper. Hand resting NEAR the book, or hand merely
      in the same image region as the book, does NOT qualify.
  (c) A pen/pencil is visible in the writing hand, OR a fingertip is unambiguously
      on the page surface.
  (d) Head is tilted DOWN toward the open page.

If you cannot see (b) AND (c) confirmed for one specific person, the verdict is
FALSE_POSITIVE. When genuinely uncertain, return UNCERTAIN with confidence ≤ 0.5.
Do NOT return TRUE_POSITIVE just because a book is visible somewhere in the frame.

Common confounders the classical pipeline misclassifies as writing:
  - Hands resting in the lap with head bent down (idle posture, no book contact)
  - A book sitting unattended on the desk while crew operates controls (static-book FP)
  - Holding a folded paper or clipboard at chest level while looking forward
  - Holding the brake handle / throttle lever (long stick with knob between the
    knees or on the console) — verify the hand is on PAPER, not on a metal control
  - Holding a railway radio handset to face/ear (brick-shaped, often with coiled cord)

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one short sentence describing what you actually see>"
}"""

_PROMPT_EATING = """You are a railway safety auditor reviewing CCTV footage from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. Up to 2 persons visible:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

A classical CV pipeline flagged this frame as "EATING OR DRINKING". The pipeline does
NOT specify which person. Your job is to VERIFY or REFUTE this classification by
examining the image directly.

The image may have person/object bounding boxes drawn on it by the pipeline. Do NOT
trust those overlays — verify the activity from the underlying scene.

True eating/drinking requires ALL of these for the SAME person:
  (a) A clearly identifiable food item, cup, bottle, or similar consumable is in their hand
  (b) That object is being moved toward the mouth OR is held at/near the lips
  (c) The hand is not on a control, lever, handset, or radio

If the object is a railway radio handset (brick-shaped, often with a coiled cord), it
is NOT eating/drinking — return FALSE_POSITIVE regardless of hand position.

When genuinely uncertain, return UNCERTAIN with confidence ≤ 0.5. Do NOT return
TRUE_POSITIVE just because a hand is near the face.

Common confounders the classical pipeline misclassifies as eating/drinking:
  - A cup or bottle resting on the desk while no one is touching it (static-object FP)
  - Person wiping face, scratching nose, adjusting cap, or yawning with hand near face
  - Hand at face but no object visible in it (idle gesture)
  - Holding a railway radio handset to face/ear

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one short sentence describing what you actually see>"
}"""

_PROMPT_PACKING = """You are a railway safety auditor reviewing CCTV footage from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. Up to 2 persons visible:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

A classical CV pipeline flagged this frame as "PACKING BAGS". The pipeline does NOT
specify which person. Your job is to VERIFY or REFUTE this classification by
examining the image directly.

The image may have person/object bounding boxes drawn on it by the pipeline. Do NOT
trust those overlays — verify the activity from the underlying scene.

True packing requires ALL of these for the SAME person:
  (a) A bag, backpack, suitcase, or duffel is visible and clearly identifiable
  (b) That person's hand is INSIDE the bag opening, or actively gripping/lifting/
      manipulating it (zipping, lifting, pushing items in/out)
  (c) Body posture is clearly oriented toward the bag (leaning, bent over)

When genuinely uncertain, return UNCERTAIN with confidence ≤ 0.5. A bag merely
visible in the frame is not packing — the hand must be on or inside it.

Common confounders the classical pipeline misclassifies as packing:
  - A bag/suitcase visible on the floor or seat but no one is interacting with it
  - Crew reaching for controls and a bag happens to be in the same image region
  - Crew member standing near a bag during a station stop (handover, not packing)
  - A piece of equipment that resembles a bag (cushion, jacket, kit) — only claim
    "bag" if you can clearly see a backpack/suitcase/duffel shape.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one short sentence describing what you actually see>"
}"""

_PROMPT_CELL_PHONE = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. The frame may contain up to 2 persons:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME activity
burst, ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames before deciding. If the activity is clearly
happening in ANY single frame, that is sufficient for TRUE_POSITIVE. If NO frame shows
the activity clearly, return FALSE_POSITIVE — do not generalise from posture alone, the
hand-on-object evidence must be visible in at least one frame. Note in `evidence_frame`
which frame number provides the strongest evidence (1-N). If only one frame is shown
or no frame is decisive, return 0.

A classical CV pipeline flagged this frame as "CELL PHONE USE". The pipeline does
NOT specify which person — either LP or ALP could be the one using a phone. Your
job is to verify whether ANYONE in the frame is actually using a personal cell phone
RIGHT NOW.

A railway RADIO HANDSET (brick-shaped, often with a coiled cord, mounted in the cab)
is OFFICIAL equipment and is NOT a cell phone — even if it is held to the ear,
the verdict must be FALSE_POSITIVE.

True cell phone use requires:
  (a) A clearly smartphone-shaped object (flat, slim, rectangular, glossy screen)
      visible in the person's hand
  (b) That object is held to the ear OR being looked at in the lap / in front of the face
  (c) The object is NOT a railway radio handset (brick shape / coiled cord)

Common confounders the classical pipeline misclassifies as cell phone use:
  - Holding a railway radio handset to the ear (it is official equipment)
  - Scratching face, adjusting cap, wiping forehead, hand to ear with nothing in it
  - A phone sitting on the desk while no one is touching it
  - A small object in the hand that you can't clearly identify — return UNCERTAIN
    rather than guessing "smartphone".

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
Cell phone use is a violation regardless of whether the train is stopped or running,
so do NOT use motion to decide the verdict — only report what you observe.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF. If verdict=FALSE_POSITIVE and you are sure no activity is happening, set confidence ≥ 0.80. If verdict=TRUE_POSITIVE and you are sure the activity is happening, set confidence ≥ 0.80. Use ≤ 0.5 only when you are genuinely unsure. Do NOT set 0.0 to mean "no activity"; that means "I am 0% sure of my verdict">,
  "object_in_hand": "smartphone" | "radio_handset" | "nothing_visible" | "unclear",
  "object_position": "at_ear" | "in_lap" | "in_front_of_face" | "on_desk" | "other",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""


_PROMPT_SLEEP = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. The frame may contain up to 2 persons:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME activity
burst, ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames before deciding. Sleep is by definition
SUSTAINED — it must be visible in MULTIPLE frames (not a single-frame artifact).
Note in `evidence_frame` which frame number provides the strongest evidence (1-N).
If only one frame is shown or no frame is decisive, return 0.

A classical CV pipeline flagged this strip as "SLEEP" for one of the persons. The
pipeline does NOT specify which person. Your job is to verify whether ANYONE in
the strip is actually sleeping or unresponsive RIGHT NOW.

True sleep requires AT LEAST TWO of these to be unambiguously visible across the strip
(consistent across the frames shown):
  (a) Eyes are clearly CLOSED for the duration of the strip (not a single-frame blink)
  (b) Body is RECLINED — head tilted back / fallen sideways / leaning heavily on the
      seat or wall; NOT a writing/reading-forward posture
  (c) Hands are completely STILL, in lap or by side; no active operation of controls
  (d) Head is motionless across frames (not turning, not nodding, not glancing)

If only ONE of (a)-(d) holds, return UNCERTAIN. If none hold, return FALSE_POSITIVE.

Common confounders the classical pipeline misclassifies as sleep:
  - Head bent down for writing or reading the logbook (eyes appear closed from
    the overhead angle, but the person is leaning FORWARD, not back/sideways)
  - Brief blink visible in only one frame (single-frame eye closure does NOT count)
  - Talking with ALP, head momentarily turned away or down
  - Resting head briefly during a station stop while still alert
  - Looking down at gauges, controls, or a phone — head is down but eyes likely open
  - LP wearing a cap that shadows the eyes — eyes may appear closed when they're not

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
Sleep is a violation regardless of motion state — do NOT use motion to decide
the verdict, only report what you observe.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF. If verdict=FALSE_POSITIVE and you are sure no sleep is happening, set confidence ≥ 0.80. If verdict=TRUE_POSITIVE and you are sure of sleep, set confidence ≥ 0.80. Use ≤ 0.5 only when you are genuinely unsure>,
  "eyes_closed": <true|false>,
  "body_reclined": <true|false>,
  "hands_still": <true|false>,
  "head_motionless_across_frames": <true|false>,
  "primary_confounder": "writing_pose" | "reading" | "cap_shadow" | "talking" | "control_op" | "blink_only" | "none" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""

_PROMPT_MIND_DIVERSION = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. The frame may contain up to 2 persons:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME activity
burst, ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames before deciding — mind diversion is a
SUSTAINED behaviour, not a single-frame glance. Note in `evidence_frame` which
frame number provides the strongest evidence (1-N). If only one frame is shown
or no frame is decisive, return 0.

A classical CV pipeline flagged this strip as "MIND DIVERSION" — the LP/ALP's
attention has drifted away from the track ahead. The pipeline does NOT specify
which person. Your job is to verify whether ANYONE in the strip is actually
sustaining attention away from the controls.

True mind_diversion requires:
  (a) The person's HEAD is clearly turned AWAY from the forward/track direction
      (toward the side window, the cab door, or backwards) for SUSTAINED time
      across multiple frames
  (b) Their gaze is NOT on the controls, gauges, or the logbook
  (c) The behaviour is consistent across the strip (not a one-off head turn)

If the person's head turns to a side gauge, the radio handset, the logbook, or
their crew partner for less than the full strip duration, that is NOT mind
diversion — return FALSE_POSITIVE.

Common confounders the classical pipeline misclassifies as mind_diversion:
  - Brief head turn to read a side gauge (returns to forward immediately)
  - Looking at the ALP/LP for crew coordination or hand gesture exchange
  - Reading the logbook (head down, not toward window)
  - Adjusting cap or wiping forehead
  - One-frame head turn caused by a vibration / track lurch

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
Mind diversion is only a violation while the train is RUNNING; if you can clearly
see the train is stopped at a station, lower your confidence in any TP verdict.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF; ≥0.80 when sure, ≤0.5 only when genuinely unsure>,
  "head_direction": "window" | "forward" | "side_gauge" | "down" | "back" | "alp_or_lp" | "unclear",
  "sustained_across_frames": <true|false>,
  "primary_confounder": "side_gauge_glance" | "alp_interaction" | "logbook_read" | "brief_turn" | "cap_adjust" | "none" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""

_PROMPT_NO_PERSON = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. The cabin normally contains
up to 2 persons (LP at the controls, ALP standing or seated nearby).

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME burst,
ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames; an unattended cabin should be empty
in EVERY frame. If a person appears in any frame, the cabin is NOT unattended.
Note in `evidence_frame` which frame number provides the strongest evidence (1-N).
If only one frame is shown or no frame is decisive, return 0.

A classical CV pipeline flagged this strip as "NO PERSON DETECTED" — the cabin
appears unattended. Your job is to verify whether the cabin is genuinely empty
or whether one or both crew members are simply OUT OF VIEW (occluded, crouched,
or partially clipped by the frame edge).

The cabin is genuinely UNATTENDED only if:
  (a) Neither LP nor ALP is visible anywhere in any frame of the strip
  (b) No partial body parts (legs, arm, head) are peeking from frame edges
  (c) No person is crouched behind the seat, control panel, or equipment

If you can see a person — even partially — in any frame, the cabin is NOT
unattended. Return FALSE_POSITIVE.

Common confounders the classical pipeline misclassifies as no-person:
  - LP or ALP bent over below the seat-back / dashboard line (occluded by furniture)
  - Person standing close to the camera, only legs/feet in the lower frame edge
  - Heavy backlight from the windshield washing out a person who is actually there
  - A jacket/uniform on the seat-back mistaken for a person being absent
  - Camera glare or compression artifacts

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
An unattended cabin while the train is RUNNING is the most serious case; while
STOPPED it can be normal during crew changeover at a station — note motion in
`train_appears_to_be` so the customer can triage accordingly.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF; ≥0.80 when sure, ≤0.5 only when genuinely unsure>,
  "cabin_empty_in_all_frames": <true|false>,
  "lp_visible_anywhere": <true|false>,
  "alp_visible_anywhere": <true|false>,
  "partial_body_in_edge": <true|false>,
  "primary_confounder": "occluded_by_seat" | "edge_clipped" | "backlight" | "jacket_on_seat" | "none" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""

_PROMPT_GROUP = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. Normally only LP and ALP
should be present. A SUPERVISOR visit is permitted (typically 3-5 people total
for a few minutes). MORE than 5 people in the cab is the violation we are
verifying.

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME burst,
ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames and pick the one that shows the
clearest count of distinct PEOPLE PHYSICALLY PRESENT in the cabin. Note that
frame in `evidence_frame`. If only one frame is shown or no frame is decisive,
return 0.

A classical CV pipeline flagged this strip as "GROUP DETECTED" — more than 5
distinct people in the cabin. The pipeline OVER-FIRES on this — duplicates,
reflections, posters, and outside-the-window people often inflate the count.
**Do NOT trust the classical flag. Count the real people yourself.**

True group_detected requires:
  (a) SIX OR MORE distinct, real people physically present INSIDE the cabin
      in at least one frame of the strip
  (b) These are not the same person counted twice (multiple bboxes on one body)
  (c) These are not posters, photos, mannequins, reflections, or people OUTSIDE
      the cab visible through the window or door

**HARD RULE on the verdict (no exceptions):**
  - If ``distinct_persons_visible_in_cabin <= 5`` → verdict MUST be FALSE_POSITIVE
  - If ``distinct_persons_visible_in_cabin >= 6`` AND no duplicate/reflection
    suspicion → verdict TRUE_POSITIVE
  - If ``distinct_persons_visible_in_cabin >= 6`` BUT you suspect duplicates,
    reflections, or wall posters → verdict UNCERTAIN

Count carefully and consistently. Do not say TRUE_POSITIVE if you only see 2 or 3
people; the rule above forbids it.

Common confounders the classical pipeline misclassifies as group:
  - The same person detected with two overlapping bboxes (over-counting)
  - Posters / safety photos on the cab walls counted as people
  - People visible through the cab window (on the platform) counted as inside
  - Mirror reflections of LP/ALP counted as additional people
  - Mannequins or training dummies (rare)

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
A 6+ person crowd in the cab while the train is RUNNING is a serious safety
issue; while STOPPED it can be normal during crew changeover or maintenance.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF; ≥0.80 when sure, ≤0.5 only when genuinely unsure>,
  "distinct_persons_visible_in_cabin": <integer count, 0 if cabin is empty>,
  "duplicate_detections_likely": <true|false>,
  "people_visible_only_through_window": <true|false>,
  "posters_or_photos_on_wall": <true|false>,
  "primary_confounder": "duplicate_bbox" | "wall_poster" | "window_view" | "reflection" | "none" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""


# Activity types that need FULL-FRAME context (rather than the hand+book ROI
# crop used for writing/eating/packing/cell_phone). Sleep needs body posture,
# mind_diversion needs head pose vs window, no_person checks cabin emptiness,
# group_detected counts distinct people — all benefit from the whole scene.
_FULL_FRAME_OBJECT_TYPES: frozenset = frozenset({
    "sleep",
    "mind_diversion",
    "no_person_detected",
    "group_detected",
})


# Activity description fragments → prompt key. The verifier matches on the
# activity's `objectType` field (writing/eating_drinking/packing_bags/cell_phone/
# sleep/mind_diversion/no_person_detected/group_detected) rather than on the
# numeric type code, so future numeric-code shuffles don't break this mapping.
_PROMPTS_BY_OBJECT_TYPE: Dict[str, str] = {
    "writing": _PROMPT_WRITING,
    "eating_drinking": _PROMPT_EATING,
    "packing_bags": _PROMPT_PACKING,
    "cell_phone": _PROMPT_CELL_PHONE,
    "sleep": _PROMPT_SLEEP,
    "mind_diversion": _PROMPT_MIND_DIVERSION,
    "no_person_detected": _PROMPT_NO_PERSON,
    "group_detected": _PROMPT_GROUP,
}


def _encode_image(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def _resolve_keyframes(activity: Dict[str, Any]) -> List[Path]:
    """Collect per-burst keyframe jpg paths from a (possibly grouped) activity.

    A grouped activity carries ``_sourceActivities`` with each burst's clip
    path; the corresponding ``*_activity.jpg`` lives alongside the clip.
    Falls back to the single ``activityImage`` field when no source-clip
    metadata is present.

    Returned in time order.
    """
    sources = activity.get("_sourceActivities") or []
    paths: List[Path] = []
    seen: set = set()
    for entry in sources:
        clip = entry.get("clip")
        if not clip:
            continue
        jpg = Path(str(clip).replace("_clip.mp4", "_activity.jpg"))
        if jpg.is_file() and str(jpg) not in seen:
            paths.append(jpg)
            seen.add(str(jpg))
    if not paths:
        single = activity.get("activityImage")
        if single:
            p = Path(str(single))
            if p.is_file():
                paths.append(p)
    return paths


def _detect_roi(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Compute a tight ROI containing all person + object bboxes that
    Pipeline-1 has rendered into the keyframe.

    Pipeline-1 draws each detected person bbox in bright GREEN and each
    object (book, cup, bottle, phone, bag, etc.) bbox in ORANGE/YELLOW.
    Returns the union of those rectangles as ``(x0, y0, x1, y1)`` so the
    VLM only sees the hand + book/cup region, not the whole cabin.

    Returns None when no qualifying bbox is found (caller should fall back
    to the full frame).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Bright green (Pipeline-1 person bbox): hue ~60, high S+V
    green = cv2.inRange(hsv, (45, 150, 150), (75, 255, 255))
    # Orange/yellow (Pipeline-1 object bbox): hue 15..35
    orange = cv2.inRange(hsv, (15, 150, 150), (35, 255, 255))
    mask = cv2.bitwise_or(green, orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rects: List[Tuple[int, int, int, int]] = []
    for c in contours:
        if cv2.contourArea(c) < 300:  # ignore noise (text, dots)
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        # Reject very thin/long rects (text labels, skeleton lines)
        if ww < 30 or hh < 30:
            continue
        rects.append((x, y, x + ww, y + hh))
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def _crop_to_roi(img: np.ndarray, padding: int = 30) -> np.ndarray:
    """Crop the keyframe to the Pipeline-1-bboxes union (with padding).

    Falls back to the original image when no usable ROI is detected.
    """
    h, w = img.shape[:2]
    roi = _detect_roi(img)
    if roi is None:
        return img
    x0, y0, x1, y1 = roi
    # Reject ROIs that already cover most of the image — no benefit cropping
    if (x1 - x0) >= 0.95 * w and (y1 - y0) >= 0.95 * h:
        return img
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return img[y0:y1, x0:x1]


def _stitch_keyframes(
    jpg_paths: List[Path],
    max_strip_width: int = 1500,
    crop_to_roi: bool = True,
) -> Optional[bytes]:
    """Stitch up to 5 keyframes left-to-right into a single labelled JPEG.

    When ``crop_to_roi=True`` (default), each frame is cropped to the
    Pipeline-1-rendered person + object bbox union before stitching, so
    the VLM sees only the hand + book/desk region at native pixel density.
    This is the right mode for fine-grained hand-on-object verdicts
    (writing, eating_drinking, packing_bags, cell_phone).

    When ``crop_to_roi=False``, frames are stitched at full resolution.
    This is the right mode for activities where the WHOLE cabin matters:
    sleep posture, head-pose for mind_diversion, "is the cabin empty"
    checks (no_person_detected), or counting people (group_detected).

    The combined strip is capped at ``max_strip_width`` either way to stay
    within vLLM's max-model-len budget. Each frame gets a ``FRAME N``
    tag in its top-left corner so the VLM can reference it via
    ``evidence_frame``. Single-frame input returns the original bytes
    unchanged (no relabel, no crop) regardless of mode.
    """
    if not jpg_paths:
        return None
    if len(jpg_paths) == 1:
        try:
            return jpg_paths[0].read_bytes()
        except OSError:
            return None

    cap = jpg_paths[:5]
    frames: List[np.ndarray] = []
    for idx, p in enumerate(cap):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if crop_to_roi:
            img = _crop_to_roi(img)
        cv2.rectangle(img, (0, 0), (110, 30), (0, 0, 0), -1)
        cv2.putText(img, f"FRAME {idx + 1}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(img)
    if not frames:
        return None

    # Pad each frame to a common height so np.hstack works.
    target_h = max(img.shape[0] for img in frames)
    padded: List[np.ndarray] = []
    for img in frames:
        h = img.shape[0]
        if h < target_h:
            img = cv2.copyMakeBorder(img, 0, target_h - h, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(img)

    strip = np.hstack(padded)

    # If the combined width exceeds the budget, downscale uniformly.
    if strip.shape[1] > max_strip_width:
        scale = max_strip_width / float(strip.shape[1])
        new_h = max(1, int(strip.shape[0] * scale))
        strip = cv2.resize(strip, (max_strip_width, new_h),
                           interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    return buf.tobytes()


def _parse_verdict(raw_text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction. Tolerates ``code``-fenced output."""
    t = (raw_text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end < 0:
        return {"parse_error": "no_json_object", "raw_text": raw_text[:300]}
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"parse_error": f"json_decode: {exc}", "raw_text": raw_text[:300]}


class VlmVerificationService:
    """Stateless verifier that talks to a vLLM endpoint over HTTP.

    The class itself is lightweight (no model loaded in-process); each call
    POSTs to ``{vlm_base_url}/chat/completions`` with one inline-encoded image.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        # Cache the verify-list as a frozen set for fast membership checks.
        self._verify_set: frozenset[str] = frozenset(
            x.strip() for x in self.settings.vlm_verify_activities.split(",") if x.strip()
        )
        logger.info(
            "[vlm] VlmVerificationService init enabled=%s shadow=%s endpoint=%s "
            "model=%s verify=%s drop_threshold=%.2f timeout=%.1fs",
            self.settings.vlm_verification_enabled,
            self.settings.vlm_shadow_mode,
            self.settings.vlm_base_url,
            self.settings.vlm_model,
            sorted(self._verify_set),
            self.settings.vlm_drop_threshold,
            self.settings.vlm_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return bool(self.settings.vlm_verification_enabled)

    def verify_activities(
        self, activities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Verify a batch of activities and return the post-filter list.

        Each input activity dict is mutated in place to add a ``vlm_review``
        sub-dict with the verdict, latency, and (in enforcement mode) a
        ``dropped`` flag.

        Returns:
            Tuple of (kept_activities, stats_dict). ``stats_dict`` keys:
            verified, skipped_type, skipped_stopped, skipped_unavailable,
            skipped_no_image, dropped, kept, uncertain, parse_errors.
        """
        stats = {
            "verified": 0,
            "skipped_type": 0,
            "skipped_stopped": 0,
            "skipped_unavailable": 0,
            "skipped_no_image": 0,
            "dropped": 0,
            "kept": 0,
            "uncertain": 0,
            "parse_errors": 0,
        }
        if not self.is_enabled() or not activities:
            return activities, stats

        kept: List[Dict[str, Any]] = []
        verified_count = 0
        cap = int(self.settings.vlm_max_activities_per_run or 0)

        for act in activities:
            # Normalize: writers use "packing bags" (space) but the registry
            # key is "packing_bags" (underscore). Match either form.
            object_type = (act.get("objectType") or "").strip().lower().replace(" ", "_")
            if object_type not in self._verify_set or object_type not in _PROMPTS_BY_OBJECT_TYPE:
                stats["skipped_type"] += 1
                kept.append(act)
                continue

            # Mirror the downstream STOPPED filter in video_controller.py: any
            # activity with motionState=STOPPED gets dropped before the
            # external API post anyway, so spending a VLM call on it is wasted
            # compute. Today's verify list (writing/eating/packing) is also
            # suppressed by gates.apply_train_stopped_suppression upstream, so
            # this branch should rarely fire — but it's a cheap guard for the
            # day cell_phone or microsleep enter the verify list (those are
            # NOT suppressed when STOPPED, see gates.py:14-15).
            if (act.get("motionState") or "").strip().upper() == "STOPPED":
                stats["skipped_stopped"] += 1
                act["vlm_review"] = {
                    "status": "SKIPPED_STOPPED",
                    "verdict": None,
                    "reason": "motionState=STOPPED; downstream filter would drop anyway",
                }
                kept.append(act)
                continue

            if cap and verified_count >= cap:
                stats["skipped_type"] += 1
                kept.append(act)
                continue

            prompt = _PROMPTS_BY_OBJECT_TYPE[object_type]
            review = self._verify_one(act, prompt, object_type)
            verified_count += 1
            act["vlm_review"] = review

            status = review.get("status")
            if status == "SKIPPED_VLM_UNAVAILABLE":
                stats["skipped_unavailable"] += 1
                kept.append(act)
                continue
            if status == "SKIPPED_NO_IMAGE":
                stats["skipped_no_image"] += 1
                kept.append(act)
                continue
            if status == "PARSE_ERROR":
                stats["parse_errors"] += 1
                kept.append(act)
                continue

            stats["verified"] += 1
            verdict = (review.get("verdict") or {}).get("verdict")
            confidence = (review.get("verdict") or {}).get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            if verdict == "UNCERTAIN":
                stats["uncertain"] += 1

            should_drop = (
                not self.settings.vlm_shadow_mode
                and verdict == "FALSE_POSITIVE"
                and confidence >= self.settings.vlm_drop_threshold
            )
            if should_drop:
                act["vlm_review"]["dropped"] = True
                stats["dropped"] += 1
                logger.info(
                    "[vlm] DROPPED activity type=%s desc=%r at t=%s "
                    "verdict=FALSE_POSITIVE conf=%.2f reason=%r",
                    act.get("activityType"),
                    act.get("des", "")[:60],
                    act.get("activityStartTime"),
                    confidence,
                    ((review.get("verdict") or {}).get("reasoning") or "")[:120],
                )
                # Skip appending — activity is dropped
                continue

            stats["kept"] += 1
            kept.append(act)

        logger.info(
            "[vlm] verification stats: verified=%d kept=%d dropped=%d uncertain=%d "
            "skipped_type=%d skipped_stopped=%d skipped_unavailable=%d "
            "parse_errors=%d shadow=%s",
            stats["verified"],
            stats["kept"] + stats["skipped_type"] + stats["skipped_stopped"]
            + stats["skipped_unavailable"] + stats["skipped_no_image"]
            + stats["parse_errors"],
            stats["dropped"],
            stats["uncertain"],
            stats["skipped_type"],
            stats["skipped_stopped"],
            stats["skipped_unavailable"],
            stats["parse_errors"],
            self.settings.vlm_shadow_mode,
        )
        return kept, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _verify_one(self, activity: Dict[str, Any], prompt: str,
                    object_type: str = "") -> Dict[str, Any]:
        # Collect all per-burst keyframes for grouped activities; falls back
        # to the single activityImage when no source-clip metadata exists.
        keyframes = _resolve_keyframes(activity)
        if not keyframes:
            return {"status": "SKIPPED_NO_IMAGE", "verdict": None,
                    "image_path": activity.get("activityImage") or ""}

        # Sleep / mind_diversion / no_person / group_detected need full-frame
        # context (body posture, head pose vs window, cabin emptiness, head-
        # count). The hand+book ROI crop used by writing/eating/packing/cell
        # would discard the relevant scene context for those.
        crop_to_roi = object_type not in _FULL_FRAME_OBJECT_TYPES

        # Stitch into a labelled multi-frame strip when more than one
        # keyframe is available; otherwise just send the single image.
        strip_bytes = _stitch_keyframes(keyframes, crop_to_roi=crop_to_roi)
        if not strip_bytes:
            return {"status": "SKIPPED_NO_IMAGE", "verdict": None,
                    "image_path": str(keyframes[0])}
        b64 = base64.b64encode(strip_bytes).decode("ascii")
        n_frames = len(keyframes)

        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 400,
            "temperature": 0.0,
        }

        url = f"{self.settings.vlm_base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(
                req, timeout=self.settings.vlm_timeout_seconds
            ) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            logger.warning(
                "[vlm] endpoint unavailable for activity type=%s at t=%s: %s",
                activity.get("activityType"),
                activity.get("activityStartTime"),
                exc,
            )
            return {
                "status": "SKIPPED_VLM_UNAVAILABLE",
                "verdict": None,
                "error": str(exc),
                "latency_sec": round(time.time() - t0, 3),
            }

        latency = round(time.time() - t0, 3)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {
                "status": "PARSE_ERROR",
                "verdict": None,
                "error": "no_choices_in_response",
                "latency_sec": latency,
            }

        parsed = _parse_verdict(text)
        if "parse_error" in parsed:
            return {
                "status": "PARSE_ERROR",
                "verdict": parsed,
                "latency_sec": latency,
            }

        return {
            "status": "OK",
            "verdict": parsed,
            "latency_sec": latency,
            "model": self.settings.vlm_model,
            "frames_sent": n_frames,
        }


# Singleton wiring (mirrors external_api_service pattern)
_vlm_service: Optional[VlmVerificationService] = None
_vlm_service_lock = threading.Lock()


def get_vlm_verification_service() -> VlmVerificationService:
    """Get the singleton VlmVerificationService instance (thread-safe)."""
    global _vlm_service
    if _vlm_service is None:
        with _vlm_service_lock:
            if _vlm_service is None:
                _vlm_service = VlmVerificationService()
    return _vlm_service


__all__ = ["VlmVerificationService", "get_vlm_verification_service"]
