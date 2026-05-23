"""Verdict parser, calibrator, telemetry writers, and motion-state logic.

Owns the structured-output side of the VLM verifier: per-activity prompt
strings (which DEFINE the JSON contract), the JSON-extraction parser,
the confidence calibrator (isotonic + temperature mapping), the
``_consistency_check`` post-VLM contradiction detector, the motion
override regexes / ``_safe_motion_state`` helper, and the JSONL
telemetry writers.

All functions copied verbatim from the original monolithic file —
behaviour is unchanged.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, Optional

from ...utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Calibrator — Wave-2 scaffolding for confidence calibration.
#
# Production CV systems should NEVER trust raw model output as if it were
# a probability. Quantised VLMs (AWQ in our case) systematically drift
# from calibration: 0.95 raw confidence does not mean 95% accuracy. Once
# we have labelled ground truth we fit a temperature-scaling or isotonic
# mapping; until then this class is a deliberate no-op that lets the
# verifier code path be unchanged when calibration data lands.
#
# The mapping file format (``vlm_calibration_path``) is intentionally
# simple JSON so it can be regenerated from a notebook without code
# changes:
#
#   {"method": "temperature", "temperature": 1.4}
#       — calibrated_p = sigmoid(logit(raw) / T)
#
#   {"method": "isotonic", "x": [0.0, 0.3, 0.6, 0.95], "y": [0.0, 0.1, 0.4, 0.85]}
#       — piecewise-linear interpolation over (x, y) breakpoints
#
#   {"method": "identity"}            — pass-through (default when missing)
#
# Per-activity calibrators ("by_object_type": {"writing": {...}}) override
# the global mapping for that activity name.
# ---------------------------------------------------------------------------
class _Calibrator:
    """Lazy-loaded confidence calibrator. Falls back to identity on any
    error so a missing / malformed file never breaks the verifier."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._loaded = False
        self._global: Dict[str, Any] = {"method": "identity"}
        self._by_type: Dict[str, Dict[str, Any]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self._path, "r") as f:
                data = json.load(f) or {}
        except (OSError, json.JSONDecodeError) as e:
            logger.info(
                "[vlm] calibration file unavailable at %s (%s); using identity",
                self._path, type(e).__name__,
            )
            return
        if not isinstance(data, dict):
            return
        if "by_object_type" in data and isinstance(data["by_object_type"], dict):
            self._by_type = {
                str(k): v for k, v in data["by_object_type"].items()
                if isinstance(v, dict)
            }
        global_cfg = {k: v for k, v in data.items() if k != "by_object_type"}
        if global_cfg:
            self._global = global_cfg
        logger.info(
            "[vlm] calibration loaded from %s: global=%s per_type_keys=%s",
            self._path, self._global.get("method", "identity"),
            list(self._by_type.keys()),
        )

    def calibrate(self, raw_conf: float, object_type: str) -> float:
        """Map raw confidence to calibrated probability, in [0, 1]."""
        self._load()
        cfg = self._by_type.get(object_type) or self._global
        method = (cfg.get("method") or "identity").lower()
        try:
            r = float(raw_conf)
        except (TypeError, ValueError):
            return 0.0
        r = max(0.0, min(1.0, r))
        if method == "identity":
            return r
        if method == "temperature":
            t = float(cfg.get("temperature", 1.0)) or 1.0
            # Stable logit/sigmoid; clamp r to avoid log(0).
            r_c = min(0.9999, max(1e-4, r))
            import math
            logit = math.log(r_c / (1.0 - r_c))
            return 1.0 / (1.0 + math.exp(-logit / t))
        if method == "isotonic":
            xs = cfg.get("x") or []
            ys = cfg.get("y") or []
            if len(xs) >= 2 and len(xs) == len(ys):
                # Piecewise-linear interp; relies on xs being sorted.
                if r <= xs[0]:
                    return float(ys[0])
                if r >= xs[-1]:
                    return float(ys[-1])
                for i in range(1, len(xs)):
                    if r <= xs[i]:
                        x0, x1 = float(xs[i - 1]), float(xs[i])
                        y0, y1 = float(ys[i - 1]), float(ys[i])
                        if x1 == x0:
                            return y1
                        return y0 + (y1 - y0) * (r - x0) / (x1 - x0)
        return r


# Process-level lazy singleton — one calibrator file per gunicorn worker
# is plenty (and avoids reload cost per call).
_calibrator: Optional[_Calibrator] = None
_calibrator_lock = threading.Lock()


def _get_calibrator(path: str) -> _Calibrator:
    global _calibrator
    if _calibrator is None or _calibrator._path != path:
        with _calibrator_lock:
            if _calibrator is None or _calibrator._path != path:
                _calibrator = _Calibrator(path)
    return _calibrator


# ---------------------------------------------------------------------------
# Disagreement and telemetry JSONL writers.
#
# Both are thread-safe append loggers gated by feature flags. They use a
# single lock per file path so concurrent writers never produce
# interleaved partial lines (jsonl readers expect one complete object
# per line).
# ---------------------------------------------------------------------------
_jsonl_locks: Dict[str, threading.Lock] = {}
_jsonl_locks_guard = threading.Lock()


def _jsonl_lock_for(path: str) -> threading.Lock:
    with _jsonl_locks_guard:
        lock = _jsonl_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _jsonl_locks[path] = lock
        return lock


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Best-effort JSONL append. Failures are logged at debug only — the
    verifier's correctness must never depend on telemetry write success."""
    try:
        from os import makedirs
        from os.path import dirname

        d = dirname(path)
        if d:
            makedirs(d, exist_ok=True)
        line = json.dumps(record, default=str)
        with _jsonl_lock_for(path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # noqa: BLE001
        logger.debug("[vlm] jsonl append failed for %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Verdict enum + parser-error sentinel
# ---------------------------------------------------------------------------
_ALLOWED_VERDICTS: frozenset = frozenset({"TRUE_POSITIVE", "FALSE_POSITIVE", "UNCERTAIN"})


_PROMPT_WRITING = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. The frame may contain up to 2 persons:
- Loco Pilot (LP): the larger, foreground person, usually seated at the controls
- Assistant Loco Pilot (ALP): the second person, often standing or in the back seat

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME activity
burst, ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames before deciding.

A classical CV pipeline flagged this strip as "WRITING IN LOG BOOK". The pipeline
fires on coarse hand-near-book proximity and is known to mislabel idle/gesturing
hands as writing. The pipeline does NOT specify which person — either LP or ALP
could be the writer. Your job is to either CONFIRM real writing or REJECT the
flag. Do not assume the pipeline is right; the prior here is roughly 50/50 in
the production stream.

The strip is typically CROPPED to the hand + book / desk region for pixel
density. Some peripheral cabin context (a sliver of door, window, or floor) is
usually still visible at the edges — use whatever you can see; do NOT invent
content you cannot see.

================ STEP 1: PER-FRAME OBSERVATION (mental, not emitted) ================
Mentally examine EACH frame of the strip in turn — for each frame, identify
the person, what their hand is doing, what (if anything) is in their hand,
and where their head is pointing. Do this BEFORE committing to a verdict.

The structured boolean fields you DO emit (``hand_actually_on_book``,
``pen_in_hand``, ``actively_handling_papers``, ``head_oriented_to_book``,
etc.) summarise these observations across the strip. Each boolean must be
true ONLY IF the condition was confirmed in at least one frame for the
SAME person. Outside-view cues go in the ``motion_evidence`` string.

================ STEP 2: WRITING VERDICT ================
There are TWO valid TP paths. Either path independently confirms writing for
the SAME person. If NEITHER path is satisfied, the verdict is FALSE_POSITIVE.

----- PATH A (open-page writing, strict) -----
ALL FOUR conditions visibly confirmed for the SAME person in at least one frame:
  (A.1) An OPEN book / notepad / paper is visible flat on the desk
  (A.2) That person's hand is in PHYSICAL CONTACT with the open page —
        `hand_contact = "touching_page"` in at least one frame.
  (A.3) EITHER a pen/pencil is clearly visible in that hand OR a bare
        fingertip is unambiguously pressed onto the page surface.
  (A.4) Head is tilted DOWN toward the open page (`head = "down_to_paper"`).

----- PATH B (railway logbook handling, holistic) -----
Indian railway logbook entries are commonly made while HOLDING the bound
logbook or its folded sheets in the hand (not laid flat on the desk). From
the overhead camera, "touching the page" is hard to distinguish from
"holding the page edge" — the salient cue is the PEN IN HAND. ALL of the
following must hold across the strip (different conditions can be confirmed
in different frames, but all must apply to the SAME person):
  (B.1) A pen / pencil is CLEARLY VISIBLE in the person's hand in at least
        one frame (`object_in_hand = "pen"` or `"pencil"`). This is the
        load-bearing requirement — without a pen this path does not apply.
  (B.2) The person is ACTIVELY HANDLING papers / a folded sheet / a logbook
        — gripping, turning, writing-on, or steadying — across at least
        one frame. Static papers untouched on the desk do NOT count.
  (B.3) Head tilted DOWN toward the papers in at least one frame
        (`head = "down_to_paper"`).
  (B.4) The pen-bearing hand is NOT on a control lever / brake handle /
        throttle / radio handset. Verify the hand is on or over PAPER.

If neither Path A nor Path B is satisfied, return FALSE_POSITIVE with
confidence ≥ 0.8 — the pipeline misfired. Do NOT default to TRUE_POSITIVE
just because hands and a book share the frame.

Common confounders that are NOT writing (return FALSE_POSITIVE):
  - Hands in the lap with head bent down and NO pen visible (idle/tired posture)
  - A book or papers sitting unattended on the desk while crew operates controls
  - One person reaches/gestures toward papers on the desk while a DIFFERENT person
    actually has the book — the proximity-fired bbox can latch onto the wrong
    person; pick the writer correctly via `which_person`, or return FALSE_POSITIVE
    if neither is genuinely writing
  - Holding a folded paper or clipboard at chest level while looking forward
    AND no pen visible (papers without a pen is "reviewing", not "writing")
  - Holding the brake handle / throttle lever — verify the hand is on PAPER,
    not on a metal control
  - Holding a railway radio handset (brick-shaped, often with coiled cord)
  - Hand extended toward the desk but not in contact AND no pen visible

================ STEP 3: TRAIN MOTION OBSERVATION ================
Independently observe whether the train is moving. Use cues OUTSIDE the cabin
(door, window, distant scenery). The strip is multi-frame, so cross-frame
comparison is your strongest tool.

  RUNNING cues (any one is sufficient, must be FRAME-SPECIFIC):
    - Visible motion blur of trees/poles/track in the outside window of a
      specific frame (cite the frame number)
    - Scenery clearly displaced between consecutive frames (e.g. a pole at the
      right edge of FRAME 1 has moved off-screen by FRAME 2)
    - Track ballast streaking past

  STOPPED cues (any one is sufficient):
    - The cabin door is OPEN (trains do not run with the door open)
    - A platform edge, platform tiles, or station building is visible outside
    - A station sign or station name board is visible
    - A person standing or walking outside the cab (people don't walk next to a
      moving train)
    - The SAME background structures (bars, grilles, walls, signs, parked
      objects) appear at the SAME screen position across 2+ frames in the strip
      — this is conclusive evidence of a stationary train and is just as strong
      as motion blur is for RUNNING

  UNCLEAR:
    - Window is dark, blocked, glare, or you cannot see enough outside

ANTI-HALLUCINATION RULES for motion:
  1. Do NOT claim "motion blur in window" unless you can name the SPECIFIC
     frame where you actually see streaking/blur. If you cannot, the correct
     answer is "unclear" or "stopped" — never default to "running".
  2. If the strip has 2+ frames and the visible outside content is identical
     across them, the train is STOPPED. State this as
     `motion_evidence: "static scenery across frames; <what you see, e.g.
     platform visible / open door / station building visible>"`. The
     downstream system parses your `motion_evidence` string for the tokens
     "open door", "platform", "station" — use those exact tokens when you
     see them.
  3. The "writing while running" rule does not apply if the train is stopped.
     Your verdict on the writing action is INDEPENDENT of motion, but motion
     is an authoritative observation that the rest of the system relies on
     to drop or keep this violation. Lying about motion drops a real
     violation OR keeps a false one — both are bad.

================ OUTPUT ================
Reply with STRICT JSON ONLY (no prose, no code fence). Emit exactly these
keys, in this order:
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0. Set ≥ 0.8 when sure (either direction).
                 ≤ 0.5 only for genuine ambiguity. Confidence is your
                 certainty in the VERDICT, not in the absence of activity>,
  "book_visible_on_desk": <true|false — Path A signal: open page on the desk>,
  "book_is_open": <true|false>,
  "hand_actually_on_book": <true|false — Path A signal: writer's hand in
                             physical contact with the open page in at least
                             one frame; "hovering near" is NOT contact>,
  "head_oriented_to_book": <true|false>,
  "pen_in_hand": <true|false — Path B signal (also satisfies Path A.3):
                   a pen / pencil is clearly visible held by the writer's
                   fingers in at least one frame. Load-bearing for Path B>,
  "actively_handling_papers": <true|false — Path B signal: writer is
                                gripping / turning / writing-on / steadying
                                papers or a folded logbook IN HAND (not
                                static-on-desk) in at least one frame>,
  "tp_path": "A" | "B" | "neither" | "unclear",
  "primary_object_in_hand": "pen" | "brake_or_throttle" | "radio_handset" | "phone" | "nothing_visible" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string citing SPECIFIC visual cue + frame number, e.g. 'FRAME 3: open door, person on platform', 'static scenery across frames; platform visible at right', 'FRAME 2: motion blur in right window'. Use 'open door' / 'platform' / 'station' verbatim when applicable>",
  "evidence_frame": <integer 1..N for strongest writing evidence; 0 if no decisive frame>,
  "reasoning": "<one short sentence: what you see, naming the FRAME number>"
}"""

_PROMPT_EATING = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
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

A classical CV pipeline flagged this frame as "EATING OR DRINKING". The pipeline does
NOT specify which person — either LP or ALP could be the one eating/drinking. Your job
is to verify whether ANYONE in the frame is actually eating or drinking RIGHT NOW.

True eating/drinking requires ALL of these to hold for the SAME person:
  (a) A clearly identifiable food item, cup, bottle, or similar consumable is in their hand
  (b) That object is being moved toward the mouth OR is held at/near the lips
  (c) The hand is not on a control/lever/handset/radio

If the object is a radio handset (brick-shaped, often with a coiled cord), it is NOT
eating/drinking — return FALSE_POSITIVE regardless of hand position.

Common confounders the classical pipeline misclassifies as eating/drinking:
  - A cup or bottle resting on the desk while no one is touching it (static-object FP)
  - Person wiping face, scratching nose, adjusting cap, or yawning with hand near face
  - Hand at face but no object visible in it (idle gesture)
  - Holding a radio handset to face/ear — ONLY claim "radio_handset" if you can
    clearly see the brick shape or coiled cord; do not invent a radio.

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
This is a separate observation; do NOT use motion alone to decide the eating verdict.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF. If verdict=FALSE_POSITIVE and you are sure no activity is happening, set confidence ≥ 0.80. If verdict=TRUE_POSITIVE and you are sure the activity is happening, set confidence ≥ 0.80. Use ≤ 0.5 only when you are genuinely unsure. Do NOT set 0.0 to mean "no activity"; that means "I am 0% sure of my verdict">,
  "primary_object_in_hand": "cup" | "bottle" | "food" | "radio_handset" | "phone" | "nothing_visible" | "unclear",
  "object_at_mouth": <true|false>,
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""

_PROMPT_PACKING = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
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

A classical CV pipeline flagged this frame as "PACKING BAGS". The pipeline does NOT
specify which person — either LP or ALP could be the one packing. Your job is to
verify whether ANYONE in the frame is actually packing/handling a bag RIGHT NOW.

True packing requires ALL of these to hold for the SAME person:
  (a) A bag, backpack, or suitcase is visible and clearly identifiable
  (b) That person's hand is INSIDE the bag opening, or actively gripping/lifting it
  (c) Body posture is clearly oriented toward the bag (not toward controls or window)

If the bag is just sitting on the floor or seat with no one touching it, return
FALSE_POSITIVE — a bag in the frame is not the same as packing.

Common confounders the classical pipeline misclassifies as packing:
  - A bag/suitcase visible on the floor or seat but no one is interacting with it
  - LP reaching for controls and a bag happens to be in the same image region
  - Crew member standing near a bag during a station stop (handover, not packing)
  - A piece of equipment that resembles a bag (cushion, jacket, kit) — only claim
    "bag" if you can clearly see a backpack/suitcase/duffel shape.

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
This is a separate observation; do NOT use motion alone to decide the packing verdict.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF. If verdict=FALSE_POSITIVE and you are sure no activity is happening, set confidence ≥ 0.80. If verdict=TRUE_POSITIVE and you are sure the activity is happening, set confidence ≥ 0.80. Use ≤ 0.5 only when you are genuinely unsure. Do NOT set 0.0 to mean "no activity"; that means "I am 0% sure of my verdict">,
  "bag_visible": <true|false>,
  "hand_in_or_on_bag": <true|false>,
  "posture_oriented_to_bag": <true|false>,
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
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


_PROMPT_SOLO_PERSON = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
Camera angle: overhead, looking down/across the cab. The cabin must always carry
TWO crew members while the train is running: the Loco Pilot (LP) at the controls
and the Assistant Loco Pilot (ALP), typically standing or seated nearby. Exactly
ONE person in a running cab is the violation we are verifying.

The image you receive may be a HORIZONTAL STRIP of 1-5 keyframes from the SAME burst,
ordered LEFT-TO-RIGHT in time. Each frame is labelled "FRAME 1", "FRAME 2", ...
in the top-left corner. Examine ALL frames before deciding — solo_person is a
SUSTAINED state, not a single-frame loss of detection. If a second person appears
in ANY frame of the strip, the cabin is NOT a solo-occupied cabin and you should
return FALSE_POSITIVE. Note in `evidence_frame` which frame number provides the
strongest evidence (1-N). If only one frame is shown or no frame is decisive,
return 0.

A classical CV pipeline flagged this strip as "SOLO PERSON" — exactly one person
in the cab. The pipeline OVER-FIRES on this — when YOLO loses recall on the
second crew member (occlusion, pose, dark frames, dedup IoU collapsing two
overlapping bboxes into one) the count drops to 1 even though both crew members
are physically present. **Do NOT trust the classical flag. Count the real people
yourself across the strip.**

True solo_person requires ALL of:
  (a) Exactly ONE distinct, real person physically present INSIDE the cabin
      in EVERY frame of the strip — never two
  (b) The single person is the LP at the controls (not an unrelated supervisor
      or worker; if the LP is missing too, this is no_person_detected, not
      solo_person)
  (c) No partial body parts (legs, arm, head) of a SECOND person are peeking
      from frame edges, behind the seat-back, or below the dashboard line in
      any frame

If you can see TWO OR MORE people — even if one is partial / occluded / barely
visible — in any frame, return FALSE_POSITIVE.

**ACTIVELY look for partial bodies before deciding** — production analysis of
this rule has shown the most common failure mode is missing a partially-visible
second person. Before declaring solo_person:
  1. Scan the UPPER edge of every frame for a clipped torso, shoulder, head,
     or arm protruding into frame.
  2. Scan THROUGH the open cabin doorway / passage for a second body at the
     back of the cab, even if only legs / a uniform sleeve / a back-of-head
     silhouette is visible.
  3. Look BEHIND the foreground person (the LP at the controls) for a second
     body partially occluded by the LP's body, the seat-back, or the
     dashboard.
  4. Look AT THE FRAME EDGES (left, right, bottom) for a leg, foot, or sleeve
     of a second person who is mostly out-of-frame.
  5. **HAND/ARM AT EDGE RULE (hard):** if you see a disembodied hand,
     forearm, sleeve cuff, finger, or hand-held object (papers, pen, radio,
     cup) entering the frame at any edge or hovering near the control panel
     WITHOUT a visible torso/head attached to it, that hand belongs to a
     SECOND person whose body is outside the camera's field of view. Set
     ``second_person_partial_or_occluded=true``. Do not require the second
     person's full body to be visible — overhead cabin cameras frequently
     clip the LP's torso when the LP leans into the controls, leaving only a
     reaching arm in frame.
If you spot ANY of the above — even with low confidence — set
``second_person_partial_or_occluded=true`` and return UNCERTAIN (not
TRUE_POSITIVE). Reserve TRUE_POSITIVE for cases where you are highly confident
the cabin contains exactly one person and there is no edge / occlusion
ambiguity anywhere in the strip.

Common confounders the classical pipeline misclassifies as solo_person:
  - ALP bent over below the seat-back / dashboard line (occluded by furniture)
  - ALP standing close to the camera with only legs/feet in the lower frame edge
  - ALP turned away from camera with only the back-of-head silhouette visible
  - **ALP standing at the BACK of the cab — partial torso / shoulder / arm
    visible at the upper frame edge or through an open doorway** (the most
    common FP archetype on this camera install)
  - LP and ALP standing close together — their bboxes overlap and the
    de-duplicator collapses them into a single detection
  - Heavy backlight from the windshield washing out a person who is actually there
  - IR / dark frames where the second person's contrast is too low for YOLO
  - A jacket / uniform on the seat-back mistaken for a person being absent

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
solo_person is ONLY a violation while the train is RUNNING. If the train is
stopped at a station, the ALP may legitimately have stepped out for an
inspection / point check / platform interaction, and a single-occupant cabin
is expected behaviour — NOT a violation. Be especially careful here: when the
keyframe strip is small or low-resolution, "motion blur in window" can be a
hallucination — only report ``running`` when you can name a SPECIFIC
running-cue (poles flashing past, track streaking, scenery clearly translating
across consecutive frames). When in doubt, return ``unclear``.

**DOOR-OPEN PRECEDENCE RULE (hard, no exceptions):** Trains do NOT run with a
cabin door open. If you see ANY of the following in ANY frame of the strip,
the train is STOPPED — set ``train_appears_to_be="stopped"`` REGARDLESS of
what you think you see in the side window:
  - Cabin door visibly ajar / open / partially open
  - Daylight or outdoor brightness flooding through a doorway
  - A person standing or walking on a platform / on the ground / on track
    ballast visible through an open doorway
  - Feet, legs, or a partial body of someone outside the cabin visible at
    floor level through a doorway
  - Platform edge, station signage, ballast, or rails visible at close range
    through an open door
Window-blur observations are UNRELIABLE when a door is open in the same strip
— low-res JPEG compression and the camera's own micro-vibration regularly
produce apparent "motion blur" in side windows even at standstill. The door
state is the authoritative cue. If door state and window state disagree, the
door wins.

**HARD RULES on the verdict (no exceptions):**
  - If ``distinct_persons_visible_in_strip >= 2`` in any frame → FALSE_POSITIVE
  - If ``train_appears_to_be != "running"`` (i.e. stopped or unclear) →
    UNCERTAIN, regardless of person count. Do NOT return TRUE_POSITIVE for a
    one-person cabin unless you can confirm the train is genuinely running
    via specific outside-the-cabin cues. This rule overrides the person-count
    rule below.
  - If ``distinct_persons_visible_in_strip == 1`` consistently across the strip
    AND the LP is the visible person AND the train is confirmed RUNNING →
    TRUE_POSITIVE
  - If ``distinct_persons_visible_in_strip == 1`` BUT you suspect a second person
    is occluded / edge-clipped / merged-by-dedup → UNCERTAIN

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF; ≥0.80 when sure, ≤0.5 only when genuinely unsure>,
  "distinct_persons_visible_in_strip": <integer: max distinct persons seen in any single frame, 0 if cabin is empty>,
  "lp_visible": <true|false>,
  "alp_visible": <true|false>,
  "second_person_partial_or_occluded": <true|false>,
  "primary_confounder": "occluded_by_seat" | "edge_clipped" | "dedup_merge" | "backlight" | "ir_dark_frame" | "jacket_on_seat" | "none" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
}"""


# Activity description fragments → prompt key. The verifier matches on the
# activity's `objectType` field (writing/eating_drinking/packing_bags/cell_phone/
# sleep/mind_diversion/no_person_detected/group_detected/solo_person) rather
# than on the numeric type code, so future numeric-code shuffles don't break
# this mapping.
_PROMPTS_BY_OBJECT_TYPE: Dict[str, str] = {
    "writing": _PROMPT_WRITING,
    "eating_drinking": _PROMPT_EATING,
    "packing_bags": _PROMPT_PACKING,
    "cell_phone": _PROMPT_CELL_PHONE,
    "sleep": _PROMPT_SLEEP,
    "mind_diversion": _PROMPT_MIND_DIVERSION,
    "no_person_detected": _PROMPT_NO_PERSON,
    "group_detected": _PROMPT_GROUP,
    "solo_person": _PROMPT_SOLO_PERSON,
}


# Per-activity contradiction rules consulted after the VLM responds. When
# the VLM emits ``TRUE_POSITIVE`` but the structured fields it filled in
# contradict the verdict (e.g. ``hand_actually_on_book=false`` for a
# writing TP), the wrapper demotes the verdict to ``UNCERTAIN`` and caps
# the confidence. This is the structural counterpart to the prompt-level
# "do not default to TRUE_POSITIVE" rule: the prompt asks the model to
# be honest, this enforces it.
def _consistency_check(
    parsed: Dict[str, Any], object_type: str
) -> Optional[str]:
    """Return a reason string if the parsed VLM verdict is internally
    inconsistent with its structured observation fields, else ``None``.

    Only fires when verdict is ``TRUE_POSITIVE`` — false-positive verdicts
    pass through unchanged because the consistency rules are designed to
    catch cooperatively-filled-but-wrong-label hallucinations, not to
    rescue genuine FPs.
    """
    if parsed.get("verdict") != "TRUE_POSITIVE":
        return None

    if object_type == "writing":
        # Two independent TP paths. Accept TP if EITHER path's structural
        # fields are coherent. Demote to UNCERTAIN only when BOTH paths
        # contradict the verdict.
        #
        # Path A (open-page writing): hand_actually_on_book + book_visible_on_desk.
        # Path B (railway-logbook handling): pen_in_hand + actively_handling_papers.
        path_a_ok = bool(
            parsed.get("hand_actually_on_book")
            and parsed.get("book_visible_on_desk")
        )
        path_b_ok = bool(
            parsed.get("pen_in_hand")
            and parsed.get("actively_handling_papers")
        )
        if not (path_a_ok or path_b_ok):
            # Compose a precise reason naming whichever paths actually failed,
            # so disagreement-queue audits can identify prompt-level blind
            # spots vs. genuine VLM hallucinations.
            reasons = []
            if not parsed.get("hand_actually_on_book"):
                reasons.append("hand_actually_on_book=false")
            if not parsed.get("book_visible_on_desk"):
                reasons.append("book_visible_on_desk=false")
            if not parsed.get("pen_in_hand"):
                reasons.append("pen_in_hand=false")
            if not parsed.get("actively_handling_papers"):
                reasons.append("actively_handling_papers=false")
            return (
                "TP claimed but neither Path A nor Path B holds: "
                + ", ".join(reasons)
            )

        # frame_observations was removed from the writing schema in the
        # 2026-05-09 max_tokens-truncation fix; the per-frame array was the
        # only thing that ever truncated and forced parse_error fallbacks.
        # The remaining structured booleans (hand_actually_on_book,
        # pen_in_hand, actively_handling_papers, head_oriented_to_book)
        # already carry the same per-frame signal as aggregates, and Path A
        # / Path B above are sufficient to catch cooperatively-filled
        # contradictory outputs without the array.
        return None

    if object_type == "eating_drinking":
        obj = (parsed.get("primary_object_in_hand") or "").lower()
        if obj in ("nothing_visible", "unclear", "radio_handset", "phone"):
            return f"TP claimed but primary_object_in_hand={obj!r}"
        if not parsed.get("object_at_mouth"):
            return "TP claimed but object_at_mouth=false"
        return None

    if object_type == "packing_bags":
        if not parsed.get("bag_visible"):
            return "TP claimed but bag_visible=false"
        if not parsed.get("hand_in_or_on_bag"):
            return "TP claimed but hand_in_or_on_bag=false"
        return None

    if object_type == "cell_phone":
        obj = (parsed.get("object_in_hand") or "").lower()
        if obj != "smartphone":
            return f"TP claimed but object_in_hand={obj!r} (not smartphone)"
        return None

    if object_type == "sleep":
        criteria_met = sum(
            bool(parsed.get(k))
            for k in (
                "eyes_closed",
                "body_reclined",
                "hands_still",
                "head_motionless_across_frames",
            )
        )
        if criteria_met < 2:
            return f"TP claimed but only {criteria_met}/4 sleep criteria met"
        return None

    if object_type == "mind_diversion":
        head_dir = (parsed.get("head_direction") or "").lower()
        # "window" / "back" are the only directions consistent with
        # mind_diversion as defined in the prompt.
        if head_dir not in ("window", "back"):
            return f"TP claimed but head_direction={head_dir!r}"
        if not parsed.get("sustained_across_frames"):
            return "TP claimed but sustained_across_frames=false"
        return None

    if object_type == "group_detected":
        n = parsed.get("distinct_persons_visible_in_cabin", 0)
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            n_int = 0
        if n_int <= 5:
            return f"TP claimed but distinct_persons_visible_in_cabin={n_int} (≤5)"
        return None

    if object_type == "no_person_detected":
        if parsed.get("lp_visible_anywhere") or parsed.get("alp_visible_anywhere"):
            return "TP claimed but lp_visible_anywhere or alp_visible_anywhere is true"
        if not parsed.get("cabin_empty_in_all_frames"):
            return "TP claimed but cabin_empty_in_all_frames=false"
        return None

    if object_type == "solo_person":
        n = parsed.get("distinct_persons_visible_in_strip", 0)
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            n_int = 0
        # The whole point of solo_person verification: if the VLM saw two or
        # more distinct people in any frame, the Pipeline-1 trigger is a FP
        # (YOLO recall drop, dedup merge, occlusion). Demote any TP that
        # contradicts its own count.
        if n_int >= 2:
            return f"TP claimed but distinct_persons_visible_in_strip={n_int} (≥2)"
        if parsed.get("alp_visible"):
            return "TP claimed but alp_visible=true"
        # Tightened 2026-05-10 after run_20260510_042945 produced 6 FP
        # solo_person events where Qwen-VL missed partially-clipped ALPs at
        # the upper frame edge. The model is now asked to set
        # ``second_person_partial_or_occluded=true`` whenever it has ANY
        # uncertainty about a second body. If it does, the Pipeline-1 raw-
        # count gate already vetoed this case in 99% of frames; in the rare
        # case both layers agree there's just one person AND the model still
        # marked partial-occlusion, we'd rather demote to UNCERTAIN than
        # ship the violation.
        if parsed.get("second_person_partial_or_occluded"):
            return "TP claimed but second_person_partial_or_occluded=true"
        # Motion-state gate (added 2026-05-10): solo_person is only a
        # violation while the train is RUNNING. If the model itself reports
        # ``train_appears_to_be`` as anything other than "running" — i.e.
        # "stopped" or "unclear" — the structural rule demotes any
        # cooperative TP. This is the post-VLM enforcement of the prompt's
        # HARD RULE 2 (see ``_PROMPT_SOLO_PERSON``). Catches cases where
        # the VLM honestly reports stopped/unclear but still emits TP
        # because the person-count rule alone would have allowed it.
        motion = (parsed.get("train_appears_to_be") or "").lower()
        if motion not in ("running",):
            return f"TP claimed but train_appears_to_be={motion!r} (not 'running')"
        return None

    return None


# ---------------------------------------------------------------------------
# Motion-override regex helpers
# ---------------------------------------------------------------------------
# A "hard stopped cue" is concrete visual evidence the VLM saw OUTSIDE the
# cabin (window or door) of a stationary state. We require word-boundary
# matches (so "destination" does not match "station" and "doorway" does not
# match "door") and reject when the same clause contains a negation token.
_HARD_STOPPED_CUE_RE = re.compile(
    r"\b(?:door\s+(?:\w+\s+){0,3}open|open\s+door|platforms?|stations?)\b",
    re.IGNORECASE,
)
# Reject when the *same clause* contains a negation/uncertainty token so
# "platform visible, no motion blur" is split per-clause and the genuine
# STOPPED observation in the first clause is honored without being
# accidentally rejected by the negation in the second.
_NEGATION_RE = re.compile(
    r"\b(?:no|not|none|without|absent|unclear|cannot|can't)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[,;.()\n]+")


def _has_hard_stopped_cue(motion_evidence: str) -> bool:
    """True when motion_evidence cites an unambiguous STOPPED visual cue.

    A clause is accepted if it contains a hard cue AND no negation token.
    Multiple clauses (split on ``,;.()`` plus newlines) are checked
    independently — one clean affirmative is enough.
    """
    if not motion_evidence:
        return False
    for clause in _CLAUSE_SPLIT_RE.split(motion_evidence):
        if _HARD_STOPPED_CUE_RE.search(clause) and not _NEGATION_RE.search(clause):
            return True
    return False


_TRUNCATION_VERDICT_RE = re.compile(
    r'"verdict"\s*:\s*"(TRUE_POSITIVE|FALSE_POSITIVE|UNCERTAIN)"'
)
_TRUNCATION_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([\d.]+)')
_TRUNCATION_BOOL_FIELDS: tuple = (
    "hand_actually_on_book",
    "book_visible_on_desk",
    "book_is_open",
    "pen_in_hand",
    "actively_handling_papers",
    "head_oriented_to_book",
    "object_at_mouth",
    "bag_visible",
    "hand_in_or_on_bag",
    "posture_oriented_to_bag",
    "eyes_closed",
    "body_reclined",
    "hands_still",
    "head_motionless_across_frames",
    "sustained_across_frames",
    "lp_visible_anywhere",
    "alp_visible_anywhere",
    "partial_body_in_edge",
    "cabin_empty_in_all_frames",
    "duplicate_detections_likely",
    "people_visible_only_through_window",
    "posters_or_photos_on_wall",
)
_TRUNCATION_STRING_FIELDS: tuple = (
    "which_person",
    "primary_object_in_hand",
    "object_in_hand",
    "object_position",
    "train_appears_to_be",
    "motion_evidence",
    "tp_path",
    "primary_confounder",
    "head_direction",
    "reasoning",
)
_TRUNCATION_INT_FIELDS: tuple = (
    "evidence_frame",
    "distinct_persons_visible_in_cabin",
)


def _parse_verdict(raw_text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction. Handles three failure modes:

    1. ``code-fence`` wrapped output (already tolerated): strip the fence.
    2. Truncated JSON missing the closing ``}`` (common when the VLM hits
       max_tokens mid-response): try appending closing braces and re-parse.
    3. Severe truncation that breaks even string literals: regex-salvage the
       verdict, confidence, and structured booleans from the partial text.
       The ``verdict-first`` prompt schema places the load-bearing fields at
       the top of the JSON, so salvage almost always recovers them.

    The regex-salvage path returns a dict with ``_salvaged_from_truncation:
    True`` so downstream code (consistency check, telemetry) can flag these
    cases for prompt-tuning audit. A salvaged verdict still goes through the
    normal verdict-enum validation in ``_verify_one_async``.
    """
    t = (raw_text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    start = t.find("{")
    if start < 0:
        return {"parse_error": "no_json_object", "raw_text": raw_text[:300]}

    # Phase 1: strict parse on the full {…} window.
    end = t.rfind("}")
    if end > start:
        try:
            return json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            # Fall through to salvage paths below — strict parse failed,
            # likely because the rfind picked a nested } and the outer one
            # is missing, or there's a syntax error inside.
            pass

    # Phase 2: truncation salvage — append closing tokens and retry. We
    # trim trailing whitespace + a dangling comma (common at the truncation
    # point) before each attempt so the synthetic close lands on a valid
    # token boundary. The inner tokens cover an unclosed array (``]``) and
    # an unclosed dict-inside-array (``}]``) which are the most frequent
    # truncation shapes.
    body = t[start:].rstrip()
    if body.endswith(","):
        body = body[:-1].rstrip()
    for closing in ("}", "]}", "}]}", '"}', '"]}'):
        try:
            return json.loads(body + closing)
        except json.JSONDecodeError:
            continue

    # Phase 3: regex salvage. Extract the verdict and confidence (the only
    # two truly load-bearing fields) plus whatever structured fields are
    # legible. This is intentionally tolerant — partial extraction is
    # better than dropping a real verdict because of one bad escape.
    body_for_regex = body
    salvaged: Dict[str, Any] = {}
    m_verdict = _TRUNCATION_VERDICT_RE.search(body_for_regex)
    if m_verdict:
        salvaged["verdict"] = m_verdict.group(1)
    m_conf = _TRUNCATION_CONFIDENCE_RE.search(body_for_regex)
    if m_conf:
        try:
            salvaged["confidence"] = float(m_conf.group(1))
        except ValueError:
            pass
    for field in _TRUNCATION_BOOL_FIELDS:
        m = re.search(r'"' + field + r'"\s*:\s*(true|false)', body_for_regex)
        if m:
            salvaged[field] = (m.group(1) == "true")
    for field in _TRUNCATION_STRING_FIELDS:
        m = re.search(r'"' + field + r'"\s*:\s*"([^"\\]*)"', body_for_regex)
        if m:
            salvaged[field] = m.group(1)
    for field in _TRUNCATION_INT_FIELDS:
        m = re.search(r'"' + field + r'"\s*:\s*(-?\d+)', body_for_regex)
        if m:
            try:
                salvaged[field] = int(m.group(1))
            except ValueError:
                pass

    # Verdict alone is enough to act on; confidence falls back to 0.5
    # (UNCERTAIN territory) when the regex can't find it.
    if "verdict" in salvaged:
        salvaged.setdefault("confidence", 0.5)
        salvaged["_salvaged_from_truncation"] = True
        logger.info(
            "[vlm] verdict salvaged from truncated JSON: verdict=%s "
            "confidence=%.2f fields=%d raw_len=%d",
            salvaged.get("verdict"), salvaged.get("confidence", 0.0),
            len(salvaged), len(raw_text or ""),
        )
        return salvaged

    return {
        "parse_error": "unable_to_salvage_verdict_from_truncation",
        "raw_text": raw_text[:300],
    }


def _safe_motion_state(activity: Dict[str, Any], verdict_dict: Dict[str, Any]) -> str:
    """Return Pipeline-1's authoritative motion state, never the VLM's free text.

    Hardening rule: the VLM emits ``motion_evidence`` (free string) and
    ``train_appears_to_be`` (semi-structured: running/stopped/unclear).
    Pipeline-1's ``train_motion_detector`` produces ``activity['motionState']``
    which is the source of truth.  We refuse to flip ``RUNNING -> STOPPED``
    based on VLM motion observations alone — that would let a jail-broken
    VLM emitting ``motion_evidence: "platform"`` silently drop a real
    violation.

    This helper exists explicitly so that future "use VLM motion as a
    tiebreaker" code paths route through one validated function instead of
    sprinkling ad-hoc regexes across the codebase.

    Returns the Pipeline-1 motion state UPPER-cased.  Allowed values are
    ``RUNNING``, ``STOPPED``, ``UNCERTAIN``.  Any unknown / missing value
    becomes ``UNCERTAIN``.

    Tiebreaker policy:
      - If Pipeline-1 says RUNNING:  return RUNNING (VLM cannot override).
      - If Pipeline-1 says STOPPED:  return STOPPED.
      - If Pipeline-1 says UNCERTAIN AND the VLM's *structured* field
        ``train_appears_to_be`` is exactly ``"stopped"``: return UNCERTAIN
        (we still don't promote to STOPPED on VLM signal alone — that
        decision belongs to the dedicated train_motion_detector).
    """
    p1 = (activity.get("motionState") or "").strip().upper()
    if p1 in ("RUNNING", "STOPPED"):
        return p1
    # Pipeline-1 was UNCERTAIN / empty.  We DO NOT promote based on VLM
    # text — return UNCERTAIN so callers fall back to the safe default
    # (treat the activity as a real violation, do not drop).
    _ = verdict_dict.get("train_appears_to_be")  # observed only, not acted upon
    return "UNCERTAIN"
