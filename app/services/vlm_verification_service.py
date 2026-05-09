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

import asyncio
import base64
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from ..utils.config import get_settings
from ..utils.logger import get_logger


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


# ---------------------------------------------------------------------------
# Circuit breaker — protects vLLM during cold-starts / GC pauses
# ---------------------------------------------------------------------------
class _CircuitBreaker:
    """Simple threshold-based circuit breaker for the VLM HTTP client.

    After ``threshold`` consecutive failures the breaker opens; subsequent
    calls short-circuit to ``SKIPPED_VLM_UNAVAILABLE`` for ``cooldown_s``
    seconds, after which it resets.  A single success closes the breaker
    early.  Used by :class:`VlmVerificationService` to avoid wasting
    ``N x timeout_seconds`` per video during a vLLM cold-start window.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._fail_count = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at > self.cooldown_s:
                self._opened_at = None
                self._fail_count = 0
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if self._fail_count >= self.threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                logger.warning(
                    "[vlm] circuit breaker OPEN after %d consecutive failures; "
                    "skipping VLM calls for %.1fs",
                    self._fail_count, self.cooldown_s,
                )

    def record_success(self) -> None:
        with self._lock:
            if self._fail_count > 0 or self._opened_at is not None:
                logger.info("[vlm] circuit breaker CLOSED (success after failures)")
            self._fail_count = 0
            self._opened_at = None

    def reset(self) -> None:
        with self._lock:
            self._fail_count = 0
            self._opened_at = None


_PROMPT_WRITING = """You are a railway safety auditor reviewing CCTV from a locomotive cabin.
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

A classical CV pipeline flagged this frame as "WRITING IN LOG BOOK". The pipeline
does NOT specify which person — either LP or ALP could be the writer. Your job
is to verify whether ANYONE in the strip is actually writing in a log book in
at least one of the frames shown.

The strip you receive has been CROPPED to the hand + book / desk region — far
window, seats, and most of the cabin walls have been removed so you can focus
on the relevant area at higher pixel density. Use the visible scene context
inside the crop to make your call.

When you ARE confident in your verdict (whether TRUE or FALSE), say so with high
confidence (≥ 0.8). Reserve UNCERTAIN + low confidence for genuinely ambiguous
cases. Do NOT default to FALSE_POSITIVE just because verification is hard —
if you can clearly see writing happening in any frame, return TRUE_POSITIVE.

Note: the scene-observation fields below (book_visible_on_desk, book_is_open,
hand_actually_on_book, head_oriented_to_book) describe WHAT YOU SEE in the
strip. Populate them based on the visual content, NOT based on what your verdict
needs. A book may be visible but with no hand on it — that is FP, but
book_visible_on_desk should still be true.

True writing requires ALL FOUR of these to be VISIBLY CONFIRMED for the SAME person
in at least one frame. Posture alone is NEVER sufficient.

  (a) An OPEN book / notepad / paper is visible (closed books do NOT count;
      bound logbooks lying flat on the desk do not count unless the pages are
      open and visible)
  (b) That person's hand is in PHYSICAL CONTACT with the open page — a fingertip
      or pen tip must be visibly touching the paper. Hand resting NEAR the book,
      hand HOLDING a closed/folded book, or hand merely in the same image region
      as the book do NOT qualify.
  (c) EITHER a pen/pencil is clearly visible in the writing hand (held by
      fingertips, NOT the brake-handle grip) OR a fingertip is unambiguously on
      the page surface
  (d) Head is tilted DOWN toward the open page (not turned forward, not toward
      the window, not toward the controls)

If you cannot see (b) AND (c) confirmed for one specific person in at least one
frame, the verdict is FALSE_POSITIVE. Holding papers at chest level, looking
sideways, reading from a clipboard, reviewing the logbook without writing, or
operating any cab control are all NOT writing. When in doubt, return UNCERTAIN
with confidence ≤ 0.5 — do NOT default to TRUE_POSITIVE.

Common confounders the classical pipeline misclassifies as writing:
  - Hands resting in the lap with head bent down (idle posture, no book contact)
  - A book sitting unattended on the desk while crew operates controls (static-book FP)
  - Holding a folded paper or clipboard at chest level while looking forward
  - Holding the brake handle / throttle lever (long stick with knob between the
    knees or on the console) — the hand may briefly look like it is reaching
    forward; verify it is on PAPER, not on a metal control
  - Holding a railway radio handset to face/ear (brick-shaped, often with coiled cord)

ALSO observe whether the train is moving, using cues OUTSIDE the cabin (window/door):
  - "running": visible motion blur in the outside window, scenery/track streaking past,
               telephone poles or trees flashing by
  - "stopped": platform infrastructure visible, station signs, people walking on a
               platform, the cabin door is OPEN (trains do not run with the door
               open), or the outside view is clearly stationary with no motion blur
  - "unclear": window is dark, blocked, glare, or you simply cannot see enough outside
               to tell — DO NOT GUESS, return "unclear"
This is a separate observation; do NOT use motion alone to decide the writing verdict.

Reply with STRICT JSON ONLY (no prose, no code fence):
{
  "verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN",
  "which_person": "LP" | "ALP" | "neither" | "unclear",
  "confidence": <float 0.0 to 1.0 — your certainty in the VERDICT ITSELF. If verdict=FALSE_POSITIVE and you are sure no activity is happening, set confidence ≥ 0.80. If verdict=TRUE_POSITIVE and you are sure the activity is happening, set confidence ≥ 0.80. Use ≤ 0.5 only when you are genuinely unsure. Do NOT set 0.0 to mean "no activity"; that means "I am 0% sure of my verdict">,
  "book_visible_on_desk": <true|false>,
  "book_is_open": <true|false>,
  "hand_actually_on_book": <true|false>,
  "head_oriented_to_book": <true|false>,
  "primary_object_in_hand": "pen" | "brake_or_throttle" | "radio_handset" | "phone" | "nothing_visible" | "unclear",
  "train_appears_to_be": "running" | "stopped" | "unclear",
  "motion_evidence": "<short string: the visual cue you used, e.g. 'platform visible', 'motion blur in window', 'no outside visible'>",
  "evidence_frame": <integer: frame number 1..N giving strongest evidence; 0 if single-frame input or no decisive frame>,
  "reasoning": "<one short sentence describing what you actually see, naming the FRAME number>"
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


# Activity types where the pre-VLM no-subject gate must NOT fire because
# "no person visible" is either the violation itself (no_person_detected)
# or operationally ambiguous (group_detected expects multiple persons but
# Pipeline-1 may render zero bboxes when its own count disagrees with the
# rendered set). For these types, the VLM is the right adjudicator.
_PRE_GATE_SKIP_OBJECT_TYPES: frozenset = frozenset({
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


def _supplement_keyframes_from_clip(
    activity: Dict[str, Any],
    existing: List[Path],
    target_n: int,
) -> List[Path]:
    """Sample additional frames from ``activityClip`` to reach ``target_n``.

    Single-burst activities (``_sourceActivities`` absent) yield exactly
    one keyframe from :func:`_resolve_keyframes`. The setting
    ``vlm_strip_target_frames`` (default 5) promises temporal evidence for
    those cases, so when fewer keyframes are present than ``target_n``
    we open the activity's ``activityClip`` and decode a small set of
    evenly-spaced frames. Frames are written next to the existing keyframe
    with a ``_supp{idx}.jpg`` suffix.

    Returns the (possibly-extended) path list in time order. Failures
    here are non-fatal: if cv2 can't open the clip we just return
    ``existing`` unchanged so the verifier proceeds with whatever it has.
    """
    if target_n <= 1 or len(existing) >= target_n:
        return existing
    clip_path = activity.get("activityClip") or ""
    if not clip_path:
        return existing
    clip = Path(str(clip_path))
    if not clip.is_file():
        return existing

    # Anchor name for sibling frames (use first existing keyframe so the
    # supplements colocate with the burst evidence, not the clip).
    anchor = existing[0] if existing else clip.with_suffix(".jpg")
    parent = anchor.parent
    stem = anchor.stem.replace("_activity", "")

    try:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            return existing
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                return existing
            need = target_n - len(existing)
            # Evenly distribute the supplementary frames across the clip,
            # avoiding the exact endpoints (which often differ from the
            # already-saved activity keyframe by only a frame or two).
            offsets: List[int] = []
            for i in range(need):
                # i runs 0..need-1; map to (i+1)/(need+1) of the clip
                pos = int(round((i + 1) * frame_count / (need + 1)))
                pos = max(0, min(frame_count - 1, pos))
                offsets.append(pos)
            # De-dup while preserving order.
            seen_ofs: set = set()
            offsets = [o for o in offsets if not (o in seen_ofs or seen_ofs.add(o))]

            new_paths: List[Path] = []
            for idx, ofs in enumerate(offsets):
                cap.set(cv2.CAP_PROP_POS_FRAMES, ofs)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                out = parent / f"{stem}_supp{idx}.jpg"
                # If we already wrote this on a prior call, reuse it.
                if not out.is_file():
                    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if out.is_file():
                    new_paths.append(out)
            if not new_paths:
                return existing
            # Place originals first (they have Pipeline-1 bbox overlays the
            # gate relies on); supplements after as additional context.
            return existing + new_paths
        finally:
            cap.release()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[vlm] clip-supplement failed for %s: %s", clip_path, e,
        )
        return existing


def _count_bboxes_in_keyframes(
    jpg_paths: List[Path],
    min_person_area: int = 1000,
    min_object_area: int = 300,
) -> Dict[str, int]:
    """Count Pipeline-1 overlay bboxes by colour across keyframes.

    Pipeline-1 renders each detected person bbox in bright GREEN (HSV hue
    ~60) and each object bbox (book / cup / bottle / bag / phone) in
    ORANGE/YELLOW (HSV hue 15-35) onto the saved ``*_activity.jpg``.

    IMPORTANT: this helper is only meaningful for paths returned by
    :func:`_resolve_keyframes` (the per-burst saved keyframes which
    carry the rendered overlay). Supplementary frames sampled from
    ``activityClip`` via :func:`_supplement_keyframes_from_clip` are
    raw decoded frames with NO bbox overlay; passing them in will
    return zero counts and defeat the gate logic that depends on
    rendering being active.

    Returns:
        ``{"with_person": int, "with_object": int, "with_any_bbox": int, "total": int}``.
        ``with_person`` and ``with_object`` are independent counts. The
        rule of thumb for callers: only enforce gate decisions when
        ``with_any_bbox > 0`` (i.e. overlay rendering is confirmed
        active), otherwise fall through to the VLM.
    """
    with_person = 0
    with_object = 0
    with_any = 0
    for p in jpg_paths:
        try:
            img = cv2.imread(str(p))
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Pipeline-1 colour palette as observed on the saved keyframes:
            #   GREEN   ~ hue 60   → person bbox
            #   YELLOW/ORANGE ~ hue 15-35 → object bbox (book, cup, etc.)
            #   MAGENTA ~ hue 140-170 → bag/backpack bbox
            #   RED     ~ hue 0 / 170-180 → keypoint markers + sometimes labels
            # We use green strictly for the person count, orange for the
            # object count, and treat green|orange|magenta|red collectively
            # as the "rendering is active" signal so a frame with only a
            # bag bbox (e.g. an empty-cab clip with a backpack on the seat)
            # still confirms the overlay is on.
            green = cv2.inRange(hsv, (45, 150, 150), (75, 255, 255))
            orange = cv2.inRange(hsv, (15, 150, 150), (35, 255, 255))
            magenta = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))
            red_low = cv2.inRange(hsv, (0, 150, 100), (10, 255, 255))
            red_high = cv2.inRange(hsv, (170, 150, 100), (180, 255, 255))

            person_found = False
            for c in cv2.findContours(
                green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                if cv2.contourArea(c) >= min_person_area:
                    _, _, w, h = cv2.boundingRect(c)
                    # A person bbox is at least ~40x40; reject thin
                    # skeleton-line fragments that survive area threshold.
                    if w >= 40 and h >= 40:
                        person_found = True
                        break

            object_found = False
            for c in cv2.findContours(
                orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                if cv2.contourArea(c) >= min_object_area:
                    _, _, w, h = cv2.boundingRect(c)
                    if w >= 30 and h >= 30:
                        object_found = True
                        break

            # Render-active probe: any sufficiently-large saturated cluster
            # in green / orange / magenta / red. The threshold is small
            # (200 px) because we only need *some* evidence of overlay
            # rendering — a bag bbox or a few keypoint markers count.
            render_active = person_found or object_found
            if not render_active:
                misc_mask = cv2.bitwise_or(
                    magenta, cv2.bitwise_or(red_low, red_high),
                )
                for c in cv2.findContours(
                    misc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]:
                    if cv2.contourArea(c) >= 200:
                        render_active = True
                        break

            if person_found:
                with_person += 1
            if object_found:
                with_object += 1
            if render_active:
                with_any += 1
        except Exception:  # noqa: BLE001
            # Any per-frame failure is non-fatal — fall back to
            # treating the frame as "unknown", which is conservatively
            # the safer choice (don't drop on shaky signal).
            continue
    return {
        "with_person": with_person,
        "with_object": with_object,
        "with_any_bbox": with_any,
        "total": len(jpg_paths),
    }


# Activity types whose Pipeline-1 trigger requires a target object
# (book / cup / bag / phone). For these, an absent orange/yellow object
# bbox in EVERY original keyframe is strong evidence of a stale-state
# Pipeline-1 trigger — the rule fired without the actual object bbox
# being detected at the keyframe time. Used by the pre-VLM object gate.
_OBJECT_REQUIRED_TYPES: frozenset = frozenset({
    "writing",
    "eating_drinking",
    "packing_bags",
    "cell_phone",
})


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
        if not parsed.get("hand_actually_on_book"):
            return "TP claimed but hand_actually_on_book=false"
        if not parsed.get("book_visible_on_desk"):
            return "TP claimed but book_visible_on_desk=false"
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

    return None


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
            logger.warning("[vlm] cv2.imread returned None for %s", p)
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


class VlmVerificationService:
    """Stateless verifier that talks to a vLLM endpoint over HTTP.

    The class itself is lightweight (no model loaded in-process); each call
    POSTs to ``{vlm_base_url}/chat/completions`` with one inline-encoded image.

    HTTP transport: a shared ``httpx.AsyncClient`` is created lazily per
    asyncio event loop (clients cannot be shared across loops).  Connection
    pooling avoids the per-activity TCP+TLS handshake of the legacy
    ``urllib`` path.  Per-activity calls are dispatched concurrently via an
    asyncio semaphore (bounded by ``max_keepalive_connections``).

    Resilience: a process-level :class:`_CircuitBreaker` opens after three
    consecutive failures and short-circuits subsequent calls for 30s, so a
    cold-start vLLM doesn't burn ``N x timeout_seconds`` per video.

    Fail-open contract: every per-activity body is wrapped in
    ``try/except``; on any exception the activity is appended to ``kept``
    with ``vlm_review = {"verdict": "SKIPPED_VLM_UNAVAILABLE"}``.  Pipeline-1
    violations are NEVER silently dropped.
    """

    # Per-loop cache of httpx.AsyncClient instances (clients are tied to a
    # specific event loop; safer to pin one per loop than to share globally).
    _async_clients: Dict[int, "httpx.AsyncClient"] = {}
    _async_clients_lock = threading.Lock()

    # Process-level circuit breaker; shared across calls so a single bad
    # cold-start window opens it for everyone.
    _breaker: _CircuitBreaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    def __init__(self) -> None:
        self.settings = get_settings()
        # Cache the verify-list as a frozen set for fast membership checks.
        self._verify_set: frozenset[str] = frozenset(
            x.strip() for x in self.settings.vlm_verify_activities.split(",") if x.strip()
        )
        # Bounded concurrency for the async dispatch.  4 in-flight matches
        # the keepalive pool below; each activity is one request.
        self._concurrency = 4
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
    # Async client lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def _get_async_client(cls, timeout_s: float) -> "httpx.AsyncClient":
        """Return a per-event-loop ``httpx.AsyncClient`` (lazy-created).

        ``httpx.AsyncClient`` instances are bound to the loop that creates
        them, so we key the cache by ``id(asyncio.get_running_loop())``.
        Within a single ``verify_activities`` invocation the same loop is
        active for all dispatched calls, so they share the keepalive pool.
        Across invocations the loop changes (``asyncio.run`` builds a new
        one each time), so a fresh client is constructed.

        The cache here is mostly a safety net for tests / future code that
        might pin a loop across multiple service calls; production benefit
        comes from the in-call pooling.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — caller is misusing the API.  Build a
            # one-shot client that the caller must close.
            return httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s),
                limits=httpx.Limits(
                    max_connections=8, max_keepalive_connections=4,
                ),
            )
        loop_key = id(loop)
        with cls._async_clients_lock:
            client = cls._async_clients.get(loop_key)
            if client is None or client.is_closed:
                # Drop any stale entries that point at closed clients to
                # avoid unbounded growth across many verify_activities
                # calls (each builds a new loop).
                stale = [k for k, v in cls._async_clients.items() if v.is_closed]
                for k in stale:
                    cls._async_clients.pop(k, None)
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout_s),
                    limits=httpx.Limits(
                        max_connections=8,
                        max_keepalive_connections=4,
                    ),
                )
                cls._async_clients[loop_key] = client
            return client

    @classmethod
    async def aclose(cls) -> None:
        """Close all cached async clients.  Safe to call from shutdown hooks."""
        with cls._async_clients_lock:
            clients = list(cls._async_clients.values())
            cls._async_clients.clear()
        for c in clients:
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("[vlm] error closing async client", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return bool(self.settings.vlm_verification_enabled)

    def verify_activities(
        self, activities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Verify a batch of activities and return the post-filter list.

        Synchronous wrapper around :meth:`verify_activities_async`.  Both
        existing call-sites (``video_processing_service.process_video``
        and ``video_controller.process_and_upload_video``) call this
        synchronously.  The latter happens to live inside an async FastAPI
        handler — when we detect an already-running event loop we spin up
        a fresh thread that owns its own loop, so we never collide with
        ``asyncio.run cannot be called from a running event loop``.

        Each input activity dict is mutated in place to add a ``vlm_review``
        sub-dict with the verdict, latency, and (in enforcement mode) a
        ``dropped`` flag.

        Fail-open contract: any unexpected exception in ``_verify_one`` is
        caught here, the activity is annotated with
        ``SKIPPED_VLM_UNAVAILABLE``, and it is kept in the output list.
        Pipeline-1 violations are never silently dropped.

        Returns:
            Tuple of (kept_activities, stats_dict). ``stats_dict`` keys:
            verified, skipped_type, skipped_stopped, skipped_unavailable,
            skipped_no_image, dropped, kept, uncertain, skipped_parse_error,
            motion_overrides.
        """
        async def _run() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
            try:
                return await self.verify_activities_async(activities)
            finally:
                # Close any clients pinned to this loop before the loop
                # tears down — avoids "unclosed client" warnings.
                await VlmVerificationService.aclose()

        # Detect whether we're already inside an event loop.  If yes
        # (FastAPI handler context), run on a worker thread that owns
        # its own loop so we can safely `asyncio.run`.
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            return asyncio.run(_run())

        # Inside a running loop: dispatch to a worker thread.
        result_box: List[Any] = []

        def _thread_main() -> None:
            try:
                result_box.append(asyncio.run(_run()))
            except BaseException as e:  # noqa: BLE001
                result_box.append(e)

        t = threading.Thread(target=_thread_main, daemon=True, name="vlm-verify")
        t.start()
        t.join()
        if not result_box:
            raise RuntimeError("vlm verify thread produced no result")
        out = result_box[0]
        if isinstance(out, BaseException):
            raise out
        return out

    async def verify_activities_async(
        self, activities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Async core of :meth:`verify_activities`.

        Two-phase design:

        1.  Iterate the input list once, classifying each activity into one
            of three buckets: ``passthrough`` (skipped types / STOPPED /
            cap-exceeded — no VLM call), ``to_verify`` (eligible for the
            VLM), or ``no_image`` (eligible but no keyframe found).
        2.  Dispatch the ``to_verify`` bucket concurrently via
            :func:`asyncio.gather` with a bounded semaphore.

        Each per-activity body is wrapped in ``try/except``; on any
        exception the activity is appended to ``kept`` with
        ``vlm_review = {"verdict": "SKIPPED_VLM_UNAVAILABLE",
        "reason": type(e).__name__}``.  This is the load-bearing invariant:
        Pipeline-1 violations must never be lost.
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
            # Spec key (docs/specs/code-review-fixes/tasks/0007-...): the
            # canonical name for "vLLM responded but its body was
            # un-parseable" is `skipped_parse_error`.  Older code used
            # `parse_errors`; renamed for cross-reference compatibility.
            "skipped_parse_error": 0,
            # Count of activities where the VLM's motion_evidence flipped
            # motionState from RUNNING -> STOPPED so the downstream filter
            # in video_controller.py drops them. Only the RUNNING -> STOPPED
            # direction is supported here; the inverse case (P1 says STOPPED
            # but VLM disagrees) requires deferring gates.apply_train_stopped_
            # suppression because suppressed activities never reach this
            # verifier in the first place.
            "motion_overrides": 0,
            # Pre-VLM no-subject gate drops (no person bbox in any keyframe).
            # Counted separately from regular VLM-driven drops for telemetry
            # so we can monitor false-drop risk during rollout.
            "pre_gate_drops": 0,
            # Post-VLM consistency overrides (TP demoted to UNCERTAIN
            # because structured fields contradicted the verdict).
            "consistency_overrides": 0,
        }
        if not self.is_enabled() or not activities:
            return activities, stats

        cap = int(self.settings.vlm_max_activities_per_run or 0)

        # Phase 1: classify activities, preserving input order in `kept`.
        # `to_verify` collects (index_in_kept, activity, prompt, object_type)
        # so we can patch the result back in-place after gather() returns.
        kept: List[Dict[str, Any]] = []
        to_verify: List[Tuple[int, Dict[str, Any], str, str]] = []
        verified_count = 0

        for act in activities:
            try:
                object_type = (act.get("objectType") or "").strip().lower().replace(" ", "_")
                if object_type not in self._verify_set or object_type not in _PROMPTS_BY_OBJECT_TYPE:
                    stats["skipped_type"] += 1
                    kept.append(act)
                    continue

                # Mirror the downstream STOPPED filter in video_controller.py.
                # Two cases:
                #   1. object_type IS in the train-stopped suppress list (writing,
                #      sleep, packing_bags, gestures, mind_diversion, eating). The
                #      gate would drop these from the API post regardless, so a
                #      VLM call is wasted compute — skip with SKIPPED_STOPPED.
                #   2. object_type is NOT in the suppress list (cell_phone,
                #      microsleep — safety-critical types in
                #      MOTION_FILTER_BYPASS_TYPES). The gate keeps these even when
                #      STOPPED, so they DO reach the API. Fall through and verify
                #      so VLM has a chance to drop the FP. Without this branch the
                #      verifier silently no-ops on the exact archetype it was added
                #      to catch (e.g. radio-handset misclassified as phone).
                if (act.get("motionState") or "").strip().upper() == "STOPPED":
                    _override = getattr(self.settings, "train_motion_stopped_suppress_list", "") or ""
                    if _override:
                        _gate_drops = {s.strip() for s in _override.split(",") if s.strip()}
                    else:
                        from app.core.gates import DEFAULT_SUPPRESSED_WHEN_STOPPED
                        _gate_drops = set(DEFAULT_SUPPRESSED_WHEN_STOPPED)
                    if object_type in _gate_drops:
                        stats["skipped_stopped"] += 1
                        act["vlm_review"] = {
                            "status": "SKIPPED_STOPPED",
                            "verdict": None,
                            "reason": "motionState=STOPPED; gate suppresses this type",
                        }
                        kept.append(act)
                        continue
                    # Else: gate would NOT drop this type — fall through to verify.

                if cap and verified_count >= cap:
                    stats["skipped_type"] += 1
                    kept.append(act)
                    continue

                # Eligible — schedule for the VLM call.
                prompt = _PROMPTS_BY_OBJECT_TYPE[object_type]
                kept_idx = len(kept)
                kept.append(act)
                to_verify.append((kept_idx, act, prompt, object_type))
                verified_count += 1
            except Exception as classify_exc:  # noqa: BLE001
                # Even classification can blow up (weird types, bad dicts).
                # Keep the activity, mark it skipped, never lose it.
                logger.exception(
                    "[vlm] classification failed for activity id=%s type=%s",
                    act.get("id"), act.get("activityType"),
                )
                act["vlm_review"] = {
                    "status": "SKIPPED_VLM_UNAVAILABLE",
                    "verdict": "SKIPPED_VLM_UNAVAILABLE",
                    "reason": type(classify_exc).__name__,
                }
                stats["skipped_unavailable"] += 1
                kept.append(act)

        # Phase 2: dispatch all eligible activities concurrently.
        if to_verify:
            sem = asyncio.Semaphore(self._concurrency)

            async def _run_one(idx: int, activity: Dict[str, Any], prompt: str,
                               obj_type: str) -> Tuple[int, Dict[str, Any]]:
                try:
                    async with sem:
                        review = await self._verify_one_async(activity, prompt, obj_type)
                    return idx, review
                except Exception as e:  # noqa: BLE001
                    # CRITICAL: never let a single activity's failure
                    # propagate up and truncate the kept list.
                    logger.exception(
                        "[vlm] verify_one failed for activity id=%s type=%s",
                        activity.get("id"), activity.get("activityType"),
                    )
                    return idx, {
                        "status": "SKIPPED_VLM_UNAVAILABLE",
                        "verdict": "SKIPPED_VLM_UNAVAILABLE",
                        "reason": type(e).__name__,
                        "error": str(e)[:200],
                    }

            results = await asyncio.gather(
                *(_run_one(i, a, p, o) for (i, a, p, o) in to_verify),
                return_exceptions=False,
            )

            # Phase 3: post-process each result, mutate the activity in
            # `kept`, and update stats / drop flags.
            #
            # Each iteration is wrapped in try/except so an unexpected shape
            # (e.g. `review` being a non-dict truthy value, or a stray
            # AttributeError from a misbehaving mock) cannot propagate out
            # of `verify_activities_async` and discard the entire `kept`
            # list mid-loop.  The activity is already in `kept` from
            # Phase 1 — a failure here just means it stays un-annotated,
            # which is acceptable under the fail-open contract.
            drop_indices: List[int] = []
            for kept_idx, review in results:
                act = kept[kept_idx]
                try:
                    act["vlm_review"] = review

                    status = review.get("status")
                    if status == "SKIPPED_VLM_UNAVAILABLE":
                        stats["skipped_unavailable"] += 1
                        continue
                    if status == "SKIPPED_NO_IMAGE":
                        stats["skipped_no_image"] += 1
                        continue
                    if status == "PARSE_ERROR":
                        stats["skipped_parse_error"] += 1
                        continue
                    if status in ("PRE_GATE_DROP_NO_SUBJECT", "PRE_GATE_DROP_NO_OBJECT"):
                        # Deterministic pre-VLM drop. Honor shadow mode
                        # (don't actually drop) but always record the
                        # decision so we can audit the gate's behaviour.
                        stats["pre_gate_drops"] += 1
                        if not self.settings.vlm_shadow_mode:
                            review["dropped"] = True
                            stats["dropped"] += 1
                            drop_indices.append(kept_idx)
                        else:
                            stats["kept"] += 1
                        continue

                    stats["verified"] += 1
                    verdict_dict = review.get("verdict") or {}
                    if verdict_dict.get("consistency_override"):
                        stats["consistency_overrides"] += 1
                    verdict = verdict_dict.get("verdict")
                    confidence = verdict_dict.get("confidence", 0.0)
                    try:
                        confidence = float(confidence)
                    except (TypeError, ValueError):
                        confidence = 0.0

                    # Wave-2: confidence calibration. Maps raw VLM
                    # confidence to calibrated probability via the
                    # learned mapping (default identity until ground
                    # truth is collected). Stored on review for audit.
                    raw_confidence = confidence
                    if getattr(self.settings, "vlm_calibration_enabled", False):
                        try:
                            cal = _get_calibrator(
                                getattr(
                                    self.settings, "vlm_calibration_path",
                                    "/opt/poc2/app/data/vlm_calibration.json",
                                )
                            )
                            object_type_for_cal = (
                                act.get("objectType") or ""
                            ).strip().lower().replace(" ", "_")
                            confidence = cal.calibrate(raw_confidence, object_type_for_cal)
                            verdict_dict["calibrated_confidence"] = round(confidence, 4)
                            verdict_dict["raw_confidence"] = round(raw_confidence, 4)
                        except Exception:  # noqa: BLE001
                            # Identity fallback on any calibration failure.
                            confidence = raw_confidence

                    # Verdict enum validation (defensive — _verify_one_async
                    # already filters, but cheap to double-check at the boundary).
                    if verdict not in _ALLOWED_VERDICTS:
                        stats["skipped_parse_error"] += 1
                        review["status"] = "PARSE_ERROR"
                        review["error"] = f"verdict not in enum: {verdict!r}"
                        continue

                    # Motion-override observability: record the helper's
                    # opinion on the activity for ops audit. _safe_motion_state
                    # is conservative — it never flips RUNNING -> STOPPED on
                    # its own and is purely diagnostic.
                    act["motion_state_after_vlm"] = _safe_motion_state(act, verdict_dict)

                    # Motion-override active path (RUNNING -> STOPPED only).
                    # If Pipeline-1's vibration/window-flow detector missed a
                    # station stop but the VLM clearly sees a hard STOPPED cue
                    # (cabin door open, platform/station visible) AND is not
                    # negated in the same clause, overwrite motionState so the
                    # downstream STOPPED-filter in video_controller.py drops
                    # the activity from the external API post. The reverse
                    # direction (VLM-says-RUNNING, P1-says-STOPPED) is not
                    # handled here — that case requires deferring
                    # gates.apply_train_stopped_suppression because
                    # suppressed activities never reach this verifier.
                    if getattr(self.settings, "vlm_motion_override_enabled", False):
                        train_obs = (verdict_dict.get("train_appears_to_be") or "").strip().lower()
                        motion_evidence = str(verdict_dict.get("motion_evidence") or "")
                        current_motion = (act.get("motionState") or "").strip().upper()
                        if (
                            train_obs == "stopped"
                            and _has_hard_stopped_cue(motion_evidence)
                            and current_motion != "STOPPED"
                        ):
                            act["motionState"] = "STOPPED"
                            review["motion_override"] = {
                                "from": current_motion or "UNKNOWN",
                                "to": "STOPPED",
                                "cue": motion_evidence[:200],
                            }
                            stats["motion_overrides"] += 1
                            logger.info(
                                "[vlm] MOTION OVERRIDE activity type=%s at t=%s: %s -> STOPPED "
                                "(cue=%r)",
                                act.get("activityType"),
                                act.get("activityStartTime"),
                                current_motion or "UNKNOWN",
                                motion_evidence[:120],
                            )

                    if verdict == "UNCERTAIN":
                        stats["uncertain"] += 1

                    should_drop = (
                        not self.settings.vlm_shadow_mode
                        and verdict == "FALSE_POSITIVE"
                        and confidence >= self.settings.vlm_drop_threshold
                    )
                    if should_drop:
                        review["dropped"] = True
                        stats["dropped"] += 1
                        drop_indices.append(kept_idx)
                        logger.info(
                            "[vlm] DROPPED activity type=%s desc=%r at t=%s "
                            "verdict=FALSE_POSITIVE conf=%.2f reason=%r",
                            act.get("activityType"),
                            (act.get("des") or "")[:60],
                            act.get("activityStartTime"),
                            confidence,
                            (verdict_dict.get("reasoning") or "")[:120],
                        )
                    else:
                        stats["kept"] += 1

                    # Wave-2 telemetry: structured JSONL line per VLM
                    # call so we can analyse verdict distribution,
                    # latency, gate impact, and drift offline. Best-effort.
                    if getattr(self.settings, "vlm_telemetry_log_enabled", True):
                        _append_jsonl(
                            getattr(
                                self.settings, "vlm_telemetry_log_path",
                                "/opt/poc2/locopilot_evidence/vlm_telemetry.jsonl",
                            ),
                            {
                                "ts": time.time(),
                                "trip_id": act.get("tripId"),
                                "activity_type": act.get("activityType"),
                                "object_type": act.get("objectType"),
                                "activity_start": act.get("activityStartTime"),
                                "motion_state": act.get("motionState"),
                                "people_count": act.get("peopleCount"),
                                "vlm_status": review.get("status"),
                                "verdict": verdict,
                                "raw_confidence": raw_confidence,
                                "calibrated_confidence": (
                                    confidence if confidence != raw_confidence else None
                                ),
                                "consistency_override": bool(
                                    verdict_dict.get("consistency_override")
                                ),
                                "frames_sent": review.get("frames_sent"),
                                "latency_sec": review.get("latency_sec"),
                                "dropped": bool(review.get("dropped")),
                            },
                        )

                    # Wave-2 disagreement queue: capture cases where the
                    # P1 verdict and the VLM verdict diverge sharply.
                    # P1 fired the activity (so its verdict is implicitly
                    # TP); a VLM FALSE_POSITIVE that didn't reach the
                    # drop threshold OR an UNCERTAIN verdict means the
                    # two pipelines disagree but the system kept the
                    # detection. These are the highest-leverage examples
                    # for ground-truthing and improving either pipeline.
                    if getattr(self.settings, "vlm_disagreement_log_enabled", True):
                        is_disagreement = (
                            verdict in ("FALSE_POSITIVE", "UNCERTAIN")
                            and not review.get("dropped")
                        )
                        if is_disagreement:
                            _append_jsonl(
                                getattr(
                                    self.settings, "vlm_disagreement_log_path",
                                    "/opt/poc2/locopilot_evidence/vlm_disagreements.jsonl",
                                ),
                                {
                                    "ts": time.time(),
                                    "trip_id": act.get("tripId"),
                                    "activity_type": act.get("activityType"),
                                    "object_type": act.get("objectType"),
                                    "activity_start": act.get("activityStartTime"),
                                    "activity_end": act.get("activityEndTime"),
                                    "motion_state": act.get("motionState"),
                                    "people_count": act.get("peopleCount"),
                                    "p1_verdict": "TRUE_POSITIVE",  # P1 fired
                                    "vlm_verdict": verdict,
                                    "vlm_calibrated_confidence": confidence,
                                    "vlm_raw_confidence": raw_confidence,
                                    "vlm_reasoning": (
                                        verdict_dict.get("reasoning") or ""
                                    )[:300],
                                    "consistency_override": (
                                        verdict_dict.get("consistency_override")
                                    ),
                                    "activity_image": act.get("activityImage"),
                                    "activity_clip": act.get("activityClip"),
                                },
                            )
                except Exception:  # noqa: BLE001
                    # Fail-open: never let a single malformed review
                    # truncate `kept`.  The activity is already retained
                    # in `kept` from Phase 1; we simply leave it
                    # un-annotated (or partially annotated) and continue.
                    logger.exception(
                        "[vlm] post-process failed for activity %s",
                        act.get("id"),
                    )
                    continue

            # Materialise drops by removing in reverse-index order.
            if drop_indices:
                drop_set = set(drop_indices)
                kept = [a for i, a in enumerate(kept) if i not in drop_set]

        logger.info(
            "[vlm] verification stats: verified=%d kept=%d dropped=%d uncertain=%d "
            "motion_overrides=%d pre_gate_drops=%d consistency_overrides=%d "
            "skipped_type=%d skipped_stopped=%d "
            "skipped_unavailable=%d skipped_parse_error=%d shadow=%s",
            stats["verified"],
            stats["kept"] + stats["skipped_type"] + stats["skipped_stopped"]
            + stats["skipped_unavailable"] + stats["skipped_no_image"]
            + stats["skipped_parse_error"],
            stats["dropped"],
            stats["uncertain"],
            stats["motion_overrides"],
            stats["pre_gate_drops"],
            stats["consistency_overrides"],
            stats["skipped_type"],
            stats["skipped_stopped"],
            stats["skipped_unavailable"],
            stats["skipped_parse_error"],
            self.settings.vlm_shadow_mode,
        )
        return kept, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _verify_one_async(
        self,
        activity: Dict[str, Any],
        prompt: str,
        object_type: str = "",
    ) -> Dict[str, Any]:
        """Async per-activity verifier.

        Resilience layers, applied in order:

        1. **Circuit breaker**: if open, return ``SKIPPED_VLM_UNAVAILABLE``
           immediately — no HTTP attempt.
        2. **Single retry on connect/timeout**: 0.5s backoff, then record
           failure on the second miss.
        3. **Tolerant body parsing**: malformed JSON / missing keys map to
           ``PARSE_ERROR`` — never an unhandled exception.

        The caller (``verify_activities_async``) wraps this in another
        try/except so any leaked exception still produces a fail-open
        ``SKIPPED_VLM_UNAVAILABLE`` review block.
        """
        # 0) Image prep — pure CPU, no network.  Wrapped in try because
        #    cv2/np can fail in interesting ways.
        try:
            keyframes = _resolve_keyframes(activity)
            if not keyframes:
                return {
                    "status": "SKIPPED_NO_IMAGE",
                    "verdict": None,
                    "image_path": activity.get("activityImage") or "",
                }

            # Pre-VLM gate runs on the ORIGINAL Pipeline-1-rendered keyframes
            # ONLY (before any clip-supplementation, which adds raw frames
            # without overlays). The gate makes two checks against the
            # rendered bbox overlays:
            #
            #   1. No-subject gate (always for non-skipped types): every
            #      keyframe lacks a person bbox → activity is bogus, drop.
            #      Catches empty-cabin hallucinations.
            #
            #   2. No-object gate (writing/eating/packing/cell_phone only):
            #      every keyframe lacks the orange/yellow target-object
            #      bbox → Pipeline-1 fired the rule on stale state, drop.
            #      Catches the "writing without book" / "eating without cup"
            #      stale-trigger archetype.
            #
            # Both gates require ``with_any_bbox > 0`` somewhere in the
            # original keyframes to confirm overlay rendering is active —
            # otherwise we fall through to the VLM rather than risk
            # mis-dropping a run with rendering disabled.
            pre_gate_enabled = bool(
                getattr(self.settings, "vlm_pre_gate_enabled", True)
            )
            if pre_gate_enabled and object_type not in _PRE_GATE_SKIP_OBJECT_TYPES:
                counts = _count_bboxes_in_keyframes(
                    keyframes,
                    min_person_area=int(
                        getattr(self.settings, "vlm_pre_gate_min_person_area", 1000)
                    ),
                )
                rendering_active = counts["with_any_bbox"] > 0

                if rendering_active and counts["with_person"] == 0:
                    logger.info(
                        "[vlm] PRE-GATE DROP (no_subject) activity type=%s at t=%s "
                        "(0/%d keyframes have a person bbox; %d had non-person bboxes)",
                        activity.get("activityType"),
                        activity.get("activityStartTime"),
                        counts["total"], counts["with_any_bbox"],
                    )
                    return {
                        "status": "PRE_GATE_DROP_NO_SUBJECT",
                        "verdict": {
                            "verdict": "FALSE_POSITIVE",
                            "confidence": 1.0,
                            "reasoning": (
                                f"pre-VLM gate: 0/{counts['total']} original keyframes "
                                f"contain a person bbox (rendering confirmed active by "
                                f"{counts['with_any_bbox']} non-person bboxes)"
                            ),
                        },
                        "model": self.settings.vlm_model,
                        "frames_sent": counts["total"],
                        "latency_sec": 0.0,
                        "pre_gate_counts": counts,
                    }

                if (
                    rendering_active
                    and object_type in _OBJECT_REQUIRED_TYPES
                    and counts["with_object"] == 0
                ):
                    logger.info(
                        "[vlm] PRE-GATE DROP (no_object) activity type=%s at t=%s "
                        "(0/%d keyframes have a target-object bbox; "
                        "person bbox count=%d)",
                        activity.get("activityType"),
                        activity.get("activityStartTime"),
                        counts["total"], counts["with_person"],
                    )
                    return {
                        "status": "PRE_GATE_DROP_NO_OBJECT",
                        "verdict": {
                            "verdict": "FALSE_POSITIVE",
                            "confidence": 1.0,
                            "reasoning": (
                                f"pre-VLM gate: 0/{counts['total']} original keyframes "
                                f"contain a target-object bbox for activity {object_type!r} "
                                f"(person bboxes present in {counts['with_person']} frames, "
                                f"so this is a stale-state trigger, not a missing render)"
                            ),
                        },
                        "model": self.settings.vlm_model,
                        "frames_sent": counts["total"],
                        "latency_sec": 0.0,
                        "pre_gate_counts": counts,
                    }

            # Honor the long-promised vlm_strip_target_frames setting:
            # for single-burst activities we sample the activityClip to
            # reach the configured target so the VLM gets temporal
            # evidence rather than a single frozen instant. Done AFTER
            # the pre-gate so supplementary raw frames don't defeat
            # the bbox-overlay-based gate decision. Cap is 5 (matches
            # _stitch_keyframes slice).
            target_n = min(int(getattr(self.settings, "vlm_strip_target_frames", 5) or 5), 5)
            if len(keyframes) < target_n:
                keyframes = _supplement_keyframes_from_clip(
                    activity, keyframes, target_n=target_n,
                )

            crop_to_roi = object_type not in _FULL_FRAME_OBJECT_TYPES
            strip_bytes = _stitch_keyframes(keyframes, crop_to_roi=crop_to_roi)
            if not strip_bytes:
                return {
                    "status": "SKIPPED_NO_IMAGE",
                    "verdict": None,
                    "image_path": str(keyframes[0]),
                }
            b64 = base64.b64encode(strip_bytes).decode("ascii")
            n_frames = len(keyframes)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[vlm] image-prep failure for activity type=%s: %s",
                activity.get("activityType"), e,
            )
            return {
                "status": "SKIPPED_VLM_UNAVAILABLE",
                "verdict": "SKIPPED_VLM_UNAVAILABLE",
                "reason": type(e).__name__,
                "error": str(e)[:200],
            }

        # 1) Circuit breaker — short-circuit before opening a socket.
        if self._breaker.is_open():
            return {
                "status": "SKIPPED_VLM_UNAVAILABLE",
                "verdict": "SKIPPED_VLM_UNAVAILABLE",
                "reason": "circuit_breaker_open",
                "latency_sec": 0.0,
            }

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
        client = self._get_async_client(self.settings.vlm_timeout_seconds)
        t0 = time.time()

        # 2) HTTP with single retry.  Treat connect/timeout errors as
        #    candidates for retry; treat HTTP 5xx as transient too.
        last_exc: Optional[BaseException] = None
        body: Optional[Dict[str, Any]] = None
        for attempt in (1, 2):
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                # Non-2xx: treat 5xx as transient (retry once); 4xx as fatal.
                if resp.status_code >= 500 and attempt == 1:
                    last_exc = httpx.HTTPStatusError(
                        f"{resp.status_code} server error",
                        request=resp.request, response=resp,
                    )
                    await asyncio.sleep(0.5)
                    continue
                if resp.status_code >= 400:
                    last_exc = httpx.HTTPStatusError(
                        f"{resp.status_code} {resp.reason_phrase}",
                        request=resp.request, response=resp,
                    )
                    break
                # Decode JSON; malformed body must NOT crash the loop.
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError) as decode_exc:
                    latency = round(time.time() - t0, 3)
                    logger.warning(
                        "[vlm] malformed JSON from vLLM (status=%s): %s",
                        resp.status_code, decode_exc,
                    )
                    # Malformed JSON is a parse error, not a circuit-breaker
                    # event — vLLM is up, just confused.  Fail-open via
                    # PARSE_ERROR (verify_activities keeps the activity).
                    return {
                        "status": "PARSE_ERROR",
                        "verdict": None,
                        "error": f"json_decode: {decode_exc}",
                        "latency_sec": latency,
                    }
                last_exc = None
                break
            except (httpx.ConnectError, httpx.ConnectTimeout,
                    httpx.ReadTimeout, httpx.WriteTimeout,
                    httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == 1:
                    await asyncio.sleep(0.5)
                    continue
                break
            except httpx.HTTPError as exc:
                # Other HTTP errors — single attempt, no retry.
                last_exc = exc
                break

        if body is None:
            # Failure path: record breaker hit and fail open.
            self._breaker.record_failure()
            latency = round(time.time() - t0, 3)
            logger.warning(
                "[vlm] endpoint unavailable for activity type=%s at t=%s: %s",
                activity.get("activityType"),
                activity.get("activityStartTime"),
                last_exc,
            )
            return {
                "status": "SKIPPED_VLM_UNAVAILABLE",
                "verdict": "SKIPPED_VLM_UNAVAILABLE",
                "reason": type(last_exc).__name__ if last_exc else "unknown",
                "error": str(last_exc)[:200] if last_exc else "",
                "latency_sec": latency,
            }

        # 3) Success path — record and parse.
        self._breaker.record_success()
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

        # Verdict enum validation — refuse unknown labels (a jail-broken
        # VLM emitting "DROP" or similar would otherwise sneak through).
        v = parsed.get("verdict")
        if v not in _ALLOWED_VERDICTS:
            return {
                "status": "PARSE_ERROR",
                "verdict": parsed,
                "error": f"verdict not in enum: {v!r}",
                "latency_sec": latency,
            }

        # Post-VLM structured-field consistency check. When the VLM emits
        # TRUE_POSITIVE but its own observation fields contradict (e.g.
        # writing TP with hand_actually_on_book=false, or cell_phone TP
        # with object_in_hand="radio_handset"), demote to UNCERTAIN with
        # capped confidence 0.5. The structural counterpart to the
        # prompt-level "do not default to TRUE_POSITIVE" rule.
        consistency_override: Optional[Dict[str, Any]] = None
        if getattr(self.settings, "vlm_consistency_check_enabled", True):
            inconsistency = _consistency_check(parsed, object_type)
            if inconsistency:
                original_verdict = parsed.get("verdict")
                try:
                    original_conf = float(parsed.get("confidence", 0.5))
                except (TypeError, ValueError):
                    original_conf = 0.5
                parsed["verdict"] = "UNCERTAIN"
                parsed["confidence"] = min(original_conf, 0.5)
                consistency_override = {
                    "original_verdict": original_verdict,
                    "original_confidence": original_conf,
                    "reason": inconsistency,
                }
                parsed["consistency_override"] = consistency_override
                logger.info(
                    "[vlm] CONSISTENCY OVERRIDE activity type=%s at t=%s: "
                    "%s -> UNCERTAIN (%s)",
                    activity.get("activityType"),
                    activity.get("activityStartTime"),
                    original_verdict,
                    inconsistency,
                )

        return {
            "status": "OK",
            "verdict": parsed,
            "latency_sec": latency,
            "model": self.settings.vlm_model,
            "frames_sent": n_frames,
        }

    # ------------------------------------------------------------------
    # Backwards-compat sync shim (used by existing tests & any caller that
    # held a reference to the old name).  Just runs the async version.
    # ------------------------------------------------------------------
    def _verify_one(
        self,
        activity: Dict[str, Any],
        prompt: str,
        object_type: str = "",
    ) -> Dict[str, Any]:
        """Synchronous wrapper around :meth:`_verify_one_async`.

        Retained for backwards compatibility with any caller that pokes the
        internal method directly.  New code should use the async variant.
        Same already-running-loop dance as :meth:`verify_activities`.
        """
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            return asyncio.run(self._verify_one_async(activity, prompt, object_type))

        result_box: List[Any] = []

        def _thread_main() -> None:
            try:
                result_box.append(
                    asyncio.run(self._verify_one_async(activity, prompt, object_type))
                )
            except BaseException as e:  # noqa: BLE001
                result_box.append(e)

        t = threading.Thread(target=_thread_main, daemon=True, name="vlm-verify-one")
        t.start()
        t.join()
        out = result_box[0]
        if isinstance(out, BaseException):
            raise out
        return out


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
