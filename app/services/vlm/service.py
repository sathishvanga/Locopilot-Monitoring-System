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
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...utils.config import get_settings
from ...utils.logger import get_logger

# Re-imports — these helpers used to live in this module.  They have been
# moved to sibling sub-modules for readability, but the original
# ``_verify_one_async`` body references them as bare module-level names,
# so we bring them back into this module's namespace verbatim.  Tests
# also monkeypatch ``vlm_verification_service._resolve_keyframes`` etc;
# the shim at ``app/services/vlm_verification_service.py`` aliases this
# module in ``sys.modules`` so those patches land here.
from .image_encoder import (  # noqa: F401  (used as module globals)
    _encode_image,
    _detect_roi,
    _crop_to_roi,
)
from .keyframe_processor import (  # noqa: F401
    _resolve_keyframes,
    _supplement_keyframes_from_clip,
    _count_bboxes_in_keyframes,
    _stitch_keyframes,
    _FULL_FRAME_OBJECT_TYPES,
    _PRE_GATE_SKIP_OBJECT_TYPES,
    _OBJECT_REQUIRED_TYPES,
)
from .vlm_client import _CircuitBreaker  # noqa: F401
from .verdict_parser import (  # noqa: F401
    _Calibrator,
    _calibrator,
    _calibrator_lock,
    _get_calibrator,
    _jsonl_locks,
    _jsonl_locks_guard,
    _jsonl_lock_for,
    _append_jsonl,
    _ALLOWED_VERDICTS,
    _PROMPT_WRITING,
    _PROMPT_EATING,
    _PROMPT_PACKING,
    _PROMPT_CELL_PHONE,
    _PROMPT_SLEEP,
    _PROMPT_MIND_DIVERSION,
    _PROMPT_NO_PERSON,
    _PROMPT_GROUP,
    _PROMPTS_BY_OBJECT_TYPE,
    _consistency_check,
    _HARD_STOPPED_CUE_RE,
    _NEGATION_RE,
    _CLAUSE_SPLIT_RE,
    _has_hard_stopped_cue,
    _parse_verdict,
    _safe_motion_state,
)


logger = get_logger(__name__)


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
    # Merged-activity sub-type helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_object_type(raw: Any) -> str:
        return (raw or "").strip().lower().replace(" ", "_") if isinstance(raw, str) else ""

    def _extract_subtype_views(
        self, activity: Dict[str, Any]
    ) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """For a ``_isCombined`` activity, return one VLM view per distinct
        verifiable sub-type.

        Concurrent-grouping merges co-occurring detections (e.g. writing+phone
        in the same minute bucket) into a single posted violation under one
        primary ``objectType``. The verifier historically only checked the
        primary type, so a real ``writing`` event would protect a co-merged
        ``cell_phone`` FP from being filtered.

        IMPORTANT — each sub-view inherits the parent's FULL ``_sourceActivities``
        list, not a sub-type-filtered subset. Rationale: the classical
        detector frequently mis-tags the best-evidence frame (e.g. the YOLO
        ``cell phone`` confusion on white logbook papers tagged the most
        clearly-writing frame as ``cell_phone``). Filtering keyframes by the
        Pipeline-1 tag would hide that evidence from the writing-prompt VLM
        call. Sending all keyframes lets each sub-type's prompt examine the
        full burst — the prompts are already written to answer "did this
        happen in any frame?", which is the correct semantic across the
        merged window.

        Returns an empty list when the activity is not combined, only one
        distinct verifiable sub-type is present, or the parallel
        ``objectTypes`` / ``activityTypes`` arrays are missing.
        """
        if not activity.get("_isCombined"):
            return []
        object_types = activity.get("objectTypes") or []
        activity_types = activity.get("activityTypes") or []
        if not object_types or len(object_types) != len(activity_types):
            return []

        # De-duplicate while preserving order; non-verifiable sub-types are
        # skipped (they pass through unchanged at the parent level).
        seen: set = set()
        views: List[Tuple[str, Any, Dict[str, Any]]] = []
        descriptions = activity.get("descriptions") or []
        for i, raw_obj in enumerate(object_types):
            sub_obj = self._normalise_object_type(raw_obj)
            if not sub_obj or sub_obj in seen:
                continue
            if sub_obj not in self._verify_set or sub_obj not in _PROMPTS_BY_OBJECT_TYPE:
                continue
            sub_act_type = activity_types[i]
            seen.add(sub_obj)

            # Shallow copy: only override the singular fields (objectType /
            # activityType / des). _sourceActivities is intentionally NOT
            # filtered — see the docstring rationale.
            view = dict(activity)
            view["objectType"] = sub_obj
            view["activityType"] = sub_act_type
            try:
                view["des"] = descriptions[i] if i < len(descriptions) else activity.get("des")
            except Exception:  # noqa: BLE001
                pass
            views.append((sub_obj, sub_act_type, view))

        # Only treat as multi-sub when at least 2 distinct verifiable
        # sub-types exist; a single sub-type falls through to the regular
        # single-entry path so we avoid the aggregation overhead.
        if len(views) < 2:
            return []
        return views

    @staticmethod
    def _strip_subtypes_from_parent(
        parent: Dict[str, Any], drop_object_types: set, drop_activity_types: set
    ) -> None:
        """Remove `drop_object_types` from the parent's parallel sub-type
        arrays. Mutates in place.

        Keeps `_sourceClips`, `_sourceActivities`, `objectTypes`,
        `activityTypes`, `descriptions` aligned. Updates singular
        `objectType` / `activityType` / `des` to the first surviving
        sub-type so downstream code (which reads the singular fields)
        still sees a coherent activity.

        ``activityClip`` and ``activityImage`` are intentionally NOT
        regenerated — the merged clip already exists on disk and the
        partial-strip case is rare enough that re-encoding per drop would
        cost more than the visual artefact of a slightly-too-long clip.
        """
        old_obj = parent.get("objectTypes") or []
        old_act = parent.get("activityTypes") or []
        old_desc = parent.get("descriptions") or []

        keep_obj: List[Any] = []
        keep_act: List[Any] = []
        keep_desc: List[Any] = []
        for i in range(min(len(old_obj), len(old_act))):
            sub_obj_norm = (
                VlmVerificationService._normalise_object_type(old_obj[i])
            )
            if sub_obj_norm in drop_object_types:
                continue
            keep_obj.append(old_obj[i])
            keep_act.append(old_act[i])
            if i < len(old_desc):
                keep_desc.append(old_desc[i])

        parent["objectTypes"] = keep_obj
        parent["activityTypes"] = keep_act
        parent["descriptions"] = keep_desc

        # Filter _sourceActivities and _sourceClips by the dropped numeric
        # activity types. _sourceClips is a flat list of paths whose
        # filenames contain the activity-type slug (e.g. "_cell_phone_") —
        # but the authoritative join key is _sourceActivities[*].clip, so we
        # rebuild _sourceClips from the surviving _sourceActivities to keep
        # the two lists consistent.
        old_src = parent.get("_sourceActivities") or []
        new_src = [s for s in old_src if s.get("activityType") not in drop_activity_types]
        parent["_sourceActivities"] = new_src
        if "_sourceClips" in parent:
            parent["_sourceClips"] = [s.get("clip") for s in new_src if s.get("clip")]

        # Update singular fields to the dominant survivor.
        if keep_obj and keep_act:
            parent["objectType"] = keep_obj[0]
            parent["activityType"] = keep_act[0]
            if keep_desc:
                # Re-derive `des` from the surviving description list, joined
                # with "; " to match how merged activities are originally
                # written by concurrent_activity_grouping_service.
                parent["des"] = "; ".join(str(d) for d in keep_desc)

    def _aggregate_subtype_results(
        self,
        parent: Dict[str, Any],
        subtype_results: List[Tuple[str, Any, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Aggregate per-sub-type reviews into a single pseudo-review.

        ``subtype_results`` is a list of ``(sub_object_type, sub_activity_type,
        review)`` tuples — one per sub-type the verifier ran. The aggregate
        feeds the existing single-result Phase-3 logic (drop / motion-override
        / telemetry) so we don't duplicate that branch for the multi-sub case.

        Drop policy:
          - For each sub-type, mark drop if status==OK AND verdict==FALSE_POSITIVE
            AND confidence >= drop_threshold. SKIPPED/PARSE_ERROR sub-types
            never count as drops (fail-open).
          - If ALL OK sub-types are drops → aggregate verdict=FALSE_POSITIVE
            with the lowest contributing confidence, so the existing Phase-3
            drop path drops the parent.
          - If SOME drops → strip those sub-types from the parent in place;
            aggregate verdict=TRUE_POSITIVE with the highest surviving
            confidence so the parent is kept.
          - If NO drops → aggregate verdict=TRUE_POSITIVE with the highest
            confidence among surviving sub-types.

        Motion observations are propagated by taking the FIRST sub-type whose
        review observed a STOPPED cue (so the existing motion-override path
        can flip motionState). If none observed STOPPED, leave empty.
        """
        # Collect each sub-type's verdict block; SKIPPED entries don't get a
        # verdict but are still reported in the audit blob.
        per_subtype_audit: Dict[str, Dict[str, Any]] = {}
        ok_subtypes: List[Tuple[str, Any, Dict[str, Any]]] = []
        any_skipped_unavailable = False
        any_parse_error = False
        latency_total = 0.0
        frames_total = 0

        for sub_obj, sub_act, review in subtype_results:
            per_subtype_audit[sub_obj] = review
            try:
                latency_total += float(review.get("latency_sec") or 0.0)
                frames_total += int(review.get("frames_sent") or 0)
            except (TypeError, ValueError):
                pass
            status = review.get("status")
            verdict_dict = review.get("verdict") or {}
            v = (verdict_dict.get("verdict") if isinstance(verdict_dict, dict)
                 else None)
            try:
                conf_log = float(verdict_dict.get("confidence") or 0.0) if isinstance(verdict_dict, dict) else 0.0
            except (TypeError, ValueError):
                conf_log = 0.0
            logger.info(
                "[vlm] SUBTYPE VERDICT: parent activity=%s sub=%s status=%s "
                "verdict=%s conf=%.2f",
                parent.get("activityType"), sub_obj, status, v, conf_log,
            )
            if status == "OK":
                ok_subtypes.append((sub_obj, sub_act, review))
            elif status == "SKIPPED_VLM_UNAVAILABLE":
                any_skipped_unavailable = True
            elif status == "PARSE_ERROR":
                any_parse_error = True
            elif status in ("PRE_GATE_DROP_NO_SUBJECT", "PRE_GATE_DROP_NO_OBJECT"):
                # Treat pre-gate drops the same as a confident FP for this
                # sub-type — they are deterministic FP signals.
                ok_subtypes.append((sub_obj, sub_act, review))

        # If nothing produced an actionable verdict, fail open.
        if not ok_subtypes:
            if any_skipped_unavailable:
                status_out = "SKIPPED_VLM_UNAVAILABLE"
            elif any_parse_error:
                status_out = "PARSE_ERROR"
            else:
                status_out = "SKIPPED_NO_IMAGE"
            return {
                "status": status_out,
                "verdict": None,
                "latency_sec": round(latency_total, 3),
                "frames_sent": frames_total,
                "model": self.settings.vlm_model,
                "subtype_reviews": per_subtype_audit,
            }

        # Decide which sub-types to drop.
        threshold = float(self.settings.vlm_drop_threshold)
        drop_object_types: set = set()
        drop_activity_types: set = set()
        survivor_max_conf = 0.0
        worst_drop_conf = 1.0
        survivor_uncertain_present = False
        propagated_train_state: Optional[str] = None
        propagated_motion_evidence: Optional[str] = None

        for sub_obj, sub_act, review in ok_subtypes:
            verdict_dict = review.get("verdict") or {}
            v = verdict_dict.get("verdict")
            try:
                conf = float(verdict_dict.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            # Pre-gate drops carry confidence=1.0 verdict=FALSE_POSITIVE
            # already, so the same predicate handles both paths.
            is_drop = (v == "FALSE_POSITIVE" and conf >= threshold)
            if is_drop:
                drop_object_types.add(sub_obj)
                drop_activity_types.add(sub_act)
                worst_drop_conf = min(worst_drop_conf, conf)
                logger.info(
                    "[vlm] SUBTYPE DROP candidate: parent activity=%s sub=%s "
                    "verdict=%s conf=%.2f reason=%r",
                    parent.get("activityType"),
                    sub_obj,
                    v,
                    conf,
                    (verdict_dict.get("reasoning") or "")[:120],
                )
            else:
                survivor_max_conf = max(survivor_max_conf, conf)
                if v == "UNCERTAIN":
                    survivor_uncertain_present = True
                # Capture motion observation from a survivor for parent override.
                if propagated_train_state is None:
                    train_obs = (verdict_dict.get("train_appears_to_be") or "").strip().lower()
                    if train_obs:
                        propagated_train_state = train_obs
                        propagated_motion_evidence = (
                            str(verdict_dict.get("motion_evidence") or "")
                        )

        all_dropped = (len(drop_object_types) == len(ok_subtypes))

        # Shadow mode: record per-sub-type verdicts for audit but never
        # strip or drop. Pipeline-1's verdict is authoritative; the
        # verifier is observe-only until the operator flips
        # vlm_shadow_mode=False. Mirrors the single-result shadow-mode
        # contract enforced by Phase-3's `should_drop` predicate.
        if self.settings.vlm_shadow_mode:
            agg_verdict = {
                "verdict": "TRUE_POSITIVE",
                "confidence": max(survivor_max_conf, worst_drop_conf if worst_drop_conf < 1.0 else 0.0),
                "reasoning": (
                    "shadow mode: per-sub-type verdicts recorded, no drops"
                ),
                "subtypes_dropped_in_enforcement": sorted(drop_object_types),
            }
            if propagated_train_state:
                agg_verdict["train_appears_to_be"] = propagated_train_state
                agg_verdict["motion_evidence"] = propagated_motion_evidence or ""
            return {
                "status": "OK",
                "verdict": agg_verdict,
                "latency_sec": round(latency_total, 3),
                "frames_sent": frames_total,
                "model": self.settings.vlm_model,
                "subtype_reviews": per_subtype_audit,
                "subtypes_dropped": [],
                "subtypes_kept": sorted(set(s for s, _, _ in ok_subtypes)),
            }

        if all_dropped:
            # Aggregate verdict drives the existing single-result drop path.
            agg_verdict = {
                "verdict": "FALSE_POSITIVE",
                "confidence": worst_drop_conf,
                "reasoning": (
                    "all verifiable sub-types ({}) returned FALSE_POSITIVE "
                    "with confidence >= {:.2f}; merged activity is fully a FP"
                ).format(",".join(sorted(drop_object_types)), threshold),
                "subtypes_dropped": sorted(drop_object_types),
            }
            if propagated_train_state:
                agg_verdict["train_appears_to_be"] = propagated_train_state
                agg_verdict["motion_evidence"] = propagated_motion_evidence or ""
            return {
                "status": "OK",
                "verdict": agg_verdict,
                "latency_sec": round(latency_total, 3),
                "frames_sent": frames_total,
                "model": self.settings.vlm_model,
                "subtype_reviews": per_subtype_audit,
                "subtypes_dropped": sorted(drop_object_types),
                "subtypes_kept": [],
            }

        # Partial drop (or no drops). Strip the FP sub-types from the parent.
        if drop_object_types:
            self._strip_subtypes_from_parent(
                parent, drop_object_types, drop_activity_types
            )
            logger.info(
                "[vlm] SUBTYPE STRIP: parent now objectTypes=%s "
                "activityTypes=%s (stripped: %s)",
                parent.get("objectTypes"),
                parent.get("activityTypes"),
                sorted(drop_object_types),
            )

        agg_v = "UNCERTAIN" if survivor_uncertain_present else "TRUE_POSITIVE"
        agg_verdict = {
            "verdict": agg_v,
            "confidence": max(survivor_max_conf, 0.0),
            "reasoning": (
                "merged activity verified per sub-type; survivors=[{}] dropped=[{}]"
            ).format(
                ",".join(sorted(set(s for s, _, _ in ok_subtypes) - drop_object_types)),
                ",".join(sorted(drop_object_types)) or "none",
            ),
            "subtypes_dropped": sorted(drop_object_types),
        }
        if propagated_train_state:
            agg_verdict["train_appears_to_be"] = propagated_train_state
            agg_verdict["motion_evidence"] = propagated_motion_evidence or ""
        return {
            "status": "OK",
            "verdict": agg_verdict,
            "latency_sec": round(latency_total, 3),
            "frames_sent": frames_total,
            "model": self.settings.vlm_model,
            "subtype_reviews": per_subtype_audit,
            "subtypes_dropped": sorted(drop_object_types),
            "subtypes_kept": sorted(
                set(s for s, _, _ in ok_subtypes) - drop_object_types
            ),
        }

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
        # `to_verify` collects (index_in_kept, activity, prompt, object_type,
        #                       parent_kept_idx, sub_object_type, sub_act_type)
        # for each VLM dispatch. For single-sub activities sub_object_type ==
        # object_type and parent_kept_idx == index_in_kept; for multi-sub
        # (concurrent-merged) activities we generate one entry per sub-type
        # and aggregate after gather(). Aggregation keys off `subtype_groups`.
        kept: List[Dict[str, Any]] = []
        to_verify: List[Tuple[int, Dict[str, Any], str, str, int, str, Any]] = []
        # Map parent_kept_idx -> ordered list of sub-types being verified for
        # that parent. A parent appears here only when len(views) >= 2.
        subtype_groups: Dict[int, List[Tuple[str, Any]]] = {}
        verified_count = 0

        for act in activities:
            try:
                object_type = (act.get("objectType") or "").strip().lower().replace(" ", "_")
                # An activity is "verifiable" if its primary objectType OR any
                # of its concurrent-merged sub-types match the verify set.
                # `_extract_subtype_views` returns [] when the activity has
                # only a single verifiable sub-type — primary path covers
                # that case. When primary is non-verifiable but a sub-type
                # IS verifiable (rare but possible after a future change to
                # concurrent grouping picking the wrong primary), the
                # sub-type path still kicks in.
                subtype_views = self._extract_subtype_views(act)
                primary_verifiable = (
                    object_type in self._verify_set
                    and object_type in _PROMPTS_BY_OBJECT_TYPE
                )
                if not primary_verifiable and not subtype_views:
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
                kept_idx = len(kept)
                kept.append(act)
                if subtype_views:
                    # Multi-sub: one VLM call per distinct verifiable sub-type.
                    # Each call uses a sub-view (shallow copy of `act` with
                    # `_sourceActivities` filtered to that sub-type's clips)
                    # so `_resolve_keyframes` sees only the right frames.
                    subtype_groups[kept_idx] = [(s, a) for (s, a, _v) in subtype_views]
                    for sub_obj, sub_act_type, sub_view in subtype_views:
                        prompt = _PROMPTS_BY_OBJECT_TYPE[sub_obj]
                        to_verify.append(
                            (kept_idx, sub_view, prompt, sub_obj, kept_idx, sub_obj, sub_act_type)
                        )
                        verified_count += 1
                else:
                    prompt = _PROMPTS_BY_OBJECT_TYPE[object_type]
                    to_verify.append(
                        (kept_idx, act, prompt, object_type, kept_idx, object_type, act.get("activityType"))
                    )
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

            async def _run_one(
                idx: int, activity: Dict[str, Any], prompt: str, obj_type: str,
                parent_kept_idx: int, sub_obj: str, sub_act: Any,
            ) -> Tuple[int, str, Any, Dict[str, Any]]:
                try:
                    async with sem:
                        review = await self._verify_one_async(activity, prompt, obj_type)
                    return parent_kept_idx, sub_obj, sub_act, review
                except Exception as e:  # noqa: BLE001
                    # CRITICAL: never let a single activity's failure
                    # propagate up and truncate the kept list.
                    logger.exception(
                        "[vlm] verify_one failed for activity id=%s type=%s sub=%s",
                        activity.get("id"), activity.get("activityType"), sub_obj,
                    )
                    return parent_kept_idx, sub_obj, sub_act, {
                        "status": "SKIPPED_VLM_UNAVAILABLE",
                        "verdict": "SKIPPED_VLM_UNAVAILABLE",
                        "reason": type(e).__name__,
                        "error": str(e)[:200],
                    }

            sub_results = await asyncio.gather(
                *(_run_one(i, a, p, o, p_idx, s_obj, s_act)
                  for (i, a, p, o, p_idx, s_obj, s_act) in to_verify),
                return_exceptions=False,
            )

            # Aggregate sub-type results into the (kept_idx, review) shape
            # the existing Phase-3 logic expects. For multi-sub parents,
            # `_aggregate_subtype_results` may mutate the parent activity in
            # `kept` to strip FP sub-types; the resulting pseudo-review then
            # carries the aggregate verdict that drives drop / motion-override
            # / telemetry uniformly with the single-sub path.
            grouped: Dict[int, List[Tuple[str, Any, Dict[str, Any]]]] = {}
            for parent_kept_idx, sub_obj, sub_act, review in sub_results:
                grouped.setdefault(parent_kept_idx, []).append(
                    (sub_obj, sub_act, review)
                )
            results: List[Tuple[int, Dict[str, Any]]] = []
            for parent_kept_idx, items in grouped.items():
                if parent_kept_idx in subtype_groups and len(items) >= 2:
                    aggregate = self._aggregate_subtype_results(
                        kept[parent_kept_idx], items
                    )
                    results.append((parent_kept_idx, aggregate))
                else:
                    # Single-sub: pass through verbatim.
                    results.append((parent_kept_idx, items[0][2]))

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
            strip_bytes = _stitch_keyframes(
                keyframes, crop_to_roi=crop_to_roi,
            )
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
            # 400 tokens: with frame_observations removed from the writing
            # schema, the worst-case response is ~250 tokens (verdict +
            # ~12 structured fields + reasoning + motion_evidence). 400
            # leaves comfortable headroom and stays well under the
            # 3072 - 2563 = 509 vLLM budget for the 5-keyframe writing
            # case (the tightest input). Shorter prompts (cell_phone,
            # sleep, etc.) all fit easily.
            "max_tokens": 400,
            "temperature": 0.0,
            # Constrain vLLM to emit a JSON object — eliminates ```code-fence
            # wrappers and stray prose that previously forced the tolerant
            # `_parse_verdict` to do extraction. Truncation still produces
            # an unparseable response, but the prompt-level reorder above
            # ensures the verdict block appears before any token cap is hit.
            "response_format": {"type": "json_object"},
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
            logger.warning(
                "[vlm] PARSE_ERROR for activity type=%s at t=%s object=%s: %s "
                "| raw_text_full=%r",
                activity.get("activityType"),
                activity.get("activityStartTime"),
                object_type,
                parsed.get("parse_error"),
                (text or "")[:1500],
            )
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
