"""
VLM Verification Service - Secondary verification using Qwen3-VL-8B

Uses a Vision-Language Model served via vLLM to verify detections from
the primary YOLO + Pose pipeline. Reduces false positives by providing
semantic understanding of the scene.

Train motion state is handled by trip schedule data, not VLM.
"""

import base64
import concurrent.futures
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Activity-specific verification prompts
# Each prompt instructs the VLM to verify whether the flagged activity is genuine
# ============================================================

ACTIVITY_PROMPTS = {
    "cell_phone": """Flagged: mobile phone usage.
TRUE: phone/smartphone clearly visible in hand.
FALSE: radio handset, walkie-talkie, control device, clipboard, reflection, shiny object, or hands not visible.""",

    "writing": """Flagged: writing or reading a document.
TRUE: hands actively engaged with a visible book, paper, or clipboard.
FALSE: hands on controls or resting in lap with no document visible. Eyes closed or head slumped = sleeping, not writing.""",

    "packing_bags": """Flagged: handling a bag/backpack.
TRUE: hands directly gripping, lifting, reaching into, or zipping a bag.
FALSE: bag nearby but not being touched, hands on lap/controls/papers near a bag, bag on floor in background.""",

    "eating_drinking": """Flagged: eating or drinking.
TRUE: cup, bottle, or food item held in hand near face.
FALSE: bottle in holder not being used, objects on shelf/dashboard not being held.""",

    "sleeping": """Flagged: operator sleeping.
TRUE: eyes closed, head slumped/drooping, body limp or slouched with no control engagement. If face not visible, judge by posture.
FALSE: actively looking at instruments with hands engaged, brief blinks.""",

    "microsleep": """Flagged: drowsiness or microsleep.
TRUE: any sign of reduced alertness — head nodding/drooping, eyes heavy or closed, slouched posture with no control engagement.
FALSE: sitting upright and alert, actively operating controls or reading instruments.""",

    "mind_diversion": """Flagged: attention diverted from track.
TRUE: clearly looking away from forward direction with no operational purpose.
FALSE: checking mirrors, signals, track infrastructure, control panel instruments, or natural head scanning.""",

    "group_detected": """Flagged: 3+ people in cab.
TRUE: 3 or more clearly distinct individuals visible.
FALSE: reflections, shadows, or partially visible figures — count only clearly visible people.""",

    "no_person_detected": """Flagged: no person in cab.
TRUE: cab is genuinely empty with no person visible.
FALSE: any part of a person (arm, leg, torso) is visible. If image is dark/unclear, return false.""",

    "lp_hand_gesture": """Flagged: LP giving hand signal but ALP not reciprocating.
Context: Indian Railways crew must exchange hand signals at signals/stations — one raises hand, other must acknowledge.
TRUE: one person has a FREE hand raised in air (not touching anything) and the other person's hands are down/on controls.
FALSE: both hands raised (coordination OK), neither signaling, or arm movement involves touching controls/switches/equipment. Operating overhead switches or controls is NOT a hand signal.""",

    "alp_hand_gesture": """Flagged: ALP giving hand signal but LP not reciprocating.
Context: Indian Railways crew must exchange hand signals at signals/stations — one raises hand, other must acknowledge.
TRUE: one person has a FREE hand raised in air (not touching anything) and the other person's hands are down/on controls.
FALSE: both hands raised (coordination OK), neither signaling, or arm movement involves touching controls/switches/equipment. Operating overhead switches or controls is NOT a hand signal.""",

    "alp_not_standing": """Flagged: ALP not standing during pre-arrival.
TRUE: ALP is clearly seated or crouching.
FALSE: ALP appears standing upright, or posture is unclear.""",
}

VERIFICATION_PROMPT_TEMPLATE = """Locomotive cab safety monitor. Analyze the image.
{activity_prompt}

Respond ONLY with JSON: {{"verified": true/false, "confidence": "high"/"medium"/"low", "reason": "brief explanation"}}"""


def build_verification_prompt(activity_type: str) -> str:
    """Build the verification prompt for a specific activity type."""
    activity_prompt = ACTIVITY_PROMPTS.get(
        activity_type,
        f"Flagged: {activity_type}. Evaluate whether this activity is genuinely occurring.",
    )
    return VERIFICATION_PROMPT_TEMPLATE.format(activity_prompt=activity_prompt)


class VLMVerificationService:
    """
    Synchronous VLM-based verification service using Qwen3-VL via vLLM.

    Verifies YOLO/Pose detections by sending cropped frame regions to
    the VLM for semantic analysis. Also determines train motion state.

    Uses a synchronous httpx.Client to avoid event-loop lifecycle issues
    in multiprocessing worker processes (spawned workers would orphan
    async clients across asyncio.run() calls).

    Features:
    - Synchronous HTTP calls to vLLM OpenAI-compatible API
    - Activity-specific verification prompts
    - Logprobs-based confidence extraction
    - Circuit breaker for VLM unavailability
    - Frame similarity caching to avoid redundant calls
    - ThreadPoolExecutor for concurrent batch verification
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8001/v1",
        model_name: str = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
        timeout: float = 10.0,
        max_retries: int = 2,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_sec: float = 60.0,
        cache_ttl_sec: float = 5.0,
        crop_padding: int = 200,
        jpeg_quality: int = 85,
        rejection_cooldown_sec: float = 3.0,
        ext_logger: Optional[logging.Logger] = None,
    ):
        self.vllm_base_url = vllm_base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.crop_padding = crop_padding
        self.jpeg_quality = jpeg_quality

        # Use caller-provided logger so spawned workers route to the right file
        self._log = ext_logger or logger

        # Synchronous HTTP client (created lazily, safe across calls)
        self._sync_client: Optional[httpx.Client] = None

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_reset_sec = circuit_breaker_reset_sec
        self._circuit_open_since: Optional[float] = None

        # Simple frame cache: hash -> (result, timestamp)
        self._cache: Dict[str, Tuple[Dict, float]] = {}
        self._cache_ttl = cache_ttl_sec

        # Rejection throttle: skip repeated VLM calls when the same
        # (activity_type, person_idx) was already rejected recently.
        # Key: "activity_type_pN" -> (result_dict, timestamp)
        self._rejection_throttle: Dict[str, Tuple[Dict, float]] = {}
        self._rejection_cooldown = rejection_cooldown_sec
        self._throttle_last_eviction: float = time.time()

        # Stats
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "throttle_hits": 0,
            "vlm_calls": 0,
            "vlm_failures": 0,
            "circuit_breaker_trips": 0,
            "verified_true": 0,
            "verified_false": 0,
        }

        self._log.info(
            f"[VLM] VLMVerificationService initialized: "
            f"url={vllm_base_url}, model={model_name}, timeout={timeout}s"
        )

    @property
    def sync_client(self) -> httpx.Client:
        """Lazy-initialize the synchronous HTTP client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.vllm_base_url,
                timeout=httpx.Timeout(self.timeout, connect=5.0),
            )
        return self._sync_client

    def close(self):
        """Close the HTTP client."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    # ─── Circuit Breaker ──────────────────────────────────────

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (VLM considered unavailable)."""
        if self._consecutive_failures < self._circuit_breaker_threshold:
            return False
        if self._circuit_open_since is None:
            self._circuit_open_since = time.time()
            self._stats["circuit_breaker_trips"] += 1
            self._log.warning("[VLM] Circuit breaker OPEN - VLM unavailable")
            return True
        # Check if reset period has elapsed
        if time.time() - self._circuit_open_since > self._circuit_breaker_reset_sec:
            self._log.info("[VLM] Circuit breaker HALF-OPEN - attempting reset")
            self._circuit_open_since = None
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0
        self._circuit_open_since = None

    def _record_failure(self):
        self._consecutive_failures += 1
        self._stats["vlm_failures"] += 1

    # ─── Frame Caching ────────────────────────────────────────

    def _compute_cache_key(self, frame_region: np.ndarray, activity_type: str) -> str:
        """Compute a simple cache key from downsampled frame region + activity type."""
        # Downsample to tiny size for fast hashing
        small = cv2.resize(frame_region, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        # Simple average hash
        avg = gray.mean()
        bits = (gray > avg).flatten()
        hash_str = "".join("1" if b else "0" for b in bits)
        return f"{activity_type}:{hash_str}"

    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if still valid."""
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                self._stats["cache_hits"] += 1
                return result
            del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: Dict):
        """Store result in cache."""
        self._cache[cache_key] = (result, time.time())
        self._log.debug(
            f"[VLM] Cache set: {result.get('activity_type', '?')} "
            f"verified={result.get('verified', '?')} (cache_size={len(self._cache)})"
        )
        # Evict old entries periodically
        if len(self._cache) > 100:
            cutoff = time.time() - self._cache_ttl
            self._cache = {
                k: (v, t) for k, (v, t) in self._cache.items() if t > cutoff
            }

    # ─── Frame Cropping ───────────────────────────────────────

    def _crop_detection_region(
        self, frame: np.ndarray, bbox: List[int], padding: Optional[int] = None
    ) -> np.ndarray:
        """Crop the detection region with padding, clamped to frame bounds."""
        pad = padding if padding is not None else self.crop_padding
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        return frame[y1:y2, x1:x2]

    def _encode_frame_b64(self, frame: np.ndarray) -> str:
        """JPEG-encode a frame and return base64 string."""
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return base64.b64encode(buffer).decode("utf-8")

    # ─── VLM API Call ─────────────────────────────────────────

    def _call_vlm(self, image_b64: str, activity_type: str) -> Dict:
        """Make a single synchronous VLM verification call to vLLM server."""
        prompt = build_verification_prompt(activity_type)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 100,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
        }

        self._log.debug(
            f"[VLM] Calling vLLM for {activity_type} "
            f"(prompt={len(prompt)} chars, model={self.model_name})"
        )
        response = self.sync_client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:200]
        self._log.debug(f"[VLM] Raw response for {activity_type}: {raw_content}")
        return data

    def _parse_vlm_response(self, response_data: Dict) -> Dict:
        """Parse VLM JSON response and extract confidence from logprobs."""
        result = {
            "train_moving": True,  # Always True - motion handled by trip schedule
            "verified": False,
            "confidence": "low",
            "reason": "parse_error",
            "vlm_confidence_score": 0.0,
            "source": "vlm",
        }

        try:
            content = response_data["choices"][0]["message"]["content"].strip()

            # Try to extract JSON from response (handle markdown code blocks)
            if "```" in content:
                # Extract JSON from code block
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]

            parsed = json.loads(content)
            result["verified"] = parsed.get("verified", False)
            result["confidence"] = parsed.get("confidence", "low")
            result["reason"] = parsed.get("reason", "no_reason")

            # Extract logprobs-based confidence for the verified field
            logprobs_confidence = self._extract_logprobs_confidence(response_data)
            if logprobs_confidence is not None:
                result["vlm_confidence_score"] = logprobs_confidence

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self._log.warning(f"[VLM] Failed to parse response: {e}")
            # Try to extract from raw text
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            result["reason"] = f"parse_error: {content[:100]}"

        return result

    def _extract_logprobs_confidence(self, response_data: Dict) -> Optional[float]:
        """Extract confidence from token-level logprobs on true/false tokens."""
        try:
            logprobs_data = response_data["choices"][0].get("logprobs", {})
            if not logprobs_data:
                return None

            content_logprobs = logprobs_data.get("content", [])
            for token_info in content_logprobs:
                token = token_info.get("token", "").strip().lower()
                if token in ("true", "false"):
                    logprob = token_info.get("logprob", -10)
                    return math.exp(logprob)
        except (KeyError, IndexError, TypeError):
            pass
        return None

    # ─── Public API ───────────────────────────────────────────

    def verify_detection_sync(
        self,
        frame: np.ndarray,
        bbox: List[int],
        activity_type: str,
        person_idx: int = 0,
    ) -> Tuple[bool, Dict]:
        """
        Verify a single detection using the VLM (synchronous).

        Args:
            frame: Full BGR frame
            bbox: Bounding box [x1, y1, x2, y2] of the detection
            activity_type: Activity type string (e.g., 'cell_phone')
            person_idx: Person index for logging

        Returns:
            Tuple of (is_confirmed, details_dict)
        """
        self._stats["total_requests"] += 1

        # Rejection throttle: if VLM already rejected this activity+person
        # recently, return the cached rejection without calling VLM again.
        throttle_key = f"{activity_type}_p{person_idx}"
        cooldown = self._rejection_cooldown
        now = time.time()
        if throttle_key in self._rejection_throttle:
            prev_result, prev_ts = self._rejection_throttle[throttle_key]
            if now - prev_ts < cooldown:
                self._stats["throttle_hits"] += 1
                self._log.debug(
                    f"[VLM] Throttled {throttle_key} "
                    f"({now - prev_ts:.1f}s since last rejection)"
                )
                return False, prev_result
            else:
                del self._rejection_throttle[throttle_key]

        # Periodic eviction of stale throttle entries (every 60s)
        if now - self._throttle_last_eviction > 60.0:
            self._throttle_last_eviction = now
            stale_keys = [
                k for k, (_, ts) in self._rejection_throttle.items()
                if now - ts > self._rejection_cooldown
            ]
            for k in stale_keys:
                del self._rejection_throttle[k]
            if stale_keys:
                self._log.debug(f"[VLM] Evicted {len(stale_keys)} stale throttle entries")

        # Circuit breaker check
        if self._is_circuit_open():
            result = self._fallback_result(activity_type, "circuit_breaker_open")
            return False, result

        # Crop detection region (activity-specific padding)
        cropped = self._crop_detection_region(frame, bbox)
        if cropped.size == 0:
            result = self._fallback_result(activity_type, "empty_crop")
            return False, result

        # Cache check
        cache_key = self._compute_cache_key(cropped, activity_type)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._log.debug(f"[VLM] Cache hit for {activity_type} person {person_idx}")
            is_confirmed = cached.get("train_moving", True) and cached.get("verified", False)
            return is_confirmed, cached

        # Encode and call VLM
        image_b64 = self._encode_frame_b64(cropped)
        self._stats["vlm_calls"] += 1

        for attempt in range(self.max_retries):
            try:
                raw_response = self._call_vlm(image_b64, activity_type)
                result = self._parse_vlm_response(raw_response)
                result["activity_type"] = activity_type
                result["person_idx"] = person_idx

                self._record_success()

                # Track stats
                if result["verified"]:
                    self._stats["verified_true"] += 1
                else:
                    self._stats["verified_false"] += 1

                # Cache the result
                self._set_cache(cache_key, result)

                self._log.info(
                    f"[VLM] {activity_type} p{person_idx}: "
                    f"moving={result['train_moving']}, "
                    f"verified={result['verified']}, "
                    f"conf={result['confidence']}, "
                    f"reason={result['reason']}"
                )

                is_confirmed = result["train_moving"] and result["verified"]

                # Throttle future calls if this was a rejection
                if not is_confirmed:
                    self._rejection_throttle[throttle_key] = (result, time.time())
                    self._log.debug(
                        f"[VLM] Rejection throttle set: {throttle_key} "
                        f"(cooldown={cooldown}s)"
                    )

                return is_confirmed, result

            except httpx.TimeoutException:
                self._log.warning(
                    f"[VLM] Timeout on attempt {attempt + 1}/{self.max_retries} "
                    f"for {activity_type} p{person_idx}"
                )
                self._record_failure()
            except httpx.HTTPStatusError as e:
                self._log.error(
                    f"[VLM] HTTP error {e.response.status_code} "
                    f"for {activity_type} p{person_idx}"
                )
                self._record_failure()
                break  # Don't retry on HTTP errors
            except Exception as e:
                self._log.error(
                    f"[VLM] Unexpected error for {activity_type} p{person_idx}: {e}"
                )
                self._record_failure()
                break

        result = self._fallback_result(activity_type, "vlm_error")
        return False, result

    def verify_batch_sync(
        self,
        frame: np.ndarray,
        activities: List[Dict],
        timestamp_sec: float = 0.0,
    ) -> Dict[str, Tuple[bool, Dict]]:
        """
        Verify a batch of activities from one frame using concurrent threads.

        Each activity gets its own HTTP request to vLLM. ThreadPoolExecutor
        provides concurrency so vLLM can batch them on the GPU.

        Args:
            frame: Full BGR frame
            activities: List of dicts with keys: type, person_idx, bbox
            timestamp_sec: Frame timestamp for logging

        Returns:
            Dict mapping 'activity_type_pN' -> (is_confirmed, details_dict)
        """
        if not activities:
            return {}

        activity_summary = ", ".join(f"{a['type']}_p{a.get('person_idx', 0)}" for a in activities)
        self._log.info(
            f"[VLM BATCH] Verifying {len(activities)} activities at t={timestamp_sec:.2f}s: "
            f"[{activity_summary}]"
        )

        # For a single activity, just call directly (no thread overhead)
        if len(activities) == 1:
            act = activities[0]
            key = f"{act['type']}_p{act.get('person_idx', 0)}"
            is_confirmed, result = self.verify_detection_sync(
                frame, act["bbox"], act["type"], act.get("person_idx", 0)
            )
            return {key: (is_confirmed, result)}

        # Fire concurrent requests via thread pool for vLLM batching
        results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(activities), 4)
        ) as executor:
            futures = {}
            for activity in activities:
                activity_type = activity["type"]
                person_idx = activity.get("person_idx", 0)
                bbox = activity["bbox"]
                key = f"{activity_type}_p{person_idx}"

                future = executor.submit(
                    self.verify_detection_sync,
                    frame, bbox, activity_type, person_idx,
                )
                futures[key] = future

            for key, future in futures.items():
                try:
                    is_confirmed, result = future.result(
                        timeout=self.timeout * self.max_retries + 5
                    )
                    results[key] = (is_confirmed, result)
                except Exception as e:
                    self._log.error(f"[VLM BATCH] Error for {key}: {e}")
                    results[key] = (False, {"source": "error", "reason": str(e)})

        confirmed = [k for k, (c, _) in results.items() if c]
        rejected = [k for k, (c, _) in results.items() if not c]
        self._log.info(
            f"[VLM BATCH] Complete: {len(confirmed)} confirmed, {len(rejected)} rejected"
            + (f" — confirmed=[{', '.join(confirmed)}]" if confirmed else "")
            + (f" — rejected=[{', '.join(rejected)}]" if rejected else "")
        )
        return results

    # ─── Fallback & Stats ─────────────────────────────────────

    def _fallback_result(self, activity_type: str, reason: str) -> Dict:
        """Return a safe fallback result when VLM is unavailable."""
        return {
            "train_moving": True,  # Assume moving (safer)
            "verified": False,  # Don't confirm (reduce false positives)
            "confidence": "low",
            "reason": f"fallback: {reason}",
            "vlm_confidence_score": 0.0,
            "source": "fallback",
            "activity_type": activity_type,
        }

    def get_stats(self) -> Dict:
        """Return service statistics."""
        return {
            **self._stats,
            "circuit_breaker_open": self._is_circuit_open(),
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear the frame cache and rejection throttle."""
        self._cache.clear()
        self._rejection_throttle.clear()

    def health_check(self) -> Dict:
        """Check if the vLLM server is reachable."""
        try:
            response = self.sync_client.get("/models")
            response.raise_for_status()
            models = response.json()
            return {"status": "healthy", "models": models}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class VLMActivityBatchCollector:
    """
    Collects activities detected for a person, to be verified in batch.

    Drop-in replacement for the existing ActivityBatchCollector pattern.
    """

    def __init__(self):
        self._activities: List[Dict] = []

    def add(self, activity_type: str, person_idx: int, bbox: List[int]):
        """Add an activity to the batch for verification."""
        self._activities.append({
            "type": activity_type,
            "person_idx": person_idx,
            "bbox": bbox,
        })

    def has_activities(self) -> bool:
        return len(self._activities) > 0

    def get_activities(self) -> List[Dict]:
        return self._activities

    def clear(self):
        self._activities.clear()
