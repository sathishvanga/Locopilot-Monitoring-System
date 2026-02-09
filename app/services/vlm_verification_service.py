"""
VLM Verification Service - Secondary verification using Qwen2.5-VL-7B

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

ACTIVITY_DESCRIPTIONS = {
    "cell_phone": "potential mobile phone usage by the locomotive operator",
    "writing": "the operator appears to be writing or reading a document/book",
    "packing_bags": "the operator appears to be packing or reaching into a bag",
    "eating_drinking": "the operator appears to be eating or drinking",
    "sleeping": "the operator may be sleeping or has their eyes closed with head drooping",
    "microsleep": "the operator may be experiencing a brief microsleep episode",
    "mind_diversion": "the operator's attention appears diverted from the track ahead",
    "group_detected": "three or more people are detected in the locomotive cab",
    "no_person_detected": "no person is detected in the locomotive cab",
    "lp_hand_gesture": "the Loco Pilot (LP) has not raised their hand for coordination",
    "alp_hand_gesture": "the Assistant Loco Pilot (ALP) has not raised their hand for coordination",
    "alp_not_standing": "the Assistant Loco Pilot (ALP) is not standing during pre-arrival",
}

ACTIVITY_RULES = {
    "cell_phone": """   - verified=true ONLY if a phone/smartphone is clearly visible in the person's hand
   - Radio handsets, walkie-talkies, control devices, clipboards -> verified=false
   - Reflections, shiny objects, or unclear items -> verified=false
   - If hands are not clearly visible -> verified=false""",

    "writing": """   - verified=true if the person's head is down and hands are positioned as if writing or reading
   - A book, paper, or clipboard in the lap or on the dashboard counts as writing
   - Hands close together with head tilted down in a writing posture -> verified=true
   - Person sitting upright looking forward with hands on controls -> verified=false
   - Hands simply resting on thighs with no writing posture -> verified=false""",

    "packing_bags": """   - verified=true ONLY if the person's hands are clearly inside or reaching into a bag
   - Hands resting near a bag but not interacting with it -> verified=false
   - Bag visible but person not touching it -> verified=false""",

    "eating_drinking": """   - verified=true ONLY if a cup, bottle, or food item is in the person's hand near their face
   - Objects on a shelf or dashboard not being held -> verified=false
   - Water bottle in a holder not being used -> verified=false""",

    "sleeping": """   - verified=true ONLY if the person clearly has eyes closed AND head is drooping or slumped
   - Person looking down at instruments or controls -> verified=false
   - Brief blinks or natural eye movements -> verified=false
   - If face is not clearly visible -> verified=false""",

    "microsleep": """   - verified=true ONLY if the person shows clear signs of drowsiness with eyes closing
   - Natural blinks or looking down -> verified=false
   - Alert person with normal eye movements -> verified=false""",

    "mind_diversion": """   - verified=true ONLY if the person is clearly distracted and looking away from the forward direction
   - Checking side mirrors, signals, or track infrastructure -> verified=false
   - Slight head movements or natural scanning of surroundings -> verified=false
   - Looking at control panel instruments -> verified=false""",

    "group_detected": """   - verified=true ONLY if you can clearly count 3 or more distinct people in the cab
   - Reflections, shadows, or partially visible people should not be counted
   - Count only clearly visible, distinct individuals""",

    "no_person_detected": """   - verified=true ONLY if the locomotive cab is genuinely empty with no person visible
   - If any part of a person (arm, leg, torso) is visible -> verified=false
   - If the image is dark/unclear and you cannot determine -> verified=false""",

    "lp_hand_gesture": """   - verified=true ONLY if the Loco Pilot clearly has NOT raised their hand
   - If the person's hand/arm is raised above shoulder level -> verified=false
   - If hands are not clearly visible -> verified=false""",

    "alp_hand_gesture": """   - verified=true ONLY if the Assistant Loco Pilot clearly has NOT raised their hand
   - If the person's hand/arm is raised above shoulder level -> verified=false
   - If hands are not clearly visible -> verified=false""",

    "alp_not_standing": """   - verified=true ONLY if the ALP is clearly seated or crouching, not standing
   - If the person appears to be standing upright -> verified=false
   - If posture is unclear -> verified=false""",
}

VERIFICATION_PROMPT_TEMPLATE = """You are a safety monitor analyzing locomotive cab footage from Indian Railways.
A detection system flagged: {activity_description}

Analyze the image and respond ONLY in this exact JSON format:
{{"verified": true/false, "confidence": "high"/"medium"/"low", "reason": "brief explanation"}}

Evaluate whether the flagged activity is genuinely occurring:
{activity_rules}

IMPORTANT: Respond with ONLY the JSON object, no other text."""


def build_verification_prompt(activity_type: str) -> str:
    """Build the verification prompt for a specific activity type."""
    description = ACTIVITY_DESCRIPTIONS.get(activity_type, f"a potential {activity_type} violation")
    rules = ACTIVITY_RULES.get(activity_type, "   - Evaluate whether this activity is genuinely occurring")
    return VERIFICATION_PROMPT_TEMPLATE.format(
        activity_description=description,
        activity_rules=rules,
    )


class VLMVerificationService:
    """
    Synchronous VLM-based verification service using Qwen2.5-VL via vLLM.

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

    # Per-activity crop padding (px). Larger padding captures more context
    # (e.g. book in lap for writing, full body posture for sleeping).
    ACTIVITY_CROP_PADDING = {
        'writing': 150,     # Need to capture book in lap
        'sleeping': 100,    # Need full body posture
        'microsleep': 100,
        'packing_bags': 100,
    }

    # Per-activity rejection cooldown (seconds). Sustained activities like
    # writing/sleeping use shorter cooldowns to allow re-checking more frequently.
    ACTIVITY_COOLDOWNS = {
        'writing': 3.0,
        'sleeping': 3.0,
        'microsleep': 3.0,
    }

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        timeout: float = 10.0,
        max_retries: int = 2,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_sec: float = 60.0,
        cache_ttl_sec: float = 5.0,
        crop_padding: int = 50,
        jpeg_quality: int = 85,
        rejection_cooldown_sec: float = 15.0,
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

        response = self.sync_client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

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
        cooldown = self.ACTIVITY_COOLDOWNS.get(activity_type, self._rejection_cooldown)
        if throttle_key in self._rejection_throttle:
            prev_result, prev_ts = self._rejection_throttle[throttle_key]
            if time.time() - prev_ts < cooldown:
                self._stats["throttle_hits"] += 1
                self._log.debug(
                    f"[VLM] Throttled {throttle_key} "
                    f"({time.time() - prev_ts:.1f}s since last rejection)"
                )
                return False, prev_result
            else:
                del self._rejection_throttle[throttle_key]

        # Circuit breaker check
        if self._is_circuit_open():
            result = self._fallback_result(activity_type, "circuit_breaker_open")
            return False, result

        # Crop detection region (activity-specific padding)
        pad = self.ACTIVITY_CROP_PADDING.get(activity_type, self.crop_padding)
        cropped = self._crop_detection_region(frame, bbox, padding=pad)
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
