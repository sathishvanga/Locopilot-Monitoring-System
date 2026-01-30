"""
VLM Verification Service - Semantic activity verification using Qwen2.5-VL

Replaces the YOLO-based voting verification with a Vision Language Model
that semantically understands activities rather than relying on geometric heuristics.

Hybrid Pipeline:
    YOLO nano (fast scan) -> Heuristic candidates -> VLM verification (semantic filter)

The VLM receives candidate frames and per-activity prompts, then confirms or
rejects each detection based on visual understanding of the scene.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..utils.config import get_settings

logger = logging.getLogger(__name__)


# Activity-specific verification prompts
# Each prompt is designed to get a clear YES/NO answer from the VLM
ACTIVITY_PROMPTS = {
    "cell_phone": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "Is this person holding, using, or talking on a mobile phone or smartphone? "
        "Look for a phone-shaped object in their hand, near their ear, or being looked at. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "sleep": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "Does this person appear to be sleeping? Signs include: eyes closed, head drooping forward "
        "or tilted to the side, slumped posture, body completely still in a resting position. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "microsleep": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "Does this person appear to be having a microsleep or nodding off briefly? "
        "Signs include: eyes briefly closing, head starting to nod, momentary loss of alertness. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "writing": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "Is this person writing in a book, logbook, or on paper? "
        "Signs include: pen/pencil in hand, book/paper on lap or surface, hand moving in writing motion, head looking down at paper. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "packing_bags": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "Is this person actively reaching into, packing, or unpacking a bag, backpack, or suitcase? "
        "Their hands must be interacting with the bag opening or contents. "
        "A bag simply being near the person does NOT count. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "mind_diversion": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "This person is a train crew member who should be facing forward watching the track. "
        "Is this person's attention clearly diverted from the front? "
        "Signs: head turned significantly sideways, looking down at something unrelated to controls, "
        "or looking backward. Minor head movements and checking instruments do NOT count. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "eating_drinking": (
        "Look at the highlighted person in this train cabin surveillance frame. "
        "Is this person eating food or drinking from a cup, bottle, or container? "
        "Signs: holding food/cup/bottle near mouth, chewing, tilting head back to drink. "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "lp_hand_gesture": (
        "Look at the people in this train cabin surveillance frame. "
        "The Loco Pilot sits on the LEFT side. Is the Loco Pilot raising their hand "
        "above shoulder level in a signaling gesture? "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "alp_hand_gesture": (
        "Look at the people in this train cabin surveillance frame. "
        "The Assistant Loco Pilot sits on the RIGHT side. Is the ALP raising their hand "
        "above shoulder level in a signaling gesture? "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"reason\": \"brief explanation\"}"
    ),
    "group_detected": (
        "Count the number of distinct people visible in this train cabin surveillance frame. "
        "Are there 3 or more people visible in the cabin? "
        "Respond with ONLY a JSON object: {\"detected\": true/false, \"count\": N, \"reason\": \"brief explanation\"}"
    ),
}

# System prompt for consistent VLM behavior
SYSTEM_PROMPT = (
    "You are a train cabin safety monitoring system analyzing surveillance camera frames. "
    "The camera is mounted behind and to the right of the crew. "
    "The Loco Pilot (LP) sits on the LEFT side, the Assistant Loco Pilot (ALP) sits on the RIGHT side. "
    "Analyze the image carefully and respond ONLY with the requested JSON format. "
    "Be conservative: only confirm an activity if you are reasonably confident it is happening."
)


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert BGR OpenCV frame to PIL RGB Image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _parse_vlm_response(response_text: str) -> Dict:
    """
    Parse VLM JSON response, handling common formatting issues.

    Conservative parsing: defaults to NOT detected on parse failure.

    Returns:
        Dict with at least 'detected' (bool) and 'reason' (str)
    """
    text = response_text.strip()

    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        result = json.loads(text)
        if "detected" in result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Conservative fallback: only match explicit JSON-like true patterns
    lower = response_text.lower()
    if "\"detected\": true" in lower or "\"detected\":true" in lower:
        return {"detected": True, "reason": "parsed_from_text"}

    # Default: not detected (conservative — avoids false positives)
    return {"detected": False, "reason": "parse_failed_default_negative"}


class VLMVerificationService:
    """
    Semantic verification service using Qwen2.5-VL.

    Replaces YOLO-based re-detection voting with VLM visual understanding.
    The VLM receives frames and natural language prompts per activity type,
    returning structured JSON with detection confirmation.
    """

    def __init__(self, vlm_model=None, vlm_processor=None):
        """
        Initialize VLM verification service.

        Args:
            vlm_model: Pre-loaded Qwen2.5-VL model instance
            vlm_processor: Pre-loaded AutoProcessor instance
        """
        self.settings = get_settings()
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self._available = vlm_model is not None and vlm_processor is not None

        logger.info("=" * 60)
        logger.info("VLMVerificationService initialized")
        logger.info(f"  vlm_enabled: {self.settings.vlm_enabled}")
        logger.info(f"  vlm_model_name: {self.settings.vlm_model_name}")
        logger.info(f"  vlm_available: {self._available}")
        logger.info(f"  vlm_max_new_tokens: {self.settings.vlm_max_new_tokens}")
        logger.info(f"  vlm_num_verification_frames: {self.settings.vlm_num_verification_frames}")
        logger.info("=" * 60)

    @property
    def available(self) -> bool:
        """Check if VLM is loaded and ready."""
        return self._available and self.settings.vlm_enabled

    def set_models(self, vlm_model: Any, vlm_processor: Any) -> None:
        """Set VLM model and processor (for lazy initialization)."""
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self._available = vlm_model is not None and vlm_processor is not None
        logger.info(f"VLM models set, available={self._available}")

    def verify_activity(
        self,
        frames: List[np.ndarray],
        activity_type: str,
        person_bbox: Optional[List[float]] = None,
    ) -> Tuple[bool, Dict]:
        """
        Verify a single activity using VLM semantic analysis.

        Args:
            frames: List of BGR frames (typically 3-5 from around the detection timestamp)
            activity_type: Activity type key (e.g., 'cell_phone', 'writing')
            person_bbox: Optional bounding box [x1, y1, x2, y2] of the person

        Returns:
            Tuple of (is_confirmed, details_dict)
        """
        if not self.available:
            logger.warning(f"[VLM] Not available, bypassing verification for {activity_type}")
            return True, {"method": "vlm_bypass", "reason": "vlm_not_available"}

        prompt_template = ACTIVITY_PROMPTS.get(activity_type)
        if prompt_template is None:
            logger.warning(f"[VLM] No prompt for activity type '{activity_type}', bypassing")
            return True, {"method": "vlm_bypass", "reason": f"no_prompt_for_{activity_type}"}

        start_time = time.time()

        try:
            # Select representative frames (first, middle, last) to limit tokens
            selected = self._select_frames(frames)

            # Optionally crop around person bbox to focus VLM attention
            if person_bbox and activity_type not in ("group_detected",):
                selected = [self._crop_person_region(f, person_bbox) for f in selected]

            # Convert to PIL images
            pil_images = [_frame_to_pil(f) for f in selected]

            # Build VLM input messages
            image_content = []
            for img in pil_images:
                image_content.append({"type": "image", "image": img})
            image_content.append({"type": "text", "text": prompt_template})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": image_content},
            ]

            # Process with VLM (inference mode — no gradient tracking, saves ~2x VRAM)
            text_input = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.vlm_processor(
                text=[text_input],
                images=pil_images,
                padding=True,
                return_tensors="pt",
            ).to(self.vlm_model.device)

            with torch.inference_mode():
                output_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=self.settings.vlm_max_new_tokens,
                    do_sample=False,
                )

            # Decode only the generated tokens (skip input)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response_text = self.vlm_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Free GPU tensors immediately to keep VRAM available for next call
            del inputs, output_ids, generated_ids

            elapsed_ms = (time.time() - start_time) * 1000

            # Parse response
            result = _parse_vlm_response(response_text)
            is_confirmed = bool(result.get("detected", False))

            logger.info(
                f"[VLM] {activity_type}: {'CONFIRMED' if is_confirmed else 'REJECTED'} "
                f"({elapsed_ms:.0f}ms) - {result.get('reason', 'N/A')}"
            )

            return is_confirmed, {
                "method": "vlm",
                "activity_type": activity_type,
                "is_confirmed": is_confirmed,
                "vlm_response": response_text[:500],
                "parsed_result": result,
                "frames_analyzed": len(selected),
                "elapsed_ms": round(elapsed_ms, 1),
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"[VLM] Error verifying {activity_type}: {e}", exc_info=True)
            # On error, fall through to confirm (don't suppress detections due to VLM failure)
            return True, {
                "method": "vlm_error_bypass",
                "activity_type": activity_type,
                "error": str(e),
                "elapsed_ms": round(elapsed_ms, 1),
            }

    def verify_batch(
        self,
        frames: List[np.ndarray],
        activities: List[Dict],
    ) -> Dict[str, Tuple[bool, Dict]]:
        """
        Verify multiple activities using shared frames.

        Args:
            frames: List of BGR frames around the detection timestamp
            activities: List of activity dicts:
                [{'type': 'cell_phone', 'person_bbox': [...], 'person_idx': 0}, ...]

        Returns:
            Dict mapping activity keys to (is_confirmed, details):
            {'cell_phone_p0': (True, {...}), 'writing_p0': (False, {...})}
        """
        if not self.available:
            logger.info("[VLM BATCH] VLM not available, bypassing all")
            return {
                f"{act['type']}_p{act.get('person_idx', 0)}": (
                    True,
                    {"method": "vlm_bypass", "reason": "vlm_not_available"},
                )
                for act in activities
            }

        start_time = time.time()
        results: Dict[str, Tuple[bool, Dict]] = {}

        logger.info(f"[VLM BATCH] Verifying {len(activities)} activities")

        for act in activities:
            activity_type = act["type"]
            person_bbox = act.get("person_bbox", [])
            person_idx = act.get("person_idx", 0)
            activity_key = f"{activity_type}_p{person_idx}"

            is_confirmed, details = self.verify_activity(
                frames=frames,
                activity_type=activity_type,
                person_bbox=person_bbox if person_bbox else None,
            )
            details["person_idx"] = person_idx
            results[activity_key] = (is_confirmed, details)

        total_ms = (time.time() - start_time) * 1000
        confirmed_count = sum(1 for v, _ in results.values() if v)
        logger.info(
            f"[VLM BATCH] Done: {confirmed_count}/{len(activities)} confirmed "
            f"({total_ms:.0f}ms total)"
        )

        return results

    def _select_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Select representative frames to send to VLM (limit token usage)."""
        max_frames = self.settings.vlm_num_verification_frames
        if len(frames) <= max_frames:
            return frames

        # Evenly sample across the frame range
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[i] for i in indices]

    def _crop_person_region(
        self, frame: np.ndarray, bbox: List[float], margin_ratio: float = 0.2
    ) -> np.ndarray:
        """
        Crop frame around person bbox with margin for context.

        Args:
            frame: Full BGR frame
            bbox: Person bounding box [x1, y1, x2, y2]
            margin_ratio: Extra margin as ratio of bbox size

        Returns:
            Cropped frame (or original if bbox is invalid)
        """
        if not bbox or len(bbox) < 4:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Add margin
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)

        cropped = frame[cy1:cy2, cx1:cx2]
        if cropped.size == 0:
            return frame
        return cropped
