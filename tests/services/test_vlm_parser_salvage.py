"""Verdict-parser truncation salvage.

The vLLM endpoint can truncate JSON output mid-response when ``max_tokens``
is reached before the model emits the closing brace. With the verdict-first
schema reorder, the load-bearing fields (``verdict``, ``confidence``, the
structured booleans) are emitted first; the parser must therefore recover
them from a truncated payload rather than dropping the entire response into
``parse_error`` (fail-open).

These tests pin the recovery contract:
  - JSON missing only the closing brace is recovered by appending ``}``.
  - JSON truncated mid-string still salvages the verdict + confidence via
    regex, with a ``_salvaged_from_truncation`` audit flag set.
  - JSON with no recognisable verdict still returns parse_error so callers
    fall open.
"""
from __future__ import annotations

import os

os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")

from app.services.vlm.verdict_parser import _parse_verdict


# Actual truncated payload from run_20260509_091436 — the merged minute-5
# writing sub-type whose verdict was lost to truncation. Verbatim from the
# vLLM response after the response_format constraint kicked in.
PROD_TRUNCATED = (
    '{\n  "verdict": "TRUE_POSITIVE",\n'
    '  "which_person": "LP",\n'
    '  "confidence": 0.95,\n'
    '  "book_visible_on_desk": true,\n'
    '  "book_is_open": true,\n'
    '  "hand_actually_on_book": true,\n'
    '  "head_oriented_to_book": true,\n'
    '  "pen_in_hand": true,\n'
    '  "actively_handling_papers": true,\n'
    '  "tp_path": "A",\n '
)


def test_recovers_verdict_from_production_truncation():
    """The exact production payload that motivated the salvage feature."""
    result = _parse_verdict(PROD_TRUNCATED)
    assert "parse_error" not in result, (
        "writing TP must be salvaged from truncated response, not lost to "
        "fail-open"
    )
    assert result["verdict"] == "TRUE_POSITIVE"
    assert result["confidence"] == 0.95
    assert result["hand_actually_on_book"] is True
    assert result["pen_in_hand"] is True
    assert result["actively_handling_papers"] is True
    assert result.get("which_person") == "LP"


def test_appends_missing_close_brace():
    """JSON truncated only at the final ``}``: simple closure recovers it
    without falling into the regex-salvage path."""
    raw = '{"verdict": "FALSE_POSITIVE", "confidence": 0.85, "reason": "test"'
    result = _parse_verdict(raw)
    assert "parse_error" not in result
    assert result["verdict"] == "FALSE_POSITIVE"
    assert result["confidence"] == 0.85
    # No salvage flag — the closure recovery path is preferred over regex.
    assert "_salvaged_from_truncation" not in result


def test_handles_code_fence_wrapper():
    """vLLM response_format=json_object usually strips fences, but legacy
    paths still emit ```json fences — keep the existing tolerance."""
    raw = '```json\n{"verdict": "UNCERTAIN", "confidence": 0.5}\n```'
    result = _parse_verdict(raw)
    assert result["verdict"] == "UNCERTAIN"
    assert result["confidence"] == 0.5


def test_garbage_input_returns_parse_error():
    """Severe truncation that loses even the verdict field must return
    parse_error so the caller fail-opens (keeps the activity)."""
    result = _parse_verdict("not json at all")
    assert "parse_error" in result


def test_truncated_array_recovered_via_close_brace():
    """JSON with an unclosed nested array still recovers via the ``]}``
    closure attempt."""
    raw = (
        '{"verdict": "TRUE_POSITIVE", "confidence": 0.9, '
        '"frame_observations": [{"frame": 1}, {"frame": 2}'
    )
    result = _parse_verdict(raw)
    assert "parse_error" not in result
    assert result["verdict"] == "TRUE_POSITIVE"


def test_salvage_flag_set_when_regex_fallback_used():
    """When even closing-brace attempts fail, the regex fallback fires and
    must mark the result with ``_salvaged_from_truncation`` so audits can
    flag the prompt for tuning. Pathological case: truncation in the middle
    of a number literal — appending ``}``/``"}`` produces invalid JSON, so
    Phase 2 can't recover and Phase 3's regex must do it."""
    raw = (
        '{"verdict": "FALSE_POSITIVE", "confidence": 0.85, '
        '"some_number": 1.2e'
    )
    result = _parse_verdict(raw)
    assert "parse_error" not in result
    assert result["verdict"] == "FALSE_POSITIVE"
    assert result["confidence"] == 0.85
    assert result.get("_salvaged_from_truncation") is True
