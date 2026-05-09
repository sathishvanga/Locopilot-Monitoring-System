"""Adversarial regression tests for the VLM verifier.

Loads keyframes from ``tests/regression/vlm_fixture/<bucket>/`` and runs
each against the deterministic gate (and optionally the live VLM, if
``VLM_BASE_URL`` is reachable). Exists to catch:

  - Prompt regressions across model upgrades (4.6 → 4.7 → next)
  - Pre-VLM gate misfires after Pipeline-1 overlay colour changes
  - Drift in confounder handling (brake handle, radio, supervisor visit)

Skips cleanly when fixtures are absent so a fresh clone passes
``pytest`` without setup.

The gate test does NOT require a vLLM endpoint — it runs the same
``_count_bboxes_in_keyframes`` helper used in production. The end-to-end
test is gated by ``RUN_VLM_E2E=1`` and requires ``VLM_BASE_URL`` reachable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest


FIXTURE_ROOT = Path(__file__).parent / "vlm_fixture"


def _load_expectations(bucket_dir: Path) -> List[Dict[str, Any]]:
    exp_path = bucket_dir / "expectations.json"
    if not exp_path.is_file():
        return []
    try:
        data = json.loads(exp_path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("entries") or []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _all_fixture_entries() -> List[Dict[str, Any]]:
    if not FIXTURE_ROOT.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for bucket_dir in sorted(FIXTURE_ROOT.iterdir()):
        if not bucket_dir.is_dir():
            continue
        bucket = bucket_dir.name
        for entry in _load_expectations(bucket_dir):
            entry = dict(entry)
            entry.setdefault("bucket", bucket)
            entry["_path"] = str(bucket_dir / entry["filename"])
            rows.append(entry)
    return rows


@pytest.fixture(scope="session")
def fixture_entries() -> List[Dict[str, Any]]:
    return _all_fixture_entries()


def test_fixture_layout_present(fixture_entries: List[Dict[str, Any]]) -> None:
    """Sanity check: at minimum the fixture directory tree exists."""
    if not FIXTURE_ROOT.is_dir():
        pytest.skip(f"fixture root not present at {FIXTURE_ROOT}")
    expected_buckets = {
        "empty_cab", "idle_person", "writing_tp", "writing_fp",
        "no_object_writing",
    }
    found = {p.name for p in FIXTURE_ROOT.iterdir() if p.is_dir()}
    missing = expected_buckets - found
    assert not missing, f"missing fixture buckets: {sorted(missing)}"


def test_pre_gate_decisions(fixture_entries: List[Dict[str, Any]]) -> None:
    """For each fixture entry with an ``expected_gate``, run the bbox
    detector and assert the gate decision matches.

    Skips the entire test when no expectations have been populated yet.
    """
    gate_entries = [e for e in fixture_entries if e.get("expected_gate") is not None]
    if not gate_entries:
        pytest.skip("no fixture entries with expected_gate populated")

    from app.services.vlm_verification_service import (
        _count_bboxes_in_keyframes,
        _OBJECT_REQUIRED_TYPES,
        _PRE_GATE_SKIP_OBJECT_TYPES,
    )

    failures: List[str] = []
    for entry in gate_entries:
        path = Path(entry["_path"])
        if not path.is_file():
            failures.append(f"{entry['filename']}: file missing")
            continue
        object_type = (entry.get("object_type") or "").lower()
        counts = _count_bboxes_in_keyframes([path])

        rendering_active = counts["with_any_bbox"] > 0
        gate_fired: str | None = None
        if rendering_active and object_type not in _PRE_GATE_SKIP_OBJECT_TYPES:
            if counts["with_person"] == 0:
                gate_fired = "PRE_GATE_DROP_NO_SUBJECT"
            elif (
                object_type in _OBJECT_REQUIRED_TYPES
                and counts["with_object"] == 0
            ):
                gate_fired = "PRE_GATE_DROP_NO_OBJECT"

        expected = entry.get("expected_gate")
        if gate_fired != expected:
            failures.append(
                f"{entry['filename']} (bucket={entry.get('bucket')}, "
                f"object_type={object_type}): expected gate={expected}, "
                f"actual={gate_fired}, counts={counts}"
            )

    assert not failures, "gate decisions diverged:\n  " + "\n  ".join(failures)


@pytest.mark.skipif(
    os.environ.get("RUN_VLM_E2E") != "1",
    reason="end-to-end VLM test gated by RUN_VLM_E2E=1 (needs VLM_BASE_URL reachable)",
)
def test_e2e_vlm_verdicts(fixture_entries: List[Dict[str, Any]]) -> None:
    """Ground-truth precision/recall against the live VLM endpoint.

    Counts how often the post-gate verdict matches the expected verdict
    on fixture entries that have one. Asserts overall precision ≥ 0.85
    and recall ≥ 0.80 — generous bounds for a starter fixture; tighten
    as the catalogue grows.
    """
    e2e_entries = [e for e in fixture_entries if e.get("expected_verdict")]
    if not e2e_entries:
        pytest.skip("no fixture entries with expected_verdict populated")

    # The full e2e harness requires constructing minimal activity dicts
    # and dispatching through VlmVerificationService. Stubbed here as a
    # follow-up (separate ticket) so the gate test can land first.
    pytest.skip("e2e harness pending — see infra ticket for completion")
