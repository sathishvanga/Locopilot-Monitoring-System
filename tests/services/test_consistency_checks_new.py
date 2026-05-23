"""Tests for the new verdict_parser consistency checks (2026-05-20):
  - writing Path C: pen_in_hand + head_oriented_to_book (book-on-lap archetype)
  - microsleep: rejects head-bowed-writing posture as a microsleep TP
  - cell_phone: synonym matching ("phone", "mobile_phone" map to smartphone)
"""

from app.services.vlm.verdict_parser import _consistency_check


# --- Writing path C ----------------------------------------------------------

def test_writing_path_c_pen_plus_head_orientation_passes():
    """The v06/v21 archetype: hand obscured by book → hand_actually_on_book=false,
    actively_handling_papers=false, but pen + head orientation visible. Previous
    code demoted to UNCERTAIN; new code accepts as TP via Path C.
    """
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "hand_actually_on_book": False,
        "book_visible_on_desk": False,
        "pen_in_hand": True,
        "actively_handling_papers": False,
        "head_oriented_to_book": True,
    }
    assert _consistency_check(parsed, "writing") is None


def test_writing_path_a_still_passes():
    """Path A (hand on book + book on desk) unchanged."""
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "hand_actually_on_book": True,
        "book_visible_on_desk": True,
        "pen_in_hand": False,
        "actively_handling_papers": False,
        "head_oriented_to_book": False,
    }
    assert _consistency_check(parsed, "writing") is None


def test_writing_path_b_still_passes():
    """Path B (pen + actively handling papers) unchanged."""
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "hand_actually_on_book": False,
        "book_visible_on_desk": False,
        "pen_in_hand": True,
        "actively_handling_papers": True,
        "head_oriented_to_book": False,
    }
    assert _consistency_check(parsed, "writing") is None


def test_writing_all_paths_fail_returns_reason():
    """No path satisfied → demote."""
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "hand_actually_on_book": False,
        "book_visible_on_desk": True,  # only desk visible, no engagement
        "pen_in_hand": False,
        "actively_handling_papers": False,
        "head_oriented_to_book": False,
    }
    reason = _consistency_check(parsed, "writing")
    assert reason is not None
    assert "Path" in reason


# --- Microsleep --------------------------------------------------------------

def test_microsleep_writing_pose_rejected():
    """The Cabin_27 t=1525 archetype: pen + paper visible + head bowed.
    Previously microsleep skipped VLM entirely; now VLM must reject this as
    a writing pose, not microsleep.
    """
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "eyes_closed": True,
        "pen_in_hand": True,
        "book_or_paper_visible_at_head": True,
        "hands_engaged_with_paperwork": True,
        "posture_slumped_or_reclined": False,
    }
    reason = _consistency_check(parsed, "microsleep")
    assert reason is not None
    assert "pen_in_hand" in reason or "paperwork" in reason


def test_microsleep_no_eyes_closed_rejected():
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "eyes_closed": False,
        "pen_in_hand": False,
        "book_or_paper_visible_at_head": False,
        "hands_engaged_with_paperwork": False,
        "posture_slumped_or_reclined": True,
    }
    reason = _consistency_check(parsed, "microsleep")
    assert reason is not None
    assert "eyes_closed" in reason


def test_microsleep_genuine_slumped_eyes_closed_passes():
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "eyes_closed": True,
        "pen_in_hand": False,
        "book_or_paper_visible_at_head": False,
        "hands_engaged_with_paperwork": False,
        "posture_slumped_or_reclined": True,
    }
    assert _consistency_check(parsed, "microsleep") is None


def test_microsleep_paper_on_desk_with_slumped_posture_still_passes():
    """Paper on desk but person is slumped (not engaged) — still microsleep TP."""
    parsed = {
        "verdict": "TRUE_POSITIVE",
        "eyes_closed": True,
        "pen_in_hand": False,
        "book_or_paper_visible_at_head": True,
        "hands_engaged_with_paperwork": False,
        "posture_slumped_or_reclined": True,
    }
    assert _consistency_check(parsed, "microsleep") is None


# --- Cell phone synonyms -----------------------------------------------------

def test_cell_phone_smartphone_canonical_passes():
    parsed = {"verdict": "TRUE_POSITIVE", "object_in_hand": "smartphone"}
    assert _consistency_check(parsed, "cell_phone") is None


def test_cell_phone_synonyms_pass():
    for syn in ("phone", "Phone", "mobile_phone", "MOBILE", "cell_phone"):
        parsed = {"verdict": "TRUE_POSITIVE", "object_in_hand": syn}
        assert _consistency_check(parsed, "cell_phone") is None, syn


def test_cell_phone_non_phone_rejected():
    for non in ("radio_handset", "pen", "wallet", "nothing_visible"):
        parsed = {"verdict": "TRUE_POSITIVE", "object_in_hand": non}
        reason = _consistency_check(parsed, "cell_phone")
        assert reason is not None, non


# --- Non-TP verdicts always pass through ------------------------------------

def test_false_positive_passes_through_unchanged():
    parsed = {"verdict": "FALSE_POSITIVE"}
    for ot in ("writing", "microsleep", "cell_phone"):
        assert _consistency_check(parsed, ot) is None
