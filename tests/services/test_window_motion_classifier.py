"""Tests for window-motion classifier banding + texture sanity (2026-05-20).

The classifier was tightened to:
  - lower the STOPPED threshold to 8.0 (FP_max=7.55 + 6% margin)
  - add an inconclusive band 8-12 where no override is applied
  - reject ROIs with std < MOTION_ROI_MIN_TEXTURE_STD as untextured
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.vlm import motion_classifier as mc


def _synth_frame(path: Path, brightness: int = 128, noise_std: int = 0) -> None:
    """Write a 1280x720 grayscale-tinted frame. Adds Gaussian noise if asked."""
    img = np.full((720, 1280, 3), brightness, dtype=np.uint8)
    if noise_std > 0:
        n = np.random.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def test_static_textured_window_triggers_stopped(tmp_path):
    """Five identical noisy frames → near-zero diff → confident STOPPED."""
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.jpg"
        # Same RNG seed across frames so the texture is identical => 0 diff,
        # but std > 0 so the texture-sanity check passes.
        np.random.seed(42)
        _synth_frame(p, brightness=120, noise_std=40)
        paths.append(p)
    result = mc.compute_window_motion_score(paths, camera_id="LP_CAM2")
    assert result is not None
    assert result["score"] < mc.MOTION_DIFF_STOPPED_THRESHOLD
    assert result["stopped"] is True
    assert result["inconclusive"] is False
    assert result["roi_std"] >= mc.MOTION_ROI_MIN_TEXTURE_STD


def test_dynamic_textured_window_is_running_not_stopped(tmp_path):
    """Different noisy frames → high diff → above the 12.0 running threshold."""
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.jpg"
        np.random.seed(i)  # different noise per frame
        _synth_frame(p, brightness=120, noise_std=80)
        paths.append(p)
    result = mc.compute_window_motion_score(paths, camera_id="LP_CAM2")
    assert result is not None
    assert result["score"] > mc.MOTION_DIFF_RUNNING_THRESHOLD
    assert result["stopped"] is False
    assert result["inconclusive"] is False


def test_untextured_black_window_returns_inconclusive(tmp_path):
    """Uniformly black ROI → roi_std≈0 → no override, inconclusive=True.

    Pre-fix this scenario would have always returned stopped=True (score
    collapses to ~0 because there's no texture to diff against), producing
    a false STOPPED override on every activity for that camera install.
    """
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.jpg"
        _synth_frame(p, brightness=0, noise_std=0)  # pure black
        paths.append(p)
    result = mc.compute_window_motion_score(paths, camera_id="LP_CAM2")
    assert result is not None
    assert result["roi_std"] < mc.MOTION_ROI_MIN_TEXTURE_STD
    assert result["stopped"] is False
    assert result["inconclusive"] is True


def test_score_in_inconclusive_band_does_not_override(tmp_path, monkeypatch):
    """A score in the 8-12 inconclusive band must NOT trigger stopped=True.

    This is the recall-protecting band: TPs with scores in the gap between
    confident-stopped (<8) and confident-running (>12) get the benefit of
    the doubt rather than being dropped.
    """
    # Patch the helper to return a deterministic score in the band.
    monkeypatch.setattr(
        mc, "_window_roi_diff", lambda a, b, roi: 10.0
    )
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        np.random.seed(99)
        _synth_frame(p, brightness=100, noise_std=20)
        paths.append(p)
    result = mc.compute_window_motion_score(paths, camera_id="LP_CAM2")
    assert result is not None
    # Median of three 10.0s is 10.0 — inside the band.
    assert result["score"] == 10.0
    assert result["stopped"] is False
    assert result["inconclusive"] is True


def test_ocr_negative_cache_ttl_evicts(monkeypatch):
    """A negative OCR cache entry must expire after the TTL so a later
    activity (with potentially better keyframes) gets a retry.
    """
    # Shorten the TTL so the test doesn't wait forever.
    monkeypatch.setattr(mc, "_OCR_NEGATIVE_CACHE_TTL_SEC", 0.05)
    # Stub OCR so it always fails — we're testing cache behavior, not OCR.
    monkeypatch.setattr(mc, "detect_camera_id", lambda p: None)

    # Clear shared cache.
    with mc._camera_cache_lock:
        mc._camera_cache.clear()

    first = mc.detect_camera_for_video("v.mp4", keyframes=["x.jpg"])
    assert first is None
    # Within TTL, second call uses cached negative without re-OCR.
    second = mc.detect_camera_for_video("v.mp4", keyframes=["x.jpg"])
    assert second is None
    # After TTL elapses, cache should evict and retry — we patch
    # detect_camera_id to return a value the second time to prove retry.
    import time
    time.sleep(0.07)
    call_count = {"n": 0}
    def _fake(_p):
        call_count["n"] += 1
        return "LP_CAM2"
    monkeypatch.setattr(mc, "detect_camera_id", _fake)
    third = mc.detect_camera_for_video("v.mp4", keyframes=["x.jpg"])
    assert third == "LP_CAM2"
    assert call_count["n"] == 1
