"""Unit tests for ``app.core.tracking.static_object_filter`` (T5).

The plan's verification one-liner expects: 3 successive calls with the same
bbox should result in suppression once ``min_frames`` is exceeded, with the
filter returning an empty list on the final call.
"""
import logging

import pytest

from app.core.tracking.static_object_filter import StaticObjectFilter


def test_static_suppression_after_min_frames():
    """3 calls with the same bbox: first 2 pass through, 3rd is suppressed."""
    f = StaticObjectFilter(
        label='backpack',
        iou_threshold=0.8,
        min_frames=2,
        enabled=True,
    )
    bbox = [10, 10, 30, 30]
    out_calls = []
    for _ in range(3):
        out_calls.append(f.filter([bbox]))

    # On the 3rd call, frame_count reaches 3 (>= min_frames=2) so the
    # detection is filtered out.
    assert out_calls[-1] == []


def test_disabled_filter_passes_through():
    f = StaticObjectFilter(
        label='backpack',
        iou_threshold=0.8,
        min_frames=2,
        enabled=False,
    )
    bbox = [10, 10, 30, 30]
    for _ in range(5):
        out = f.filter([bbox])
    assert out == [bbox]


def test_empty_detections_returned_as_is():
    f = StaticObjectFilter(
        label='phone',
        iou_threshold=0.7,
        min_frames=5,
        enabled=True,
    )
    assert f.filter([]) == []


def test_phone_log_string_byte_identical(caplog):
    """The phone suppression log line must remain byte-identical (operators
    grep ``[STATIC PHONE]`` and ``— likely panel instrument``)."""
    logger = logging.getLogger('test_static_phone_log')
    logger.setLevel(logging.DEBUG)
    f = StaticObjectFilter(
        label='phone',
        iou_threshold=0.7,
        min_frames=2,
        enabled=True,
        log_level='info',
        logger=logger,
    )
    bbox = [100, 200, 150, 260]
    with caplog.at_level(logging.INFO, logger='test_static_phone_log'):
        for _ in range(3):
            f.filter([bbox])

    suppression_messages = [r.message for r in caplog.records if '[STATIC PHONE]' in r.message]
    assert suppression_messages, "expected at least one [STATIC PHONE] suppression log"
    msg = suppression_messages[-1]
    assert msg.startswith('[STATIC PHONE] Suppressed static phone at [100, 200, 150, 260] ')
    assert 'likely panel instrument' in msg
    # Em-dash and trailing parenthesis structure preserved verbatim.
    assert '— likely panel instrument)' in msg


def test_backpack_log_string_byte_identical(caplog):
    """The backpack suppression log line must remain byte-identical."""
    logger = logging.getLogger('test_static_backpack_log')
    logger.setLevel(logging.DEBUG)
    f = StaticObjectFilter(
        label='backpack',
        iou_threshold=0.8,
        min_frames=2,
        enabled=True,
        log_level='debug',
        logger=logger,
    )
    bbox = [10, 10, 30, 30]
    with caplog.at_level(logging.DEBUG, logger='test_static_backpack_log'):
        for _ in range(3):
            f.filter([bbox])

    msgs = [r.message for r in caplog.records if '[STATIC BACKPACK]' in r.message]
    assert msgs, "expected at least one [STATIC BACKPACK] suppression log"
    msg = msgs[-1]
    assert msg.startswith('[STATIC BACKPACK] Suppressed static backpack at [10, 10, 30, 30] ')
    assert 'frames)' in msg
