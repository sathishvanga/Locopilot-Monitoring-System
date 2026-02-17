# Task 0019: Voting cache stores raw video frames -- up to 1.9GB RAM

- **Issue ID:** C-08
- **Priority:** Phase 4 - Memory & Performance (Item 19)
- **Severity:** CRITICAL
- **Category:** Memory
- **File:** `app/services/voting_verification_service.py:48, 98-119`

## Description

LRU cache (max 32 entries) stores raw numpy frames. With 10 frames of 1080p per entry (~60MB each), max capacity consumes ~1.9GB.

## Fix

Reduce `max_size` to 4-8, add memory budget check, or cache only inference results (not raw frames).
