# Task 0016: No retry logic for external API calls

- **Issue ID:** H-20
- **Priority:** Phase 3 - Security & Reliability (Item 16)
- **Severity:** HIGH
- **Category:** Reliability
- **File:** `app/services/external_api_service.py:103, 189`

## Description

Both `_post_no_events` and `_post_violations` make single POST requests. If CVVR API is temporarily unavailable, all violation data for the trip is permanently lost.

## Fix

Add retry with exponential backoff. Consider dead-letter queue for persistent failures.
