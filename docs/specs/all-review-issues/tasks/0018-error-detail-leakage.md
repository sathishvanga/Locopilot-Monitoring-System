# Task 0018: Error messages expose internal details in production

- **Issue ID:** M-22
- **Priority:** Phase 3 - Security & Reliability (Item 18)
- **Severity:** MEDIUM
- **Category:** Security
- **File:** `app/main.py:305`

## Description

Exception messages (file paths, database details) returned to clients in 500 responses.

## Fix

Return generic messages in production; log full errors internally.
