# Task 0013: CORS wildcard + credentials

- **Issue ID:** C-05
- **Priority:** Phase 3 - Security & Reliability (Item 13)
- **Severity:** CRITICAL
- **Category:** Security
- **File:** CORS middleware configuration

## Description

CORS wildcard origin (`*`) combined with credentials mode creates a security vulnerability. Any origin can make authenticated requests to the API.

## Fix

Replace wildcard with explicit allowed origins list. Configure per-environment allowed origins in settings.
