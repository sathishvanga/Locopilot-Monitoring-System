# CR-018: Commented-out debug code remaining in production

- **Severity:** Medium
- **Category:** Code Quality / Dead Code
- **Lines:** 3449-3528

## Description

A large block of commented-out debug code (~80 lines) remains in the source, adding clutter and confusion about intended behavior.

## Suggested Fix

Remove all commented-out code. Use version control (git) to preserve history if the code is needed later.
