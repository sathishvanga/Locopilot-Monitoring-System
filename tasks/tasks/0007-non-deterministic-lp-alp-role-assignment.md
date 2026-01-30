# CR-007: Non-deterministic LP/ALP role assignment based on bounding box area

- **Severity:** High
- **Category:** Bug / Logic Error
- **Lines:** 5754

## Description

Person roles (LP vs ALP) are assigned based on bounding box area. When persons have similar sizes, roles can flip between frames, breaking hand gesture coordination logic that depends on stable role assignments.

## Suggested Fix

Implement temporal role tracking using position continuity (IoU-based tracking across frames) or a simple tracker like SORT to maintain consistent person identities.
