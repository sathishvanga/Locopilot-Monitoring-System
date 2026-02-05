# CR-001: God Class violates Single Responsibility Principle

- **Severity:** Critical
- **Category:** Architecture / Code Organization
- **Lines:** 150-7195 (entire class)

## Description

The `LocopilotActivityMonitor` class handles everything: model loading, frame sampling, object detection, pose estimation, sleep/writing/packing detection, hand gestures, person identification, video I/O, and evidence management. This is untestable in isolation and violates the single responsibility principle.

## Affected Code

The entire `LocopilotActivityMonitor` class spanning ~7,000 lines with 50+ methods.

## Suggested Fix

Decompose into smaller classes: `ActivityDetector`, `FrameSampler`, `ModelManager`, `ActivityStateTracker`, and `EvidenceCollector`. Each class should own one concern and communicate via well-defined interfaces.
