# CR-017: Inconsistent logging with multiple logger instances

- **Severity:** Medium
- **Category:** Code Quality / Logging
- **Lines:** 131-133, 170, and various methods

## Description

The module uses three different logging approaches: `self.logger` (instance-level), module-level `gesture_logger` and `monitor_logger`, and fresh loggers created inside methods. This creates inconsistent log output and makes filtering difficult.

## Affected Code

```python
gesture_logger = _setup_module_logger('HandGestureDetection')   # Line 132
monitor_logger = _setup_module_logger('LocopilotMonitor')       # Line 133
self.logger = _setup_module_logger(...)                         # Line 170
```

## Suggested Fix

Consolidate to a single logger hierarchy. Use `self.logger` consistently throughout the class and child loggers (e.g., `self.logger.getChild('gesture')`) for subsystem-specific logging.
