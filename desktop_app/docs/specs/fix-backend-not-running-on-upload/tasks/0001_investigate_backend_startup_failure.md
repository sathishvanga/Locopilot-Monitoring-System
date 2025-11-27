# Task 0001: Investigate Backend Startup Failure

## Metadata

| Field | Value |
|-------|-------|
| **Task Number** | 0001 |
| **Feature** | fix-backend-not-running-on-upload |
| **Category** | services |
| **Component(s)** | BackendManager, FastAPI backend |
| **Agent Type** | python-backend-expert |
| **Estimated Size** | M |
| **Status** | pending |
| **Priority** | high |
| **Dependencies** | None |

## Overview

Thoroughly investigate why the backend fails to auto-start in packaged macOS applications and document findings.

## Description

The BackendManager attempts to automatically start the FastAPI backend when the desktop app launches. This works correctly in development mode but fails silently in packaged (.app) applications.

### Background

From MACOS_APP_STATUS.md:
- Backend auto-start is disabled in packaged app due to PyInstaller limitations
- System Python lacks bundled dependencies (torch, ultralytics, etc.)
- macOS security restrictions on subprocess spawning from .app bundles

### Problem Statement

We need to understand the exact failure mechanism to implement the right fix:
1. Does subprocess.Popen fail entirely?
2. Does it start but fail to import dependencies?
3. Is the path resolution incorrect?
4. Are there permission issues?

## Acceptance Criteria

- [ ] Document exact failure point in backend startup
- [ ] Identify all error conditions encountered
- [ ] Test with and without system Python installed
- [ ] Document path resolution behavior
- [ ] Identify feasible solutions
- [ ] Create detailed findings document
- [ ] Recommend approach for fixes

## Implementation Details

### Files to Analyze

| File | Purpose |
|------|---------|
| `services/backend_manager.py` | Current implementation |
| `MACOS_APP_STATUS.md` | Known issues documentation |
| `build_config/build_macos.spec` | PyInstaller configuration |
| Desktop app logs | Runtime behavior |

### Investigation Steps

1. **Test in Development Mode**
   - Verify backend starts correctly
   - Log all startup steps
   - Confirm health check works

2. **Build and Test Packaged App**
   ```bash
   cd build_config
   pyinstaller build_macos.spec
   open dist/LocopilotCVVR.app
   ```
   - Monitor logs during startup
   - Check for subprocess errors
   - Verify path resolution

3. **Test Python Availability**
   - Test with system Python installed
   - Test without system Python
   - Test with Python in different locations

4. **Test Dependency Loading**
   - Check if uvicorn is importable
   - Test FastAPI import
   - Verify torch/ultralytics availability

5. **Analyze sys._MEIPASS**
   - Log actual _MEIPASS value
   - Check Contents/Frameworks vs Contents/Resources
   - Verify backend code location

## Testing Strategy

### Manual Testing

- [ ] Run app in development mode - backend should start
- [ ] Build packaged app - backend should fail
- [ ] Install system Python - test if behavior changes
- [ ] Remove system Python - test fallback
- [ ] Check logs for error patterns

### Logging Points

Add comprehensive logging to:
```python
def start_backend(self) -> bool:
    logger.info(f"=== Backend Startup Debug ===")
    logger.info(f"Is packaged: {self._is_packaged()}")
    logger.info(f"sys.executable: {sys.executable}")
    logger.info(f"sys._MEIPASS: {getattr(sys, '_MEIPASS', 'Not set')}")
    logger.info(f"Backend path: {backend_path}")
    logger.info(f"Python exe: {python_exe}")
    logger.info(f"Command: {' '.join(cmd)}")
    # ... attempt startup, log results
```

## Technical Notes

### Known PyInstaller Behaviors

- `sys.executable` is the frozen executable, not Python
- `sys._MEIPASS` points to temporary extraction directory
- Subprocesses inherit frozen environment
- System Python is independent installation

### Questions to Answer

1. What does `subprocess.Popen()` return in packaged app?
2. Does the process start but fail immediately?
3. What is the actual error message?
4. Can we capture stderr from the subprocess?
5. Is there a Python interpreter we can use?

## Implementation Checklist

- [ ] Add debug logging to BackendManager
- [ ] Build and run packaged app
- [ ] Collect logs from multiple test scenarios
- [ ] Document all error conditions
- [ ] Test on clean macOS system
- [ ] Test on system with Python installed
- [ ] Analyze subprocess behavior
- [ ] Document findings in investigation report
- [ ] Recommend solution approach

## Expected Findings Document

Create `INVESTIGATION_FINDINGS.md` with:

```markdown
# Backend Startup Investigation Findings

## Test Environment
- macOS version:
- Python installed: Yes/No
- Python location:
- PyInstaller version:

## Failure Analysis

### Scenario 1: No System Python
[What happens, error messages, logs]

### Scenario 2: System Python Present
[What happens, error messages, logs]

### Scenario 3: Manual Backend Start
[Does app detect it?]

## Root Causes
1. [Cause 1]
2. [Cause 2]

## Recommended Solutions
1. [Solution 1 - pros/cons]
2. [Solution 2 - pros/cons]

## Next Steps
[Specific tasks to implement solution]
```

## Related Tasks

- Task 0002: Implement recommended fix
- Task 0003: Add health check endpoint
- Task 0004: Improve error messaging

## Status History

| Date | Status | Notes |
|------|--------|-------|
| 2024-01-16 | pending | Initial creation |

---

## Notes

This is a research task. Take time to thoroughly understand the problem before proposing solutions.
