# Requirements: Fix Backend Not Running on Upload

## Functional Requirements

### FR-001: Backend Health Endpoint
**Priority:** High
**Category:** Backend

The FastAPI backend MUST provide a `/health` endpoint that returns the backend's operational status.

**Acceptance Criteria:**
- Endpoint accessible at `GET /health`
- Returns 200 OK when backend is healthy
- Returns JSON with status information
- Response time < 100ms
- Documented in API documentation

### FR-002: Enhanced Backend Health Checking
**Priority:** High
**Category:** Desktop App

The BackendManager MUST reliably detect if the backend is running and responsive.

**Acceptance Criteria:**
- Uses `/health` endpoint for verification
- Returns both running status and health information
- Handles connection failures gracefully
- Timeout after 2 seconds
- Logs health check results

### FR-003: Improved Backend Path Resolution
**Priority:** High
**Category:** Desktop App

The BackendManager MUST correctly locate the backend code in both development and packaged modes.

**Acceptance Criteria:**
- Correctly resolves path in development mode
- Correctly resolves path in packaged macOS .app
- Handles both Frameworks and Resources MEIPASS locations
- Logs resolved path for debugging
- Validates backend path exists before starting

### FR-004: Better Error Reporting
**Priority:** Medium
**Category:** Desktop App

The system MUST provide clear, actionable error messages when backend fails to start.

**Acceptance Criteria:**
- Specific error codes for different failure types
- User-friendly error messages
- Suggested solutions for each error type
- Error messages logged for debugging
- UI shows helpful dialog with next steps

### FR-005: Configuration Options
**Priority:** Medium
**Category:** Desktop App

The system MUST support multiple backend operation modes.

**Acceptance Criteria:**
- `backend_mode` configuration option
- Modes: auto, manual, remote
- `auto_start_backend` flag
- `fallback_to_remote` option
- Settings documented in README

### FR-006: Graceful Fallback
**Priority:** High
**Category:** Desktop App

The application MUST continue functioning even if backend fails to start.

**Acceptance Criteria:**
- App launches successfully without backend
- Remote processing available as fallback
- User informed of fallback mode
- Can switch to local backend if started later
- No crashes from backend failures

---

## Non-Functional Requirements

### NFR-001: Performance
**Category:** Performance

- Backend health check: < 100ms response time
- Backend startup: < 10 seconds
- Health check timeout: 2 seconds max
- No UI blocking during backend operations

### NFR-002: Reliability
**Category:** Reliability

- Backend startup success rate: 95%+ in packaged apps
- Health check accuracy: 99%+
- No false positives (claiming running when not)
- No false negatives (claiming not running when it is)

### NFR-003: Usability
**Category:** User Experience

- Clear status indicators for backend state
- Error messages understandable by non-technical users
- One-click fallback to remote processing
- No required manual configuration

### NFR-004: Maintainability
**Category:** Code Quality

- Comprehensive logging for debugging
- Unit test coverage: 80%+
- Code follows project conventions
- Type hints on all functions
- Docstrings on public methods

### NFR-005: Compatibility
**Category:** Platform Support

- macOS 11+ (Big Sur and later)
- Both Intel and Apple Silicon
- Python 3.11+
- Works with and without system Python

### NFR-006: Security
**Category:** Security

- No sensitive data in error messages
- No exposure of system paths to users
- Subprocess spawning uses safe defaults
- No arbitrary code execution

---

## Technical Constraints

### TC-001: PyInstaller Limitations
**Description:** PyInstaller frozen executables cannot directly spawn Python subprocesses

**Impact:**
- Cannot use `sys.executable` in packaged app
- Must find alternative Python interpreter
- May need to bundle Python environment

**Mitigation:**
- Try system Python as fallback
- Document manual backend startup option
- Consider bundling Python in future version

### TC-002: macOS Security Restrictions
**Description:** macOS restricts subprocess spawning from .app bundles

**Impact:**
- Subprocess.Popen may fail in packaged apps
- Security prompts for unsigned apps
- Gatekeeper may block backend startup

**Mitigation:**
- Use start_new_session flag
- Document code signing requirements
- Provide manual startup alternative

### TC-003: Dependency Management
**Description:** System Python may not have required dependencies

**Impact:**
- uvicorn, fastapi, torch, etc. may not be installed
- Cannot assume any packages available
- Different Python versions have different package locations

**Mitigation:**
- Check for dependencies before starting
- Provide clear error messages
- Document dependency installation
- Consider bundling dependencies

---

## User Stories

### US-001: Seamless Local Processing
**As a** desktop app user
**I want** the backend to start automatically without manual steps
**So that** I can immediately use local video processing

**Acceptance Criteria:**
- App starts backend automatically on launch
- No configuration required from user
- Works on fresh Python installation
- Backend ready within 10 seconds

### US-002: Clear Status Visibility
**As a** desktop app user
**I want** to know if local processing is available
**So that** I can decide whether to wait or use remote processing

**Acceptance Criteria:**
- Status indicator shows backend state
- Distinguish between "starting", "running", "failed"
- Estimated time until ready shown
- Can manually refresh status

### US-003: Easy Troubleshooting
**As a** desktop app user
**I want** clear guidance when backend fails
**So that** I can fix the problem myself

**Acceptance Criteria:**
- Error message explains what went wrong
- Lists specific steps to resolve
- Provides alternative options
- Links to documentation

### US-004: Manual Backend Control
**As a** power user
**I want** to start the backend manually
**So that** I can have more control over the process

**Acceptance Criteria:**
- Configuration option to disable auto-start
- Instructions for manual backend startup
- App detects manually-started backend
- Can stop backend from app

---

## Dependencies

### Internal Dependencies

| Dependency | Component | Reason |
|------------|-----------|--------|
| BackendManager | services/backend_manager.py | Core component being fixed |
| Settings | utils/config.py | Configuration management |
| Logger | utils/logger.py | Logging infrastructure |
| TripsController | controllers/trips_controller.py | Error handling integration |

### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| FastAPI | >=0.104.0 | Backend framework |
| uvicorn | >=0.24.0 | ASGI server |
| requests | >=2.31.0 | HTTP client for health checks |
| PySide6 | >=6.6.0 | UI framework |

---

## Success Criteria

The feature is considered successful when:

1. **Backend Starts Reliably**
   - 95%+ success rate in packaged macOS apps
   - Startup completes within 10 seconds
   - Works on clean macOS systems

2. **Health Monitoring Works**
   - `/health` endpoint responds < 100ms
   - Health checks accurate 99%+ of time
   - False positive/negative rate < 1%

3. **User Experience Improved**
   - Clear status indication
   - Helpful error messages
   - Seamless fallback to remote
   - Zero support tickets about "backend not running"

4. **Code Quality Maintained**
   - All tests pass
   - 80%+ test coverage on new code
   - No regressions in existing features
   - Follows project conventions

5. **Documentation Complete**
   - README updated with backend info
   - Troubleshooting guide created
   - MACOS_APP_STATUS.md updated
   - Code comments comprehensive

---

## Out of Scope

The following are explicitly OUT OF SCOPE for this feature:

1. **Bundling Full Python Environment** - Deferred to future version due to complexity
2. **Windows Support** - Focus on macOS first, Windows later
3. **Backend Performance Optimization** - Separate feature
4. **Automatic Backend Updates** - Not needed for v1
5. **Backend Service Management** - Too complex for initial implementation
6. **Multi-Backend Support** - Single backend sufficient

---

## Acceptance Testing

### Test Scenario 1: Fresh macOS System
**Given:** Clean macOS system with no Python installed
**When:** User launches the desktop app
**Then:**
- App launches successfully
- Backend startup attempted
- Clear error message if Python missing
- Remote processing available as fallback

### Test Scenario 2: System with Python
**Given:** macOS system with Python 3.11+ and dependencies
**When:** User launches the desktop app
**Then:**
- Backend starts automatically
- Health check confirms running
- Status indicator shows "Running"
- Local processing available

### Test Scenario 3: Backend Already Running
**Given:** Backend manually started before app launch
**When:** User launches the desktop app
**Then:**
- App detects existing backend
- Reuses existing backend instance
- Does not spawn duplicate process
- Status indicator shows "Running"

### Test Scenario 4: Backend Crashes During Use
**Given:** App running with backend active
**When:** Backend process crashes
**Then:**
- App detects backend failure via health check
- Shows user-friendly error message
- Offers to restart backend
- Falls back to remote processing

### Test Scenario 5: Port Already in Use
**Given:** Another process using port 8000
**When:** App tries to start backend
**Then:**
- Detects port conflict
- Shows specific error message
- Suggests using different port or stopping other process
- Falls back to remote processing

---

## Metrics and Monitoring

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Backend startup success rate | ≥95% | Logs analysis |
| Average startup time | <10s | Performance monitoring |
| Health check response time | <100ms | API monitoring |
| User error rate | <5% | Error logs |
| Support tickets (backend) | <2/month | Support system |

### Monitoring Requirements

- Log all backend startup attempts
- Log startup failures with error codes
- Track health check success/failure
- Monitor backend process lifecycle
- Alert on repeated failures

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2024-01-16 | Initial requirements | System |

---

*Last Updated: 2024*
