# Feature: Fix Backend Not Running on Video Upload

## Epic Description

This feature addresses the production issue where the local FastAPI backend fails to start automatically when the desktop application is launched, resulting in errors during video upload operations.

## Problem Statement

Users are experiencing the following issue:
- When attempting to upload videos in the desktop application, they receive an error indicating the backend is not running
- The local backend is supposed to auto-start when the app launches, but this fails in packaged (PyInstaller) applications
- This prevents users from utilizing local video processing features

## Current Behavior

### In Development Mode
- Backend starts successfully when app launches
- Local video processing works as expected
- Videos are processed locally before uploading to S3

### In Packaged App (macOS .app)
- Backend auto-start fails silently
- User sees warning dialog about backend not available
- Only remote processing (direct upload) works
- Local processing features unavailable

## Root Causes

Based on analysis of `MACOS_APP_STATUS.md` and `services/backend_manager.py`:

1. **PyInstaller Subprocess Limitations**
   - Packaged executable cannot easily spawn Python subprocesses
   - System Python lacks bundled dependencies (torch, ultralytics, etc.)
   - macOS security restrictions on subprocess spawning from .app bundles

2. **Path Resolution Issues**
   - Backend path detection fails in some macOS bundle configurations
   - sys._MEIPASS points to different locations (Frameworks vs Resources)

3. **Missing Dependencies**
   - System Python doesn't have required packages
   - Virtual environment not accessible from packaged app

## Goals

### Primary Goals
1. **Investigate Root Cause**: Thoroughly analyze why backend auto-start fails in packaged apps
2. **Implement Reliable Solution**: Fix backend initialization to work in packaged environment
3. **Add Health Monitoring**: Implement robust backend health checking
4. **Improve User Experience**: Better error messages and fallback handling

### Secondary Goals
1. Document limitations clearly
2. Provide alternative deployment options
3. Enable manual backend configuration

## User Stories

### Story 1: Auto-Start Backend in Packaged App
**As a** desktop app user
**I want** the backend to start automatically when I launch the app
**So that** I can use local video processing without manual setup

**Acceptance Criteria:**
- Backend starts automatically in packaged macOS app
- Startup completes within 10 seconds
- User sees success indication in UI
- No Python environment required from user

### Story 2: Clear Error Messages
**As a** desktop app user
**I want** clear error messages if the backend cannot start
**So that** I understand what's wrong and how to fix it

**Acceptance Criteria:**
- Error dialog explains the issue clearly
- Provides actionable steps to resolve
- Offers fallback to remote processing
- Doesn't block app usage

### Story 3: Backend Health Monitoring
**As a** desktop app developer
**I want** robust backend health checking
**So that** the app can detect and handle backend failures gracefully

**Acceptance Criteria:**
- Health check verifies backend is responsive
- Periodic health checks during runtime
- Automatic restart on failure (if possible)
- Logs health status for debugging

## Scope

### In Scope

#### Phase 1: Investigation
- Analyze backend startup failure in packaged apps
- Identify exact failure points
- Document current behavior
- Research alternative approaches

#### Phase 2: Core Fix
- Fix backend manager initialization
- Improve path resolution
- Add health check endpoint
- Implement better error handling

#### Phase 3: User Experience
- Update error messages
- Add status indicators
- Document known limitations
- Provide configuration options

### Out of Scope

- Bundling Python interpreter in app (too complex for v1)
- Separate backend as system service (future consideration)
- Windows-specific fixes (focus on macOS first)
- Backend performance optimization (separate feature)

## Success Metrics

### Technical Metrics
- Backend startup success rate: **95%+** in packaged apps
- Startup time: **< 10 seconds**
- Health check response time: **< 100ms**
- Zero crashes from backend-related errors

### User Experience Metrics
- Reduced support tickets about "backend not running"
- User can complete upload workflow without manual backend setup
- Clear understanding of fallback options

## Timeline

### Phase 1: Investigation (Tasks 0001)
**Duration:** 1-2 days
- Investigate root cause thoroughly
- Document findings
- Identify potential solutions

### Phase 2: Implementation (Tasks 0002-0003)
**Duration:** 3-5 days
- Fix backend manager
- Add health check
- Test in packaged environment

### Phase 3: Polish (Task 0004)
**Duration:** 1-2 days
- Improve error messages
- Update documentation
- Final testing

**Total Estimate:** 5-9 days

## Dependencies

### External Dependencies
- PyInstaller packaging process
- macOS .app bundle structure
- System Python availability
- FastAPI/uvicorn dependencies

### Internal Dependencies
- BackendManager service
- TripsController (for error handling)
- Configuration system
- Logging infrastructure

## Risks and Mitigations

### Risk 1: PyInstaller Fundamental Limitation
**Probability:** High
**Impact:** High
**Mitigation:**
- Research alternative approaches (standalone backend)
- Document limitations clearly
- Provide manual backend startup option
- Fall back to remote processing

### Risk 2: Platform-Specific Issues
**Probability:** Medium
**Impact:** Medium
**Mitigation:**
- Test on multiple macOS versions
- Document platform requirements
- Provide platform-specific instructions

### Risk 3: User Environment Variations
**Probability:** Medium
**Impact:** Low
**Mitigation:**
- Make backend optional
- Provide detailed troubleshooting guide
- Support remote processing fallback

## Alternative Solutions Considered

### Option 1: Bundle Python Interpreter in App
**Pros:**
- Complete self-contained solution
- No system Python required

**Cons:**
- Massive app size (1GB+)
- Complex build process
- Difficult to maintain

**Decision:** Not chosen for v1, consider for future

### Option 2: Separate Backend Service
**Pros:**
- Cleaner separation
- More reliable
- Easier updates

**Cons:**
- Requires separate installation
- More complex for users
- Service management complexity

**Decision:** Document as future option

### Option 3: Remote Backend Only
**Pros:**
- Simple, no local backend
- Always works

**Cons:**
- Slower (upload unprocessed video)
- Requires internet
- Less privacy

**Decision:** Available as fallback, not primary solution

## Related Features

- Video Upload Workflow
- Local Video Processing
- S3 Integration
- Error Handling System

## Documentation Updates Required

- README.md - Update installation instructions
- QUICK_START.md - Explain backend status
- MACOS_APP_STATUS.md - Update with fixes
- New troubleshooting guide

---

## Next Steps

1. Review and approve this feature overview
2. Create detailed task breakdown
3. Begin implementation with Task 0001
4. Regular progress updates

---

*Created: 2024*
*Status: Pending Approval*
