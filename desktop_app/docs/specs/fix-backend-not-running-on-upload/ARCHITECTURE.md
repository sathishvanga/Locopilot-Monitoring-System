# Architecture: Backend Auto-Start System

## Overview

This document describes the architecture of the backend auto-start system, analyzes the current implementation, identifies problems, and proposes solutions.

---

## Current Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                Desktop Application                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │              main.py (Entry Point)                 │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │          launcher.py (Bootstrap)                   │  │
│  │  - Initializes logging                             │  │
│  │  - Starts BackendManager                           │  │
│  │  - Launches PySide6 UI                             │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │         BackendManager Service                     │  │
│  │                                                     │  │
│  │  Methods:                                           │  │
│  │  - _is_packaged() → bool                           │  │
│  │  - _get_backend_path() → Path                      │  │
│  │  - is_backend_running() → bool                     │  │
│  │  - start_backend() → bool                          │  │
│  │  - stop_backend() → None                           │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                                │
│             Spawns subprocess (fails in packaged app)    │
│                         │                                │
└─────────────────────────┼────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           FastAPI Backend (Subprocess)                   │
│                                                          │
│  - app/main.py                                           │
│  - YOLO model integration                                │
│  - Video processing                                      │
│  - S3 upload                                             │
│  - Runs on port 8000                                     │
└──────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. BackendManager (`services/backend_manager.py`)

**Responsibilities:**
- Detect if running from packaged app
- Locate backend code path
- Check if backend is running
- Start backend subprocess
- Monitor backend health
- Stop backend gracefully

**Current Implementation:**

```python
class BackendManager:
    def start_backend(self) -> bool:
        # Get backend path
        backend_path = self._get_backend_path()

        # Get Python executable
        python_exe = sys.executable  # Problem: packaged exe, not Python

        if self._is_packaged():
            # Try to find system Python
            system_python = shutil.which('python3')  # May not exist
            if system_python:
                python_exe = system_python

        # Build uvicorn command
        cmd = [python_exe, "-m", "uvicorn", "app.main:app", ...]

        # Start subprocess
        self.backend_process = subprocess.Popen(cmd, ...)  # Fails!
```

**Issues:**
1. System Python may not exist
2. System Python lacks dependencies (torch, ultralytics, etc.)
3. Subprocess spawning restricted in macOS .app bundles
4. No fallback if startup fails

#### 2. Path Resolution

**Development Mode:**
```
/Users/user/project/
├── desktop_app/
│   ├── main.py
│   └── services/backend_manager.py
└── app/
    └── main.py  ← Backend found at ../app
```

**Packaged Mode (macOS):**
```
LocopilotCVVR.app/
└── Contents/
    ├── MacOS/
    │   └── LocopilotCVVR  ← Executable
    ├── Frameworks/  ← sys._MEIPASS may point here
    └── Resources/
        └── app/  ← Backend should be here
            └── main.py
```

**Problem:** sys._MEIPASS inconsistent (Frameworks vs Resources)

#### 3. Health Checking

**Current:**
```python
def is_backend_running(self) -> bool:
    # Check port open
    sock.connect_ex(('localhost', 8000))

    # Try health endpoint
    response = requests.get(f"{url}/health")  # No /health endpoint!
    return response.status_code == 200
```

**Problem:** No `/health` endpoint exists in backend

---

## Problem Analysis

### Problem 1: PyInstaller Subprocess Limitations

**Root Cause:**
- `sys.executable` in packaged app is the frozen executable, not Python
- Cannot spawn Python subprocesses from frozen executable
- System Python is separate installation, lacks dependencies

**Evidence:**
- Works in development (regular Python)
- Fails in packaged app (frozen executable)
- Documented PyInstaller limitation

**Impact:**
- Backend never starts in packaged apps
- Users cannot use local processing

### Problem 2: Missing Health Endpoint

**Root Cause:**
- Backend (`app/main.py`) has no `/health` endpoint
- Health check assumes it exists
- Returns False even if backend is running

**Impact:**
- Cannot verify backend is truly running
- Cannot distinguish between "not started" and "started but unresponsive"

### Problem 3: Inadequate Error Handling

**Root Cause:**
- Startup failures logged but not surfaced to user
- Generic "backend unavailable" warning
- No troubleshooting guidance

**Impact:**
- Users don't understand what's wrong
- Cannot self-resolve issues
- Increased support burden

### Problem 4: No Fallback Strategy

**Root Cause:**
- Either backend works or it doesn't
- No graceful degradation
- No alternative deployment options

**Impact:**
- Features completely unavailable
- Poor user experience

---

## Proposed Architecture

### Solution 1: Bundled Backend Option

**Approach:** Bundle backend dependencies with app

**Architecture:**
```
LocopilotCVVR.app/
└── Contents/
    └── Resources/
        ├── app/  ← Backend code
        ├── venv/  ← Virtual environment with dependencies
        │   ├── bin/python3
        │   └── lib/python3.11/site-packages/
        └── models/  ← YOLO model
```

**Implementation:**
```python
def _get_bundled_python(self) -> Optional[Path]:
    """Get bundled Python interpreter"""
    if self._is_packaged():
        resources = Path(sys._MEIPASS).parent / 'Resources'
        venv_python = resources / 'venv' / 'bin' / 'python3'
        if venv_python.exists():
            return venv_python
    return None

def start_backend(self) -> bool:
    # Try bundled Python first
    python_exe = self._get_bundled_python()

    if not python_exe:
        # Fall back to system Python
        python_exe = shutil.which('python3')

    # ... rest of implementation
```

**Pros:**
- Self-contained solution
- No system dependencies
- Reliable across systems

**Cons:**
- Larger app size (+300MB for dependencies)
- More complex build process
- Requires bundling virtual environment

### Solution 2: Health Check Enhancement

**Add `/health` endpoint to backend:**

```python
# app/main.py

@app.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns:
        dict: Status information
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

**Enhanced health checking:**

```python
def is_backend_running(self) -> tuple[bool, Optional[dict]]:
    """
    Check if backend is running and get status

    Returns:
        tuple: (is_running: bool, status: Optional[dict])
    """
    try:
        response = requests.get(
            f"{settings.local_backend_url}/health",
            timeout=2
        )
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None
```

### Solution 3: Improved Error Handling

**Detailed error tracking:**

```python
class BackendStartupError:
    """Track specific backend startup failures"""
    PYTHON_NOT_FOUND = "python_not_found"
    DEPENDENCIES_MISSING = "dependencies_missing"
    PORT_IN_USE = "port_in_use"
    STARTUP_TIMEOUT = "startup_timeout"
    UNKNOWN = "unknown"

def start_backend(self) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Start backend with detailed error reporting

    Returns:
        tuple: (success: bool, error_code: Optional[str], error_msg: Optional[str])
    """
    # ... implementation with specific error detection
```

**User-friendly error messages:**

```python
ERROR_MESSAGES = {
    "python_not_found": (
        "Python not found on your system.\n\n"
        "Solutions:\n"
        "1. Install Python 3.11+ from python.org\n"
        "2. Use remote processing (slower but works)\n"
        "3. Start backend manually (see docs)"
    ),
    "dependencies_missing": (
        "Backend dependencies not installed.\n\n"
        "Solutions:\n"
        "1. Install requirements: pip install -r requirements.txt\n"
        "2. Use remote processing instead\n"
        "3. Contact support for help"
    ),
    # ... other messages
}
```

### Solution 4: Configuration-Based Fallback

**Add configuration options:**

```python
# utils/config.py

class Settings(BaseSettings):
    # Backend configuration
    auto_start_backend: bool = Field(
        default=True,
        description="Attempt to auto-start backend"
    )

    backend_mode: Literal["auto", "manual", "remote"] = Field(
        default="auto",
        description="Backend startup mode"
    )

    fallback_to_remote: bool = Field(
        default=True,
        description="Use remote processing if local fails"
    )
```

**Mode-based behavior:**

- **auto**: Try to start backend, fall back to remote
- **manual**: Expect user to start backend separately
- **remote**: Skip local backend entirely

---

## Decision Matrix

### Solution Comparison

| Solution | Complexity | Reliability | App Size | User Impact |
|----------|-----------|-------------|----------|-------------|
| Bundled Backend | High | Very High | +300MB | Best (works always) |
| Health Check | Low | Medium | +0 | Good (better detection) |
| Error Handling | Medium | Medium | +0 | Good (better UX) |
| Config Fallback | Low | High | +0 | Good (flexibility) |

### Recommended Approach

**Phase 1:** Quick Wins (Tasks 0001-0004)
1. Add `/health` endpoint
2. Fix backend manager initialization
3. Improve error messages
4. Add configuration options

**Phase 2:** Long-term Solution (Future)
1. Bundle Python environment in app
2. Automated build process for bundling
3. Comprehensive testing across macOS versions

---

## Architecture Decision Records

### ADR-001: Use Health Check Endpoint

**Status:** Approved

**Context:** Need reliable way to verify backend is running

**Decision:** Add `/health` GET endpoint to FastAPI backend

**Rationale:**
- Standard REST API pattern
- Lightweight and fast
- Easy to implement
- Widely understood

**Consequences:**
- Backend must be modified
- Desktop app health check updated
- Better visibility into backend status

### ADR-002: Defer Full Backend Bundling

**Status:** Approved

**Context:** Bundling full Python environment is complex

**Decision:** Implement quick fixes first, bundle backend in future version

**Rationale:**
- Quick wins provide immediate value
- More time to perfect bundling approach
- Can test alternatives (separate service, etc.)

**Consequences:**
- Users may still need Python installed
- Manual backend option documented
- Remote processing as reliable fallback

### ADR-003: Configuration-Based Modes

**Status:** Approved

**Context:** Different deployment scenarios need flexibility

**Decision:** Add `backend_mode` configuration option

**Rationale:**
- Supports multiple deployment models
- Users can choose what works for them
- Easy to maintain and test

**Consequences:**
- More configuration options
- Need to document each mode
- Testing complexity increases

---

## Technical Specifications

### Backend Health Endpoint

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-16T10:30:00Z",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: Backend is healthy
- `503 Service Unavailable`: Backend is starting up or unhealthy

### BackendManager Interface

**Updated Methods:**

```python
class BackendManager:
    def is_backend_running(self) -> tuple[bool, Optional[dict]]:
        """Check if backend is running, return status"""

    def start_backend(self) -> tuple[bool, Optional[str], Optional[str]]:
        """Start backend, return (success, error_code, error_msg)"""

    def get_backend_status(self) -> dict:
        """Get comprehensive backend status"""
```

---

## Testing Strategy

See `TESTING_STRATEGY.md` for comprehensive testing approach.

---

## Future Enhancements

### Bundled Backend (v2.0)

- Bundle complete Python virtual environment
- Include all dependencies
- Self-contained solution

### Separate Backend Service (v2.5)

- Backend as standalone service
- System service installation
- Always-on architecture

### Cloud Backend Option (v3.0)

- Remote backend deployment
- Multi-tenant support
- Reduced desktop app complexity

---

*Last Updated: 2024*
