# Testing Strategy: Fix Backend Not Running on Upload

## Testing Overview

This document outlines the comprehensive testing approach for the backend auto-start fixes.

## Test Pyramid

```
        ┌─────────────┐
        │   Manual    │  ← Integration, UAT
        ├─────────────┤
        │ Integration │  ← Cross-component tests
        ├─────────────┤
        │    Unit     │  ← Individual functions
        └─────────────┘
```

##Unit Tests

### Backend Health Endpoint Tests

**File:** `app/tests/test_health.py`

```python
def test_health_endpoint_returns_200():
    """Test health endpoint returns 200 OK"""

def test_health_endpoint_returns_json():
    """Test health endpoint returns valid JSON"""

def test_health_endpoint_has_required_fields():
    """Test response contains status, timestamp, version"""
```

**Coverage Target:** 100%

### BackendManager Tests

**File:** `desktop_app/tests/test_backend_manager.py`

```python
def test_is_packaged_development_mode():
    """Test _is_packaged() returns False in development"""

def test_is_packaged_frozen_mode():
    """Test _is_packaged() returns True when frozen"""

def test_get_backend_path_development():
    """Test path resolution in development mode"""

def test_get_backend_path_packaged():
    """Test path resolution in packaged mode"""

def test_is_backend_running_when_running():
    """Test health check detects running backend"""

def test_is_backend_running_when_not_running():
    """Test health check detects backend is down"""

def test_start_backend_success():
    """Test successful backend startup"""

def test_start_backend_python_not_found():
    """Test error handling when Python not found"""

def test_start_backend_port_in_use():
    """Test error handling when port already used"""
```

**Coverage Target:** 80%+

## Integration Tests

### End-to-End Backend Startup

**Test:** Backend starts and becomes healthy
```python
def test_e2e_backend_startup():
    """Test complete backend startup workflow"""
    manager = BackendManager()
    success, error_code, error_msg = manager.start_backend()

    assert success is True
    assert manager.is_backend_running()[0] is True

    # Cleanup
    manager.stop_backend()
```

### Health Check Integration

**Test:** Health check correctly identifies backend state
```python
def test_health_check_integration():
    """Test health check with real backend"""
    # Start backend
    # Wait for startup
    # Check health
    # Verify response
```

## Manual Testing

### Scenario 1: Fresh macOS Install
- [ ] Install app on clean macOS system
- [ ] Launch app
- [ ] Verify error message if Python missing
- [ ] Install Python
- [ ] Restart app
- [ ] Verify backend starts

### Scenario 2: Packaged App
- [ ] Build .app with PyInstaller
- [ ] Launch packaged app
- [ ] Verify backend behavior
- [ ] Check health indicator
- [ ] Try video upload

### Scenario 3: Manual Backend
- [ ] Start backend manually
- [ ] Launch app
- [ ] Verify app detects existing backend
- [ ] No duplicate processes

### Scenario 4: Error Conditions
- [ ] Port 8000 already in use
- [ ] Python not in PATH
- [ ] Backend crashes during use
- [ ] Network timeout

## Performance Testing

### Metrics to Measure

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Health check response time | <100ms | API timing |
| Backend startup time | <10s | Process timing |
| Health check CPU usage | <1% | Activity Monitor |
| Memory overhead | <50MB | Memory profiler |

## Platform Testing

### macOS Versions
- [ ] macOS 11 (Big Sur)
- [ ] macOS 12 (Monterey)
- [ ] macOS 13 (Ventura)
- [ ] macOS 14 (Sonoma)

### Architectures
- [ ] Intel (x86_64)
- [ ] Apple Silicon (arm64)

## Regression Testing

### Existing Features to Verify
- [ ] Video upload still works
- [ ] Login/authentication unchanged
- [ ] Trip viewing functional
- [ ] Settings preserved
- [ ] Logout works

## Test Automation

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements_desktop.txt
      - name: Run tests
        run: pytest tests/ -v --cov
```

## Test Data

### Mock Data Requirements
- Sample video files (small, for testing)
- Mock API responses
- Mock health check responses
- Test configuration files

## Success Criteria

Testing is complete when:
- [ ] All unit tests pass (100% pass rate)
- [ ] Integration tests pass
- [ ] Manual test scenarios completed
- [ ] Code coverage ≥80%
- [ ] No critical bugs found
- [ ] Performance targets met
- [ ] Regression tests pass

---

*Last Updated: 2024*
