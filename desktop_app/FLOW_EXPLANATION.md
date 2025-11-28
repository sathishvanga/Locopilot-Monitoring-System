# Desktop App Flow Explanation

This document explains the complete flow of the Locopilot CVVR Desktop Application from startup to video upload.

## Architecture Overview

The application follows **MVC (Model-View-Controller)** architecture:

- **Models** (`models/`): Pydantic data models for API communication
- **Views** (`views/`): PySide6 UI components (LoginView, TripsView)
- **Controllers** (`controllers/`): Business logic connecting views and services
- **Services** (`services/`): API clients and processing services
- **Utils** (`utils/`): Configuration, logging, and utilities

---

## 1. Application Startup Flow

### Entry Point: `launcher.py` → `main.py`

```
launcher.py (PyInstaller entry point)
    ↓
main.py::main()
    ↓
QApplication initialization
    ↓
MainWindow creation
    ↓
Backend auto-start (if enabled)
    ↓
Show LoginView
```

**Detailed Steps:**

1. **Launcher (`launcher.py`)**:
   - Sets up Python path for packaged/development mode
   - Configures SSL certificates for HTTPS
   - Sets multiprocessing start method (spawn)
   - Enables faulthandler for crash reporting
   - Calls `desktop_app.main.main()`

2. **Main Function (`main.py::main()`)**:
   - Initializes logging and settings
   - Creates QApplication (Qt event loop)
   - Sets application properties (name, version, organization)
   - Enables high DPI scaling
   - Creates and shows MainWindow
   - Starts Qt event loop

3. **MainWindow Initialization (`main.py::MainWindow.__init__`)**:
   - Creates BackendManager (manages FastAPI backend process)
   - Initializes services:
     - `AuthService` - User authentication
     - `TripService` - Fetch pending trips
     - `UploadService` - S3 uploads (legacy, kept for compatibility)
     - `LocalProcessingService` - Video processing via local backend
   - Sets up UI (LoginView and TripsView in QStackedWidget)
   - Sets up controllers (LoginController, TripsController)
   - Auto-starts backend if `CVVR_AUTO_START_BACKEND=true` (default)

4. **Backend Auto-Start (`main.py::_start_backend_async`)**:
   - Uses QTimer to start backend asynchronously (non-blocking)
   - BackendManager starts FastAPI backend using Gunicorn
   - Backend runs on `http://localhost:8000`

---

## 2. Login Flow

```
User enters credentials
    ↓
LoginView emits login_clicked signal
    ↓
LoginController::_on_login_clicked()
    ↓
LoginWorker (QThread) - Background login
    ↓
AuthService::login() - API call
    ↓
POST /auth/user/loginByMobilePassword
    ↓
LoginResponse received
    ↓
AuthService stores token
    ↓
LoginController emits login_success signal
    ↓
MainWindow::_on_login_success()
    ↓
Switch to TripsView
    ↓
TripsController loads trips
```

**Detailed Steps:**

1. **User Input** (`views/login_view.py`):
   - User enters mobile number and password
   - Clicks "Log In" button
   - View emits `login_clicked` signal with credentials

2. **Controller Handling** (`controllers/login_controller.py`):
   - `LoginController::_on_login_clicked()` validates inputs
   - Sets view to loading state
   - Creates `LoginWorker` (QThread) to avoid blocking UI
   - Starts worker thread

3. **Background Login** (`controllers/login_controller.py::LoginWorker`):
   - Worker calls `AuthService::login(username, password)`
   - Makes POST request to `/auth/user/loginByMobilePassword`
   - Parses response and extracts JWT token
   - Emits `login_success` or `login_failed` signal

4. **Authentication Service** (`services/auth_service.py`):
   - Validates and sanitizes inputs
   - Creates `LoginRequest` model
   - Makes API call via `APIClient`
   - Parses `LoginAPIResponse` (wrapped response)
   - Stores token in `AuthState`
   - Returns `LoginResponse` with user info

5. **Success Handling**:
   - `LoginController::_on_login_success()` clears password field
   - Emits `login_success` signal to MainWindow
   - MainWindow switches to TripsView
   - Updates auth tokens in services
   - TripsController automatically loads trips

---

## 3. Trips View Flow

```
TripsView displayed
    ↓
TripsController::_load_trips()
    ↓
LoadTripsWorker (QThread) - Background fetch
    ↓
TripService::get_pending_trips()
    ↓
GET /cvvr/cvvrTrips/getAllPendingTrips
    ↓
Trips loaded and displayed in table
```

**Detailed Steps:**

1. **View Initialization** (`views/trips_view.py`):
   - Displays table with pending trips
   - Shows "Upload Video" button for each trip
   - Shows "Refresh" and "Logout" buttons

2. **Controller Initialization** (`controllers/trips_controller.py`):
   - `TripsController` connects view signals
   - Sets auth token from `AuthService`
   - Automatically calls `_load_trips()` on init

3. **Loading Trips**:
   - `_load_trips()` creates `LoadTripsWorker` (QThread)
   - Worker calls `TripService::get_pending_trips()`
   - Makes GET request to `/cvvr/cvvrTrips/getAllPendingTrips`
   - Parses `TripsAPIResponse` (wrapped response)
   - Returns list of `TripModel` objects

4. **Display Trips**:
   - `_on_trips_loaded()` receives trips list
   - Calls `view.load_trips(trips)` to populate table
   - Updates UI with trip data (UUID, vehicle number, etc.)

---

## 4. Video Upload Flow (Simplified Workflow)

```
User clicks "Upload Video" button
    ↓
TripsController::_on_upload_clicked()
    ↓
File dialog - User selects video
    ↓
UploadService::validate_file()
    ↓
TripsController::_start_upload_workflow()
    ↓
UploadProcessWorker (QThread) - Background processing
    ↓
LocalProcessingService::process_and_upload_video()
    ↓
POST /api/v1/video/process-and-upload (Local Backend)
    ↓
Backend processes video (YOLO detection)
    ↓
Backend uploads original video to S3
    ↓
Backend uploads evidence clips to S3
    ↓
Response with S3 URLs
    ↓
Upload success - UI updated
```

**Detailed Steps:**

1. **User Action** (`views/trips_view.py`):
   - User clicks "Upload Video" button for a trip
   - View emits `upload_clicked` signal with trip UUID

2. **Controller Handling** (`controllers/trips_controller.py`):
   - `_on_upload_clicked()` checks if backend is running
   - Shows warning if backend is not available (once per session)
   - Opens file dialog for video selection
   - Validates file using `UploadService::validate_file()`
   - Calls `_start_upload_workflow()`

3. **Upload Workflow**:
   - Creates `UploadProcessWorker` (QThread)
   - Gets auth token from `AuthService`
   - Starts worker thread

4. **Background Processing** (`controllers/trips_controller.py::UploadProcessWorker`):
   - Worker calls `LocalProcessingService::process_and_upload_video()`
   - This is the **simplified workflow** - single API call handles everything

5. **Local Processing Service** (`services/local_processing_service.py`):
   - Checks if backend is running (`is_backend_running()`)
   - Validates video file exists and is not empty
   - Prepares multipart form data:
     - `video_file`: Video file
     - `tripId`: Trip UUID
     - `subFolderName`: "cvvr"
     - `authToken`: JWT token (for S3 upload)
   - Makes POST request to `/api/v1/video/process-and-upload`

6. **Backend Processing** (`app/controllers/video_controller.py`):
   - Receives video file and parameters
   - Processes video using YOLO model (activity detection)
   - Extracts evidence clips
   - Uploads original video to S3 (using auth token)
   - Uploads evidence clips to S3
   - Returns response with:
     - `video_url`: S3 URL of original video
     - `evidence_clips`: List of S3 URLs for clips
     - `activities_count`: Number of activities detected
     - `clips_uploaded`: Number of clips uploaded

7. **Response Handling**:
   - `LocalProcessingService` parses response
   - Returns `ProcessingResult` with success status and URLs
   - Worker emits `upload_success` or `upload_failed` signal

8. **UI Update**:
   - `_on_upload_success()` updates button state to "completed"
   - Shows success message
   - `_on_upload_failed()` updates button state to "error"
   - Shows error message

---

## 5. Backend Management Flow

```
Application Startup
    ↓
BackendManager::start_backend()
    ↓
Check if backend already running
    ↓
If not running:
    - Find backend path (packaged or development)
    - Find gunicorn_config.py
    - Start Gunicorn subprocess
    - Wait for health check
    ↓
Backend running on localhost:8000
    ↓
Application Shutdown
    ↓
BackendManager::stop_backend()
    ↓
Graceful termination (SIGTERM)
    ↓
Force kill if needed (SIGKILL)
```

**Detailed Steps:**

1. **Backend Manager** (`services/backend_manager.py`):
   - Detects if running in packaged mode (PyInstaller)
   - Finds backend path (works in both packaged and dev mode)
   - Finds `gunicorn_config.py` or uses inline settings
   - Uses system Python (if packaged) or current Python (if dev)
   - Starts Gunicorn subprocess with:
     - `app.main:app` (FastAPI application)
     - Config from `gunicorn_config.py`
     - Bind to `127.0.0.1:8000`
   - Waits for backend to become healthy (health check)

2. **Health Checking** (`utils/backend_health.py`):
   - Checks `GET /health` endpoint
   - Verifies backend responds within timeout
   - Returns True if healthy, False otherwise

3. **Shutdown**:
   - On application close, `MainWindow::closeEvent()` is called
   - Calls `BackendManager::stop_backend()`
   - Only stops if backend was started by us
   - Sends SIGTERM for graceful shutdown
   - Sends SIGKILL if graceful shutdown fails

---

## 6. Service Layer Architecture

### AuthService
- **Purpose**: User authentication with remote API
- **Methods**:
  - `login(username, password)` - Authenticate user
  - `logout()` - Clear auth state
  - `get_token()` - Get JWT token
  - `get_user_info()` - Get user info
- **State**: Stores token in `AuthState` object

### TripService
- **Purpose**: Fetch pending trips from remote API
- **Methods**:
  - `get_pending_trips()` - Fetch all pending trips
  - `refresh_trips()` - Alias for get_pending_trips
- **Auth**: Requires JWT token

### UploadService
- **Purpose**: Upload files to S3 (legacy, kept for compatibility)
- **Methods**:
  - `upload_file()` - Upload single file to S3
  - `upload_multiple_files()` - Upload multiple files
  - `validate_file()` - Validate file before upload
- **Note**: Main workflow now uses backend endpoint, but this service is kept for validation

### LocalProcessingService
- **Purpose**: Interface with local FastAPI backend for video processing
- **Methods**:
  - `process_and_upload_video()` - **Main method** - Single call handles everything
  - `is_backend_running()` - Check backend health
  - `wait_for_backend()` - Wait for backend to become available
- **Endpoint**: `/api/v1/video/process-and-upload`

### BackendManager
- **Purpose**: Manage FastAPI backend process lifecycle
- **Methods**:
  - `start_backend()` - Start backend process
  - `stop_backend()` - Stop backend process
  - `is_backend_running()` - Check if backend is running
  - `get_backend_status()` - Get backend status info

---

## 7. Threading Model

The application uses **QThread** for all network operations to prevent UI blocking:

1. **LoginWorker**: Handles login API call
2. **LoadTripsWorker**: Handles trips fetch API call
3. **UploadProcessWorker**: Handles video processing and upload

**Benefits**:
- UI remains responsive during network operations
- Progress updates via signals
- Proper cleanup on completion

**Signal Flow**:
```
Worker Thread
    ↓
Emit signal (login_success, trips_loaded, upload_success)
    ↓
Controller slot receives signal
    ↓
Update UI (main thread)
```

---

## 8. Error Handling

### Network Errors
- **ConnectionError**: "Cannot connect to server"
- **Timeout**: "Request timed out"
- **HTTPError**: Parses error message from response

### Validation Errors
- File validation (size, format, existence)
- Input validation (username, password)
- Backend availability checks

### User Feedback
- Error dialogs (QMessageBox)
- Button state updates (processing, error, completed)
- Loading indicators

---

## 9. Configuration

Configuration is managed via `utils/config.py`:

- **Environment Variables** (with `CVVR_` prefix):
  - `CVVR_API_BASE_URL`: Remote API URL
  - `CVVR_LOCAL_BACKEND_URL`: Local backend URL
  - `CVVR_AUTO_START_BACKEND`: Auto-start backend (default: true)
  - `CVVR_LOG_LEVEL`: Logging level
  - `CVVR_DEBUG`: Debug mode

- **Settings Object**:
  - Centralized configuration
  - Default values for all settings
  - Type-safe access

---

## 10. Key Design Decisions

### 1. Simplified Upload Workflow
- **Old**: Desktop app processes video → uploads video → uploads clips (3 separate operations)
- **New**: Single backend endpoint handles everything (1 operation)
- **Benefit**: Video sent once, more efficient, less error-prone

### 2. Backend Auto-Start
- Backend starts automatically on app launch
- Uses Gunicorn for production-ready server
- Works in both packaged and development mode

### 3. MVC Architecture
- Clear separation of concerns
- Controllers handle business logic
- Views handle UI only
- Services handle API communication

### 4. Threading for Network Operations
- All network calls in background threads
- UI remains responsive
- Progress updates via signals

### 5. Wrapped API Responses
- Handles both new wrapped format and legacy format
- Backward compatible
- Type-safe with Pydantic models

---

## Summary

The desktop app flow is:

1. **Startup**: Launcher → Main → MainWindow → Backend auto-start → LoginView
2. **Login**: User credentials → LoginController → AuthService → API → Token stored → TripsView
3. **Trips**: TripsController → TripService → API → Display trips
4. **Upload**: User selects video → TripsController → LocalProcessingService → Backend → Processing + S3 upload → Success
5. **Shutdown**: MainWindow close → BackendManager → Stop backend

The entire workflow is designed to be:
- **User-friendly**: Clear UI, progress indicators, error messages
- **Efficient**: Single API call for upload workflow
- **Robust**: Error handling, validation, health checks
- **Maintainable**: MVC architecture, clear separation of concerns

