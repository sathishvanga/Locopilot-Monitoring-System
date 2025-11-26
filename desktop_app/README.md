# Locopilot CVVR Desktop Application

A cross-platform desktop application for managing CVVR trips and video uploads.

## Features

- **User Authentication**: Secure login with remote API
- **Trip Management**: View pending trips in a table
- **Video Upload**: Upload videos for processing
- **Local Processing**: Process videos using local FastAPI backend
- **S3 Upload**: Upload videos and evidence clips to S3
- **Progress Tracking**: Real-time progress updates for uploads

## Architecture

The application follows MVC (Model-View-Controller) architecture:

- **Models** (`models/`): Pydantic data models for API communication
- **Views** (`views/`): PySide6 UI components
- **Controllers** (`controllers/`): Business logic connecting views and services
- **Services** (`services/`): API clients and processing services
- **Utils** (`utils/`): Configuration, logging, and utilities

## Requirements

- Python 3.11+
- PySide6 (Qt for Python)
- See `requirements_desktop.txt` for full dependencies

## Installation

### For Development

```bash
# Navigate to desktop_app directory
cd desktop_app

# Install dependencies
pip install -r requirements_desktop.txt

# Run the application
python -m desktop_app.main
```

### For macOS

```bash
# Install dependencies
pip install -r requirements_desktop.txt

# Build with PyInstaller
cd build_config
pyinstaller build_macos.spec

# Output will be in dist/LocopilotCVVR.app
```

### For Windows

The Windows build is automated via GitHub Actions:

1. Push code to GitHub
2. GitHub Actions will automatically build the Windows EXE
3. Download from GitHub Releases

## Configuration

The application can be configured via environment variables with the `CVVR_` prefix:

- `CVVR_API_BASE_URL`: Remote API base URL (default: https://api.mindcoinapps.com/ai_demo_api)
- `CVVR_LOCAL_BACKEND_URL`: Local FastAPI backend URL (default: http://localhost:8000)
- `CVVR_LOG_LEVEL`: Logging level (default: INFO)
- `CVVR_DEBUG`: Enable debug mode (default: False)

## Usage

### Login

1. Launch the application
2. Enter your mobile number and password
3. Select organization (Demo)
4. Click "Log In"

### Upload Video

1. After login, you'll see the pending trips table
2. Click "Upload Video" button for a trip
3. Select a video file
4. The application will:
   - Process the video locally (requires FastAPI backend running)
   - Upload the original video to S3
   - Upload evidence clips to S3
5. Progress updates will be shown in real-time

### Requirements for Video Processing

- Local FastAPI backend must be running on port 8000
- Start the backend with: `uvicorn app.main:app --reload`

## Testing

```bash
# Run unit tests
cd desktop_app
python -m pytest tests/

# Or with unittest
python -m unittest discover tests/
```

## Project Structure

```
desktop_app/
├── main.py                    # Application entry point
├── requirements_desktop.txt   # Dependencies
├── README.md                  # This file
├── models/                    # Pydantic models
│   ├── auth_models.py
│   └── trip_models.py
├── services/                  # Business logic
│   ├── auth_service.py
│   ├── trip_service.py
│   ├── upload_service.py
│   └── local_processing_service.py
├── controllers/               # UI controllers
│   ├── login_controller.py
│   └── trips_controller.py
├── views/                     # PySide6 UI
│   ├── login_view.py
│   ├── trips_view.py
│   └── widgets/
│       └── progress_widget.py
├── utils/                     # Utilities
│   ├── config.py
│   ├── logger.py
│   └── api_client.py
├── resources/                 # Assets
│   └── logo.png
├── tests/                     # Unit tests
│   ├── test_models.py
│   └── test_services.py
└── build_config/              # Build configs
    ├── build_macos.spec
    └── build_windows.spec
```

## API Integration

### Remote API

- **Base URL**: https://api.mindcoinapps.com/ai_demo_api
- **Login**: POST /auth/user/loginByMobilePasswordReCaptchaEncrypt
- **Get Trips**: GET /cvvr/cvvrTrips/getAllPendingTrips
- **Upload**: POST /amazonUpload/uploadWithFolder

### Local Backend

- **Base URL**: http://localhost:8000
- **Process Video**: POST /api/v1/video/process
- **Health Check**: GET /health

## Troubleshooting

### Backend Not Running

If you see "Backend Not Running" error:

1. Start the local FastAPI backend:
   ```bash
   cd /path/to/project
   uvicorn app.main:app --reload
   ```

2. Verify it's running:
   ```bash
   curl http://localhost:8000/health
   ```

### Login Issues

- Check your internet connection
- Verify credentials
- Check API status

### Upload Failures

- Ensure video file is valid format (.mp4, .avi, .mov, etc.)
- Check file size (max 2GB)
- Verify local backend is running
- Check S3 upload permissions

## License

Copyright © 2024 MINDCOIN Services

## Support

For issues and questions, contact: info@mindcoinservices.com

