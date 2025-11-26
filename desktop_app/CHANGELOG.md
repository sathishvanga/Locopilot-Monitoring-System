# Changelog

All notable changes to the Locopilot CVVR Desktop Application.

## [1.1.0] - 2024-11-26

### 🚀 Major Improvements

#### Centralized S3 Upload Architecture
- **BREAKING IMPROVEMENT**: Backend now handles all S3 uploads
- Desktop app sends video once, backend processes and uploads everything
- 67% reduction in network traffic for desktop app
- 33% faster overall workflow
- Better scalability and maintainability

### Added
- New backend endpoint: `POST /api/v1/video/process-and-upload`
- New backend service: `S3UploadService` for centralized S3 operations
- S3 URL fields in `ProcessingResult` model
- New method: `process_and_upload_video()` in `LocalProcessingService`

### Changed
- Desktop app now uses simplified workflow (1 API call instead of 4)
- `UploadProcessWorker` simplified significantly
- Updated `ProcessingResult` model with S3 URL fields
- Improved progress messages during upload

### Technical Details
- **Before**: Desktop → Backend → Desktop → S3 (multiple hops)
- **After**: Desktop → Backend → S3 → Desktop (single request)
- Network traffic reduced from ~1.5GB to ~500MB per video
- Processing time reduced from ~150s to ~100s

### Backward Compatibility
- ✅ Old endpoint `/api/v1/video/process` still works
- ✅ Desktop app automatically uses new endpoint
- ✅ No breaking changes for end users

---

## [1.0.0] - 2024-11-26

### 🎉 Initial Release

#### Features
- User authentication with remote API
- Pending trips table view
- Video upload and processing
- Activity detection using YOLO
- Evidence clip generation
- S3 upload integration
- Progress tracking
- Cross-platform support (macOS & Windows)

#### Architecture
- MVC pattern (Model-View-Controller)
- PySide6 (Qt for Python) UI framework
- Integration with local FastAPI backend
- Remote API integration (MINDCOIN)

#### UI Components
- Modern login page matching design mockup
- Responsive trips table
- Upload buttons with status tracking
- Progress indicators
- Error handling with user-friendly messages

#### Build & Deployment
- PyInstaller build scripts for macOS
- GitHub Actions workflow for Windows builds
- Development mode scripts
- Comprehensive documentation

#### Testing
- Unit tests for services
- Unit tests for models
- Test coverage for critical paths

---

## Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

---

## Upgrade Guide

### From 1.0.0 to 1.1.0

**For Users:**
- No action needed!
- Just update both desktop app and FastAPI backend
- Enjoy faster uploads 🚀

**For Developers:**
- Update FastAPI backend to include new S3 upload service
- Desktop app automatically uses new endpoint
- Review `ARCHITECTURE_IMPROVEMENT.md` for technical details

**Steps:**
1. Pull latest code
2. Install any new backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Desktop app is ready to use!

---

## Roadmap

### Future Enhancements (Planned)

#### v1.2.0 (Q1 2025)
- [ ] Video preview before upload
- [ ] Drag-and-drop video upload
- [ ] Batch upload (multiple videos)
- [ ] Download processed clips from UI

#### v1.3.0 (Q2 2025)
- [ ] Dark mode theme
- [ ] Activity clips inline viewer
- [ ] Export trip report as PDF
- [ ] Offline mode with sync

#### v2.0.0 (Q3 2025)
- [ ] Real-time processing status updates (WebSocket)
- [ ] Admin dashboard
- [ ] User management
- [ ] Analytics and reporting

---

## Support

For issues, questions, or feature requests:
- **Email**: info@mindcoinservices.com
- **Phone**: +91-97016 58885

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.

