"""
Trips controller - Manages trips view and upload workflow
"""

import os
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal, QThread

from ..views.trips_view import TripsView
from ..services.auth_service import AuthService
from ..services.trip_service import TripService
from ..services.upload_service import UploadService
from ..services.local_processing_service import LocalProcessingService
from ..models.trip_models import TripModel
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class LoadTripsWorker(QThread):
    """Worker thread for loading trips"""
    
    trips_loaded = Signal(list)  # List[TripModel]
    load_failed = Signal(str)  # error_message
    
    def __init__(self, trip_service: TripService):
        super().__init__()
        self.trip_service = trip_service
    
    def run(self):
        """Load trips in background"""
        success, trips, error = self.trip_service.get_pending_trips()
        
        if success:
            self.trips_loaded.emit(trips)
        else:
            self.load_failed.emit(error or "Failed to load trips")


class UploadProcessWorker(QThread):
    """
    Worker thread for complete upload and processing workflow
    
    NEW SIMPLIFIED WORKFLOW:
    1. Call backend endpoint that handles EVERYTHING:
       - Process video locally (YOLO detection)
       - Upload original video to S3
       - Upload evidence clips to S3
    2. Get back S3 URLs
    
    Much more efficient - video is only sent once to backend!
    """
    
    progress_update = Signal(str)  # status_message
    upload_success = Signal()
    upload_failed = Signal(str)  # error_message
    
    def __init__(
        self,
        video_path: str,
        trip_uuid: str,
        local_processing: LocalProcessingService,
        auth_token: str = None
    ):
        super().__init__()
        self.video_path = video_path
        self.trip_uuid = trip_uuid
        self.local_processing = local_processing
        self.auth_token = auth_token
    
    def run(self):
        """Execute simplified upload workflow"""
        try:
            # Single call - backend handles everything!
            self.progress_update.emit("Processing and uploading to S3...")
            logger.info(f"Starting process and upload for trip {self.trip_uuid}")
            
            result = self.local_processing.process_and_upload_video(
                video_path=self.video_path,
                trip_id=self.trip_uuid,
                auth_token=self.auth_token
            )
            
            if not result.success:
                self.upload_failed.emit(result.error or "Processing and upload failed")
                return
            
            logger.info(
                f"Complete workflow finished - "
                f"Activities: {result.activities_count}, "
                f"Video URL: {result.video_url}, "
                f"Clips uploaded: {result.clips_uploaded}"
            )
            
            # Success
            self.progress_update.emit("Completed successfully!")
            self.upload_success.emit()
            
        except Exception as e:
            logger.error(f"Upload workflow error: {e}", exc_info=True)
            self.upload_failed.emit(f"Unexpected error: {str(e)}")


class TripsController(QObject):
    """
    Controller for trips view
    
    Handles trip loading, video upload, and processing workflow
    """
    
    # Signals
    logout_requested = Signal()
    
    def __init__(
        self,
        view: TripsView,
        auth_service: AuthService,
        trip_service: TripService,
        upload_service: UploadService,
        local_processing_service: LocalProcessingService
    ):
        """
        Initialize trips controller
        
        Args:
            view: Trips view
            auth_service: Authentication service
            trip_service: Trip service
            upload_service: Upload service (deprecated - kept for compatibility)
            local_processing_service: Local processing service
        """
        super().__init__()
        self.view = view
        self.auth_service = auth_service
        self.trip_service = trip_service
        self.upload_service = upload_service  # Kept for compatibility
        self.local_processing = local_processing_service
        
        self.load_worker = None
        self.upload_worker = None
        
        # Set auth tokens
        token = self.auth_service.get_token()
        if token:
            self.trip_service.set_auth_token(token)
            # upload_service no longer needed for main workflow
        
        # Connect signals
        self.view.upload_clicked.connect(self._on_upload_clicked)
        self.view.refresh_clicked.connect(self._load_trips)
        self.view.logout_clicked.connect(self._on_logout_clicked)
        
        logger.info("Trips controller initialized")
        
        # Load trips on initialization
        self._load_trips()
    
    def _load_trips(self):
        """Load pending trips from API"""
        logger.info("Loading pending trips")
        self.view.set_loading(True)
        
        # Create and start worker
        self.load_worker = LoadTripsWorker(self.trip_service)
        self.load_worker.trips_loaded.connect(self._on_trips_loaded)
        self.load_worker.load_failed.connect(self._on_load_failed)
        self.load_worker.start()
    
    def _on_trips_loaded(self, trips):
        """
        Handle successful trips load
        
        Args:
            trips: List of TripModel objects
        """
        logger.info(f"Loaded {len(trips)} trips")
        self.view.set_loading(False)
        self.view.load_trips(trips)
    
    def _on_load_failed(self, error_message: str):
        """
        Handle failed trips load
        
        Args:
            error_message: Error description
        """
        logger.error(f"Failed to load trips: {error_message}")
        self.view.set_loading(False)
        self.view.show_error(f"Failed to load trips:\n{error_message}")
    
    def _on_upload_clicked(self, trip_uuid: str):
        """
        Handle upload button click
        
        Args:
            trip_uuid: Trip UUID
        """
        logger.info(f"Upload clicked for trip: {trip_uuid}")
        
        # Check if backend is running (only show warning once per session)
        # Flag is stored in service to persist across controller recreations
        if not self.local_processing.is_backend_running() and not self.local_processing.backend_warning_shown:
            self.local_processing.backend_warning_shown = True
            response = QMessageBox.warning(
                self.view,
                "Local Processing Unavailable",
                "The local video processing backend is not running.\n\n"
                "You can still upload videos directly to the server.\n\n"
                "Note: Without local processing, videos will be uploaded as-is without analysis.\n\n"
                "Continue with upload?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes  # Default to Yes
            )
            
            if response == QMessageBox.No:
                return
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if not file_path:
            logger.info("Video selection cancelled")
            return
        
        # Validate file
        success, error = self.upload_service.validate_file(file_path)
        if not success:
            self.view.show_error(f"Invalid file:\n{error}")
            return
        
        logger.info(f"Selected video: {file_path}")
        
        # Start upload workflow
        self._start_upload_workflow(trip_uuid, file_path)
    
    def _start_upload_workflow(self, trip_uuid: str, video_path: str):
        """
        Start the complete upload and processing workflow
        
        NEW: Backend handles everything - processing + S3 upload in one call
        
        Args:
            trip_uuid: Trip UUID
            video_path: Path to video file
        """
        logger.info(f"Starting simplified workflow - Trip: {trip_uuid}, Video: {video_path}")
        
        # Update button state
        self.view.set_upload_button_state(trip_uuid, "processing")
        
        # Get auth token for S3 upload (backend needs it)
        auth_token = self.auth_service.get_token()
        
        # Create and start worker with NEW simplified workflow
        self.upload_worker = UploadProcessWorker(
            video_path,
            trip_uuid,
            self.local_processing,
            auth_token  # Pass token to backend
        )
        
        self.upload_worker.progress_update.connect(
            lambda msg: self._on_progress_update(trip_uuid, msg)
        )
        self.upload_worker.upload_success.connect(
            lambda: self._on_upload_success(trip_uuid)
        )
        self.upload_worker.upload_failed.connect(
            lambda error: self._on_upload_failed(trip_uuid, error)
        )
        
        self.upload_worker.start()
    
    def _on_progress_update(self, trip_uuid: str, message: str):
        """
        Handle progress update
        
        Args:
            trip_uuid: Trip UUID
            message: Progress message
        """
        logger.info(f"Progress [{trip_uuid}]: {message}")
        
        # Update button text based on progress
        if "Processing" in message:
            self.view.set_upload_button_state(trip_uuid, "processing", "⚙️ Processing...")
        elif "Uploading" in message:
            self.view.set_upload_button_state(trip_uuid, "uploading", "⏳ Uploading...")
    
    def _on_upload_success(self, trip_uuid: str):
        """
        Handle successful upload
        
        Args:
            trip_uuid: Trip UUID
        """
        logger.info(f"Upload workflow completed successfully for trip: {trip_uuid}")
        
        # Update button state
        self.view.set_upload_button_state(trip_uuid, "completed")
        
        # Show success message
        self.view.show_info("Video uploaded and processed successfully!")
    
    def _on_upload_failed(self, trip_uuid: str, error_message: str):
        """
        Handle failed upload
        
        Args:
            trip_uuid: Trip UUID
            error_message: Error description
        """
        logger.error(f"Upload workflow failed for trip {trip_uuid}: {error_message}")
        
        # Update button state
        self.view.set_upload_button_state(trip_uuid, "error", "❌ Retry")
        
        # Show error message
        self.view.show_error(f"Upload failed:\n{error_message}")
    
    def _on_logout_clicked(self):
        """Handle logout button click"""
        response = QMessageBox.question(
            self.view,
            "Logout",
            "Are you sure you want to logout?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            logger.info("User logging out")
            self.auth_service.logout()
            self.logout_requested.emit()

