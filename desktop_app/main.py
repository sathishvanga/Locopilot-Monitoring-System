"""
Locopilot CVVR Desktop Application

Main entry point for the desktop application
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QStackedWidget, QMainWindow
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon

from .views.login_view import LoginView
from .views.trips_view import TripsView
from .controllers.login_controller import LoginController
from .controllers.trips_controller import TripsController
from .services.auth_service import AuthService
from .services.trip_service import TripService
from .services.upload_service import UploadService
from .services.local_processing_service import LocalProcessingService
from .services.backend_manager import BackendManager
from .utils.logger import setup_logging, get_logger
from .utils.config import get_settings


# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window
    
    Manages navigation between login and trips views
    """
    
    def __init__(self):
        """Initialize main window"""
        super().__init__()
        
        # Initialize backend manager (but don't start yet to avoid blocking)
        self.backend_manager = BackendManager()
        
        # Initialize services
        self.auth_service = AuthService()
        self.trip_service = TripService()
        self.upload_service = UploadService()
        self.local_processing_service = LocalProcessingService()
        
        # Setup UI
        self._setup_ui()
        self._setup_controllers()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Setup main window UI"""
        # Window properties
        self.setWindowTitle(settings.app_name)
        self.resize(settings.window_width, settings.window_height)
        
        # Center window on screen
        screen = QApplication.primaryScreen().geometry()
        window_rect = self.frameGeometry()
        window_rect.moveCenter(screen.center())
        self.move(window_rect.topLeft())
        
        # Create stacked widget for navigation
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create views
        self.login_view = LoginView()
        self.trips_view = TripsView()
        
        # Add views to stack
        self.stacked_widget.addWidget(self.login_view)
        self.stacked_widget.addWidget(self.trips_view)
        
        # Start with login view
        self.stacked_widget.setCurrentWidget(self.login_view)
        
        # Apply global stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
        """)
    
    def showEvent(self, event):
        """
        Handle show event - start backend after window is visible
        
        Args:
            event: Show event
        """
        super().showEvent(event)
        
        # Start backend after window is shown to avoid blocking UI
        if settings.auto_start_backend and not hasattr(self, '_backend_started'):
            self._backend_started = True
            # Use QTimer to start backend in the next event loop iteration
            QTimer.singleShot(100, self._start_backend_async)
    
    def _start_backend_async(self):
        """Start backend asynchronously after UI is shown"""
        logger.info("Auto-starting local backend...")
        backend_started = self.backend_manager.start_backend()
        if backend_started:
            logger.info("Backend is ready")
        else:
            logger.warning("Backend failed to start - video processing will not be available")
    
    def _setup_controllers(self):
        """Setup controllers and connect signals"""
        # Login controller
        self.login_controller = LoginController(
            self.login_view,
            self.auth_service
        )
        self.login_controller.login_success.connect(self._on_login_success)
        
        # Trips controller
        self.trips_controller = TripsController(
            self.trips_view,
            self.auth_service,
            self.trip_service,
            self.upload_service,
            self.local_processing_service
        )
        self.trips_controller.logout_requested.connect(self._on_logout_requested)
    
    def _on_login_success(self):
        """Handle successful login - navigate to trips view"""
        logger.info("Navigating to trips view")
        
        # Update service tokens
        token = self.auth_service.get_token()
        if token:
            self.trip_service.set_auth_token(token)
            self.upload_service.set_auth_token(token)
        
        # Switch to trips view
        self.stacked_widget.setCurrentWidget(self.trips_view)
        
        # Reload trips controller to refresh data
        self.trips_controller = TripsController(
            self.trips_view,
            self.auth_service,
            self.trip_service,
            self.upload_service,
            self.local_processing_service
        )
        self.trips_controller.logout_requested.connect(self._on_logout_requested)
    
    def _on_logout_requested(self):
        """Handle logout - navigate to login view"""
        logger.info("Navigating to login view")
        
        # Switch to login view
        self.stacked_widget.setCurrentWidget(self.login_view)
        
        # Clear login inputs
        self.login_view.clear_inputs()
    
    def closeEvent(self, event):
        """
        Handle window close event
        
        Args:
            event: Close event
        """
        logger.info("Application closing")
        
        # Stop backend if it was started by us
        if hasattr(self, 'backend_manager'):
            logger.info("Stopping backend...")
            self.backend_manager.stop_backend()
        
        event.accept()


def main():
    """
    Main entry point for the application
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(settings.app_name)
    app.setApplicationVersion(settings.app_version)
    app.setOrganizationName("MINDCOIN Services")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info("Application started successfully")
    
    # Start event loop
    exit_code = app.exec()
    
    logger.info(f"Application exited with code: {exit_code}")
    logger.info("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

