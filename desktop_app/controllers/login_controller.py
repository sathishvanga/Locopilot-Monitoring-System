"""
Login controller - Connects login view with authentication service
"""

from typing import Optional
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal, QThread

from ..views.login_view import LoginView
from ..services.auth_service import AuthService
from ..models.auth_models import LoginResponse
from ..utils.logger import get_logger


logger = get_logger(__name__)


class LoginWorker(QThread):
    """
    Worker thread for login operation
    
    Prevents UI blocking during network requests
    """
    
    # Signals
    login_success = Signal(object)  # LoginResponse
    login_failed = Signal(str)  # error_message
    
    def __init__(self, auth_service: AuthService, username: str, password: str):
        """
        Initialize login worker
        
        Args:
            auth_service: Authentication service
            username: Username
            password: Password
        """
        super().__init__()
        self.auth_service = auth_service
        self.username = username
        self.password = password
    
    def run(self) -> None:
        """Execute login in background thread"""
        logger.info(f"Login worker started for user: {self.username}")
        
        try:
            success, user_info, error = self.auth_service.login(
                self.username,
                self.password
            )
            
            if success and user_info:
                self.login_success.emit(user_info)
            else:
                self.login_failed.emit(error or "Login failed")
        except Exception as e:
            logger.error(f"Login worker error: {e}", exc_info=True)
            self.login_failed.emit(f"Unexpected error: {str(e)}")
    
    def __del__(self) -> None:
        """Cleanup worker thread"""
        try:
            if self.isRunning():
                self.terminate()
                self.wait(3000)  # Wait up to 3 seconds
        except Exception:
            pass


class LoginController(QObject):
    """
    Controller for login view
    
    Handles login logic and connects view to authentication service
    """
    
    # Signals
    login_success = Signal()
    
    def __init__(self, view: LoginView, auth_service: AuthService):
        """
        Initialize login controller
        
        Args:
            view: Login view
            auth_service: Authentication service
        """
        super().__init__()
        self.view = view
        self.auth_service = auth_service
        self.login_worker = None
        
        # Connect signals
        self.view.login_clicked.connect(self._on_login_clicked)
        
        logger.info("Login controller initialized")
    
    def _on_login_clicked(self, username: str, password: str):
        """
        Handle login button click
        
        Args:
            username: Entered username
            password: Entered password
        """
        # Validate inputs
        if not username:
            self._show_error("Please enter your mobile number")
            return
        
        if not password:
            self._show_error("Please enter your password")
            return
        
        # Set loading state
        self.view.set_loading(True)
        logger.info(f"Initiating login for user: {username}")
        
        # Cleanup previous worker if exists
        if self.login_worker is not None and self.login_worker.isRunning():
            self.login_worker.terminate()
            self.login_worker.wait(1000)
        
        # Create and start login worker
        self.login_worker = LoginWorker(self.auth_service, username, password)
        self.login_worker.login_success.connect(self._on_login_success)
        self.login_worker.login_failed.connect(self._on_login_failed)
        self.login_worker.finished.connect(lambda: self.view.set_loading(False))
        self.login_worker.finished.connect(self._cleanup_worker)
        self.login_worker.start()
    
    def _cleanup_worker(self) -> None:
        """Cleanup worker thread after completion"""
        if self.login_worker is not None:
            try:
                if self.login_worker.isRunning():
                    self.login_worker.terminate()
                    self.login_worker.wait(1000)
            except Exception as e:
                logger.warning(f"Error cleaning up login worker: {e}")
            finally:
                self.login_worker = None
    
    def _on_login_success(self, user_info: LoginResponse) -> None:
        """
        Handle successful login
        
        Args:
            user_info: LoginResponse object
        """
        logger.info(f"Login successful: {user_info.name} ({user_info.mobileNumber})")
        
        # Clear password field
        self.view.clear_inputs()
        
        # Show success message (optional)
        # QMessageBox.information(
        #     self.view,
        #     "Login Successful",
        #     f"Welcome, {user_info.name}!"
        # )
        
        # Emit success signal to navigate to main view
        self.login_success.emit()
    
    def _on_login_failed(self, error_message: str):
        """
        Handle failed login
        
        Args:
            error_message: Error description
        """
        logger.warning(f"Login failed: {error_message}")
        self._show_error(error_message)
        
        # Clear password field
        self.view.clear_inputs()
    
    def _show_error(self, message: str):
        """
        Show error message dialog
        
        Args:
            message: Error message
        """
        QMessageBox.critical(
            self.view,
            "Login Error",
            message
        )

