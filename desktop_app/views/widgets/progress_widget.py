"""
Progress widget for upload/processing status
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ProgressWidget(QWidget):
    """
    Custom widget for showing upload/processing progress
    
    Combines a progress bar with status label
    """
    
    # Signals
    cancelled = Signal()
    
    def __init__(self, parent=None):
        """
        Initialize progress widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(25)
        layout.addWidget(self.progress_bar)
        
        # Set fixed height for widget
        self.setFixedHeight(60)
    
    def set_status(self, status: str):
        """
        Update status text
        
        Args:
            status: Status message
        """
        self.status_label.setText(status)
    
    def set_progress(self, value: int):
        """
        Update progress value
        
        Args:
            value: Progress value (0-100)
        """
        self.progress_bar.setValue(min(max(0, value), 100))
    
    def set_state_ready(self):
        """Set widget to ready state"""
        self.status_label.setText("Ready")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("")
    
    def set_state_uploading(self, progress: int = 0):
        """
        Set widget to uploading state
        
        Args:
            progress: Upload progress (0-100)
        """
        self.status_label.setText("Uploading...")
        self.progress_bar.setValue(progress)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #6C4EFF;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #6C4EFF;
            }
        """)
    
    def set_state_processing(self, progress: int = 0):
        """
        Set widget to processing state
        
        Args:
            progress: Processing progress (0-100)
        """
        self.status_label.setText("Processing...")
        self.progress_bar.setValue(progress)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #FF9500;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #FF9500;
            }
        """)
    
    def set_state_completed(self):
        """Set widget to completed state"""
        self.status_label.setText("Completed ✓")
        self.status_label.setStyleSheet("color: #34C759; font-weight: bold;")
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #34C759;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #34C759;
            }
        """)
    
    def set_state_error(self, error_message: str = "Error"):
        """
        Set widget to error state
        
        Args:
            error_message: Error message to display
        """
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: #FF3B30; font-weight: bold;")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #FF3B30;
                border-radius: 5px;
                text-align: center;
                background-color: #ffebee;
            }
            QProgressBar::chunk {
                background-color: #FF3B30;
            }
        """)
    
    def reset(self):
        """Reset widget to initial state"""
        self.set_state_ready()
        self.status_label.setStyleSheet("")

