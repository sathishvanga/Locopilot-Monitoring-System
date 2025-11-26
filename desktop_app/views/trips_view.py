"""
Trips view UI - Display pending trips with upload functionality
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor
from typing import List, Dict, Any

from ..models.trip_models import TripModel


class TripsView(QWidget):
    """
    Trips table view with upload functionality
    
    Displays pending trips in a table with upload buttons
    """
    
    # Signals
    upload_clicked = Signal(str)  # trip_uuid
    refresh_clicked = Signal()
    logout_clicked = Signal()
    
    def __init__(self, parent=None):
        """
        Initialize trips view
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.trips_data: List[TripModel] = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Set background
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
        """)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("CVVR Pending Trips")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #333;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Refresh button
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.setFixedSize(120, 40)
        self.refresh_button.setCursor(Qt.PointingHandCursor)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #6C4EFF;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A3DD9;
            }
            QPushButton:pressed {
                background-color: #4A2DB8;
            }
        """)
        self.refresh_button.clicked.connect(self.refresh_clicked.emit)
        header_layout.addWidget(self.refresh_button)
        
        # Logout button
        self.logout_button = QPushButton("Logout")
        self.logout_button.setFixedSize(100, 40)
        self.logout_button.setCursor(Qt.PointingHandCursor)
        self.logout_button.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E02B1F;
            }
            QPushButton:pressed {
                background-color: #C01B0E;
            }
        """)
        self.logout_button.clicked.connect(self.logout_clicked.emit)
        header_layout.addWidget(self.logout_button)
        
        main_layout.addLayout(header_layout)
        
        # Table widget
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                gridline-color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #E8E0FF;
            }
            QHeaderView::section {
                background-color: #6C4EFF;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # Define columns
        self.columns = [
            "UUID",
            "Date/Time",
            "From Station",
            "To Station",
            "Section",
            "Train No",
            "Loco No",
            "Created By",
            "Analysis Type",
            "Status",
            "Action"
        ]
        
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        
        # Configure table
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        
        # Set column widths
        header = self.table.horizontalHeader()
        for i, col in enumerate(self.columns):
            if col == "UUID":
                header.setSectionResizeMode(i, QHeaderView.Fixed)
                self.table.setColumnWidth(i, 250)
            elif col == "Action":
                header.setSectionResizeMode(i, QHeaderView.Fixed)
                self.table.setColumnWidth(i, 150)
            else:
                header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        main_layout.addWidget(self.table)
        
        # Status label
        self.status_label = QLabel("Loading trips...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 14px; padding: 20px;")
        main_layout.addWidget(self.status_label)
    
    def load_trips(self, trips: List[TripModel]):
        """
        Load trips into table
        
        Args:
            trips: List of trip models
        """
        self.trips_data = trips
        self.table.setRowCount(len(trips))
        
        for row, trip in enumerate(trips):
            # UUID
            uuid_item = QTableWidgetItem(trip.uuid[:20] + "..." if len(trip.uuid) > 20 else trip.uuid)
            uuid_item.setFlags(uuid_item.flags() & ~Qt.ItemIsEditable)
            uuid_item.setToolTip(trip.uuid)  # Full UUID on hover
            self.table.setItem(row, 0, uuid_item)
            
            # Date/Time
            datetime_item = QTableWidgetItem(trip.dateTime or "N/A")
            datetime_item.setFlags(datetime_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, datetime_item)
            
            # From Station
            from_item = QTableWidgetItem(trip.fromStation or trip.fromStationId or "N/A")
            from_item.setFlags(from_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, from_item)
            
            # To Station
            to_item = QTableWidgetItem(trip.toStation or trip.toStationId or "N/A")
            to_item.setFlags(to_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, to_item)
            
            # Section
            section_item = QTableWidgetItem(trip.sectionName or trip.sectionId or "N/A")
            section_item.setFlags(section_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 4, section_item)
            
            # Train No
            train_item = QTableWidgetItem(trip.trainNo or "N/A")
            train_item.setFlags(train_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 5, train_item)
            
            # Loco No
            loco_item = QTableWidgetItem(trip.locoNo or "N/A")
            loco_item.setFlags(loco_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6, loco_item)
            
            # Created By
            created_item = QTableWidgetItem(trip.createdBy or "N/A")
            created_item.setFlags(created_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 7, created_item)
            
            # Analysis Type
            analysis_item = QTableWidgetItem(trip.analysisType or "N/A")
            analysis_item.setFlags(analysis_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 8, analysis_item)
            
            # Status
            status_text = "Pending" if trip.status is None else str(trip.status)
            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 9, status_item)
            
            # Action button
            upload_button = QPushButton("📤 Upload Video")
            upload_button.setFixedHeight(35)
            upload_button.setCursor(Qt.PointingHandCursor)
            upload_button.setStyleSheet("""
                QPushButton {
                    background-color: #34C759;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2DB04A;
                }
                QPushButton:pressed {
                    background-color: #25993C;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
            upload_button.clicked.connect(lambda checked, uuid=trip.uuid: self.upload_clicked.emit(uuid))
            self.table.setCellWidget(row, 10, upload_button)
        
        # Update status
        if len(trips) == 0:
            self.status_label.setText("No pending trips found")
            self.status_label.show()
            self.table.hide()
        else:
            self.status_label.hide()
            self.table.show()
    
    def set_loading(self, loading: bool):
        """
        Set loading state
        
        Args:
            loading: True to show loading, False to hide
        """
        if loading:
            self.status_label.setText("Loading trips...")
            self.status_label.show()
            self.table.hide()
            self.refresh_button.setEnabled(False)
        else:
            self.refresh_button.setEnabled(True)
    
    def set_upload_button_state(self, trip_uuid: str, state: str, text: str = None):
        """
        Update upload button state for a specific trip
        
        Args:
            trip_uuid: Trip UUID
            state: Button state ('ready', 'uploading', 'processing', 'completed', 'error')
            text: Optional button text override
        """
        # Find row for this trip
        for row in range(self.table.rowCount()):
            uuid_item = self.table.item(row, 0)
            if uuid_item and trip_uuid in uuid_item.toolTip():
                button = self.table.cellWidget(row, 10)
                if isinstance(button, QPushButton):
                    if state == "uploading":
                        button.setEnabled(False)
                        button.setText(text or "⏳ Uploading...")
                    elif state == "processing":
                        button.setEnabled(False)
                        button.setText(text or "⚙️ Processing...")
                    elif state == "completed":
                        button.setEnabled(False)
                        button.setText("✓ Completed")
                        button.setStyleSheet("""
                            QPushButton {
                                background-color: #34C759;
                                color: white;
                                border: none;
                                border-radius: 6px;
                                font-size: 12px;
                                font-weight: bold;
                            }
                        """)
                    elif state == "error":
                        button.setEnabled(True)
                        button.setText(text or "❌ Retry")
                        button.setStyleSheet("""
                            QPushButton {
                                background-color: #FF3B30;
                                color: white;
                                border: none;
                                border-radius: 6px;
                                font-size: 12px;
                                font-weight: bold;
                            }
                            QPushButton:hover {
                                background-color: #E02B1F;
                            }
                        """)
                    else:  # ready
                        button.setEnabled(True)
                        button.setText(text or "📤 Upload Video")
                break
    
    def show_error(self, message: str):
        """
        Show error message
        
        Args:
            message: Error message to display
        """
        QMessageBox.critical(self, "Error", message)
    
    def show_info(self, message: str):
        """
        Show info message
        
        Args:
            message: Info message to display
        """
        QMessageBox.information(self, "Information", message)

