"""
Login view UI
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor
from pathlib import Path
import os


class LoginView(QWidget):
    """
    Login page UI
    
    Exact replication of the CVVRS login design
    """
    
    # Signals
    login_clicked = Signal(str, str)  # username, password
    
    def __init__(self, parent=None):
        """
        Initialize login view
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.password_visible = False
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI components matching exact design"""
        # Main layout with gradient background
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Set background gradient (purple - exact colors from design)
        self.setStyleSheet("""
            LoginView {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #C5B3E6,
                    stop: 1 #B8A5D9
                );
            }
        """)
        
        # Add vertical spacer at top to center content
        main_layout.addStretch(1)
        
        # Center container
        center_layout = QHBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        
        # Add horizontal spacers for centering
        center_layout.addStretch(1)
        
        # Login card (white rounded with shadow)
        login_card = QFrame()
        login_card.setObjectName("loginCard")
        login_card.setStyleSheet("""
            QFrame#loginCard {
                background-color: white;
                border-radius: 20px;
            }
        """)
        login_card.setFixedSize(420, 580)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 10)
        login_card.setGraphicsEffect(shadow)
        
        card_layout = QVBoxLayout(login_card)
        card_layout.setContentsMargins(40, 50, 40, 40)
        card_layout.setSpacing(15)
        
        # Logo - Video camera icon in purple circle
        logo_container = QHBoxLayout()
        logo_container.setAlignment(Qt.AlignCenter)
        
        logo_label = QLabel("🎥")
        logo_font = QFont()
        logo_font.setPointSize(42)
        logo_label.setFont(logo_font)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedSize(90, 90)
        logo_label.setStyleSheet("""
            QLabel {
                background-color: #7C5CFC;
                border-radius: 45px;
            }
        """)
        logo_container.addWidget(logo_label)
        card_layout.addLayout(logo_container)
        
        card_layout.addSpacing(5)
        
        # Title - CVVRS
        title_label = QLabel("CVVRS")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(26)
        title_font.setBold(True)
        title_font.setFamily("Arial")
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #333333;")
        card_layout.addWidget(title_label)
        
        card_layout.addSpacing(25)
        
        # Organization dropdown with building icon
        self.org_combo = QComboBox()
        self.org_combo.addItem("🏢  Demo")
        self.org_combo.setFixedHeight(50)
        self.org_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 12px 15px;
                font-size: 15px;
                background-color: #FAFAFA;
                color: #666666;
            }
            QComboBox:hover {
                border: 1px solid #7C5CFC;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 15px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #999;
                margin-right: 10px;
            }
        """)
        card_layout.addWidget(self.org_combo)
        
        # Mobile number input with user icon
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("👤  Mobile Number")
        self.username_input.setFixedHeight(50)
        self.username_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 12px 15px;
                padding-left: 15px;
                font-size: 15px;
                background-color: white;
                color: #333333;
            }
            QLineEdit:focus {
                border: 2px solid #7C5CFC;
                padding: 11px 14px;
                padding-left: 14px;
            }
            QLineEdit::placeholder {
                color: #B0B0B0;
            }
        """)
        card_layout.addWidget(self.username_input)
        
        # Password input container with lock icon and eye icon
        password_container = QWidget()
        password_layout = QHBoxLayout(password_container)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.setSpacing(0)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("🔒  Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedHeight(50)
        self.password_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 12px 50px 12px 15px;
                font-size: 15px;
                background-color: white;
                color: #333333;
            }
            QLineEdit:focus {
                border: 2px solid #7C5CFC;
                padding: 11px 49px 11px 14px;
            }
            QLineEdit::placeholder {
                color: #B0B0B0;
            }
        """)
        password_layout.addWidget(self.password_input)
        
        # Eye icon button for show/hide password
        self.toggle_password_btn = QPushButton("👁️")
        self.toggle_password_btn.setFixedSize(40, 40)
        self.toggle_password_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_password_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 18px;
                margin-right: 10px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 5px;
            }
        """)
        self.toggle_password_btn.clicked.connect(self._toggle_password_visibility)
        
        # Position eye icon over password field
        password_container.setFixedHeight(50)
        password_layout.addWidget(self.toggle_password_btn)
        password_layout.setAlignment(self.toggle_password_btn, Qt.AlignRight | Qt.AlignVCenter)
        password_layout.setContentsMargins(0, 0, 5, 0)
        
        # Adjust layout to overlay button
        self.toggle_password_btn.setParent(self.password_input)
        self.toggle_password_btn.move(
            self.password_input.width() - 45,
            5
        )
        
        card_layout.addWidget(self.password_input)
        
        # Skip reCAPTCHA section (as per requirements)
        card_layout.addSpacing(15)
        
        # Login button - exact purple from design
        self.login_button = QPushButton("Log In")
        self.login_button.setFixedHeight(50)
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #7C5CFC;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 17px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #6C4EFF;
            }
            QPushButton:pressed {
                background-color: #5A3DD9;
            }
            QPushButton:disabled {
                background-color: #D0D0D0;
                color: #888888;
            }
        """)
        self.login_button.clicked.connect(self._on_login_clicked)
        card_layout.addWidget(self.login_button)
        
        card_layout.addStretch()
        
        # Footer - "Powered by MINDCOIN" with exact styling
        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignCenter)
        footer_layout.setSpacing(5)
        
        powered_label = QLabel("Powered by")
        powered_label.setStyleSheet("color: #888888; font-size: 13px;")
        footer_layout.addWidget(powered_label)
        
        # MINDCOIN text with orange dot
        mindcoin_label = QLabel("MINDC🟠IN")
        mindcoin_label.setStyleSheet("color: #333333; font-size: 13px; font-weight: 600; letter-spacing: 0.5px;")
        footer_layout.addWidget(mindcoin_label)
        
        card_layout.addLayout(footer_layout)
        
        # Add card to center layout
        center_layout.addWidget(login_card)
        center_layout.addStretch(1)
        
        main_layout.addLayout(center_layout)
        
        # Add vertical spacer at bottom
        main_layout.addStretch(1)
        
        # Contact info footer (absolute bottom of screen)
        contact_label = QLabel("Contact : +91-97016 58885 | Email: info@mindcoinservices.com | Ver-06.11.01")
        contact_label.setAlignment(Qt.AlignCenter)
        contact_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 11px;
                padding: 15px;
                background-color: transparent;
            }
        """)
        main_layout.addWidget(contact_label)
        
        # Connect Enter key to login
        self.username_input.returnPressed.connect(self._on_login_clicked)
        self.password_input.returnPressed.connect(self._on_login_clicked)
    
    def resizeEvent(self, event):
        """Handle resize to reposition toggle password button"""
        super().resizeEvent(event)
        if hasattr(self, 'toggle_password_btn') and hasattr(self, 'password_input'):
            self.toggle_password_btn.move(
                self.password_input.width() - 45,
                5
            )
    
    def _toggle_password_visibility(self):
        """Toggle password visibility"""
        if self.password_visible:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_btn.setText("👁️")
            self.password_visible = False
        else:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_btn.setText("👁️‍🗨️")
            self.password_visible = True
    
    def _on_login_clicked(self):
        """Handle login button click"""
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        self.login_clicked.emit(username, password)
    
    def set_loading(self, loading: bool):
        """
        Set loading state
        
        Args:
            loading: True to disable inputs, False to enable
        """
        self.username_input.setEnabled(not loading)
        self.password_input.setEnabled(not loading)
        self.org_combo.setEnabled(not loading)
        self.login_button.setEnabled(not loading)
        
        if loading:
            self.login_button.setText("Logging in...")
        else:
            self.login_button.setText("Log In")
    
    def clear_inputs(self):
        """Clear input fields"""
        self.password_input.clear()
    
    def get_credentials(self) -> tuple[str, str]:
        """
        Get entered credentials
        
        Returns:
            tuple[str, str]: (username, password)
        """
        return self.username_input.text().strip(), self.password_input.text().strip()

