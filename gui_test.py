import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QComboBox, QLineEdit,
                             QVBoxLayout, QHBoxLayout, QFileDialog)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QPainterPath, QColor, QFont
import sounddevice as sd

class SwitchButton(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 45)
        self._is_checked = False
        self._track_color = QColor("#E0E0E0")
        self._thumb_color = QColor("#FFFFFF")
        self._track_color_checked = QColor("#2196F3")  # Material Blue
        self._thumb_position = 2
        
        # Setup animation
        self.animation = QPropertyAnimation(self, b"position", self)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.setDuration(200)

    def get_position(self):
        return self._thumb_position

    def set_position(self, pos):
        self._thumb_position = pos
        self.update()

    position = Property(float, get_position, set_position)

    def mousePressEvent(self, event):
        self._is_checked = not self._is_checked
        self.animation.setStartValue(self._thumb_position)
        if self._is_checked:
            self.animation.setEndValue(27)
        else:
            self.animation.setEndValue(2)
        self.animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw track
        track_opacity = 0.6 if not self._is_checked else 1.0
        painter.setOpacity(track_opacity)
        track_color = self._track_color if not self._is_checked else self._track_color_checked
        painter.setBrush(track_color)
        painter.setPen(Qt.NoPen)
        track_path = QPainterPath()
        track_path.addRoundedRect(0, 8, 54, 18, 9, 9)
        painter.drawPath(track_path)

        # Draw ON/OFF text
        painter.setOpacity(1.0)
        painter.setPen(QColor("#FFFFFF"))
        font = QFont()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        if self._is_checked:
            painter.drawText(7, 20, "ON")
        else:
            painter.drawText(30, 20, "OFF")

        # Draw thumb
        painter.setOpacity(1.0)
        painter.setBrush(self._thumb_color)
        shadow = QColor(0, 0, 0, 30)
        painter.setPen(shadow)
        thumb_path = QPainterPath()
        thumb_path.addEllipse(self._thumb_position, 4, 26, 26)
        painter.drawPath(thumb_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Conversion")
        self.setFixedSize(500, 400)
        
        self._setup_ui()
        self._setup_styles()
        self._setup_connections()

    def _setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Voice Header (Label and Switch)
        voice_header = QHBoxLayout()
        self.voice_label = QLabel("Voice")
        self.switch = SwitchButton()
        voice_header.addWidget(self.voice_label)
        voice_header.addStretch()
        voice_header.addWidget(self.switch)
        main_layout.addLayout(voice_header)

        # File Selection
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select audio file...")
        self.file_path.setReadOnly(True)
        self.select_button = QPushButton("Select\nVoice")
        self.select_button.setFixedWidth(60)
        self.select_button.setFixedHeight(50)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.select_button)
        main_layout.addLayout(file_layout)

        # Device Selection with Labels
        # Input Device
        input_layout = QVBoxLayout()
        input_layout.setSpacing(4)
        input_label = QLabel("Select Input Device")
        self.input_combo = QComboBox()
        
        # Get input devices
        input_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        input_device_names = []
        self.input_device_ids = []  # Store device IDs
        default_input_idx = 0  # Default index to select
        
        for i, device in enumerate(input_devices):
            if device['max_input_channels'] > 0:  # Input device
                # Get hostapi name
                hostapi_name = hostapis[device['hostapi']]['name']
                name = f"{device['name']} ({hostapi_name})"
                input_device_names.append(name)
                self.input_device_ids.append(i)
                if i == sd.default.device[0]:  # Check if this is the default input device
                    default_input_idx = len(input_device_names) - 1

        self.input_combo.addItems(input_device_names)
        self.input_combo.setCurrentIndex(default_input_idx)
        self.input_combo.view().setSpacing(0)
        self.input_combo.view().setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_combo)
        main_layout.addLayout(input_layout)

        # Output Device
        output_layout = QVBoxLayout()
        output_layout.setSpacing(4)
        output_label = QLabel("Select Output Device")
        self.output_combo = QComboBox()
        
        # Get output devices
        output_device_names = []
        self.output_device_ids = []  # Store device IDs
        default_output_idx = 0  # Default index to select
        
        for i, device in enumerate(input_devices):
            if device['max_output_channels'] > 0:  # Output device
                # Get hostapi name
                hostapi_name = hostapis[device['hostapi']]['name']
                name = f"{device['name']} ({hostapi_name})"
                output_device_names.append(name)
                self.output_device_ids.append(i)
                if i == sd.default.device[1]:  # Check if this is the default output device
                    default_output_idx = len(output_device_names) - 1

        self.output_combo.addItems(output_device_names)
        self.output_combo.setCurrentIndex(default_output_idx)
        self.output_combo.view().setSpacing(0)
        self.output_combo.view().setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_combo)
        main_layout.addLayout(output_layout)

        main_layout.addStretch()

    def _setup_connections(self):
        self.select_button.clicked.connect(self.select_audio_file)

    def select_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3)"
        )
        
        if file_name:
            # 절대 경로로 표시
            self.file_path.setText(str(Path(file_name).absolute()))

    def _setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QLabel {
                color: #333333;
                font-size: 13px;
                font-weight: bold;
            }
            QComboBox {
                border: 2px solid #E0E0E0;
                border-radius: 4px;
                padding: 3px 8px;
                background: white;
                font-size: 13px;
                height: 30px;
            }
            QComboBox:hover {
                border-color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::item {
                color: #333333;
                padding: 5px;
                height: 20px;
                padding-left: 7px;
            }
            QComboBox::item:selected {
                background-color: #E3F2FD;
                color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #666;
                margin-right: 5px;
            }
            QComboBox::indicator {
                width: 13px;
                height: 13px;
                left: 7px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
                min-width: 60px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QLineEdit {
                border: 2px solid #E0E0E0;
                border-radius: 4px;
                padding: 8px;
                min-height: 30px;
                background-color: #F5F5F5;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
