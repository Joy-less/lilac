import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QComboBox, QLineEdit,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QStyledItemDelegate,
                             QMessageBox)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QPainterPath, QColor, QFont
import sounddevice as sd
from core import RealtimeVoiceConverter

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

class ComboBoxItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._check_icon = "✓"  # 체크 표시 문자

    def paint(self, painter, option, index):
        # 기본 아이템 그리기
        super().paint(painter, option, index)
        
        # 현재 선택된 아이템인 경우 체크 표시 추가
        if index.row() == index.model().parent().currentIndex():
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 폰트 설정
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            
            # 체크 표시 색상 설정 (파란색)
            painter.setPen(QColor("#2196F3"))
            
            # 체크 표시 위치 계산 (오른쪽 정렬)
            check_width = painter.fontMetrics().horizontalAdvance(self._check_icon)
            x = option.rect.right() - check_width - 10  # 오른쪽 여백
            y = option.rect.center().y() + painter.fontMetrics().height() / 3
            
            # 체크 표시 그리기
            painter.drawText(x, y, self._check_icon)
            painter.restore()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Conversion")
        self.setFixedSize(500, 400)
        
        self.setStyleSheet("background-color: white;")
        self.converter = None  # Voice converter 인스턴스 저장용
        
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
        
        # 커스텀 델리게이트 설정
        self.input_combo.setItemDelegate(ComboBoxItemDelegate(self.input_combo))
        
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
        
        # 커스텀 델리게이트 설정
        self.output_combo.setItemDelegate(ComboBoxItemDelegate(self.output_combo))
        
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
        self.switch.mousePressEvent = self.handle_switch_click  # 스위치 이벤트 오버라이드

    def handle_switch_click(self, event):
        if not self.file_path.text():
            QMessageBox.warning(self, "Warning", "Please select a voice file first.")
            return
            
        # 기존 스위치 동작 수행
        self.switch._is_checked = not self.switch._is_checked
        self.switch.animation.setStartValue(self.switch._thumb_position)
        
        if self.switch._is_checked:
            self.switch.animation.setEndValue(27)
            self.start_voice_conversion()
        else:
            self.switch.animation.setEndValue(2)
            self.stop_voice_conversion()
            
        self.switch.animation.start()

    def start_voice_conversion(self):
        try:
            # 현재 선택된 디바이스 인덱스 가져오기
            input_idx = self.input_device_ids[self.input_combo.currentIndex()]
            output_idx = self.output_device_ids[self.output_combo.currentIndex()]
            
            # Voice Converter 인스턴스 생성
            self.converter = RealtimeVoiceConverter(
                model_path='vc/model.pth',  # 모델 경로 설정
                target_voice_path=self.file_path.text(),  # 선택된 음성 파일 경로
                device='cuda' if torch.cuda.is_available() else 'cpu',
                input_device=input_idx,
                output_device=output_idx
            )
            
            # 변환 시작
            self.converter.start()
            
            # UI 컴포넌트 비활성화
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)
            self.select_button.setEnabled(False)
            self.file_path.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start voice conversion: {str(e)}")
            self.switch._is_checked = False
            self.switch.animation.setStartValue(self.switch._thumb_position)
            self.switch.animation.setEndValue(2)
            self.switch.animation.start()

    def stop_voice_conversion(self):
        try:
            if self.converter:
                self.converter.stop()
                self.converter = None
            
            # UI 컴포넌트 활성화
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
            self.select_button.setEnabled(True)
            self.file_path.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop voice conversion: {str(e)}")

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

    def closeEvent(self, event):
        # 프로그램 종료 시 voice converter 정리
        if self.converter:
            self.stop_voice_conversion()
        event.accept()

    def _setup_styles(self):
        self.setStyleSheet(open('stylesheet.qss', 'r', encoding='utf-8').read())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 어플리케이션 레벨에서 기본 색상 팔레트 설정
    app.setPalette(app.style().standardPalette())
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())