# GUI + DSP
# Created by Caden Craddock and Matthew Reyna
# 4/7/2026
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QSpacerItem, QSizePolicy,
    QFileDialog, QMessageBox, QFrame, QComboBox, QSpinBox, QCheckBox,
    QGridLayout, QSlider
)
from gpiozero import Button
from dsp import applyFilter
from signal import pause
from PyQt6.QtCore import Qt, QUrl, pyqtSignal


# -----------------------------
# Shared UI Helpers
# -----------------------------
def make_title(text: str, pt: int = 60) -> QLabel:
    lbl = QLabel(text)
    font = lbl.font()
    font.setPointSize(pt)
    font.setBold(True)
    lbl.setFont(font)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("color: white;")
    return lbl


def make_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet("background-color: rgb(200, 200, 200); height: 3px;")
    return line


def make_big_button(text: str, bg_rgb: str, hover_rgb: str, pressed_rgb: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(320, 120)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {bg_rgb};
            color: white;
            font-size: 28px;
            font-weight: bold;
            border-radius: 18px;
        }}
        QPushButton:hover {{
            background-color: {hover_rgb};
        }}
        QPushButton:pressed {{
            background-color: {pressed_rgb};
        }}
        """
    )
    return btn


def make_back_button(page: QWidget) -> QPushButton:
    btn = QPushButton("Back to Menu")
    btn.setFixedSize(260, 80)
    btn.setStyleSheet(
        """
        QPushButton {
            font-size: 24px;
            padding: 12px 22px;
            border-radius: 14px;
            background-color: rgb(80, 80, 80);
            color: white;
        }
        QPushButton:hover {
            background-color: rgb(70, 70, 70);
        }
        QPushButton:pressed {
            background-color: rgb(50, 50, 50);
        }
        """
    )
    btn.clicked.connect(page.go_back_to_menu)
    return btn


def make_action_button(text: str, normal_rgb: str, hover_rgb: str, pressed_rgb: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(240, 90)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            font-size: 28px;
            padding: 12px 20px;
            border-radius: 14px;
            background-color: {normal_rgb};
            color: white;
        }}
        QPushButton:hover {{
            background-color: {hover_rgb};
        }}
        QPushButton:pressed {{
            background-color: {pressed_rgb};
        }}
        """
    )
    return btn


def make_debug_exit_button(main_window: QMainWindow) -> QPushButton:
    btn = QPushButton("X", main_window)
    btn.setFixedSize(48, 48)
    btn.setToolTip("Debug Exit")
    btn.setStyleSheet(
        """
        QPushButton {
            background-color: rgb(200, 0, 0);
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 10px;
        }
        QPushButton:hover {
            background-color: rgb(170, 0, 0);
        }
        QPushButton:pressed {
            background-color: rgb(140, 0, 0);
        }
        """
    )
    btn.clicked.connect(main_window.close)
    btn.raise_()
    return btn


def make_card_widget(inner_layout: QVBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(inner_layout)
    w.setStyleSheet(
        "background-color: rgb(20,20,20);"
        "border: 2px solid white;"
        "border-radius: 16px;"
        "padding: 18px;"
    )
    return w


# -----------------------------
# FFT Plot Page
# -----------------------------
class FFTDisplayPage(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.current_file = None
        self.canvas = None
        self.figure = None
        self.ax = None
        self.canvas_container = None
        self.status_label = None
        self.file_label = None
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        layout.addWidget(make_title("FFT Display", pt=42))
        layout.addWidget(make_divider())

        self.file_label = QLabel("Selected file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet(
            "font-size: 18px; color: white; padding: 6px;"
        )
        layout.addWidget(self.file_label)

        self.status_label = QLabel("Status: Waiting for audio file.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 17px; padding: 10px; color: white;"
            "border: 2px solid white; border-radius: 12px;"
            "background-color: rgb(20,20,20);"
        )
        layout.addWidget(self.status_label)

        # Large expandable FFT region
        fft_shell = QWidget()
        fft_shell.setStyleSheet(
            "background-color: white;"
            "border-radius: 16px;"
            "border: 2px solid rgb(220,220,220);"
        )
        fft_shell_layout = QVBoxLayout(fft_shell)
        fft_shell_layout.setContentsMargins(10, 10, 10, 10)

        self.canvas_container = QVBoxLayout()
        self.canvas_container.setContentsMargins(0, 0, 0, 0)

        self.placeholder_label = QLabel("FFT graph will appear here after an audio file is selected.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "background-color: white;"
            "color: black;"
            "font-size: 20px;"
            "padding: 20px;"
            "min-height: 420px;"
        )

        self.canvas_container.addWidget(self.placeholder_label)
        fft_shell_layout.addLayout(self.canvas_container)

        layout.addWidget(fft_shell, stretch=1)

        # Bottom touch row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(14)

        choose_btn = self.make_touch_button(
            "Choose File",
            "rgb(30,144,255)",
            "rgb(20,120,220)",
            "rgb(15,100,200)",
            220,
            72
        )
        replot_btn = self.make_touch_button(
            "Replot",
            "rgb(128,128,128)",
            "rgb(110,110,110)",
            "rgb(90,90,90)",
            180,
            72
        )
        clear_btn = self.make_touch_button(
            "Clear",
            "rgb(180,180,180)",
            "rgb(160,160,160)",
            "rgb(130,130,130)",
            180,
            72
        )
        back_btn = self.make_touch_button(
            "Back",
            "rgb(80,80,80)",
            "rgb(70,70,70)",
            "rgb(50,50,50)",
            180,
            72
        )

        choose_btn.clicked.connect(self.choose_file)
        replot_btn.clicked.connect(self.replot_current_file)
        clear_btn.clicked.connect(self.clear_plot)
        back_btn.clicked.connect(self.go_back_to_menu)

        btn_row.addStretch()
        btn_row.addWidget(choose_btn)
        btn_row.addWidget(replot_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(back_btn)
        btn_row.addStretch()

        layout.addLayout(btn_row)

    def make_touch_button(self, text, normal_rgb, hover_rgb, pressed_rgb, w, h):
        btn = QPushButton(text)
        btn.setFixedSize(w, h)
        btn.setStyleSheet(
            f"""
            QPushButton {{
                font-size: 24px;
                font-weight: bold;
                border-radius: 14px;
                background-color: {normal_rgb};
                color: white;
            }}
            QPushButton:hover {{
                background-color: {hover_rgb};
            }}
            QPushButton:pressed {{
                background-color: {pressed_rgb};
            }}
            """
        )
        return btn

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a);;All Files (*.*)"
        )
        if file_path:
            self.load_and_plot_file(file_path)

    def load_and_plot_file(self, file_path):
        self.current_file = Path(file_path)
        self.file_label.setText(f"Selected file: {self.current_file}")

        try:
            self.plot_fft(self.current_file)
            self.status_label.setText("Status: FFT displayed successfully.")
        except Exception as e:
            self.status_label.setText("Status: FFT failed.")
            QMessageBox.warning(self, "FFT Error", str(e))

    def replot_current_file(self):
        if self.current_file is None:
            QMessageBox.information(self, "FFT", "No audio file is currently selected.")
            return

        try:
            self.plot_fft(self.current_file)
            self.status_label.setText("Status: FFT replotted successfully.")
        except Exception as e:
            self.status_label.setText("Status: FFT failed.")
            QMessageBox.warning(self, "FFT Error", str(e))

    def clear_plot(self):
        self.current_file = None
        self.file_label.setText("Selected file: (none)")
        self.status_label.setText("Status: Waiting for audio file.")

        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
            self.figure = None
            self.ax = None

        if self.placeholder_label.parent() is None:
            self.canvas_container.addWidget(self.placeholder_label)
        self.placeholder_label.show()

    def plot_fft(self, file_path):
        import numpy as np
        import soundfile as sf
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        def to_mono(x):
            if x.ndim == 1:
                return x
            return np.mean(x, axis=1)

        def next_pow2(n):
            return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

        x, fs = sf.read(str(file_path), always_2d=False)
        x = to_mono(x)
        x = np.asarray(x, dtype=np.float64)

        start = 0.0
        duration = 1.0

        start_samp = int(start * fs)
        end_samp = int((start + duration) * fs)
        start_samp = max(0, start_samp)
        end_samp = min(len(x), end_samp)

        if end_samp <= start_samp + 1:
            raise ValueError("Selected segment is too short.")

        seg = x[start_samp:end_samp]
        seg = seg - np.mean(seg)

        N = len(seg)
        window = np.hanning(N)
        seg_w = seg * window

        nfft = next_pow2(N)
        X = np.fft.rfft(seg_w, n=nfft)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        mag_db = 20.0 * np.log10(mag + 1e-12)

        if self.canvas is None:
            self.figure = Figure(figsize=(10, 5.8), facecolor="white")
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)

            self.placeholder_label.hide()
            self.canvas_container.addWidget(self.canvas)

        self.ax.clear()
        self.ax.plot(freqs, mag_db, linewidth=1.0)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title(f"FFT Magnitude Spectrum - {Path(file_path).name}")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, fs / 2.0)
        self.figure.tight_layout()
        self.canvas.draw()

    def go_back_to_menu(self):
        self.main_window.show_menu()


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    gpioFilterSignal = pyqtSignal(str)
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Digital Audio Post Processor")
        self.resize(1400, 900)
        self.setStyleSheet("background-color: rgb(0, 0, 0);")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu_page = MenuPage(parent=self)
        self.load_file_page = LoadFilePage(parent=self)
        self.record_page = RecordPage(parent=self)
        self.settings_page = SettingsPage(parent=self)
        self.io_page = IOPage(parent=self)
        self.fft_page = FFTDisplayPage(parent=self)

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.load_file_page)
        self.stack.addWidget(self.record_page)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.io_page)
        self.stack.addWidget(self.fft_page)

        # Shared Variables
        self.currAudio = None  # Holds the current audio file, unfiltered
        self.currFS = None         # Holds the sample rate
        self.procAudio = None  # Holds processed audio, filtered
        self.currFilterMode = None # Holds current state of  filter selected 

        self.debug_exit_btn = make_debug_exit_button(self)
        self._position_debug_exit_button()

        self.show_menu()
        
        self.gpioFilterSignal.connect(self.handleGPIO)

        # ----------- Filter GPIO Setup --------------------
        self.lpfButton = Button(17, pull_up = True, bounce_time = 0.2)
        self.hpfButton = Button(27, pull_up = True, bounce_time = 0.1)
        self.bpfButton = Button(22, pull_up = True, bounce_time = 0.1)

        # Filter function calls on pressed
        self.lpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("LPF")
        self.hpfButton.when_pressed = self.hpfPressed
        self.bpfButton.when_pressed = self.bpfPressed

    # Filter Button Press Functions
    def handleGPIO(self, mode)
        
    def hpfPressed(self):
        self.gpioFilterPressed("HPF")
    def bpfPressed(self):
        self.gpioFilterPressed("BPF")

    def gpioFilterPressed(self, mode):
        # only respond when on the listening state
        if self.stack.currentWidget() is not self.io_page:
            return
        
        if self.currAudio is None:
            self.io_page.status_label.setText("Status: No file loaded.")
            return
        
        try:
            self.io_page.status_label.setText(f"Status: Applying {mode}...")
            output_file = self.applyFiltertoCurrAudio(self.currAudio,mode)

            if output_file is not None:
                self.io_page.status_label.setText(f"Status: {mode} applied.")
                self.io_page.file_label.setText(f"Selected file: {Path(output_file).name}")

                # play automatically after filtering
                self.io_page.play_audio()
        except Exception as e:
            self.io_page.status_label.setText(f"Status: {mode} failed :(")
            QMessageBox.warning(self,"Filter Error",str(e))

    def _position_debug_exit_button(self):
        self.debug_exit_btn.move(12, 12)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_debug_exit_button()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def show_load_file(self):
        self.stack.setCurrentWidget(self.load_file_page)

    def show_record(self):
        self.stack.setCurrentWidget(self.record_page)

    def show_settings(self):
        self.stack.setCurrentWidget(self.settings_page)

    def show_io(self):
        self.stack.setCurrentWidget(self.io_page)

    def show_fft_page(self, file_path=None):
        self.stack.setCurrentWidget(self.fft_page)
        if file_path is not None:
            self.fft_page.load_and_plot_file(file_path)
    # Apply filter to current audio file 
    def applyFiltertoCurrAudio(self, mode):
        if self.currAudio is None:
            return None
        
        outputPath = Path("processed_output.wav") # add MM/DD/YYY later

        # Call apply filter from dsp.py
        applyFilter(
            str(self.currAudio),
            str(outputPath),
            mode,
            normalize= True
        )

        self.procAudio = outputPath
        self.currFilterMode = mode

        return outputPath
    


# -----------------------------
# Menu Page (2x2 Grid)
# -----------------------------
class MenuPage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.addItem(QSpacerItem(20, 200, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(make_title("DIGITAL AUDIO POST PROCESSOR", pt=65))
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(35)

        self.btn_load = make_big_button("Load File", "rgb(30, 144, 255)", "rgb(20, 120, 220)", "rgb(15, 100, 200)")
        self.btn_record = make_big_button("Record", "rgb(255, 165, 0)", "rgb(230, 140, 0)", "rgb(200, 120, 0)")
        self.btn_settings = make_big_button("Settings", "rgb(128, 128, 128)", "rgb(110, 110, 110)", "rgb(90, 90, 90)")
        self.btn_io = make_big_button("Listen", "rgb(34, 139, 34)", "rgb(24, 110, 24)", "rgb(14, 90, 14)")

        grid.addWidget(self.btn_load, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_record, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_settings, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_io, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        wrap = QHBoxLayout()
        wrap.addStretch()
        wrap.addLayout(grid)
        wrap.addStretch()

        layout.addLayout(wrap)
        layout.addItem(QSpacerItem(20, 200, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.btn_load.clicked.connect(self.main_window.show_load_file)
        self.btn_record.clicked.connect(self.main_window.show_record)
        self.btn_settings.clicked.connect(self.main_window.show_settings)
        self.btn_io.clicked.connect(self.main_window.show_io)


# -----------------------------
# Load File Page
# -----------------------------
class LoadFilePage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.selected_path: Path | None = None
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(make_title("Load Audio File", pt=54))
        layout.addWidget(make_divider())
        layout.addSpacing(20)

        self.path_label = QLabel("Selected file: (none)")
        self.path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_label.setStyleSheet("font-size: 24px; padding: 12px; color: white;")
        layout.addWidget(self.path_label)

        self.status_label = QLabel("Status: Waiting for file selection.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 22px; padding: 14px; border: 2px solid white; border-radius: 14px; "
            "background-color: black; color: white;"
        )
        layout.addWidget(self.status_label)

        layout.addSpacing(30)

        row = QHBoxLayout()
        row.addStretch()

        pick_btn = make_action_button("Choose File", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        fft_btn = make_action_button("Open FFT", "rgb(128,128,128)", "rgb(110,110,110)", "rgb(90,90,90)")
        clear_btn = make_action_button("Clear", "rgb(180,180,180)", "rgb(160,160,160)", "rgb(130,130,130)")

        pick_btn.clicked.connect(self.choose_file)
        fft_btn.clicked.connect(self.open_fft_for_current_file)
        clear_btn.clicked.connect(self.clear_file)

        row.addWidget(pick_btn)
        row.addSpacing(18)
        row.addWidget(fft_btn)
        row.addSpacing(18)
        row.addWidget(clear_btn)
        row.addStretch()

        layout.addLayout(row)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))
        bottom.addStretch()
        layout.addLayout(bottom)

        layout.addSpacing(30)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a);;All Files (*.*)"
        )
        if file_path:
            self.selected_path = Path(file_path)
            self.path_label.setText(f"Selected file: {self.selected_path}")
            self.status_label.setText("Status: File selected.")

            # load file into current, unprocessed var
            self.main_window.currAudio = self.selected_path
            # Processed file can be nothing
            self.main_window.procAudio = None

    def open_fft_for_current_file(self):
        if self.selected_path is None:
            QMessageBox.information(self, "FFT", "Please choose an audio file first.")
            return
        self.main_window.show_fft_page(self.selected_path)

    def clear_file(self):
        self.selected_path = None
        self.path_label.setText("Selected file: (none)")
        self.status_label.setText("Status: Waiting for file selection.")

    def go_back_to_menu(self):
        self.main_window.show_menu()


# -----------------------------
# Record Page
# -----------------------------
class RecordPage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent

        self.is_recording = False
        self.recorded_file: Path | None = None
        self.uploaded_file: Path | None = None

        self.sample_rate = 44100
        self.channels = 1
        self.recorded_chunks = []
        self.stream = None

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(make_title("Record Audio", pt=54))
        layout.addWidget(make_divider())
        layout.addSpacing(20)

        self.state_label = QLabel("State: Idle")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet(
            "font-size: 30px; font-weight: bold; padding: 18px; border: 2px solid white; "
            "border-radius: 16px; background-color: black; color: white;"
        )
        layout.addWidget(self.state_label)

        self.file_label = QLabel("Current file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet(
            "font-size: 22px; padding: 10px; color: white;"
        )
        layout.addWidget(self.file_label)

        layout.addSpacing(30)

        row1 = QHBoxLayout()
        row1.addStretch()

        start_btn = make_action_button("Start", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        stop_btn = make_action_button("Stop", "rgb(255,99,71)", "rgb(230,80,60)", "rgb(200,60,50)")

        start_btn.clicked.connect(self.start_recording)
        stop_btn.clicked.connect(self.stop_recording)

        row1.addWidget(start_btn)
        row1.addSpacing(18)
        row1.addWidget(stop_btn)
        row1.addStretch()

        layout.addLayout(row1)
        layout.addSpacing(20)

        row2 = QHBoxLayout()
        row2.addStretch()

        upload_btn = make_action_button("Upload Audio", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        fft_btn = make_action_button("Open FFT", "rgb(128,128,128)", "rgb(110,110,110)", "rgb(90,90,90)")
        clear_btn = make_action_button("Clear File", "rgb(180,180,180)", "rgb(160,160,160)", "rgb(130,130,130)")

        upload_btn.clicked.connect(self.upload_audio_file)
        fft_btn.clicked.connect(self.open_fft_for_current_file)
        clear_btn.clicked.connect(self.clear_current_file)

        row2.addWidget(upload_btn)
        row2.addSpacing(18)
        row2.addWidget(fft_btn)
        row2.addSpacing(18)
        row2.addWidget(clear_btn)
        row2.addStretch()

        layout.addLayout(row2)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))
        bottom.addStretch()
        layout.addLayout(bottom)

        layout.addSpacing(30)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.recorded_chunks.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            QMessageBox.information(self, "Recording", "Recording is already in progress.")
            return

        try:
            self.recorded_chunks = []
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self.audio_callback
            )
            self.stream.start()
            self.is_recording = True
            self.state_label.setText("State: Recording from microphone...")
        except Exception as e:
            QMessageBox.warning(self, "Microphone Error", f"Could not start recording:\n{e}")

    def stop_recording(self):
        if not self.is_recording:
            QMessageBox.information(self, "Recording", "No recording is currently in progress.")
            return

        try:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_recording = False
        except Exception as e:
            QMessageBox.warning(self, "Recording Error", f"Could not stop recording cleanly:\n{e}")
            return

        if not self.recorded_chunks:
            self.state_label.setText("State: Recording stopped, but no audio was captured.")
            return

        audio_data = np.concatenate(self.recorded_chunks, axis=0)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recorded Audio",
            "recording.wav",
            "WAV Files (*.wav)"
        )

        if not file_path:
            self.state_label.setText("State: Recording stopped. Save cancelled.")
            return

        try:
            sf.write(file_path, audio_data, self.sample_rate)
            self.recorded_file = Path(file_path)
            self.uploaded_file = None
            self.file_label.setText(f"Current file: {self.recorded_file}")
            self.state_label.setText("State: Recording saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save recording:\n{e}")
            self.state_label.setText("State: Failed to save recording.")
        
        # load file into current, unprocessed var
        self.main_window.currAudio = self.recorded_file
        # Processed file can be nothing
        self.main_window.procAudio = None

    def upload_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a);;All Files (*.*)"
        )

        if not file_path:
            return

        self.uploaded_file = Path(file_path)
        self.recorded_file = None
        self.file_label.setText(f"Current file: {self.uploaded_file}")
        self.state_label.setText("State: Uploaded audio file selected.")

        # load file into current, unprocessed var
        self.main_window.currAudio = self.uploaded_file
        # Processed file can be nothing
        self.main_window.procAudio = None

    def open_fft_for_current_file(self):
        current_file = self.get_current_file()

        if current_file is None:
            QMessageBox.information(self, "FFT", "Please record or upload an audio file first.")
            return

        self.main_window.show_fft_page(current_file)

    def get_current_file(self):
        if self.recorded_file is not None:
            return self.recorded_file
        if self.uploaded_file is not None:
            return self.uploaded_file
        return None

    def clear_current_file(self):
        self.recorded_file = None
        self.uploaded_file = None
        self.file_label.setText("Current file: (none)")
        self.state_label.setText("State: Idle")

    def go_back_to_menu(self):
        if self.is_recording and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.is_recording = False
            self.stream = None

        self.main_window.show_menu()


# -----------------------------
# Settings Page
# -----------------------------
class SettingsPage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(make_title("Settings", pt=54))
        layout.addWidget(make_divider())
        layout.addSpacing(25)

        button_row_1 = QHBoxLayout()
        button_row_1.addStretch()

        btn_audio = make_action_button("Audio", "rgb(128,128,128)", "rgb(110,110,110)", "rgb(90,90,90)")
        btn_display = make_action_button("Display", "rgb(128,128,128)", "rgb(110,110,110)", "rgb(90,90,90)")
        btn_reset = make_action_button("Reset", "rgb(200,0,0)", "rgb(170,0,0)", "rgb(140,0,0)")

        btn_audio.clicked.connect(lambda: QMessageBox.information(self, "Settings", "Audio settings (placeholder)."))
        btn_display.clicked.connect(lambda: QMessageBox.information(self, "Settings", "Display settings (placeholder)."))
        btn_reset.clicked.connect(self.reset_settings)

        button_row_1.addWidget(btn_audio)
        button_row_1.addSpacing(18)
        button_row_1.addWidget(btn_display)
        button_row_1.addSpacing(18)
        button_row_1.addWidget(btn_reset)
        button_row_1.addStretch()

        layout.addLayout(button_row_1)
        layout.addSpacing(25)

        panel = QVBoxLayout()

        sr_row = QHBoxLayout()
        sr_label = QLabel("Sample rate (Hz):")
        sr_label.setStyleSheet("font-size: 22px; color: white;")
        self.sr_box = QSpinBox()
        self.sr_box.setRange(8000, 192000)
        self.sr_box.setSingleStep(1000)
        self.sr_box.setValue(48000)
        self.sr_box.setStyleSheet("font-size: 22px; padding: 6px;")
        sr_row.addWidget(sr_label)
        sr_row.addStretch()
        sr_row.addWidget(self.sr_box)
        panel.addLayout(sr_row)

        fmt_row = QHBoxLayout()
        fmt_label = QLabel("Default output format:")
        fmt_label.setStyleSheet("font-size: 22px; color: white;")
        self.fmt_box = QComboBox()
        self.fmt_box.addItems(["WAV", "FLAC", "MP3"])
        self.fmt_box.setStyleSheet("font-size: 22px; padding: 6px;")
        fmt_row.addWidget(fmt_label)
        fmt_row.addStretch()
        fmt_row.addWidget(self.fmt_box)
        panel.addLayout(fmt_row)

        self.norm_check = QCheckBox("Enable normalization")
        self.norm_check.setChecked(True)
        self.norm_check.setStyleSheet("font-size: 22px; color: white;")
        panel.addWidget(self.norm_check)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setStyleSheet(
            "background: rgb(20,20,20); border: 2px solid white; border-radius: 16px; padding: 18px;"
        )
        layout.addWidget(panel_widget)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()

        save_btn = make_action_button("Save", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        save_btn.clicked.connect(self.save_settings)

        bottom.addWidget(save_btn)
        bottom.addSpacing(18)
        bottom.addWidget(make_back_button(self))
        bottom.addStretch()

        layout.addLayout(bottom)
        layout.addSpacing(30)

    def save_settings(self):
        QMessageBox.information(
            self,
            "Settings",
            f"Saved:\nSample rate: {self.sr_box.value()} Hz\nFormat: {self.fmt_box.currentText()}\nNormalization: {self.norm_check.isChecked()}"
        )

    def reset_settings(self):
        self.sr_box.setValue(48000)
        self.fmt_box.setCurrentText("WAV")
        self.norm_check.setChecked(True)
        QMessageBox.information(self, "Settings", "Settings reset to defaults (placeholder).")

    def go_back_to_menu(self):
        self.main_window.show_menu()


# -----------------------------
# I/O Page
# -----------------------------
class IOPage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent

        self.audio_file: Path | None = None

        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0.7)

        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        layout.addWidget(make_title("Audio Listening", pt=54))
        layout.addWidget(make_divider())

        self.status_label = QLabel("Status: Waiting for audio file.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 22px; padding: 14px; border: 2px solid white; "
            "border-radius: 14px; background-color: rgb(20,20,20); color: white;"
        )
        layout.addWidget(self.status_label)

        layout.addSpacing(10)

        # File label
        self.file_label = QLabel("Selected file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("font-size: 22px; color: white; padding: 10px;")
        layout.addWidget(self.file_label)

        layout.addSpacing(20)

        # Control buttons
        button_row = QHBoxLayout()
        button_row.addStretch()

        upload_btn = make_action_button(
            "Upload Audio",
            "rgb(30,144,255)",
            "rgb(20,120,220)",
            "rgb(15,100,200)"
        )

        play_btn = make_action_button(
            "Play",
            "rgb(34,139,34)",
            "rgb(24,110,24)",
            "rgb(14,90,14)"
        )

        stop_btn = make_action_button(
            "Stop",
            "rgb(200,0,0)",
            "rgb(170,0,0)",
            "rgb(140,0,0)"
        )

        upload_btn.clicked.connect(self.upload_audio_file)
        play_btn.clicked.connect(self.play_audio)
        stop_btn.clicked.connect(self.stop_audio)

        button_row.addWidget(upload_btn)
        button_row.addSpacing(18)
        button_row.addWidget(play_btn)
        button_row.addSpacing(18)
        button_row.addWidget(stop_btn)
        button_row.addStretch()

        layout.addLayout(button_row)

        layout.addSpacing(30)

        # Volume control
        volume_layout = QHBoxLayout()

        volume_label = QLabel("Volume")
        volume_label.setStyleSheet("font-size: 20px; color: white;")

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)

        self.volume_slider.valueChanged.connect(self.change_volume)

        volume_layout.addStretch()
        volume_layout.addWidget(volume_label)
        volume_layout.addSpacing(12)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addStretch()

        layout.addLayout(volume_layout)

        layout.addStretch()

        # Bottom row
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(make_back_button(self))
        bottom_row.addStretch()

        layout.addLayout(bottom_row)

    def upload_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a);;All Files (*.*)"
        )

        if not file_path:
            return

        self.audio_file = Path(file_path)
        self.file_label.setText(f"Selected file: {self.audio_file.name}")
        self.status_label.setText("Status: Audio file loaded.")

    def play_audio(self):
        audioToPlay = self.main_window.procAudio or self.main_window.currAudio
        if audioToPlay is None:
            QMessageBox.information(self, "Audio", "Please load or record audio file first.")
            return

        url = QUrl.fromLocalFile(str(audioToPlay.resolve()))
        self.player.setSource(url)
        self.player.play()

        self.status_label.setText(f"Status: Playing {Path(audioToPlay).name}")
        self.file_label.setText(f"Selected file:  {Path(audioToPlay).name}")

    def stop_audio(self):
        self.player.stop()
        self.status_label.setText("Status: Playback stopped.")

    def change_volume(self, value):
        self.audio_output.setVolume(value / 100)

    def go_back_to_menu(self):
        self.player.stop()
        self.main_window.show_menu()

# -----------------------------
# Entry Point
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

