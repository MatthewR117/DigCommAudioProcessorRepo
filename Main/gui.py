# GUI + DSP | 4/20/2026
# Current Features:
# - Main 3 filters, EQ, Compressors
# - Temp output and save output
# - Scrolling Audio Waveform

# Created by Matthew Reyna and Cadden Craddock

import sys
from pathlib import Path
import datetime
import os
import shutil
import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFileDialog, QMessageBox,
    QFrame, QSlider, QGridLayout
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from gpiozero import Button
from dsp import applyFilter

QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
#  Global variable to hold current date for file output
date = datetime.datetime.now().strftime("%Y-%m-%d")

# -----------------------------
#  !!!!!!!!!!!!!!!!!!!!! MAIN RETARD MAIN !!!!!!!!!!!!!!!!
# -----------------------------
class MainWindow(QMainWindow): 
    
    gpioFilterSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Digital Audio Post Processor")
        self.setFixedSize(1024, 600)
        self.setStyleSheet("background-color: rgb(70, 70, 70);") # Edit RGB values to change theme color

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu_page = MenuPage(parent=self)
        self.upload_page = UploadAudioPage(parent=self)
        self.record_page = RecordAudioPage(parent=self)

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.record_page)

        self.debug_exit_btn = make_debug_exit_button(self)
        self._position_debug_exit_button()
        
        # ---- Shared Variables ----
        self.currAudio = None  # Holds the current audio file, unfiltered
        self.currFS = None         # Holds the sample rate
        self.procAudio = None  # Holds processed audio, filtered
        self.currFilterMode = None # Holds current state of  filter selected
        self.tempAudio = None # Holds temp filtered audio, get deleted lil bro

        self.gpioFilterSignal.connect(self.handleGPIO)
        self.show_menu()

        # ----------- Filter GPIO Setup --------------------
        # Top Buttons
        self.lpfButton = Button(17, pull_up=True, bounce_time=0.2)
        self.hpfButton = Button(27, pull_up=True, bounce_time=0.2)
        self.bpfButton = Button(22, pull_up=True, bounce_time=0.2)
        # Bottom Buttons
        self.autoQButton = Button(16, pull_up=True, bounce_time=0.2)
        self.eqButton = Button(23, pull_up=True, bounce_time=0.2)
        self.compButton = Button(15, pull_up=True, bounce_time=0.2)
        self.pwrButton = Button(20, pull_up=True, bounce_time=0.2)
        # ------------ GPIO Pressed ----------------------------------
        self.lpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("LPF")
        self.hpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("HPF")
        self.bpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("BPF")

        self.autoQButton.when_pressed = lambda: self.gpioFilterSignal.emit("AUTO")
        self.eqButton.when_pressed = lambda: self.gpioFilterSignal.emit("EQ")
        self.compButton.when_pressed = lambda: self.gpioFilterSignal.emit("COMP")
        self.pwrButton.when_pressed = lambda: self.gpioFilterSignal.emit("PWR")
        #--------------------------------------------------------------------------
    
    # Filter Button Press Function
    def handleGPIO(self, mode):
        if self.stack.currentWidget() is not self.upload_page:
            return

        if self.currAudio is None:
            self.upload_page.status_label.setText("Status: No audio file loaded.")
            return

        try:
            self.upload_page.status_label.setText(f"Status: Applying {mode}...")
            
            # hold the filtered audio in temp, unsaved
            tempOutFile = self.applyFilterToCurrAudio(mode)

            if tempOutFile is not None:
                self.procAudio = tempOutFile
                self.upload_page.status_label.setText(f"Status: {mode} applied.")
                self.upload_page.file_label.setText(f"Selected file: {Path(tempOutFile).name}")

                if self.upload_page.current_plot_mode == "waveform":
                    self.upload_page.plot_waveform(Path(tempOutFile))
                else:
                    self.upload_page.plot_fft(Path(tempOutFile))
                # Play audio after filtering (maybe make it live rather than restart)
                self.upload_page.play_audio()

        except Exception as e:
            self.upload_page.status_label.setText(f"Status: {mode} failed.")
            QMessageBox.warning(self, "Processing Error", str(e))
    

    # --------------------- Apply filter to current audio function -----------------
    def applyFilterToCurrAudio(self, mode):
        if self.currAudio is None:
            return None
        
        # delete temp audio 
        self.deleteTemp()

        # Save file name for TEMP filtered audio
        output_path = Path(f"TEMP_{self.currAudio.stem}_{mode}_{date}.wav")

        applyFilter(
            str(self.currAudio),
            str(output_path),
            mode,
            normalize=True
        )

        self.tempAudio = output_path
        self.currFilterMode = mode
        return output_path

    # delete temp audio function
    def deleteTemp(self):
        if self.tempAudio is not None:
            try:
                if os.path.exists(self.procAudio):
                    # delete temp file     
                     os.remove(self.procAudio)
            except Exception:
                pass
            # reset current mode and processed audio to none
            self.procAudio = None
            self.currFilterMode = None 
                    

    def _position_debug_exit_button(self):
        self.debug_exit_btn.move(10, 10)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_debug_exit_button()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def show_upload_audio(self):
        self.stack.setCurrentWidget(self.upload_page)
        self.upload_page.on_page_shown()

    def show_record_audio(self):
        self.stack.setCurrentWidget(self.record_page)
        
# -----------------------------
# Shared UI Helpers
# -----------------------------
def make_title(text: str, pt: int = 32) -> QLabel:
    lbl = QLabel(text)
    font = lbl.font()
    font.setPointSize(pt)
    font.setBold(True)
    lbl.setFont(font)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: white;")
    return lbl


def make_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet("background-color: rgb(200, 200, 200); height: 2px;")
    return line


def make_big_button(text: str, bg_rgb: str, hover_rgb: str, pressed_rgb: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(260, 90)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {bg_rgb};
            color: white;
            font-size: 22px;
            font-weight: bold;
            border-radius: 16px;
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


def make_action_button(text: str, normal_rgb: str, hover_rgb: str, pressed_rgb: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(150, 58)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            font-size: 17px;
            padding: 8px 10px;
            border-radius: 12px;
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


def make_back_button(page: QWidget) -> QPushButton:
    btn = QPushButton("Back to Menu")
    btn.setFixedSize(190, 60)
    btn.setStyleSheet(
        """
        QPushButton {
            font-size: 18px;
            padding: 8px 14px;
            border-radius: 12px;
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


def make_debug_exit_button(main_window: QMainWindow) -> QPushButton:
    btn = QPushButton("X", main_window)
    btn.setFixedSize(40, 40)
    btn.setToolTip("Debug Exit")
    btn.setStyleSheet(
        """
        QPushButton {
            background-color: rgb(200, 0, 0);
            color: white;
            font-size: 18px;
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


def open_audio_file_dialog(parent: QWidget, title: str) -> str:
    dialog = QFileDialog(parent, title)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setNameFilter("Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg);;All Files (*.*)")
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

    if dialog.exec():
        return dialog.selectedFiles()[0]
    return ""


def save_wav_file_dialog(parent: QWidget, title: str, default_name: str = "recording.wav") -> str:
    dialog = QFileDialog(parent, title)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dialog.setFileMode(QFileDialog.FileMode.AnyFile)
    dialog.selectFile(default_name)
    dialog.setNameFilter("WAV Files (*.wav)")
    dialog.setDefaultSuffix("wav")
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

    if dialog.exec():
        return dialog.selectedFiles()[0]
    return ""


# -----------------------------
#   $$$$$$$$$$$$  Upload Audio Page $$$$$$$$$$$$$$
# -----------------------------
class UploadAudioPage(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.main_window = parent

        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0.7)

        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        self.canvas = None
        self.figure = None
        self.current_plot_mode = "waveform"
        
        # Variables for live waveform
        self.wave_timer = QTimer()
        self.wave_timer.setInterval(30) # time interval updates every 30 ms
        self.wave_timer.timeout.connect(self.updateWaveform)
        
        self.live_audio_data = None
        self.live_fs = None # live sample rate
        self.live_window_sec = 0.20 # show last 0.2 seconds
        self.live_line = None
        self.live_ax = None
        self.play_started = False

        self.build_ui()

    # save audio function 
    def saveAudio(self):
        if self.main_window.procAudio is None:
            QMessageBox.information(self, "Save Audio", "No filtered audio to save, chud")
            return
        
        # Save filtered to .wav
        outputFilt = f"{self.main_window.currAudio.stem}_{self.main_window.currFilterMode}_{date}.wav" # "filename_filter_date.wav"
        savePath = save_wav_file_dialog(self, "Save Filtered Audio",outputFilt)

        if not savePath:
            return
        
        try:
            shutil.copy(self.main_window.procAudio, savePath)
            self.status_label.setText(f"Status: Saved as {Path(savePath).name}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 55, 18, 18)
        layout.setSpacing(10)

        layout.addWidget(make_title("Upload Audio", pt=12))
        #layout.addWidget(make_divider())

        self.file_label = QLabel("Selected file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-size: 16px; color: white; padding: 4px;")
        layout.addWidget(self.file_label)

        self.status_label = QLabel("Status: Waiting for audio file.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 16px; padding: 8px; color: white;"
            "border: 2px solid white; border-radius: 12px;"
            "background-color: rgb(20,20,20);"
        )
        layout.addWidget(self.status_label)

        plot_shell = QWidget()
        plot_shell.setStyleSheet(
            "background-color: white;"
            "border-radius: 16px;"
            "border: 2px solid rgb(220,220,220);"
        )
        plot_shell_layout = QVBoxLayout(plot_shell)
        plot_shell_layout.setContentsMargins(8, 8, 8, 8)

        self.canvas_container = QVBoxLayout()
        self.canvas_container.setContentsMargins(0, 0, 0, 0)

        self.placeholder_label = QLabel(
            "FFT or waveform will appear here after an audio file is selected."
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "background-color: white;"
            "color: black;"
            "font-size: 18px;"
            "padding: 20px;"
            "min-height: 240px;"
        )
        self.canvas_container.addWidget(self.placeholder_label)
        plot_shell_layout.addLayout(self.canvas_container)

        layout.addWidget(plot_shell, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch()

        choose_btn = make_action_button("Choose File", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        play_btn = make_action_button("Play", "rgb(34,139,34)", "rgb(24,110,24)", "rgb(14,90,14)")
        stop_btn = make_action_button("Stop", "rgb(200,0,0)", "rgb(170,0,0)", "rgb(140,0,0)")
        fft_btn = make_action_button("Show FFT", "rgb(128,128,128)", "rgb(110,110,110)", "rgb(90,90,90)")
        wave_btn = make_action_button("Waveform", "rgb(128,128,128)", "rgb(110,110,110)", "rgb(90,90,90)")
        clear_btn = make_action_button("Clear", "rgb(180,180,180)", "rgb(160,160,160)", "rgb(130,130,130)")
        save_btn = make_action_button("Save Audio", "rgb(255,140,0)","rgb(230,120,0)","rgb(200,100,0)")
        
        choose_btn.clicked.connect(self.prompt_for_audio_file)
        play_btn.clicked.connect(self.play_audio)
        stop_btn.clicked.connect(self.stop_audio)
        fft_btn.clicked.connect(self.show_fft)
        wave_btn.clicked.connect(self.show_waveform)
        clear_btn.clicked.connect(self.clear_audio)
        save_btn.clicked.connect(self.saveAudio)

        button_row.addWidget(choose_btn)
        button_row.addWidget(play_btn)
        button_row.addWidget(stop_btn)
        button_row.addWidget(fft_btn)
        button_row.addWidget(wave_btn)
        button_row.addWidget(clear_btn)
        button_row.addWidget(save_btn)
        button_row.addStretch()

        layout.addLayout(button_row)

        volume_layout = QHBoxLayout()
        volume_layout.addStretch()

        volume_label = QLabel("Volume")
        volume_label.setStyleSheet("font-size: 18px; color: white;")

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedWidth(220)
        self.volume_slider.valueChanged.connect(self.change_volume)

        volume_layout.addWidget(volume_label)
        volume_layout.addSpacing(10)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addStretch()

        layout.addLayout(volume_layout)

        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))
        bottom.addStretch()
        layout.addLayout(bottom)

    def on_page_shown(self):
        if self.main_window.currAudio is None and self.main_window.procAudio is None:
            QTimer.singleShot(150, self.prompt_for_audio_file)

    def prompt_for_audio_file(self):
        file_path = open_audio_file_dialog(self, "Select Audio File")
        if not file_path:
            if self.main_window.currAudio is None and self.main_window.procAudio is None:
                self.status_label.setText("Status: No file selected.")
            return

        self.load_audio_file(file_path)
        
    # init waveform display page function
    def waveformInit(self):
        self._ensure_canvas()
        self.figure.clear()
        
        self.live_ax = self.figure.add_subplot(111)
        self.live_ax.set_title("Audio Waveform")
        self.live_ax.set_xlabel("Time (seconds)")
        self.live_ax.set_ylabel("Amplitude")
        self.live_ax.set_xlim(0,self.live_window_sec)
        self.live_ax.set_ylim(-1.0,1.0)
        #self.live_ax.set_gid(True, alpha = 0.3)
        
        t = np.linspace(0, self.live_window_sec,1000)
        y = np.zeros_like(t)
        
        # drawing line
        self.live_line, = self.live_ax.plot(t,y,linewidth = 1.0)
        self.figure.tight_layout()
        self.canvas.draw()
        

    # Load audio file function
    def load_audio_file(self, file_path: str):
        self.main_window.deleteTemp()
        audio_path = Path(file_path)

        self.main_window.currAudio = audio_path
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None

        self.file_label.setText(f"Selected file: {audio_path.name}")
        # plot mode when audio file is init loaded
        self.current_plot_mode = "waveform"
        
        self.waveformInit()
        self.status_label.setText("Status: Audio loaded.")
        
        """
        try:
            self.plot_fft(audio_path)
            self.status_label.setText("Status: Audio loaded. FFT displayed.")
        except Exception as e:
            self.status_label.setText("Status: Audio loaded, but FFT failed.")
            QMessageBox.warning(self, "FFT Error", str(e))
        """

    def get_active_audio_path(self):
        return self.main_window.procAudio or self.main_window.currAudio
    
    # Play audio function
    def play_audio(self):
        audio_to_play = self.get_active_audio_path()
        if audio_to_play is None:
            QMessageBox.information(self, "Audio", "Please load or record an audio file first.")
            return

        self.player.stop()
        url = QUrl.fromLocalFile(str(Path(audio_to_play).resolve()))
        self.player.setSource(url)
        
        # start init waveform scrolling
        self.waveformInit()
        self.player.play()
        self.wave_timer.start()
        
        
        self.status_label.setText(f"Status: Playing {Path(audio_to_play).stem}")
        self.file_label.setText(f"Selected file: {Path(audio_to_play).name}")
    
    # Stop audio function
    def stop_audio(self):
        self.player.stop()
        self.wave_timer.stop()
        
        if self.get_active_audio_path() is None:
            self.status_label.setText("Status: Waiting for audio file.")
        else:
            self.status_label.setText("Status: Playback stopped.")
            
    # Update audio waveform for "scrolling" effect
    def updateWaveform(self):
        if self.live_audio_data is None or self.live_fs is None:
            return
        
        # QMediaPlayer position is in miliseconds
        pos = self.player.position()
        current_sample = int((pos / 1000.0) * self.live_fs)
        
        window_samples = int(self.live_window_seconds * self.live_fs)
        start = max(0, current_sample - window_samples)
        end = current_sample
        
        segment = self.live_audio_data[start:end]
        
        if len(segment) < window_samples:
            padded = np.zeros(window_samples, dtype = np.float64)
            padded[-len(segment):] = segment
            segment = padded
        
        t = np.linspace(0, self.live_window_seconds, window_samples, endpoint = False)
        
        if self.live_line is not None:
            self.live_line.set_data(t,segnent)
        
        if self.live_ax is not None:
            self.live_ax.setxlim(0,self.live_window_seconds)
            self.live_ax.setylim(-1.0, 1.0)
            
        if self.canvas is not None:
            self.canvas.draw_idle()
            
        # stop timer when playback ends
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState and pos > 0:
            self.wave_timer.stop()
        

    def show_fft(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            QMessageBox.information(self, "FFT", "Please select an audio file first.")
            return

        try:
            self.current_plot_mode = "fft"
            self.plot_fft(Path(audio_path))
            self.status_label.setText("Status: FFT displayed.")
        except Exception as e:
            self.status_label.setText("Status: FFT failed.")
            QMessageBox.warning(self, "FFT Error", str(e))

    def show_waveform(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            QMessageBox.information(self, "Waveform", "Please select an audio file first.")
            return

        try:
            self.current_plot_mode = "waveform"
            self.plot_waveform(Path(audio_path))
            self.status_label.setText("Status: Waveform displayed.")
        except Exception as e:
            self.status_label.setText("Status: Waveform failed.")
            QMessageBox.warning(self, "Waveform Error", str(e))

    # Clear audio function
    def clear_audio(self):
        self.main_window.deleteTemp()
        self.player.stop()
        self.wave_timer.stop()

        self.main_window.currAudio = None
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None
        self.live_audio_data = None
        self.live_fs = None
        self.live_line = None
        self.live_ax = None

        self.file_label.setText("Selected file: (none)")
        self.status_label.setText("Status: Waiting for audio file.")
        self.current_plot_mode = "waveform"

        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
            self.figure = None

        if self.placeholder_label.parent() is None:
            self.canvas_container.addWidget(self.placeholder_label)
        self.placeholder_label.show()

    def change_volume(self, value):
        self.audio_output.setVolume(value / 100.0)

    def _ensure_canvas(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        if self.canvas is None:
            self.figure = Figure(figsize=(8, 4.0), facecolor="white")
            self.canvas = FigureCanvas(self.figure)
            self.placeholder_label.hide()
            self.canvas_container.addWidget(self.canvas)

    def _load_audio_mono(self, file_path: Path):
        x, fs = sf.read(str(file_path), always_2d=False)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        return x, fs

    def plot_waveform(self, file_path: Path):
        x, fs = self._load_audio_mono(file_path)

        max_samples = min(len(x), fs * 5)
        x = x[:max_samples]
        t = np.linspace(0, len(x) / fs, num=len(x), endpoint=False)

        self._ensure_canvas()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(t, x, linewidth=0.8)
        ax.set_title(f"Waveform - {file_path.name}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_fft(self, file_path: Path):
        x, fs = self._load_audio_mono(file_path)

        end_samp = min(len(x), int(1.0 * fs))
        if end_samp <= 1:
            raise ValueError("Selected segment is too short.")

        seg = x[:end_samp]
        seg = seg - np.mean(seg)

        n = len(seg)
        seg_w = seg * np.hanning(n)
        nfft = 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

        x_fft = np.fft.rfft(seg_w, n=nfft)
        mag = np.abs(x_fft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        mag_db = 20.0 * np.log10(mag + 1e-12)

        self._ensure_canvas()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(freqs, mag_db, linewidth=1.0)
        ax.set_title(f"FFT - {file_path.name}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fs / 2.0)

        self.figure.tight_layout()
        self.canvas.draw()

    def go_back_to_menu(self):
        self.player.stop()
        self.main_window.show_menu()


# -----------------------------
# Record Audio Page
# -----------------------------
class RecordAudioPage(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.main_window = parent

        self.is_recording = False
        self.sample_rate = 44100
        self.channels = 1
        self.recorded_chunks = []
        self.stream = None

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 55, 18, 18)
        layout.setSpacing(10)

        layout.addWidget(make_title("Record Audio", pt=28))
        layout.addWidget(make_divider())

        self.state_label = QLabel("State: Idle")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 12px; border: 2px solid white;"
            "border-radius: 16px; background-color: black; color: white;"
        )
        layout.addWidget(self.state_label)

        self.info_label = QLabel(
            "Use Start to record from the microphone.\n"
            "When Stop is pressed, you will choose a .wav filename to save."
        )
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 16px; color: white; padding: 6px;")
        layout.addWidget(self.info_label)

        layout.addStretch()

        row = QHBoxLayout()
        row.addStretch()

        start_btn = make_big_button("Start", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        stop_btn = make_big_button("Stop", "rgb(255,99,71)", "rgb(230,80,60)", "rgb(200,60,50)")

        start_btn.clicked.connect(self.start_recording)
        stop_btn.clicked.connect(self.stop_recording)

        row.addWidget(start_btn)
        row.addSpacing(20)
        row.addWidget(stop_btn)
        row.addStretch()

        layout.addLayout(row)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))
        bottom.addStretch()
        layout.addLayout(bottom)

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

        file_path = save_wav_file_dialog(self, "Save Recorded Audio", "recording.wav")
        if not file_path:
            self.state_label.setText("State: Recording stopped. Save cancelled.")
            return

        try:
            sf.write(file_path, audio_data, self.sample_rate)
            saved_name = Path(file_path).name

            self.main_window.currAudio = Path(file_path)
            self.main_window.procAudio = None
            self.main_window.currFilterMode = None

            self.state_label.setText(f"State: Recording saved successfully as {saved_name}")
            QMessageBox.information(
                self,
                "Recording Saved",
                f"Recording saved as:\n{saved_name}\n\nYou can now open it from the Upload Audio page."
            )
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save recording:\n{e}")
            self.state_label.setText("State: Failed to save recording.")

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
# Menu Page
# -----------------------------
class MenuPage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 55, 20, 20)
        layout.setSpacing(14)

        layout.addWidget(make_title("DIGITAL AUDIO POST PROCESSOR", pt=34))
        layout.addSpacing(12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(20)

        self.btn_upload = make_big_button(
            "Upload Audio",
            "rgb(30,144,255)",
            "rgb(20,120,220)",
            "rgb(15,100,200)"
        )
        self.btn_record = make_big_button(
            "Record Audio",
            "rgb(255,165,0)",
            "rgb(230,140,0)",
            "rgb(200,120,0)"
        )

        grid.addWidget(self.btn_upload, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_record, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        wrap = QHBoxLayout()
        wrap.addStretch()
        wrap.addLayout(grid)
        wrap.addStretch()

        layout.addStretch()
        layout.addLayout(wrap)
        layout.addStretch()

        self.btn_upload.clicked.connect(self.main_window.show_upload_audio)
        self.btn_record.clicked.connect(self.main_window.show_record_audio)


# -----------------------------
# Entry Point
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
