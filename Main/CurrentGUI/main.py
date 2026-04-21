import sys
from pathlib import Path
import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFileDialog, QMessageBox,
    QFrame, QGridLayout, QListWidget, QListWidgetItem
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from dsp import applyFilter, get_bode_data

# Desktop-safe GPIO import
try:
    from gpiozero import Button
    GPIO_IMPORT_OK = True
except Exception:
    Button = None
    GPIO_IMPORT_OK = False


QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)


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
    btn.setFixedSize(132, 48)
    btn.setStyleSheet(
        f"""
        QPushButton {{
            font-size: 15px;
            padding: 6px 8px;
            border-radius: 10px;
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


def make_back_button(page: QWidget, text: str = "Back to Menu") -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(190, 56)
    btn.setStyleSheet(
        """
        QPushButton {
            font-size: 17px;
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
# Audio Browser Page
# -----------------------------
class AudioBrowserPage(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.current_directory = Path.home()
        self.audio_exts = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}
        self.build_ui()
        self.load_audio_files()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 46, 12, 10)
        layout.setSpacing(8)

        layout.addWidget(make_title("Audio Browser", pt=22))

        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 13px; color: white; padding: 2px;")
        layout.addWidget(self.path_label)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                color: black;
                border: 2px solid rgb(40,40,40);
                border-radius: 10px;
                font-size: 16px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 10px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: rgb(30,144,255);
                color: white;
            }
        """)
        self.file_list.itemDoubleClicked.connect(self.open_selected_item)
        layout.addWidget(self.file_list, stretch=1)

        top_buttons = QHBoxLayout()
        top_buttons.setSpacing(6)
        top_buttons.addStretch()

        up_btn = make_action_button("Up", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        refresh_btn = make_action_button("Refresh", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        open_btn = make_action_button("Open", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")

        up_btn.clicked.connect(self.go_up_directory)
        refresh_btn.clicked.connect(self.load_audio_files)
        open_btn.clicked.connect(self.open_selected_item)

        top_buttons.addWidget(up_btn)
        top_buttons.addWidget(refresh_btn)
        top_buttons.addWidget(open_btn)
        top_buttons.addStretch()

        layout.addLayout(top_buttons)

        bottom = QHBoxLayout()
        bottom.addStretch()
        back_btn = make_back_button(self, text="Back to Upload")
        bottom.addWidget(back_btn)
        bottom.addStretch()
        layout.addLayout(bottom)

    def load_audio_files(self):
        self.file_list.clear()
        self.path_label.setText(f"Folder: {self.current_directory}")

        try:
            entries = sorted(
                self.current_directory.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower())
            )

            for entry in entries:
                if entry.is_dir():
                    item = QListWidgetItem(f"[Folder] {entry.name}")
                    item.setData(Qt.ItemDataRole.UserRole, str(entry))
                    item.setData(Qt.ItemDataRole.UserRole + 1, "dir")
                    self.file_list.addItem(item)
                elif entry.suffix.lower() in self.audio_exts:
                    item = QListWidgetItem(entry.name)
                    item.setData(Qt.ItemDataRole.UserRole, str(entry))
                    item.setData(Qt.ItemDataRole.UserRole + 1, "file")
                    self.file_list.addItem(item)

        except Exception as e:
            QMessageBox.warning(self, "Browser Error", str(e))

    def open_selected_item(self):
        item = self.file_list.currentItem()
        if item is None:
            return

        path = Path(item.data(Qt.ItemDataRole.UserRole))
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

        if item_type == "dir":
            self.current_directory = path
            self.load_audio_files()
        else:
            self.main_window.upload_page.load_audio_file(str(path))
            self.main_window.show_upload_audio()

    def go_up_directory(self):
        parent = self.current_directory.parent
        if parent != self.current_directory:
            self.current_directory = parent
            self.load_audio_files()

    def go_back_to_menu(self):
        self.main_window.show_upload_audio()


# -----------------------------
# Upload Audio Page
# -----------------------------
class UploadAudioPage(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.main_window = parent

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.canvas = None
        self.figure = None
        self.current_plot_mode = None
        self.waveform_ax = None
        self.playhead_line = None
        self.current_waveform_duration = None
        self.last_playhead_draw_ms = -1

        self.player.positionChanged.connect(self.update_playhead)

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 46, 10, 10)
        layout.setSpacing(6)

        layout.addWidget(make_title("DSP Touch Interface", pt=22))

        self.file_label = QLabel("Selected file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-size: 14px; color: white; padding: 2px;")
        layout.addWidget(self.file_label)

        plot_shell = QWidget()
        plot_shell.setStyleSheet(
            "background-color: white;"
            "border-radius: 10px;"
            "border: 2px solid rgb(40,40,40);"
        )
        plot_shell_layout = QVBoxLayout(plot_shell)
        plot_shell_layout.setContentsMargins(4, 4, 4, 4)

        self.canvas_container = QVBoxLayout()
        self.canvas_container.setContentsMargins(0, 0, 0, 0)

        self.placeholder_label = QLabel("")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "background-color: white;"
            "color: black;"
            "font-size: 18px;"
            "padding: 20px;"
            "min-height: 360px;"
        )
        self.canvas_container.addWidget(self.placeholder_label)
        plot_shell_layout.addLayout(self.canvas_container)

        layout.addWidget(plot_shell, stretch=1)

        # Plot buttons
        plot_row = QHBoxLayout()
        plot_row.setSpacing(6)
        plot_row.addStretch()

        choose_btn = make_action_button("Choose File", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        fft_btn = make_action_button("FFT", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        bode_btn = make_action_button("Bode", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        wave_btn = make_action_button("Waveform", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")

        choose_btn.clicked.connect(self.main_window.show_audio_browser)
        fft_btn.clicked.connect(self.show_fft)
        bode_btn.clicked.connect(self.show_bode)
        wave_btn.clicked.connect(self.show_waveform)

        plot_row.addWidget(choose_btn)
        plot_row.addWidget(fft_btn)
        plot_row.addWidget(bode_btn)
        plot_row.addWidget(wave_btn)
        plot_row.addStretch()

        layout.addLayout(plot_row)

        # Playback buttons
        play_row = QHBoxLayout()
        play_row.setSpacing(6)
        play_row.addStretch()

        play_btn = make_action_button("Play", "rgb(34,139,34)", "rgb(24,110,24)", "rgb(14,90,14)")
        pause_btn = make_action_button("Pause", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        stop_btn = make_action_button("Stop", "rgb(200,0,0)", "rgb(170,0,0)", "rgb(140,0,0)")
        clear_btn = make_action_button("Clear", "rgb(180,180,180)", "rgb(160,160,160)", "rgb(130,130,130)")

        play_btn.clicked.connect(self.play_audio)
        pause_btn.clicked.connect(self.pause_audio)
        stop_btn.clicked.connect(self.stop_audio)
        clear_btn.clicked.connect(self.clear_audio)

        play_row.addWidget(play_btn)
        play_row.addWidget(pause_btn)
        play_row.addWidget(stop_btn)
        play_row.addWidget(clear_btn)
        play_row.addStretch()

        layout.addLayout(play_row)

        # Bottom row
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.time_label = QLabel("Playback position: 0.00 s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("font-size: 14px; color: white; padding: 2px;")

        bottom.addWidget(self.time_label)
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))

        layout.addLayout(bottom)

    def on_page_shown(self):
        pass

    def load_audio_file(self, file_path: str):
        audio_path = Path(file_path)

        self.main_window.currAudio = audio_path
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None

        self.file_label.setText(f"Selected file: {audio_path.name}")
        self.time_label.setText("Playback position: 0.00 s")
        self.current_plot_mode = None

        self._reset_plot_area()

    def get_active_audio_path(self):
        return self.main_window.procAudio or self.main_window.currAudio

    def play_audio(self):
        audio_to_play = self.get_active_audio_path()
        if audio_to_play is None:
            QMessageBox.information(self, "Audio", "Please load or record an audio file first.")
            return

        self.player.stop()
        url = QUrl.fromLocalFile(str(Path(audio_to_play).resolve()))
        self.player.setSource(url)
        self.player.play()

        self.file_label.setText(f"Selected file: {Path(audio_to_play).name}")

    def pause_audio(self):
        self.player.pause()
        current_time_sec = self.player.position() / 1000.0
        self.time_label.setText(f"Paused at: {current_time_sec:.2f} s")

    def stop_audio(self):
        self.player.stop()
        self.time_label.setText("Playback position: 0.00 s")
        self.last_playhead_draw_ms = -1
        if self.current_plot_mode == "waveform" and self.playhead_line is not None:
            self.playhead_line.set_xdata([0, 0])
            self.canvas.draw_idle()

    def show_fft(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            QMessageBox.information(self, "FFT", "Please select an audio file first.")
            return
        try:
            self.current_plot_mode = "fft"
            self.plot_fft(Path(audio_path))
        except Exception as e:
            QMessageBox.warning(self, "FFT Error", str(e))

    def show_waveform(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            QMessageBox.information(self, "Waveform", "Please select an audio file first.")
            return
        try:
            self.current_plot_mode = "waveform"
            self.plot_waveform(Path(audio_path))
        except Exception as e:
            QMessageBox.warning(self, "Waveform Error", str(e))

    def show_bode(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            QMessageBox.information(self, "Bode", "Please select an audio file first.")
            return
        try:
            self.current_plot_mode = "bode"
            self.plot_bode_placeholder(Path(audio_path))
        except Exception as e:
            QMessageBox.warning(self, "Bode Error", str(e))

    def clear_audio(self):
        self.player.stop()

        self.main_window.currAudio = None
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None

        self.file_label.setText("Selected file: (none)")
        self.time_label.setText("Playback position: 0.00 s")
        self.current_plot_mode = None
        self.playhead_line = None
        self.current_waveform_duration = None
        self.waveform_ax = None
        self.last_playhead_draw_ms = -1

        self._reset_plot_area()

    def _reset_plot_area(self):
        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
            self.figure = None

        self.placeholder_label.setText("")
        if self.placeholder_label.parent() is None:
            self.canvas_container.addWidget(self.placeholder_label)
        self.placeholder_label.show()

    def _ensure_canvas(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        if self.canvas is None:
            self.figure = Figure(figsize=(8.4, 5.2), facecolor="white")
            self.canvas = FigureCanvas(self.figure)
            self.placeholder_label.hide()
            self.canvas_container.addWidget(self.canvas)

    def _load_audio_mono(self, file_path: Path):
        x, fs = sf.read(str(file_path), always_2d=False)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        return x, fs

    def update_playhead(self, position_ms):
        t_sec = position_ms / 1000.0

        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.time_label.setText(f"Playback position: {t_sec:.2f} s")

        if self.current_plot_mode != "waveform":
            return
        if self.playhead_line is None:
            return
        if self.current_waveform_duration is None:
            return

        if self.last_playhead_draw_ms >= 0 and (position_ms - self.last_playhead_draw_ms) < 100:
            return
        self.last_playhead_draw_ms = position_ms

        if t_sec < 0:
            t_sec = 0
        if t_sec > self.current_waveform_duration:
            t_sec = self.current_waveform_duration

        self.playhead_line.set_xdata([t_sec, t_sec])
        self.canvas.draw_idle()

    def plot_waveform(self, file_path: Path):
        x, fs = self._load_audio_mono(file_path)

        self.current_waveform_duration = len(x) / fs

        max_display_points = 4000
        if len(x) > max_display_points:
            idx = np.linspace(0, len(x) - 1, max_display_points, dtype=int)
            x_plot = x[idx]
            t_plot = idx / fs
        else:
            x_plot = x
            t_plot = np.linspace(0, len(x) / fs, num=len(x), endpoint=False)

        self._ensure_canvas()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(t_plot, x_plot, linewidth=0.6)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.current_waveform_duration)

        self.playhead_line = ax.axvline(0, linestyle="--", linewidth=1.5)
        self.waveform_ax = ax

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

        self.playhead_line = None
        self.current_waveform_duration = None
        self.waveform_ax = None
        self.last_playhead_draw_ms = -1

        self._ensure_canvas()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(freqs, mag_db, linewidth=1.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fs / 2.0)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_bode_placeholder(self, file_path: Path):
        _, fs = self._load_audio_mono(file_path)

        mode = self.main_window.currFilterMode
        if mode is None:
            raise ValueError("No active filter selected. Apply a filter first to view its Bode plot.")

        w, mag_db = get_bode_data(mode, fs)
        if w is None or mag_db is None:
            raise ValueError(f"Bode plot is not supported for mode: {mode}")

        self.playhead_line = None
        self.current_waveform_duration = None
        self.waveform_ax = None
        self.last_playhead_draw_ms = -1

        self._ensure_canvas()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.semilogx(w, mag_db, linewidth=1.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(10, fs / 2.0)

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
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    gpioFilterSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Digital Audio Post Processor")
        self.setFixedSize(1024, 600)
        self.setStyleSheet("background-color: rgb(100, 100, 100);")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu_page = MenuPage(parent=self)
        self.upload_page = UploadAudioPage(parent=self)
        self.record_page = RecordAudioPage(parent=self)
        self.browser_page = AudioBrowserPage(parent=self)

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.record_page)
        self.stack.addWidget(self.browser_page)

        self.currAudio = None
        self.procAudio = None
        self.currFilterMode = None

        self.gpioFilterSignal.connect(self.handleGPIO)
        self.show_menu()

        # Desktop-safe GPIO setup
        self.gpio_enabled = False

        try:
            if GPIO_IMPORT_OK:
                self.lpfButton = Button(17, pull_up=True, bounce_time=0.05)
                self.hpfButton = Button(27, pull_up=True, bounce_time=0.05)
                self.bpfButton = Button(22, pull_up=True, bounce_time=0.05)

                self.autoQButton = Button(16, pull_up=True, bounce_time=0.05)
                self.eqButton = Button(23, pull_up=True, bounce_time=0.05)
                self.compButton = Button(15, pull_up=True, bounce_time=0.05)
                self.pwrButton = Button(20, pull_up=True, bounce_time=0.05)

                self.lpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("LPF")
                self.hpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("HPF")
                self.bpfButton.when_pressed = lambda: self.gpioFilterSignal.emit("BPF")

                self.autoQButton.when_pressed = lambda: self.gpioFilterSignal.emit("AUTO")
                self.eqButton.when_pressed = lambda: self.gpioFilterSignal.emit("EQ")
                self.compButton.when_pressed = lambda: self.gpioFilterSignal.emit("COMP")
                self.pwrButton.when_pressed = lambda: self.gpioFilterSignal.emit("PWR")

                self.gpio_enabled = True
                print("GPIO initialized successfully.")
            else:
                print("gpiozero import unavailable. Running in desktop mode.")

        except Exception as e:
            self.gpio_enabled = False
            print(f"GPIO unavailable on this machine. Running in desktop mode. Details: {e}")

    def handleGPIO(self, mode):
        if self.stack.currentWidget() is not self.upload_page:
            return

        if self.currAudio is None:
            return

        try:
            output_file = self.applyFilterToCurrAudio(mode)

            if output_file is not None:
                self.procAudio = output_file
                self.currFilterMode = mode
                self.upload_page.file_label.setText(f"Selected file: {Path(output_file).name}")

                if self.upload_page.current_plot_mode == "waveform":
                    self.upload_page.plot_waveform(Path(output_file))
                elif self.upload_page.current_plot_mode == "bode":
                    self.upload_page.plot_bode_placeholder(Path(output_file))
                elif self.upload_page.current_plot_mode == "fft":
                    self.upload_page.plot_fft(Path(output_file))

                self.upload_page.play_audio()

        except Exception as e:
            QMessageBox.warning(self, "Processing Error", str(e))

    def applyFilterToCurrAudio(self, mode):
        if self.currAudio is None:
            return None

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path(f"{self.currAudio.stem}_{mode}_{timestamp}.wav")

        applyFilter(
            str(self.currAudio),
            str(output_path),
            mode,
            normalize=True
        )

        self.procAudio = output_path
        self.currFilterMode = mode
        return output_path

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def show_upload_audio(self):
        self.stack.setCurrentWidget(self.upload_page)
        self.upload_page.on_page_shown()

    def show_record_audio(self):
        self.stack.setCurrentWidget(self.record_page)

    def show_audio_browser(self):
        self.browser_page.load_audio_files()
        self.stack.setCurrentWidget(self.browser_page)


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
