# MainGUI
# GUI + DSP | Updated copy
# Current Features:
# - Low-pass, High-pass, Band-pass, EQ, Compress, and Notch filters
# - Upload page with in-GUI audio browser, waveform, FFT, save, and Q display
# - Record page for saving microphone input
# - Live Audio page for real-time microphone pass-through/filtering
#
# Created by Matthew Reyna and Caden Craddock
# Updated with Upload-only Q/potentiometer logic

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
    QFrame, QGridLayout, QComboBox, QListWidget,
    QListWidgetItem
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# GPIO is optional so the GUI can still run on a laptop.
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except Exception:
    Button = None
    GPIO_AVAILABLE = False

# SPI is optional so the GUI can still run when the ADC is not connected.
try:
    import spidev
    SPI_AVAILABLE = True
except Exception:
    spidev = None
    SPI_AVAILABLE = False

from dsp import applyFilter, applyLiveFilter

QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)

# Global date used for output filenames.
date = datetime.datetime.now().strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------------------------------------------------
# Potentiometer / ADC Reader
# ----------------------------------------------------------------------------------------------------------------------
class PotReader:
    def __init__(self):
        self.enabled = False
        self.spi = None

        if SPI_AVAILABLE:
            try:
                self.spi = spidev.SpiDev()
                self.spi.open(0, 0)
                self.spi.max_speed_hz = 500000
                self.enabled = True
                print("SPI ADC connected.")
            except Exception as e:
                print("SPI ADC unavailable:", e)
        else:
            print("spidev unavailable. Potentiometer disabled.")

    def read_channel(self, channel):
        if not self.enabled:
            return None

        resp = self.spi.xfer2([1, (8 + channel) << 4, 0])
        value = ((resp[1] & 3) << 8) | resp[2]
        return value

    def read_q(self):
        if not self.enabled:
            return None

        # CH2 is used for Q factor.
        val = self.read_channel(2)

        # Map ADC range 0-1023 to Q range 1-30.
        # Higher Q = narrower notch. Lower Q = wider notch.
        q = 1.0 + (val / 1023.0) * 29.0
        return q

    def close(self):
        if self.spi is not None:
            self.spi.close()


# ----------------------------------------------------------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    gpioFilterSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Digital Audio Post Processor")
        self.setFixedSize(1024, 600)
        self.setStyleSheet("background-color: rgb(150, 150, 150);")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Create pages.
        self.menu_page = MenuPage(parent=self)
        self.live_page = LiveAudioPage(parent=self)
        self.upload_page = UploadAudioPage(parent=self)
        self.record_page = RecordAudioPage(parent=self)
        self.browser_page = AudioBrowserPage(parent=self)

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.live_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.record_page)
        self.stack.addWidget(self.browser_page)

        # ---- Shared Audio Variables ----
        self.currAudio = None
        self.currFS = None
        self.procAudio = None
        self.currFilterMode = None
        self.tempAudio = None
        self.audioPos = None
        self.liveFilterMode = None
        #---------------------------------------
        # ---- Upload-only Q Factor Control ----
        # The Q dial only matters on Upload now. Live and Record ignore it.
        self.qFactor = 30.0
        self.qLockedByButton = False

        # ---- Potentiometer Setup ----
        # This timer only updates Q when the Upload page is active.
        self.pot_reader = PotReader()

        self.pot_timer = QTimer()
        self.pot_timer.setInterval(100)
        self.pot_timer.timeout.connect(self.update_pot_controls)
        self.pot_timer.start()

        # Route physical GPIO presses depending on which page is active.
        self.gpioFilterSignal.connect(self.routeGPIO)

        self.show_menu()

        # ----------- Filter GPIO Setup --------------------
        self.gpio_enabled = False

        if GPIO_AVAILABLE:
            try:
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
                self.lpfButton.when_activated = lambda: self.gpioFilterSignal.emit("LPF")
                self.hpfButton.when_activated = lambda: self.gpioFilterSignal.emit("HPF")
                self.bpfButton.when_activated = lambda: self.gpioFilterSignal.emit("BPF")
                self.autoQButton.when_activated = lambda: self.gpioFilterSignal.emit("AUTO_ON")
                self.eqButton.when_activated = lambda: self.gpioFilterSignal.emit("EQ")
                self.compButton.when_activated = lambda: self.gpioFilterSignal.emit("COMP")
                self.pwrButton.when_activated = lambda: self.gpioFilterSignal.emit("PWR")
                # ------------- GPIO Not Pressed ------------------------------
                self.lpfButton.when_deactivated = lambda: self.gpioFilterSignal.emit(None)
                self.hpfButton.when_deactivated = lambda: self.gpioFilterSignal.emit(None)
                self.bpfButton.when_deactivated = lambda: self.gpioFilterSignal.emit(None)
                self.autoQButton.when_deactivated = lambda: self.gpioFilterSignal.emit("AUTO_OFF")
                self.eqButton.when_deactivated = lambda: self.gpioFilterSignal.emit(None)
                self.compButton.when_deactivated = lambda: self.gpioFilterSignal.emit(None)

                self.gpio_enabled = True
                print("GPIO buttons connected.")
            except Exception as e:
                print("GPIO not available:", e)
        else:
            print("GPIO unavailable. Running without physical buttons.")

    # ------------------------------------------------------------------------------------------------------------------
    # Upload File Filtering
    # ------------------------------------------------------------------------------------------------------------------
    def handleGPIO(self, mode):
        if self.stack.currentWidget() is not self.upload_page:
            return

        if self.currAudio is None:
            print("No audio file is loaded.")
            return

        current_pos = self.upload_page.player.position()

        try:
            print(f"Applying {mode}...")

            tempOutFile = self.applyFilterToCurrAudio(mode)

            if tempOutFile is not None:
                self.procAudio = tempOutFile
                self.upload_page.file_label.setText(f"Selected file: {Path(tempOutFile).name}")

                if self.upload_page.current_plot_mode == "waveform":
                    self.upload_page.plot_waveform(Path(tempOutFile))
                else:
                    self.upload_page.plot_fft(Path(tempOutFile))

                self.upload_page.play_audio_at_position(current_pos)
                print(f"{mode} applied.")

        except Exception as e:
            print(f"{mode} failed.")
            QMessageBox.warning(self, "Processing Error", str(e))

    def applyFilterToCurrAudio(self, mode):
        if self.currAudio is None:
            return None

        self.deleteTemp()

        output_path = Path(f"TEMP_{self.currAudio.stem}_{mode}_{date}.wav")

        # File-based DSP. Keep this simple unless dsp.py is updated to accept q_factor.
        applyFilter(str(self.currAudio), str(output_path), mode, normalize=True)

        self.tempAudio = output_path
        self.currFilterMode = mode
        return output_path

    # ------------------------------------------------------------------------------------------------------------------
    # GPIO Router
    # ------------------------------------------------------------------------------------------------------------------
    def routeGPIO(self, mode):
        if mode is None:
            return

        current = self.stack.currentWidget()

        # Upload-only Auto Q behavior.
        if current == self.upload_page:
            self.toggleFilter(mode)
            if mode == "AUTO_ON":
                self.qFactor = 30.0
                self.qLockedByButton = True
                self.update_q_displays()
                print("Auto Q ON: Q locked at 30.0")
                return

            if mode == "AUTO_OFF":
                self.qLockedByButton = False
                self.update_q_displays()
                print("Auto Q OFF: dial control restored")
                return

        # Live ignores Auto Q and dial logic.
        elif current == self.live_page:
            self.toggleLiveFilter(mode)
            if mode in ("AUTO_ON", "AUTO_OFF", "AUTO"):
                print("Auto Q ignored on Live page.")
                return

        # Record and other pages ignore filter GPIO.
        else:
            return

    # ------------------------------------------------------------------------------------------------------------------
    # Upload-only Q / Potentiometer Logic
    # ------------------------------------------------------------------------------------------------------------------
    def update_pot_controls(self):
        # Q dial only updates while Upload page is visible.
        if self.stack.currentWidget() is not self.upload_page:
            return

        q = self.pot_reader.read_q()

        if q is None:
            return

        # If Auto Q button locked the value, ignore the dial.
        if self.qLockedByButton:
            self.update_q_displays()
            return

        self.qFactor = q
        self.update_q_displays()

    def get_current_q(self):
        return self.qFactor

    def update_q_displays(self):
        source = "AUTO" if self.qLockedByButton else "DIAL"
        text = f"Q: {self.get_current_q():.2f} | {source}"

        if hasattr(self.upload_page, "q_label"):
            self.upload_page.q_label.setText(text)

    # ------------------------------------------------------------------------------------------------------------------
    # Utility / Navigation
    # ------------------------------------------------------------------------------------------------------------------
    def deleteTemp(self):
        if self.tempAudio is not None:
            try:
                if self.procAudio is not None and os.path.exists(self.procAudio):
                    os.remove(self.procAudio)
            except Exception:
                pass

            self.tempAudio = None
            self.procAudio = None
            self.currFilterMode = None

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def show_upload_audio(self):
        self.stack.setCurrentWidget(self.upload_page)
        self.upload_page.on_page_shown()
        self.upload_page.setFocus()
        self.update_q_displays()

    def show_record_audio(self):
        self.stack.setCurrentWidget(self.record_page)

    def show_live_audio(self):
        self.stack.setCurrentWidget(self.live_page)
        self.live_page.setFocus()

    def show_audio_browser(self):
        self.browser_page.load_files()
        self.stack.setCurrentWidget(self.browser_page)

    def toggleLiveFilter(self, mode):
        if self.liveFilterMode == mode:
            self.liveFilterMode = None
            print(f"{mode} live filter off.")
        else:
            self.liveFilterMode = mode
            print(f"{mode} live filter on.")

    def toggleFilter(self, mode):
        if self.stack.currentWidget() is not self.upload_page:
            return

        if self.currAudio is None:
            return

        # If same filter is active, turn it off.
        if self.currFilterMode == mode and self.procAudio is not None:
            current_pos = self.upload_page.player.position()
            self.deleteTemp()
            print(f"{mode} turned off.")
            self.upload_page.file_label.setText(f"Selected file: {Path(self.currAudio).name}")

            if self.upload_page.current_plot_mode == "waveform":
                self.upload_page.waveformInit()
            else:
                self.upload_page.plot_fft(Path(self.currAudio))

            self.upload_page.play_audio_at_position(current_pos)
            return

        # Otherwise, turn the filter on.
        self.handleGPIO(mode)


# ----------------------------------------------------------------------------------------------------------------------
# Shared UI Helpers
# ----------------------------------------------------------------------------------------------------------------------
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
    btn.setFixedSize(300, 90)
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
    btn.setFixedSize(120, 60)
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
    btn.setFixedSize(130, 60)
    btn.setStyleSheet(
        """
        QPushButton {
            font-size: 16px;
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


def open_audio_file_dialog(parent: QWidget, title: str) -> str:
    dialog = QFileDialog(parent, title)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setNameFilter("Audio Files (*.wav *.mp3);;All Files (*.*)")
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


# ----------------------------------------------------------------------------------------------------------------------
# Audio Browser Page
# ----------------------------------------------------------------------------------------------------------------------
class AudioBrowserPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.current_directory = Path.home()
        self.audio_exts = {".wav", ".mp3"}

        self.build_ui()
        self.load_files()

    def build_ui(self):
        layout = QVBoxLayout(self)

        title = make_title("Audio Browser", pt=18)
        layout.addWidget(title)

        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.path_label)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet(
            """
            QListWidget {
                background-color: white;
                color: black;
                font-size: 16px;
                border-radius: 10px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background-color: rgb(30,144,255);
                color: white;
            }
            """
        )
        self.file_list.itemDoubleClicked.connect(self.open_selected_item)
        layout.addWidget(self.file_list, stretch=1)

        button_row = QHBoxLayout()

        up_btn = make_action_button("Up", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        open_btn = make_action_button("Open", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        back_btn = make_back_button(self)

        up_btn.clicked.connect(self.go_up)
        open_btn.clicked.connect(self.open_selected_item)

        button_row.addWidget(up_btn)
        button_row.addWidget(open_btn)
        button_row.addWidget(back_btn)

        layout.addLayout(button_row)

    def load_files(self):
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

    def open_selected_item(self, item=None):
        if item is None:
            item = self.file_list.currentItem()

        if item is None:
            return

        path = Path(item.data(Qt.ItemDataRole.UserRole))
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

        if item_type == "dir":
            self.current_directory = path
            self.load_files()
        else:
            self.main_window.upload_page.load_audio_file(str(path))
            self.main_window.show_upload_audio()

    def go_up(self):
        parent = self.current_directory.parent
        if parent != self.current_directory:
            self.current_directory = parent
            self.load_files()

    def go_back_to_menu(self):
        self.main_window.show_upload_audio()


# ----------------------------------------------------------------------------------------------------------------------
# Upload Audio Page
# ----------------------------------------------------------------------------------------------------------------------
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

        self.wave_timer = QTimer()
        self.wave_timer.setInterval(30)
        self.wave_timer.timeout.connect(self.updateWaveform)

        self.live_audio_data = None
        self.live_fs = None
        self.live_window_sec = 0.2
        self.live_line = None
        self.live_ax = None
        self.play_started = False
        self.file_dialog_open = False

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.build_ui()

    def saveAudio(self):
        if self.main_window.procAudio is None:
            QMessageBox.information(self, "Save Audio", "No filtered audio to save.")
            return

        outputFilt = f"{self.main_window.currAudio.stem}_{self.main_window.currFilterMode}_{date}.wav"
        savePath = save_wav_file_dialog(self, "Save Filtered Audio", outputFilt)

        if not savePath:
            return

        try:
            shutil.copy(self.main_window.procAudio, savePath)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def play_audio_at_position(self, position_ms):
        audio_to_play = self.get_active_audio_path()
        if audio_to_play is None:
            QMessageBox.warning(self, "Audio", "Please load or record an audio file.")
            return

        try:
            self.live_audio_data, self.live_fs = self.load_audio_mono(Path(audio_to_play))
        except Exception as e:
            QMessageBox.warning(self, "Waveform Error", f"Could not load waveform data:\n{e}")
            return

        self.player.stop()

        url = QUrl.fromLocalFile(str(Path(audio_to_play).resolve()))
        self.player.setSource(url)

        self.waveformInit()
        self.player.play()

        # Delay setPosition slightly so QMediaPlayer has time to load the new source.
        QTimer.singleShot(100, lambda: self.player.setPosition(position_ms))

        self.wave_timer.start()
        self.file_label.setText(f"Selected file: {Path(audio_to_play).name}")

    def keyPressEvent(self, event):
        try:
            if event.key() == Qt.Key.Key_L:
                print("Low-Pass key pressed.")
                self.main_window.toggleFilter("LPF")

            elif event.key() == Qt.Key.Key_H:
                print("High-Pass key pressed.")
                self.main_window.toggleFilter("HPF")

            elif event.key() == Qt.Key.Key_B:
                print("Band-Pass key pressed.")
                self.main_window.toggleFilter("BPF")

            elif event.key() == Qt.Key.Key_E:
                print("Equalization key pressed.")
                self.main_window.toggleFilter("EQ")

            elif event.key() == Qt.Key.Key_C:
                print("Compression key pressed.")
                self.main_window.toggleFilter("COMP")

            elif event.key() == Qt.Key.Key_N:
                print("Notch key pressed.")
                self.main_window.toggleFilter("NOTCH")

            else:
                super().keyPressEvent(event)

        except Exception as e:
            print("keyPressEvent error:", e)

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(10)

        title_row = QHBoxLayout()

        left_spacer = QWidget()
        left_spacer.setFixedWidth(180)

        self.title_label = make_title("UPLOAD AUDIO", 12)

        self.plot_select = QComboBox()
        self.plot_select.addItem("Choose File")
        self.plot_select.addItem("Waveform")
        self.plot_select.addItem("FFT")
        self.plot_select.setFixedSize(180, 50)
        self.plot_select.setStyleSheet(
            """
            QComboBox {
                font-size: 17px;
                padding: 6px;
                border-radius: 10px;
                background-color: rgb(128,128,128);
                color: white;
            }
            """
        )
        self.plot_select.activated[int].connect(self.change_plot_type)

        title_row.addWidget(left_spacer)
        title_row.addStretch()
        title_row.addWidget(self.title_label)
        title_row.addStretch()
        title_row.addWidget(self.plot_select)

        layout.addLayout(title_row)

        self.q_label = QLabel("Q: 30.00 | DIAL")
        self.q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.q_label.setStyleSheet("font-size: 14px; color: white;")
        layout.addWidget(self.q_label)

        self.file_label = QLabel("Selected file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-size: 16px; color: white; padding: 4px;")
        layout.addWidget(self.file_label)

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

        self.placeholder_label = QLabel("FFT or waveform will appear here after an audio file is selected.")
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

        play_btn = make_action_button("Play", "rgb(34,139,34)", "rgb(24,110,24)", "rgb(14,90,14)")
        stop_btn = make_action_button("Stop", "rgb(200,0,0)", "rgb(170,0,0)", "rgb(140,0,0)")
        notch_btn = make_action_button("Notch", "rgb(17,17,232)", "rgb(17,17,202)", "rgb(8,8,138)")
        clear_btn = make_action_button("Clear", "rgb(180,180,180)", "rgb(160,160,160)", "rgb(130,130,130)")
        save_btn = make_action_button("Save Audio", "rgb(255,140,0)", "rgb(230,120,0)", "rgb(200,100,0)")

        play_btn.clicked.connect(self.play_audio)
        stop_btn.clicked.connect(self.stop_audio)
        notch_btn.clicked.connect(self.notch_audio)
        clear_btn.clicked.connect(self.clear_audio)
        save_btn.clicked.connect(self.saveAudio)

        button_row.addWidget(play_btn)
        button_row.addWidget(stop_btn)
        button_row.addWidget(notch_btn)
        button_row.addWidget(clear_btn)
        button_row.addWidget(save_btn)
        button_row.addWidget(make_back_button(self))
        button_row.addStretch()

        layout.addLayout(button_row)

    def change_plot_type(self, index):
        try:
            plot_type = self.plot_select.itemText(index)
            print("Dropdown selected:", plot_type)

            if plot_type == "Choose File":
                self.main_window.show_audio_browser()

                self.plot_select.blockSignals(True)
                self.plot_select.setCurrentText("Waveform")
                self.plot_select.blockSignals(False)
                return

            audio_path = self.get_active_audio_path()
            if audio_path is None:
                QMessageBox.information(self, "Audio", "Please load an audio file first.")
                return

            if plot_type == "Waveform":
                self.show_waveform()
            elif plot_type == "FFT":
                self.show_fft()

        except Exception as e:
            print("Dropdown error:", e)

    def on_page_shown(self):
        pass

    def prompt_for_audio_file(self):
        if self.file_dialog_open:
            return

        self.file_dialog_open = True
        file_path = open_audio_file_dialog(self, "Select Audio File")
        self.file_dialog_open = False

        if not file_path:
            return

        self.load_audio_file(file_path)

    def waveformInit(self):
        self._ensure_canvas()
        self.figure.clear()

        self.live_ax = self.figure.add_subplot(111)
        self.live_ax.set_xticks([])
        self.live_ax.set_yticks([])

        t = np.linspace(0, self.live_window_sec, 1000)
        y = np.zeros_like(t)

        for spine in self.live_ax.spines.values():
            spine.set_visible(False)

        self.figure.set_facecolor("grey")
        self.live_ax.set_facecolor("black")

        self.live_line, = self.live_ax.plot(t, y, linewidth=1.5, color="red")
        self.figure.tight_layout()
        self.canvas.draw()

    def load_audio_file(self, file_path: str):
        self.main_window.deleteTemp()
        audio_path = Path(file_path)

        self.main_window.currAudio = audio_path
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None

        self.file_label.setText(f"Selected file: {audio_path.name}")
        self.current_plot_mode = "waveform"

        self.waveformInit()

    def get_active_audio_path(self):
        return self.main_window.procAudio or self.main_window.currAudio

    def play_audio(self, startPos=None):
        audio_to_play = self.get_active_audio_path()
        if audio_to_play is None:
            QMessageBox.information(self, "Audio", "Please load or record an audio file first.")
            return

        try:
            self.live_audio_data, self.live_fs = self.load_audio_mono(Path(audio_to_play))
        except Exception as e:
            QMessageBox.warning(self, "Waveform error", f"Could not load waveform data:\n{e}")
            return

        self.player.stop()

        url = QUrl.fromLocalFile(str(Path(audio_to_play).resolve()))
        self.player.setSource(url)

        self.waveformInit()

        if startPos is not None:
            self.player.setPosition(startPos)

        self.player.play()
        self.wave_timer.start()

        self.file_label.setText(f"Selected file: {Path(audio_to_play).name}")

    def stop_audio(self):
        self.player.stop()
        self.wave_timer.stop()

    def notch_audio(self):
        self.main_window.toggleFilter("NOTCH")

    def updateWaveform(self):
        if self.live_audio_data is None or self.live_fs is None:
            return

        pos = self.player.position()
        current_sample = int((pos / 1000.0) * self.live_fs)

        window_samples = int(self.live_window_sec * self.live_fs)
        start = max(0, current_sample - window_samples)
        end = current_sample

        segment = self.live_audio_data[start:end]

        if len(segment) < window_samples:
            padded = np.zeros(window_samples, dtype=np.float64)
            if len(segment) > 0:
                padded[-len(segment):] = segment
            segment = padded

        t = np.linspace(start / self.live_fs, end / self.live_fs, window_samples)

        if self.live_line is not None:
            self.live_line.set_data(t, segment)

        if self.live_ax is not None:
            self.live_ax.set_xlim(t[0], t[-1])
            self.live_ax.set_ylim(-1.0, 1.0)

        if self.canvas is not None:
            self.canvas.draw_idle()

        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState and pos > 0:
            self.wave_timer.stop()

    def show_fft(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            return

        try:
            self.current_plot_mode = "fft"
            self.plot_fft(Path(audio_path))
        except Exception as e:
            QMessageBox.warning(self, "FFT Error", str(e))

    def show_waveform(self):
        audio_path = self.get_active_audio_path()
        if audio_path is None:
            return

        self.current_plot_mode = "waveform"
        self.waveformInit()

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

    def load_audio_mono(self, file_path: Path):
        x, fs = sf.read(str(file_path), always_2d=False)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        return x, fs

    def plot_waveform(self, file_path: Path):
        x, fs = self.load_audio_mono(file_path)

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
        x, fs = self.load_audio_mono(file_path)

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
        self.wave_timer.stop()
        self.main_window.show_menu()


# ----------------------------------------------------------------------------------------------------------------------
# Record Audio Page
# ----------------------------------------------------------------------------------------------------------------------
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

        self.info_label = QLabel(
            "Use Start to record from the microphone. When Stop is pressed, you will choose a .wav filename to save."
        )
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 16px; color: white; padding: 6px;")
        layout.addWidget(self.info_label)
        layout.addWidget(make_divider())

        layout.addStretch()

        row = QVBoxLayout()
        row.addStretch()

        start_btn = make_big_button("Start", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        stop_btn = make_big_button("Stop", "rgb(255,99,71)", "rgb(230,80,60)", "rgb(200,60,50)")

        row.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        row.addSpacing(20)
        row.addWidget(stop_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        row.addStretch()

        start_btn.clicked.connect(self.start_recording)
        stop_btn.clicked.connect(self.stop_recording)

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
            self.info_label.setText("Status: Recording started...")
            self.is_recording = True
        except Exception as e:
            QMessageBox.warning(self, "Microphone Error", f"Could not start recording:\n{e}")

    def stop_recording(self):
        if not self.is_recording:
            QMessageBox.information(self, "Recording", "No recording is currently in progress.")
            return

        try:
            self.stream.stop()
            self.info_label.setText("Status: Recording complete.")
            self.stream.close()
            self.stream = None
            self.is_recording = False
        except Exception as e:
            QMessageBox.warning(self, "Recording Error", f"Could not stop recording cleanly:\n{e}")
            return

        if not self.recorded_chunks:
            return

        audio_data = np.concatenate(self.recorded_chunks, axis=0)

        file_path = save_wav_file_dialog(self, "Save Recorded Audio", "recording.wav")
        if not file_path:
            return

        try:
            sf.write(file_path, audio_data, self.sample_rate)
            saved_name = Path(file_path).name

            self.main_window.currAudio = Path(file_path)
            self.main_window.procAudio = None
            self.main_window.currFilterMode = None

            QMessageBox.information(
                self,
                "Recording Saved",
                f"Recording saved as:\n{saved_name}\n\nYou can now open it from the Upload Audio page."
            )
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save recording:\n{e}")

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


# ----------------------------------------------------------------------------------------------------------------------
# Live Audio Page
# ----------------------------------------------------------------------------------------------------------------------
class LiveAudioPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent

        self.sample_rate = 44100
        self.block_size = 4096
        self.channels = 1

        self.stream = None
        self.live_buffer = np.zeros(self.block_size * 10)

        self.canvas = None
        self.figure = None
        self.ax = None
        self.line = None

        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_waveform)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(make_title("Live Audio", pt=18))

        self.status_label = QLabel("Live audio stopped!")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: white;")
        layout.addWidget(self.status_label)

        plot_shell = QWidget()
        plot_shell.setStyleSheet(
            "background-color: white;"
            "border-radius: 16px;"
            "border: 2px solid rgb(220,220,220);"
        )

        plot_layout = QVBoxLayout(plot_shell)

        self.canvas_container = QVBoxLayout()
        plot_layout.addLayout(self.canvas_container)

        layout.addWidget(plot_shell, stretch=1)

        row = QHBoxLayout()
        row.addStretch()

        self.start_btn = make_action_button("Start Live", "rgb(34,139,34)", "rgb(24,110,24)", "rgb(14,90,14)")
        self.stop_btn = make_action_button("Stop Live", "rgb(200,0,0)", "rgb(170,0,0)", "rgb(140,0,0)")

        self.start_btn.clicked.connect(self.start_live_audio)
        self.stop_btn.clicked.connect(self.stop_live_audio)

        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(make_back_button(self))
        row.addStretch()

        layout.addLayout(row)

    def ensure_canvas(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        if self.canvas is None:
            self.figure = Figure(figsize=(8, 4), facecolor="white")
            self.canvas = FigureCanvas(self.figure)
            self.canvas_container.addWidget(self.canvas)

            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor("black")

            self.ax.set_ylim(-1.0, 1.0)
            self.ax.set_xlim(0, len(self.live_buffer))

            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.line, = self.ax.plot(self.live_buffer, linewidth=1.5)
            self.figure.tight_layout()
            self.canvas.draw()

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)

        audio = indata[:, 0].copy()

        mode = self.main_window.liveFilterMode

        # Live Audio intentionally ignores the Upload-page Q dial.
        # It uses whatever default behavior applyLiveFilter has for each mode.
        processed = applyLiveFilter(audio, self.sample_rate, mode)

        outdata[:, 0] = processed

        self.live_buffer = np.roll(self.live_buffer, -len(processed))
        self.live_buffer[-len(processed):] = processed

    def start_live_audio(self):
        if self.stream is not None:
            return

        self.ensure_canvas()

        try:
            print("Starting live audio...")
            print("Default devices:", sd.default.device)

            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype="float32",
                callback=self.audio_callback
            )

            self.stream.start()
            self.timer.start()
            self.status_label.setText("Live audio running!")
            self.setFocus()

        except Exception as e:
            print("Live Audio Error:", e)
            QMessageBox.warning(self, "Live Audio Error", str(e))

    def stop_live_audio(self):
        self.timer.stop()

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        self.status_label.setText("Live audio stopped!")

    def update_waveform(self):
        if self.line is not None:
            self.line.set_ydata(self.live_buffer)
            self.canvas.draw_idle()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_L:
            self.main_window.toggleLiveFilter("LPF")
        elif event.key() == Qt.Key.Key_H:
            self.main_window.toggleLiveFilter("HPF")
        elif event.key() == Qt.Key.Key_B:
            self.main_window.toggleLiveFilter("BPF")
        elif event.key() == Qt.Key.Key_E:
            self.main_window.toggleLiveFilter("EQ")
        elif event.key() == Qt.Key.Key_C:
            self.main_window.toggleLiveFilter("COMP")
        elif event.key() == Qt.Key.Key_N:
            self.main_window.toggleLiveFilter("NOTCH")
        else:
            super().keyPressEvent(event)

    def go_back_to_menu(self):
        self.stop_live_audio()
        self.main_window.show_menu()


# ----------------------------------------------------------------------------------------------------------------------
# Menu Page
# ----------------------------------------------------------------------------------------------------------------------
class MenuPage(QWidget):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.main_window = parent
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 40, 20, 20)
        layout.setSpacing(0)

        layout.addWidget(make_title("DIGITAL AUDIO", pt=34))
        layout.addWidget(make_title("POST PROCESSOR", pt=34))
        layout.addSpacing(20)
        layout.addWidget(make_divider())

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(20)

        self.btn_upload = make_big_button("Upload Audio", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        self.btn_record = make_big_button("Record Audio", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        self.btn_live = make_big_button("Live Audio", "rgb(27,212,64)", "rgb(24,178,55)", "rgb(19,138,43)")

        grid.addWidget(self.btn_upload, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_record, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_live, 2, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        wrap = QHBoxLayout()
        wrap.addStretch()
        wrap.addLayout(grid)
        wrap.addStretch()

        layout.addStretch()
        layout.addLayout(wrap)
        layout.addStretch()

        self.btn_upload.clicked.connect(self.main_window.show_upload_audio)
        self.btn_record.clicked.connect(self.main_window.show_record_audio)
        self.btn_live.clicked.connect(self.main_window.show_live_audio)


# ----------------------------------------------------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
