# main.py
# Main GUI for the Digital Audio Post Processor
# Handles page navigation, audio playback, and waveform/FFT/Bode displays.
# Achieves GPIO filter triggers and interaction with dsp.py.
#-----------------------------------------------------------------------------------------------------------------------
# Import GUI creation essentials, audio, time, and graph libraries.
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
import datetime
import sys

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from fontTools.merge import layout

from dsp import applyFilter, get_bode_data
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFileDialog, QMessageBox, QFrame, QGridLayout, QListWidget, QListWidgetItem, QSlider)
#-----------------------------------------------------------------------------------------------------------------------
# Attempt to import GPIO support (gpiozero) for Raspberry Pi hardware buttons.
# If unavailable (like when on my laptop), disable GPIO functionality.
# When disabled, the GUI can still run without crashing the program.
try:
    from gpiozero import Button
    GPIO_IMPORT_OK = True
except Exception:
    Button = None
    GPIO_IMPORT_OK = False

# Force Qt to use software rendering instead of GPU/OpenGL.
# This improves functionality on Raspberry Pi 5 displays and prevents the GUI from crashing.
QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
# ----------------------------------------------------------------------------------------------------------------------
# Shared UI Helpers - These helper functions create reusable styled UI elements
# (titles, buttons, dividers) so that the design stays consistent across ALL pages
# without having to repeat code. In short it just makes the program less repetitive.
def make_title(text: str, pt: int = 32) -> QLabel:

    lbl = QLabel(text)    # Create a large, bold, centered label for page titles.
    font = lbl.font()     # Access and modify the font properties.
    font.setPointSize(pt) # Control text size.
    font.setBold(True)    # Make the title bold font.
    lbl.setFont(font)

    # Center the text both horizontally and vertically.
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # Allow wrapping so long titles don't overflow smaller screens (like the 7" display).
    lbl.setWordWrap(True)

    # Set text color to white to match the dark GUI theme.
    lbl.setStyleSheet("color: white;")
    return lbl

def make_divider() -> QFrame:
    # Create a horizontal line used to visually separate sections.
    line = QFrame()

    # Set shape and style of the divider.
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)

    # Style the divider (light gray line).
    line.setStyleSheet("background-color: rgb(200, 200, 200); height: 2px;")
    return line

def make_big_button(text: str, bg_rgb: str, hover_rgb: str, pressed_rgb: str) -> QPushButton:
    # Create a large button used on main menus (useful for touch-screen).
    btn = QPushButton(text)

    # Fixed size ensures consistent layout across screens.
    btn.setFixedSize(260, 90)

    # Apply styling for normal, hover, and pressed states.
    btn.setStyleSheet(
        f"""
        QPushButton {{background-color: {bg_rgb}; color: white; font-size: 22px; font-weight: bold; border-radius: 16px;}}
        QPushButton:hover {{background-color: {hover_rgb};}}
        QPushButton:pressed {{background-color: {pressed_rgb};}}
        """)
    return btn

def make_action_button(text: str, normal_rgb: str, hover_rgb: str, pressed_rgb: str) -> QPushButton:
    # Smaller button used for actions (play, pause, stop, FFT, etc.).
    btn = QPushButton(text)

    # Smaller size compared to main menu buttons.
    btn.setFixedSize(132, 48)

    # Style for consistent UI behavior.
    btn.setStyleSheet(
        f"""
        QPushButton {{font-size: 15px; padding: 6px 8px; border-radius: 10px; background-color: {normal_rgb}; color: white;}}
        QPushButton:hover {{background-color: {hover_rgb};}}
        QPushButton:pressed {{background-color: {pressed_rgb};}}
        """)
    return btn

def make_back_button(page: QWidget, text: str = "Back to Menu") -> QPushButton:
    # Create a standardized "Back" button used across any menu.
    btn = QPushButton(text)

    # Medium-sized button for navigation.
    btn.setFixedSize(190, 56)

    # Style for consistent UI behavior.
    btn.setStyleSheet("""
        QPushButton {font-size: 17px; padding: 8px 14px; border-radius: 12px; background-color: rgb(80, 80, 80); color: white;}
        QPushButton:hover {background-color: rgb(70, 70, 70);}
        QPushButton:pressed {background-color: rgb(50, 50, 50);}
        """)
    # Connect button click to the page's "go back" function.
    btn.clicked.connect(page.go_back_to_menu)
    return btn

def save_wav_file_dialog(parent: QWidget, title: str, default_name: str = "recording.wav") -> str:
    # Open a file save dialog for exporting recorded audio as a .wav file.
    dialog = QFileDialog(parent, title)

    # Configure dialog behavior.
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dialog.setFileMode(QFileDialog.FileMode.AnyFile)

    # Suggest a default filename.
    dialog.selectFile(default_name)

    # Restrict visible file types.
    dialog.setNameFilter("WAV Files (*.wav)")
    dialog.setDefaultSuffix("wav")

    # Use Qt's internal file dialog (prevents OS theme issues ʜᴏᴘᴇғᴜʟʟʏ).
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

    # If user confirms, return selected file path..
    if dialog.exec():
        return dialog.selectedFiles()[0]
    return "" # If canceled, return emtpy string.
# ----------------------------------------------------------------------------------------------------------------------
# Audio Browser Page - Provides an in-app file browser for navigating folders and selecting audio
# files without using the computer's default "file explorer" window.
class AudioBrowserPage(QWidget):
    def __init__(self, parent: QMainWindow):
        # Initialize the browser page and store a reference to the main
        # window so this page can switch screens and send selected files
        # back to the Upload page.
        super().__init__(parent)
        self.main_window = parent

        # Start browsing from the user's home directory.
        self.current_directory = Path.home()

        # Supported audio files shown. Never even heard of some of these to be honest.
        self.audio_exts = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}

        # Build the visible layout and then populate the file list.
        self.build_ui()
        self.load_audio_files()

    def build_ui(self):
        # Create the main vertical layout for the browser page.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(3)

        # Page title
        layout.addWidget(make_title("Audio Browser", pt=22))

        # Label showing the currently opened folder path.
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 13px; color: white; padding: 2px;")
        layout.addWidget(self.path_label)

        # List widget used to show folders and supported audio files.
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {background-color: white; color: black; border: 2px solid rgb(40,40,40); border-radius: 10px; font-size: 16px; padding: 4px;}
            QListWidget::item {padding: 10px; margin: 2px;}
            QListWidget::item:selected {background-color: rgb(30,144,255); color: white;}
            """)
        # Double-clicking an item either opens a folder or loads an audio file.
        self.file_list.itemDoubleClicked.connect(self.open_selected_item)
        layout.addWidget(self.file_list, stretch=1)

        # Row of browser control buttons.
        top_buttons = QHBoxLayout()
        top_buttons.setSpacing(6)
        top_buttons.addStretch()

        up_btn = make_action_button("Up", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        refresh_btn = make_action_button("Refresh", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        open_btn = make_action_button("Open", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")

        # Connect buttons to their actions.
        up_btn.clicked.connect(self.go_up_directory)
        refresh_btn.clicked.connect(self.load_audio_files)
        open_btn.clicked.connect(self.open_selected_item)

        top_buttons.addWidget(up_btn)
        top_buttons.addWidget(refresh_btn)
        top_buttons.addWidget(open_btn)
        top_buttons.addStretch()

        layout.addLayout(top_buttons)

        # Bottom navigation row.
        bottom = QHBoxLayout()
        bottom.addStretch()
        back_btn = make_back_button(self, text="Back to Upload")
        bottom.addWidget(back_btn)
        bottom.addStretch()
        layout.addLayout(bottom)

    def load_audio_files(self):
        # Clear the current file list before reloading.
        self.file_list.clear()

        # Update the path label to show the current folder.
        self.path_label.setText(f"Folder: {self.current_directory}")

        try: # Sort entries so folders appear before files then sort alphabetically.
            entries = sorted(self.current_directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for entry in entries:
                if entry.is_dir():
                    # Shows folders in the list so the user can browse deeper.
                    item = QListWidgetItem(f"[Folder] {entry.name}")
                    item.setData(Qt.ItemDataRole.UserRole, str(entry))
                    item.setData(Qt.ItemDataRole.UserRole + 1, "dir")
                    self.file_list.addItem(item)
                elif entry.suffix.lower() in self.audio_exts:
                    # Only show files if they are supported audio formats.
                    item = QListWidgetItem(entry.name)
                    item.setData(Qt.ItemDataRole.UserRole, str(entry))
                    item.setData(Qt.ItemDataRole.UserRole + 1, "file")
                    self.file_list.addItem(item)

        except Exception as e:
            # Show an error if the folder couldn't be read.
            QMessageBox.warning(self, "Browser Error", str(e))

    def open_selected_item(self):
        # Get the item currently in the list.
        item = self.file_list.currentItem()
        if item is None:
            return

        # Recover the hidden path and type stored inside the item.
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

        if item_type == "dir":
            # If the user selected a folder, enter that folder and reload the file list.
            self.current_directory = path
            self.load_audio_files()
        else:
            # If the user selected an audio file, send it to the upload page and switch back there.
            self.main_window.upload_page.load_audio_file(str(path))
            self.main_window.show_upload_audio()

    def go_up_directory(self):
        # Move to the parent directory, unless already at the top.
        parent = self.current_directory.parent
        if parent != self.current_directory:
            self.current_directory = parent
            self.load_audio_files()

    def go_back_to_menu(self):
        # Return to the Upload page instead of the main menu.
        self.main_window.show_upload_audio()
# ----------------------------------------------------------------------------------------------------------------------
# Upload Audio Page - The main analysis page of the GUI program. Allows uers to load audio,
# display FFT/Bode/Waveform plots, adjust filter cutoff sliders, control playback, and
# monitor the playback position with a moving line on the waveform.
class UploadAudioPage(QWidget):
    def __init__(self, parent: QMainWindow):
        # Initiliaze the page and store a reference to the main window,
        # so this page can access shared audio state and switch pages.
        super().__init__(parent)
        self.main_window = parent

        # Media player objects used for audio playback inside the GUI.
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # References for the embedded matplotlib plot.
        self.canvas = None
        self.figure = None

        # Track which plot is currently being shown.
        # Possible values are: None, "fft", "bode", or "waveform."
        self.current_plot_mode = None

        # Store waveform plotting state for the moving playback line.
        self.waveform_ax = None
        self.playhead_line = None
        self.current_waveform_duration = None

        # Used to limit how often the playhead redraws,
        # which prevents GUI lag during playback.
        self.last_playhead_draw_ms = -1

        # Update the playhead position whenever the media player reports a new playback time.
        self.player.positionChanged.connect(self.update_playhead)

        # Build the visible interface.
        self.build_ui()

    def build_ui(self):
        # Main vertical layout for the Upload/Analysis page.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 46, 10, 10)
        layout.setSpacing(6)

        # Page title shown near the top of the display.
        layout.addWidget(make_title("DSP Touch Interface", pt=18))

        # Label showing which audio file is currently loaded.
        self.file_label = QLabel("Selected file: (none)")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("font-size: 12px; color: white; padding: 0px;")
        layout.addWidget(self.file_label)

        # Outer container for the graph area. This gives the plot a framed white background so it POPS.
        plot_shell = QWidget()
        plot_shell.setStyleSheet("background-color: white;" "border-radius: 10px;" "border: 2px solid rgb(40,40,40);")
        plot_shell_layout = QVBoxLayout(plot_shell)
        plot_shell_layout.setContentsMargins(4, 4, 4, 4)

        # Layout that will hold either a blank placeholder or the matplotlib graph canvas.
        self.canvas_container = QVBoxLayout()
        self.canvas_container.setContentsMargins(0, 0, 0, 0)

        # Blank placeholder shown when no graph has been selected yet.
        self.placeholder_label = QLabel("")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("background-color: white;" "color: black;" "font-size: 18px;" "padding: 20px;" "min-height: 280px;")
        self.canvas_container.addWidget(self.placeholder_label)
        plot_shell_layout.addLayout(self.canvas_container)

        # Add the graph area to the page and let it expand to take most of the space.
        layout.addWidget(plot_shell, stretch=1)

        # Row of buttons used to choose a file and switch plot types.
        plot_row = QHBoxLayout()
        plot_row.setSpacing(6)
        plot_row.addStretch()

        choose_btn = make_action_button("Choose File", "rgb(30,144,255)", "rgb(20,120,220)", "rgb(15,100,200)")
        fft_btn = make_action_button("FFT", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        bode_btn = make_action_button("Bode", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")
        wave_btn = make_action_button("Waveform", "rgb(90,90,90)", "rgb(75,75,75)", "rgb(60,60,60)")

        # Connect each button to its action.
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

        # Label for the low cutoff slider.
        self.low_cut_label = QLabel("Low Cutoff: 500 Hz")
        self.low_cut_label.setStyleSheet("font-size: 14px; color: white; padding: 2px;")
        layout.addWidget(self.low_cut_label)

        # Slider controlling the low cutoff frequency.
        # This affects HPF and the low edge of BPF.
        self.low_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_cut_slider.setRange(1, 22)
        self.low_cut_slider.setValue(1)
        self.low_cut_slider.valueChanged.connect(self.update_cutoff_labels)
        layout.addWidget(self.low_cut_slider)

        # Label for the high cutoff slider.
        self.high_cut_label = QLabel("High Cutoff: 2400 Hz")
        self.high_cut_label.setStyleSheet("font-size: 14px; color: white; padding: 2px;")
        layout.addWidget(self.high_cut_label)

        # Slider controlling the high cutoff frequency.
        # This affects LPF and the high edge of BPF.
        self.high_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_cut_slider.setRange(2, 22)
        self.high_cut_slider.setValue(3)
        self.high_cut_slider.valueChanged.connect(self.update_cutoff_labels)
        layout.addWidget(self.high_cut_slider)

        # Playback control buttons.
        play_row = QHBoxLayout()
        play_row.setSpacing(6)
        play_row.addStretch()

        play_btn = make_action_button("Play", "rgb(34,139,34)", "rgb(24,110,24)", "rgb(14,90,14)")
        pause_btn = make_action_button("Pause", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        stop_btn = make_action_button("Stop", "rgb(200,0,0)", "rgb(170,0,0)", "rgb(140,0,0)")
        clear_btn = make_action_button("Clear", "rgb(180,180,180)", "rgb(160,160,160)", "rgb(130,130,130)")

        # Connect playback buttons to audio control methods.
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

        # Debug filter buttons used to simulate GPIO hardware presses while on a desktop/laptop.
        # Comment out the code block here when testing on the real deal.
        '''
        debug_row = QHBoxLayout()
        debug_row.setSpacing(6)
        debug_row.addStretch()

        test_lpf = make_action_button("Test LPF", "rgb(90,90,140)", "rgb(80,80,130)", "rgb(70,70,120)")
        test_hpf = make_action_button("Test HPF", "rgb(90,90,140)", "rgb(80,80,130)", "rgb(70,70,120)")
        test_bpf = make_action_button("Test BPF", "rgb(90,90,140)", "rgb(80,80,130)", "rgb(70,70,120)")
        test_eq = make_action_button("Test EQ", "rgb(90,90,140)", "rgb(80,80,130)", "rgb(70,70,120)")
        test_comp = make_action_button("Test COMP", "rgb(140,90,140)", "rgb(130,80,130)", "rgb(120,70,120)")

        # Trigger the same filter-handling logic used by the physical rig.
        test_lpf.clicked.connect(lambda: self.main_window.handleGPIO("LPF"))
        test_hpf.clicked.connect(lambda: self.main_window.handleGPIO("HPF"))
        test_bpf.clicked.connect(lambda: self.main_window.handleGPIO("BPF"))
        test_eq.clicked.connect(lambda: self.main_window.handleGPIO("EQ"))
        test_comp.clicked.connect(lambda: self.main_window.handleGPIO("COMP"))

        debug_row.addWidget(test_lpf)
        debug_row.addWidget(test_hpf)
        debug_row.addWidget(test_bpf)
        debug_row.addWidget(test_eq)
        debug_row.addWidget(test_comp)
        debug_row.addStretch()

        layout.addLayout(debug_row)
        '''
        # Bottom row contains playback time information on the left and a navigation button on the right.
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.time_label = QLabel("Playback position: 0.00 s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("font-size: 14px; color: white; padding: 2px;")

        bottom.addWidget(self.time_label)
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))

        layout.addLayout(bottom)

        # Initialize the slider labels once all widgets are created.
        self.update_cutoff_labels()

    def get_low_cutoff(self):
        # Convert the slider value from kHz steps into Hz before passing it to the DSP code.
        return self.low_cut_slider.value() * 1000

    def get_high_cutoff(self):
        # Convert the slider value from kHz steps into Hz.
        return self.high_cut_slider.value() * 1000

    def update_cutoff_labels(self):
        # Read current slider values (displayed in kHz-style steps).
        low_khz = self.low_cut_slider.value()
        high_khz = self.high_cut_slider.value()

        # Prevent invalid band-pass settings where low cutoff
        # is equal to or higher than high cutoff.
        if low_khz >= high_khz:
            if self.sender() == self.low_cut_slider:
                # If the low slider moved too high, push the high slider up.
                high_khz = low_khz + 1
                self.high_cut_slider.blockSignals(True)
                self.high_cut_slider.setValue(min(high_khz, self.high_cut_slider.maximum()))
                self.high_cut_slider.blockSignals(False)
                high_khz = self.high_cut_slider.value()
            else:
                # If the high slider moved too low, pull the low slider down.
                low_khz = high_khz - 1
                self.low_cut_slider.blockSignals(True)
                self.low_cut_slider.setValue(max(low_khz, self.low_cut_slider.minimum()))
                self.low_cut_slider.blockSignals(False)
                low_khz = self.low_cut_slider.value()

        # Update the labels shown on the screen.
        self.low_cut_label.setText(f"Low Cutoff: {low_khz} kHz")
        self.high_cut_label.setText(f"High Cutoff: {high_khz} kHz")

    # This function runs whenever the Upload page is opened. Right now
    # it does not automatically load or plot anything, but it exists so
    # that behavior can be added later if needed.
    def on_page_shown(self):
        pass

    def load_audio_file(self, file_path: str):
        # Conver the selected file path into a Path object for easier file handling.
        audio_path = Path(file_path)

        # Store this file as the current "raw/original" audio in the main window.
        # Reset any processed version since a new file has now been selected.
        self.main_window.currAudio = audio_path
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None

        # Update the file label so the user sees which file is loaded.
        self.file_label.setText(f"Selected file: {audio_path.name}")

        # Reset playback and plotting state for the new file.
        self.time_label.setText("Playback position: 0.00 s")
        self.current_plot_mode = None

        # Clear the graph area so it stays blank until the user explicity chooses a chart type.
        self._reset_plot_area()

    # Return the processed audio file if one exists; otherwise return the original
    # loaded audio file. This makes playback and plotting automatically use the
    # newest filtered version after a DSP mode is applied.
    def get_active_audio_path(self):
        return self.main_window.procAudio or self.main_window.currAudio


    def play_audio(self):
        # Determine which file should be played... Processed audio if available,
        # otherwise just plays the original audio file.
        audio_to_play = self.get_active_audio_path()

        # If no file is loaded at all, show a warning and stop.
        if audio_to_play is None:
            QMessageBox.information(self, "Audio", "Please load or record an audio file first.")
            return

        # Stop any currently playing audio before starting the new source.
        self.player.stop()

        # Convert the file path into a Qt URL so that QMediaPlayer can read it.
        url = QUrl.fromLocalFile(str(Path(audio_to_play).resolve()))
        self.player.setSource(url)

        # Start a playback.
        self.player.play()

        # Update the displayed filename in case playback is now using a
        # processed version instead of the original.
        self.file_label.setText(f"Selected file: {Path(audio_to_play).name}")

    def pause_audio(self):
        # Pause audio playback at the current position.
        self.player.pause()

        # Convert playback position from milliseconds to seconds
        # so that the user can read and write the value more easily.
        current_time_sec = self.player.position() / 1000.0

        # Freeze the displayed time so the user knows exactly where playback was paused.
        self.time_label.setText(f"Paused at: {current_time_sec:.2f} s")

    def stop_audio(self):
        # Fully stop audio playback and return to the beginning.
        self.player.stop()

        # Reset the playback position text shown in the bottom-left label.
        self.time_label.setText("Playback position: 0.00 s")

        # Reset so future playback starts fresh.
        self.last_playhead_draw_ms = -1

        # If the waveform is currently visible, move the vertical playback line back to the start.
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
            self.plot_bode(Path(audio_path))
        except Exception as e:
            QMessageBox.warning(self, "Bode Error", str(e))

    def clear_audio(self):
        # Stop any audio currently playing.
        self.player.stop()

        # Remove both original and processed audio from shared app state.
        self.main_window.currAudio = None
        self.main_window.procAudio = None
        self.main_window.currFilterMode = None

        # Reset visible labels.
        self.file_label.setText("Selected file: (none)")
        self.time_label.setText("Playback position: 0.00 s")

        # Reset plot-related state variables.
        self.current_plot_mode = None
        self.playhead_line = None
        self.current_waveform_duration = None
        self.waveform_ax = None
        self.last_playhead_draw_ms = -1

        # Clear the graph area and return to the blank placeholder from before selection.
        self._reset_plot_area()

    def _reset_plot_area(self):
        # If a matplotlib canvas exists, remove it from the layout
        # and delete it so the plot area becomes blank again.
        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
            self.figure = None

        # Show the placeholder again instead of a graph.
        self.placeholder_label.setText("")
        if self.placeholder_label.parent() is None:
            self.canvas_container.addWidget(self.placeholder_label)
        self.placeholder_label.show()

    def _ensure_canvas(self):
        # Import matplotllib canvas and figure classes only when needed.
        # This keeps plotting setup localized to the graph functions.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        # If the canvas does not exist yet, create it and replace the placeholder.
        if self.canvas is None:
            self.figure = Figure(figsize=(8.4, 5.2), facecolor="white")
            self.canvas = FigureCanvas(self.figure)

            # Hide the blank placeholder once a real graph is ready.
            self.placeholder_label.hide()

            # Add the canvas into the plot area layout.
            self.canvas_container.addWidget(self.canvas)

    def _load_audio_mono(self, file_path: Path):
        # Read audio samples and sample rate from disk.
        x, fs = sf.read(str(file_path), always_2d=False)

        # Convert the audio into a NumPy float array for plotting/math.
        x = np.asarray(x, dtype=np.float64)

        # If the audio is stereo, average the two channels into mono.
        # This simplifies plotting and FFT display to a single waveform.
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        return x, fs

    def update_playhead(self, position_ms):
        # Convert playback position from milliseconds to seconds
        # so it matches the time axis used on the waveform plot.
        t_sec = position_ms / 1000.0

        # While audio is actively playing, continuously update the
        # playback-position label shown at the bottom-left of the page.
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.time_label.setText(f"Playback position: {t_sec:.2f} s")

        # Only move the playback cursor when the waveform plot is active.
        # FFT and Bode are frequency-domain plots, so a time cursor doesn't
        # make sense for those graphs.
        if self.current_plot_mode != "waveform":
            return
        if self.playhead_line is None:
            return
        if self.current_waveform_duration is None:
            return

        # Limit how often the cursor redraws to avoid lag issues. Redrawing
        # too often on a small display can slow the GUI down.
        if self.last_playhead_draw_ms >= 0 and (position_ms - self.last_playhead_draw_ms) < 100:
            return
        self.last_playhead_draw_ms = position_ms

        # Clamp the cursor so it stays within the plotted waveform range.
        if t_sec < 0:
            t_sec = 0
        if t_sec > self.current_waveform_duration:
            t_sec = self.current_waveform_duration

        # Move the vertical playhead line to the current time.
        self.playhead_line.set_xdata([t_sec, t_sec])

        # Request a lightweight redraw of the graph.
        self.canvas.draw_idle()

    def plot_waveform(self, file_path: Path):
        # Load the selected audio file and convert it to mono if needed.
        x, fs = self._load_audio_mono(file_path)

        # Store total duration so the playback cursor knows the full
        # left-to-right time range of the waveform.
        self.current_waveform_duration = len(x) / fs

        # Limit the number of plotted points for performance.
        # Longer files can have hundreds of thousands of samples,
        # whcih would make the GUI lag badly if all were drawn.
        max_display_points = 4000
        if len(x) > max_display_points:
            # Select evenly spaced sample indices across the full signal.
            idx = np.linspace(0, len(x) - 1, max_display_points, dtype=int)
            x_plot = x[idx]
            t_plot = idx / fs
        else:
            # If the audio is short enough, plot it directly.
            x_plot = x
            t_plot = np.linspace(0, len(x) / fs, num=len(x), endpoint=False)

        # Create the matplotlib canvas if it does not already exist.
        self._ensure_canvas()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Plot waveform amplitude as a function of time.
        ax.plot(t_plot, x_plot, linewidth=0.6)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        # Force the x-axis to cover the full duration of the audio,
        # even though the displayed points were downsampled.
        ax.set_xlim(0, self.current_waveform_duration)

        # Create the vertical playback cursor and start it at time zero.
        self.playhead_line = ax.axvline(0, linestyle="--", linewidth=1.5)

        # Store the axis in case it is needed later.
        self.waveform_ax = ax

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_fft(self, file_path: Path):
        # Load audio and convert to mono if needed.
        x, fs = self._load_audio_mono(file_path)

        # Use only the first second of audio for the FFT display.
        # This keeps the spectrum calculation fast and consistent.
        end_samp = min(len(x), int(1.0 * fs))
        if end_samp <= 1:
            raise ValueError("Selected segment is too short.")

        # Extract the segment used for frequency analysis.
        seg = x[:end_samp]

        # Remove DC offset so that the FFT plot is cleaner and not dominated
        # by a zero-frequency bias.
        seg = seg - np.mean(seg)

        # Apply a Hanning window before the FFT to reduce spectral leaking.
        n = len(seg)
        seg_w = seg * np.hanning(n)

        # Choose an FFT size that is the next power of 2.
        # This is efficient and gives a smooth frequency axis.
        nfft = 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

        # Compute the one-sided real FFT.
        x_fft = np.fft.rfft(seg_w, n=nfft)
        mag = np.abs(x_fft)

        # Create frequency axis in Hz.
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        # Convert magnitude to decibels for easier interpretation.
        mag_db = 20.0 * np.log10(mag + 1e-12)

        # FFT is not a time-domain plot, so reset waveform-specific state.
        self.playhead_line = None
        self.current_waveform_duration = None
        self.waveform_ax = None
        self.last_playhead_draw_ms = -1

        # Create canvas and draw the FFT.
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

    def plot_bode(self, file_path: Path):
        # Load the file only to recover its sample rate.
        # The Bode plot itself is based on the active filter
        # response, not the audio content itself.
        _, fs = self._load_audio_mono(file_path)

        # Determine which filter mode is currently active.
        mode = self.main_window.currFilterMode
        if mode is None:
            raise ValueError("No active filter selected. Apply a filter first to view its Bode plot.")

        # Ask dsp.py for the filter's magnitude response using the current slider cutoff settings.
        w, mag_db = get_bode_data(mode, fs, low_cutoff=self.get_low_cutoff(), high_cutoff=self.get_high_cutoff())

        # If the current mode does not support a Bode plot, stop here.
        if w is None or mag_db is None:
            raise ValueError(f"Bode plot is not supported for mode: {mode}")

        # Bode is not a time-domain plot, so reset waveform cursor state.
        self.playhead_line = None
        self.current_waveform_duration = None
        self.waveform_ax = None
        self.last_playhead_draw_ms = -1

        # Create canvas and draw the Bode plot.
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
# ----------------------------------------------------------------------------------------------------------------------
# Record Audio Page - This handles microphone recording within the GUI. It lets the user start
# and stop recording, save the captured audio as a .wav file, and then make that file available
# to the Upload page for playback, analysis, and filtering.
class RecordAudioPage(QWidget):
    def __init__(self, parent: QMainWindow):
        # Initiliaze the recording page and store a reference
        # to the main window for shared app state and navigation.
        super().__init__(parent)
        self.main_window = parent

        # Recording state flag so the GUI knows whether recording is active.
        self.is_recording = False

        # Audio recording settings.
        self.sample_rate = 44100  # Standard audio sample rate (44.1kHz)
        self.channels = 1         # Mono recording

        # List used to store chunks of microphone input as they arrive.
        self.recorded_chunks = []

        # Will hold the active sound device input stream while recording.
        self.stream = None

        # Build the visible page layout.
        self.build_ui()

    def build_ui(self):
        # Main vertical layout for the recording page.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 55, 18, 18)
        layout.setSpacing(10)

        # Page title.
        layout.addWidget(make_title("Record Audio", pt=28))
        layout.addWidget(make_divider())

        # Label showing the current recording state.
        self.state_label = QLabel("State: Idle")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 12px; border: 2px solid white;"
            "border-radius: 16px; background-color: black; color: white;")
        layout.addWidget(self.state_label)

        # Instruction text for the user.
        self.info_label = QLabel("Use Start to record from the microphone.\n" "When Stop is pressed, you will choose a .wav filename to save.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 16px; color: white; padding: 6px;")
        layout.addWidget(self.info_label)

        # Add empty space above the button row.
        layout.addStretch()

        row = QHBoxLayout()
        row.addStretch()

        # Large touch-friendly buttons for starting and stopping recording.
        start_btn = make_big_button("Start", "rgb(255,165,0)", "rgb(230,140,0)", "rgb(200,120,0)")
        stop_btn = make_big_button("Stop", "rgb(255,99,71)", "rgb(230,80,60)", "rgb(200,60,50)")

        # Connect buttons to recording control methods.
        start_btn.clicked.connect(self.start_recording)
        stop_btn.clicked.connect(self.stop_recording)

        row.addWidget(start_btn)
        row.addSpacing(20)
        row.addWidget(stop_btn)
        row.addStretch()

        layout.addLayout(row)

        layout.addStretch()

        # Bottom navigation row.
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(make_back_button(self))
        bottom.addStretch()
        layout.addLayout(bottom)

    def audio_callback(self, indata, frames, time, status):
        # This callback runs repeatedly while the microphone stream is active.
        # Each call delivers a chunk of recorded audio samples.

        # Print any stream warnings/errors for debugging purposes.
        if status:
            print(status)

        # Store a copy of the current audio chunk so all chunks can later
        # be combined into one complete recording.
        self.recorded_chunks.append(indata.copy())

    def start_recording(self):
        # Prevent starting a second recording if one is already active.
        if self.is_recording:
            QMessageBox.information(self, "Recording", "Recording is already in progress.")
            return

        try:
            # Clear any previous recording data before starting a new session.
            self.recorded_chunks = []

            # Create an audio input stream from the microphone.
            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype="float32", callback=self.audio_callback)

            # Start capturing microphone input.
            self.stream.start()
            self.is_recording = True

            # Update status text shown to the user.
            self.state_label.setText("State: Recording from microphone...")

        except Exception as e:
            # Show an error if the microphone or stream could not be opened.
            QMessageBox.warning(self, "Microphone Error", f"Could not start recording:\n{e}")

    def stop_recording(self):
        # Prevent stopping if no recording is currently active.
        if not self.is_recording:
            QMessageBox.information(self, "Recording", "No recording is currently in progress.")
            return

        try:
            # Stop and close the audio stream cleanly.
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_recording = False

        except Exception as e:
            # Show an error if the stream could not be closed properly.
            QMessageBox.warning(self, "Recording Error", f"Could not stop recording cleanly:\n{e}")
            return

        # If no audio chunks were captured, inform the user.
        if not self.recorded_chunks:
            self.state_label.setText("State: Recording stopped, but no audio was captured.")
            return

        # Combine all recorded chunks into one continuous NumPy array.
        audio_data = np.concatenate(self.recorded_chunks, axis=0)

        # Ask the user where to save the .wav file.
        file_path = save_wav_file_dialog(self, "Save Recorded Audio", "recording.wav")
        if not file_path:
            self.state_label.setText("State: Recording stopped. Save cancelled.")
            return

        try:
            # Write the audio to disk as a .wav file.
            sf.write(file_path, audio_data, self.sample_rate)
            saved_name = Path(file_path).name

            # Store the newly saved file as the current audio in the main app state.
            self.main_window.currAudio = Path(file_path)
            self.main_window.procAudio = None
            self.main_window.currFilterMode = None

            # Update page status and notify the user.
            self.state_label.setText(f"State: Recording saved successfully as {saved_name}")
            QMessageBox.information(self, "Recording Saved", f"Recording saved as:\n{saved_name}\n\nYou can now open it from the Upload Audio page.")
        except Exception as e:
            # Show an error if the file could not be saved.
            QMessageBox.warning(self, "Save Error", f"Could not save recording:\n{e}")
            self.state_label.setText("State: Failed to save recording.")

    def go_back_to_menu(self):
        # If the user leaves the page while recording is still active,
        # stop and close the stream to avoid leaving the microphone on.
        if self.is_recording and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.is_recording = False
            self.stream = None

        # Return to the main menu page.
        self.main_window.show_menu()
# ----------------------------------------------------------------------------------------------------------------------
# Main Window - The central controller for the application. It creates all GUI
# pages, stores shared audio/filter state, handles page navigation, receives
# GPIO or debug filter triggers, and calls the DSP processing functions.
class MainWindow(QMainWindow):
    # Custom Qt signal used to send filter mode names (LPF, HPF, etc.)
    # from GPIO button presses into the GUI thread safely.
    gpioFilterSignal = pyqtSignal(str)

    def __init__(self):
        # Initilizes the main application window.
        super().__init__()

        # Set window title and fixed size for the small 7" touchscreen display.
        self.setWindowTitle("Digital Audio Post Processor")
        self.setFixedSize(1024, 600)

        # Set overall background color for the GUI.
        self.setStyleSheet("background-color: rgb(100, 100, 100);")

        # Create a stacked widget so multiple pages can exist in the same
        # window and be switched without opening separate windows.
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Create all major pages of the program.
        self.menu_page = MenuPage(parent=self)
        self.upload_page = UploadAudioPage(parent=self)
        self.record_page = RecordAudioPage(parent=self)
        self.browser_page = AudioBrowserPage(parent=self)

        # Add the pages to the stacked widget.
        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.record_page)
        self.stack.addWidget(self.browser_page)

        # Shared audio/filter state used across multiple pages.
        self.currAudio = None       # Original/current audio file.
        self.procAudio = None       # Processed audio file AFTER applying a filter.
        self.currFilterMode = None  # Name of the currently active filter mode.

        # Connect GPIO/debug filter signals to the method that processes them.
        self.gpioFilterSignal.connect(self.handleGPIO)

        # Show the main menu first when the program starts.
        self.show_menu()

        # Track whether hardware GPIO buttons are available.
        self.gpio_enabled = False

        try:
            if GPIO_IMPORT_OK:
                # Create Button objects for each physical control input.
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

                # When a physical button is pressed, emit the matching mode name.
                # Into the GUI logic using the custom Qt signal.
                self.autoQButton.when_pressed = lambda: self.gpioFilterSignal.emit("AUTO")
                self.eqButton.when_pressed = lambda: self.gpioFilterSignal.emit("EQ")
                self.compButton.when_pressed = lambda: self.gpioFilterSignal.emit("COMP")
                self.pwrButton.when_pressed = lambda: self.gpioFilterSignal.emit("PWR")

                self.gpio_enabled = True
                print("GPIO initialized successfully.")
            else:
                # If gpiozero is unavailable, stay in desktop-safe mode.
                print("gpiozero import unavailable. Running in desktop mode.")

        except Exception as e:
            # If GPIO setup fails (common on non-Pi systems), disable hardware control
            # but allow the rest of the GUI to function as intended. :)
            self.gpio_enabled = False
            print(f"GPIO unavailable on this machine. Running in desktop mode. Details: {e}")

    def handleGPIO(self, mode):
        # Only allow filter application while the Upload page is active. This
        # prevents filters from being triggered while on unrelated pages.
        if self.stack.currentWidget() is not self.upload_page:
            return

        # If no audio file is currently loaded, ignore the filter request.
        if self.currAudio is None:
            return

        try:
            # Process the current audio using the selected filter mode.
            output_file = self.applyFilterToCurrAudio(mode)

            if output_file is not None:
                # Store the processed file and remember which mode produced it.
                self.procAudio = output_file
                self.currFilterMode = mode

                # Update the file label so the user sees the processed filename.
                self.upload_page.file_label.setText(f"Selected file: {Path(output_file).name}")

                # Redraw whichever plot is currently active using the processed file.
                if self.upload_page.current_plot_mode == "waveform":
                    self.upload_page.plot_waveform(Path(output_file))
                elif self.upload_page.current_plot_mode == "bode":
                    self.upload_page.plot_bode(Path(output_file))
                elif self.upload_page.current_plot_mode == "fft":
                    self.upload_page.plot_fft(Path(output_file))

                # Start playback of the newly processed audio.
                self.upload_page.play_audio()

        except Exception as e:
            QMessageBox.warning(self, "Processing Error", str(e))

    def applyFilterToCurrAudio(self, mode):
        # If there is no currently loaded audio, nothing can be processed.
        if self.currAudio is None:
            return None

        # Create a timestamped output filename so each processed file is unique.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path(f"{self.currAudio.stem}_{mode}_{timestamp}.wav")

        # Call the DSP module to apply the selected filter using the
        # current slider cutoff settings from the Upload page.
        applyFilter(str(self.currAudio), str(output_path), mode, normalize=True, low_cutoff=self.upload_page.get_low_cutoff(),
            high_cutoff=self.upload_page.get_high_cutoff())

        # Save the processed file and remember the active mode.
        self.procAudio = output_path
        self.currFilterMode = mode
        return output_path

    # Switch the stacked widget to the main menu page.
    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    # Switch to the Upload/Analysis page and let that page run any page-entry logic it needs.
    def show_upload_audio(self):
        self.stack.setCurrentWidget(self.upload_page)
        self.upload_page.on_page_shown()

    # Switch to the microphone recording page.
    def show_record_audio(self):
        self.stack.setCurrentWidget(self.record_page)

    # Refresh the browser contents, then show the embedded file browser page.
    def show_audio_browser(self):
        self.browser_page.load_audio_files()
        self.stack.setCurrentWidget(self.browser_page)
# ----------------------------------------------------------------------------------------------------------------------
# Menu Page - It's the startup screen of the application. It provides the main navigation
# buttons that send the user either to the Upload/Analysis page or the Record Audio page.
class MenuPage(QWidget):
    def __init__(self, parent: MainWindow):
        # Initialize the menu page and keep a reference to the main window so
        # this page can switch to other pages when buttons are pressed.
        super().__init__(parent)
        self.main_window = parent

        # Build the visual layout for the main menu.
        self.build_ui()

    def build_ui(self):
        # Main vertical layout for the startup menu.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 55, 20, 20)
        layout.setSpacing(14)

        # Main application title shown at startup.
        layout.addWidget(make_title("DIGITAL AUDIO POST PROCESSOR", pt=34))
        layout.addSpacing(12)

        # Grid layout used to center the two large menu buttons.
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(20)

        # Button that opens the Upload/Analysis workflow.
        self.btn_upload = make_big_button("Upload Audio","rgb(30,144,255)","rgb(20,120,220)","rgb(15,100,200)")

        # Button that opens the microphone recording workflow.
        self.btn_record = make_big_button("Record Audio","rgb(255,165,0)","rgb(230,140,0)","rgb(200,120,0)")

        # Place the buttons into the grid.
        grid.addWidget(self.btn_upload, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.btn_record, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        # Wrap the grid in a horizontal layout with stretch spacers
        # so the buttons remain centered on the screen..
        wrap = QHBoxLayout()
        wrap.addStretch()
        wrap.addLayout(grid)
        wrap.addStretch()

        # Add vertical stretch so the button group sits nicely in the middle.
        layout.addStretch()
        layout.addLayout(wrap)
        layout.addStretch()

        # Connect the menu buttons to page-switching methods in MainWindow.
        self.btn_upload.clicked.connect(self.main_window.show_upload_audio)
        self.btn_record.clicked.connect(self.main_window.show_record_audio)
# ----------------------------------------------------------------------------------------------------------------------
# Entry Point - The literal entry point for the program. Creates the Qt application, builds
# the main window, shows the GUI, and starts the Qt event loop.
def main():
    # Create the Qt application object required for all PyQt programs.
    app = QApplication(sys.argv)

    # Create the main GUI window.
    w = MainWindow()

    # Show the main window on screen.
    w.show()

    # Start the Qt event loop so the application stays running
    # and responds to user input until it is closed.
    sys.exit(app.exec())

if __name__ == "__main__":
    # Run the program only when this file is executed directly.
    main()
