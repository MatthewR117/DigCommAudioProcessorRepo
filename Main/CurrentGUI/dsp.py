# dsp.py
# Signal-processing backend for the Digital Audio Post Processor.
# Contains LPF, HPF, BPF, EQ, compression, normalization, and
# Bode-response helper functions used by the GUI.

from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, freqz, sosfreqz

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utility
# -----------------------------
def clamp_frequency(value, low, high):
    return max(low, min(high, value))


def normalizeAudio(x):
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return x


def dbToLinear(db):
    return 10 ** (db / 20)


def linearToDb(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


# -----------------------------
# Filter Builders
# -----------------------------
def get_lpf_sos(fs, cutoff=3000):
    cutoff = clamp_frequency(cutoff, 20, int(fs / 2) - 1)
    order = 4
    wn = cutoff / (0.5 * fs)
    return butter(order, wn, btype="low", output="sos")


def get_hpf_sos(fs, cutoff=1000):
    cutoff = clamp_frequency(cutoff, 20, int(fs / 2) - 1)
    order = 4
    wn = cutoff / (0.5 * fs)
    return butter(order, wn, btype="high", output="sos")


def get_bpf_sos(fs, lowcut=500, highcut=2400):
    nyq = 0.5 * fs

    lowcut = clamp_frequency(lowcut, 20, int(fs / 2) - 2)
    highcut = clamp_frequency(highcut, lowcut + 1, int(fs / 2) - 1)

    order = 4
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", output="sos")

# -----------------------------
# DSP Modes
# -----------------------------
def LPF(x, fs, cutoff=3000):
    sos = get_lpf_sos(fs, cutoff=cutoff)
    return sosfiltfilt(sos, x)

def HPF(x, fs, cutoff=1000):
    sos = get_hpf_sos(fs, cutoff=cutoff)
    return sosfiltfilt(sos, x)

def BPF(x, fs, lowcut=500, highcut=2400):
    sos = get_bpf_sos(fs, lowcut=lowcut, highcut=highcut)
    return sosfiltfilt(sos, x)

def NOTCH(x, fs):
    f0 = 60
    q = 30
    b, a = iirnotch(f0, q, fs)
    y = filtfilt(b, a, x)
    return y, f0, q

def threeBandEQ(x, fs):
    low_gain = 3.0
    mid_gain = 0.3
    high_gain = 0.2

    low_fc = 100
    high_fc = 5000
    order = 4

    sos_low = butter(order, low_fc / (0.5 * fs), btype="low", output="sos")
    sos_mid = butter(order, [low_fc / (0.5 * fs), high_fc / (0.5 * fs)], btype="band", output="sos")
    sos_high = butter(order, high_fc / (0.5 * fs), btype="high", output="sos")

    low_band = sosfiltfilt(sos_low, x)
    mid_band = sosfiltfilt(sos_mid, x)
    high_band = sosfiltfilt(sos_high, x)

    y = low_gain * low_band + mid_gain * mid_band + high_gain * high_band
    return y
# ----------------------------------------------------------------------------------------------------------------------
# Compression - Tracks the amplitude "envelope" of the signal over time.
# This smooths rapid changes so compression reacts naturally
# instead of jumping angrily sample-by-sample.
def envelopeFollower(signal_abs, sample_rate, attack_ms, release_ms):
    # Convert attack/release times into smoothing coefficients.
    attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms * 0.001))
    release_coeff = np.exp(-1.0 / (sample_rate * release_ms * 0.001))

    env = np.zeros_like(signal_abs)
    prev = 0.0

    for i, sample in enumerate(signal_abs):
        # Use faster response when signal increases (attack),
        # slower response when signal decreases (release).
        if sample > prev:
            coeff = attack_coeff
        else:
            coeff = release_coeff

        # Exponential smoothing to track signal level.
        prev = coeff * prev + (1.0 - coeff) * sample
        env[i] = prev

    return env

def compressChannel(x, sample_rate, threshold_db=-24.0, ratio=8.0, attack_ms=3.0, release_ms=100.0, makeup_gain_db=2.0):
    # Core dynamic range compression function. Reduces loud parts of the signal while
    # keeping quieter parts, making the overall audio more balanced.

    x = x.astype(np.float64)

    # Track signal loudness over time using an envelope follower.
    env = envelopeFollower(signal_abs=np.abs(x), sample_rate=sample_rate, attack_ms=attack_ms, release_ms=release_ms)

    # Convert envelope amplitude to decibels for threshold comparison.
    env_db = linearToDb(env)

    # Initialize gain reduction array (in dB).
    gain_reduction_db = np.zeros_like(env_db)

    # Identify where signal exceeds the threshold.
    over_threshold = env_db > threshold_db

    # Apply compression curve: Above threshold, reduce dynamic range based on ratio.
    gain_reduction_db[over_threshold] = (
        threshold_db
        + (env_db[over_threshold] - threshold_db) / ratio
        - env_db[over_threshold]
    )

    # Combine compression with makeup gain to restore loudness.
    total_gain_db = gain_reduction_db + makeup_gain_db
    total_gain_linear = dbToLinear(total_gain_db)

    # Apply the computed gain to the signal.
    return x * total_gain_linear

# Soft limiter to prevent harsh clipping after compression. Uses
# tanh to smoothly "smoosh" peaks instead of cutting them off.
def softLimitAudio(x, limit=0.70):
    return limit * np.tanh(x / limit)


def COMP(x, fs):
    # Full compression pipeline: 1. Apply dynamic range compression
    #                            2. Apply soft limiting to prevent peaks from clipping.
    y = compressChannel(x=x, sample_rate=fs, threshold_db=-24.0, ratio=8.0, attack_ms=3.0, release_ms=100.0, makeup_gain_db=2.0)

    # Smoothly limit extreme peaks after compression.
    y = softLimitAudio(y, limit=0.70)
    return y

# -----------------------------
# Main Apply Filter Function
# -----------------------------
def applyFilter(
    infile,
    outfile,
    mode,
    normalize=True,
    low_cutoff=500,
    high_cutoff=2400
):
    x, fs = sf.read(infile, always_2d=False)

    if x.ndim == 2:
        x = np.mean(x, axis=1)

    x = np.asarray(x, dtype=np.float64)

    if mode == "LPF":
        y = LPF(x, fs, cutoff=high_cutoff)
    elif mode == "HPF":
        y = HPF(x, fs, cutoff=low_cutoff)
    elif mode == "BPF":
        y = BPF(x, fs, lowcut=low_cutoff, highcut=high_cutoff)
    elif mode == "EQ":
        y = threeBandEQ(x, fs)
    elif mode == "COMP":
        y = COMP(x, fs)
    else:
        y = x

    if normalize:
        y = normalizeAudio(y)

    sf.write(outfile, y, fs)
    return outfile
# ----------------------------------------------------------------------------------------------------------------------
# Bode Data - Compute the frequency response of the 3-band EQ filter. This
# is used for generating the Bode plot of the EQ mode.
def get_eq_response(fs, worN=4096):
    # Gains applied to each frequency band.
    low_gain = 3.0
    mid_gain = 0.3
    high_gain = 0.2

    low_fc = 100
    high_fc = 5000
    order = 4

    # Create filters for each band.
    sos_low = butter(order, low_fc / (0.5 * fs), btype="low", output="sos")
    sos_mid = butter(order, [low_fc / (0.5 * fs), high_fc / (0.5 * fs)], btype="band", output="sos")
    sos_high = butter(order, high_fc / (0.5 * fs), btype="high", output="sos")

    # Compute frequency response of each band.
    w, h_low = sosfreqz(sos_low, worN=worN, fs=fs)
    _, h_mid = sosfreqz(sos_mid, worN=worN, fs=fs)
    _, h_high = sosfreqz(sos_high, worN=worN, fs=fs)

    # Combine the bands using their respective gains.
    h_total = low_gain * h_low + mid_gain * h_mid + high_gain * h_high
    return w, h_total


def get_bode_data(mode, fs, low_cutoff=500, high_cutoff=2400, worN=4096):
    # Generate magnitude response (Bode plot data) for the selected filter
    # mode. This describes how the filter affects different frequencies.
    if mode == "LPF":
        sos = get_lpf_sos(fs, cutoff=high_cutoff)
        w, h = sosfreqz(sos, worN=worN, fs=fs)
    elif mode == "HPF":
        sos = get_hpf_sos(fs, cutoff=low_cutoff)
        w, h = sosfreqz(sos, worN=worN, fs=fs)
    elif mode == "BPF":
        sos = get_bpf_sos(fs, lowcut=low_cutoff, highcut=high_cutoff)
        w, h = sosfreqz(sos, worN=worN, fs=fs)
    elif mode == "EQ":
        # Special case: EQ uses a combination of filters instead of one.
        w, h = get_eq_response(fs, worN=worN)
    else:
        # Modes like "COMP" do not have traditional linear frequency response.
        return None, None

    # Convert magnitude to decibels for plotting.
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    return w, mag_db

# -----------------------------
# Optional Plot Helpers
# -----------------------------
def plotTime(x, y, fs, title="Time Domain", t0=0, t1=0.05):
    n = len(x)
    t = np.arange(n) / fs

    i0 = int(t0 * fs)
    i1 = int(t1 * fs)

    plt.figure(figsize=(12, 5))
    plt.plot(t[i0:i1], x[i0:i1], label="Original")
    plt.plot(t[i0:i1], y[i0:i1], label="Filtered")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plotBode(b, a, fs, title="Bode Plot"):
    w, h = freqz(b, a, worN=65536)
    freq = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    plt.figure(figsize=(10, 5))
    plt.semilogx(freq, mag_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.xlim(10, fs / 2)
    plt.show()


def plotFFT_compare(x, y, fs):
    n = len(x)

    x_fft = np.fft.rfft(x)
    y_fft = np.fft.rfft(y)

    f = np.fft.rfftfreq(n, d=1 / fs)

    mag_x = np.abs(x_fft)
    mag_y = np.abs(y_fft)

    plt.figure(figsize=(10, 5))
    plt.plot(f, mag_x, label="Original")
    plt.plot(f, mag_y, label="Processed", alpha=0.7)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Comparison")
    plt.xlim(0, 8000)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
