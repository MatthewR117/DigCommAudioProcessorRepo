# DSP Code for Digital Audio Post Processor
# Holds all functions for digital filters:
# Lowpass, Highpass, Bandpass, Notch, EQ, Compression
# Rewritten to include COMP mode in applyFilter()

from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, freqz

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


# ------------ Filter Functions ------------------

# LPF
def LPF(x, fs):
    fc = 3000
    order = 4
    nyq = 0.5 * fs
    wn = fc / nyq

    sos = butter(order, wn, btype="low", output="sos")
    return sosfiltfilt(sos, x)


# HPF
def HPF(x, fs):
    fc = 1000
    order = 4
    nyq = 0.5 * fs
    wn = fc / nyq

    sos = butter(order, wn, btype="high", output="sos")
    return sosfiltfilt(sos, x)


# BPF
def BPF(x, fs):
    lowcut = 500
    highcut = 2400
    order = 4
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


# NOTCH
def NOTCH(x, fs):
    f0 = 60
    q = 30
    b, a = iirnotch(f0, q, fs)
    y = filtfilt(b, a, x)
    return y, f0, q


# 3-Band EQ
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


# ------------ Compression Helpers ------------------

def dbToLinear(db):
    return 10 ** (db / 20)


def linearToDb(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def envelopeFollower(signal_abs, sample_rate, attack_ms, release_ms):
    attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms * 0.001))
    release_coeff = np.exp(-1.0 / (sample_rate * release_ms * 0.001))

    env = np.zeros_like(signal_abs)
    prev = 0.0

    for i, sample in enumerate(signal_abs):
        if sample > prev:
            coeff = attack_coeff
        else:
            coeff = release_coeff

        prev = coeff * prev + (1.0 - coeff) * sample
        env[i] = prev

    return env


def compressChannel(
    x,
    sample_rate,
    threshold_db=-24.0,
    ratio=8.0,
    attack_ms=3.0,
    release_ms=100.0,
    makeup_gain_db=2.0
):
    x = x.astype(np.float64)

    env = envelopeFollower(
        signal_abs=np.abs(x),
        sample_rate=sample_rate,
        attack_ms=attack_ms,
        release_ms=release_ms
    )

    env_db = linearToDb(env)
    gain_reduction_db = np.zeros_like(env_db)

    over_threshold = env_db > threshold_db
    gain_reduction_db[over_threshold] = (
        threshold_db
        + (env_db[over_threshold] - threshold_db) / ratio
        - env_db[over_threshold]
    )

    total_gain_db = gain_reduction_db + makeup_gain_db
    total_gain_linear = dbToLinear(total_gain_db)

    return x * total_gain_linear


def softLimitAudio(x, limit=0.70):
    return limit * np.tanh(x / limit)


def COMP(x, fs):
    y = compressChannel(
        x=x,
        sample_rate=fs,
        threshold_db=-24.0,
        ratio=8.0,
        attack_ms=3.0,
        release_ms=100.0,
        makeup_gain_db=2.0
    )
    y = softLimitAudio(y, limit=0.70)
    return y


# ------------ Utility Functions ------------------

def normalizeAudio(x):
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return x


# ------------ Main Apply Filter Function ------------------

def applyFilter(infile, outfile, mode, normalize=True):
    x, fs = sf.read(infile, always_2d=False)

    # Convert stereo to mono for now
    if x.ndim == 2:
        x = np.mean(x, axis=1)

    x = np.asarray(x, dtype=np.float64)

    if mode == "LPF":
        y = LPF(x, fs)
    elif mode == "HPF":
        y = HPF(x, fs)
    elif mode == "BPF":
        y = BPF(x, fs)
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


# ------------ Optional Plot Functions ------------------

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
