# DSP Code for Digital Audio Post Processor
# Holds all functions for digital filters:
# Lowpass, Highpass, Bandpass, Notch, EQ
# Created by Matthew Reyna

from scipy import signal
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, lfilter
from scipy.signal import freqz

import soundfile as sf
import sounddevice as sd
import datetime
# !! May or may not need !!
import matplotlib.pyplot as plt
import os
import numpy as np


# ------------ Filter Functions ------------------
# LPF
def LPF(x, fs):
    fc = 3000  # !! NEEDS TO BE VARIABLE VIA GUI !!
    order = 4
    nyq = 0.5 * fs
    Wn = fc / nyq

    # Construct Lowpass filter with specs
    sos = butter(order, Wn, btype="low", output="sos")

    # Plot Bode
    # plotBode(sos,fs,"Low-Pass Filter Bode Plot")
    # y = sosfiltfilt(sos,x)

    # Plot time graph
    # plotTime(x,y,fs,title = "Low-Pass Filter Time Graph")
    return sosfiltfilt(sos, x)

# HPF
def HPF(x, fs):
    fc = 1000  # !! NEEDS TO BE VARIABLE VIA GUI !!
    order = 4
    nyq = 0.5 * fs
    Wn = fc / nyq

    # Construct Highpass filter with specs
    sos = butter(order, Wn, btype="high", output="sos")

    # Plot Bode
    # plotBode(sos,fs,"High-Pass Filter Bode Plot")
    y = sosfiltfilt(sos, x)

    # Plot time graph
    # plotTime(x,y,fs,title = "High-Pass Filter Time Graph")
    return sosfiltfilt(sos, x)


# BPF
def BPF(x, fs):
    lowcut = 500  # Hz Temp Default  !! NEEDS TO BE VARIABLE VIA GUI !!
    highcut = 2400  # Hz Temp Default !! NEEDS TO BE VARIABLE VIA GUI !!
    order = 4
    nyq = 0.5 * fs
    low = lowcut / nyq  # low and high act as fc
    high = highcut / nyq

    sos = butter(order, [low, high], btype='band', output="sos")
    # plot bode
    # plotBode(sos,fs, title="Band-Pass Bode Plot")
    # y = sosfiltfilt(sos,x)

    # plot time graph
    # plotTime(x,y,fs,title="Band-Pass Filter Time Graph")
    return sosfiltfilt(sos, x)


# NOTCH
def NOTCH(x, fs):
    # notchCount = int(input("Enter number of notches to filter (max of 4): ").strip())
    # !! NEEDS TO BE VARIABLE VIA GUI !!
    f0 = 500  # Hz
    Q = 30
    # create filter
    b, a = signal.iirnotch(f0, Q, fs)

    # convert to SOS for sosfiltfilt (TF2!!!!!!!!!)
    sos = signal.tf2sos(b, a)
    # plot bode
    # plotBode(b,a,fs, title="Notch Bode Plot")
    # y = filtfilt(b,a,x)
    return signal.sosfiltfilt(sos, x)


# 3 Band Multi Eq
def threeBandEQ(x, fs):
    # !! NEEDS TO BE VARIABLE VIA GUI !!
    # gain in dB
    lowGain = 3.0
    midGain = 0.3
    highGain = 0.2

    low_fc = 100
    high_fc = 5000
    order = 4

    # The 3 band "filters"
    sosLow = butter(order, low_fc / (0.5 * fs), btype='low', output='sos')  # Like LPF
    sosMid = butter(order, [low_fc / (0.5 * fs), high_fc / (0.5 * fs)], btype='band', output='sos')  # Like BPF
    sosHigh = butter(order, high_fc / (0.5 * fs), btype='high', output='sos')  # Like HPF

    # The 3 bands (low, mid and high)
    lowBand = sosfiltfilt(sosLow, x)
    midBand = sosfiltfilt(sosMid, x)
    highBand = sosfiltfilt(sosHigh, x)

    # combine gain and bands into one signal
    y = lowGain * lowBand + midGain * midBand + highGain * highBand
    return y


# ------------------------ Compressor Shi ------------------
# Linear to Db function
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


# Compressor filter audio idk whatever ts is for
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


# ------------ end compressor -------------------
# dB to Linear func
def dbToLinear(db):
    return 10 ** (db / 20)


# Audio Normalizer
def normalizeAudio(x):
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return x


# ---------- !!! Apply filter function !!! ----------
def applyFilter(infile, outfile, mode, normalize=True):
    x, fs = sf.read(infile, always_2d=False)

    # convert to mono for rn
    if x.ndim == 2:
        x = np.mean(x, axis=1)

    x = np.asarray(x, dtype=np.float64)

    # Filter states
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
    elif mode == "NOTCH":
        y = NOTCH(x, fs)
    else:
        y = x

    if normalize:
        y = normalizeAudio(y)

    sf.write(outfile, y, fs)
    return outfile


# !! May or may not need !!
# ------------ Plot Functions ------------------
# Plot TimeFunction
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


# Bode Plot
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


# FFT Compare
def plotFFT_compare(x, y, fs):
    N = len(x)

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    f = np.fft.rfftfreq(N, d=1 / fs)

    magX = np.abs(X)
    magY = np.abs(Y)

    plt.figure(figsize=(10, 5))
    plt.plot(f, magX, label="Original")
    plt.plot(f, magY, label="Equalized", alpha=0.7)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Comparison")
    plt.xlim(0, 8000)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------
# ******* MATTHEW I'M ADDING THIS TO WORK ON THE LIVE AUDIO PLS DON'T KILL ME *******
#------------------------------------------------------------------------------------
# This function filters one small chunk of live microphone audio. The audio_block is NOT
# a file... Rather it is the current block of samples coming from the microphone callback.
def applyLiveFilter(audio_block, fs, mode):
    # This if block applies a low-pass filter.
    if mode == "LPF": # Keeps lower frequencies and reduces higher ones.
        cutoff = 3000
        # Convert cutoff from Hz into normalized frequency.
        # Butterworth filters expect cutoff as a fraction of Nyquist frequency.
        b, a = butter(4, cutoff / (0.5 * fs), btype = "low")
        return lfilter(b, a, audio_block) # Apply filter to the current audio block.
    # This elif block applies a high-pass filter.
    elif mode == "HPF": # Keeps higher frequencies and reduces lower ones.
        cutoff = 1000
        b, a = butter(4, cutoff / (0.5 * fs), btype = "high")
        return lfilter(b, a, audio_block)
    # This elif block applies a band-pass filter.
    elif mode == "BPF": # Keeps only frequencies BETWEEN low and high cutoff.
        low_cutoff = 500
        high_cutoff = 3000
        b, a = butter(4, [low_cutoff / (0.5 * fs), high_cutoff / (0.5 * fs)], btype = "band")
        return lfilter(b, a, audio_block)
    elif mode == "COMP": # Increases the loudness of an audio without clipping.
        threshold = 0.2 # Level where compression starts.
        ratio = 4.0     # How strong the compression is.

        # Make a copy so the original isn't modified directly.
        processed = np.copy(audio_block)

        for i in range(len(processed)):
            sample = processed[i]
            abs_sample = abs(sample)

            if abs_sample > threshold:
                # Compress the amount ABOVE threshold.
                excess = abs_sample - threshold
                # Reduce that excess by the compression ratio.
                compressed_excess = excess / ratio
                # New amplitude = threshold + compressed portion.
                new_sample = threshold + compressed_excess
                # Restore original sign (+ or -).
                processed[i] = np.sign(sample) * new_sample
        return processed
    elif mode == "EQ": # Boosts lows and reduces mids/highs slightly.
        low_gain = 1.8
        mid_gain = 0.7
        high_gain = 0.8

        low_fc = 300
        high_fc = 3000

        # Low band
        b_low, a_low = butter(2, low_fc / (0.5 * fs), btype = "low")
        low_band = lfilter(b_low, a_low, audio_block)

        # Mid band
        b_mid, a_mid = butter(2, [low_fc / (0.5 * fs), high_fc / (0.5 * fs)], btype = "band")
        mid_band = lfilter(b_mid, a_mid, audio_block)

        # High band
        b_high, a_high = butter(2, high_fc / (0.5 * fs), btype = "high")
        high_band = lfilter(b_high, a_high, audio_block)

        # Mix bands back together with gain changes.
        processed = (low_gain * low_band + mid_gain * mid_band + high_gain * high_band)

        # Prevent clipping issues.
        processed = np.clip(processed, -1.0, 1.0)
        return processed
    elif mode == "NOTCH": # Reduces one narrow problem frequency while leaving all else the same.
        notch_freq = 60.0 # Frequency to remove.
        q = 30.0          # Higher Q = narrower notch.
        b, a = iirnotch(notch_freq, q, fs)
        return lfilter(b, a, audio_block)
    return audio_block
