# DSP Code for Digital Audio Post Processor 
# Holds all functions for digital filters:
# Lowpass, Highpass, Bandpass, Notch, EQ
# Created by Matthew Reyna
from scipy import signal
from scipy.signal import butter,iirnotch, filtfilt, sosfiltfilt
from scipy.signal import freqz

import soundfile as sf
import sounddevice as sd 
# !! May or may not need !!
import matplotlib.pyplot as plt 
import os
import numpy as np

# ------------ Filter Functions ------------------
# Initialize Functions
# LPF
def LPF(x,fs):       
    fc = 3000 # !! USING 3 kHz FOR NOW, UNTIL MORE GUI !!
    order = 4
    nyq = 0.5 * fs
    Wn = fc / nyq

    # Construct Lowpass filter with specs
    sos = butter(order, Wn, btype = "low", output="sos")

    # Plot Bode
    plotBode(sos,fs,"Low-Pass Filter Bode Plot")
    y = sosfiltfilt(sos,x)

    # Plot time graph 
    plotTime(x,y,fs,title = "Low-Pass Filter Time Graph")
    return sosfiltfilt(sos, x)
    
# HPF
def HPF(x,fs):                                    
    fc = 1000 # !! USING 1 kHz FOR NOW, UNTIL MORE GUI !!
    order = 4
    nyq = 0.5 * fs
    Wn = fc / nyq

    # Construct Highpass filter with specs
    sos = butter(order, Wn, btype = "high", output = "sos")

    # Plot Bode
    plotBode(sos,fs,"High-Pass Filter Bode Plot")
    y = sosfiltfilt(sos,x)

    # Plot time graph 
    plotTime(x,y,fs,title = "High-Pass Filter Time Graph")
    return sosfiltfilt(sos,x)

# BPF 
def BPF(x,fs):  
    lowcut = 500 # Hz Temp Default
    highcut = 2400 # Hz Temp Default
    order = 4
    nyq = 0.5 * fs
    low = lowcut / nyq # low and high act as fc
    high = highcut / nyq 

    sos = butter(order, [low,high], btype='band', output="sos")
    # plot bode
    plotBode(sos,fs, title="Band-Pass Bode Plot")
    y = sosfiltfilt(sos,x)

    # plot time graph
    plotTime(x,y,fs,title="Band-Pass Filter Time Graph")
    return sosfiltfilt(sos,x)

# NOTCH
def NOTCH(x,fs):
    #notchCount = int(input("Enter number of notches to filter (max of 4): ").strip())
    f0 = 60 # Hz
    Q = 30
    # create filter
    b,a = iirnotch(f0,Q,fs)
    # plot bode
    plotBode(b,a,fs, title="Notch Bode Plot")
    y = filtfilt(b,a,x)
    return y,f0,Q
   
# 3 Band Multi Eq
def threeBandEQ(x,fs,lowGain,midGain,highGain,low_fc = 300, high_fc = 3000,order=4):
    # The 3 band "filters"
    sosLow = butter(order, low_fc/(0.5*fs),btype='low',output='sos') #Like LPF
    sosMid = butter(order, [low_fc/(0.5*fs),high_fc/(0.5*fs)],btype='band',output='sos') #Like BPF
    sosHigh = butter(order, high_fc/(0.5*fs),btype='high',output='sos') #Like HPF

    # The 3 bands (low, mid and high)
    lowBand = sosfiltfilt(sosLow, x)
    midBand = sosfiltfilt(sosMid, x)
    highBand = sosfiltfilt(sosHigh, x)

    # combine gain and bands into one signal 
    y = lowGain * lowBand + midGain * midBand + highGain * highBand
    return y

# dB to Linear func
def dbToLinear(db):
    return 10**(db/20)    

# Audio Normalizer
def normalizeAudio(x):
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return x

# Apply filter function
def applyFilter(infile, outfile, normalize = True):
    x,fs = sf.read(infile, always_2d = False)
    mode = "mode"

    # convert to mono for rn 
    if x.ndim == 2:
        x = np.mean(x, axis = 1)
    
    x = np.asarray(x, dtype=np.float64)

    # Filter states
    if mode == "LPF":
        y = LPF(x,fs)
    elif mode == "HPF":
        y = HPF(x,fs)
    elif mode == "BPF":
        y = BPF(x,fs)
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

    plt.figure(figsize=(12,5))
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

    freq = w * fs / (2*np.pi)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    plt.figure(figsize=(10,5))
    plt.semilogx(freq, mag_db)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)

    plt.grid(True, which="both")
    plt.xlim(10, fs/2)

    plt.show()

# FFT Compare 
def plotFFT_compare(x, y, fs):
    N = len(x)

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    f = np.fft.rfftfreq(N, d=1/fs)

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
