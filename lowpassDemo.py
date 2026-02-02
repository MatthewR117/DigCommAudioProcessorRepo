# Lowpass Demo Code
# Created by Matthew Reyna on 1/28/2026
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt

# Demo values
T = 5.0                     # Period (s)
fs = 50.0                # Sample Rate (Hz)
cutoff = 2.0        # Cutoff Freq
order = 2

n = int(T * fs)
t = np.linspace(0, T, n, endpoint = False)

# Create signal with noise
sig = np.sin(2*np.pi*1.2*t)
noise = 1.5*np.cos(2*np.pi*9*t) + 0.5*np.sin(2*np.pi*12*t)

data = sig + noise

# Lowpass filter function
def lowpassFilter(data,  cutoff ,fs, order):
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype = 'low')
    y = filtfilt(b, a, data)
    return y

# Apply lowpass filter
y = lowpassFilter(data, cutoff, fs, order)

# Display results
plt.figure()
plt.plot(t, data, label = "Signal with noise")
plt.plot(t, y, linewidth = 2, label = "Lowpass filtered signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Lowpass Filter Demo")
plt.legend()
plt.grid(True)
plt.show()
