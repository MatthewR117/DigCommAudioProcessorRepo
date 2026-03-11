# Master Python File that contains filters created before 2/25/26
# Filters include: Low-Pass, High-Pass, Band-Pass, and Notch Filter
# created by Matthew Reyna on 2/25/2026
from scipy import signal
from scipy.signal import butter,iirnotch, filtfilt
from scipy.signal import freqz
#from signal import pause

import RPi.GPIO as GPIO
import time
import matplotlib.pyplot as plt 
import os
import numpy as np
import soundfile as sf
import sounddevice as sd 

#------------------ Load .wav Function ------------------------
def loadWav():
    filename = input("Enter filename ending in .wav: ").strip()

    # input validation
    if not os.path.exists(filename):
        print("ERROR: filename not found:", filename)
        return None, None, None
    
    x,fs =  sf.read(filename, always_2d= False)

    # Convert to float32 for filtering
    x = x.astype(np.float32)

    # if stereo audio, use left channel
    if x.ndim == 2:
        x = x[:,0]

    base = os.path.splitext(os.path.basename(filename))[0]

    # Confirm file specs
    print(f"Loaded '{filename}' | fs = {fs} Hz | samples = {len(x)}" )

    return x,int(fs), base 

#------------------ Record audio function ------------------
def recordAudio():
    fs = 48000
    duration = 10
    print(f"Recording {duration} seconds at {fs} Hz")

    x = sd.rec(int(duration*fs), samplerate= fs, channels=1, dtype="float32")
    sd.wait()
    print("End Recording")
    x = x.flatten()

    # save unfiltered recording
    sf.write("master_unfiltered.wav", x, fs)
    print("file saved.")
    return x,fs,"master_unfiltered.wav"



# ------------ Filter Functions ------------------
# Initialize Functions
# LPF
def initLPF(x,fs):        # Low-Pass Initialize
    fc = float(input("Enter LPF cutoff freq in Hz (Ex: 3000): ").strip())
    order = int(input("Enter order of LPF (example 4): ").strip())
    nyq = 0.5 * fs
    Wn = fc / nyq

    b,a = butter(order, Wn, btype="low")
    # Plot Bode
    plotBode(b,a,fs,"Low-Pass Filter Bode Plot")
    y = filtfilt(b,a,x)

    # Plot time graph 
    plotTime(x,y,fs,title = "Low-Pass Filter Time Graph")
    return y, fc, order
    
# HPF
def initHPF(x,fs):                                    # High-Pass Initialize
    fc = float(input("Enter HPF cutoff freq in Hz (Ex: 1000): ").strip())
    order = int(input("Enter order of HPF (example 4): ").strip())
    nyq = 0.5 * fs
    Wn = fc / nyq

    b,a = butter(order, Wn, btype="high")
    # Plot Bode
    plotBode(b,a,fs,"High-Pass Filter Bode Plot")
    y = filtfilt(b,a,x)

    # Plot time graph 
    plotTime(x,y,fs,title = "High-Pass Filter Time Graph")
    return y, fc, order

# BPF
def initBPF(x,fs):  # Band-Pass Initialize
    lowcut = float(input("Enter BPF low cutoff freq in Hz (Ex: 500): ").strip())
    highcut = float(input("Enter BPF high cutoff freq in Hz (Ex: 2400): ").strip())
    order = int(input("Enter order of HPF (example 4): ").strip())
    nyq = 0.5 * fs
    low = lowcut / nyq # low and high act as fc
    high = highcut / nyq 

    b, a = butter(order, [low,high], btype='band')
    # plot bode
    plotBode(b,a,fs, title="Band-Pass Bode Plot")
    y = filtfilt(b,a,x)

    # plot time graph
    plotTime(x,y,fs,title="Band-Pass Filter Time Graph")
    return y,low,high,order

# NOTCH
def initNOTCH(x,fs):
    #notchCount = int(input("Enter number of notches to filter (max of 4): ").strip())
    f0 = float(input("Enter frequency to filter in Hz (EX: 60): ").strip())
    Q = int(input("Enter Q factor for Notch filter (EX: 30): ").strip())
    # create filter
    b,a = iirnotch(f0,Q,fs)
    # plot bode
    plotBode(b,a,fs, title="Notch Bode Plot")
    y = filtfilt(b,a,x)
    return y,f0,Q
   


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

#------------------ Terminal  Menu Functions ------------------
# Main Menu Function
def mainMenu():
    print("-- DIGITAL AUDIO POST PROCESSOR MASTER FILE --")
    print("                      v1                                                                        ")
    print("1. Load .wav file")
    print("2. Record Live Audio")
    print("3. Exit")

# filter menu
# --  Filter GPIO Specs: -- 
# LPF = GPIO(18) ; pin 12 
# HPF = GPIO(23) ; pin 16
# BPF = GPIO(24) ; pin 18

def filterMenu(x,fs,base_name):
    while True:
        # GPIO Setup
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP) # LPF Input pin with pullup
        GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP) # HPF Input pin with internal pullup
        GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP) # BPF Input pin with internal pullup

        # menu
        print("\n-- FILTER MENU --")
        print("1.) Apply Low Pass")
        print("2.) Apply High Pass")
        print("3.) Apply Band Pass")
        print("4.) Apply Notch")
        print("5.) Save Audio")
        print("6.) Back to Main Menu")

        filtChoice = input("> ").strip()

        # input choices
        #------- LPF -----------
        if filtChoice == "1" | GPIO.input(18):
            y, fc, order = initLPF(x,fs)
            x = y # update audio to be filtered
            print(f"Applied Low-Pass filter")
            print(f"Specs: fc = {fc} Hz | order = {order}")
        #------- HPF -----------
        elif filtChoice == "2" | GPIO.input(23):
            y, fc, order = initHPF(x,fs)
            x = y # update audio to be filtered
            print(f"Applied HPF-Pass filter")
            print(f"Specs: fc = {fc} Hz | order = {order}")
        #------- BPF -----------
        elif filtChoice == "3" | GPIO.input(24):
            y, low,high, order = initBPF(x,fs)
            x = y # update audio to be filtered
            print("Applied BPF-Pass filter")
            print(f"Specs: Low = {low} Hz | High = {high} Hz | order = {order}")
         #------- NOTCH -----------
        elif filtChoice == "4":
            y,fc,Q_fact = initNOTCH(x,fs)
            print("Applied Notch Filter")


         #------- SAVE ------------
        elif filtChoice == "5":
            # prompt user to name filtered audio file
            outname = input(f"Output name (no . wav) [{base_name}_out]: ").strip()
            # if user enters nothing
            if outname == "":
                outname = base_name + "_out"
            sf.write(outname + ".wav",x,fs)
            print("Saved: ", outname + ".wav")
        elif filtChoice == "6":
            return # back to mainMenu()
        else:
            print("Invalid input")
            


# ------------ Main Loop -------------------
while True:
    mainMenu()
    menuIn = input("> ").strip()

    if (menuIn == "1"):
        print("Load WAV selected\n")

        x,fs,base = loadWav()
        if x is not None:
            filterMenu(x,fs,base)
    elif(menuIn == "2"):
        print("Record selected\n")

        x,fs,base = recordAudio()
        filterMenu(x,fs,base)
    elif (menuIn == "3"):
        print("Terminating program...")
        print("Goodbye")
        break
    else:
        print("Invalid selection, please try again.")