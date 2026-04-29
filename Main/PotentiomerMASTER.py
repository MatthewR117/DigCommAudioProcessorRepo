import spidev
import time
import os

# --- SPI SETUP ---
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000

# --- STATE ---
last_volume = -1
last_q = -1
last_bit = -1

Q_factor = 0
bit_depth = 16  # control value only (no DSP here)


# --- SPI READ ---
def read_channel(channel):
    resp = spi.xfer2([1, (8 + channel) << 4, 0])
    value = ((resp[1] & 3) << 8) | resp[2]
    return value


# --- CONTROLS ---
def set_volume(percent):
    global last_volume
    percent = max(0, min(100, percent))

    # This affects ENTIRE SYSTEM audio (including your GUI)
    if abs(percent - last_volume) >= 2:
        os.system(f"amixer sset Master {percent}%")
        last_volume = percent


def set_q_factor(percent):
    global Q_factor, last_q
    percent = max(0, min(100, percent))

    if abs(percent - last_q) >= 2:
        Q_factor = percent
        last_q = percent
        print(f"Q_factor: {Q_factor}")


def set_bitcrusher(percent):
    global bit_depth, last_bit
    percent = max(0, min(100, percent))

    # Map 0–100 → 4–16 bits
    new_bit = int(4 + (percent / 100) * 12)

    if new_bit != last_bit:
        bit_depth = new_bit
        last_bit = new_bit
        print(f"Bit depth (control only): {bit_depth}")


# --- MAIN LOOP ---
try:
    while True:
        # CH0 → Volume (GLOBAL)
        val0 = read_channel(0)
        volume = int((val0 / 1023) * 100)
        set_volume(volume)

        # CH1 → Bitcrusher (control value only)
        val1 = read_channel(1)
        crush = int((val1 / 1023) * 100)
        set_bitcrusher(crush)

        # CH2 → Q factor
        val2 = read_channel(2)
        q_val = int((val2 / 1023) * 100)
        set_q_factor(q_val)

        print(f"VOL={volume}% | BIT={bit_depth} | Q={Q_factor}")

        time.sleep(0.05)

except KeyboardInterrupt:
    spi.close()