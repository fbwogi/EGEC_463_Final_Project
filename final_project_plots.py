import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os

df = pd.read_csv("project_heart_rate_data.csv")

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Clean data
for col in ["time_seconds", "selected_raw", "filtered_latest", "bpm", "bpm_fft", "hrv_ms"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["filtered_latest"])

time = df["time_seconds"].values
raw = df["selected_raw"].values
filtered = df["filtered_latest"].values
bpm = df["bpm"].values
bpm_fft = df["bpm_fft"].values if "bpm_fft" in df.columns else None
hrv = df["hrv_ms"].values if "hrv_ms" in df.columns else None

peaks, _ = find_peaks(
    filtered,
    distance=20,
    prominence=np.std(filtered) * 0.5
)

signal = filtered
fs = 25  # Sample rate in Hz

n_fft = 1024
fft_vals = np.abs(np.fft.rfft(signal * np.hamming(len(signal)), n=n_fft))
freq = np.fft.rfftfreq(n_fft, 1/fs)

mask = (freq >= 0) & (freq <= 3.0)
plt.figure(figsize=(12, 5))
plt.plot(freq[mask], fft_vals[mask], linewidth=2)

plt.title("Filtered PPG Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "ppg_frequency_spectrum.png"), dpi=300)
plt.show()

# Raw PPG frequency spectrum
fs = 25
n_fft = 1024

raw_centered = raw - np.mean(raw)
raw_windowed = raw_centered * np.hamming(len(raw_centered))

raw_fft_vals = np.abs(np.fft.rfft(raw_windowed, n=n_fft))
raw_freq = np.fft.rfftfreq(n_fft, d=1/fs)

mask = (raw_freq >= 0) & (raw_freq <= 5)

plt.figure(figsize=(12, 5))
plt.plot(raw_freq[mask], raw_fft_vals[mask])
plt.title("Raw PPG Signal Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_raw_ppg_frequency_spectrum.png"), dpi=300)
plt.close()

# Raw PPG
plt.figure(figsize=(12, 5))
plt.plot(time, raw)
plt.title("Raw PPG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Raw IR Value")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_raw_ppg_signal.png"), dpi=300)
plt.close()

# Filtered PPG with peaks
plt.figure(figsize=(12, 5))
plt.plot(time, filtered, label="Filtered PPG")
plt.plot(time[peaks], filtered[peaks], "x", label="Detected Peaks")
plt.title("Filtered PPG Signal with Detected Peaks")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_filtered_ppg_peaks.png"), dpi=300)
plt.close()

# BPM only
plt.figure(figsize=(12, 5))
plt.plot(time, bpm, label="BPM from Peaks")
plt.title("BPM Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("BPM")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_bpm_trend.png"), dpi=300)
plt.close()

# BPM vs FFT BPM
if bpm_fft is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(time, bpm, label="BPM from Peak Detection")
    plt.plot(time, bpm_fft, "--", label="BPM from FFT")
    plt.title("BPM Comparison: Peak Detection vs FFT")
    plt.xlabel("Time (seconds)")
    plt.ylabel("BPM")
    plt.ylim(40, 120)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "project_bpm_vs_fft.png"), dpi=300)
    plt.close()

# Difference plot
if bpm_fft is not None:
    bpm_error = np.abs(bpm - bpm_fft)

    plt.figure(figsize=(12, 5))
    plt.plot(time, bpm_error)
    plt.axhline(10, linestyle="--", label="10 BPM mismatch threshold")
    plt.title("Absolute Difference Between Peak BPM and FFT BPM")
    plt.xlabel("Time (seconds)")
    plt.ylabel("|BPM - BPM_FFT|")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "project_bpm_fft_error.png"), dpi=300)
    plt.close()

# HRV plot
if hrv is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(time, hrv)
    plt.title("Heart Rate Variability Estimate Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("HRV (ms)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "project_hrv_trend.png"), dpi=300)
    plt.close()

#hrv bs bpm
if hrv is not None and bpm is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(time, hrv, label="HRV")
    plt.plot(time, bpm, label="BPM")
    plt.title("HRV and BPM Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "project_hrv_bpm_trend.png"), dpi=300)
    plt.close()

print(f"Plots saved to folder: {output_dir}/")