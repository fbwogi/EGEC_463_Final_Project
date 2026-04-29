import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os

# Load CSV
df = pd.read_csv("project_heart_rate_data.csv")

# Create output folder
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Clean data
df["filtered_latest"] = pd.to_numeric(df["filtered_latest"], errors="coerce")
df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
df = df.dropna(subset=["filtered_latest"])

time = df["time_seconds"].values
raw = df["selected_raw"].values
filtered = df["filtered_latest"].values
bpm = df["bpm"].values

# Detect peaks
peaks, _ = find_peaks(
    filtered,
    distance=20,
    prominence=np.std(filtered) * 0.5
)

# Raw plot
plt.figure(figsize=(12, 5))
plt.plot(time, raw)
plt.title("Raw PPG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Raw IR Value")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_raw_ppg_signal.png"), dpi=300)
plt.close()

# Filtered plot
plt.figure(figsize=(12, 5))
plt.plot(time, filtered, label="Filtered PPG")
plt.plot(time[peaks], filtered[peaks], "x", label="Peaks")
plt.title("Filtered PPG Signal with Peaks")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_filtered_ppg_peaks.png"), dpi=300)
plt.close()

# BPM plot
plt.figure(figsize=(12, 5))
plt.plot(time, bpm)
plt.title("BPM Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("BPM")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "project_bpm_trend.png"), dpi=300)
plt.close()

print(f"Plots saved to folder: {output_dir}/")