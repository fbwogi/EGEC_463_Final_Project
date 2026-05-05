import csv
import os
import signal 
import sys 
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple   

import numpy as np
from  scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from max30102 import MAX30102

@dataclass
class Config:
    sample_rate_hz: int = 25
    window_seconds: int = 15
    lowcut_hz: float = 0.7
    highcut_hz: float = 3.0
    filter_order: int = 3
    min_bpm: int = 40
    max_bpm: int = 120
    use_red_channel: bool = False
    csv_filename: str = "project_heart_rate_data.csv"
    print_interval_seconds: float = 0.5
    duration_seconds: int = 60  # Set to None for indefinite monitoring
    enable_live_plot: bool = False
CFG = Config()

#sensor reader
class MAX30102Reader:
    def __init__(self, config: Config):
        self.config = config
        self.sensor = MAX30102()
        self.sensor.setup()
        self.red_buffer: Deque[int] = deque(maxlen=self.config.sample_rate_hz * self.config.window_seconds)
        self.ir_buffer: Deque[int] = deque(maxlen=self.config.sample_rate_hz * self.config.window_seconds)

    def read_samples(self):
        red, ir = self.sensor.read_sequential(amount=1)
        if len(red) == 0 or len(ir) == 0:
            raise RuntimeError("Failed to read data from MAX30102 sensor")
        
        red_raw = int(red[0])
        ir_raw = int(ir[0])
        return red_raw, ir_raw

    def shutdown(self):
        try: 
            self.sensor.shutdown()
        except Exception as e:
            pass

#signal processing
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1 or low >= high:
        raise ValueError(f"Invalid filter limits. lowcut={lowcut}, highcut={highcut}, fs={fs} "
                         "Check sample_rate_hz")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(raw_signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    if len(raw_signal) < max(30, order * 3):
        return raw_signal - np.mean(raw_signal)
    centered = raw_signal - np.mean(raw_signal)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, centered)
    return y

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    if len(signal) == 0:
        return signal
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-9:
        return signal - mean
    return (signal - mean) / std

def estimate_bpm(filtered_signal: np.ndarray, fs: float, min_bpm: int, max_bpm: int):
    #detect peaks in filtered ppg waveform 
    if len(filtered_signal) < int(fs * 5):
        return None, np.array([]), None
    normalized_signal = normalize_signal(filtered_signal)
    #if max bpm is 180, min distance between peaks is 60/180 = 0.333s
    #min_distance = int(fs * (60 / max_bpm))  
    min_distance = int(fs * (60 / max_bpm) * 1.2) #increase spacing between peaks

    #adaptive prominence 
    #adaptive_prominence = max(0.3, 0.5 * np.std(normalized_signal) )
    adaptive_prominence = 0.6 * np.std(normalized_signal) #lets make it less sensitive to noise

    peaks, _ = find_peaks(normalized_signal, distance=max(1, min_distance), prominence=adaptive_prominence)
    peak_intervals = np.diff(peaks) / fs

    if len(peaks) < 2:
        return None, peaks , None

    # Calculate the times of the peaks
    peak_times = peaks / fs
    rr_intervals = np.diff(peak_times)
    bpm_values = 60 / rr_intervals

    #remove irrational bpm values
    valid_bpm_values = bpm_values[(bpm_values >= min_bpm) & (bpm_values <= max_bpm)]
    if len(valid_bpm_values) == 0:
        return None, peaks, None 

    avg_bpm = float(np.mean(valid_bpm_values))
    
    valid_rr_intervals = rr_intervals[(bpm_values >= min_bpm) & (bpm_values <= max_bpm)]
    hrv_ms = float(np.std(valid_rr_intervals) * 1000) if len(valid_rr_intervals) > 1 else None
    return avg_bpm, peaks, hrv_ms

def estimate_bpm_fft(signal, fs):
    n = len(signal)
    if n < fs * 5:
        return None

    # Compute the FFT
    freq = np.fft.rfftfreq(n, 1/fs)
    fft_magnitude = np.abs(np.fft.rfft(signal))

    # Find the frequency with the highest amplitude (excluding DC)
    mask = (freq >= 0.7) & (freq <= 3.0)  # Filter for frequencies between 0 and 3 Hz
    if not np.any(mask):
        return None
    
    freqs_band = freq[mask]
    fft_band = fft_magnitude[mask]

    # Find the frequency with the highest amplitude in the filtered range
    if len(fft_band) == 0:
        return None

    dominant_freq = freqs_band[np.argmax(fft_band)]
    bpm_fft = dominant_freq * 60

    return float(bpm_fft)

def get_signal_quality(raw_signal: np.ndarray, filtered_window: np.ndarray, peaks: np.ndarray, hrv_ms:Optional[float]) -> str:
    if len(raw_signal) < 20:
        return "Wait....Collecting More Info........"
    raw_std = np.std(raw_signal)
    filtered_std = np.std(filtered_window)

    if raw_std < 50:
        return "Poor"
    
    if filtered_std < 0.01:
        return "Poor"

    if len(peaks) < 3:
        return "Fair"

    if hrv_ms is not None and hrv_ms > 250:
        return "Unstable"

    return "Good"

#CSV Logging
class CSVLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time_seconds", "red_raw", "ir_raw", "selected_raw", "filtered_latest", "bpm", "bpm_fft", "hrv_ms", "signal_quality"])

    def write(self, time_seconds: float, red_raw: int, ir_raw: int, selected_raw: int, filtered_latest: Optional[float], bpm: Optional[float], bpm_fft: Optional[float], hrv_ms: Optional[float], signal_quality: str):
        self.writer.writerow([f"{time_seconds:.4f}", red_raw, ir_raw, selected_raw, f"{filtered_latest:.2f}" if filtered_latest is not None else "None", f"{bpm:.2f}" if bpm is not None else "None", f"{bpm_fft:.2f}" if bpm_fft is not None else "None", f"{hrv_ms:.2f}" if hrv_ms is not None else "None", signal_quality])
        self.file.flush()

    def close(self):
        self.file.close()   

#heart rate monitor display
class HeartRateMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.reader = MAX30102Reader(config)
        self.logger = CSVLogger(config.csv_filename)

        self.max_samples = config.sample_rate_hz * config.window_seconds
        self.red_buffer: Deque[int] = deque(maxlen=self.max_samples)
        self.ir_buffer: Deque[int] = deque(maxlen=self.max_samples)
        self.bpm_smooth_buffer: Deque[float] = deque(maxlen=5)

        self.running = True
        self.start_time = time.time()
        self.last_print_time = 0.0
        if self.config.enable_live_plot:
            self.live_plot()

    def stop(self, *_args):
        self.running = False

    def print_dashboard(self, current_time: float, selected_raw: int, filtered_latest: Optional[float], bpm: Optional[float], hrv_ms: Optional[float], signal_quality: str):
        sys.stdout.write("\033[2J\033[H")
        bpm_text = f"{bpm:.2f}" if bpm is not None else "Calculating..."
        sys.stdout.flush()
        print("------------------------------------------")
        print(" Raspberry Pi PPG Heart Rate Monitor ")
        print("------------------------------------------")
        print(f"Time                : {current_time:.2f}s")
        print("------------------------------------------")
        print(f"Sensor              : MAX30102")
        print(f"Selected Channel    : {'Red' if self.config.use_red_channel else 'IR'}")
        print(f"Sample Rate         : {self.config.sample_rate_hz} Hz ")
        print(f"Window Size         : {self.config.window_seconds} s")
        print("------------------------------------------")
        print(f"Latest Raw Value    : {selected_raw}")
        print(f"Latest Filtered     : {filtered_latest:.2f}" if filtered_latest is not None else "Latest Filtered     : N/A")
        print(f"Estimated BPM       : {bpm_text}")
        print(f"HRV (ms)            : {hrv_ms:.2f}" if hrv_ms is not None else "HRV (ms)            : N/A")
        print(f"Signal Quality      : {signal_quality}")
        print("------------------------------------------") 
        print(f"CSV Log File       : {self.config.csv_filename}")
        print("Press Ctrl+C to stop the monitor")
        print(f"------------------------------------------")
    
    def compute_fft(self, filtered_window):
        fs = self.config.sample_rate_hz
        n_fft = 1024

        widowed = filtered_window * np.hamming(len(filtered_window))

        fft = np.abs(np.fft.fft(widowed, n=n_fft))
        freqs = np.fft.fftfreq(n_fft, 1/fs)

        mask = (freqs >= 0) & (freqs <= 3)
    
        return freqs[mask], fft[mask]

    def live_plot(self):
        plt.ion()
        self.fig, (self.ax_raw, self.ax_filtered, self.ax_bpm, self.ax_fft) = plt.subplots(4, 1, figsize=(11, 9))
        self.raw_line, = self.ax_raw.plot([], [])
        self.filtered_line, = self.ax_filtered.plot([], [], color="lime")
        self.bpm_line, = self.ax_bpm.plot([], [], label="BPM Peaks")
        self.bpm_fft_line, = self.ax_bpm.plot([], [], "--", label="BPM FFT")
        self.fft_line, = self.ax_fft.plot([], [])

        self.ax_raw.set_title("Live Raw PPG Signal")
        self.ax_raw.set_ylabel("Raw IR")

        self.ax_filtered.set_title("Live Filtered PPG Signal")
        self.ax_filtered.set_ylabel("Filtered")
        self.ax_filtered.set_facecolor("black")
        self.ax_filtered.tick_params(colors="lime")
        for spine in self.ax_filtered.spines.values():
            spine.set_color("lime")

        self.ax_bpm.set_title("Live BPM Trend")
        self.ax_bpm.set_ylabel("BPM")
        self.ax_bpm.legend()
        self.ax_bpm.set_ylim(40, 120)

        self.ax_fft.set_title("Live FFT Spectrum")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.ax_fft.set_xlim(0, 3)

        self.live_times = deque(maxlen=self.max_samples)
        self.live_bpm = deque(maxlen=self.max_samples)
        self.live_bpm_fft = deque(maxlen=self.max_samples)

        plt.tight_layout()


    def update_plot(self, current_time: float, raw_window: np.ndarray, filtered_window: np.ndarray, bpm: float, bpm_fft: float):
        n = len(raw_window)
        #t = np.linspace(current_time - self.config.window_seconds, current_time, n)
        t_window = np.linspace(
            current_time - n / self.config.sample_rate_hz,
            current_time,
            n
        )

        self.raw_line.set_data(t_window, raw_window)
        self.filtered_line.set_data(t_window, filtered_window)

        self.live_times.append(current_time)
        self.live_bpm.append(bpm if bpm is not None else np.nan)
        self.live_bpm_fft.append(bpm_fft if bpm_fft is not None else np.nan)

        self.bpm_line.set_data(list(self.live_times), list(self.live_bpm))
        self.bpm_fft_line.set_data(list(self.live_times), list(self.live_bpm_fft))

        fft_freqs, fft_mag = self.compute_fft(filtered_window)
        self.fft_line.set_data(fft_freqs, fft_mag)

        for ax in [self.ax_raw, self.ax_filtered, self.ax_bpm, self.ax_fft]:
            ax.relim()
            ax.autoscale_view()

        self.ax_bpm.set_ylim(40, 120)
        self.ax_fft.set_xlim(0, 3.0)

        plt.pause(0.001)

    def run(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        print("Starting Heart Rate Monitor...")
        print("Gently place your finger on the sensor and wait for readings to stabilize.")
        time.sleep(1)

        target_delay = 1.0 / self.config.sample_rate_hz

        try:
            while self.running:
                red_raw, ir_raw = self.reader.read_samples()
                current_time = time.time() - self.start_time
                selected_raw = red_raw if self.config.use_red_channel else ir_raw
                if current_time >= self.config.duration_seconds:
                    print("Monitoring duration reached. Stopping...")
                    self.running = False
                    continue

                '''if self.config.use_red_channel:
                    selected_raw = red_raw
                else:
                    selected_raw = ir_raw'''

                self.reader.red_buffer.append(red_raw)
                self.reader.ir_buffer.append(ir_raw)

                filtered_latest = None
                bpm = None
                hrv_ms = None
                signal_quality = 'Collecting data...'
                bpm_fft = None

                if len(self.reader.red_buffer) == self.reader.red_buffer.maxlen:
                    raw_window = np.array(self.reader.red_buffer) if self.config.use_red_channel else np.array(self.reader.ir_buffer)
                    filtered_window = bandpass_filter(raw_window, self.config.lowcut_hz, self.config.highcut_hz, self.config.sample_rate_hz, order=self.config.filter_order)
                    filtered_window = np.convolve(filtered_window, np.ones(5)/5, mode='same')
                    bpm, peaks, hrv_ms = estimate_bpm(filtered_window, self.config.sample_rate_hz, self.config.min_bpm, self.config.max_bpm)
                    bpm_fft = estimate_bpm_fft(filtered_window, self.config.sample_rate_hz)
                    # Reject bad values
                    if bpm is not None and (bpm < 40 or bpm > 120):
                        bpm = None

                    if bpm_fft is not None and (bpm_fft < 40 or bpm_fft > 120):
                        bpm_fft = None

                    # Smooth
                    if bpm is not None:
                        self.bpm_smooth_buffer.append(bpm)
                        #weights = np.linspace(1, 2, len(self.bpm_smooth_buffer))
                        #bpm = float(np.average(self.bpm_smooth_buffer, weights=weights))
                        bpm = np.median(self.bpm_smooth_buffer)

                    
                    signal_quality = get_signal_quality(raw_window, filtered_window, peaks, hrv_ms)
                    filtered_latest = float(filtered_window[-1]) if len(filtered_window) > 0 else None


                    if bpm_fft is not None and bpm is not None:
                        if abs(bpm - bpm_fft) > 10:
                            signal_quality = 'Inconsistent (FFT mismatch)'

                    

                    #filtered_latest = float(filtered_window[-1]) if len(filtered_window) > 0 else None
                    self.logger.write(current_time, red_raw, ir_raw, selected_raw, filtered_latest, bpm, bpm_fft, hrv_ms, signal_quality)

                    if self.config.enable_live_plot:
                        self.update_plot(current_time, raw_window, filtered_window, bpm, bpm_fft)
                
                if current_time - self.last_print_time >= self.config.print_interval_seconds:
                        #bpm_text = f"{bpm:.2f}" if bpm is not None else "Calculating..."
                        #print(f"Time: {current_time:.2f}s | BPM: {bpm_text} | Signal Quality: {signal_quality}")
                    self.print_dashboard(current_time, selected_raw, filtered_latest, bpm, hrv_ms, signal_quality)
                    self.last_print_time = current_time

                time.sleep(1 / self.config.sample_rate_hz)
        except KeyboardInterrupt:
            print("Shutting down gracefully...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\nShutting down sensor...")
        self.logger.close()
        self.reader.shutdown()

        if self.config.enable_live_plot:
            plt.savefig("final_live_plot.png", dpi=300)
            print("Live plot saved to final_live_plot.png")
            plt.show()

        print(f"Data saved to {self.config.csv_filename}")

if __name__ == "__main__":
    monitor = HeartRateMonitor(CFG)
    monitor.run()