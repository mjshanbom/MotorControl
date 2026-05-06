"""
Rigol DS1000Z — Stable single-channel scope GUI with correct PII / PD.
Rewritten for deterministic waveform capture (matches known-good processing).
"""

import datetime
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pyvisa

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ─────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────

def _load_cal_table(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.txt")
    freqs, factors = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                freqs.append(float(p[0]))
                factors.append(float(p[1]))
    return np.array(freqs), np.array(factors)

_CAL_FREQS, _CAL_FACTORS = _load_cal_table()


# ─────────────────────────────────────────────────────────────
# Signal processing (UNCHANGED CORE)
# ─────────────────────────────────────────────────────────────

RHO_C = 1.54e6

def process_waveform(voltage, dt):
    v_ac = voltage - voltage.mean()

    n = len(v_ac)
    nfft = n * 16
    fft_mag = np.abs(np.fft.rfft(v_ac * np.hanning(n), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=dt)

    mask = freqs > 0
    if mask.any():
        sub_mag = fft_mag[mask]
        sub_freqs = freqs[mask]
        peak_i = int(np.argmax(sub_mag))
        fft_freq = sub_freqs[peak_i]
    else:
        fft_freq = 0

    freq = fft_freq
    freq_mhz = freq / 1e6

    cal_idx = int(np.argmin(np.abs(_CAL_FREQS - freq_mhz)))
    cal = _CAL_FACTORS[cal_idx]

    pii = float(np.sum((v_ac / (cal * 1e-6))**2) * dt) / (RHO_C * 1e4)

    cumulative = np.cumsum(v_ac**2) * dt
    total = cumulative[-1]
    t1 = np.searchsorted(cumulative, 0.1 * total) * dt
    t2 = np.searchsorted(cumulative, 0.9 * total) * dt
    pd = (t2 - t1) * 1.25

    return {
        "freq": freq,
        "pii": pii,
        "pd": pd,
        "v_pp": float(voltage.max() - voltage.min()),
        "v_rms": float(np.sqrt(np.mean(v_ac**2))),
    }


# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────

class ScopeGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Scope — Stable Capture")
        self.geometry("1000x700")

        self._scope = None
        self._rm = None
        self._q = queue.Queue()

        self._build_ui()
        self._poll()

    # ─────────────────────────────────────────────────────────

    def _build_ui(self):

        top = tk.Frame(self)
        top.pack(fill=tk.X)

        self._visa = tk.StringVar(value="USB0::6833::1303::DS1ZE278M01562::0::INSTR")

        tk.Entry(top, textvariable=self._visa, width=50).pack(side=tk.LEFT)

        tk.Button(top, text="Connect", command=self._connect).pack(side=tk.LEFT)
        tk.Button(top, text="Capture", command=self._capture).pack(side=tk.LEFT)

        self._status = tk.StringVar(value="Idle")
        tk.Label(self, textvariable=self._status).pack()

        self._fig = Figure(figsize=(6,4))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────────────────────────────────────

    def _connect(self):
        self._status.set("Connecting...")
        threading.Thread(target=self._thread_connect, daemon=True).start()

    def _thread_connect(self):
        try:
            self._rm = pyvisa.ResourceManager()
            self._scope = self._rm.open_resource(self._visa.get())
            self._scope.timeout = 5000
            self._q.put(("status", "Connected"))
        except Exception as e:
            self._q.put(("status", str(e)))

    # ─────────────────────────────────────────────────────────

    def _capture(self):
        threading.Thread(target=self._thread_capture, daemon=True).start()

   # ONLY showing the part that changed — everything else remains identical

def _thread_capture(self):
    try:
        ch = self._chan.get()
        coup = "AC" if self._ac_var.get() else "DC"

        # --- Force stable acquisition ---
        self._scope.write(f":{ch}:COUP {coup}")
        self._scope.write(":STOP")
        time.sleep(0.2)

        self._scope.write(f":WAV:SOUR {ch}")
        self._scope.write(":WAV:MODE RAW")
        self._scope.write(":WAV:FORM BYTE")
        self._scope.write(":WAV:STAR 1")
        self._scope.write(":WAV:STOP 2048")

        # --- Read preamble ---
        pre = self._scope.query(":WAV:PRE?").strip().split(",")

        dt        = float(pre[4])
        x_origin  = float(pre[5])
        y_inc     = float(pre[7])
        y_origin  = float(pre[8])
        y_ref     = float(pre[9])

        # --- Request data ---
        self._scope.write(":WAV:DATA?")
        raw = self._scope.read_raw()

        # Ensure we got enough bytes to parse header
        while len(raw) < 100:
            raw += self._scope.read_raw()

        # Parse binary block header
        n_digits = int(chr(raw[1]))
        n_bytes  = int(raw[2:2+n_digits])
        start    = 2 + n_digits

        # Ensure full payload received
        while len(raw) < start + n_bytes:
            raw += self._scope.read_raw()

        data = raw[start:start+n_bytes]

        # --- Convert to voltage ---
        samples = np.frombuffer(data, dtype=np.uint8)
        voltage = (samples - y_ref) * y_inc + y_origin
        t       = x_origin + np.arange(len(voltage)) * dt

        # Resume acquisition
        self._scope.write(":RUN")

        # --- Process (UNCHANGED CORE) ---
        results = process_waveform(voltage, dt)

        # --- Return to GUI ---
        self._q.put({
            "type": "capture",
            "t": t,
            "v": voltage,
            "results": results,
            "dt": dt,
            "x_origin": x_origin
        })

    except Exception as exc:
        self._q.put({"type": "error", "text": str(exc)})
        self._q.put({"type": "capture_done"})
    # ─────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()

                if msg[0] == "status":
                    self._status.set(msg[1])

                elif msg[0] == "plot":
                    t, v = msg[1], msg[2]
                    self._ax.clear()
                    self._ax.plot(t, v)
                    self._canvas.draw()

        except queue.Empty:
            pass

        self.after(100, self._poll)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ScopeGUI().mainloop()