#!/usr/bin/env python3
"""
trigger_sweep.py — diagnostic tool for B-mode trigger analysis.

Sweeps the oscilloscope trigger level from high to low and captures a waveform
at each level that fires.  Saves two CSVs:

  trigger_sweep_summary.csv   — one row per level: fired?, v_pp, frequency
  trigger_sweep_waveforms.csv — all raw waveform samples for fired levels

Feed these back to Claude so the signal landscape can be analysed.
"""

import csv
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from pathlib import Path

import numpy as np
import pyvisa


# ── Scope helpers ─────────────────────────────────────────────────────────── #

def _get_preamble(scope, channel):
    scope.write(f":WAV:SOUR CHAN{channel}")
    scope.write(":WAV:MODE RAW")
    scope.write(":WAV:FORM BYTE")
    scope.write(":WAV:STAR 1")
    scope.write(":WAV:STOP 2048")
    return {
        "x_increment": float(scope.query(":WAV:XINC?")),
        "x_origin":    float(scope.query(":WAV:XOR?")),
        "x_reference": float(scope.query(":WAV:XREF?")),
        "y_increment": float(scope.query(":WAV:YINC?")),
        "y_origin":    float(scope.query(":WAV:YOR?")),
        "y_reference": float(scope.query(":WAV:YREF?")),
    }


def _read_waveform(scope, pre):
    """Read raw bytes from a stopped scope; return voltage array."""
    scope.write(":WAV:DATA?")
    time.sleep(0.20)
    raw_bytes = scope.read_raw()

    n_digits   = int(chr(raw_bytes[1]))
    n_data     = int(raw_bytes[2:2 + n_digits])
    data_start = 2 + n_digits
    while len(raw_bytes) < data_start + n_data:
        raw_bytes += scope.read_raw()

    raw = np.frombuffer(raw_bytes[data_start:data_start + n_data], dtype=np.uint8)
    voltage = (raw - pre["y_reference"]) * pre["y_increment"] + pre["y_origin"]
    return voltage


def _compute_freq(voltage, dt):
    """FFT with zero-padding + parabolic interpolation."""
    n    = len(voltage)
    nfft = n * 16
    v_ac = voltage - voltage.mean()
    mag  = np.abs(np.fft.rfft(v_ac * np.hanning(n), n=nfft))
    freq = np.fft.rfftfreq(nfft, d=dt)
    mask = freq > 0
    if not mask.any():
        return 0.0
    sub_mag, sub_freq = mag[mask], freq[mask]
    pk = int(np.argmax(sub_mag))
    if 0 < pk < len(sub_mag) - 1:
        a, b, g = sub_mag[pk - 1], sub_mag[pk], sub_mag[pk + 1]
        corr = 0.5 * (a - g) / (a - 2 * b + g)
        return float(sub_freq[pk] + corr * (sub_freq[1] - sub_freq[0]))
    return float(sub_freq[pk])


def _wait_td(scope, wait_s, poll=0.05):
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if scope.query(":TRIG:STAT?").strip() == "TD":
            return True
        time.sleep(poll)
    return False


# ── Sweep worker ──────────────────────────────────────────────────────────── #

def _run_sweep(visa, channel, trig_src, v_high, v_low, v_step,
               wait_s, out_dir, msg_q, stop_evt):
    rm    = pyvisa.ResourceManager()
    scope = rm.open_resource(visa)
    scope.timeout = 15_000

    try:
        # One-time scope setup
        scope.write(f":TRIG:EDGE:SOUR CHAN{trig_src}")
        scope.write(":TRIG:EDGE:SLOP POS")
        scope.write(":TRIG:SWE NORM")
        scope.write(f":WAV:SOUR CHAN{channel}")
        scope.write(":WAV:MODE RAW")
        scope.write(":WAV:FORM BYTE")
        scope.write(":WAV:STAR 1")
        scope.write(":WAV:STOP 2048")

        # Build level list  high → low
        levels = []
        v = v_high
        while v >= v_low - 1e-9:
            levels.append(round(v, 6))
            v = round(v - v_step, 6)

        msg_q.put(f"Sweeping {len(levels)} levels: "
                  f"{v_high:.4f} V → {v_low:.4f} V, step {v_step:.4f} V")

        out_dir = Path(out_dir)
        sum_path  = out_dir / "trigger_sweep_summary.csv"
        wave_path = out_dir / "trigger_sweep_waveforms.csv"

        sum_fields  = ["trigger_level_V", "fired", "v_pp_V", "v_max_V",
                       "v_min_V", "frequency_MHz"]
        wave_fields = ["trigger_level_V", "sample_index", "time_us", "voltage_V"]

        with open(sum_path, "w", newline="") as sf, \
             open(wave_path, "w", newline="") as wf:

            sw = csv.DictWriter(sf, fieldnames=sum_fields)
            ww = csv.DictWriter(wf, fieldnames=wave_fields)
            sw.writeheader()
            ww.writeheader()

            for level in levels:
                if stop_evt.is_set():
                    break

                scope.write(f":TRIG:EDGE:LEV {level:.6f}")
                scope.write(":RUN")
                time.sleep(0.02)

                fired = _wait_td(scope, wait_s)

                if not fired:
                    sw.writerow({"trigger_level_V": level, "fired": 0,
                                 "v_pp_V": "", "v_max_V": "", "v_min_V": "",
                                 "frequency_MHz": ""})
                    sf.flush()
                    msg_q.put(f"  {level:+.4f} V  —  no trigger")
                    continue

                # Scope fired: stop and capture
                scope.write(":STOP")
                time.sleep(0.05)

                try:
                    pre     = _get_preamble(scope, channel)
                    voltage = _read_waveform(scope, pre)
                    dt      = pre["x_increment"]
                    freq    = _compute_freq(voltage, dt)
                    v_pp    = float(voltage.max() - voltage.min())
                    v_max   = float(voltage.max())
                    v_min   = float(voltage.min())

                    sw.writerow({"trigger_level_V": level, "fired": 1,
                                 "v_pp_V": f"{v_pp:.6f}",
                                 "v_max_V": f"{v_max:.6f}",
                                 "v_min_V": f"{v_min:.6f}",
                                 "frequency_MHz": f"{freq/1e6:.4f}"})
                    sf.flush()

                    for i, (vi, ti) in enumerate(
                            zip(voltage, np.arange(len(voltage)) * dt * 1e6)):
                        ww.writerow({"trigger_level_V": level,
                                     "sample_index": i,
                                     "time_us": f"{ti:.4f}",
                                     "voltage_V": f"{vi:.6f}"})
                    wf.flush()

                    msg_q.put(f"  {level:+.4f} V  →  FIRED  "
                              f"Vpp={v_pp:.4f} V  f={freq/1e6:.3f} MHz")

                except Exception as e:
                    msg_q.put(f"  {level:+.4f} V  →  capture error: {e}")

        msg_q.put(f"\nDone.\n  Summary  : {sum_path}\n  Waveforms: {wave_path}")

    except Exception as e:
        import traceback
        msg_q.put(f"ERROR: {e}\n{traceback.format_exc()}")
    finally:
        try:
            scope.write(":RUN")
        except Exception:
            pass
        scope.close()
        rm.close()
        msg_q.put("__DONE__")


# ── GUI ───────────────────────────────────────────────────────────────────── #

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trigger Level Sweep — Diagnostic")
        self.resizable(True, True)
        self._stop_evt = threading.Event()
        self._msg_q    = queue.Queue()
        self._build_ui()
        self._poll()

    def _build_ui(self):
        pad = dict(padx=6, pady=3)

        # ── Connection ──────────────────────────────────────────────────────
        f0 = ttk.LabelFrame(self, text="Scope")
        f0.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(f0, text="VISA address:").grid(row=0, column=0, sticky="e", **pad)
        self._visa = tk.StringVar(value="USB0::0x1AB1::0x0517::DS1ZE278M01562::INSTR")
        ttk.Entry(f0, textvariable=self._visa, width=52).grid(
            row=0, column=1, columnspan=3, sticky="ew", **pad)

        ttk.Label(f0, text="Measure ch:").grid(row=1, column=0, sticky="e", **pad)
        self._chan = tk.StringVar(value="1")
        ttk.Combobox(f0, textvariable=self._chan,
                     values=["1", "2", "3", "4"], width=4).grid(
            row=1, column=1, sticky="w", **pad)

        ttk.Label(f0, text="Trigger ch:").grid(row=1, column=2, sticky="e", **pad)
        self._tsrc = tk.StringVar(value="1")
        ttk.Combobox(f0, textvariable=self._tsrc,
                     values=["1", "2", "3", "4"], width=4).grid(
            row=1, column=3, sticky="w", **pad)

        # ── Sweep parameters ────────────────────────────────────────────────
        f1 = ttk.LabelFrame(self, text="Sweep parameters")
        f1.grid(row=1, column=0, sticky="ew", **pad)

        def _lbl_entry(parent, row, col, label, var, width=10):
            ttk.Label(parent, text=label).grid(row=row, column=col,     sticky="e", **pad)
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=col + 1, sticky="w", **pad)

        self._high = tk.StringVar(value="1.0")
        self._low  = tk.StringVar(value="0.02")
        self._step = tk.StringVar(value="0.02")
        self._wait = tk.StringVar(value="0.5")
        _lbl_entry(f1, 0, 0, "High (V):",       self._high)
        _lbl_entry(f1, 1, 0, "Low  (V):",        self._low)
        _lbl_entry(f1, 2, 0, "Step (V):",        self._step)
        _lbl_entry(f1, 3, 0, "Wait/level (s):",  self._wait)

        # ── Output directory ────────────────────────────────────────────────
        f2 = ttk.LabelFrame(self, text="Output directory")
        f2.grid(row=2, column=0, sticky="ew", **pad)
        self._outdir = tk.StringVar(value=str(Path.home() / "Desktop"))
        ttk.Entry(f2, textvariable=self._outdir, width=52).grid(
            row=0, column=0, sticky="ew", **pad)
        ttk.Button(f2, text="Browse…", command=self._browse).grid(
            row=0, column=1, **pad)

        # ── Buttons ─────────────────────────────────────────────────────────
        bf = ttk.Frame(self)
        bf.grid(row=3, column=0, **pad)
        ttk.Button(bf, text="Run Sweep", command=self._run).pack(side="left", padx=4)
        ttk.Button(bf, text="Stop",      command=self._stop).pack(side="left", padx=4)

        # ── Log ─────────────────────────────────────────────────────────────
        self._log = scrolledtext.ScrolledText(
            self, width=80, height=22, state="disabled")
        self._log.grid(row=4, column=0, sticky="nsew", **pad)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

    def _browse(self):
        d = filedialog.askdirectory()
        if d:
            self._outdir.set(d)

    def _log_line(self, text):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _run(self):
        self._stop_evt.clear()
        try:
            args = (
                self._visa.get(),
                self._chan.get(),
                self._tsrc.get(),
                float(self._high.get()),
                float(self._low.get()),
                float(self._step.get()),
                float(self._wait.get()),
                self._outdir.get(),
                self._msg_q,
                self._stop_evt,
            )
        except ValueError as e:
            self._log_line(f"Config error: {e}")
            return
        threading.Thread(target=_run_sweep, args=args, daemon=True).start()

    def _stop(self):
        self._stop_evt.set()

    def _poll(self):
        try:
            while True:
                msg = self._msg_q.get_nowait()
                if msg != "__DONE__":
                    self._log_line(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
