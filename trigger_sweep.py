#!/usr/bin/env python3
"""
trigger_sweep.py — trigger-level sweep diagnostic and final capture tool.

Sweeps the oscilloscope trigger from high to low using the same raw-capture
pipeline as scan_gui.py (preamble caching, chunked IEEE 488.2 reads, proper
TD polling).  For each fired level the full waveform is captured and analysed
(v_pp, v_max, v_min, frequency, Vpp).

Outputs:
  trigger_sweep_summary.csv   — one row per level: fired?, v_pp, freq, …
  trigger_sweep_waveforms.csv — raw voltage samples for every fired level

A "Final Capture" button captures a single waveform at the currently chosen
trigger level and saves it as trigger_final_capture.csv.
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


# ── Scope helpers (mirrors scan_gui.py) ──────────────────────────────────── #

def setup_scope(scope, channel):
    """Configure WAV parameters and return a cached preamble dict."""
    scope.write(f":WAV:SOUR {channel}")
    scope.write(":WAV:MODE RAW")
    scope.write(":WAV:FORM BYTE")
    scope.write(":WAV:STAR 1")
    scope.write(":WAV:STOP 2048")
    parts = scope.query(":WAV:PRE?").strip().split(",")
    return {
        "x_increment": float(parts[4]),
        "x_origin":    float(parts[5]),
        "y_increment": float(parts[7]),
        "y_origin":    float(parts[8]),
        "y_reference": float(parts[9]),
    }


def capture(scope, pre):
    """Scope must already be stopped. Read waveform, return stats dict or None."""
    scope.write(":WAV:DATA?")
    time.sleep(0.20)
    raw_bytes = scope.read_raw()

    # IEEE 488.2 definite-length block: #N<N-digit byte count><data>
    n_digits   = int(chr(raw_bytes[1]))
    n_data     = int(raw_bytes[2:2 + n_digits])
    data_start = 2 + n_digits
    while len(raw_bytes) < data_start + n_data:
        raw_bytes += scope.read_raw()

    raw = np.frombuffer(raw_bytes[data_start:data_start + n_data], dtype=np.uint8)
    if len(raw) == 0:
        scope.write(":RUN")
        return None

    voltage = (raw - pre["y_reference"]) * pre["y_increment"] + pre["y_origin"]
    scope.write(":RUN")

    dt   = pre["x_increment"]
    v_ac = voltage - voltage.mean()

    # FFT frequency with zero-padding and parabolic interpolation
    n      = len(v_ac)
    nfft   = n * 16
    mag    = np.abs(np.fft.rfft(v_ac * np.hanning(n), n=nfft))
    freqs  = np.fft.rfftfreq(nfft, d=dt)
    mask   = freqs > 0
    if mask.any():
        sub_mag, sub_freq = mag[mask], freqs[mask]
        pk = int(np.argmax(sub_mag))
        if 0 < pk < len(sub_mag) - 1:
            a, b, g = sub_mag[pk-1], sub_mag[pk], sub_mag[pk+1]
            denom = a - 2*b + g
            corr  = 0.5*(a - g)/denom if abs(denom) > 1e-12 else 0.0
            freq  = sub_freq[pk] + corr*(sub_freq[1] - sub_freq[0])
        else:
            freq = sub_freq[pk]
    else:
        freq = 0.0

    return {
        "voltage":   voltage,
        "dt":        dt,
        "v_pp":      float(voltage.max() - voltage.min()),
        "v_max":     float(voltage.max()),
        "v_min":     float(voltage.min()),
        "v_rms":     float(np.sqrt(np.mean(v_ac**2))),
        "frequency": freq,
    }


def arm_and_capture(scope, trig_lev, pre, wait_s=2.0):
    """Arm scope at trig_lev, poll for TD, stop, capture. Returns stats or None."""
    scope.write(f":TRIG:EDGE:LEV {trig_lev:.6f}")
    scope.write(":TRIG:SWE NORM")
    scope.write(":RUN")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if scope.query(":TRIG:STAT?").strip() == "TD":
            break
        time.sleep(0.05)
    else:
        return None   # timed out — do not read stale memory
    scope.write(":STOP")
    time.sleep(0.05)
    return capture(scope, pre)


# ── Sweep worker ──────────────────────────────────────────────────────────── #

def _run_sweep(visa, channel, trig_src, v_high, v_low, v_step,
               wait_s, out_dir, msg_q, stop_evt):
    rm    = pyvisa.ResourceManager()
    scope = rm.open_resource(visa)
    scope.timeout = 15_000

    try:
        scope.write(f":TRIG:EDGE:SOUR {trig_src}")
        scope.write(":TRIG:EDGE:SLOP POS")
        scope.write(":TRIG:SWE NORM")

        pre = setup_scope(scope, channel)
        msg_q.put(f"Preamble: dt={pre['x_increment']:.3g}s  "
                  f"y_inc={pre['y_increment']:.3g} V/cnt")

        levels = []
        v = v_high
        while v >= v_low - 1e-9:
            levels.append(round(v, 6))
            v = round(v - v_step, 6)

        msg_q.put(f"Sweeping {len(levels)} levels: "
                  f"{v_high:.4f} V → {v_low:.4f} V, step {v_step:.4f} V")

        out_dir   = Path(out_dir)
        sum_path  = out_dir / "trigger_sweep_summary.csv"
        wave_path = out_dir / "trigger_sweep_waveforms.csv"

        sum_fields  = ["trigger_level_V", "fired",
                       "v_pp_V", "v_max_V", "v_min_V", "v_rms_V", "frequency_MHz"]
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

                stats = arm_and_capture(scope, level, pre, wait_s=wait_s)

                if stats is None:
                    sw.writerow({"trigger_level_V": level, "fired": 0,
                                 "v_pp_V": "", "v_max_V": "", "v_min_V": "",
                                 "v_rms_V": "", "frequency_MHz": ""})
                    sf.flush()
                    msg_q.put(f"  {level:+.4f} V  —  no trigger")
                    continue

                sw.writerow({
                    "trigger_level_V": level,
                    "fired":           1,
                    "v_pp_V":          f"{stats['v_pp']:.6f}",
                    "v_max_V":         f"{stats['v_max']:.6f}",
                    "v_min_V":         f"{stats['v_min']:.6f}",
                    "v_rms_V":         f"{stats['v_rms']:.6f}",
                    "frequency_MHz":   f"{stats['frequency']/1e6:.4f}",
                })
                sf.flush()

                dt      = stats["dt"]
                voltage = stats["voltage"]
                for i, vi in enumerate(voltage):
                    ww.writerow({
                        "trigger_level_V": level,
                        "sample_index":    i,
                        "time_us":         f"{i * dt * 1e6:.4f}",
                        "voltage_V":       f"{vi:.6f}",
                    })
                wf.flush()

                msg_q.put(f"  {level:+.4f} V  →  FIRED  "
                          f"Vpp={stats['v_pp']:.4f} V  "
                          f"f={stats['frequency']/1e6:.3f} MHz")

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


def _run_final_capture(visa, channel, trig_src, trig_lev, wait_s, out_dir, msg_q):
    rm    = pyvisa.ResourceManager()
    scope = rm.open_resource(visa)
    scope.timeout = 15_000

    try:
        scope.write(f":TRIG:EDGE:SOUR {trig_src}")
        scope.write(":TRIG:EDGE:SLOP POS")

        pre   = setup_scope(scope, channel)
        stats = arm_and_capture(scope, trig_lev, pre, wait_s=wait_s)

        if stats is None:
            msg_q.put(f"Final capture: no trigger at {trig_lev:.4f} V within {wait_s}s")
            msg_q.put("__DONE__")
            return

        out_path = Path(out_dir) / "trigger_final_capture.csv"
        dt       = stats["dt"]
        voltage  = stats["voltage"]
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_index", "time_us", "voltage_V"])
            for i, vi in enumerate(voltage):
                w.writerow([i, f"{i * dt * 1e6:.4f}", f"{vi:.6f}"])

        msg_q.put(
            f"Final capture: Vpp={stats['v_pp']:.4f} V  "
            f"f={stats['frequency']/1e6:.3f} MHz\n"
            f"  Saved: {out_path}"
        )

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
        self.title("Trigger Level Sweep")
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
        self._chan = tk.StringVar(value="CHAN1")
        ttk.Combobox(f0, textvariable=self._chan,
                     values=["CHAN1","CHAN2","CHAN3","CHAN4"], width=6).grid(
            row=1, column=1, sticky="w", **pad)

        ttk.Label(f0, text="Trigger ch:").grid(row=1, column=2, sticky="e", **pad)
        self._tsrc = tk.StringVar(value="CHAN1")
        ttk.Combobox(f0, textvariable=self._tsrc,
                     values=["CHAN1","CHAN2","CHAN3","CHAN4"], width=6).grid(
            row=1, column=3, sticky="w", **pad)

        # ── Sweep parameters ────────────────────────────────────────────────
        f1 = ttk.LabelFrame(self, text="Sweep parameters")
        f1.grid(row=1, column=0, sticky="ew", **pad)

        def _le(parent, row, col, label, var, width=10):
            ttk.Label(parent, text=label).grid(row=row, column=col,   sticky="e", **pad)
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=col+1, sticky="w", **pad)

        self._high = tk.StringVar(value="1.0")
        self._low  = tk.StringVar(value="0.02")
        self._step = tk.StringVar(value="0.02")
        self._wait = tk.StringVar(value="0.5")
        _le(f1, 0, 0, "High (V):",      self._high)
        _le(f1, 1, 0, "Low  (V):",      self._low)
        _le(f1, 2, 0, "Step (V):",      self._step)
        _le(f1, 3, 0, "Wait/level (s):", self._wait)

        # ── Final capture trigger level ──────────────────────────────────────
        f1b = ttk.LabelFrame(self, text="Final capture")
        f1b.grid(row=2, column=0, sticky="ew", **pad)
        self._final_trig = tk.StringVar(value="0.30")
        _le(f1b, 0, 0, "Trigger (V):", self._final_trig)

        # ── Output directory ────────────────────────────────────────────────
        f2 = ttk.LabelFrame(self, text="Output directory")
        f2.grid(row=3, column=0, sticky="ew", **pad)
        self._outdir = tk.StringVar(value=str(Path.home() / "Desktop"))
        ttk.Entry(f2, textvariable=self._outdir, width=52).grid(
            row=0, column=0, sticky="ew", **pad)
        ttk.Button(f2, text="Browse…", command=self._browse).grid(
            row=0, column=1, **pad)

        # ── Buttons ─────────────────────────────────────────────────────────
        bf = ttk.Frame(self)
        bf.grid(row=4, column=0, **pad)
        ttk.Button(bf, text="Run Sweep",      command=self._run).pack(side="left", padx=4)
        ttk.Button(bf, text="Final Capture",  command=self._final).pack(side="left", padx=4)
        ttk.Button(bf, text="Stop",           command=self._stop).pack(side="left", padx=4)

        # ── Log ─────────────────────────────────────────────────────────────
        self._log = scrolledtext.ScrolledText(
            self, width=80, height=24, state="disabled")
        self._log.grid(row=5, column=0, sticky="nsew", **pad)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

    def _browse(self):
        d = filedialog.askdirectory()
        if d:
            self._outdir.set(d)

    def _log_line(self, text):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _parse(self):
        def _f(v): return float(v.get().replace(",", "."))
        return (
            self._visa.get(),
            self._chan.get(),
            self._tsrc.get(),
            _f(self._high),
            _f(self._low),
            _f(self._step),
            _f(self._wait),
            self._outdir.get(),
        )

    def _run(self):
        self._stop_evt.clear()
        try:
            visa, chan, tsrc, high, low, step, wait, outdir = self._parse()
        except ValueError as e:
            self._log_line(f"Config error: {e}")
            return
        threading.Thread(
            target=_run_sweep,
            args=(visa, chan, tsrc, high, low, step, wait, outdir,
                  self._msg_q, self._stop_evt),
            daemon=True,
        ).start()

    def _final(self):
        self._stop_evt.clear()
        try:
            visa, chan, tsrc, _, _, _, wait, outdir = self._parse()
            trig = float(self._final_trig.get().replace(",", "."))
        except ValueError as e:
            self._log_line(f"Config error: {e}")
            return
        self._log_line(f"Final capture at {trig:.4f} V…")
        threading.Thread(
            target=_run_final_capture,
            args=(visa, chan, tsrc, trig, wait, outdir, self._msg_q),
            daemon=True,
        ).start()

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
