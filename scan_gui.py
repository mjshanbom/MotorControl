#!/usr/bin/env python3
"""GUI front-end for the 2D motor grid scan."""

import csv
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

import numpy as np
import pyvisa


# ── Calibration table (Data.txt: freq_MHz, cal_factor) ───────────────────── #

def _load_cal_table(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.txt")
    freqs, factors = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                freqs.append(float(parts[0]))
                factors.append(float(parts[1]))
    return np.array(freqs), np.array(factors)

_CAL_FREQS, _CAL_FACTORS = _load_cal_table()

from velmex_vp9000 import VP9000


# ── Scan logic (runs in background thread) ───────────────────────────────── #

def configure_scope(scope, cfg):
    """Apply user-selected scope settings before a scan."""
    scope.write(f":TIM:MODE {cfg['scope_mode']}")
    scope.write(f":TIM:SCAL {cfg['scope_sdiv']}")
    scope.write(f":TIM:OFFS {cfg['scope_delay']}")
    ch = cfg["channel"]
    scope.write(f":{ch}:SCAL {cfg['scope_vdiv']}")
    scope.write(f":ACQ:TYPE {cfg['scope_acq_type']}")
    scope.write(f":ACQ:MDEP {cfg['scope_mdep']}")
    scope.write(":TRIG:MODE EDGE")
    scope.write(f":TRIG:EDGE:SOUR {cfg['scope_trig_src']}")
    scope.write(f":TRIG:EDGE:LEV {cfg['scope_trig_lev']}")
    scope.write(f":TRIG:HOLD {cfg['scope_holdoff']}")


def setup_scope(scope, channel, ac_coupling=False):
    """Configure WAV parameters once and return cached preamble constants."""
    if ac_coupling:
        scope.write(f":{channel}:COUP AC")
    else:
        scope.write(f":{channel}:COUP DC")
    scope.write(f":WAV:SOUR {channel}")
    scope.write(":WAV:MODE NORM")
    scope.write(":WAV:FORM BYTE")
    scope.write(":WAV:STAR 1")
    scope.write(":WAV:STOP 1020")
    preamble    = scope.query(":WAV:PRE?").strip().split(",")
    return {
        "n_points":    int(preamble[2]),
        "x_increment": float(preamble[4]),
        "x_origin":    float(preamble[5]),
        "y_increment": float(preamble[7]),
        "y_origin":    float(preamble[8]),
        "y_reference": float(preamble[9]),
    }


_first_capture = True   # module-level flag for one-time debug log


def capture(scope, pre, stop_event=None):
    """Scope already stopped by caller. Read waveform using cached preamble.
    Returns None if stop_event fires during the wait."""
    global _first_capture

    scope.write(":WAV:DATA?")
    # 0.2 s wait in small steps so stop_event can interrupt
    for _ in range(20):
        time.sleep(0.01)
        if stop_event is not None and stop_event.is_set():
            scope.write(":RUN")
            return None
    raw_bytes = scope.read_raw()         # first chunk (header + data, or just header)

    # parse IEEE 488.2 definite-length block header: #N<N digits of byte count><data>
    n_digits   = int(chr(raw_bytes[1]))
    n_data     = int(raw_bytes[2:2 + n_digits])
    data_start = 2 + n_digits

    # accumulate additional chunks if the first read was short
    while len(raw_bytes) < data_start + n_data:
        raw_bytes += scope.read_raw()

    raw = np.frombuffer(raw_bytes[data_start:data_start + n_data], dtype=np.uint8)
    voltage = (raw - pre["y_reference"]) * pre["y_increment"] + pre["y_origin"]

    if _first_capture:
        _first_capture = False
        print(f"[capture debug] raw_len={len(raw)}  n_data={n_data}")
        print(f"[capture debug] raw min={raw.min()}  max={raw.max()}  mean={raw.mean():.1f}")
        print(f"[capture debug] voltage min={voltage.min():.6g}  max={voltage.max():.6g}  mean={voltage.mean():.6g}")

    scope.write(":RUN")

    dt = pre["x_increment"]
    v_ac = voltage - voltage.mean()   # remove DC for RMS / PII, matching scope behaviour

    # Frequency: zero-crossing counter.
    # Only count crossings where the signal nearby exceeds 15% of peak,
    # ignoring noise crossings outside the burst region.
    # Falls back to FFT if fewer than 2 crossings are found.
    threshold = 0.15 * np.max(np.abs(v_ac))
    raw_crosses = np.where((v_ac[:-1] < 0) & (v_ac[1:] >= 0))[0]
    crosses = [i for i in raw_crosses
               if np.max(np.abs(v_ac[max(0, i - 5):i + 6])) > threshold]
    if len(crosses) >= 2:
        # linear interpolation for sub-sample accuracy on each crossing
        refined = []
        for i in crosses:
            frac = -v_ac[i] / (v_ac[i + 1] - v_ac[i])
            refined.append((i + frac) * dt)
        periods = np.diff(refined)
        freq = float(1.0 / np.mean(periods))
    else:
        nfft    = 131072
        window  = np.hanning(len(v_ac))
        fft_mag = np.abs(np.fft.rfft(v_ac * window, n=nfft))
        k = int(np.argmax(fft_mag[1:]) + 1)
        if 1 < k < len(fft_mag) - 1:
            a, b, c = fft_mag[k - 1], fft_mag[k], fft_mag[k + 1]
            denom   = a - 2 * b + c
            k_frac  = k + (a - c) / (2 * denom) if denom != 0 else k
        else:
            k_frac = k
        freq = float(k_frac / (nfft * dt))

    freq_mhz = freq / 1e6
    cal_idx    = int(np.argmin(np.abs(_CAL_FREQS - freq_mhz)))
    cal_factor = _CAL_FACTORS[cal_idx]
    RHO_C      = 1.5e6  # acoustic impedance of water (Pa·s/m)
    pii = float(np.sum((v_ac / (cal_factor * 1e-6))**2) * dt) / (RHO_C * 1e4)  # J/cm²
    pd  = float(len(voltage) * dt)              # s     (capture window duration)

    return {
        "n_samples":  len(voltage),
        "timebase_s": dt,
        "x_origin_s": pre["x_origin"],
        "v_min":      float(voltage.min()),
        "v_max":      float(voltage.max()),
        "v_pp":       float(voltage.max() - voltage.min()),
        "v_mean":     float(voltage.mean()),
        "v_rms":      float(np.sqrt(np.mean(v_ac**2))),
        "frequency":  freq,
        "pii":        pii,
        "pd":         pd,
    }


def run_connect(cfg, msg_q):
    """Open serial port + scope. No commands sent to the VP9000.
    After this returns, put the VP9000 in online mode manually, then Start Scan."""
    try:
        msg_q.put({"type": "log", "text": f"Opening motor port {cfg['port']}..."})
        motor = VP9000(cfg["port"], baudrate=cfg["baudrate"], bytesize=cfg["bytesize"],
                       parity=cfg["parity"], stopbits=cfg["stopbits"])
        msg_q.put({"type": "log", "text": "Motor port open. Now put VP9000 in online mode."})

        msg_q.put({"type": "log", "text": f"Connecting to scope at {cfg['visa']}..."})
        rm = pyvisa.ResourceManager("@py")
        scope = rm.open_resource(cfg["visa"])
        scope.timeout = 3000
        scope.chunk_size = 1024 * 1024
        msg_q.put({"type": "log", "text": "Scope connected. Ready to scan."})

        msg_q.put({"type": "connected", "motor": motor, "scope": scope})

    except Exception as exc:
        msg_q.put({"type": "connect_error", "text": str(exc)})


def run_scan(cfg, motor, scope, msg_q, stop_event):
    """Runs in a daemon thread. Posts status dicts to msg_q."""
    global _first_capture
    _first_capture = True
    try:
        motor.set_speed(cfg["x_motor"], cfg["speed"])
        motor.set_speed(cfg["y_motor"], cfg["speed"])
        motor.set_speed(cfg["z_motor"], cfg["speed"])

        xh = cfg["x_range"] // 2
        yh = cfg["y_range"] // 2
        zh = cfg["z_range"] // 2
        x_positions = list(range(cfg["x_center"] - xh, cfg["x_center"] + xh + cfg["x_step"], cfg["x_step"]))
        y_positions = list(range(cfg["y_center"] - yh, cfg["y_center"] + yh + cfg["y_step"], cfg["y_step"]))
        z_positions = list(range(cfg["z_center"] - zh, cfg["z_center"] + zh + cfg["z_step"], cfg["z_step"]))
        total = len(x_positions) * len(y_positions) * len(z_positions)

        fieldnames = ["x_mm", "y_mm", "z_mm", "v_pp", "v_max", "v_min",
                      "frequency", "pii", "pd"]

        msg_q.put({"type": "log", "text": "Configuring scope..."})
        configure_scope(scope, cfg)
        pre = setup_scope(scope, cfg["channel"], ac_coupling=cfg["ac_coupling"])
        msg_q.put({"type": "log", "text":
            f"Scope preamble: y_inc={pre['y_increment']:.6g} V/cnt  "
            f"y_origin={pre['y_origin']:.4g} V  y_ref={pre['y_reference']:.4g} cnt"})

        msg_q.put({"type": "log", "text": f"Starting scan — {total} points."})

        with open(cfg["output"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            done = 0
            for z in z_positions:
                if stop_event.is_set():
                    break
                msg_q.put({"type": "log", "text": f"Moving Z → {z}..."})
                motor.move_absolute(cfg["z_motor"], z)
                motor.wait_for_done(stop_event=stop_event)
                msg_q.put({"type": "log", "text": f"Z at {z}."})

                for y in y_positions:
                    if stop_event.is_set():
                        break
                    msg_q.put({"type": "log", "text": f"  Moving Y → {y}..."})
                    motor.move_absolute(cfg["y_motor"], y)
                    motor.wait_for_done(stop_event=stop_event)
                    msg_q.put({"type": "log", "text": f"  Y at {y}."})

                    for x in x_positions:
                        if stop_event.is_set():
                            break
                        msg_q.put({"type": "log", "text": f"    Moving X → {x}..."})
                        motor.move_absolute(cfg["x_motor"], x)
                        scope.write(":STOP")  # Stop scope while motor is moving
                        motor.wait_for_done(stop_event=stop_event)
                        if stop_event.is_set():
                            break
                        time.sleep(cfg["settle"])

                        stats = capture(scope, pre, stop_event=stop_event)
                        if stats is None:
                            break
                        row = {
                            "x_mm":      x / cfg["x_spmm"],
                            "y_mm":      y / cfg["y_spmm"],
                            "z_mm":      z / cfg["z_spmm"],
                            "v_pp":      stats["v_pp"],
                            "v_max":     stats["v_max"],
                            "v_min":     stats["v_min"],
                            "frequency": stats["frequency"],
                            "pii":       stats["pii"],
                            "pd":        stats["pd"],
                        }
                        writer.writerow(row)
                        f.flush()

                        done += 1
                        msg_q.put({
                            "type":      "progress",
                            "done":      done,
                            "total":     total,
                            "x":         x,
                            "y":         y,
                            "z":         z,
                            "v_pp":      stats["v_pp"],
                            "v_rms":     stats["v_rms"],
                            "v_mean":    stats["v_mean"],
                            "frequency": stats["frequency"],
                            "pii":       stats["pii"],
                        })

            if stop_event.is_set():
                try:
                    scope.write(":RUN")
                except Exception:
                    pass
            else:
                motor.move_absolute(cfg["x_motor"], cfg["x_center"])
                motor.move_absolute(cfg["y_motor"], cfg["y_center"])
                motor.move_absolute(cfg["z_motor"], cfg["z_center"])
                motor.wait_for_done(stop_event=stop_event)

        msg_q.put({"type": "done", "output": cfg["output"], "aborted": stop_event.is_set()})

    except Exception as exc:
        msg_q.put({"type": "error", "text": str(exc)})


# ── GUI ──────────────────────────────────────────────────────────────────── #

class ScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VP9000 2D Grid Scan")
        self.resizable(False, False)

        self._msg_q     = queue.Queue()
        self._stop_evt  = threading.Event()
        self._thread    = None
        self._motor     = None
        self._scope     = None

        self._build_ui()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ──────────────────────────────────────────────────── #

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── Connection ───────────────────────────────────────────────────── #
        conn = ttk.LabelFrame(self, text="Connection")
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self._port = self._row(conn, 0, "Motor port",  "/dev/tty.usbserial-FTEHI1FO")
        self._visa = self._row(conn, 1, "Scope VISA",  "USB0::6833::1303::DS1ZE278M01562::0::INSTR")
        self._chan = self._row(conn, 2, "Channel",     "CHAN1")

        self._ac_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(conn, text="AC coupling", variable=self._ac_var).grid(
            row=2, column=2, padx=8, sticky="w")

        # RS-232 settings sub-row
        rs = ttk.Frame(conn)
        rs.grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        ttk.Label(rs, text="Baud").pack(side="left")
        self._baud = tk.StringVar(value="9600")
        ttk.Combobox(rs, textvariable=self._baud, values=["9600","19200","38400","57600","115200"],
                     width=8, state="readonly").pack(side="left", padx=(2,10))
        ttk.Label(rs, text="Data bits").pack(side="left")
        self._bits = tk.StringVar(value="7")
        ttk.Combobox(rs, textvariable=self._bits, values=["7","8"],
                     width=3, state="readonly").pack(side="left", padx=(2,10))
        ttk.Label(rs, text="Parity").pack(side="left")
        self._parity = tk.StringVar(value="E")
        ttk.Combobox(rs, textvariable=self._parity, values=["N","E","O"],
                     width=3, state="readonly").pack(side="left", padx=(2,10))
        ttk.Label(rs, text="Stop bits").pack(side="left")
        self._stop = tk.StringVar(value="2")
        ttk.Combobox(rs, textvariable=self._stop, values=["1","2"],
                     width=3, state="readonly").pack(side="left", padx=(2,0))

        # ── Motors ───────────────────────────────────────────────────────── #
        mot = ttk.LabelFrame(self, text="Motors")
        mot.grid(row=1, column=0, sticky="nsew", **pad)

        self._xm  = self._row(mot, 0, "X motor #", "1")
        self._ym  = self._row(mot, 1, "Y motor #", "2")
        self._zm  = self._row(mot, 2, "Z motor #", "3")
        self._spd = self._row(mot, 3, "Speed (steps/s)", "1000")
        self._set = self._row(mot, 4, "Settle (s)", "0.05")

        # ── Grid ─────────────────────────────────────────────────────────── #
        grid = ttk.LabelFrame(self, text="Scan Grid (steps)")
        grid.grid(row=1, column=1, sticky="nsew", **pad)

        ttk.Label(grid, text="").grid(row=0, column=0)
        ttk.Label(grid, text="Center",   width=8).grid(row=0, column=1)
        ttk.Label(grid, text="Range",    width=8).grid(row=0, column=2)
        ttk.Label(grid, text="Step",     width=8).grid(row=0, column=3)
        ttk.Label(grid, text="Steps/mm", width=8).grid(row=0, column=4)

        self._xc, self._xr, self._xst, self._x_spmm = self._axis_row(grid, 1, "X")
        self._yc, self._yr, self._yst, self._y_spmm = self._axis_row(grid, 2, "Y")
        self._zc, self._zr, self._zst, self._z_spmm = self._axis_row(grid, 3, "Z")

        # ── Scope Config ─────────────────────────────────────────────────── #
        sc = ttk.LabelFrame(self, text="Scope Config")
        sc.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        def _sc_combo(parent, row, col, label, var, values, width=9):
            ttk.Label(parent, text=label).grid(row=row, column=col*2,   sticky="w", padx=(8,2), pady=3)
            ttk.Combobox(parent, textvariable=var, values=values, width=width,
                         state="readonly").grid(row=row, column=col*2+1, sticky="w", padx=(0,8), pady=3)

        def _sc_entry(parent, row, col, label, var, width=9):
            ttk.Label(parent, text=label).grid(row=row, column=col*2,   sticky="w", padx=(8,2), pady=3)
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=col*2+1, sticky="w", padx=(0,8), pady=3)

        self._sc_mode     = tk.StringVar(value="MAIN")
        self._sc_sdiv     = tk.StringVar(value="0.001")
        self._sc_delay    = tk.StringVar(value="0.0")
        self._sc_vdiv     = tk.StringVar(value="1.0")
        self._sc_acq_type = tk.StringVar(value="NORM")
        self._sc_mdep     = tk.StringVar(value="AUTO")
        self._sc_trig_src = tk.StringVar(value="CHAN1")
        self._sc_trig_lev = tk.StringVar(value="0.0")
        self._sc_holdoff  = tk.StringVar(value="1e-7")

        _sc_combo(sc, 0, 0, "Mode",         self._sc_mode,     ["MAIN","XY","ROLL"])
        _sc_entry(sc, 0, 1, "s/div",         self._sc_sdiv)
        _sc_entry(sc, 0, 2, "Delay (s)",     self._sc_delay)
        _sc_entry(sc, 1, 0, "V/div",         self._sc_vdiv)
        _sc_combo(sc, 1, 1, "Acquire type",  self._sc_acq_type, ["NORM","AVER","PEAK","HRES"], width=6)
        _sc_combo(sc, 1, 2, "Mem depth",     self._sc_mdep,
                  ["AUTO","12000","120000","1200000","12000000"], width=10)
        _sc_combo(sc, 2, 0, "Trig source",   self._sc_trig_src, ["CHAN1","CHAN2","CHAN3","CHAN4","EXT","ACL"])
        _sc_entry(sc, 2, 1, "Trig level (V)",self._sc_trig_lev)
        _sc_entry(sc, 2, 2, "Holdoff (s)",   self._sc_holdoff)

        self._apply_scope_btn = ttk.Button(sc, text="Apply Now",
                                           command=self._apply_scope_cfg, state="disabled")
        self._apply_scope_btn.grid(row=3, column=0, columnspan=6, pady=(2, 4))

        # ── Output ───────────────────────────────────────────────────────── #
        out = ttk.LabelFrame(self, text="Output")
        out.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        self._out_var = tk.StringVar(value="scan_results.csv")
        ttk.Entry(out, textvariable=self._out_var, width=50).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(out, text="Browse…", command=self._browse).grid(row=0, column=1, padx=4)

        # ── Progress ─────────────────────────────────────────────────────── #
        self._progress = ttk.Progressbar(self, mode="determinate")
        self._progress.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 0))

        self._status_var = tk.StringVar(value="Not connected.")
        ttk.Label(self, textvariable=self._status_var).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=10)

        # ── Log ──────────────────────────────────────────────────────────── #
        self._log = scrolledtext.ScrolledText(self, height=10, width=72, state="disabled",
                                              font=("Menlo", 11))
        self._log.grid(row=6, column=0, columnspan=2, padx=8, pady=4)

        # ── Buttons ──────────────────────────────────────────────────────── #
        btn = ttk.Frame(self)
        btn.grid(row=7, column=0, columnspan=2, pady=(0, 8))

        self._connect_btn    = ttk.Button(btn, text="Connect", command=self._connect)
        self._connect_btn.pack(side="left", padx=6)
        self._disconnect_btn = ttk.Button(btn, text="Disconnect", command=self._disconnect, state="disabled")
        self._disconnect_btn.pack(side="left", padx=6)
        self._start_btn = ttk.Button(btn, text="Start Scan", command=self._start, state="disabled")
        self._start_btn.pack(side="left", padx=6)
        self._stop_btn  = ttk.Button(btn, text="Stop", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=6)

    def _row(self, parent, r, label, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=44).grid(row=r, column=1, padx=6, pady=2)
        return var

    def _axis_row(self, parent, r, axis):
        ttk.Label(parent, text=axis).grid(row=r, column=0, padx=6, pady=2)
        defaults = {
            "X": ("0", "5000", "500", "160"),
            "Y": ("0", "5000", "500", "160"),
            "Z": ("0", "0",    "1",   "160"),
        }
        vs = []
        for c, val in enumerate(defaults[axis], start=1):
            v = tk.StringVar(value=val)
            ttk.Entry(parent, textvariable=v, width=8).grid(row=r, column=c, padx=4, pady=2)
            vs.append(v)
        return vs

    def _axis_positions(self, center, half_range, step):
        """Return integer step positions centered on center."""
        half = half_range // 2
        start = center - half
        stop  = center + half
        return list(range(start, stop + step, step))

    def _browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._out_var.set(path)

    # ── Connection control ───────────────────────────────────────────────── #

    def _collect_cfg(self):
        return {
            "port":     self._port.get(),
            "visa":     self._visa.get(),
            "channel":  self._chan.get(),
            "output":   self._out_var.get(),
            "baudrate": int(self._baud.get()),
            "bytesize": int(self._bits.get()),
            "parity":   self._parity.get(),
            "stopbits":    int(self._stop.get()),
            "ac_coupling": self._ac_var.get(),
            "x_motor":  int(self._xm.get()),
            "y_motor":  int(self._ym.get()),
            "z_motor":  int(self._zm.get()),
            "speed":    int(self._spd.get()),
            "settle":   float(self._set.get()),
            "x_center": int(self._xc.get()),
            "x_range":  int(self._xr.get()),
            "x_step":   int(self._xst.get()),
            "y_center": int(self._yc.get()),
            "y_range":  int(self._yr.get()),
            "y_step":   int(self._yst.get()),
            "z_center": int(self._zc.get()),
            "z_range":  int(self._zr.get()),
            "z_step":   int(self._zst.get()),
            "x_spmm":   float(self._x_spmm.get()),
            "y_spmm":   float(self._y_spmm.get()),
            "z_spmm":   float(self._z_spmm.get()),
            # scope config
            "scope_mode":     self._sc_mode.get(),
            "scope_sdiv":     self._sc_sdiv.get(),
            "scope_delay":    self._sc_delay.get(),
            "scope_vdiv":     self._sc_vdiv.get(),
            "scope_acq_type": self._sc_acq_type.get(),
            "scope_mdep":     self._sc_mdep.get(),
            "scope_trig_src": self._sc_trig_src.get(),
            "scope_trig_lev": self._sc_trig_lev.get(),
            "scope_holdoff":  self._sc_holdoff.get(),
        }

    def _connect(self):
        try:
            cfg = self._collect_cfg()
        except ValueError as e:
            self._log_line(f"Config error: {e}")
            return

        self._connect_btn.config(state="disabled")
        self._status_var.set("Connecting…")
        threading.Thread(target=run_connect, args=(cfg, self._msg_q), daemon=True).start()

    def _disconnect(self):
        if self._motor:
            try:
                self._motor.send("K")
            except Exception:
                pass
            # Keep serial port open — reopening it would cause another DTR glitch.
        if self._scope:
            try:
                self._scope.close()
            except Exception:
                pass
            self._scope = None
        self._connect_btn.config(state="normal")
        self._disconnect_btn.config(state="disabled")
        self._start_btn.config(state="disabled")
        self._apply_scope_btn.config(state="disabled")
        self._status_var.set("Disconnected.")
        self._log_line("Disconnected.")

    def _on_close(self):
        if self._motor:
            try:
                self._motor.close()
            except Exception:
                pass
        self._disconnect()
        self.destroy()

    # ── Scope config ─────────────────────────────────────────────────────── #

    def _apply_scope_cfg(self):
        if self._scope is None:
            return
        try:
            cfg = self._collect_cfg()
        except ValueError as e:
            self._log_line(f"Config error: {e}")
            return

        def _send():
            try:
                configure_scope(self._scope, cfg)
                self._msg_q.put({"type": "log", "text": "Scope config applied."})
            except Exception as exc:
                self._msg_q.put({"type": "log", "text": f"Scope config error: {exc}"})

        threading.Thread(target=_send, daemon=True).start()

    # ── Scan control ─────────────────────────────────────────────────────── #

    def _start(self):
        try:
            cfg = self._collect_cfg()
        except ValueError as e:
            self._log_line(f"Config error: {e}")
            return

        self._stop_evt.clear()
        self._progress["value"] = 0
        self._status_var.set("Running…")
        self._start_btn.config(state="disabled")
        self._disconnect_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        self._thread = threading.Thread(
            target=run_scan,
            args=(cfg, self._motor, self._scope, self._msg_q, self._stop_evt),
            daemon=True)
        self._thread.start()

    def _stop(self):
        self._stop_evt.set()
        self._status_var.set("Stopping…")
        self._stop_btn.config(state="disabled")

    # ── Message polling ──────────────────────────────────────────────────── #

    def _poll(self):
        try:
            while True:
                msg = self._msg_q.get_nowait()
                t = msg["type"]

                if t == "log":
                    self._log_line(msg["text"])

                elif t == "connected":
                    self._motor = msg["motor"]
                    self._scope = msg["scope"]
                    self._connect_btn.config(state="disabled")
                    self._disconnect_btn.config(state="normal")
                    self._start_btn.config(state="normal")
                    self._apply_scope_btn.config(state="normal")
                    self._status_var.set("Connected. Put VP9000 online, then Start Scan.")

                elif t == "connect_error":
                    self._log_line(f"Connect error: {msg['text']}")
                    self._status_var.set("Connection failed.")
                    self._connect_btn.config(state="normal")

                elif t == "progress":
                    pct = msg["done"] / msg["total"] * 100
                    self._progress["value"] = pct
                    self._status_var.set(
                        f"[{msg['done']}/{msg['total']}]  "
                        f"x={msg['x']}  y={msg['y']}  z={msg['z']}  "
                        f"Vpp={msg['v_pp']:.4f} V"
                    )
                    self._log_line(
                        f"[{msg['done']:>4}/{msg['total']}] "
                        f"x={msg['x']:>6}  y={msg['y']:>6}  z={msg['z']:>6}  "
                        f"Vpp={msg['v_pp']:.4f} V  "
                        f"f={msg['frequency']:.1f} Hz  PII={msg['pii']:.3e} V²s"
                    )

                elif t == "done":
                    label = "Aborted." if msg["aborted"] else f"Done. Saved to {msg['output']}"
                    self._log_line(label)
                    self._status_var.set(label)
                    self._progress["value"] = 0 if msg["aborted"] else 100
                    self._start_btn.config(state="normal")
                    self._disconnect_btn.config(state="normal")
                    self._stop_btn.config(state="disabled")

                elif t == "error":
                    self._log_line(f"ERROR: {msg['text']}")
                    self._status_var.set("Error — see log.")
                    self._start_btn.config(state="normal")
                    self._disconnect_btn.config(state="normal")
                    self._stop_btn.config(state="disabled")

        except queue.Empty:
            pass

        self.after(100, self._poll)

    def _log_line(self, text):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")


if __name__ == "__main__":
    ScanApp().mainloop()
