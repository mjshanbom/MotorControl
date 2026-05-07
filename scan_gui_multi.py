#!/usr/bin/env python3
"""GUI front-end for 2D motor grid scan — multi-capture reliable version.

Each scan point collects a minimum number of consistent waveforms before recording.
Trigger source: CHAN2 (pulser sync). Waveform channel: CHAN1 (hydrophone).
Each accepted capture must have v_pp >= consistency_frac * previous accepted capture.
Misfires (below threshold) are discarded without updating the baseline.
The capture with the highest v_pp is written to the CSV.
"""

import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

import numpy as np
import pyvisa
import serial.tools.list_ports


# ── Calibration table (Data.txt: freq_MHz, cal_factor) ───────────────────── #

def _load_cal_table(path=None):
    if path is None:
        if getattr(sys, "frozen", False):
            user_path = os.path.join(os.path.dirname(sys.executable), "Data.txt")
            path = user_path if os.path.exists(user_path) else os.path.join(sys._MEIPASS, "Data.txt")
        else:
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

# Standard Rigol V/div steps (volts)
_VDIV_STEPS = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050,
               0.100, 0.200, 0.500, 1.0, 2.0, 5.0, 10.0]


def _best_vdiv(v_pp, n_divs=8, fill=0.85):
    """Smallest standard V/div that fits v_pp within fill*screen without clipping."""
    target = v_pp / (n_divs * fill)
    for step in _VDIV_STEPS:
        if step >= target:
            return step
    return _VDIV_STEPS[-1]


# ── Scope helpers ─────────────────────────────────────────────────────────── #

def configure_scope(scope, cfg):
    """Apply user-selected scope settings before a scan."""
    scope.write(f":TIM:MODE {cfg['scope_mode']}")
    scope.write(f":TIM:SCAL {cfg['scope_sdiv']}")
    scope.write(f":TIM:OFFS {cfg['scope_delay']}")
    ch = cfg["channel"]
    scope.write(f":{ch}:SCAL {cfg['scope_vdiv']}")
    scope.write(f":{ch}:OFFS 0")
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
    scope.write(":WAV:MODE RAW")
    scope.write(":WAV:FORM BYTE")
    scope.write(":WAV:STAR 1")
    scope.write(":WAV:STOP 2048")
    preamble = scope.query(":WAV:PRE?").strip().split(",")
    if float(preamble[4]) == 0:   # scope mid-acquisition — retry once
        time.sleep(0.15)
        preamble = scope.query(":WAV:PRE?").strip().split(",")
    return {
        "n_points":    2048,
        "x_increment": float(preamble[4]),
        "x_origin":    float(preamble[5]),
        "y_increment": float(preamble[7]),
        "y_origin":    float(preamble[8]),
        "y_reference": float(preamble[9]),
    }


_first_capture = True   # module-level flag for one-time debug log


def capture(scope, pre, stop_event=None):
    """Scope already stopped by caller. Read waveform using cached preamble.
    Returns None if stop_event fires or dt is zero (transitional preamble)."""
    global _first_capture

    dt = pre["x_increment"]
    if dt <= 0:
        scope.write(":RUN")
        return None

    scope.write(":WAV:DATA?")
    for _ in range(20):
        time.sleep(0.01)
        if stop_event is not None and stop_event.is_set():
            scope.write(":RUN")
            return None
    raw_bytes = scope.read_raw()

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

    if _first_capture:
        _first_capture = False
        print(f"[capture debug] raw_len={len(raw)}  n_data={n_data}")
        print(f"[capture debug] raw min={raw.min()}  max={raw.max()}  mean={raw.mean():.1f}")
        print(f"[capture debug] voltage min={voltage.min():.6g}  max={voltage.max():.6g}  mean={voltage.mean():.6g}")

    scope.write(":RUN")

    v_ac = voltage - voltage.mean()

    n = len(v_ac)
    nfft      = n * 16
    fft_mag   = np.abs(np.fft.rfft(v_ac * np.hanning(n), n=nfft))
    fft_freqs = np.fft.rfftfreq(nfft, d=dt)
    mask = fft_freqs > 0
    if mask.any():
        sub_mag   = fft_mag[mask]
        sub_freqs = fft_freqs[mask]
        peak_i    = int(np.argmax(sub_mag))
        if 0 < peak_i < len(sub_mag) - 1:
            alpha, beta, gamma = sub_mag[peak_i-1], sub_mag[peak_i], sub_mag[peak_i+1]
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-12:
                correction = 0.5 * (alpha - gamma) / denom
                fft_freq = sub_freqs[peak_i] + correction * (sub_freqs[1] - sub_freqs[0])
            else:
                fft_freq = sub_freqs[peak_i]
        else:
            fft_freq = sub_freqs[peak_i]
    else:
        fft_freq = 0.0

    threshold   = 0.15 * np.max(np.abs(v_ac))
    half_win    = max(3, int(round(0.75 / (fft_freq * dt)))) if fft_freq > 0 else 50
    raw_crosses = np.where((v_ac[:-1] < 0) & (v_ac[1:] >= 0))[0]
    crosses = [i for i in raw_crosses
               if np.max(v_ac[max(0, i - half_win):i + half_win + 1]) >  threshold
               and np.min(v_ac[max(0, i - half_win):i + half_win + 1]) < -threshold]
    if len(crosses) >= 2:
        refined = []
        for i in crosses:
            frac = -v_ac[i] / (v_ac[i + 1] - v_ac[i])
            refined.append((i + frac) * dt)
        periods = np.diff(refined)
        zc_freq = float(1.0 / np.median(periods))
        if fft_freq > 0 and abs(zc_freq - fft_freq) / fft_freq < 0.20:
            freq = zc_freq
        else:
            freq = fft_freq
    else:
        freq = fft_freq if fft_freq > 0 else 0.0

    freq_mhz = freq / 1e6
    cal_idx    = int(np.argmin(np.abs(_CAL_FREQS - freq_mhz)))
    cal_factor = _CAL_FACTORS[cal_idx]
    RHO_C      = 1.54e6
    pii = float(np.sum((v_ac / (cal_factor * 1e-6))**2) * dt) / (RHO_C * 1e4)

    cumulative = np.cumsum(v_ac ** 2) * dt
    total_energy = cumulative[-1]
    t1 = int(np.searchsorted(cumulative, 0.10 * total_energy)) * dt
    t2 = int(np.searchsorted(cumulative, 0.90 * total_energy)) * dt
    pd = float((t2 - t1) * 1.25)

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


def _arm_and_capture(scope, trig_lev, pre, stop_event=None, timeout=2.0):
    """Arm scope, wait for TD, stop, capture. Returns None on timeout or stop."""
    if trig_lev is not None:
        scope.write(f":TRIG:EDGE:LEV {trig_lev:.4f}")
    scope.write(":TRIG:SWE NORM")
    scope.write(":RUN")
    t0 = time.time()
    triggered = False
    while time.time() - t0 < timeout:
        if stop_event is not None and stop_event.is_set():
            scope.write(":STOP")
            return None
        if scope.query(":TRIG:STAT?").strip() == "TD":
            triggered = True
            break
        time.sleep(0.05)
    scope.write(":STOP")
    if not triggered:
        return None
    return capture(scope, pre, stop_event=stop_event)


# ── Multi-capture ─────────────────────────────────────────────────────────── #

def _multi_capture(scope, trig_lev, pre, min_n, max_attempts,
                   consistency_frac, msg_q, stop_event, timeout=2.0,
                   min_signal_vpp=0.0):
    """Collect min_n consistent waveforms, returning the one with highest v_pp.

    Each accepted capture must have v_pp >= min_signal_vpp (absolute floor) AND
    >= consistency_frac * previous accepted v_pp. The absolute floor prevents a
    first misfire (no B-mode pulse between frames) from collapsing the baseline
    to near-zero, which would let all subsequent misfires pass the 2/3 check.
    Rejected captures do NOT update the baseline.

    If max_attempts is reached with fewer than min_n valid captures but at least
    one, logs a warning and returns the best available. Returns None only if zero
    valid captures were collected."""
    valid    = []
    prev_vpp = None
    rejected = 0

    for _attempt in range(max_attempts):
        if stop_event.is_set():
            return None

        if trig_lev is not None:
            scope.write(f":TRIG:EDGE:LEV {trig_lev:.4f}")
        scope.write(":TRIG:SWE NORM")
        scope.write(":RUN")
        t0 = time.time()
        triggered = False
        while time.time() - t0 < timeout:
            if stop_event.is_set():
                scope.write(":STOP")
                return None
            if scope.query(":TRIG:STAT?").strip() == "TD":
                triggered = True
                break
            time.sleep(0.05)
        scope.write(":STOP")

        if not triggered:
            continue

        stats = capture(scope, pre, stop_event=stop_event)
        if stats is None:
            continue

        vpp = stats["v_pp"]

        # Absolute floor — rejects noise/between-frame captures regardless of baseline.
        if vpp < min_signal_vpp:
            rejected += 1
            continue

        if prev_vpp is not None and vpp < consistency_frac * prev_vpp:
            rejected += 1
            continue   # misfire — baseline unchanged

        valid.append(stats)
        prev_vpp = vpp

        if len(valid) >= min_n:
            break

    if not valid:
        return None

    if len(valid) < min_n:
        msg_q.put({"type": "log", "text":
            f"    Warning: only {len(valid)}/{min_n} captures "
            f"({rejected} rejected) — using best available"})

    return max(valid, key=lambda s: s["v_pp"])


# ── Connection helpers ────────────────────────────────────────────────────── #

def run_online(motor, msg_q):
    try:
        msg_q.put({"type": "log", "text": "Sending online command..."})
        motor.enable_online_mode()
        msg_q.put({"type": "online"})
    except Exception as exc:
        msg_q.put({"type": "log", "text": f"Online error: {exc}"})


def run_connect(cfg, msg_q):
    try:
        msg_q.put({"type": "log", "text": f"Opening motor port {cfg['port']}..."})
        motor = VP9000(cfg["port"], baudrate=cfg["baudrate"], bytesize=cfg["bytesize"],
                       parity=cfg["parity"], stopbits=cfg["stopbits"])
        msg_q.put({"type": "log", "text": "Motor port open. Now put VP9000 in online mode."})

        msg_q.put({"type": "log", "text": f"Connecting to scope at {cfg['visa']}..."})
        try:
            rm = pyvisa.ResourceManager()
            rm.list_resources()
        except Exception:
            rm = pyvisa.ResourceManager("@py")
        scope = rm.open_resource(cfg["visa"])
        scope.timeout = 3000
        scope.chunk_size = 1024 * 1024
        msg_q.put({"type": "log", "text": "Scope connected. Ready to scan."})
        msg_q.put({"type": "connected", "motor": motor, "scope": scope})

    except Exception as exc:
        msg_q.put({"type": "connect_error", "text": str(exc)})


# ── Scan thread ───────────────────────────────────────────────────────────── #

def run_scan(cfg, motor, scope, msg_q, stop_event):
    """Runs in a daemon thread. Posts status dicts to msg_q."""
    global _first_capture
    _first_capture = True
    try:
        motor.set_speed(cfg["x_motor"], cfg["speed"])
        motor.set_speed(cfg["y_motor"], cfg["speed"])
        motor.set_speed(cfg["z_motor"], cfg["speed"])

        def _positions(center, half_range, step):
            if step == 0 or half_range == 0:
                return [center]
            return list(range(center - half_range, center + half_range + step, step))

        xh = cfg["x_range"] // 2
        yh = cfg["y_range"] // 2
        x_positions = _positions(cfg["x_center"], xh, cfg["x_step"])
        y_positions = _positions(cfg["y_center"], yh, cfg["y_step"])

        zs, ze, zst = cfg["z_start"], cfg["z_stop"], cfg["z_step"]
        if zst == 0 or zs == ze:
            z_positions = [zs]
        else:
            direction = 1 if ze >= zs else -1
            z_positions = list(range(zs, ze + direction * zst, direction * zst))
        total = len(x_positions) * len(y_positions) * len(z_positions)

        fieldnames = ["x_mm", "y_mm", "z_mm", "v_pp", "v_max", "v_min",
                      "frequency", "pii", "pd"]

        msg_q.put({"type": "log", "text": "Configuring scope..."})
        configure_scope(scope, cfg)
        pre = setup_scope(scope, cfg["channel"], ac_coupling=cfg["ac_coupling"])
        msg_q.put({"type": "log", "text":
            f"Scope preamble: y_inc={pre['y_increment']:.6g} V/cnt  "
            f"y_origin={pre['y_origin']:.4g} V  y_ref={pre['y_reference']:.4g} cnt  "
            f"dt={pre['x_increment']:.3g} s"})

        msg_q.put({"type": "log", "text": f"Starting scan — {total} points."})

        with open(cfg["output"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            done = 0
            for z in z_positions:
                if stop_event.is_set():
                    break
                if len(z_positions) > 1:
                    msg_q.put({"type": "log", "text": f"Moving Z → {z}..."})
                    motor.move_absolute(cfg["z_motor"], z)
                    motor.wait_for_done(stop_event=stop_event)
                    msg_q.put({"type": "log", "text": f"Z at {z}."})

                if cfg.get("auto_delay_z") and cfg.get("speed_of_sound", 0) > 0:
                    z_m       = z / cfg["z_spmm"] / 1000.0
                    z_start_m = cfg["z_start"] / cfg["z_spmm"] / 1000.0
                    new_delay = cfg["scope_delay"] + (z_m - z_start_m) / cfg["speed_of_sound"]
                    scope.write(f":TIM:OFFS {new_delay:.9f}")
                    msg_q.put({"type": "log", "text":
                        f"  Z delay → {new_delay*1e6:.2f} µs"
                        f"  (Δz={( z_m - z_start_m)*1000:.1f} mm)"})

                for y in y_positions:
                    if stop_event.is_set():
                        break
                    if len(y_positions) > 1:
                        msg_q.put({"type": "log", "text": f"  Moving Y → {y}..."})
                        motor.move_absolute(cfg["y_motor"], y)
                        motor.wait_for_done(stop_event=stop_event)
                        msg_q.put({"type": "log", "text": f"  Y at {y}."})

                    for x in x_positions:
                        if stop_event.is_set():
                            break
                        if len(x_positions) > 1:
                            msg_q.put({"type": "log", "text": f"    Moving X → {x}..."})
                            motor.move_absolute(cfg["x_motor"], x)
                            scope.write(":STOP")
                            motor.wait_for_done(stop_event=stop_event)
                        else:
                            scope.write(":STOP")
                        if stop_event.is_set():
                            break
                        time.sleep(cfg["settle"])

                        trig_lev = cfg["scope_trig_lev"]

                        # ── Pilot capture for autoscale ───────────────────── #
                        if cfg.get("autoscale_vdiv"):
                            pilot = _arm_and_capture(scope, trig_lev, pre,
                                                     stop_event=stop_event, timeout=2.0)
                            if pilot is not None:
                                pilot_vpp = pilot["v_pp"]
                                new_vdiv  = max(_best_vdiv(pilot_vpp), cfg.get("vdiv_floor", 0.050))
                                cur_vdiv  = float(cfg["scope_vdiv"])
                                # Always scale up (prevents clipping).
                                # Only scale down if pilot looks like a real signal
                                # (>= autoscale_min_vpp) — protects against misfires
                                # shrinking V/div and clipping subsequent captures.
                                scale_down_ok = pilot_vpp >= cfg.get("autoscale_min_vpp", 0.10)
                                if new_vdiv != cur_vdiv and (new_vdiv > cur_vdiv or scale_down_ok):
                                    scope.write(f":{cfg['channel']}:SCAL {new_vdiv:.4f}")
                                    time.sleep(0.30)
                                    cfg["scope_vdiv"] = new_vdiv
                                    pre = setup_scope(scope, cfg["channel"],
                                                      ac_coupling=cfg["ac_coupling"])
                                    msg_q.put({"type": "log", "text":
                                        f"    Autoscale: {cur_vdiv*1000:.0f} mV"
                                        f" → {new_vdiv*1000:.0f} mV/div"
                                        f"  (pilot Vpp={pilot_vpp*1000:.0f} mV)"})

                        # ── Multi-capture ─────────────────────────────────── #
                        stats = _multi_capture(
                            scope, trig_lev=trig_lev, pre=pre,
                            min_n=cfg["min_captures"],
                            max_attempts=cfg["max_attempts"],
                            consistency_frac=cfg["consistency_frac"],
                            msg_q=msg_q, stop_event=stop_event,
                            min_signal_vpp=cfg.get("min_signal_vpp", 0.0),
                        )
                        if stats is None:
                            if stop_event.is_set():
                                break
                            msg_q.put({"type": "log", "text":
                                "    No valid captures — skipping point"})
                            continue

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
                if x_positions[-1] != cfg["x_center"]:
                    motor.move_absolute(cfg["x_motor"], cfg["x_center"])
                if y_positions[-1] != cfg["y_center"]:
                    motor.move_absolute(cfg["y_motor"], cfg["y_center"])
                if z_positions[-1] != cfg["z_start"]:
                    motor.move_absolute(cfg["z_motor"], cfg["z_start"])
                motor.wait_for_done(stop_event=stop_event)

        msg_q.put({"type": "done", "output": cfg["output"], "aborted": stop_event.is_set()})

    except Exception as exc:
        msg_q.put({"type": "error", "text": str(exc)})


# ── GUI ──────────────────────────────────────────────────────────────────── #

class ScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VP9000 Multi-Capture Grid Scan")
        self.resizable(False, False)

        self._msg_q    = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread   = None
        self._motor    = None
        self._scope    = None

        self._build_ui()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ──────────────────────────────────────────────────── #

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── Connection ───────────────────────────────────────────────────── #
        conn = ttk.LabelFrame(self, text="Connection")
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self._port = self._row(conn, 0, "Motor port",  "COM3")
        ttk.Button(conn, text="Find", command=self._find_port).grid(
            row=0, column=2, padx=(0, 8), pady=2, sticky="w")
        self._visa = self._row(conn, 1, "Scope VISA",  "USB0::0x1AB1::0x0517::DS1ZE278M01562::INSTR")
        ttk.Button(conn, text="Find", command=self._find_visa).grid(
            row=1, column=2, padx=(0, 8), pady=2, sticky="w")
        self._chan = self._row(conn, 2, "Channel",     "CHAN1")

        self._ac_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(conn, text="AC coupling", variable=self._ac_var).grid(
            row=2, column=2, padx=8, sticky="w")

        ttk.Label(conn, text="Lab").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        self._lab = tk.StringVar(value="Acoustics Lab (200 steps/mm)")
        ttk.Combobox(conn, textvariable=self._lab,
                     values=["Optics Lab (160 steps/mm)", "Acoustics Lab (200 steps/mm)"],
                     state="readonly", width=28).grid(row=3, column=1, sticky="w", padx=6, pady=2)

        rs = ttk.Frame(conn)
        rs.grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=2)
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
        grid = ttk.LabelFrame(self, text="Scan Grid (mm)")
        grid.grid(row=1, column=1, sticky="nsew", **pad)

        ttk.Label(grid, text="").grid(row=0, column=0)
        ttk.Label(grid, text="Center(mm)", width=9).grid(row=0, column=1)
        ttk.Label(grid, text="Range(mm)",  width=9).grid(row=0, column=2)
        ttk.Label(grid, text="Step(mm)",   width=9).grid(row=0, column=3)

        self._xc, self._xr, self._xst = self._axis_row(grid, 1, "X")
        self._yc, self._yr, self._yst = self._axis_row(grid, 2, "Y")

        ttk.Label(grid, text="Z").grid(row=3, column=0, padx=6, pady=2)
        ttk.Label(grid, text="Start(mm)", width=9).grid(row=3, column=1)
        ttk.Label(grid, text="Stop(mm)",  width=9).grid(row=3, column=2)
        ttk.Label(grid, text="Step(mm)",  width=9).grid(row=3, column=3)
        ttk.Label(grid, text="").grid(row=4, column=0)
        self._zstart = tk.StringVar(value="0")
        self._zstop  = tk.StringVar(value="30")
        self._zst    = tk.StringVar(value="0.25")
        ttk.Entry(grid, textvariable=self._zstart, width=8).grid(row=4, column=1, padx=4, pady=2)
        ttk.Entry(grid, textvariable=self._zstop,  width=8).grid(row=4, column=2, padx=4, pady=2)
        ttk.Entry(grid, textvariable=self._zst,    width=8).grid(row=4, column=3, padx=4, pady=2)

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
        self._sc_trig_src = tk.StringVar(value="CHAN2")   # CH2 = pulser sync trigger
        self._sc_trig_lev = tk.StringVar(value="0.5")
        self._sc_holdoff  = tk.StringVar(value="1e-7")
        self._min_cap     = tk.StringVar(value="5")
        self._max_att     = tk.StringVar(value="20")
        self._consist     = tk.StringVar(value="0.67")
        self._autoscale_vdiv  = tk.BooleanVar(value=True)
        self._as_min_vpp      = tk.StringVar(value="0.10")
        self._vdiv_floor      = tk.StringVar(value="0.050")
        self._min_sig_vpp     = tk.StringVar(value="0.02")
        self._auto_delay_z    = tk.BooleanVar(value=True)
        self._sos             = tk.StringVar(value="1500")

        _sc_combo(sc, 0, 0, "Mode",         self._sc_mode,     ["MAIN","XY","ROLL"])
        _sc_entry(sc, 0, 1, "s/div",         self._sc_sdiv)
        _sc_entry(sc, 0, 2, "Delay (s)",     self._sc_delay)
        _sc_entry(sc, 1, 0, "V/div",         self._sc_vdiv)
        _sc_combo(sc, 1, 1, "Acquire type",  self._sc_acq_type, ["NORM","AVER","PEAK","HRES"], width=6)
        _sc_combo(sc, 1, 2, "Mem depth",     self._sc_mdep,
                  ["AUTO","12000","120000","1200000","12000000"], width=10)
        _sc_combo(sc, 2, 0, "Trig source",   self._sc_trig_src,
                  ["CHAN1","CHAN2","CHAN3","CHAN4","EXT","ACL"])
        _sc_entry(sc, 2, 1, "Trig level (V)",self._sc_trig_lev)
        _sc_entry(sc, 2, 2, "Holdoff (s)",   self._sc_holdoff)

        # Multi-capture parameters
        _sc_entry(sc, 3, 0, "Min captures",  self._min_cap,  width=4)
        _sc_entry(sc, 3, 1, "Max attempts",  self._max_att,  width=4)
        _sc_entry(sc, 3, 2, "Consistency",   self._consist,  width=6)

        # Autoscale row
        ttk.Checkbutton(sc, text="Auto V/div", variable=self._autoscale_vdiv).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=(8, 2), pady=3)
        _sc_entry(sc, 4, 1, "Min Vpp to scale down (V)", self._as_min_vpp, width=6)
        _sc_entry(sc, 4, 2, "Min signal Vpp (V)",        self._min_sig_vpp, width=6)
        _sc_entry(sc, 4, 3, "V/div floor (V)",           self._vdiv_floor,  width=6)

        # Z delay row
        ttk.Checkbutton(sc, text="Auto delay with Z", variable=self._auto_delay_z).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=(8, 2), pady=3)
        _sc_entry(sc, 5, 1, "Speed of sound (m/s)", self._sos, width=6)

        self._apply_scope_btn = ttk.Button(sc, text="Apply Now",
                                           command=self._apply_scope_cfg, state="disabled")
        self._apply_scope_btn.grid(row=6, column=0, columnspan=6, pady=(2, 4))

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
        self._online_btn = ttk.Button(btn, text="Online", command=self._go_online, state="disabled")
        self._online_btn.pack(side="left", padx=6)
        self._start_btn = ttk.Button(btn, text="Start Scan", command=self._start, state="disabled")
        self._start_btn.pack(side="left", padx=6)
        self._stop_btn  = ttk.Button(btn, text="Stop", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=6)

    def _find_port(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            tk.messagebox.showinfo("Find Ports", "No serial ports found.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Select Serial Port")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Select a port:").pack(padx=12, pady=(10, 4))
        lb = tk.Listbox(dlg, listvariable=tk.StringVar(value=ports),
                        selectmode="single", width=40, height=min(len(ports), 10))
        lb.pack(padx=12, pady=4)
        lb.select_set(0)
        def _select():
            sel = lb.curselection()
            if sel:
                self._port.set(ports[sel[0]])
            dlg.destroy()
        ttk.Button(dlg, text="Select", command=_select).pack(pady=(4, 10))
        dlg.grab_set()

    def _find_visa(self):
        try:
            try:
                rm = pyvisa.ResourceManager()
                resources = rm.list_resources()
            except Exception:
                rm = pyvisa.ResourceManager("@py")
                resources = rm.list_resources()
            rm.close()
        except Exception as e:
            tk.messagebox.showerror("VISA Error", str(e))
            return
        if not resources:
            tk.messagebox.showinfo("Find Devices", "No VISA devices found.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Select VISA Device")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Select a device:").pack(padx=12, pady=(10, 4))
        lb = tk.Listbox(dlg, listvariable=tk.StringVar(value=resources),
                        selectmode="single", width=60, height=min(len(resources), 10))
        lb.pack(padx=12, pady=4)
        lb.select_set(0)
        def _select():
            sel = lb.curselection()
            if sel:
                self._visa.set(resources[sel[0]])
            dlg.destroy()
        ttk.Button(dlg, text="Select", command=_select).pack(pady=(4, 10))
        dlg.grab_set()

    def _row(self, parent, r, label, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=44).grid(row=r, column=1, padx=6, pady=2)
        return var

    def _axis_row(self, parent, r, axis):
        ttk.Label(parent, text=axis).grid(row=r, column=0, padx=6, pady=2)
        defaults = {
            "X": ("0", "4",   "0.1"),
            "Y": ("0", "0",   "3"),
        }
        vs = []
        for c, val in enumerate(defaults[axis], start=1):
            v = tk.StringVar(value=val)
            ttk.Entry(parent, textvariable=v, width=8).grid(row=r, column=c, padx=4, pady=2)
            vs.append(v)
        return vs

    def _browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._out_var.set(path)

    # ── Connection control ───────────────────────────────────────────────── #

    @staticmethod
    def _f(var):
        return float(var.get().replace(",", "."))

    def _collect_cfg(self):
        lab_spmm = {"Optics Lab (160 steps/mm)": 160, "Acoustics Lab (200 steps/mm)": 200}
        spmm = lab_spmm[self._lab.get()]
        f = self._f
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
            "settle":   f(self._set),
            "x_spmm":   spmm,
            "y_spmm":   spmm,
            "z_spmm":   spmm,
            "x_center": int(round(f(self._xc)  * spmm)),
            "x_range":  int(round(f(self._xr)  * spmm)),
            "x_step":   int(round(f(self._xst) * spmm)),
            "y_center": int(round(f(self._yc)  * spmm)),
            "y_range":  int(round(f(self._yr)  * spmm)),
            "y_step":   int(round(f(self._yst) * spmm)),
            "z_start":  int(round(f(self._zstart) * spmm)),
            "z_stop":   int(round(f(self._zstop)  * spmm)),
            "z_step":   int(round(f(self._zst)    * spmm)),
            "scope_mode":     self._sc_mode.get(),
            "scope_sdiv":     f(self._sc_sdiv),
            "scope_delay":    f(self._sc_delay),
            "scope_vdiv":     f(self._sc_vdiv),
            "scope_acq_type": self._sc_acq_type.get(),
            "scope_mdep":     self._sc_mdep.get(),
            "scope_trig_src": self._sc_trig_src.get(),
            "scope_trig_lev": f(self._sc_trig_lev),
            "scope_holdoff":  f(self._sc_holdoff),
            "min_captures":      int(self._min_cap.get()),
            "max_attempts":      int(self._max_att.get()),
            "consistency_frac":  f(self._consist),
            "autoscale_vdiv":    self._autoscale_vdiv.get(),
            "autoscale_min_vpp": f(self._as_min_vpp),
            "vdiv_floor":        f(self._vdiv_floor),
            "min_signal_vpp":    f(self._min_sig_vpp),
            "auto_delay_z":      self._auto_delay_z.get(),
            "speed_of_sound":    f(self._sos),
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

    def _go_online(self):
        self._online_btn.config(state="disabled")
        threading.Thread(target=run_online, args=(self._motor, self._msg_q), daemon=True).start()

    def _disconnect(self):
        if self._motor:
            try:
                self._motor.send("K")
            except Exception:
                pass
        if self._scope:
            try:
                self._scope.close()
            except Exception:
                pass
            self._scope = None
        self._connect_btn.config(state="normal")
        self._disconnect_btn.config(state="disabled")
        self._online_btn.config(state="disabled")
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
                    self._online_btn.config(state="normal")
                    self._apply_scope_btn.config(state="normal")
                    self._status_var.set("Connected. Click Online to enable the controller.")

                elif t == "online":
                    self._online_btn.config(state="disabled")
                    self._start_btn.config(state="normal")
                    self._status_var.set("Controller online. Ready to scan.")
                    self._log_line("Controller online. Ready.")

                elif t == "connect_error":
                    self._log_line(f"Connect error: {msg['text']}")
                    self._status_var.set("Connection failed.")
                    self._connect_btn.config(state="normal")

                elif t == "progress":
                    pct = msg["done"] / msg["total"] * 100 if msg["total"] > 0 else 0
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
                    self._online_btn.config(state="normal")

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
