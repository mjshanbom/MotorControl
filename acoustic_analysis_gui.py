#!/usr/bin/env python3
"""Acoustic parameter calculator.

Loads scan CSV files produced by scan_gui / scan_gui_multi and computes:
  Isppa, ISPTA, Power, p+, p-, MI, Fc, PD, X/Y beam widths, Peak centre.

Input files all share the scan CSV format:
  x_mm, y_mm, z_mm, v_pp, v_max, v_min, frequency, pii, pd

Calibration: Data.txt (freq_MHz  sensitivity_mV_per_MPa) alongside this script.
"""

import csv
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np


# ── Calibration ───────────────────────────────────────────────────────────── #

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


def _sensitivity_v_per_pa(freq_hz):
    """Hydrophone sensitivity in V/Pa at freq_hz (linearly interpolated from Data.txt).
    Data.txt columns: freq_MHz  sensitivity_V_per_MPa
    Conversion V/MPa → V/Pa: multiply by 1e-6.
    """
    f = freq_hz / 1e6
    return float(np.interp(f, _CAL_FREQS, _CAL_FACTORS)) * 1e-6


# ── Data loading ──────────────────────────────────────────────────────────── #

def load_scan_csv(path):
    """Return list of float-valued row dicts from a scan CSV."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (ValueError, TypeError):
                pass
    return rows


def load_waveform_txt(path):
    """Load a scope-GUI waveform .txt file (# header + dt + t_start + samples).
    Returns a single-element list in scan CSV row format for use as the
    Waveform input.  v_max / v_min are derived from the raw voltage samples.
    """
    meta = {}
    numerics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                if ':' in content:
                    key, _, rest = content.partition(':')
                    tokens = rest.strip().split()
                    if tokens:
                        meta[key.strip().lower()] = tokens[0]
            else:
                try:
                    numerics.append(float(line))
                except ValueError:
                    pass

    # First two numerics are dt and t_start; the rest are voltage samples.
    samples = numerics[2:] if len(numerics) > 2 else numerics

    def _f(key, default=0.0):
        try:
            return float(meta.get(key, default))
        except (ValueError, TypeError):
            return default

    v_pp      = _f('vpp')
    frequency = _f('freq')
    pii       = _f('pii')
    pd_s      = _f('pd')
    v_max     = max(samples) if samples else  v_pp / 2
    v_min     = min(samples) if samples else -v_pp / 2

    return [{
        "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0,
        "v_pp":  v_pp,  "v_max": v_max,  "v_min": v_min,
        "frequency": frequency, "pii": pii, "pd": pd_s,
    }]


def load_any(path):
    """Auto-detect file format and load it."""
    with open(path) as f:
        first = next((l for l in f if l.strip()), "")
    if first.startswith('#'):
        return load_waveform_txt(path)
    return load_scan_csv(path)


# ── Beam width ────────────────────────────────────────────────────────────── #

def beam_width_1d(positions, amplitudes, threshold_db):
    """
    Full beam width at threshold_db (e.g. -6) relative to peak amplitude,
    using linear interpolation between samples.
    Returns width in the same units as positions, or None if crossings not found.
    """
    pos = np.array(positions, dtype=float)
    amp = np.array(amplitudes, dtype=float)
    order = np.argsort(pos)
    pos, amp = pos[order], amp[order]

    peak_i = int(np.argmax(amp))
    thresh = amp[peak_i] * 10 ** (threshold_db / 20.0)

    def _cross(i, j):
        dy = amp[j] - amp[i]
        return None if abs(dy) < 1e-15 else pos[i] + (thresh - amp[i]) / dy * (pos[j] - pos[i])

    left_x = None
    for i in range(peak_i - 1, -1, -1):
        if amp[i] <= thresh:
            left_x = _cross(i, i + 1)
            break

    right_x = None
    for i in range(peak_i, len(amp) - 1):
        if amp[i + 1] <= thresh:
            right_x = _cross(i, i + 1)
            break

    if left_x is not None and right_x is not None:
        return right_x - left_x
    return None


# ── Metric computation ────────────────────────────────────────────────────── #

def compute_metrics(waveform_rows, xy_rows, x_rows, y_rows, prf_hz, bw_db,
                    scan_mode="scanned", aprt_area_cm2=None, z_sp_cm=None,
                    lines_per_frame=1):
    """
    Return ordered list of (parameter_label, value_str, unit_str) tuples.
    Any input list may be empty; missing inputs produce 'N/A' entries.
    scan_mode: "scanned" (B-mode) or "unscanned" (M-mode / Doppler).
    aprt_area_cm2: transducer aperture area in cm² (required for TIS/TIB).
    z_sp_cm: focal / spatial-peak depth in cm from transducer face (for derating).
    lines_per_frame: number of scan lines per frame (B-mode); effective PRF at
        the spatial peak = prf_hz / lines_per_frame.  Use 1 for unscanned modes.
    """
    NA = "N/A"
    out = []
    fc_mhz       = None
    W_mW         = None
    I_SPTA_mWcm2 = None
    p_pos_mpa    = None
    p_neg_mpa    = None
    isppa_wcm2   = None

    # ── Reference row (spatial peak for waveform-derived metrics) ──────── #
    if waveform_rows:
        peak = max(waveform_rows, key=lambda r: r.get("pii", 0.0))
    elif xy_rows:
        peak = max(xy_rows, key=lambda r: r.get("pii", 0.0))
    else:
        peak = None

    # ── Frequency & pulse duration ──────────────────────────────────────── #
    if peak is not None:
        freq_hz = peak.get("frequency", 0.0)
        fc_mhz  = freq_hz / 1e6
        out.append(("Fc",             f"{fc_mhz:.3f}",           "MHz"))
        out.append(("Pulse Duration",  f"{peak.get('pd',0)*1e6:.2f}", "µs"))
    else:
        out.append(("Fc",            NA, "MHz"))
        out.append(("Pulse Duration", NA, "µs"))

    # ── Pressure & MI ───────────────────────────────────────────────────── #
    if peak is not None and peak.get("frequency", 0) > 0:
        sens    = _sensitivity_v_per_pa(peak["frequency"])   # V/Pa
        p_pos   = peak.get("v_max", 0.0) / sens              # Pa
        p_neg   = abs(peak.get("v_min", 0.0)) / sens         # Pa
        p_pos_mpa = p_pos / 1e6
        p_neg_mpa = p_neg / 1e6
        fc_mhz  = peak["frequency"] / 1e6
        mi      = p_neg_mpa / np.sqrt(fc_mhz) if fc_mhz > 0 else 0.0
        out.append(("p+ (free field)",  f"{p_pos_mpa:.4f}", "MPa"))
        out.append(("p- (free field)",  f"{p_neg_mpa:.4f}", "MPa"))
        out.append(("MI (free field)",  f"{mi:.3f}",        ""))
    else:
        out.append(("p+ (free field)", NA, "MPa"))
        out.append(("p- (free field)", NA, "MPa"))
        out.append(("MI (free field)", NA, ""))

    # Effective PRF at the spatial peak (accounts for B-mode scan geometry).
    eff_prf = prf_hz / max(1, lines_per_frame)

    # ── ISPPA & ISPTA ───────────────────────────────────────────────────── #
    if peak is not None:
        pii_peak = peak.get("pii", 0.0)
        pd_s     = peak.get("pd",  0.0)

        if pd_s > 0:
            isppa = pii_peak / pd_s
            isppa_wcm2 = isppa
            out.append(("ISPPA (free field)", f"{isppa:.3f}", "W/cm²"))
        else:
            out.append(("ISPPA (free field)", NA + " (PD=0)", "W/cm²"))

        if prf_hz > 0:
            ispta = pii_peak * eff_prf
            I_SPTA_mWcm2 = ispta * 1000
            out.append(("ISPTA (free field)", f"{ispta * 1000:.3f}", "mW/cm²"))
        else:
            out.append(("ISPTA (free field)", NA + " (PRF=0)", "mW/cm²"))
    else:
        out.append(("ISPPA (free field)", NA, "W/cm²"))
        out.append(("ISPTA (free field)", NA, "mW/cm²"))

    # ── Total power from 2D scan ────────────────────────────────────────── #
    if xy_rows and eff_prf > 0:
        x_set = sorted(set(round(r["x_mm"], 6) for r in xy_rows))
        y_set = sorted(set(round(r["y_mm"], 6) for r in xy_rows))
        if len(x_set) >= 2 and len(y_set) >= 2:
            dx_cm = abs(x_set[1] - x_set[0]) / 10.0
            dy_cm = abs(y_set[1] - y_set[0]) / 10.0
            area  = dx_cm * dy_cm   # cm²
            power_w = eff_prf * sum(r.get("pii", 0.0) * area for r in xy_rows)
            W_mW = power_w * 1000
            out.append(("Total Power", f"{W_mW:.3f}", "mW"))
        else:
            out.append(("Total Power", NA + " (need 2-D grid)", "mW"))
    elif not xy_rows:
        out.append(("Total Power", NA + " (no ScanXY)", "mW"))
    else:
        out.append(("Total Power", NA + " (PRF=0)", "mW"))

    # ── Derated values (.3 — 0.3 dB/cm/MHz path to focal depth) ───────── #
    # Pressure derating uses sqrt(DF) because DF is for intensity (∝ p²).
    derate_ok = (z_sp_cm is not None and z_sp_cm > 0 and
                 fc_mhz  is not None and fc_mhz  > 0)
    if derate_ok:
        df_i  = 10 ** (-0.3 * fc_mhz * z_sp_cm / 10)   # intensity
        df_p  = df_i ** 0.5                              # pressure

        def _d(val, factor, fmt):
            return f"{val * factor:{fmt}}" if val is not None else NA + " (no data)"

        out.append(("p+.3 (derated)",    _d(p_pos_mpa,  df_p, ".4f"), "MPa"))
        out.append(("p-.3 (derated)",    _d(p_neg_mpa,  df_p, ".4f"), "MPa"))
        if p_neg_mpa is not None and fc_mhz > 0:
            out.append(("MI.3 (derated)", f"{p_neg_mpa * df_p / np.sqrt(fc_mhz):.3f}", ""))
        else:
            out.append(("MI.3 (derated)", NA + " (no data)", ""))
        out.append(("ISPPA.3 (derated)", _d(isppa_wcm2,  df_i, ".3f"), "W/cm²"))
        out.append(("ISPTA.3 (derated)", _d(I_SPTA_mWcm2, df_i, ".3f"), "mW/cm²"))
        out.append(("Power.3 (derated)", _d(W_mW,         df_i, ".3f"), "mW"))
    else:
        for lbl in ("p+.3", "p-.3", "MI.3", "ISPPA.3", "ISPTA.3", "Power.3"):
            out.append((f"{lbl} (derated)", NA + " (need focal depth)", ""))

    # ── Peak centre from 2D scan ────────────────────────────────────────── #
    if xy_rows:
        pk = max(xy_rows, key=lambda r: r.get("pii", 0.0))
        out.append(("Peak Centre (Pc)",
                    f"x={pk['x_mm']:.2f}, y={pk['y_mm']:.2f}", "mm"))
    else:
        out.append(("Peak Centre (Pc)", NA + " (no ScanXY)", "mm"))

    # ── Beam widths ─────────────────────────────────────────────────────── #
    db_label = f"{int(bw_db)} dB" if bw_db == int(bw_db) else f"{bw_db} dB"

    for axis_label, rows, col in (("X Beam Width", x_rows, "x_mm"),
                                   ("Y Beam Width", y_rows, "y_mm")):
        src = "ScanX" if col == "x_mm" else "ScanY"
        if rows:
            pos = [r[col]    for r in rows]
            amp = [r.get("v_pp", 0.0) for r in rows]
            bw  = beam_width_1d(pos, amp, bw_db)
            if bw is not None:
                out.append((f"{axis_label} ({db_label})", f"{bw:.3f}", "mm"))
            else:
                out.append((f"{axis_label} ({db_label})",
                            NA + " (crossings not found)", "mm"))
        else:
            out.append((f"{axis_label} ({db_label})",
                        f"{NA} (no {src})", "mm"))

    # ── Thermal Indices (NEMA ODS / IEC 60601-2-37 simplified model) ──────── #
    # Derating: 0.3 dB/cm/MHz through soft tissue to focal depth z_sp.
    # Scanned (B-mode):   TIS/TIB use power term only.
    # Unscanned (M-mode): TIS/TIB use max(power term, intensity term).
    # TIC uses underated power — assumes bone is at the surface.

    mode_str = "Scanned" if scan_mode == "scanned" else "Unscanned"
    ti_ok = (aprt_area_cm2 is not None and aprt_area_cm2 > 0 and
             z_sp_cm is not None and z_sp_cm > 0 and
             fc_mhz is not None and fc_mhz > 0)

    if not ti_ok:
        for lbl in ("TIS", "TIB", "TIC"):
            out.append((f"{lbl} ({mode_str})", NA + " (need aperture & depth)", ""))
    else:
        df   = 10 ** (-0.3 * fc_mhz * z_sp_cm / 10)          # 0.3 dB/cm/MHz derating
        w03  = (W_mW * df)         if W_mW         is not None else None
        i03  = (I_SPTA_mWcm2 * df) if I_SPTA_mWcm2 is not None else None
        A, z, f = aprt_area_cm2, z_sp_cm, fc_mhz

        # TIS ── soft tissue
        if w03 is not None:
            tis_p = w03 * f ** (1 / 3) / (210 * A ** 0.5)          # power term
            if scan_mode == "scanned" or i03 is None:
                tis = tis_p
            else:
                tis = max(tis_p, i03 * z / (210 * f ** 0.25))       # intensity term
            out.append((f"TIS ({mode_str})", f"{tis:.3f}", ""))
        else:
            out.append((f"TIS ({mode_str})", NA + " (no ScanXY)", ""))

        # TIB ── bone at focus
        if w03 is not None:
            tib_p = f ** 0.5 * w03 / (40 * A ** 0.5)               # power term
            if scan_mode == "scanned" or i03 is None:
                tib = tib_p
            else:
                tib = max(tib_p, i03 * z * f ** 0.5 / 40)          # intensity term
            out.append((f"TIB ({mode_str})", f"{tib:.3f}", ""))
        else:
            out.append((f"TIB ({mode_str})", NA + " (no ScanXY)", ""))

        # TIC ── cranial (underated — bone at surface, no tissue path)
        if W_mW is not None:
            out.append((f"TIC ({mode_str})", f"{W_mW / 40:.3f}", ""))
        else:
            out.append((f"TIC ({mode_str})", NA + " (no ScanXY)", ""))

    return out


# ── GUI ──────────────────────────────────────────────────────────────────── #

class AnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic Parameter Calculator")
        self.resizable(False, False)
        self._results = []
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── Input files ──────────────────────────────────────────────────── #
        ff = ttk.LabelFrame(self, text="Input Files")
        ff.grid(row=0, column=0, sticky="ew", **pad)

        self._wf_var = tk.StringVar()
        self._xy_var = tk.StringVar()
        self._x_var  = tk.StringVar()
        self._y_var  = tk.StringVar()

        file_rows = [
            ("Waveform",  self._wf_var, "Single capture at focus — optional; falls back to ScanXY peak"),
            ("Scan XY",   self._xy_var, "2D spatial scan — for Power, ISPPA/ISPTA, Peak Centre"),
            ("Scan X",    self._x_var,  "1D X scan — for X beam width"),
            ("Scan Y",    self._y_var,  "1D Y scan — for Y beam width"),
        ]
        for r, (label, var, tip) in enumerate(file_rows):
            ttk.Label(ff, text=label, width=10, anchor="w").grid(
                row=r, column=0, padx=(6, 2), pady=3, sticky="w")
            ttk.Entry(ff, textvariable=var, width=50).grid(
                row=r, column=1, padx=2, pady=3)
            ttk.Button(ff, text="Browse…",
                       command=lambda v=var: self._browse(v)).grid(
                row=r, column=2, padx=(2, 4), pady=3)
            ttk.Label(ff, text=tip, foreground="gray").grid(
                row=r, column=3, padx=(4, 8), pady=3, sticky="w")

        # ── Parameters ───────────────────────────────────────────────────── #
        pf = ttk.LabelFrame(self, text="Parameters")
        pf.grid(row=1, column=0, sticky="ew", **pad)

        self._prf_var    = tk.StringVar(value="40")
        self._lines_var  = tk.StringVar(value="1")
        self._bw_db_var  = tk.StringVar(value="-6")

        ttk.Label(pf, text="PRF (Hz)").grid(
            row=0, column=0, padx=(8, 2), pady=5, sticky="w")
        ttk.Entry(pf, textvariable=self._prf_var, width=10).grid(
            row=0, column=1, padx=(0, 16), pady=5)
        ttk.Label(pf, text="Lines per frame").grid(
            row=0, column=2, padx=(0, 2), pady=5, sticky="w")
        ttk.Entry(pf, textvariable=self._lines_var, width=6).grid(
            row=0, column=3, padx=(0, 16), pady=5)
        ttk.Label(pf, text="Beam width threshold (dB)").grid(
            row=0, column=4, padx=(0, 2), pady=5, sticky="w")
        ttk.Entry(pf, textvariable=self._bw_db_var, width=6).grid(
            row=0, column=5, padx=(0, 8), pady=5)
        ttk.Label(pf, text="−6 dB = standard  |  Lines per frame: 1 = unscanned / M-mode",
                  foreground="gray").grid(
            row=1, column=0, columnspan=6, padx=(8, 8), pady=(0, 5), sticky="w")

        # ── Thermal Index Parameters ─────────────────────────────────────── #
        tf = ttk.LabelFrame(self, text="Thermal Index Parameters")
        tf.grid(row=2, column=0, sticky="ew", **pad)

        self._ti_mode_var   = tk.StringVar(value="Scanned")
        self._aprt_var      = tk.StringVar(value="")
        self._z_sp_var      = tk.StringVar(value="")

        ttk.Label(tf, text="Scan mode").grid(
            row=0, column=0, padx=(8, 2), pady=5, sticky="w")
        ttk.OptionMenu(tf, self._ti_mode_var, "Scanned", "Scanned", "Unscanned").grid(
            row=0, column=1, padx=(0, 16), pady=5, sticky="w")
        ttk.Label(tf, text="Aperture area (cm²)").grid(
            row=0, column=2, padx=(0, 2), pady=5, sticky="w")
        ttk.Entry(tf, textvariable=self._aprt_var, width=8).grid(
            row=0, column=3, padx=(0, 16), pady=5)
        ttk.Label(tf, text="Focal depth (cm)").grid(
            row=0, column=4, padx=(0, 2), pady=5, sticky="w")
        ttk.Entry(tf, textvariable=self._z_sp_var, width=8).grid(
            row=0, column=5, padx=(0, 8), pady=5)
        ttk.Label(tf,
                  text="Leave aperture/depth blank to skip TI",
                  foreground="gray").grid(
            row=0, column=6, padx=(8, 8), pady=5, sticky="w")

        # ── Calculate ────────────────────────────────────────────────────── #
        ttk.Button(self, text="Calculate", command=self._calculate).grid(
            row=3, column=0, pady=(4, 2))

        # ── Results table ────────────────────────────────────────────────── #
        rf = ttk.LabelFrame(self, text="Results")
        rf.grid(row=4, column=0, sticky="nsew", **pad)

        self._tree = ttk.Treeview(rf, columns=("value", "unit"),
                                  show="tree headings", height=14)
        self._tree.heading("#0",     text="Parameter",  anchor="w")
        self._tree.heading("value",  text="Value",      anchor="e")
        self._tree.heading("unit",   text="Unit",       anchor="w")
        self._tree.column("#0",    width=250, anchor="w")
        self._tree.column("value", width=160, anchor="e")
        self._tree.column("unit",  width=80,  anchor="w")

        vsb = ttk.Scrollbar(rf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        vsb.grid(row=0, column=1, sticky="ns", pady=4, padx=(0, 4))

        ttk.Button(rf, text="Export CSV…", command=self._export).grid(
            row=1, column=0, columnspan=2, pady=(0, 6))

        # ── Status ───────────────────────────────────────────────────────── #
        self._status = tk.StringVar(value="Load files and click Calculate.")
        ttk.Label(self, textvariable=self._status, foreground="gray").grid(
            row=5, column=0, sticky="w", padx=10, pady=(0, 6))

    def _browse(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.csv *.txt"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _calculate(self):
        try:
            prf    = float(self._prf_var.get())
            bw_db  = float(self._bw_db_var.get())
            lines  = max(1, int(float(self._lines_var.get())))
        except ValueError:
            self._status.set("Error: PRF, lines per frame, and beam-width threshold must be numbers.")
            return

        def _load(var):
            p = var.get().strip()
            return load_any(p) if p else []

        try:
            wf = _load(self._wf_var)
            xy = _load(self._xy_var)
            x  = _load(self._x_var)
            y  = _load(self._y_var)
        except Exception as exc:
            self._status.set(f"File load error: {exc}")
            return

        if not any([wf, xy, x, y]):
            self._status.set("No files loaded — browse for at least one input.")
            return

        scan_mode = "scanned" if self._ti_mode_var.get() == "Scanned" else "unscanned"
        try:
            aprt = float(self._aprt_var.get()) if self._aprt_var.get().strip() else None
        except ValueError:
            self._status.set("Error: Aperture area must be a number.")
            return
        try:
            z_sp = float(self._z_sp_var.get()) if self._z_sp_var.get().strip() else None
        except ValueError:
            self._status.set("Error: Focal depth must be a number.")
            return

        try:
            self._results = compute_metrics(wf, xy, x, y, prf, bw_db,
                                            scan_mode=scan_mode,
                                            aprt_area_cm2=aprt,
                                            z_sp_cm=z_sp,
                                            lines_per_frame=lines)
        except Exception as exc:
            self._status.set(f"Calculation error: {exc}")
            return

        for item in self._tree.get_children():
            self._tree.delete(item)
        for param, value, unit in self._results:
            self._tree.insert("", "end", text=param, values=(value, unit))

        self._status.set(f"Done — {len(self._results)} metrics calculated.")

    def _export(self):
        if not self._results:
            self._status.set("Nothing to export — run Calculate first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value", "Unit"])
            writer.writerows(self._results)
        self._status.set(f"Exported → {os.path.basename(path)}")


if __name__ == "__main__":
    AnalysisApp().mainloop()
