#!/usr/bin/env python3
"""
Analyze a waveform file using the same calculations as scan_gui.capture().

File format (same as Rigol exported waveform):
  Line 1: dt (x_increment in seconds)
  Line 2: x_origin (seconds)
  Line 3+: voltage samples (volts)

Usage:
  python analyze_waveform.py <waveform_file> [<waveform_file2> ...]
"""

import os
import sys

import numpy as np


# ── Calibration table ────────────────────────────────────────────────────── #

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


def get_cal_factor(freq_hz):
    freq_mhz = freq_hz / 1e6
    idx = int(np.argmin(np.abs(_CAL_FREQS - freq_mhz)))
    return float(_CAL_FACTORS[idx]), float(_CAL_FREQS[idx])


# ── Waveform loading ─────────────────────────────────────────────────────── #

def load_waveform(path):
    """Return (dt, timescale, voltage_array) from a waveform text file.
    Line 1: dt (time per sample in seconds)
    Line 2: timescale (s/div, sign ignored)
    Line 3+: voltage samples (volts)
    """
    with open(path, encoding='latin-1') as f:
        lines = [l.strip() for l in f if l.strip()]
    dt        = float(lines[0])
    timescale = abs(float(lines[1]))
    samples = []
    for line in lines[2:]:
        for token in line.split():
            try:
                samples.append(float(token))
            except ValueError:
                pass
    voltage = np.array(samples)
    return dt, timescale, voltage


# ── Analysis ─────────────────────────────────────────────────────────────── #

def analyze(dt, voltage):
    """Replicate the calculations in scan_gui.capture()."""
    v_ac = voltage - voltage.mean()

    # Frequency: zero-crossing counter with amplitude threshold
    threshold = 0.15 * np.max(np.abs(v_ac))
    raw_crosses = np.where((v_ac[:-1] < 0) & (v_ac[1:] >= 0))[0]
    crosses = [i for i in raw_crosses
               if np.max(v_ac[max(0, i - 50):i + 51]) >  threshold
               and np.min(v_ac[max(0, i - 50):i + 51]) < -threshold]
    if len(crosses) >= 2:
        refined = []
        for i in crosses:
            frac = -v_ac[i] / (v_ac[i + 1] - v_ac[i])
            refined.append((i + frac) * dt)
        periods = np.diff(refined)
        freq = float(1.0 / np.mean(periods))
    else:
        # Single-cycle fallback: peak-to-trough half-period.
        # Robust for arbitrary frequencies; accuracy limited only by sample
        # resolution, not FFT bin spacing.
        idx_peak   = int(np.argmax(v_ac))
        idx_trough = int(np.argmin(v_ac))
        half_period = abs(idx_peak - idx_trough) * dt
        freq = float(1.0 / (2 * half_period)) if half_period > 0 else 0.0

    RHO_C = 1.54e6  # acoustic impedance of water (Pa·s/m)

    cal_factor, cal_freq_mhz = get_cal_factor(freq)
    pii = float(np.sum((v_ac / (cal_factor * 1e-6)) ** 2) * dt) / (RHO_C * 1e4)  # J/cm²

    # Pulse duration: 10–90% cumulative energy × 1.25 correction factor
    cumulative = np.cumsum(v_ac ** 2) * dt
    total_energy = cumulative[-1]
    t1 = int(np.searchsorted(cumulative, 0.10 * total_energy)) * dt
    t2 = int(np.searchsorted(cumulative, 0.90 * total_energy)) * dt
    pd = float((t2 - t1) * 1.25)

    return {
        "n_samples":      len(voltage),
        "timebase_s":     dt,
        "v_min":          float(voltage.min()),
        "v_max":          float(voltage.max()),
        "v_pp":           float(voltage.max() - voltage.min()),
        "v_mean":         float(voltage.mean()),
        "v_rms":          float(np.sqrt(np.mean(v_ac ** 2))),
        "frequency_hz":   freq,
        "cal_freq_mhz":   cal_freq_mhz,
        "cal_factor":     cal_factor,
        "pii":            pii,
        "pd":             pd,
    }


def print_results(path, timescale, results):
    r = results
    print(f"\n{'─'*50}")
    print(f"File       : {path}")
    print(f"Timescale  : {timescale:.4g} s/div")
    print(f"Samples    : {r['n_samples']}  dt={r['timebase_s']:.4g} s")
    print(f"V_min      : {r['v_min']:.6g} V")
    print(f"V_max      : {r['v_max']:.6g} V")
    print(f"V_pp       : {r['v_pp']:.6g} V")
    print(f"V_mean     : {r['v_mean']:.6g} V")
    print(f"V_rms      : {r['v_rms']:.6g} V  (AC)")
    print(f"Frequency  : {r['frequency_hz']:.4g} Hz  ({r['frequency_hz']/1e6:.4g} MHz)")
    print(f"Cal lookup : {r['cal_freq_mhz']:.4g} MHz  →  factor={r['cal_factor']:.6g}")
    print(f"PII        : {r['pii']:.6e} J/cm²")
    print(f"PD         : {r['pd']:.6g} s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for waveform_path in sys.argv[1:]:
        try:
            dt, timescale, voltage = load_waveform(waveform_path)
            results = analyze(dt, voltage)
            print_results(waveform_path, timescale, results)
        except Exception as e:
            print(f"ERROR processing {waveform_path}: {e}", file=sys.stderr)
