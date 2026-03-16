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
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    dt        = float(lines[0])
    timescale = abs(float(lines[1]))
    voltage   = np.array([float(v) for v in lines[2:]])
    return dt, timescale, voltage


# ── Analysis ─────────────────────────────────────────────────────────────── #

def analyze(dt, voltage):
    """Replicate the calculations in scan_gui.capture()."""
    v_ac = voltage - voltage.mean()

    # Frequency via Hanning-windowed zero-padded FFT
    nfft   = 131072
    window = np.hanning(len(v_ac))
    fft_mag = np.abs(np.fft.rfft(v_ac * window, n=nfft))
    k = int(np.argmax(fft_mag[1:]) + 1)
    if 1 < k < len(fft_mag) - 1:
        a, b, c = fft_mag[k - 1], fft_mag[k], fft_mag[k + 1]
        denom  = a - 2 * b + c
        k_frac = k + (a - c) / (2 * denom) if denom != 0 else k
    else:
        k_frac = k
    freq = float(k_frac / (nfft * dt))

    RHO_C = 1.5e6   # acoustic impedance of water (Pa·s/m)

    cal_factor, cal_freq_mhz = get_cal_factor(freq)
    pii = float(np.sum((v_ac / (cal_factor * 1e-6)) ** 2) * dt) / (RHO_C * 1e4)  # J/cm²
    pd  = float(len(voltage) * dt)

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
