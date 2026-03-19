#!/usr/bin/env python3
"""
Ultrasound beam profile plot — mirrors the LabVIEW CalculateMaxPII display.

Three curves plotted vs scan position:
  PII   — raw (no attenuation correction)
  PII.3 — derated at 0.3 dB/(MHz·cm)
  PII.6 — derated at 0.6 dB/(MHz·cm)

Scan types handled automatically:
  Z scan  — z varies, x/y fixed  → plot vs depth (attenuation per point)
  X scan  — x varies, z/y fixed  → plot vs X position
  Y scan  — y varies, z/x fixed  → plot vs Y position
  XY scan — x and y both vary    → prompt for X or Y axis, slice through peak
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Derating coefficients in dB/(MHz·cm)
ALPHA_3 = 0.3
ALPHA_6 = 0.6

_TSV_COLS = ["x_mm", "y_mm", "z_mm", "v_pp", "v_max", "v_min",
             "frequency", "pii", "pd"]


def _load_scan(path):
    """Load scan results from either a CSV (with header) or a headerless TSV.

    Frequency is normalised to Hz on load.  CSV files (scan_gui output) store
    frequency in Hz; headerless TSV files store frequency in MHz.
    """
    with open(path, encoding="latin-1") as f:
        first = f.readline()
    if first.strip().startswith("x_mm"):
        return pd.read_csv(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_TSV_COLS,
                     engine="python")
    # TSV stores frequency in MHz — convert to Hz
    df["frequency"] = df["frequency"] * 1e6
    return df


def correct_pii(pii, freq_hz, z_mm, alpha_db):
    """Derate PII: PII × 10^(−α_dB × f_MHz × z_cm / 10)."""
    return pii * 10 ** (-alpha_db * (freq_hz / 1e6) * (z_mm / 10.0) / 10)


def _dark_axes(ax):
    ax.set_facecolor("#111111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.grid(True, color="#2a2a2a", linewidth=0.6)


def _draw_plot(profile, pos, axis_lbl, title):
    """Build and show the standard three-curve plot."""
    pii_raw = profile["pii"].values
    pii3    = profile["pii3"].values
    pii6    = profile["pii6"].values

    idx_raw = np.argmax(pii_raw)
    idx_3   = np.argmax(pii3)
    idx_6   = np.argmax(pii6)

    max_pii  = pii_raw[idx_raw];  pos_pii  = pos[idx_raw]
    max_pii3 = pii3[idx_3];       pos_pii3 = pos[idx_3]
    max_pii6 = pii6[idx_6];       pos_pii6 = pos[idx_6]

    fig = plt.figure(figsize=(11, 6), facecolor="#2a2a2a")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)

    ax = fig.add_subplot(gs[0])
    _dark_axes(ax)

    ax.plot(pos, pii_raw, color="#FF6633", linewidth=1.5, label="PII")
    ax.plot(pos, pii3,    color="#6699FF", linewidth=1.5, label="PII.3")
    ax.plot(pos, pii6,    color="#DDDDDD", linewidth=1.5, label="PII.6")

    ax.set_ylabel("PII (J/cm²)", color="#aaaaaa")
    ax.set_xlabel(axis_lbl, color="#aaaaaa")
    ax.set_xlim(pos[0], pos[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2E}"))
    ax.legend(loc="upper right", facecolor="#333333", edgecolor="#555555",
              labelcolor="white", fontsize=9)
    ax.set_title(title, color="#cccccc", pad=6)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#2a2a2a")
    ax2.axis("off")

    metrics = [
        ("Max PII",   f"{max_pii:.3E}",  "Pos Max PII",   f"{pos_pii:.2f}"),
        ("Max PII.3", f"{max_pii3:.3E}", "Pos Max PII.3", f"{pos_pii3:.2f}"),
        ("Max PII.6", f"{max_pii6:.3E}", "Pos Max PII.6", f"{pos_pii6:.2f}"),
    ]
    col_x = [0.05, 0.22, 0.45, 0.62]
    for row_i, (lbl_v, val_v, lbl_p, val_p) in enumerate(metrics):
        y = 0.85 - row_i * 0.30
        for col, txt, clr in [
            (col_x[0], lbl_v, "#aaaaaa"),
            (col_x[1], val_v, "#ffffff"),
            (col_x[2], lbl_p, "#aaaaaa"),
            (col_x[3], val_p, "#ffffff"),
        ]:
            ax2.text(col, y, txt, transform=ax2.transAxes,
                     color=clr, fontsize=9, va="top",
                     fontfamily="monospace")

    plt.tight_layout(pad=0.6)

    print(f"\n{title}")
    print(f"  Max PII   = {max_pii:.4e}  @ {pos_pii:.2f} mm")
    print(f"  Max PII.3 = {max_pii3:.4e}  @ {pos_pii3:.2f} mm")
    print(f"  Max PII.6 = {max_pii6:.4e}  @ {pos_pii6:.2f} mm")

    plt.show()


def plot_beam(path):
    df = _load_scan(path)

    z_unique = sorted(df["z_mm"].unique())
    x_unique = sorted(df["x_mm"].unique())
    y_unique = sorted(df["y_mm"].unique())

    z_varies = len(z_unique) > 1
    x_varies = len(x_unique) > 1
    y_varies = len(y_unique) > 1

    # ── Z scan: depth profile ──────────────────────────────────────────────── #
    if z_varies and not x_varies and not y_varies:
        profile = df.sort_values("z_mm").copy()
        # Attenuation correction uses each point's own depth
        profile["pii3"] = correct_pii(profile["pii"], profile["frequency"],
                                      profile["z_mm"], ALPHA_3)
        profile["pii6"] = correct_pii(profile["pii"], profile["frequency"],
                                      profile["z_mm"], ALPHA_6)
        pos = profile["z_mm"].values
        x0, y0 = x_unique[0], y_unique[0]
        title = f"Z Scan — X = {x0} mm, Y = {y0} mm"
        _draw_plot(profile, pos, "Z (mm)", title)
        return

    # ── Lateral scan(s): select Z plane first ─────────────────────────────── #
    if z_varies:
        print("Available Z planes (mm):", z_unique)
        z_val = float(input("Enter Z value to plot: "))
    else:
        z_val = z_unique[0]

    df = df[df["z_mm"] == z_val].copy()
    df["pii3"] = correct_pii(df["pii"], df["frequency"], z_val, ALPHA_3)
    df["pii6"] = correct_pii(df["pii"], df["frequency"], z_val, ALPHA_6)

    x_unique = sorted(df["x_mm"].unique())
    y_unique = sorted(df["y_mm"].unique())
    is_2d    = len(x_unique) > 1 and len(y_unique) > 1

    if is_2d:
        axes_choice = input("Plot along X or Y axis? [X/Y]: ").strip().upper()
        if axes_choice == "Y":
            peak_row = df.loc[df["pii6"].idxmax()]
            fixed_x  = peak_row["x_mm"]
            profile  = df[df["x_mm"] == fixed_x].sort_values("y_mm")
            pos      = profile["y_mm"].values
            axis_lbl = "Y (mm)"
        else:
            peak_row = df.loc[df["pii6"].idxmax()]
            fixed_y  = peak_row["y_mm"]
            profile  = df[df["y_mm"] == fixed_y].sort_values("x_mm")
            pos      = profile["x_mm"].values
            axis_lbl = "X (mm)"
    elif len(x_unique) > 1:
        profile  = df.sort_values("x_mm")
        pos      = profile["x_mm"].values
        axis_lbl = "X (mm)"
    else:
        profile  = df.sort_values("y_mm")
        pos      = profile["y_mm"].values
        axis_lbl = "Y (mm)"

    title = f"Beam Profile — Z = {z_val} mm"
    _draw_plot(profile, pos, axis_lbl, title)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select scan results file",
            filetypes=[("Scan files", "*.csv *.txt"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            print("No file selected.")
            sys.exit(0)
    plot_beam(path)
