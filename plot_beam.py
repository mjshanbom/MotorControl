#!/usr/bin/env python3
"""Plot a 2D PII heatmap from a scan_results.csv file."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_beam(csv_path):
    df = pd.read_csv(csv_path)

    # If multiple Z planes, prompt user to pick one
    z_planes = sorted(df["z_mm"].unique())
    if len(z_planes) > 1:
        print("Available Z planes (mm):", z_planes)
        z_val = float(input("Enter Z value to plot: "))
    else:
        z_val = z_planes[0]

    df = df[df["z_mm"] == z_val]

    x = sorted(df["x_mm"].unique())
    y = sorted(df["y_mm"].unique())

    grid = np.full((len(y), len(x)), np.nan)
    x_idx = {v: i for i, v in enumerate(x)}
    y_idx = {v: i for i, v in enumerate(y)}

    for _, row in df.iterrows():
        grid[y_idx[row["y_mm"]], x_idx[row["x_mm"]]] = row["pii"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x, y, grid, cmap="hot", shading="nearest",
                       norm=Normalize(vmin=0, vmax=np.nanmax(grid)))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PII (J/cm²)")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"PII Beam Map — Z = {z_val} mm")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scan_results.csv"
    plot_beam(path)
