#!/usr/bin/env python3
"""Plot oscilloscope waveform from Data_*.txt files."""

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_waveform(path):
    with open(path) as f:
        vals = [float(v) for v in f.read().strip().split("\t")]
    x_increment = vals[0]
    x_origin    = vals[1]
    voltage     = np.array(vals[2:])
    time        = x_origin + np.arange(len(voltage)) * x_increment
    return time, voltage


def main():
    files = sys.argv[1:] or sorted(glob.glob("Data_*.txt"))
    if not files:
        print("No Data_*.txt files found.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 4))

    for path in files:
        time, voltage = load_waveform(path)
        label = path.replace("Data_", "").replace(".txt", "")
        ax.plot(time * 1e6, voltage * 1e3, label=label, linewidth=0.8)

    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Oscilloscope Waveform")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
