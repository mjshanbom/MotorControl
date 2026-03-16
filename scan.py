#!/usr/bin/env python3
"""
2D grid scan: moves motor 1 (X) and motor 2 (Y) through a grid,
captures scope waveform at each point, and saves summary stats to CSV.
"""

import argparse
import csv
import time
import numpy as np
import pyvisa

from velmex_vp9000 import VP9000


def parse_args():
    p = argparse.ArgumentParser(description="2D motor grid scan with scope capture")

    p.add_argument("--port",       default="/dev/tty.usbserial-FTEHI1FO", help="VP9000 serial port")
    p.add_argument("--baudrate",   type=int, default=9600,  help="Baud rate (default: 9600)")
    p.add_argument("--bytesize",   type=int, default=7,     help="Data bits: 7 or 8 (default: 7)")
    p.add_argument("--parity",     default="E",             help="Parity: N, E, or O (default: E)")
    p.add_argument("--stopbits",   type=int, default=2,     help="Stop bits: 1 or 2 (default: 2)")
    p.add_argument("--visa",       default="USB0::6833::1303::DS1ZE265M00121::0::INSTR", help="Scope VISA address")
    p.add_argument("--channel",    default="CHAN1",            help="Scope channel (default: CHAN1)")
    p.add_argument("--output",     default="scan_results.csv", help="Output CSV file")

    p.add_argument("--x-motor",    type=int,   default=1,    help="Motor number for X axis (default: 1)")
    p.add_argument("--y-motor",    type=int,   default=2,    help="Motor number for Y axis (default: 2)")

    p.add_argument("--x-start",   type=int,   default=0,    help="X start position in steps (default: 0)")
    p.add_argument("--x-stop",    type=int,   default=5000, help="X stop position in steps (default: 5000)")
    p.add_argument("--x-step",    type=int,   default=500,  help="X step size in steps (default: 500)")

    p.add_argument("--y-start",   type=int,   default=0,    help="Y start position in steps (default: 0)")
    p.add_argument("--y-stop",    type=int,   default=5000, help="Y stop position in steps (default: 5000)")
    p.add_argument("--y-step",    type=int,   default=500,  help="Y step size in steps (default: 500)")

    p.add_argument("--speed",     type=int,   default=1000, help="Motor speed in steps/sec (default: 1000)")
    p.add_argument("--settle",    type=float, default=0.05, help="Settle time in seconds after each move (default: 0.05)")

    return p.parse_args()


def setup_scope(scope, channel):
    """Configure WAV parameters once and return cached preamble constants."""
    scope.write(f":WAV:SOUR {channel}")
    scope.write(":WAV:MODE NORM")
    scope.write(":WAV:FORM BYTE")
    scope.write(":WAV:STAR 1")
    scope.write(":WAV:STOP 1200")
    preamble    = scope.query(":WAV:PRE?").strip().split(",")
    return {
        "n_points":    int(preamble[2]),
        "x_increment": float(preamble[4]),
        "x_origin":    float(preamble[5]),
        "y_increment": float(preamble[7]),
        "y_origin":    float(preamble[8]),
        "y_reference": float(preamble[9]),
    }


def capture(scope, pre) -> dict:
    """Scope already stopped by caller. Read waveform using cached preamble."""
    s = scope
    s.write(":WAV:DATA?")
    time.sleep(0.3)
    raw_bytes  = s.read_raw()
    n_digits   = int(chr(raw_bytes[1]))
    data_start = 2 + n_digits
    raw = np.frombuffer(raw_bytes[data_start:data_start + pre["n_points"]], dtype=np.uint8)

    voltage = (raw - pre["y_reference"]) * pre["y_increment"] + pre["y_origin"]
    s.write(":RUN")

    return {
        "n_samples":   len(voltage),
        "timebase_s":  pre["x_increment"],
        "x_origin_s":  pre["x_origin"],
        "v_min":       float(voltage.min()),
        "v_max":       float(voltage.max()),
        "v_pp":        float(voltage.max() - voltage.min()),
        "v_mean":      float(voltage.mean()),
        "v_rms":       float(np.sqrt(np.mean(voltage**2))),
    }


def main():
    args = parse_args()

    x_positions = range(args.x_start, args.x_stop + args.x_step, args.x_step)
    y_positions = range(args.y_start, args.y_stop + args.y_step, args.y_step)
    total = len(list(x_positions)) * len(list(y_positions))

    rm = pyvisa.ResourceManager("@py")
    scope = rm.open_resource(args.visa)
    scope.timeout = 10000
    scope.chunk_size = 1024 * 1024

    fieldnames = ["x_steps", "y_steps", "n_samples", "timebase_s",
                  "x_origin_s", "v_min", "v_max", "v_pp", "v_mean", "v_rms"]

    with VP9000(args.port, baudrate=args.baudrate, bytesize=args.bytesize,
                parity=args.parity, stopbits=args.stopbits) as motor, \
         open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        motor.enable_online_mode()
        motor.set_speed(args.x_motor, args.speed)
        motor.set_speed(args.y_motor, args.speed)

        pre = setup_scope(scope, args.channel)

        done = 0
        for y in y_positions:
            motor.move_absolute(args.y_motor, y)
            motor.wait_for_done()

            for x in x_positions:
                motor.move_absolute(args.x_motor, x)
                scope.write(":STOP")  # Stop scope while motor is moving
                motor.wait_for_done()
                time.sleep(args.settle)

                stats = capture(scope, pre)
                row = {"x_steps": x, "y_steps": y, **stats}
                writer.writerow(row)
                f.flush()

                done += 1
                print(f"[{done}/{total}] x={x}, y={y}  "
                      f"Vpp={stats['v_pp']:.4f} V  Vrms={stats['v_rms']:.4f} V")

        motor.move_absolute(args.x_motor, args.x_start)
        motor.move_absolute(args.y_motor, args.y_start)
        motor.wait_for_done()

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
