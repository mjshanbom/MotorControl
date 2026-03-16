#!/usr/bin/env python3
"""
Quick diagnostic: open the VP9000 serial port and let you type raw commands.
Sends each line you type, prints whatever the controller responds.
Ctrl+C to quit.

Usage:
    python motor_diag.py [port] [baudrate]

Defaults: /dev/tty.usbserial-FTEHI1FO  9600
"""

import sys
import time
import serial

PORT     = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbserial-FTEHI1FO"
BAUDRATE = int(sys.argv[2]) if len(sys.argv) > 2 else 9600

print(f"Connecting to {PORT} at {BAUDRATE} baud...")
ser = serial.Serial(PORT, baudrate=BAUDRATE, bytesize=serial.SEVENBITS,
                    parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_TWO,
                    timeout=1.0)
time.sleep(0.2)
print("Connected. Type commands (e.g. F, V, K). Ctrl+C to quit.\n")

try:
    while True:
        cmd = input("cmd> ").strip()
        if not cmd:
            continue
        ser.write((cmd + "\r").encode("ascii"))
        time.sleep(0.2)
        response = ser.read(ser.in_waiting or 1)
        if response:
            print(f"  <- {response!r}  (decoded: {response.decode('ascii', errors='replace').strip()!r})")
        else:
            print("  <- (no response)")
except KeyboardInterrupt:
    print("\nDone.")
finally:
    ser.close()
