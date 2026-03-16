import pyvisa
import numpy as np
import time

rm = pyvisa.ResourceManager("@py")
scope = rm.open_resource("USB0::6833::1303::DS1ZE265M00121::0::INSTR")
scope.timeout = 10000
scope.chunk_size = 1024 * 1024

scope.write(":STOP")
# Wait until the scope has actually stopped acquiring
for _ in range(20):
    if scope.query(":TRIG:STAT?").strip() == "STOP":
        break
    time.sleep(0.1)
else:
    raise RuntimeError("Scope did not stop in time")
scope.write(":WAV:SOUR CHAN1")
scope.write(":WAV:MODE NORM")
scope.write(":WAV:FORM BYTE")
scope.write(":WAV:STAR 1")
scope.write(":WAV:STOP 1200")

# Parse preamble to get scaling factors
preamble = scope.query(":WAV:PRE?").strip().split(",")
print("Preamble:", preamble)

x_increment = float(preamble[4])
x_origin    = float(preamble[5])
x_reference = float(preamble[6])
y_increment = float(preamble[7])
y_origin    = float(preamble[8])
y_reference = float(preamble[9])

# Read waveform and manually strip the TMC block header (#N<length><data>)
scope.write(":WAV:DATA?")
time.sleep(0.5)
raw_bytes = scope.read_raw()
n_digits = int(chr(raw_bytes[1]))
data_start = 2 + n_digits
n_points = int(preamble[2])
raw = np.frombuffer(raw_bytes[data_start:data_start + n_points], dtype=np.uint8)

print("Samples:", len(raw))

# Convert raw ADC counts to voltage and time
voltage = (raw - y_reference) * y_increment + y_origin
time    = (np.arange(len(raw)) - x_reference) * x_increment + x_origin

print("Time range:    {:.6f} s  to  {:.6f} s".format(time[0], time[-1]))
print("Voltage range: {:.4f} V  to  {:.4f} V".format(voltage.min(), voltage.max()))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(time * 1e3, voltage)  # time in ms for readability
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.title("CH1 Waveform")
plt.grid(True)
plt.tight_layout()
plt.show()

scope.write(":RUN")