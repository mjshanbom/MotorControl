import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("334.TXT")
dt = 1.5e-9              # change if needed
fs = 1 / dt

# remove DC
y = y - np.mean(y)

# apply window
w = np.hanning(len(y))
yw = y * w

# zero pad for smoother spectrum
Nfft = 8 * len(y)

Y = np.fft.rfft(yw, n=Nfft)
freq = np.fft.rfftfreq(Nfft, d=dt)
mag = np.abs(Y)

# ignore DC
i = np.argmax(mag[1:]) + 1

# quadratic interpolation around peak
if 1 <= i < len(mag) - 1:
    a = mag[i-1]
    b = mag[i]
    c = mag[i+1]
    denom = (a - 2*b + c)
    if denom != 0:
        p = 0.5 * (a - c) / denom
    else:
        p = 0
else:
    p = 0

df = freq[1] - freq[0]
f_est = freq[i] + p * df

print(f"Estimated frequency: {f_est:.3f} Hz")
print(f"Estimated frequency: {f_est/1e6:.6f} MHz")

plt.figure(figsize=(8,5))
plt.plot(freq/1e6, mag)
plt.axvline(f_est/1e6, linestyle='--')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title("FFT with interpolated peak")
plt.grid(True)
plt.show()