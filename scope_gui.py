"""
Rigol DS1000Z — Single-channel scope GUI with PII / PD readout.
"""

import datetime
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pyvisa

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Calibration table ─────────────────────────────────────────────────────── #

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

# ── Theme ─────────────────────────────────────────────────────────────────── #

BG_DARK   = "#1a1a1a"
BG_PANEL  = "#2a2a2a"
BG_WIDGET = "#333333"
FG_TEXT   = "#e0e0e0"
FG_DIM    = "#888888"
ACCENT    = "#555555"

# ── Signal processing (mirrors scan_gui.capture) ──────────────────────────── #

RHO_C = 1.54e6   # acoustic impedance of water (Pa·s/m)

def process_waveform(voltage, dt):
    """Compute frequency, PII, and PD from a captured waveform."""
    v_ac = voltage - voltage.mean()

    # Frequency: FFT primary (robust against noise, ringing, and low samples/cycle)
    n = len(v_ac)
    fft_mag = np.abs(np.fft.rfft(v_ac * np.hanning(n)))
    freqs   = np.fft.rfftfreq(n, d=dt)
    mask = freqs > 0  # exclude DC bin only
    if mask.any():
        sub_mag   = fft_mag[mask]
        sub_freqs = freqs[mask]
        peak_i = int(np.argmax(sub_mag))
        # Parabolic interpolation for sub-bin accuracy
        if 0 < peak_i < len(sub_mag) - 1:
            alpha, beta, gamma = sub_mag[peak_i-1], sub_mag[peak_i], sub_mag[peak_i+1]
            correction = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            fft_freq = sub_freqs[peak_i] + correction * (sub_freqs[1] - sub_freqs[0])
        else:
            fft_freq = sub_freqs[peak_i]
    else:
        fft_freq = 0.0

    # Zero-crossing counter with adaptive window (±0.75 cycles at FFT-estimated frequency)
    threshold    = 0.15 * np.max(np.abs(v_ac))
    half_win     = max(3, int(round(0.75 / (fft_freq * dt)))) if fft_freq > 0 else 50
    raw_crosses  = np.where((v_ac[:-1] < 0) & (v_ac[1:] >= 0))[0]
    crosses      = [i for i in raw_crosses
                    if np.max(v_ac[max(0, i - half_win):i + half_win + 1]) >  threshold
                    and np.min(v_ac[max(0, i - half_win):i + half_win + 1]) < -threshold]
    if len(crosses) >= 2:
        refined = []
        for i in crosses:
            frac = -v_ac[i] / (v_ac[i + 1] - v_ac[i])
            refined.append((i + frac) * dt)
        periods = np.diff(refined)
        # Use median — robust against spurious crossings from ringing/reflections
        zc_freq = float(1.0 / np.median(periods))
        # Accept zero-crossing result only if it agrees with FFT within 20%
        if fft_freq > 0 and abs(zc_freq - fft_freq) / fft_freq < 0.20:
            freq = zc_freq
        else:
            freq = fft_freq
    else:
        freq = fft_freq if fft_freq > 0 else 0.0

    freq_mhz   = freq / 1e6
    cal_idx    = int(np.argmin(np.abs(_CAL_FREQS - freq_mhz)))
    cal_factor = _CAL_FACTORS[cal_idx]

    pii = float(np.sum((v_ac / (cal_factor * 1e-6))**2) * dt) / (RHO_C * 1e4)

    cumulative   = np.cumsum(v_ac**2) * dt
    total_energy = cumulative[-1]
    t1 = int(np.searchsorted(cumulative, 0.10 * total_energy)) * dt
    t2 = int(np.searchsorted(cumulative, 0.90 * total_energy)) * dt
    pd = float((t2 - t1) * 1.25)

    return {
        "freq":   freq,
        "pii":    pii,
        "pd":     pd,
        "v_pp":   float(voltage.max() - voltage.min()),
        "v_rms":  float(np.sqrt(np.mean(v_ac**2))),
        "v_mean": float(voltage.mean()),
    }

# ── GUI ───────────────────────────────────────────────────────────────────── #

class ScopeGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Scope — Single Channel")
        self.configure(bg=BG_DARK)
        self.minsize(900, 680)

        self._scope        = None
        self._rm           = None
        self._q            = queue.Queue()
        self._last_capture = None   # stores (voltage, dt, x_origin, results)

        # Apply dark ttk theme
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox",
            fieldbackground=BG_WIDGET, background=BG_WIDGET,
            foreground=FG_TEXT, selectbackground=BG_WIDGET,
            selectforeground=FG_TEXT, arrowcolor=FG_TEXT)
        style.map("TCombobox",
            fieldbackground=[("readonly", BG_WIDGET)],
            foreground=[("readonly", FG_TEXT)],
            background=[("readonly", BG_WIDGET)])

        self._build_ui()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────── #

    def _build_ui(self):
        self._build_conn_bar()

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Left panel: scope config
        left = tk.Frame(body, bg=BG_PANEL, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_config_panel(left)

        # Right: plot + results
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot(right)
        self._build_results(right)

        self._build_status_bar()

    # ── Connection bar ────────────────────────────────────────────────────── #

    def _build_conn_bar(self):
        bar = tk.Frame(self, bg=BG_PANEL, pady=4)
        bar.pack(fill=tk.X, padx=6, pady=(4, 0))

        tk.Label(bar, text="VISA:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(8, 4))

        self._visa = tk.StringVar(value="USB0::6833::1303::DS1ZE278M01562::0::INSTR")
        tk.Entry(bar, textvariable=self._visa, width=44,
                 bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief=tk.FLAT, font=("Courier", 10)).pack(side=tk.LEFT, padx=4)

        tk.Button(bar, text="Find", command=self._find_visa,
                  bg="#444444", fg=FG_TEXT, relief=tk.FLAT,
                  font=("Helvetica", 10)).pack(side=tk.LEFT, padx=4)

        self._btn_conn = tk.Button(bar, text="Connect", width=10,
            command=self._connect, bg="#336633", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#448844", font=("Helvetica", 10, "bold"))
        self._btn_conn.pack(side=tk.LEFT, padx=4)

        self._btn_disc = tk.Button(bar, text="Disconnect", width=10,
            command=self._disconnect, bg="#663333", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#884444", font=("Helvetica", 10, "bold"),
            state=tk.DISABLED)
        self._btn_disc.pack(side=tk.LEFT, padx=4)

        self._conn_lbl = tk.Label(bar, text="Not connected", bg=BG_PANEL,
            fg="#FF6666", font=("Helvetica", 9, "italic"))
        self._conn_lbl.pack(side=tk.LEFT, padx=12)

    # ── Config panel ──────────────────────────────────────────────────────── #

    def _build_config_panel(self, parent):
        pad = {"padx": 8, "pady": 3}

        def _lbl_entry(frame, row, text, var, width=10):
            tk.Label(frame, text=text, bg=BG_PANEL, fg=FG_TEXT,
                     font=("Helvetica", 9)).grid(row=row, column=0, sticky="w", **pad)
            tk.Entry(frame, textvariable=var, width=width,
                     bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
                     relief=tk.FLAT, font=("Courier", 9)).grid(
                     row=row, column=1, sticky="w", **pad)

        def _lbl_combo(frame, row, text, var, values, width=10):
            tk.Label(frame, text=text, bg=BG_PANEL, fg=FG_TEXT,
                     font=("Helvetica", 9)).grid(row=row, column=0, sticky="w", **pad)
            ttk.Combobox(frame, textvariable=var, values=values,
                         width=width, state="readonly",
                         font=("Helvetica", 9)).grid(
                         row=row, column=1, sticky="w", **pad)

        # ── Channel ──
        ch_frame = tk.LabelFrame(parent, text="Channel", bg=BG_PANEL, fg=FG_TEXT,
                                 font=("Helvetica", 10, "bold"), relief=tk.GROOVE,
                                 bd=1, padx=6, pady=4)
        ch_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(ch_frame, text="Channel:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9)).grid(row=0, column=0, sticky="w", **pad)
        self._chan = tk.StringVar(value="CHAN1")
        ttk.Combobox(ch_frame, textvariable=self._chan,
                     values=["CHAN1", "CHAN2", "CHAN3", "CHAN4"],
                     width=8, state="readonly",
                     font=("Helvetica", 9)).grid(row=0, column=1, sticky="w", **pad)

        self._ac_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ch_frame, text="AC coupling", variable=self._ac_var,
                       bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_WIDGET,
                       activebackground=BG_PANEL,
                       font=("Helvetica", 9)).grid(
                       row=1, column=0, columnspan=2, sticky="w", padx=8)

        # ── Scope Config ──
        sc_frame = tk.LabelFrame(parent, text="Scope Config", bg=BG_PANEL, fg=FG_TEXT,
                                  font=("Helvetica", 10, "bold"), relief=tk.GROOVE,
                                  bd=1, padx=6, pady=4)
        sc_frame.pack(fill=tk.X, padx=8, pady=4)

        self._sc_mode     = tk.StringVar(value="MAIN")
        self._sc_sdiv     = tk.StringVar(value="0.0000002")
        self._sc_delay    = tk.StringVar(value="0.0")
        self._sc_vdiv     = tk.StringVar(value="1.0")
        self._sc_acq_type = tk.StringVar(value="NORM")
        self._sc_mdep     = tk.StringVar(value="AUTO")
        self._sc_trig_src = tk.StringVar(value="CHAN1")
        self._sc_trig_lev = tk.StringVar(value="0.0")
        self._sc_holdoff  = tk.StringVar(value="1e-7")

        _lbl_combo(sc_frame, 0, "Mode",        self._sc_mode,     ["MAIN","XY","ROLL"])
        _lbl_entry(sc_frame, 1, "s/div",        self._sc_sdiv)
        _lbl_entry(sc_frame, 2, "Delay (s)",    self._sc_delay)
        _lbl_entry(sc_frame, 3, "V/div",        self._sc_vdiv)
        _lbl_combo(sc_frame, 4, "Acq type",     self._sc_acq_type, ["NORM","AVER","PEAK","HRES"], width=7)
        _lbl_combo(sc_frame, 5, "Mem depth",    self._sc_mdep,
                   ["AUTO","12000","120000","1200000","12000000"], width=9)
        _lbl_combo(sc_frame, 6, "Trig source",  self._sc_trig_src, ["CHAN1","CHAN2","CHAN3","CHAN4","EXT","ACL"])
        _lbl_entry(sc_frame, 7, "Trig level (V)",self._sc_trig_lev)
        _lbl_entry(sc_frame, 8, "Holdoff (s)",  self._sc_holdoff)

        self._btn_apply = tk.Button(sc_frame, text="Apply Config",
            command=self._apply_config, bg="#444466", fg=FG_TEXT,
            relief=tk.FLAT, activebackground="#6666aa",
            font=("Helvetica", 9), state=tk.DISABLED)
        self._btn_apply.grid(row=9, column=0, columnspan=2,
                             sticky="ew", padx=4, pady=(4, 2))

        # ── Capture buttons ──
        btn_frame = tk.Frame(parent, bg=BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        self._btn_capture = tk.Button(btn_frame, text="Capture",
            command=self._capture, bg="#445544", fg=FG_TEXT,
            relief=tk.FLAT, activebackground="#556655",
            font=("Helvetica", 10, "bold"), state=tk.DISABLED)
        self._btn_capture.pack(fill=tk.X, pady=2)

        self._btn_run = tk.Button(btn_frame, text="Run",
            command=lambda: self._send(":RUN", "Running."),
            bg="#336633", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#448844",
            font=("Helvetica", 10, "bold"), state=tk.DISABLED)
        self._btn_run.pack(fill=tk.X, pady=2)

        self._btn_stop = tk.Button(btn_frame, text="Stop",
            command=lambda: self._send(":STOP", "Stopped."),
            bg="#663333", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#884444",
            font=("Helvetica", 10, "bold"), state=tk.DISABLED)
        self._btn_stop.pack(fill=tk.X, pady=2)

        self._btn_save = tk.Button(btn_frame, text="Save Waveform",
            command=self._save_waveform, bg="#444444", fg=FG_TEXT,
            relief=tk.FLAT, activebackground="#666666",
            font=("Helvetica", 10, "bold"), state=tk.DISABLED)
        self._btn_save.pack(fill=tk.X, pady=2)

        self._all_btns = [self._btn_apply, self._btn_capture,
                          self._btn_run, self._btn_stop]

    # ── Waveform plot ─────────────────────────────────────────────────────── #

    def _build_plot(self, parent):
        self._fig = Figure(figsize=(7, 4), dpi=96, facecolor=BG_DARK)
        self._ax  = self._fig.add_subplot(111)
        self._style_ax()

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._trace, = self._ax.plot([], [], color="#FFD700", linewidth=0.8)

    def _style_ax(self):
        self._ax.set_facecolor(BG_DARK)
        self._ax.tick_params(colors=FG_DIM)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(ACCENT)
        self._ax.grid(True, color="#303030", linestyle="--", linewidth=0.5)
        self._ax.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        self._ax.set_ylabel("Voltage (V)", color=FG_DIM, fontsize=9)
        self._fig.tight_layout(pad=0.8)

    # ── Results bar ───────────────────────────────────────────────────────── #

    def _build_results(self, parent):
        bar = tk.Frame(parent, bg=BG_PANEL, pady=6)
        bar.pack(fill=tk.X)

        fields = [
            ("Vpp",   "v_pp",  "{:.4g} V"),
            ("Vrms",  "v_rms", "{:.4g} V"),
            ("Freq",  "freq",  "{:.5g} Hz"),
            ("PII",   "pii",   "{:.4e} J/cm²"),
            ("PD",    "pd",    "{:.4e} s"),
        ]

        self._result_labels = {}
        for label, key, _ in fields:
            cell = tk.Frame(bar, bg=BG_PANEL)
            cell.pack(side=tk.LEFT, padx=12)
            tk.Label(cell, text=label + ":", bg=BG_PANEL, fg=FG_DIM,
                     font=("Helvetica", 8)).pack()
            val = tk.Label(cell, text="—", bg=BG_PANEL, fg="#88DDFF",
                           font=("Courier", 10, "bold"), width=18)
            val.pack()
            self._result_labels[key] = (val, _)

    # ── Status bar ────────────────────────────────────────────────────────── #

    def _build_status_bar(self):
        self._status = tk.StringVar(value="Not connected.")
        bar = tk.Frame(self, bg="#111111", pady=2)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(bar, textvariable=self._status, bg="#111111", fg=FG_DIM,
                 font=("Helvetica", 9), anchor="w").pack(fill=tk.X, padx=8)

    # ── Connection ────────────────────────────────────────────────────────── #

    @staticmethod
    def _open_rm():
        """Return a ResourceManager, preferring NI-VISA, falling back to @py."""
        try:
            rm = pyvisa.ResourceManager()
            rm.list_resources()   # probe — raises if no backend
            return rm
        except Exception:
            return pyvisa.ResourceManager("@py")

    def _find_visa(self):
        try:
            rm = self._open_rm()
            resources = list(rm.list_resources())
            rm.close()
        except Exception:
            resources = []
        if not resources:
            import tkinter.messagebox as mb
            mb.showinfo("Find VISA", "No VISA resources found.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Select VISA Resource")
        dlg.resizable(False, False)
        tk.Label(dlg, text="Select a resource:").pack(padx=12, pady=(10, 4))
        lb = tk.Listbox(dlg, listvariable=tk.StringVar(value=resources),
                        selectmode="single", width=56, height=min(len(resources), 10))
        lb.pack(padx=12, pady=4)
        lb.select_set(0)
        def _select():
            sel = lb.curselection()
            if sel:
                self._visa.set(resources[sel[0]])
            dlg.destroy()
        tk.Button(dlg, text="Select", command=_select).pack(pady=(4, 10))
        dlg.grab_set()

    def _connect(self):
        self._btn_conn.config(state=tk.DISABLED)
        self._status.set("Connecting…")
        threading.Thread(target=self._thread_connect, daemon=True).start()

    def _thread_connect(self):
        try:
            self._rm    = self._open_rm()
            self._scope = self._rm.open_resource(self._visa.get())
            self._scope.timeout    = 3000
            self._scope.chunk_size = 1024 * 1024
            idn = self._scope.query("*IDN?").strip()
            self._q.put({"type": "connected", "idn": idn})
        except Exception as exc:
            self._q.put({"type": "error", "text": str(exc)})
            self._q.put({"type": "conn_reenable"})

    def _disconnect(self):
        if self._scope:
            try:
                self._scope.close()
            except Exception:
                pass
            self._scope = None
        if self._rm:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None
        self._btn_conn.config(state=tk.NORMAL)
        self._btn_disc.config(state=tk.DISABLED)
        for b in self._all_btns:
            b.config(state=tk.DISABLED)
        self._conn_lbl.config(text="Not connected", fg="#FF6666")
        self._status.set("Disconnected.")

    def _on_close(self):
        self._disconnect()
        self.destroy()

    # ── Scope commands ────────────────────────────────────────────────────── #

    def _send(self, cmd, msg="Done."):
        def _run():
            try:
                self._scope.write(cmd)
                self._q.put({"type": "status", "text": msg})
            except Exception as exc:
                self._q.put({"type": "error", "text": str(exc)})
        threading.Thread(target=_run, daemon=True).start()

    def _apply_config(self):
        def _run():
            try:
                ch = self._chan.get()
                self._scope.write(f":TIM:MODE {self._sc_mode.get()}")
                self._scope.write(f":TIM:SCAL {self._sc_sdiv.get()}")
                self._scope.write(f":TIM:OFFS {self._sc_delay.get()}")
                self._scope.write(f":{ch}:SCAL {self._sc_vdiv.get()}")
                coup = "AC" if self._ac_var.get() else "DC"
                self._scope.write(f":{ch}:COUP {coup}")
                self._scope.write(f":ACQ:TYPE {self._sc_acq_type.get()}")
                self._scope.write(f":ACQ:MDEP {self._sc_mdep.get()}")
                self._scope.write(":TRIG:MODE EDGE")
                self._scope.write(f":TRIG:EDGE:SOUR {self._sc_trig_src.get()}")
                self._scope.write(f":TRIG:EDGE:LEV {self._sc_trig_lev.get()}")
                self._scope.write(f":TRIG:HOLD {self._sc_holdoff.get()}")
                self._q.put({"type": "status", "text": "Config applied."})
            except Exception as exc:
                self._q.put({"type": "error", "text": str(exc)})
        threading.Thread(target=_run, daemon=True).start()

    def _capture(self):
        self._btn_capture.config(state=tk.DISABLED)
        self._status.set("Capturing…")
        threading.Thread(target=self._thread_capture, daemon=True).start()

    def _thread_capture(self):
        try:
            ch = self._chan.get()
            coup = "AC" if self._ac_var.get() else "DC"
            self._scope.write(f":{ch}:COUP {coup}")
            self._scope.write(f":WAV:SOUR {ch}")
            self._scope.write(":WAV:MODE NORM")
            self._scope.write(":WAV:FORM BYTE")
            self._scope.write(":WAV:STAR 1")
            self._scope.write(":WAV:STOP 1200")

            preamble    = self._scope.query(":WAV:PRE?").strip().split(",")
            x_increment = float(preamble[4])
            x_origin    = float(preamble[5])
            y_increment = float(preamble[7])
            y_origin    = float(preamble[8])
            y_reference = float(preamble[9])

            self._scope.write(":STOP")
            time.sleep(0.2)
            self._scope.write(":WAV:DATA?")
            time.sleep(0.2)
            raw = self._scope.read_raw()
            while len(raw) < 10:
                raw += self._scope.read_raw()

            n_digits   = int(chr(raw[1]))
            n_data     = int(raw[2:2 + n_digits])
            data_start = 2 + n_digits
            while len(raw) < data_start + n_data:
                raw += self._scope.read_raw()

            samples = np.frombuffer(raw[data_start:data_start + n_data], dtype=np.uint8)
            voltage = (samples - y_reference) * y_increment + y_origin
            t       = x_origin + np.arange(len(samples)) * x_increment

            self._scope.write(":RUN")

            results = process_waveform(voltage, x_increment)
            self._q.put({"type": "capture", "t": t, "v": voltage,
                         "results": results, "dt": x_increment,
                         "x_origin": x_origin})

        except Exception as exc:
            self._q.put({"type": "error", "text": str(exc)})
            self._q.put({"type": "capture_done"})

    # ── Queue polling ─────────────────────────────────────────────────────── #

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                t   = msg["type"]

                if t == "connected":
                    self._conn_lbl.config(text=f"Connected: {msg['idn']}", fg="#66FF88")
                    self._btn_conn.config(state=tk.DISABLED)
                    self._btn_disc.config(state=tk.NORMAL)
                    for b in self._all_btns:
                        b.config(state=tk.NORMAL)
                    self._status.set(f"Connected: {msg['idn']}")

                elif t == "conn_reenable":
                    self._btn_conn.config(state=tk.NORMAL)

                elif t == "status":
                    self._status.set(msg["text"])

                elif t == "error":
                    self._status.set(f"Error: {msg['text']}")

                elif t == "capture":
                    self._last_capture = (msg["v"], msg["dt"], msg["x_origin"], msg["results"])
                    self._update_plot(msg["t"], msg["v"])
                    self._update_results(msg["results"])
                    self._btn_capture.config(state=tk.NORMAL)
                    self._btn_save.config(state=tk.NORMAL)
                    self._status.set("Capture complete.")

                elif t == "capture_done":
                    self._btn_capture.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _save_waveform(self):
        if self._last_capture is None:
            return
        voltage, dt, x_origin, results = self._last_capture

        default_name = "waveform_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        path = filedialog.asksaveasfilename(
            title="Save Waveform",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        with open(path, "w") as f:
            # Header: scope settings and computed results
            f.write(f"# Saved:      {datetime.datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"# Channel:    {self._chan.get()}\n")
            f.write(f"# Coupling:   {'AC' if self._ac_var.get() else 'DC'}\n")
            f.write(f"# s/div:      {self._sc_sdiv.get()}\n")
            f.write(f"# V/div:      {self._sc_vdiv.get()}\n")
            f.write(f"# Delay (s):  {self._sc_delay.get()}\n")
            f.write(f"# Acq type:   {self._sc_acq_type.get()}\n")
            f.write(f"# Mem depth:  {self._sc_mdep.get()}\n")
            f.write(f"# Trig src:   {self._sc_trig_src.get()}\n")
            f.write(f"# Trig lev:   {self._sc_trig_lev.get()}\n")
            f.write(f"# Holdoff:    {self._sc_holdoff.get()}\n")
            f.write(f"# Samples:    {len(voltage)}\n")
            f.write(f"# Vpp:        {results['v_pp']:.6g} V\n")
            f.write(f"# Vrms:       {results['v_rms']:.6g} V\n")
            f.write(f"# Vmean:      {results['v_mean']:.6g} V\n")
            f.write(f"# Freq:       {results['freq']:.6g} Hz\n")
            f.write(f"# PII:        {results['pii']:.6e} J/cm2\n")
            f.write(f"# PD:         {results['pd']:.6e} s\n")
            # Data block (compatible with analyze_waveform.py)
            f.write(f"{dt}\n")
            f.write(f"{x_origin}\n")
            for v in voltage:
                f.write(f"{v}\n")

        self._status.set(f"Saved: {os.path.basename(path)}")

    def _update_plot(self, t, v):
        self._trace.set_data(t, v)
        self._ax.relim()
        self._ax.autoscale_view()
        self._style_ax()
        self._canvas.draw_idle()

    def _update_results(self, r):
        fmt_map = {
            "v_pp":  "{:.4g} V",
            "v_rms": "{:.4g} V",
            "freq":  "{:.5g} Hz",
            "pii":   "{:.4e} J/cm²",
            "pd":    "{:.4e} s",
        }
        for key, (lbl, _) in self._result_labels.items():
            val = r.get(key)
            if val is not None:
                lbl.config(text=fmt_map[key].format(val))


if __name__ == "__main__":
    ScopeGUI().mainloop()
