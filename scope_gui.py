"""
Rigol DS1000Z Oscilloscope GUI
Standalone tkinter + matplotlib frontend using pyvisa.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import numpy as np

import pyvisa

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_COLORS = {
    1: "#FFD700",   # gold
    2: "#00CC44",   # green
    3: "#4488FF",   # blue
    4: "#FF3333",   # red
}

BG_DARK    = "#1a1a1a"
BG_PANEL   = "#2a2a2a"
BG_WIDGET  = "#333333"
FG_TEXT    = "#e0e0e0"
FG_DIM     = "#888888"
ACCENT     = "#555555"

INVALID_MEAS = 9.9e+37
MEAS_ITEMS  = ["FREQ", "PER", "VPP", "VRMS", "VAVG", "VMAX", "VMIN"]
MEAS_LABELS = ["Freq", "Period", "Vpp", "Vrms", "Vmean", "Vmax", "Vmin"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_voltage(v):
    """Format a voltage value with automatic unit selection."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1.0:
        return f"{v:.4g} V"
    elif abs(v) >= 1e-3:
        return f"{v * 1e3:.4g} mV"
    else:
        return f"{v * 1e6:.4g} µV"


def fmt_time(s):
    """Format a time value with automatic unit selection."""
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "—"
    if abs(s) >= 1.0:
        return f"{s:.4g} s"
    elif abs(s) >= 1e-3:
        return f"{s * 1e3:.4g} ms"
    elif abs(s) >= 1e-6:
        return f"{s * 1e6:.4g} µs"
    else:
        return f"{s * 1e9:.4g} ns"


def fmt_freq(hz):
    """Format a frequency value."""
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return "—"
    if hz >= 1e6:
        return f"{hz / 1e6:.4g} MHz"
    elif hz >= 1e3:
        return f"{hz / 1e3:.4g} kHz"
    else:
        return f"{hz:.4g} Hz"


def parse_ieee_block(raw_bytes):
    """
    Parse IEEE 488.2 definite-length binary block.
    raw_bytes[0]  == ord('#')
    raw_bytes[1]  == n_digits (ASCII digit)
    raw_bytes[2:2+n_digits] == decimal byte-count string
    Returns (payload_bytes, byte_count).
    """
    if len(raw_bytes) < 2 or chr(raw_bytes[0]) != '#':
        raise ValueError("Not an IEEE 488.2 block header")
    n_digits = int(chr(raw_bytes[1]))
    if n_digits == 0:
        raise ValueError("Indefinite-length IEEE block not supported")
    byte_count = int(raw_bytes[2:2 + n_digits])
    payload_start = 2 + n_digits
    return raw_bytes[payload_start: payload_start + byte_count], byte_count


# ---------------------------------------------------------------------------
# Scope worker  (all I/O runs in daemon threads; results pushed via queue)
# ---------------------------------------------------------------------------

class ScopeWorker:
    """Thin wrapper around a pyvisa resource; called from worker threads."""

    def __init__(self):
        self.rm    = None
        self.scope = None

    # ------------------------------------------------------------------
    def connect(self, address):
        self.rm = pyvisa.ResourceManager("@py")
        self.scope = self.rm.open_resource(address)
        self.scope.timeout    = 3000
        self.scope.chunk_size = 1024 * 1024
        idn = self.scope.query("*IDN?").strip()
        return idn

    def disconnect(self):
        try:
            if self.scope:
                self.scope.close()
        except Exception:
            pass
        try:
            if self.rm:
                self.rm.close()
        except Exception:
            pass
        self.scope = None
        self.rm    = None

    def write(self, cmd):
        self.scope.write(cmd)

    def query(self, cmd):
        return self.scope.query(cmd).strip()

    # ------------------------------------------------------------------
    def fetch_waveform(self, channel):
        """
        Fetch waveform for *channel* (1-4).
        Returns (t_array, v_array) or raises.
        """
        src = f"CHAN{channel}"
        self.scope.write(f":WAV:SOUR {src}")
        self.scope.write(":WAV:MODE NORM")
        self.scope.write(":WAV:FORM BYTE")
        self.scope.write(":WAV:STAR 1")
        self.scope.write(":WAV:STOP 1200")

        # Preamble  (DS1000Z fields):
        # 0=format, 1=type, 2=points, 3=count,
        # 4=x_inc,  5=x_orig, 6=x_ref,
        # 7=y_inc,  8=y_orig, 9=y_ref
        preamble    = self.scope.query(":WAV:PRE?").strip().split(",")
        x_increment = float(preamble[4])
        x_origin    = float(preamble[5])
        y_increment = float(preamble[7])
        y_origin    = float(preamble[8])
        y_reference = float(preamble[9])

        # Request data then accumulate chunks
        self.scope.write(":WAV:DATA?")
        raw = b""
        while True:
            chunk = self.scope.read_raw()
            raw  += chunk
            if len(chunk) < self.scope.chunk_size:
                break

        payload, _ = parse_ieee_block(raw)
        raw_samples = np.frombuffer(payload, dtype=np.uint8).astype(np.float64)

        voltage = (raw_samples - y_reference) * y_increment + y_origin
        t       = x_origin + np.arange(len(raw_samples)) * x_increment
        return t, voltage

    # ------------------------------------------------------------------
    def read_settings(self):
        """
        Query the scope for current settings.
        Returns a dict of string values keyed by setting name.
        """
        s = {}
        s["tim_scal"] = self.query(":TIM:SCAL?")
        for ch in range(1, 5):
            s[f"ch{ch}_scal"] = self.query(f":CHAN{ch}:SCAL?")
            s[f"ch{ch}_offs"] = self.query(f":CHAN{ch}:OFFS?")
            s[f"ch{ch}_coup"] = self.query(f":CHAN{ch}:COUP?")
            s[f"ch{ch}_disp"] = self.query(f":CHAN{ch}:DISP?")
        s["trig_sour"] = self.query(":TRIG:EDGE:SOUR?")
        s["trig_slop"] = self.query(":TRIG:EDGE:SLOP?")
        s["trig_lev"]  = self.query(":TRIG:EDGE:LEV?")
        s["trig_swe"]  = self.query(":TRIG:SWE?")
        return s

    # ------------------------------------------------------------------
    def read_measurements(self, channel):
        """
        Read all measurement items for *channel*.
        Returns dict {item_name: raw_string}.
        """
        results = {}
        for item in MEAS_ITEMS:
            try:
                val = self.query(f":MEAS:ITEM? {item},CHAN{channel}").strip()
                results[item] = val
            except Exception:
                results[item] = str(INVALID_MEAS)
        return results


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class OscilloscopeGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Rigol DS1000Z — Scope GUI")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(1100, 720)

        self.worker    = ScopeWorker()
        self.connected = False
        self.q         = queue.Queue()

        # Auto-capture state
        self.auto_var  = tk.BooleanVar(value=False)
        self._auto_thread = None

        # Per-channel enabled flags (CH1 on by default)
        self.ch_enabled = [tk.BooleanVar(value=(i == 0)) for i in range(4)]

        # ch_widgets is a list-of-dicts, index 0 = CH1
        self.ch_widgets = []

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._build_visa_bar()

        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left sidebar
        sidebar = tk.Frame(content, bg=BG_PANEL, width=230, relief=tk.FLAT)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Right area: control buttons + plot + measurements
        right = tk.Frame(content, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_control_buttons(right)
        self._build_plot(right)
        self._build_measurements(right)
        self._build_trigger_bar(right)

        self._build_status_bar()

    # ---- VISA bar --------------------------------------------------------

    def _build_visa_bar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, pady=4)
        bar.pack(fill=tk.X, padx=4, pady=(4, 0))

        tk.Label(bar, text="VISA Address:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(8, 4))

        self.visa_addr = tk.StringVar(value="USB0::6833::1303::DS1ZE278M01562::0::INSTR")
        tk.Entry(bar, textvariable=self.visa_addr, width=38,
                 bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief=tk.FLAT, font=("Courier", 10)).pack(side=tk.LEFT, padx=4)

        self.btn_connect = tk.Button(
            bar, text="Connect", width=10, command=self._on_connect,
            bg="#336633", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#448844", font=("Helvetica", 10, "bold"))
        self.btn_connect.pack(side=tk.LEFT, padx=4)

        self.btn_disconnect = tk.Button(
            bar, text="Disconnect", width=10, command=self._on_disconnect,
            bg="#663333", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#884444", font=("Helvetica", 10, "bold"),
            state=tk.DISABLED)
        self.btn_disconnect.pack(side=tk.LEFT, padx=4)

        tk.Button(
            bar, text="Find Devices", command=self._find_devices,
            bg="#334455", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#445566", font=("Helvetica", 9)
        ).pack(side=tk.LEFT, padx=4)

        self.conn_label = tk.Label(
            bar, text="Not connected", bg=BG_PANEL, fg="#FF6666",
            font=("Helvetica", 9, "italic"))
        self.conn_label.pack(side=tk.LEFT, padx=12)

    # ---- Sidebar ---------------------------------------------------------

    def _build_sidebar(self, parent):
        canvas    = tk.Canvas(parent, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        inner     = tk.Frame(canvas, bg=BG_PANEL)

        inner.bind("<Configure>",
                   lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Per-channel sections
        for ch in range(1, 5):
            self.ch_widgets.append(self._build_channel_section(inner, ch))

        tk.Frame(inner, bg=ACCENT, height=2).pack(fill=tk.X, padx=6, pady=6)
        self._build_timebase_section(inner)
        tk.Frame(inner, bg=ACCENT, height=2).pack(fill=tk.X, padx=6, pady=6)
        self._build_trigger_section(inner)

    def _build_channel_section(self, parent, ch):
        color = CHANNEL_COLORS[ch]
        frame = tk.LabelFrame(
            parent, text=f"  CH{ch}  ", fg=color, bg=BG_PANEL,
            font=("Helvetica", 10, "bold"), relief=tk.GROOVE, bd=1,
            padx=6, pady=4)
        frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        w = {}

        # Enable checkbox
        row0 = tk.Frame(frame, bg=BG_PANEL)
        row0.pack(fill=tk.X)
        cb = tk.Checkbutton(
            row0, text="Enable", variable=self.ch_enabled[ch - 1],
            bg=BG_PANEL, fg=color, selectcolor=BG_WIDGET,
            activebackground=BG_PANEL, font=("Helvetica", 9),
            state=tk.DISABLED)
        cb.pack(side=tk.LEFT)
        w["enable_cb"] = cb

        # V/div
        row1 = tk.Frame(frame, bg=BG_PANEL)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="V/div:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9), width=8, anchor="w").pack(side=tk.LEFT)
        vdiv_var   = tk.StringVar(value="1.0")
        vdiv_entry = tk.Entry(
            row1, textvariable=vdiv_var, width=8,
            bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
            relief=tk.FLAT, font=("Courier", 9), state=tk.DISABLED)
        vdiv_entry.pack(side=tk.LEFT, padx=2)
        w["vdiv_var"]   = vdiv_var
        w["vdiv_entry"] = vdiv_entry

        # Offset
        row2 = tk.Frame(frame, bg=BG_PANEL)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="Offset V:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9), width=8, anchor="w").pack(side=tk.LEFT)
        offs_var   = tk.StringVar(value="0.0")
        offs_entry = tk.Entry(
            row2, textvariable=offs_var, width=8,
            bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
            relief=tk.FLAT, font=("Courier", 9), state=tk.DISABLED)
        offs_entry.pack(side=tk.LEFT, padx=2)
        w["offs_var"]   = offs_var
        w["offs_entry"] = offs_entry

        # Coupling
        row3 = tk.Frame(frame, bg=BG_PANEL)
        row3.pack(fill=tk.X, pady=2)
        tk.Label(row3, text="Coupling:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9), width=8, anchor="w").pack(side=tk.LEFT)
        coup_var  = tk.StringVar(value="DC")
        coup_menu = ttk.Combobox(
            row3, textvariable=coup_var, values=["DC", "AC", "GND"],
            width=6, state=tk.DISABLED, font=("Helvetica", 9))
        coup_menu.pack(side=tk.LEFT, padx=2)
        w["coup_var"]  = coup_var
        w["coup_menu"] = coup_menu

        # Set button
        row4 = tk.Frame(frame, bg=BG_PANEL)
        row4.pack(fill=tk.X, pady=(4, 2))
        set_btn = tk.Button(
            row4, text=f"Set CH{ch}",
            command=lambda c=ch: self._apply_channel(c),
            bg="#444466", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#6666aa", font=("Helvetica", 9),
            state=tk.DISABLED)
        set_btn.pack(fill=tk.X)
        w["set_btn"] = set_btn

        return w

    def _build_timebase_section(self, parent):
        frame = tk.LabelFrame(
            parent, text="  Timebase  ", fg=FG_TEXT, bg=BG_PANEL,
            font=("Helvetica", 10, "bold"), relief=tk.GROOVE, bd=1,
            padx=6, pady=4)
        frame.pack(fill=tk.X, padx=6)

        row = tk.Frame(frame, bg=BG_PANEL)
        row.pack(fill=tk.X)
        tk.Label(row, text="s/div:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9), width=8, anchor="w").pack(side=tk.LEFT)
        self.tim_var   = tk.StringVar(value="0.001")
        self.tim_entry = tk.Entry(
            row, textvariable=self.tim_var, width=10,
            bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
            relief=tk.FLAT, font=("Courier", 9), state=tk.DISABLED)
        self.tim_entry.pack(side=tk.LEFT, padx=2)

        self.tim_set_btn = tk.Button(
            frame, text="Set Timebase", command=self._apply_timebase,
            bg="#444466", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#6666aa", font=("Helvetica", 9),
            state=tk.DISABLED)
        self.tim_set_btn.pack(fill=tk.X, pady=(4, 0))

    def _build_trigger_section(self, parent):
        # Trigger controls are in the always-visible bar below the plot.
        tk.Label(parent, text="↓ Trigger controls below plot ↓",
                 bg=BG_PANEL, fg=FG_DIM,
                 font=("Helvetica", 8, "italic")).pack(pady=4)

    # ---- Control buttons -------------------------------------------------

    def _build_control_buttons(self, parent):
        bar = tk.Frame(parent, bg=BG_DARK, pady=4)
        bar.pack(fill=tk.X)

        self._ctrl_buttons = []

        btn_specs = [
            ("Run",    "#336633", "#448844", self._on_run),
            ("Stop",   "#663333", "#884444", self._on_stop),
            ("Single", "#334466", "#445588", self._on_single),
            ("Capture","#445544", "#556655", self._on_capture),
        ]
        for label, bg, abg, cmd in btn_specs:
            b = tk.Button(
                bar, text=label, width=9, command=cmd,
                bg=bg, fg=FG_TEXT, relief=tk.FLAT,
                activebackground=abg, font=("Helvetica", 10, "bold"),
                state=tk.DISABLED)
            b.pack(side=tk.LEFT, padx=4)
            self._ctrl_buttons.append(b)

        # Auto checkbox
        self.auto_cb = tk.Checkbutton(
            bar, text="Auto", variable=self.auto_var,
            command=self._on_auto_toggle,
            bg=BG_DARK, fg=FG_TEXT, selectcolor=BG_WIDGET,
            activebackground=BG_DARK, font=("Helvetica", 10),
            state=tk.DISABLED)
        self.auto_cb.pack(side=tk.LEFT, padx=8)
        self._ctrl_buttons.append(self.auto_cb)

        # Read from scope
        self.read_btn = tk.Button(
            bar, text="Read from scope", command=self._on_read_settings,
            bg="#444444", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#666666", font=("Helvetica", 10),
            state=tk.DISABLED)
        self.read_btn.pack(side=tk.LEFT, padx=4)
        self._ctrl_buttons.append(self.read_btn)

    # ---- Plot ------------------------------------------------------------

    def _build_plot(self, parent):
        plot_frame = tk.Frame(parent, bg=BG_DARK)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 4.5), dpi=96, facecolor=BG_DARK)
        self.ax  = self.fig.add_subplot(111)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # One trace per channel (hidden until data arrives)
        self._traces = {}
        for ch in range(1, 5):
            line, = self.ax.plot([], [], color=CHANNEL_COLORS[ch],
                                 linewidth=0.8, label=f"CH{ch}")
            self._traces[ch] = line

        self.ax.legend(
            loc="upper right", facecolor=BG_PANEL,
            edgecolor=ACCENT, labelcolor=FG_TEXT, fontsize=8)

    def _style_axes(self):
        self.ax.set_facecolor(BG_DARK)
        self.ax.tick_params(colors=FG_DIM)
        self.ax.xaxis.label.set_color(FG_DIM)
        self.ax.yaxis.label.set_color(FG_DIM)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(ACCENT)
        self.ax.grid(True, color="#303030", linestyle="--", linewidth=0.5)
        self.ax.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        self.ax.set_ylabel("Voltage (V)", color=FG_DIM, fontsize=9)
        self.fig.tight_layout(pad=0.8)

    # ---- Measurements bar -----------------------------------------------

    def _build_measurements(self, parent):
        bar = tk.Frame(parent, bg=BG_PANEL, pady=4)
        bar.pack(fill=tk.X, pady=(2, 0))

        tk.Label(bar, text="Measurements:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(8, 10))

        self._meas_labels = {}
        for item, label in zip(MEAS_ITEMS, MEAS_LABELS):
            cell = tk.Frame(bar, bg=BG_PANEL)
            cell.pack(side=tk.LEFT, padx=6)
            tk.Label(cell, text=label + ":", bg=BG_PANEL, fg=FG_DIM,
                     font=("Helvetica", 8)).pack(side=tk.TOP)
            val_lbl = tk.Label(cell, text="—", bg=BG_PANEL, fg="#88DDFF",
                               font=("Courier", 9, "bold"), width=10)
            val_lbl.pack(side=tk.TOP)
            self._meas_labels[item] = val_lbl

    # ---- Trigger bar (always visible, below measurements) ----------------

    def _build_trigger_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_PANEL, pady=4)
        bar.pack(fill=tk.X, pady=(2, 0))

        tk.Label(bar, text="Trigger:", bg=BG_PANEL, fg=FG_TEXT,
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(8, 6))

        def _lbl(text):
            tk.Label(bar, text=text, bg=BG_PANEL, fg=FG_DIM,
                     font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(6, 1))

        _lbl("Source")
        self.trig_sour_var  = tk.StringVar(value="CHAN1")
        self.trig_sour_menu = ttk.Combobox(
            bar, textvariable=self.trig_sour_var,
            values=["CHAN1", "CHAN2", "CHAN3", "CHAN4", "EXT", "ACL"],
            width=6, state=tk.DISABLED, font=("Helvetica", 9))
        self.trig_sour_menu.pack(side=tk.LEFT, padx=2)

        _lbl("Slope")
        self.trig_slop_var  = tk.StringVar(value="POS")
        self.trig_slop_menu = ttk.Combobox(
            bar, textvariable=self.trig_slop_var,
            values=["POS", "NEG", "RFAL"],
            width=5, state=tk.DISABLED, font=("Helvetica", 9))
        self.trig_slop_menu.pack(side=tk.LEFT, padx=2)

        _lbl("Level (V)")
        self.trig_lev_var   = tk.StringVar(value="0.0")
        self.trig_lev_entry = tk.Entry(
            bar, textvariable=self.trig_lev_var, width=7,
            bg=BG_WIDGET, fg=FG_TEXT, insertbackground=FG_TEXT,
            relief=tk.FLAT, font=("Courier", 9), state=tk.DISABLED)
        self.trig_lev_entry.pack(side=tk.LEFT, padx=2)

        _lbl("Sweep")
        self.trig_swe_var  = tk.StringVar(value="AUTO")
        self.trig_swe_menu = ttk.Combobox(
            bar, textvariable=self.trig_swe_var,
            values=["AUTO", "NORM", "SING"],
            width=5, state=tk.DISABLED, font=("Helvetica", 9))
        self.trig_swe_menu.pack(side=tk.LEFT, padx=2)

        self.trig_apply_btn = tk.Button(
            bar, text="Apply", command=self._apply_trigger,
            bg="#444466", fg=FG_TEXT, relief=tk.FLAT,
            activebackground="#6666aa", font=("Helvetica", 9),
            state=tk.DISABLED)
        self.trig_apply_btn.pack(side=tk.LEFT, padx=(6, 2))

    # ---- Status bar ------------------------------------------------------

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready.")
        bar = tk.Frame(self.root, bg="#111111", pady=2)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(bar, textvariable=self.status_var, bg="#111111", fg=FG_DIM,
                 font=("Helvetica", 9), anchor="w").pack(fill=tk.X, padx=8)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _enable_controls(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        ro    = "readonly" if enabled else tk.DISABLED

        for b in self._ctrl_buttons:
            try:
                b.configure(state=state)
            except Exception:
                pass

        for w in self.ch_widgets:
            w["enable_cb"].configure(state=state)
            w["vdiv_entry"].configure(state=state)
            w["offs_entry"].configure(state=state)
            w["coup_menu"].configure(state=ro)
            w["set_btn"].configure(state=state)

        self.tim_entry.configure(state=state)
        self.tim_set_btn.configure(state=state)
        self.trig_lev_entry.configure(state=state)
        self.trig_apply_btn.configure(state=state)
        self.trig_sour_menu.configure(state=ro)
        self.trig_slop_menu.configure(state=ro)
        self.trig_swe_menu.configure(state=ro)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _find_devices(self):
        self._set_status("Scanning for VISA devices…")
        def _scan():
            try:
                rm = pyvisa.ResourceManager("@py")
                resources = rm.list_resources()
                self.q.put(("devices", resources))
            except Exception as exc:
                self.q.put(("error", f"Scan failed: {exc}"))
        threading.Thread(target=_scan, daemon=True).start()

    def _on_connect(self):
        addr = self.visa_addr.get().strip()
        if not addr:
            messagebox.showerror("Error", "Enter a VISA address.")
            return
        self._set_status(f"Connecting to {addr} …")
        self.btn_connect.configure(state=tk.DISABLED)
        threading.Thread(target=self._thread_connect, args=(addr,),
                         daemon=True).start()

    def _thread_connect(self, addr):
        try:
            idn = self.worker.connect(addr)
            self.q.put(("connected", idn))
        except Exception as exc:
            self.q.put(("error", f"Connect failed: {exc}"))
            self.q.put(("connect_btn_reenable", None))

    def _on_disconnect(self):
        self.auto_var.set(False)
        self._set_status("Disconnecting…")
        threading.Thread(target=self._thread_disconnect, daemon=True).start()

    def _thread_disconnect(self):
        try:
            self.worker.disconnect()
            self.q.put(("disconnected", None))
        except Exception as exc:
            self.q.put(("error", f"Disconnect error: {exc}"))

    def _on_run(self):
        self._send_command(":RUN", "Running.")

    def _on_stop(self):
        self._send_command(":STOP", "Stopped.")

    def _on_single(self):
        self._send_command(":SING", "Single trigger armed.")

    def _on_capture(self):
        self._set_status("Capturing waveforms…")
        threading.Thread(target=self._thread_capture, daemon=True).start()

    def _on_auto_toggle(self):
        if self.auto_var.get():
            self._set_status("Auto capture enabled.")
            self._auto_thread = threading.Thread(
                target=self._auto_loop, daemon=True)
            self._auto_thread.start()
        else:
            self._set_status("Auto capture disabled.")

    def _on_read_settings(self):
        self._set_status("Reading settings from scope…")
        threading.Thread(target=self._thread_read_settings,
                         daemon=True).start()

    # ---- Apply channel settings -----------------------------------------

    def _apply_channel(self, ch):
        idx     = ch - 1
        w       = self.ch_widgets[idx]
        enabled = self.ch_enabled[idx].get()
        vdiv    = w["vdiv_var"].get().strip()
        offs    = w["offs_var"].get().strip()
        coup    = w["coup_var"].get().strip()

        def _send():
            try:
                disp = "ON" if enabled else "OFF"
                self.worker.write(f":CHAN{ch}:DISP {disp}")
                self.worker.write(f":CHAN{ch}:SCAL {vdiv}")
                self.worker.write(f":CHAN{ch}:OFFS {offs}")
                self.worker.write(f":CHAN{ch}:COUP {coup}")
                self.q.put(("status", f"CH{ch} settings applied."))
            except Exception as exc:
                self.q.put(("error", f"CH{ch} set error: {exc}"))

        threading.Thread(target=_send, daemon=True).start()

    def _apply_timebase(self):
        s_div = self.tim_var.get().strip()

        def _send():
            try:
                self.worker.write(f":TIM:SCAL {s_div}")
                self.q.put(("status", f"Timebase set to {s_div} s/div."))
            except Exception as exc:
                self.q.put(("error", f"Timebase set error: {exc}"))

        threading.Thread(target=_send, daemon=True).start()

    def _apply_trigger(self):
        sour = self.trig_sour_var.get()
        slop = self.trig_slop_var.get()
        lev  = self.trig_lev_var.get().strip()
        swe  = self.trig_swe_var.get()

        def _send():
            try:
                self.worker.write(":TRIG:MODE EDGE")
                self.worker.write(f":TRIG:EDGE:SOUR {sour}")
                self.worker.write(f":TRIG:EDGE:SLOP {slop}")
                self.worker.write(f":TRIG:EDGE:LEV {lev}")
                self.worker.write(f":TRIG:SWE {swe}")
                self.q.put(("status", "Trigger settings applied."))
            except Exception as exc:
                self.q.put(("error", f"Trigger set error: {exc}"))

        threading.Thread(target=_send, daemon=True).start()

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def _send_command(self, cmd, ok_msg="Done."):
        def _send():
            try:
                self.worker.write(cmd)
                self.q.put(("status", ok_msg))
            except Exception as exc:
                self.q.put(("error", str(exc)))

        threading.Thread(target=_send, daemon=True).start()

    def _thread_capture(self):
        """Capture waveforms for all enabled channels, then read measurements."""
        try:
            results = {}
            for ch in range(1, 5):
                if self.ch_enabled[ch - 1].get():
                    try:
                        t_arr, v_arr = self.worker.fetch_waveform(ch)
                        results[ch]  = (t_arr, v_arr)
                    except Exception as exc:
                        self.q.put(("warning", f"CH{ch} fetch error: {exc}"))

            # Measurements from the first active channel
            meas_ch = next(
                (ch for ch in range(1, 5) if self.ch_enabled[ch - 1].get()),
                None)
            meas = {}
            if meas_ch is not None:
                try:
                    meas = self.worker.read_measurements(meas_ch)
                except Exception:
                    pass

            self.q.put(("waveforms", (results, meas)))
        except Exception as exc:
            self.q.put(("error", f"Capture error: {exc}"))

    def _auto_loop(self):
        """Continuously capture while auto_var is set; sleeps ~0.5 s between captures."""
        while self.auto_var.get() and self.connected:
            self._thread_capture()
            # Sleep in small steps so we can exit cleanly
            elapsed = 0.0
            while elapsed < 0.5:
                if not self.auto_var.get() or not self.connected:
                    return
                time.sleep(0.05)
                elapsed += 0.05

    def _thread_read_settings(self):
        try:
            settings = self.worker.read_settings()
            self.q.put(("settings", settings))
        except Exception as exc:
            self.q.put(("error", f"Read settings error: {exc}"))

    # ------------------------------------------------------------------
    # Queue polling  (main thread only)
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.q.get_nowait()
                self._handle_message(msg_type, payload)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _handle_message(self, msg_type, payload):
        if msg_type == "connected":
            self.connected = True
            self.conn_label.configure(
                text=f"Connected: {payload}", fg="#66FF88")
            self.btn_connect.configure(state=tk.DISABLED)
            self.btn_disconnect.configure(state=tk.NORMAL)
            self._enable_controls(True)
            self._set_status(f"Connected: {payload}")

        elif msg_type == "disconnected":
            self.connected = False
            self.conn_label.configure(text="Not connected", fg="#FF6666")
            self.btn_connect.configure(state=tk.NORMAL)
            self.btn_disconnect.configure(state=tk.DISABLED)
            self._enable_controls(False)
            self._set_status("Disconnected.")

        elif msg_type == "connect_btn_reenable":
            self.btn_connect.configure(state=tk.NORMAL)

        elif msg_type == "status":
            self._set_status(payload)

        elif msg_type == "warning":
            self._set_status(f"Warning: {payload}")

        elif msg_type == "error":
            self._set_status(f"Error: {payload}")
            messagebox.showerror("Scope Error", payload)

        elif msg_type == "waveforms":
            waveform_dict, meas = payload
            self._update_plot(waveform_dict)
            self._update_measurements(meas)
            self._set_status(f"Captured {len(waveform_dict)} channel(s).")

        elif msg_type == "settings":
            self._apply_settings_to_gui(payload)
            self._set_status("Settings loaded from scope.")

        elif msg_type == "devices":
            if not payload:
                messagebox.showinfo("Find Devices", "No VISA devices found.\n\n"
                    "Make sure the instrument is powered on and connected via USB or LAN.")
            else:
                win = tk.Toplevel(self.root)
                win.title("Found VISA Devices")
                win.configure(bg=BG_DARK)
                tk.Label(win, text="Click a device to use its address:",
                         bg=BG_DARK, fg=FG_TEXT,
                         font=("Helvetica", 10)).pack(padx=12, pady=(10, 4))
                lb = tk.Listbox(win, bg=BG_WIDGET, fg=FG_TEXT,
                                font=("Courier", 10), width=60, height=len(payload),
                                selectbackground="#445566")
                lb.pack(padx=12, pady=4)
                for r in payload:
                    lb.insert(tk.END, r)
                def _use():
                    sel = lb.curselection()
                    if sel:
                        self.visa_addr.set(payload[sel[0]])
                    win.destroy()
                tk.Button(win, text="Use Selected", command=_use,
                          bg="#336633", fg=FG_TEXT, relief=tk.FLAT,
                          font=("Helvetica", 10, "bold")).pack(pady=(0, 10))
                win.grab_set()
            self._set_status(f"Found {len(payload)} device(s).")

    # ------------------------------------------------------------------
    # Plot update
    # ------------------------------------------------------------------

    def _update_plot(self, waveform_dict):
        for ch in range(1, 5):
            trace = self._traces[ch]
            if ch in waveform_dict:
                t_arr, v_arr = waveform_dict[ch]
                trace.set_data(t_arr, v_arr)
                trace.set_visible(True)
            else:
                trace.set_data([], [])
                trace.set_visible(False)

        self.ax.relim()
        self.ax.autoscale_view()
        self._style_axes()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Measurements update
    # ------------------------------------------------------------------

    def _update_measurements(self, meas):
        for item in MEAS_ITEMS:
            lbl = self._meas_labels.get(item)
            if lbl is None:
                continue
            raw = meas.get(item)
            if raw is None:
                lbl.configure(text="—")
                continue
            try:
                val = float(raw)
            except (ValueError, TypeError):
                lbl.configure(text="—")
                continue
            if abs(val) >= INVALID_MEAS * 0.9:
                lbl.configure(text="—")
                continue
            if item == "FREQ":
                lbl.configure(text=fmt_freq(val))
            elif item == "PER":
                lbl.configure(text=fmt_time(val))
            else:
                lbl.configure(text=fmt_voltage(val))

    # ------------------------------------------------------------------
    # Populate GUI from scope settings
    # ------------------------------------------------------------------

    def _apply_settings_to_gui(self, s):
        try:
            self.tim_var.set(s.get("tim_scal", "0.001"))
        except Exception:
            pass

        for ch in range(1, 5):
            idx = ch - 1
            w   = self.ch_widgets[idx]
            try:
                w["vdiv_var"].set(s.get(f"ch{ch}_scal", "1.0"))
            except Exception:
                pass
            try:
                w["offs_var"].set(s.get(f"ch{ch}_offs", "0.0"))
            except Exception:
                pass
            try:
                coup = s.get(f"ch{ch}_coup", "DC").upper()
                w["coup_var"].set(coup)
            except Exception:
                pass
            try:
                disp    = s.get(f"ch{ch}_disp", "0").strip().upper()
                enabled = disp in ("1", "ON", "YES")
                self.ch_enabled[idx].set(enabled)
            except Exception:
                pass

        # Trigger source
        try:
            self.trig_sour_var.set(
                s.get("trig_sour", "CHAN1").strip().upper())
        except Exception:
            pass

        # Trigger slope  (scope may return long-form words)
        try:
            slop = s.get("trig_slop", "POS").strip().upper()
            slop_map = {
                "POSITIVE": "POS", "POS": "POS",
                "NEGATIVE": "NEG", "NEG": "NEG",
                "RFALLING": "RFAL", "RFAL": "RFAL",
            }
            self.trig_slop_var.set(slop_map.get(slop, slop))
        except Exception:
            pass

        try:
            self.trig_lev_var.set(s.get("trig_lev", "0.0"))
        except Exception:
            pass

        # Sweep mode  (scope may return long-form words)
        try:
            swe = s.get("trig_swe", "AUTO").strip().upper()
            swe_map = {
                "AUTO":   "AUTO",
                "NORMAL": "NORM", "NORM": "NORM",
                "SINGLE": "SING", "SING": "SING",
            }
            self.trig_swe_var.set(swe_map.get(swe, swe))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()

    # Dark ttk theme overrides
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "TCombobox",
        fieldbackground=BG_WIDGET,
        background=BG_WIDGET,
        foreground=FG_TEXT,
        selectbackground=BG_WIDGET,
        selectforeground=FG_TEXT,
        arrowcolor=FG_TEXT)
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", BG_WIDGET)],
        foreground=[("readonly", FG_TEXT)],
        background=[("readonly", BG_WIDGET)])
    style.configure(
        "TScrollbar",
        background=BG_PANEL,
        troughcolor=BG_DARK,
        arrowcolor=FG_DIM)

    OscilloscopeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
