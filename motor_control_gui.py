#!/usr/bin/env python3
"""Simple 3-axis manual jog GUI for the Velmex VP9000."""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import serial.tools.list_ports

from velmex_vp9000 import VP9000


# ── Background workers ───────────────────────────────────────────────────── #

def run_connect(cfg, msg_q):
    try:
        msg_q.put({"type": "log", "text": f"Opening {cfg['port']}…"})
        motor = VP9000(cfg["port"], baudrate=cfg["baudrate"],
                       bytesize=cfg["bytesize"], parity=cfg["parity"],
                       stopbits=cfg["stopbits"])
        msg_q.put({"type": "log", "text": "Port open. Press Online to enable computer control."})
        msg_q.put({"type": "connected", "motor": motor})
    except Exception as exc:
        msg_q.put({"type": "connect_error", "text": str(exc)})


def run_online(motor, msg_q):
    try:
        motor.enable_online_mode()
        msg_q.put({"type": "online"})
    except Exception as exc:
        msg_q.put({"type": "error", "text": str(exc)})


def run_move(motor, motor_num, steps, speed, stop_event, msg_q):
    try:
        motor.set_speed(motor_num, speed)
        motor.move_relative(motor_num, steps)
        motor.wait_for_done(stop_event=stop_event)
        msg_q.put({"type": "move_done", "motor": motor_num, "steps": steps})
    except Exception as exc:
        msg_q.put({"type": "error", "text": str(exc)})


def run_home(motor, motor_num, speed, stop_event, msg_q):
    try:
        motor.set_speed(motor_num, speed)
        motor.home(motor_num)
        motor.wait_for_done(stop_event=stop_event)
        msg_q.put({"type": "homed", "motor": motor_num})
    except Exception as exc:
        msg_q.put({"type": "error", "text": str(exc)})


# ── GUI ──────────────────────────────────────────────────────────────────── #

LAB_SPMM = {"Optics Lab (160 steps/mm)": 160, "Acoustics Lab (200 steps/mm)": 200}

AXIS_DEFAULTS = [
    {"label": "X", "motor": "1", "step": "3.0"},
    {"label": "Y", "motor": "2", "step": "3.0"},
    {"label": "Z", "motor": "3", "step": "0.25"},
]


class MotorControlApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VP9000 Motor Control")
        self.resizable(False, False)

        self._msg_q    = queue.Queue()
        self._stop_evt = threading.Event()
        self._motor    = None
        self._moving   = False

        # Tracked positions per motor (1-indexed, index 0 unused)
        self._positions = [0, 0, 0, 0, 0]

        self._build_ui()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ──────────────────────────────────────────────────── #

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── Connection ───────────────────────────────────────────────────── #
        conn = ttk.LabelFrame(self, text="Connection")
        conn.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(conn, text="Motor port").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self._port = tk.StringVar(value="/dev/tty.usbserial-FTEHI1FO")
        ttk.Entry(conn, textvariable=self._port, width=36).grid(row=0, column=1, padx=6, pady=2)
        ttk.Button(conn, text="Find", command=self._find_port).grid(row=0, column=2, padx=(0, 8), pady=2)

        rs = ttk.Frame(conn)
        rs.grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=2)
        for label, var_name, values, default, width in [
            ("Baud",      "_baud",   ["9600","19200","38400","57600","115200"], "9600", 8),
            ("Data bits", "_bits",   ["7","8"],                                "7",    3),
            ("Parity",    "_parity", ["N","E","O"],                            "E",    3),
            ("Stop bits", "_stop",   ["1","2"],                                "2",    3),
        ]:
            ttk.Label(rs, text=label).pack(side="left")
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            ttk.Combobox(rs, textvariable=var, values=values, width=width,
                         state="readonly").pack(side="left", padx=(2, 10))

        ttk.Label(conn, text="Speed (steps/s)").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self._speed = tk.StringVar(value="1000")
        ttk.Entry(conn, textvariable=self._speed, width=10).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(conn, text="Lab").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        self._lab = tk.StringVar(value="Optics Lab (160 steps/mm)")
        ttk.Combobox(conn, textvariable=self._lab, values=list(LAB_SPMM.keys()),
                     state="readonly", width=28).grid(row=3, column=1, sticky="w", padx=6, pady=2)

        # ── Axes ─────────────────────────────────────────────────────────── #
        axes_frame = ttk.LabelFrame(self, text="Axes")
        axes_frame.grid(row=1, column=0, sticky="ew", **pad)

        headers = ["Axis", "Motor #", "Step (mm)", "Position (mm)", "Jog", "Home"]
        for col, h in enumerate(headers):
            ttk.Label(axes_frame, text=h, font=("", 10, "bold")).grid(
                row=0, column=col, padx=6, pady=(4, 2))

        self._axis_widgets = []
        for row_idx, defaults in enumerate(AXIS_DEFAULTS, start=1):
            row_widgets = {}

            # Label
            ttk.Label(axes_frame, text=defaults["label"], width=3).grid(
                row=row_idx, column=0, padx=6, pady=4)

            # Motor #
            motor_var = tk.StringVar(value=defaults["motor"])
            ttk.Spinbox(axes_frame, textvariable=motor_var, from_=1, to=4,
                        width=4).grid(row=row_idx, column=1, padx=6)
            row_widgets["motor"] = motor_var

            # Step size (mm)
            step_var = tk.StringVar(value=defaults["step"])
            ttk.Entry(axes_frame, textvariable=step_var, width=8).grid(
                row=row_idx, column=2, padx=6)
            row_widgets["step"] = step_var

            # Position display (mm)
            pos_var = tk.StringVar(value="0.000")
            ttk.Label(axes_frame, textvariable=pos_var, width=10,
                      relief="sunken", anchor="e").grid(row=row_idx, column=3, padx=6)
            row_widgets["pos"] = pos_var

            # Jog buttons
            jog_frame = ttk.Frame(axes_frame)
            jog_frame.grid(row=row_idx, column=4, padx=6)
            neg_btn = ttk.Button(jog_frame, text="  −  ", width=4,
                                 command=lambda r=row_idx-1: self._jog(r, -1))
            neg_btn.pack(side="left", padx=2)
            pos_btn = ttk.Button(jog_frame, text="  +  ", width=4,
                                 command=lambda r=row_idx-1: self._jog(r, +1))
            pos_btn.pack(side="left", padx=2)
            row_widgets["neg_btn"] = neg_btn
            row_widgets["pos_btn"] = pos_btn

            # Home button
            home_btn = ttk.Button(axes_frame, text="Home",
                                  command=lambda r=row_idx-1: self._home(r))
            home_btn.grid(row=row_idx, column=5, padx=6)
            row_widgets["home_btn"] = home_btn

            self._axis_widgets.append(row_widgets)

        # ── Status & log ─────────────────────────────────────────────────── #
        self._status_var = tk.StringVar(value="Not connected.")
        ttk.Label(self, textvariable=self._status_var).grid(
            row=2, column=0, sticky="w", padx=10, pady=(4, 0))

        self._log = scrolledtext.ScrolledText(self, height=8, width=60,
                                              state="disabled", font=("Menlo", 11))
        self._log.grid(row=3, column=0, padx=8, pady=4)

        # ── Buttons ──────────────────────────────────────────────────────── #
        btn = ttk.Frame(self)
        btn.grid(row=4, column=0, pady=(0, 8))

        self._connect_btn = ttk.Button(btn, text="Connect", command=self._connect)
        self._connect_btn.pack(side="left", padx=6)
        self._online_btn = ttk.Button(btn, text="Online", command=self._go_online,
                                      state="disabled")
        self._online_btn.pack(side="left", padx=6)
        self._disconnect_btn = ttk.Button(btn, text="Disconnect",
                                          command=self._disconnect, state="disabled")
        self._disconnect_btn.pack(side="left", padx=6)
        self._stop_btn = ttk.Button(btn, text="Stop", command=self._stop,
                                    state="disabled")
        self._stop_btn.pack(side="left", padx=6)

        self._set_jog_buttons_state("disabled")

    # ── Helpers ──────────────────────────────────────────────────────────── #

    def _set_jog_buttons_state(self, state):
        for w in self._axis_widgets:
            w["neg_btn"].config(state=state)
            w["pos_btn"].config(state=state)
            w["home_btn"].config(state=state)

    def _speed_val(self):
        try:
            return int(self._speed.get())
        except ValueError:
            return 1000

    # ── Connection ───────────────────────────────────────────────────────── #

    def _connect(self):
        cfg = {
            "port":     self._port.get(),
            "baudrate": int(self._baud.get()),
            "bytesize": int(self._bits.get()),
            "parity":   self._parity.get(),
            "stopbits": int(self._stop.get()),
        }
        self._connect_btn.config(state="disabled")
        self._status_var.set("Connecting…")
        threading.Thread(target=run_connect, args=(cfg, self._msg_q), daemon=True).start()

    def _find_port(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            messagebox.showinfo("Find Ports", "No serial ports found.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Select Serial Port")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Select a port:").pack(padx=12, pady=(10, 4))
        lb = tk.Listbox(dlg, listvariable=tk.StringVar(value=ports),
                        selectmode="single", width=40, height=min(len(ports), 10))
        lb.pack(padx=12, pady=4)
        lb.select_set(0)
        def _select():
            sel = lb.curselection()
            if sel:
                self._port.set(ports[sel[0]])
            dlg.destroy()
        ttk.Button(dlg, text="Select", command=_select).pack(pady=(4, 10))
        dlg.grab_set()

    def _go_online(self):
        if self._motor is None:
            return
        self._online_btn.config(state="disabled")
        self._status_var.set("Going online…")
        self._log_line("Sending online command (F)…")
        threading.Thread(target=run_online, args=(self._motor, self._msg_q),
                         daemon=True).start()

    def _disconnect(self):
        if self._motor:
            try:
                self._motor.send("K")
            except Exception:
                pass
        self._motor = None
        self._connect_btn.config(state="normal")
        self._online_btn.config(state="disabled")
        self._disconnect_btn.config(state="disabled")
        self._stop_btn.config(state="disabled")
        self._set_jog_buttons_state("disabled")
        self._status_var.set("Disconnected.")
        self._log_line("Disconnected.")

    def _on_close(self):
        if self._motor:
            try:
                self._motor.close()
            except Exception:
                pass
        self.destroy()

    # ── Motion ───────────────────────────────────────────────────────────── #

    def _jog(self, axis_idx, direction):
        if self._moving or self._motor is None:
            return
        w = self._axis_widgets[axis_idx]
        try:
            motor_num = int(w["motor"].get())
            step_mm   = float(w["step"].get())
        except ValueError:
            self._log_line("Invalid motor # or step size.")
            return
        spmm  = LAB_SPMM[self._lab.get()]
        steps = int(round(direction * step_mm * spmm))
        self._start_move(motor_num, steps)

    def _home(self, axis_idx):
        if self._moving or self._motor is None:
            return
        w = self._axis_widgets[axis_idx]
        try:
            motor_num = int(w["motor"].get())
        except ValueError:
            self._log_line("Invalid motor #.")
            return
        self._moving = True
        self._set_jog_buttons_state("disabled")
        self._stop_btn.config(state="normal")
        self._stop_evt.clear()
        self._status_var.set(f"Homing motor {motor_num}…")
        threading.Thread(target=run_home,
                         args=(self._motor, motor_num, self._speed_val(),
                               self._stop_evt, self._msg_q),
                         daemon=True).start()

    def _start_move(self, motor_num, steps):
        self._moving = True
        self._set_jog_buttons_state("disabled")
        self._stop_btn.config(state="normal")
        self._stop_evt.clear()
        direction = "+" if steps >= 0 else ""
        self._status_var.set(f"Moving motor {motor_num} {direction}{steps} steps…")
        threading.Thread(target=run_move,
                         args=(self._motor, motor_num, steps, self._speed_val(),
                               self._stop_evt, self._msg_q),
                         daemon=True).start()

    def _stop(self):
        self._stop_evt.set()
        if self._motor:
            try:
                self._motor.send("K")
            except Exception:
                pass
        self._status_var.set("Stopping…")
        self._stop_btn.config(state="disabled")

    # ── Message polling ──────────────────────────────────────────────────── #

    def _poll(self):
        try:
            while True:
                msg = self._msg_q.get_nowait()
                t = msg["type"]

                if t == "log":
                    self._log_line(msg["text"])

                elif t == "connected":
                    self._motor = msg["motor"]
                    self._connect_btn.config(state="disabled")
                    self._online_btn.config(state="normal")
                    self._disconnect_btn.config(state="normal")
                    self._status_var.set("Connected — press Online to enable control.")

                elif t == "online":
                    self._online_btn.config(state="disabled")
                    self._set_jog_buttons_state("normal")
                    self._log_line("Controller online. Ready.")
                    self._status_var.set("Ready.")

                elif t == "connect_error":
                    self._log_line(f"Connect error: {msg['text']}")
                    self._status_var.set("Connection failed.")
                    self._connect_btn.config(state="normal")

                elif t == "move_done":
                    motor_num = msg["motor"]
                    self._positions[motor_num] += msg["steps"]
                    spmm = LAB_SPMM[self._lab.get()]
                    for w in self._axis_widgets:
                        if int(w["motor"].get()) == motor_num:
                            pos_mm = self._positions[motor_num] / spmm
                            w["pos"].set(f"{pos_mm:.3f}")
                    self._log_line(
                        f"Motor {motor_num} moved {msg['steps']:+d} steps  "
                        f"→ {self._positions[motor_num] / spmm:.3f} mm")
                    self._moving = False
                    self._set_jog_buttons_state("normal")
                    self._stop_btn.config(state="disabled")
                    self._status_var.set("Ready.")

                elif t == "homed":
                    motor_num = msg["motor"]
                    self._positions[motor_num] = 0
                    for w in self._axis_widgets:
                        if int(w["motor"].get()) == motor_num:
                            w["pos"].set("0.000")
                    self._log_line(f"Motor {motor_num} homed.")
                    self._moving = False
                    self._set_jog_buttons_state("normal")
                    self._stop_btn.config(state="disabled")
                    self._status_var.set("Ready.")

                elif t == "error":
                    self._log_line(f"ERROR: {msg['text']}")
                    self._status_var.set("Error — see log.")
                    self._moving = False
                    self._set_jog_buttons_state("normal")
                    self._stop_btn.config(state="disabled")

        except queue.Empty:
            pass

        self.after(100, self._poll)

    def _log_line(self, text):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")


if __name__ == "__main__":
    MotorControlApp().mainloop()
