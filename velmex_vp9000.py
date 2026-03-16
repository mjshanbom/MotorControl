import serial
import time


class VP9000:
    """
    Controller for the Velmex VP9000 motor controller via serial (RS-232).

    The VP9000 uses a simple ASCII command protocol.
    Commands are sent as strings; the controller echoes them back.
    """

    PARITY_MAP = {
        "N": serial.PARITY_NONE,
        "E": serial.PARITY_EVEN,
        "O": serial.PARITY_ODD,
    }

    def __init__(self, port: str, baudrate: int = 9600, bytesize: int = 7,
                 parity: str = "E", stopbits: int = 2, timeout: float = 0.3):
        """
        Connect to the VP9000.

        Args:
            port:     Serial port, e.g. '/dev/tty.usbserial-XXXX' or 'COM3'
            baudrate: Baud rate — must match VP9000 DIP switch setting (default 9600)
            bytesize: Data bits: 7 or 8 (default 8)
            parity:   'N' none, 'E' even, 'O' odd (default 'N')
            stopbits: 1 or 2 (default 1)
            timeout:  Read timeout in seconds
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.SEVENBITS if bytesize == 7 else serial.EIGHTBITS,
            parity=self.PARITY_MAP.get(parity.upper(), serial.PARITY_NONE),
            stopbits=serial.STOPBITS_TWO if stopbits == 2 else serial.STOPBITS_ONE,
            timeout=timeout,
        )
        time.sleep(0.5)

    def close(self):
        """Close the serial connection."""
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ #
    # Low-level communication
    # ------------------------------------------------------------------ #

    def send(self, command: str):
        """Send a raw ASCII command."""
        self.ser.write((command + "\r").encode("ascii"))
        self.ser.flush()

    def read_response(self) -> str:
        """Read a response line from the controller (VP9000 uses CR, not LF)."""
        return self.ser.read_until(b"\r").decode("latin-1").strip()

    def query(self, command: str) -> str:
        """Send a command and return the response."""
        self.send(command)
        return self.read_response()

    # ------------------------------------------------------------------ #
    # Motor control commands
    # ------------------------------------------------------------------ #

    def enable_online_mode(self, retries: int = 10):
        """Put the controller in online (computer control) mode and confirm."""
        # Send F repeatedly over ~2 seconds to cover the recovery window after
        # the port-open DTR glitch, then verify with V.
        for _ in range(8):
            self.ser.write(b"F\r")
            self.ser.flush()
            time.sleep(0.25)

        last_status = ""
        for _ in range(retries):
            self.ser.write(b"V\r")
            self.ser.flush()
            time.sleep(0.3)
            data = self.ser.read(self.ser.in_waiting or 1)
            text = data.decode("latin-1").strip() if data else ""
            last_status = text[-1] if text else ""
            if last_status == "R":
                return
            self.ser.write(b"F\r")
            self.ser.flush()
            time.sleep(0.5)
        raise RuntimeError(f"VP9000 did not come online (last status: {last_status!r})")

    def stop(self):
        """Kill all motion immediately."""
        self.send("K")

    def run_program(self):
        """Execute the current program (R command)."""
        self.send("R")

    def home(self, motor: int):
        """
        Home a motor (move to limit switch).

        Args:
            motor: Motor number (1–4)
        """
        self.send(f"C,I{motor}M-0,R")

    def move_absolute(self, motor: int, steps: int):
        """
        Move a motor to an absolute step position.

        Args:
            motor: Motor number (1–4)
            steps: Target position in steps
        """
        self.send(f"C,IA{motor}M{steps},R")

    def move_relative(self, motor: int, steps: int):
        """
        Move a motor by a relative number of steps.

        Args:
            motor: Motor number (1–4)
            steps: Steps to move (negative = reverse)
        """
        self.send(f"C,I{motor}M{steps},R")

    def set_speed(self, motor: int, speed: int):
        """
        Set the speed for a motor.

        Args:
            motor: Motor number (1–4)
            speed: Speed in steps/second (controller-dependent range)
        """
        self.send(f"S{motor}M{speed}")

    def get_position(self) -> int:
        """
        Query the current position (all motors).
        Returns the first position value from the response.

        Returns:
            Current step position as an integer (e.g. +0000000 → 0)
        """
        response = self.query("C,X")
        return int(response.split()[0])

    def wait_for_done(self, poll_interval: float = 0.1, timeout: float = 30.0,
                      stop_event=None):
        """
        Block until all motion is complete.
        The VP9000 returns 'R' when idle/ready and '^' when busy.

        Args:
            poll_interval: How often to poll (seconds)
            timeout: Raise TimeoutError if motion takes longer than this (seconds)
            stop_event: optional threading.Event; if set, sends K and returns early
        """
        deadline = time.monotonic() + timeout
        next_log = time.monotonic() + 5.0
        time.sleep(0.05)  # Give controller time to start executing
        while True:
            if stop_event is not None and stop_event.is_set():
                self.send("K")  # Kill motion immediately
                return
            if self.ser.in_waiting:        # discard stale bytes before polling
                self.ser.read(self.ser.in_waiting)
            self.ser.write(b"V\r")
            self.ser.flush()
            time.sleep(0.05)
            # Read one byte — returns immediately when byte arrives, not waiting for \r
            byte = self.ser.read(1)
            status = byte.decode("latin-1").strip() if byte else ""
            if status == "R":  # 'R' means Ready (idle)
                break
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError(f"wait_for_done timed out after {timeout}s (last status: {status!r})")
            if now >= next_log:
                print(f"[wait_for_done] still running, status={status!r}")
                next_log = now + 5.0
            time.sleep(poll_interval)


# ------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    PORT = "/dev/tty.usbserial-XXXX"  # <-- change this to your port

    with VP9000(PORT) as controller:
        controller.enable_online_mode()

        # Set speed for motor 1
        controller.set_speed(1, 1000)

        # Move motor 1 forward 5000 steps
        controller.move_relative(1, 5000)
        controller.wait_for_done()

        print("Position:", controller.get_position(1))

        # Move motor 1 back to absolute position 0
        controller.move_absolute(1, 0)
        controller.wait_for_done()

        print("Done. Position:", controller.get_position(1))
