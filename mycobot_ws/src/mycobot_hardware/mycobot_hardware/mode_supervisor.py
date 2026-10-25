"""Start and stop the launch files that share the myCobot serial port.

The supervisor does not open the serial port. It only runs ``ros2 launch``
and refuses a second mode that would open ``/dev/ttyTHS1`` or reuse a page port.
"""

import os
import signal
import subprocess
import threading
from collections import deque


STOP_GRACE_SECONDS = 12.0
TERM_GRACE_SECONDS = 3.0
LOG_LINES = 40


class Mode:
    """One launch file the supervisor is allowed to start."""

    def __init__(
        self,
        mode_id,
        title,
        package,
        launch_file,
        summary,
        owns_serial,
        moves_arm,
        ports,
        page_port,
    ):
        """Store the launch identity and the resources it claims."""
        self.mode_id = mode_id
        self.title = title
        self.package = package
        self.launch_file = launch_file
        self.summary = summary
        self.owns_serial = owns_serial
        self.moves_arm = moves_arm
        self.ports = tuple(ports)
        self.page_port = page_port

    def command(self):
        """Return the ``ros2 launch`` command for this mode."""
        return ["ros2", "launch", self.package, self.launch_file]


MODES = (
    Mode(
        "feedback",
        "Feedback only",
        "mycobot_hardware",
        "physical_bringup.launch.py",
        "Publishes joint state and the robot model. Motion stays off.",
        True,
        False,
        (),
        None,
    ),
    Mode(
        "plot",
        "Joint offset plot",
        "mycobot_hardware",
        "joint_plot.launch.py",
        "Draws commanded and measured joints. Does not open the serial port.",
        False,
        False,
        (8091,),
        8091,
    ),
    Mode(
        "diagnostic",
        "Joint diagnostic",
        "mycobot_hardware",
        "joint_diagnostic.launch.py",
        "Moves one joint at a time by 2 degrees at speed 5.",
        True,
        True,
        (8090,),
        8090,
    ),
    Mode(
        "demo",
        "Joint demo",
        "mycobot_hardware",
        "demo_control.launch.py",
        "Enables arm and gripper motion, including the demo page.",
        True,
        True,
        (8088, 8091),
        8088,
    ),
    Mode(
        "pick",
        "Pick and place",
        "mycobot_hardware",
        "pick_place.launch.py",
        "Enables Cartesian motion, the gripper, cameras, and the pick page.",
        True,
        True,
        (8092,),
        8092,
    ),
)


def launch_process(command):
    """Start one launch from the command mapping."""
    return subprocess.Popen(**command)


def conflict_reason(mode, active_modes):
    """Return why ``mode`` cannot start, or None when it can."""
    for other in active_modes:
        if mode.owns_serial and other.owns_serial:
            return f"{other.title} already owns the serial port"
        shared = sorted(set(mode.ports) & set(other.ports))
        if shared:
            return f"Port {shared[0]} is already used by {other.title}"
    return None


class _Running:
    """Bookkeeping for one live or recently exited launch."""

    def __init__(self, mode, process):
        """Remember the process and the log lines read from it."""
        self.mode = mode
        self.process = process
        self.lines = deque(maxlen=LOG_LINES)
        self.stopping = False


class ModeSupervisor:
    """Run at most one serial-owning launch, plus a plot when the port is free."""

    def __init__(self, popen=None, killpg=None, modes=MODES):
        """Keep the mode catalog and the process calls used to control it."""
        self._popen = launch_process if popen is None else popen
        self._killpg = os.killpg if killpg is None else killpg
        self._modes = {mode.mode_id: mode for mode in modes}
        self._lock = threading.Lock()
        self._running = {}

    def mode(self, mode_id):
        """Return one catalog entry, or None."""
        return self._modes.get(mode_id)

    def start(self, mode_id, confirmed=False):
        """Start one mode when it does not collide with a live launch."""
        mode = self._modes.get(mode_id)
        if mode is None:
            return False, f"Unknown mode {mode_id}"
        if mode.moves_arm and not confirmed:
            return False, "Confirm before starting a mode that moves the arm"
        with self._lock:
            self._reap_unlocked()
            current = self._running.get(mode_id)
            if current is not None and self._is_active(current):
                return False, f"{mode.title} is already running"
            reason = conflict_reason(mode, self._active_modes_unlocked())
            if reason is not None:
                return False, reason
            try:
                process = self._popen(self._launch_command(mode))
            except OSError as error:
                return False, f"Could not start {mode.title}: {error}"
            running = _Running(mode, process)
            self._running[mode_id] = running
            if process.stdout is not None:
                thread = threading.Thread(
                    target=self._read_output,
                    args=(running,),
                    name=f"supervisor_{mode_id}",
                    daemon=True,
                )
                thread.start()
        return True, f"Started {mode.title}"

    def stop(self, mode_id):
        """Interrupt one launch and wait until its process exits."""
        with self._lock:
            running = self._running.get(mode_id)
            if running is None or not self._is_active(running):
                self._reap_unlocked()
                return False, f"{mode_id} is not running"
            running.stopping = True
            process = running.process
        self._signal_and_wait(process)
        with self._lock:
            self._reap_unlocked()
        return True, f"Stopped {running.mode.title}"

    def stop_all(self):
        """Interrupt every launch this supervisor started."""
        with self._lock:
            active = [
                mode_id
                for mode_id, running in self._running.items()
                if self._is_active(running)
            ]
        messages = [self.stop(mode_id)[1] for mode_id in active]
        if not messages:
            return True, "Nothing was running"
        return True, "; ".join(messages)

    def snapshot(self):
        """Return every mode with its live status, block reason, and log."""
        with self._lock:
            self._reap_unlocked()
            active = self._active_modes_unlocked()
            modes = []
            for mode in self._modes.values():
                running = self._running.get(mode.mode_id)
                status, returncode, log = self._record_status(running)
                blocked = None
                if status not in ("running", "stopping"):
                    blocked = conflict_reason(mode, active)
                modes.append({
                    "id": mode.mode_id,
                    "title": mode.title,
                    "summary": mode.summary,
                    "moves_arm": mode.moves_arm,
                    "owns_serial": mode.owns_serial,
                    "page_port": mode.page_port,
                    "status": status,
                    "returncode": returncode,
                    "block_reason": blocked,
                    "log": log,
                })
        return {"modes": modes}

    def _launch_command(self, mode):
        """Build the process arguments for one ``ros2 launch``."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        return {
            "args": mode.command(),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "start_new_session": True,
            "text": True,
            "bufsize": 1,
            "env": env,
        }

    def _read_output(self, running):
        """Append launch output until the process closes its pipe."""
        stream = running.process.stdout
        for line in stream:
            with self._lock:
                running.lines.append(line.rstrip("\n"))

    def _signal_and_wait(self, process):
        """Interrupt a launch, then terminate it if it does not exit."""
        if process.poll() is not None:
            return
        self._signal(process, signal.SIGINT)
        if self._wait(process, STOP_GRACE_SECONDS):
            return
        self._signal(process, signal.SIGTERM)
        if self._wait(process, TERM_GRACE_SECONDS):
            return
        self._signal(process, signal.SIGKILL)
        self._wait(process, TERM_GRACE_SECONDS)

    def _signal(self, process, signum):
        """Signal the launch process group, or the process if that fails."""
        if process.poll() is not None:
            return
        try:
            self._killpg(process.pid, signum)
        except ProcessLookupError:
            return
        except OSError:
            try:
                process.send_signal(signum)
            except OSError:
                return

    def _wait(self, process, timeout):
        """Return True when the process exits within the timeout."""
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _reap_unlocked(self):
        """Refresh return codes. Caller must hold the lock."""
        for running in self._running.values():
            running.process.poll()

    def _active_modes_unlocked(self):
        """Return modes whose processes have not exited."""
        return [
            running.mode
            for running in self._running.values()
            if self._is_active(running)
        ]

    def _is_active(self, running):
        """Return True while the launch process is still alive."""
        return running.process.poll() is None

    def _record_status(self, running):
        """Return status, exit code, and log lines for one mode."""
        if running is None:
            return "idle", None, []
        returncode = running.process.poll()
        log = list(running.lines)
        if running.stopping and returncode is None:
            return "stopping", None, log
        if returncode is None:
            return "running", None, log
        return "exited", returncode, log
