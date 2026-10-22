#!/usr/bin/env python3
"""Plot measured joint angles against the angles that were commanded."""

import json
import math
import os
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from rclpy.executors import (
    ExternalShutdownException,
    MultiThreadedExecutor,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState

HISTORY_SECONDS = 30.0
SAMPLE_SECONDS = 0.1
TOLERANCE_DEGREES = 0.5
STALE_SECONDS = 1.0
STILL_SECONDS = 1.0
STILL_BAND_DEGREES = 0.15
DIRECTION_SECONDS = 2.0
DIRECTION_COMMAND_DEGREES = 1.0
DIRECTION_MEASURED_DEGREES = 0.3
LAG_HOLD_SECONDS = 5.0
JOINT_NAMES = (
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
)


class PlotHttpServer(ThreadingHTTPServer):
    """Serve plot requests without blocking process shutdown."""

    daemon_threads = True
    allow_reuse_address = True


class JointPlot(Node):
    """Serve commanded and measured joint traces without using the arm."""

    def __init__(self):
        """Subscribe to both joint topics and open the plot page."""
        super().__init__("mycobot_joint_plot")
        self.declare_parameter("ui_host", "0.0.0.0")
        self.declare_parameter("ui_port", 8091)

        self._lock = threading.Lock()
        self._measured = deque()
        self._commanded = deque()
        self._lag_until = [0.0] * 6
        self._server = None

        self.create_subscription(
            JointState,
            "joint_states",
            self._on_measured,
            10,
        )
        self.create_subscription(
            JointState,
            "mycobot/commanded_joint_states",
            self._on_commanded,
            10,
        )

        host = str(self.get_parameter("ui_host").value)
        port = int(self.get_parameter("ui_port").value)
        self._page_path = _page_path()
        try:
            self._server = PlotHttpServer((host, port), PlotHandler)
        except OSError as error:
            raise RuntimeError(
                f"Could not listen on {host}:{port}: {error}"
            ) from error
        self._server.plot_node = self
        threading.Thread(
            target=self._server.serve_forever,
            name="mycobot_joint_plot_http",
            daemon=True,
        ).start()
        self.get_logger().info(
            f"Joint offset plot: http://{host}:{port}"
        )

    def close_server(self):
        """Stop accepting browser connections."""
        server = self._server
        self._server = None
        if server is None:
            return
        server.shutdown()
        server.server_close()

    def page_bytes(self):
        """Read the plot page from the installed HTML file."""
        with open(self._page_path, "rb") as page_file:
            return page_file.read()

    def history_payload(self):
        """Return the last 30 seconds and a label for each joint."""
        now = self._now()
        with self._lock:
            measured = list(self._measured)
            commanded = list(self._commanded)
            lag_until = list(self._lag_until)
            joints = []
            for index in range(6):
                label, error, lag_until[index] = classify_joint(
                    measured,
                    commanded,
                    index,
                    now,
                    lag_until[index],
                )
                joints.append({
                    "name": f"J{index + 1}",
                    "label": label,
                    "error_degrees": _round_or_none(error),
                    "measured": _trace(
                        measured,
                        index,
                        now,
                        fresh_only=True,
                    ),
                    "commanded": _trace(
                        commanded,
                        index,
                        now,
                        fresh_only=False,
                    ),
                })
            self._lag_until = lag_until
        return {"joints": joints}

    def _on_measured(self, message):
        """Store a new measured sample and ignore a repeated stamp."""
        self._store(self._measured, message, skip_repeat=True)

    def _on_commanded(self, message):
        """Store a commanded sample."""
        self._store(self._commanded, message, skip_repeat=False)

    def _store(self, series, message, skip_repeat):
        """Append one six-joint sample and drop traces older than 30s."""
        stamp = self._stamp_seconds(message.header.stamp)
        values = _angles_degrees(message)
        with self._lock:
            if skip_repeat and series and series[-1][0] == stamp:
                return
            series.append((stamp, values))
            cutoff = self._now() - HISTORY_SECONDS
            while series and series[0][0] < cutoff:
                series.popleft()

    def _stamp_seconds(self, stamp):
        """Return a header stamp in seconds, or the node clock if unset."""
        seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if seconds <= 0.0:
            return self._now()
        return seconds

    def _now(self):
        """Return the node clock in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9


class PlotHandler(BaseHTTPRequestHandler):
    """HTTP handlers for the plot page and its history API."""

    def do_GET(self):
        """Serve the page or the joint history JSON."""
        path = self.path.split("?", 1)[0]
        node = self.server.plot_node
        if path in ("/", "/plot"):
            body = node.page_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/history":
            body = json.dumps(node.history_payload()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, format, *args):
        """Keep ordinary page polls out of the launch log."""
        del format, args


def classify_joint(
    measured_series,
    commanded_series,
    joint_index,
    now,
    lag_until,
):
    """Return the label, error, and lag-until time for one joint."""
    measured = _hold(measured_series, joint_index, now)
    commanded = _hold(commanded_series, joint_index, now)
    if measured is None or commanded is None:
        return "waiting", None, lag_until

    age = now - measured[0]
    error = measured[1] - commanded[1]
    if age > STALE_SECONDS:
        return "missed read", None, lag_until

    command_change = _change(
        commanded_series,
        joint_index,
        now,
        DIRECTION_SECONDS,
    )
    measured_change = _change(
        measured_series,
        joint_index,
        now,
        DIRECTION_SECONDS,
    )
    if (
        command_change is not None
        and measured_change is not None
        and abs(command_change) >= DIRECTION_COMMAND_DEGREES
        and measured_change * command_change < 0
        and abs(measured_change) >= DIRECTION_MEASURED_DEGREES
    ):
        return "wrong direction", error, lag_until

    recent_command = _change(
        commanded_series,
        joint_index,
        now,
        STILL_SECONDS,
    )
    recent_measured = _change(
        measured_series,
        joint_index,
        now,
        STILL_SECONDS,
    )
    command_still = (
        recent_command is not None
        and abs(recent_command) < STILL_BAND_DEGREES
    )
    measured_still = (
        recent_measured is not None
        and abs(recent_measured) < STILL_BAND_DEGREES
    )
    if command_still and measured_still:
        if abs(error) > TOLERANCE_DEGREES:
            return "offset", error, lag_until
        if now < lag_until:
            return "lag", error, lag_until
        return "tracking", error, lag_until

    if abs(error) > TOLERANCE_DEGREES:
        return "lag", error, now + LAG_HOLD_SECONDS
    return "tracking", error, lag_until


def _hold(series, joint_index, time_s):
    """Return (stamp, degrees) at or before time_s."""
    found = None
    for stamp, values in series:
        if stamp > time_s:
            break
        value = values[joint_index]
        if value is None or not math.isfinite(value):
            continue
        found = (stamp, value)
    return found


def _change(series, joint_index, now, window):
    """Return how far one joint moved over the given window."""
    older = _hold(series, joint_index, now - window)
    newer = _hold(series, joint_index, now)
    if older is None or newer is None:
        return None
    return newer[1] - older[1]


def _trace(series, joint_index, now, fresh_only):
    """Downsample one joint across the last 30 seconds."""
    points = []
    time_s = now - HISTORY_SECONDS
    while time_s <= now + 1e-9:
        held = _hold(series, joint_index, time_s)
        if held is not None:
            stale = fresh_only and time_s - held[0] > STALE_SECONDS
            if not stale:
                points.append([
                    round(time_s - now, 3),
                    round(held[1], 3),
                ])
        time_s += SAMPLE_SECONDS
    return points


def _angles_degrees(message):
    """Map a joint-state message into six degree values."""
    values = [None] * 6
    names = list(message.name)
    for index, name in enumerate(JOINT_NAMES):
        if name not in names:
            continue
        position_index = names.index(name)
        if position_index >= len(message.position):
            continue
        radians = float(message.position[position_index])
        if math.isfinite(radians):
            values[index] = math.degrees(radians)
    return tuple(values)


def _round_or_none(value):
    """Round a degree value for JSON, or return None."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, 3)


def _page_path():
    """Locate the installed page, or the copy next to this module."""
    try:
        share_path = os.path.join(
            get_package_share_directory("mycobot_hardware"),
            "joint_plot_page.html",
        )
    except PackageNotFoundError:
        share_path = ""
    if share_path and os.path.isfile(share_path):
        return share_path
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "joint_plot_page.html",
    )


def main(args=None):
    """Run the offset plot until the process is interrupted."""
    rclpy.init(args=args)
    node = JointPlot()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close_server()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
