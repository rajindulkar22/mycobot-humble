#!/usr/bin/env python3
"""Browser page that starts and stops the myCobot launch modes."""

import json
import os
import threading
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

from mycobot_hardware.mode_supervisor import ModeSupervisor


class SupervisorHttpServer(ThreadingHTTPServer):
    """Serve supervisor requests without blocking process shutdown."""

    daemon_threads = True
    allow_reuse_address = True


class SupervisorUi(Node):
    """Serve the mode page. This node does not open the robot serial port."""

    def __init__(self, supervisor=None):
        """Open the page and keep a handle on the launches it starts."""
        super().__init__("mycobot_supervisor")
        self.declare_parameter("ui_host", "0.0.0.0")
        self.declare_parameter("ui_port", 8093)
        self._supervisor = ModeSupervisor() if supervisor is None else supervisor
        self._server = None
        self._page_path = _page_path()

        host = str(self.get_parameter("ui_host").value)
        port = int(self.get_parameter("ui_port").value)
        try:
            self._server = SupervisorHttpServer((host, port), SupervisorHandler)
        except OSError as error:
            raise RuntimeError(
                f"Could not listen on {host}:{port}: {error}"
            ) from error
        self._server.supervisor_node = self
        thread = threading.Thread(
            target=self._server.serve_forever,
            name="mycobot_supervisor_http",
            daemon=True,
        )
        thread.start()
        self.get_logger().info(f"Supervisor page: http://{host}:{port}")

    def destroy_node(self):
        """Stop every launch this page started, then destroy the node."""
        self._supervisor.stop_all()
        self.close_server()
        return super().destroy_node()

    def close_server(self):
        """Stop accepting browser connections."""
        server = self._server
        self._server = None
        if server is None:
            return
        server.shutdown()
        server.server_close()

    def snapshot(self):
        """Return the mode list for the page."""
        return self._supervisor.snapshot()

    def start_mode(self, payload):
        """Start the requested mode after the page confirms motion."""
        mode_id = str(payload.get("mode") or "")
        confirmed = payload.get("confirm") is True
        ok, message = self._supervisor.start(mode_id, confirmed=confirmed)
        return (200 if ok else 409), _result(ok, message)

    def stop_mode(self, payload):
        """Stop one mode, or all of them when requested."""
        if payload.get("all") is True:
            ok, message = self._supervisor.stop_all()
        else:
            ok, message = self._supervisor.stop(str(payload.get("mode") or ""))
        return (200 if ok else 409), _result(ok, message)

    def page_bytes(self):
        """Read the supervisor page from the installed HTML file."""
        with open(self._page_path, "rb") as page_file:
            return page_file.read()


class SupervisorHandler(BaseHTTPRequestHandler):
    """HTTP handlers for the supervisor page and its actions."""

    def do_GET(self):
        """Serve the page or the current mode status."""
        path = self.path.split("?", 1)[0]
        node = self.server.supervisor_node
        if path == "/":
            self._send_bytes(200, "text/html; charset=utf-8", node.page_bytes())
            return
        if path == "/api/state":
            self._send_json(200, node.snapshot())
            return
        self._send_json(404, _result(False, "Not found"))

    def do_POST(self):
        """Start or stop a launch mode."""
        path = self.path.split("?", 1)[0]
        node = self.server.supervisor_node
        try:
            payload = self._read_json()
            if path == "/api/start":
                status, body = node.start_mode(payload)
            elif path == "/api/stop":
                status, body = node.stop_mode(payload)
            else:
                status, body = 404, _result(False, "Not found")
        except ValueError as error:
            status, body = 400, _result(False, str(error))
        self._send_json(status, body)

    def log_message(self, fmt, *args):
        """Keep request logs out of the robot terminal."""
        return

    def _read_json(self):
        """Read a JSON object from the request body."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Request body is not JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be an object")
        return payload

    def _send_json(self, status, payload):
        """Send one JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, "application/json", body)

    def _send_bytes(self, status, content_type, body):
        """Send one complete HTTP response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _result(ok, message):
    """Return the JSON object used by start and stop."""
    return {"success": ok, "message": message}


def _page_path():
    """Locate the installed page, or the copy next to this module."""
    try:
        share_path = os.path.join(
            get_package_share_directory("mycobot_hardware"),
            "supervisor_page.html",
        )
    except PackageNotFoundError:
        share_path = ""
    if share_path and os.path.isfile(share_path):
        return share_path
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "supervisor_page.html",
    )


def main(args=None):
    """Run the supervisor page until the process is interrupted."""
    rclpy.init(args=args)
    node = SupervisorUi()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._supervisor.stop_all()
        node.close_server()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
