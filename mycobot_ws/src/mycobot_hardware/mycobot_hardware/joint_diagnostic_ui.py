#!/usr/bin/env python3
"""Browser diagnostic for one-joint measured feedback."""

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from mycobot_interfaces.srv import MoveJoint
from rclpy.executors import (
    ExternalShutdownException,
    MultiThreadedExecutor,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from mycobot_hardware.joint_diagnostic import (
    DIAGNOSTIC_TOLERANCE_DEGREES,
    INITIAL_POSITION_DEGREES,
    abort_reason,
    assess_move,
    step_toward,
    summarize_reports,
)

DIAGNOSTIC_STEP_DEGREES = 2.0
DIAGNOSTIC_SPEED = 5
MAX_HOME_STEPS = 100
MOVE_TIMEOUT_SECONDS = 20.0
STATUS_TIMEOUT_SECONDS = 5.0
FEEDBACK_WAIT_SECONDS = 1.0
JOINT_NAMES = (
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
)


class DiagnosticHttpServer(ThreadingHTTPServer):
    """Serve diagnostic requests without blocking shutdown."""

    daemon_threads = True
    allow_reuse_address = True


class JointDiagnosticUi(Node):
    """Serve measured joint checks through the existing hardware services."""

    def __init__(self):
        """Subscribe to feedback and open the diagnostic page."""
        super().__init__("mycobot_joint_diagnostic_ui")
        self.declare_parameter("ui_host", "0.0.0.0")
        self.declare_parameter("ui_port", 8090)

        self._lock = threading.Lock()
        self._measured = [None] * 6
        self._commanded = [None] * 6
        self._feedback_stamp = None
        self._feedback_received = None
        self._motion_lock = threading.Lock()
        self._stop_requested = threading.Event()
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
        self._move_client = self.create_client(MoveJoint, "mycobot/move_joint")
        self._status_client = self.create_client(Trigger, "mycobot/status")
        self._stop_client = self.create_client(Trigger, "mycobot/stop")

        host = str(self.get_parameter("ui_host").value)
        port = int(self.get_parameter("ui_port").value)
        self._page_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "joint_diagnostic_page.html",
        )
        try:
            self._server = DiagnosticHttpServer((host, port), DiagnosticHandler)
        except OSError as error:
            raise RuntimeError(
                f"Could not listen on {host}:{port}: {error}"
            ) from error
        self._server.diagnostic_node = self
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.get_logger().info(
            f"Joint diagnostic page: http://<this-computer>:{port}"
        )

    def close_server(self):
        """Stop the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def state_payload(self):
        """Return the latest measured and commanded angles."""
        with self._lock:
            measured = list(self._measured)
            commanded = list(self._commanded)
            received = self._feedback_received
        age = None if received is None else time.monotonic() - received
        return {
            "joints": [
                {
                    "joint_id": index + 1,
                    "name": name,
                    "measured_degrees": _json_number(measured[index]),
                    "commanded_degrees": _json_number(commanded[index]),
                }
                for index, name in enumerate(JOINT_NAMES)
            ],
            "feedback_age_seconds": None if age is None else round(age, 3),
            "services_ready": self._services_ready(),
            "checking": self._motion_lock.locked(),
            "step_degrees": DIAGNOSTIC_STEP_DEGREES,
            "speed": DIAGNOSTIC_SPEED,
            "tolerance_degrees": DIAGNOSTIC_TOLERANCE_DEGREES,
        }

    def read_status(self):
        """Read controller status through the hardware service."""
        response, error = _call_service(
            self._status_client,
            Trigger.Request(),
            STATUS_TIMEOUT_SECONDS,
        )
        if error:
            return {"success": False, "message": error}
        return {
            "success": bool(response.success),
            "message": response.message,
        }

    def stop_motion(self):
        """Ask the hardware node to stop and cancel a multi-joint run."""
        self._stop_requested.set()
        response, error = _call_service(
            self._stop_client,
            Trigger.Request(),
            STATUS_TIMEOUT_SECONDS,
        )
        if error:
            return {"success": False, "message": error}
        return {
            "success": bool(response.success),
            "message": response.message,
        }

    def check_joint(self, joint_id):
        """Move one joint out by 2° and back only if that move succeeds."""
        if joint_id < 1 or joint_id > 6:
            raise ValueError("joint_id must be from 1 to 6")
        if not self._motion_lock.acquire(blocking=False):
            return {
                "success": False,
                "message": "A joint check is already running",
            }
        self._stop_requested.clear()
        try:
            return self._check_joint_locked(joint_id)
        finally:
            self._motion_lock.release()

    def check_all(self):
        """Check every joint and keep going after a failed move."""
        if not self._motion_lock.acquire(blocking=False):
            return {
                "success": False,
                "message": "A joint check is already running",
            }
        self._stop_requested.clear()
        reports = []
        try:
            for joint_id in range(1, 7):
                if self._stop_requested.is_set():
                    break
                reports.append(self._check_joint_locked(joint_id))
            return {
                "success": (
                    not self._stop_requested.is_set()
                    and len(reports) == 6
                    and all(item.get("verdict") == "tracked" for item in reports)
                ),
                "aborted": False,
                "reports": reports,
                "summary": summarize_reports(reports),
                "stopped": self._stop_requested.is_set(),
            }
        finally:
            self._motion_lock.release()

    def go_initial(self):
        """Walk every joint to 0° unless a move fails or Stop is pressed."""
        if not self._motion_lock.acquire(blocking=False):
            return {
                "success": False,
                "message": "A joint check is already running",
            }
        self._stop_requested.clear()
        joints = []
        try:
            for joint_id, goal in enumerate(INITIAL_POSITION_DEGREES, start=1):
                if self._stop_requested.is_set():
                    break
                joints.append(self._home_joint(joint_id, goal))
                if not joints[-1]["success"]:
                    break
            reached = (
                len(joints) == len(INITIAL_POSITION_DEGREES)
                and all(joint["success"] for joint in joints)
                and not self._stop_requested.is_set()
            )
            if self._stop_requested.is_set():
                message = "Move to the initial position stopped"
            elif reached:
                message = "Initial position reached"
            elif joints and not joints[-1]["success"]:
                message = joints[-1].get("message") or "Initial position did not finish"
            else:
                message = "Initial position did not finish"
            return {
                "success": reached,
                "aborted": False,
                "message": message,
                "joints": joints,
                "stopped": self._stop_requested.is_set(),
            }
        finally:
            self._motion_lock.release()

    def _home_joint(self, joint_id, goal):
        snapshot = self._snapshot()
        current = snapshot["measured"][joint_id - 1]
        start = current
        if current is None:
            return {
                "joint_id": joint_id,
                "success": False,
                "aborted": True,
                "abort_reason": "feedback",
                "start_degrees": None,
                "final_degrees": None,
                "steps": 0,
                "message": "Joint feedback is not available yet",
                "notes": [
                    "Invalid feedback stopped the sequence. "
                    "No recovery move was sent."
                ],
            }
        steps = 0
        message = "Already within 0.5° of 0°"
        while abs(current - goal) > DIAGNOSTIC_TOLERANCE_DEGREES:
            if self._stop_requested.is_set():
                message = "Move to the initial position stopped"
                break
            if steps >= MAX_HOME_STEPS:
                message = "Initial-position step limit reached before 0°"
                break
            target = step_toward(current, goal, DIAGNOSTIC_STEP_DEGREES)
            response, error = self._move(joint_id, round(target, 3))
            steps += 1
            if response is None:
                message = error
                break
            if (
                not response.success
                and abort_reason(False, response.message) == "movement"
                and "exceeds" in response.message
                and _finite(response.start_degrees)
            ):
                target = step_toward(
                    response.start_degrees,
                    goal,
                    DIAGNOSTIC_STEP_DEGREES,
                )
                response, error = self._move(joint_id, round(target, 3))
                steps += 1
                if response is None:
                    message = error
                    break
            if not response.success:
                message = response.message
                if _finite(response.final_degrees):
                    current = float(response.final_degrees)
                break
            if not _finite(response.final_degrees):
                message = response.message or "Final angle was not finite"
                break
            current = float(response.final_degrees)
            message = response.message
        succeeded = abs(current - goal) <= DIAGNOSTIC_TOLERANCE_DEGREES
        return {
            "joint_id": joint_id,
            "success": succeeded,
            "aborted": not succeeded and not self._stop_requested.is_set(),
            "abort_reason": (
                None
                if succeeded or self._stop_requested.is_set()
                else abort_reason(False, message)
            ),
            "start_degrees": _json_number(start),
            "final_degrees": _json_number(current),
            "steps": steps,
            "message": message,
            "notes": [
                "This initial-position sequence stopped on this joint. "
                "No later joint was commanded."
            ] if not succeeded and not self._stop_requested.is_set() else [],
        }

    def _check_joint_locked(self, joint_id):
        """Move one joint out and back. The caller holds the motion lock."""
        before = self._snapshot()
        start_angle = before["measured"][joint_id - 1]
        if start_angle is None:
            assessment = assess_move(
                joint_id,
                None,
                None,
                None,
                False,
                "Joint feedback is not available yet",
                [],
            )
            return _report(
                joint_id,
                None,
                None,
                assessment,
                [],
                "Joint feedback is not available yet",
                aborted=True,
                abort_reason_name="feedback",
            )

        target = round(start_angle + DIAGNOSTIC_STEP_DEGREES, 3)
        outward, outward_error = self._move(joint_id, target)
        after = self._wait_for_fresh_feedback(before["received"])
        companions = _companion_shifts(
            joint_id,
            before["measured"],
            after["measured"],
        )
        if outward is None:
            assessment = assess_move(
                joint_id,
                start_angle,
                target,
                None,
                False,
                outward_error,
                companions,
            )
            return _report(
                joint_id,
                None,
                None,
                assessment,
                companions,
                outward_error,
                target,
                aborted=True,
                abort_reason_name=abort_reason(False, outward_error),
                start_override=start_angle,
            )

        assessment = assess_move(
            joint_id,
            outward.start_degrees,
            target,
            outward.final_degrees,
            bool(outward.success),
            outward.message,
            companions,
        )
        reason = abort_reason(bool(outward.success), outward.message)
        if reason is not None:
            assessment["notes"] = list(assessment["notes"]) + [
                "The sequence stopped. This joint was not returned, "
                "no other joint will be tested, and initial position "
                "was not started."
            ]
            return _report(
                joint_id,
                outward,
                None,
                assessment,
                companions,
                outward.message,
                target,
                aborted=True,
                abort_reason_name=reason,
            )

        return_target = _return_target(
            outward.start_degrees,
            outward.final_degrees,
        )
        returned, return_error = self._move(joint_id, return_target)
        if returned is None:
            return_report = {
                "target_degrees": return_target,
                "message": return_error,
                "success": False,
            }
            return_reason = abort_reason(False, return_error)
        else:
            return_report = {
                "target_degrees": return_target,
                "start_degrees": _json_number(returned.start_degrees),
                "final_degrees": _json_number(returned.final_degrees),
                "success": bool(returned.success),
                "message": returned.message,
            }
            return_reason = abort_reason(
                bool(returned.success),
                returned.message,
            )
        if return_reason is not None:
            assessment["notes"] = list(assessment["notes"]) + [
                "The outward move succeeded, but the return move failed. "
                "No other joint was tested and initial position was not started."
            ]
        return _report(
            joint_id,
            outward,
            return_report,
            assessment,
            companions,
            outward.message,
            target,
            aborted=return_reason is not None,
            abort_reason_name=return_reason,
        )

    def _move(self, joint_id, target):
        request = MoveJoint.Request()
        request.joint_id = joint_id
        request.target_degrees = float(target)
        request.speed = DIAGNOSTIC_SPEED
        request.execute = True
        self.get_logger().info(
            f"Diagnostic J{joint_id} -> {target:.2f}° at speed "
            f"{DIAGNOSTIC_SPEED}"
        )
        return _call_service(
            self._move_client,
            request,
            MOVE_TIMEOUT_SECONDS,
        )

    def _snapshot(self):
        with self._lock:
            return {
                "measured": list(self._measured),
                "received": self._feedback_received,
            }

    def _wait_for_fresh_feedback(self, previous_received):
        deadline = time.monotonic() + FEEDBACK_WAIT_SECONDS
        while time.monotonic() < deadline:
            snapshot = self._snapshot()
            if (
                previous_received is None
                or snapshot["received"] != previous_received
            ):
                return snapshot
            time.sleep(0.05)
        return self._snapshot()

    def _services_ready(self):
        return (
            self._move_client.service_is_ready()
            and self._status_client.service_is_ready()
            and self._stop_client.service_is_ready()
        )

    def _on_measured(self, message):
        self._store_angles(message, commanded=False)

    def _on_commanded(self, message):
        self._store_angles(message, commanded=True)

    def _store_angles(self, message, commanded):
        angles = [None] * 6
        for index, name in enumerate(JOINT_NAMES):
            if name not in message.name:
                continue
            position_index = list(message.name).index(name)
            if position_index >= len(message.position):
                continue
            radians = float(message.position[position_index])
            if math.isfinite(radians):
                angles[index] = math.degrees(radians)
        with self._lock:
            if commanded:
                self._commanded = angles
                return
            self._measured = angles
            self._feedback_received = time.monotonic()
            stamp = message.header.stamp
            self._feedback_stamp = stamp.sec + stamp.nanosec * 1e-9


class DiagnosticHandler(BaseHTTPRequestHandler):
    """HTTP handlers for the diagnostic page."""

    def do_GET(self):
        """Serve the page or the live feedback JSON."""
        path = self.path.split("?", 1)[0]
        node = self.server.diagnostic_node
        if path in ("/", "/diagnostic"):
            self._serve_page(node)
            return
        if path == "/api/state":
            self._send_json(200, node.state_payload())
            return
        self._send_json(404, {"success": False, "message": "Not found"})

    def do_POST(self):
        """Run a status read, one joint check, all joints, or stop."""
        path = self.path.split("?", 1)[0]
        node = self.server.diagnostic_node
        try:
            payload = self._read_json()
            if path == "/api/status":
                body = node.read_status()
            elif path == "/api/check_joint":
                body = node.check_joint(_joint_id(payload.get("joint_id")))
            elif path == "/api/check_all":
                body = node.check_all()
            elif path == "/api/initial":
                body = node.go_initial()
            elif path == "/api/stop":
                body = node.stop_motion()
            else:
                self._send_json(404, {"success": False, "message": "Not found"})
                return
            self._send_json(200, body)
        except ValueError as error:
            self._send_json(400, {"success": False, "message": str(error)})

    def log_message(self, format, *args):
        """Drop the default access log."""
        del format, args

    def _serve_page(self, node):
        try:
            with open(node._page_path, "rb") as page_file:
                body = page_file.read()
        except OSError as error:
            self._send_json(
                500,
                {"success": False, "message": f"Could not read the page: {error}"},
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length < 0 or length > 8192:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _report(
    joint_id,
    response,
    return_report,
    assessment,
    companions,
    message,
    target=None,
    aborted=False,
    abort_reason_name=None,
    start_override=None,
):
    start = start_override if response is None else response.start_degrees
    final = None if response is None else response.final_degrees
    return {
        "joint_id": joint_id,
        "name": JOINT_NAMES[joint_id - 1],
        "success": assessment["verdict"] == "tracked" and not aborted,
        "aborted": aborted,
        "abort_reason": abort_reason_name,
        "verdict": assessment["verdict"],
        "notes": assessment["notes"],
        "command_delta": _json_number(assessment["command_delta"]),
        "measured_delta": _json_number(assessment["measured_delta"]),
        "error_degrees": _json_number(assessment["error_degrees"]),
        "target_degrees": _json_number(target),
        "start_degrees": _json_number(start),
        "final_degrees": _json_number(final),
        "message": message,
        "return_move": return_report,
        "companions": [
            {
                "joint_id": companion_id,
                "shift_degrees": _json_number(shift),
            }
            for companion_id, shift in companions
        ],
    }


def _return_target(start, final):
    if not _finite(start) or not _finite(final):
        return start
    delta = float(start) - float(final)
    if abs(delta) <= DIAGNOSTIC_STEP_DEGREES:
        return round(float(start), 3)
    return round(
        float(final) + math.copysign(DIAGNOSTIC_STEP_DEGREES, delta),
        3,
    )


def _companion_shifts(joint_id, before, after):
    shifts = []
    for index, (start, end) in enumerate(zip(before, after), start=1):
        if index == joint_id:
            continue
        if _finite(start) and _finite(end):
            shifts.append((index, end - start))
        else:
            shifts.append((index, None))
    return shifts


def _joint_id(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("joint_id must be an integer from 1 to 6")
    if value < 1 or value > 6:
        raise ValueError("joint_id must be an integer from 1 to 6")
    return value


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _json_number(value):
    if not _finite(value):
        return None
    return round(float(value), 3)


def _call_service(client, request, timeout_seconds):
    if not client.wait_for_service(timeout_sec=2.0):
        return None, "Hardware service is not available"
    future = client.call_async(request)
    finished = threading.Event()
    future.add_done_callback(lambda _future: finished.set())
    if not finished.wait(timeout_seconds):
        return None, "Hardware service call timed out"
    exception = future.exception()
    if exception is not None:
        return None, f"Hardware service call failed: {exception}"
    result = future.result()
    if result is None:
        return None, "Hardware service returned no response"
    return result, ""


def main(args=None):
    """Run the diagnostic page until the process is interrupted."""
    rclpy.init(args=args)
    node = JointDiagnosticUi()
    executor = MultiThreadedExecutor(num_threads=4)
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
