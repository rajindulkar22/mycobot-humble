"""Measure how the wrist image moves when the tool moves in X and Y.

Run inside the container with the pick stack up. The arm moves only in
small steps at travel height above one cap. Prints the 2x2 matrix that
maps a robot move (mm) to a cap shift in the wrist image (px).
"""

import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image

sys.path.insert(0, "/root/mycobot_ws/src/mycobot_vision/scripts")
import pick_place  # noqa: E402
from mycobot_interfaces.srv import MoveCoords  # noqa: E402

CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else "red_cap"
BASE_X = float(sys.argv[2]) if len(sys.argv) > 2 else 195.0
BASE_Y = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
BASE_Z = 150.0
STEP_MM = 10.0
ORIENTATION = (-170.0, 3.0, -125.0)


class Probe(Node):
    def __init__(self):
        super().__init__("wrist_jacobian_probe")
        self._bridge = CvBridge()
        self._frame = None
        self._stamp = 0.0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, "/wrist_camera/image_raw", self._on_image, qos,
        )
        self._move = self.create_client(MoveCoords, "mycobot/move_coords")

    def _on_image(self, message):
        self._frame = self._bridge.imgmsg_to_cv2(message, "bgr8")
        self._stamp = time.monotonic()

    def move(self, x, y, z):
        request = MoveCoords.Request()
        request.x, request.y, request.z = float(x), float(y), float(z)
        request.rx, request.ry, request.rz = ORIENTATION
        request.speed = 15
        request.execute = True
        future = self._move.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=25.0)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f"move failed: {None if result is None else result.message}")
        return list(result.final_coords)[:3]

    def cap_pixel(self, samples=5):
        found = []
        hint = None
        started = time.monotonic()
        while len(found) < samples and time.monotonic() - started < 6.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._frame is None or self._stamp < started:
                continue
            center = pick_place.wrist_cap_center(self._frame, CLASS_NAME, hint=hint)
            self._frame = None
            if center is not None:
                found.append(center)
                hint = center
        if len(found) < 3:
            return None
        return np.median(np.array(found), axis=0)


def main():
    rclpy.init()
    probe = Probe()
    if not probe._move.wait_for_service(timeout_sec=10.0):
        raise RuntimeError("move_coords not available")
    offsets = [
        (0.0, 0.0),
        (STEP_MM, 0.0),
        (-STEP_MM, 0.0),
        (0.0, 0.0),
        (0.0, STEP_MM),
        (0.0, -STEP_MM),
        (0.0, 0.0),
    ]
    rows = []
    for dx, dy in offsets:
        arrived = probe.move(BASE_X + dx, BASE_Y + dy, BASE_Z)
        time.sleep(1.0)
        pixel = probe.cap_pixel()
        print(f"cmd {dx:+.0f},{dy:+.0f} arrived {np.round(arrived, 1)} pixel {None if pixel is None else np.round(pixel, 1)}", flush=True)
        rows.append((np.array(arrived[:2]), pixel))
    moves, shifts = [], []
    reference = [r for r in rows if r[1] is not None]
    if len(reference) < 4:
        print("JACOBIAN none: cap not seen often enough")
        return
    centre = rows[0]
    for arrived, pixel in rows[1:]:
        if pixel is None or centre[1] is None:
            continue
        move = arrived - centre[0]
        if np.linalg.norm(move) < 3.0:
            continue
        moves.append(move)
        shifts.append(pixel - centre[1])
    moves = np.array(moves)
    shifts = np.array(shifts)
    solution, _, _, _ = np.linalg.lstsq(moves, shifts, rcond=None)
    jacobian = solution.T
    residual = shifts - (jacobian @ moves.T).T
    print("JACOBIAN", np.round(jacobian, 3).tolist())
    print("RESIDUAL_PX", np.round(np.abs(residual).max(), 1))
    print("GAIN_MM_PER_PX", np.round(-np.linalg.inv(jacobian), 4).tolist())
    probe.move(BASE_X, BASE_Y, BASE_Z)


if __name__ == "__main__":
    main()
