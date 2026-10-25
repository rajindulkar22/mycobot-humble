#!/usr/bin/env python3
"""Hover open fingers over each cap and record pad-vs-centre error."""
import json
import math
import os
import time
import urllib.request

import cv2
import numpy as np
import rclpy
from mycobot_interfaces.srv import MoveCoords, SetGripper
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

OUT = "/root/mycobot_ws/.grasp_measure"
RX, RY, RZ = -170.0, 3.0, -125.0
TRAVEL_Z = 150.0
HOVER_Z = 108.0
SPEED = 12

# Affine from current accepted pairs (src yaml).
PAIRS = [
    (222.3, 285.5, 171.2, -118.4),
    (341.4, 211.3, 248.7, -42.6),
    (340.4, 333.0, 257.2, -129.2),
    (260.8, 201.1, 173.2, -54.7),
    (185.4, 228.4, 138.7, -88.4),
]


def fit_affine(pairs):
    A = np.array([[u, v, 1.0] for u, v, _x, _y in pairs], float)
    bx = np.array([x for _u, _v, x, _y in pairs], float)
    by = np.array([y for _u, _v, _x, y in pairs], float)
    cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    return cx, cy


def project(cx, cy, u, v):
    x = cx[0] * u + cx[1] * v + cx[2]
    y = cy[0] * u + cy[1] * v + cy[2]
    return x, y


def get_state():
    raw = urllib.request.urlopen("http://127.0.0.1:8092/api/state", timeout=5).read()
    return json.loads(raw)


def get_image():
    raw = urllib.request.urlopen("http://127.0.0.1:8092/api/camera.jpg", timeout=5).read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def finger_pads(image):
    """Return left-pad, right-pad, midpoint in image pixels, or None."""
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White plastic fingers.
    white = cv2.inRange(hsv, (0, 0, 185), (180, 55, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    h, w = image.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80 or area > 8000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        # Ignore huge background and the far-left housing.
        if x < 20 or y < 10:
            continue
        if bw > w * 0.45 or bh > h * 0.45:
            continue
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        blobs.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area, bw, bh))
    if len(blobs) < 2:
        return None
    # Prefer two nearby similar-size blobs (the two fingers).
    best = None
    for i, a in enumerate(blobs):
        for b in blobs[i + 1 :]:
            gap = math.hypot(a[0] - b[0], a[1] - b[1])
            if gap < 18 or gap > 90:
                continue
            score = abs(a[2] - b[2]) + abs(a[1] - b[1]) * 0.3
            if best is None or score < best[0]:
                best = (score, a, b)
    if best is None:
        # Fallback: two rightmost blobs (gripper is usually right of the wrist).
        blobs.sort(key=lambda t: t[0], reverse=True)
        a, b = blobs[0], blobs[1]
    else:
        a, b = best[1], best[2]
    if a[0] <= b[0]:
        left, right = a, b
    else:
        left, right = b, a
    mid = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    return {
        "left": (left[0], left[1]),
        "right": (right[0], right[1]),
        "mid": mid,
        "gap_px": math.hypot(left[0] - right[0], left[1] - right[1]),
    }


class Measure(Node):
    def __init__(self):
        super().__init__("grasp_offset_measure")
        self.pose = None
        self.create_subscription(Float64MultiArray, "mycobot/tool_pose_raw", self._on_pose, 10)
        self.coords = self.create_client(MoveCoords, "mycobot/move_coords")
        self.grip = self.create_client(SetGripper, "mycobot/set_gripper")
        if not self.coords.wait_for_service(timeout_sec=8.0):
            raise RuntimeError("move_coords missing")
        if not self.grip.wait_for_service(timeout_sec=8.0):
            raise RuntimeError("set_gripper missing")

    def _on_pose(self, msg):
        if len(msg.data) >= 6:
            self.pose = [float(v) for v in msg.data[:6]]

    def wait_pose(self, timeout=4.0):
        deadline = time.time() + timeout
        while time.time() < deadline and self.pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pose

    def call_move(self, x, y, z, speed=SPEED):
        last = None
        for attempt in range(3):
            req = MoveCoords.Request()
            req.x, req.y, req.z = float(x), float(y), float(z)
            req.rx, req.ry, req.rz = RX, RY, RZ
            req.speed = int(speed)
            req.execute = True
            fut = self.coords.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=40.0)
            res = fut.result()
            if res is None:
                last = RuntimeError("move_coords timed out")
                time.sleep(0.8)
                continue
            print(
                f"MOVE {x:.1f},{y:.1f},{z:.1f} try={attempt+1} "
                f"success={res.success} {res.message} "
                f"final={list(res.final_coords)[:3]}"
            )
            if res.success:
                time.sleep(0.3)
                return res
            last = RuntimeError(res.message)
            # -1 is a serial flake; wait and retry.
            if "error -1" not in (res.message or ""):
                break
            time.sleep(1.0)
        raise last

    def call_grip(self, state, speed=15):
        req = SetGripper.Request()
        req.state = int(state)
        req.speed = int(speed)
        req.execute = True
        fut = self.grip.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=12.0)
        res = fut.result()
        if res is None:
            raise RuntimeError("set_gripper timed out")
        print(f"GRIP state={state} success={res.success} {res.message} jaw={res.final_value}")
        return res


def annotate(image, cap, pads, path):
    vis = image.copy()
    cu, cv = int(round(cap[0])), int(round(cap[1]))
    cv2.circle(vis, (cu, cv), 8, (0, 255, 255), 2)
    cv2.drawMarker(vis, (cu, cv), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    if pads:
        lu, lv = int(round(pads["left"][0])), int(round(pads["left"][1]))
        ru, rv = int(round(pads["right"][0])), int(round(pads["right"][1]))
        mu, mv = int(round(pads["mid"][0])), int(round(pads["mid"][1]))
        cv2.circle(vis, (lu, lv), 6, (255, 80, 80), 2)
        cv2.circle(vis, (ru, rv), 6, (80, 80, 255), 2)
        cv2.circle(vis, (mu, mv), 5, (0, 255, 0), 2)
        cv2.line(vis, (lu, lv), (ru, rv), (0, 255, 0), 1)
        cv2.line(vis, (mu, mv), (cu, cv), (255, 255, 0), 1)
    cv2.imwrite(path, vis)


def main():
    os.makedirs(OUT, exist_ok=True)
    cx, cy = fit_affine(PAIRS)
    print("affine cx", cx)
    print("affine cy", cy)
    print("du+10 ->", project(cx, cy, 10, 0)[0] - project(cx, cy, 0, 0)[0],
          project(cx, cy, 10, 0)[1] - project(cx, cy, 0, 0)[1])
    print("dv+10 ->", project(cx, cy, 0, 10)[0] - project(cx, cy, 0, 0)[0],
          project(cx, cy, 0, 10)[1] - project(cx, cy, 0, 0)[1])

    rclpy.init()
    node = Measure()
    pose = node.wait_pose()
    print("pose", pose)
    if pose is None:
        raise RuntimeError("no tool pose")

    state = get_state()
    dets = state.get("detections") or {}
    print("dets", {k: (round(v["u"],1), round(v["v"],1), v.get("x_mm"), v.get("y_mm")) for k, v in dets.items()})
    if not dets:
        raise RuntimeError("no detections")

    node.call_grip(0, 15)
    travel_z = max(TRAVEL_Z, float(pose[2]))
    if float(pose[2]) < TRAVEL_Z - 5.0:
        node.call_move(pose[0], pose[1], travel_z, 12)
    else:
        print(f"already high enough at z={pose[2]:.1f}, skip park lower")

    results = {}
    order = ["white_cap", "black_cap", "red_cap"]
    for name in order:
        det = dets.get(name)
        if not det:
            print("skip missing", name)
            continue
        x, y = project(cx, cy, det["u"], det["v"])
        print(f"\n=== {name} mapped {x:.1f},{y:.1f} px {det['u']:.1f},{det['v']:.1f} ===")
        node.call_move(x, y, travel_z, 12)
        node.call_move(x, y, HOVER_Z, 8)
        time.sleep(0.6)
        # refresh detection after settle
        state = get_state()
        det2 = (state.get("detections") or {}).get(name) or det
        img = get_image()
        raw_path = os.path.join(OUT, f"{name}_raw.jpg")
        vis_path = os.path.join(OUT, f"{name}_vis.jpg")
        cv2.imwrite(raw_path, img)
        pads = finger_pads(img)
        cap = (float(det2["u"]), float(det2["v"]))
        annotate(img, cap, pads, vis_path)
        rec = {
            "cap_px": cap,
            "mapped": (x, y),
            "pads": pads,
        }
        if pads:
            du = pads["mid"][0] - cap[0]
            dv = pads["mid"][1] - cap[1]
            mx, my = project(cx, cy, pads["mid"][0], pads["mid"][1])
            rec["du"] = du
            rec["dv"] = dv
            rec["pad_mapped"] = (mx, my)
            # Offset that would put the pad midpoint on the cap.
            rec["offset_x"] = x - mx
            rec["offset_y"] = y - my
            print(
                f"padL={pads['left'][0]:.1f},{pads['left'][1]:.1f} "
                f"padR={pads['right'][0]:.1f},{pads['right'][1]:.1f} "
                f"mid={pads['mid'][0]:.1f},{pads['mid'][1]:.1f} "
                f"cap={cap[0]:.1f},{cap[1]:.1f} "
                f"dpx=({du:+.1f},{dv:+.1f}) offset_mm=({rec['offset_x']:+.1f},{rec['offset_y']:+.1f})"
            )
        else:
            print("pads not found")
        results[name] = rec
        node.call_move(x, y, travel_z, 10)

    # Park away from the table centre so the overview is clear.
    node.call_move(110.0, -30.0, travel_z, 12)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("\nDONE", json.dumps(results, indent=2, default=float))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
