#!/usr/bin/env python3
"""Run inside the container: median refined red-cap pixel from raw frames.

Prints one JSON line: {"raw": [u, v], "refined": [u, v], "n": count}.
Uses refine_cap_center from pick_place.py so the pixel matches picking.
"""

import json
import statistics
import sys
import time
import urllib.request

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

sys.path.insert(0, "/root/mycobot_ws/src/mycobot_vision/scripts")
from pick_place import refine_cap_center  # noqa: E402

CLASS = sys.argv[1] if len(sys.argv) > 1 else "red_cap"
WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SAVE = sys.argv[3] if len(sys.argv) > 3 else ""


class Grab(Node):
    def __init__(self):
        super().__init__("recal_measure_cap")
        self.bridge = CvBridge()
        self.frame = None
        self.stamp = 0.0
        self.create_subscription(
            Image, "/overview_camera/image_rect", self.on_image,
            qos_profile_sensor_data,
        )

    def on_image(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.stamp = time.monotonic()


def detection():
    with urllib.request.urlopen(
        "http://127.0.0.1:8092/api/state", timeout=5
    ) as resp:
        state = json.loads(resp.read().decode("utf-8"))
    return (state.get("detections") or {}).get(CLASS)


def main():
    rclpy.init()
    node = Grab()
    raw, refined = [], []
    last = 0.0
    deadline = time.monotonic() + 15.0
    while len(refined) < WANT and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.frame is None or node.stamp == last:
            continue
        last = node.stamp
        det = detection()
        if not det or float(det.get("age_seconds") or 9) > 1.0:
            continue
        u, v = float(det["u"]), float(det["v"])
        ru, rv = refine_cap_center(
            node.frame, CLASS, u, v,
            float(det.get("size_x", 0.0)), float(det.get("size_y", 0.0)),
        )
        raw.append((u, v))
        refined.append((ru, rv))
        if SAVE and len(refined) == 1:
            cv2.imwrite(SAVE, node.frame)
        time.sleep(0.12)
    out = {"n": len(refined)}
    if refined:
        out["raw"] = [round(statistics.median(p[i] for p in raw), 2)
                      for i in range(2)]
        out["refined"] = [round(statistics.median(p[i] for p in refined), 2)
                          for i in range(2)]
        out["spread"] = [
            round(max(p[i] for p in refined) - min(p[i] for p in refined), 2)
            for i in range(2)
        ]
    print("CAPPIX " + json.dumps(out))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
