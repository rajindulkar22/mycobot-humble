#!/usr/bin/env python3
"""Touch table points and save overview frames for pad-centre measurement.

Usage: recal_collect.py x,y [x,y ...]
Writes recal_pt_<x>_<y>.jpg, recal_diff_<x>_<y>.jpg and appends to
recal_points.jsonl (commanded + measured pose, tip estimate).
"""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/er/mycobot_humble/test_runs")
import auto_calibrate as ac  # noqa: E402

OUT = "/home/er/mycobot_humble/test_runs/"
Z_TOUCH = ac.TABLE_Z + 1.5
TRAVEL = 150.0


def median_frame(count=5, gap=0.15):
    frames = []
    for _ in range(count):
        frames.append(ac.camera_bgr())
        time.sleep(gap)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def pose_avg(samples=6):
    poses = []
    for _ in range(samples):
        pose = (ac.http_get("/api/state").get("wizard") or {}).get(
            "tool_pose_mm"
        )
        if pose:
            poses.append(pose[:3])
        time.sleep(0.15)
    return [round(float(v), 2) for v in np.mean(poses, axis=0)]


def tip_estimate(bg, img):
    """Lowest point (max v) of the arm silhouette in the diff."""
    diff = cv2.absdiff(
        cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0),
        cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0),
    )
    mask = np.uint8(diff > 28) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None, mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == biggest)
    vmax = ys.max()
    sel = ys >= vmax - 3
    return (float(xs[sel].mean()), float(vmax)), mask


def main():
    bg = cv2.imread(OUT + "recal_bg.jpg")
    closed = os.environ.get("GRIP") == "closed"
    suffix = "_c" if closed else ""
    ac.focus()
    ac.set_gripper(1 if closed else 0, 15)
    time.sleep(0.8)
    for arg in sys.argv[1:]:
        x, y = (float(v) for v in arg.split(","))
        ac.move_coords(x, y, TRAVEL, speed=12)
        ac.move_coords(x, y, Z_TOUCH + 15, speed=8)
        if closed:
            ac.set_gripper(1, 15)
        ac.move_coords(x, y, Z_TOUCH, speed=5)
        time.sleep(1.5)
        img = median_frame()
        pose = pose_avg()
        tip, mask = tip_estimate(bg, img)
        tag = "%d_%d%s" % (round(x), round(y), suffix)
        cv2.imwrite(OUT + "recal_pt_%s.jpg" % tag, img)
        cv2.imwrite(OUT + "recal_diff_%s.jpg" % tag, mask)
        row = {"cmd": [x, y, Z_TOUCH], "pose": pose, "tip": tip, "tag": tag}
        print("POINT", json.dumps(row))
        with open(OUT + "recal_points.jsonl", "a") as handle:
            handle.write(json.dumps(row) + "\n")
        ac.move_coords(x, y, TRAVEL, speed=10)


if __name__ == "__main__":
    main()
