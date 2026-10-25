#!/usr/bin/env python3
"""Cap-based table calibration: place a held cap, measure, regrasp.

The cap must already be held in the gripper. For each spot the arm
releases the cap at a known pose, parks out of view, reads the refined
cap pixel (same refine_cap_center as picking), returns, and regrasps.
Spots tagged fit/check are also recorded in the :8092 wizard at the
regrasp pose. Every spot is appended to recal_cap_points.jsonl.

Usage: recal_cap_loop.py [--wizard] tag:x,y [tag:x,y ...]
  tag is fit, check or extra.
"""

import json
import subprocess
import sys
import time

sys.path.insert(0, "/home/er/mycobot_humble/test_runs")
import auto_calibrate as ac  # noqa: E402

OUT = "/home/er/mycobot_humble/test_runs/"
RELEASE_Z = 70.0
GRASP_Z = 66.0
TRAVEL = 150.0
PARK = (150.0, 110.0, 200.0)

_raw_set_gripper = ac.set_gripper
CARRYING = {"value": True}


def _guarded_set_gripper(state, speed=20):
    if CARRYING["value"] and state == 0:
        state = 1
    return _raw_set_gripper(state, speed)


ac.set_gripper = _guarded_set_gripper


def open_jaws():
    CARRYING["value"] = False
    return _raw_set_gripper(0, 10)


def close_jaws():
    CARRYING["value"] = True
    return _raw_set_gripper(1, 10)


def measure_cap(save_path=""):
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /root/mycobot_ws/install/setup.bash && "
        "cd /root/mycobot_ws/src/mycobot_vision/scripts && "
        "timeout 40 python3 /tmp/recal_measure_cap.py red_cap 10 %s"
        % save_path
    )
    result = subprocess.run(
        ["docker", "exec", "mycobot-humble-jetson", "bash", "-lc", cmd],
        capture_output=True, text=True, timeout=60,
    )
    for line in result.stdout.splitlines():
        if line.startswith("CAPPIX "):
            return json.loads(line[7:])
    raise RuntimeError("measure failed: " + result.stdout[-400:]
                       + result.stderr[-400:])


def pose_avg(samples=6):
    poses = []
    for _ in range(samples):
        pose = (ac.http_get("/api/state").get("wizard") or {}).get(
            "tool_pose_mm"
        )
        if pose:
            poses.append(pose[:3])
        time.sleep(0.15)
    return [round(sum(p[i] for p in poses) / len(poses), 2) for i in range(3)]


def main():
    args = sys.argv[1:]
    wizard = "--wizard" in args
    args = [a for a in args if a != "--wizard"]
    ac.focus()
    for item in args:
        tag, xy = item.split(":")
        x, y = (float(v) for v in xy.split(","))
        print("\n=== %s %.1f %.1f ===" % (tag, x, y), flush=True)
        ac.move_coords(x, y, TRAVEL, speed=12)
        ac.move_coords(x, y, RELEASE_Z + 15, speed=8)
        ac.move_coords(x, y, RELEASE_Z, speed=5)
        time.sleep(0.8)
        release_pose = pose_avg()
        open_jaws()
        time.sleep(1.0)
        ac.move_coords(x, y, TRAVEL, speed=8)
        ac.move_coords(*PARK, speed=12)
        time.sleep(1.2)
        name = "%s_%d_%d" % (tag, round(x), round(y))
        pix = measure_cap("/tmp/recal_cap_%s.jpg" % name)
        print("CAP", name, json.dumps(pix), flush=True)
        if not pix.get("n"):
            raise RuntimeError("cap not seen after release at %s" % name)
        ru, rv = pix["refined"]
        ac.move_coords(x, y, TRAVEL, speed=12)
        ac.move_coords(x, y, GRASP_Z + 15, speed=8)
        ac.move_coords(x, y, GRASP_Z, speed=5)
        time.sleep(1.0)
        grasp_pose = pose_avg()
        wiz_msg = None
        if wizard and tag in ("fit", "check"):
            ac.http_post("/api/calibrate/hold", {})
            time.sleep(0.4)
            ac.http_post("/api/calibrate/click", {"u": ru, "v": rv})
            body = ac.http_post("/api/calibrate/capture", {})
            wiz_msg = body.get("message")
            print("WIZARD", tag, body.get("success"), wiz_msg, flush=True)
        close_jaws()
        time.sleep(0.8)
        row = {
            "name": name, "tag": tag, "cmd": [x, y],
            "release_pose": release_pose, "grasp_pose": grasp_pose,
            "pix": pix, "wizard": wiz_msg,
        }
        with open(OUT + "recal_cap_points.jsonl", "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print("ROW", json.dumps(row), flush=True)
        ac.move_coords(x, y, TRAVEL, speed=8)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAILED:", error, flush=True)
        sys.exit(1)
