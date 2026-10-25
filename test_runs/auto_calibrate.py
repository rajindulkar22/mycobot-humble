#!/usr/bin/env python3
"""Drive the arm through fit spots and auto-capture pad centres via :8092."""

import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

BASE = "http://127.0.0.1:8092"
TABLE_Z = 64.0
HOVER_Z = 110.0
TRAVEL_Z = 150.0
RX, RY, RZ = -170.0, 3.0, -125.0
SPEED = 10
# Spread for Accept limits (≥100×80 px). Low-v ≈ front, high-v ≈ far Y-.
FIT = [
    (171.0, -118.0),  # mid-left
    (250.0, -95.0),   # mid-right
    (255.0, -125.0),  # far-right high-v
    (200.0, -45.0),   # front low-v
    (220.0, -132.0),  # far high-v
]
CHECK = (235.0, -75.0)
DROPS = {
    None: (250.0, -50.0),
    "red_cap": (270.0, 25.0),
    "white_cap": (270.0, -85.0),
    "black_cap": (270.0, -25.0),
}


def http_post(path, body=None):
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as decode_error:
            raise RuntimeError(
                "HTTP %s %s: %s" % (error.code, path, raw)
            ) from decode_error


def http_get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def camera_bgr():
    with urllib.request.urlopen(BASE + "/api/camera.jpg", timeout=30) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode overview JPEG")
    return image


def _pixel_ok(u, v, width, height):
    """Reject edge / cable false positives."""
    return (
        width * 0.16 < u < width * 0.84
        and height * 0.16 < v < height * 0.88
    )


def find_pad_center(image, hint=None):
    """Find open-pad midpoint; robust to dark frames and cable clutter."""
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y0 = int(height * 0.12)
    crop = gray[y0:, :]
    median = float(np.median(crop))
    # Relative + absolute thresholds so dark/shadowed frames still work.
    thresholds = []
    for delta in (20.0, 28.0, 35.0, 45.0):
        thresholds.append(min(median - delta, 100.0))
    for absolute in (45.0, 55.0, 70.0, 85.0, 100.0):
        thresholds.append(absolute)
    best = None
    best_score = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    x_cable = int(width * 0.14)
    for threshold in thresholds:
        if threshold < 12.0:
            continue
        mask = np.uint8(crop < threshold) * np.uint8(255)
        mask[:, :x_cable] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        tips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 45.0 or area > 2500.0:
                continue
            _center, size, _angle = cv2.minAreaRect(contour)
            short_side = min(float(size[0]), float(size[1]))
            long_side = max(float(size[0]), float(size[1]))
            if short_side < 3.5 or long_side / max(short_side, 1e-6) > 3.2:
                continue
            points = contour[:, 0, :]
            lowest = points[points[:, 1] >= points[:, 1].max() - 2]
            tip_u = float(lowest[:, 0].mean())
            tip_v = float(lowest[:, 1].mean()) + y0
            if not _pixel_ok(tip_u, tip_v, width, height):
                continue
            tips.append((tip_u, tip_v, area))
        if len(tips) < 2:
            continue
        for index, first in enumerate(tips):
            for second in tips[index + 1 :]:
                gap = math.hypot(first[0] - second[0], first[1] - second[1])
                if gap < 16.0 or gap > 70.0:
                    continue
                if abs(first[1] - second[1]) > 40.0:
                    continue
                mid = (
                    (first[0] + second[0]) / 2.0,
                    (first[1] + second[1]) / 2.0,
                )
                if not _pixel_ok(mid[0], mid[1], width, height):
                    continue
                area_ratio = max(first[2], second[2]) / max(
                    min(first[2], second[2]), 1.0
                )
                if area_ratio > 4.0:
                    continue
                # Prefer compact similar pads near image centre / hint.
                score = (first[2] + second[2]) / (
                    1.0 + abs(first[1] - second[1]) + abs(gap - 35.0)
                )
                score -= area_ratio * 15.0
                score -= abs(mid[0] - width * 0.5) * 0.05
                score -= abs(mid[1] - height * 0.45) * 0.04
                if hint is not None:
                    score -= math.hypot(mid[0] - hint[0], mid[1] - hint[1]) * 0.35
                if best is None or score > best_score:
                    best = mid
                    best_score = score
    return best


def docker_ros(cmd):
    full = (
        "source /opt/ros/humble/setup.bash && "
        "source /root/mycobot_ws/install/setup.bash && "
        + cmd
    )
    result = subprocess.run(
        ["docker", "exec", "mycobot-humble-jetson", "bash", "-lc", full],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "docker ros failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
    return result.stdout


def move_coords(x, y, z, speed=SPEED):
    cmd = (
        "ros2 service call /mycobot/move_coords "
        "mycobot_interfaces/srv/MoveCoords "
        "\"{x: %.3f, y: %.3f, z: %.3f, rx: %.3f, ry: %.3f, rz: %.3f, "
        "speed: %d, execute: true}\""
        % (x, y, z, RX, RY, RZ, speed)
    )
    out = docker_ros(cmd)
    print("move", round(x, 1), round(y, 1), round(z, 1), "->", out.strip()[-200:])
    if "success=False" in out or "success: false" in out.lower():
        if "success=True" not in out and "success: true" not in out.lower():
            # Clear stop latch / error -1 and retry once.
            focus()
            set_gripper(0, 15)
            time.sleep(0.8)
            out = docker_ros(cmd)
            print(
                "move-retry",
                round(x, 1),
                round(y, 1),
                round(z, 1),
                "->",
                out.strip()[-200:],
            )
            if "success=False" in out or "success: false" in out.lower():
                if "success=True" not in out and "success: true" not in out.lower():
                    raise RuntimeError("move_coords failed: " + out)
    return out


def set_gripper(state, speed=20):
    cmd = (
        "ros2 service call /mycobot/set_gripper "
        "mycobot_interfaces/srv/SetGripper "
        "\"{state: %d, speed: %d, execute: true}\"" % (state, speed)
    )
    out = docker_ros(cmd)
    print("gripper", state, "->", out.strip()[-160:])
    return out


def focus():
    out = docker_ros(
        "ros2 service call /mycobot/focus_servos std_srvs/srv/Trigger {}"
    )
    print("focus", out.strip()[-120:])


def wait_pose_near(x, y, z, tol_xy=12.0, tol_z=10.0, seconds=20.0):
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline:
        state = http_get("/api/state")
        pose = (state.get("wizard") or {}).get("tool_pose_mm")
        last = pose
        if pose and len(pose) >= 3:
            if (
                abs(pose[0] - x) <= tol_xy
                and abs(pose[1] - y) <= tol_xy
                and abs(pose[2] - z) <= tol_z
            ):
                return pose
        time.sleep(0.25)
    raise RuntimeError("pose not near target; last=%s want=%s" % (last, (x, y, z)))


def _pixel_from_auto(label, body):
    wiz = body.get("wizard") or http_get("/api/state").get("wizard") or {}
    pairs = list(wiz.get("pairs") or [])
    check = wiz.get("check")
    spot = check if check and label.startswith("check") else (
        pairs[-1] if pairs else None
    )
    if spot is None:
        return None
    return (float(spot["u"]), float(spot["v"]))


# Seed from live overview after camera/view shift (rejects ~u=150 false blob).
_SEED_PAIRS = [
    (171.0, -118.0, 217.0, 151.0),
    (250.0, -95.0, 337.0, 147.0),
    (255.0, -125.0, 340.0, 187.0),
    (200.0, -55.0, 270.0, 100.0),
    (145.0, -105.0, 190.0, 160.0),
]


def _affine_from_pairs(pairs_xyz_uv):
    matrix = np.array(
        [[p[0], p[1], 1.0] for p in pairs_xyz_uv], dtype=float
    )
    uv = np.array([[p[2], p[3]] for p in pairs_xyz_uv], dtype=float)
    coeff, *_ = np.linalg.lstsq(matrix, uv, rcond=None)
    return coeff


_SEED_AFFINE = _affine_from_pairs(_SEED_PAIRS)


def _predict_uv(x, y, pairs):
    """Predict pad pixel; prefer live pairs, else seeded affine + v bias."""
    live = [
        (float(p["x"]), float(p["y"]), float(p["u"]), float(p["v"]))
        for p in pairs
        if p.get("u") is not None
    ]
    if len(live) >= 3:
        coeff = _affine_from_pairs(live)
        pred = np.array([x, y, 1.0], dtype=float) @ coeff
        return float(pred[0]), float(pred[1])
    seed = np.array([x, y, 1.0], dtype=float) @ _SEED_AFFINE
    su, sv = float(seed[0]), float(seed[1])
    if live:
        # Correct vertical shift from camera angle change using fit1-like bias.
        biases = []
        for lx, ly, lu, lv in live:
            s = np.array([lx, ly, 1.0], dtype=float) @ _SEED_AFFINE
            biases.append((lu - float(s[0]), lv - float(s[1])))
        bu = float(np.mean([b[0] for b in biases]))
        bv = float(np.mean([b[1] for b in biases]))
        return su + bu, sv + bv
    return su, sv


def capture_at_pose(label, x, y, z_table, hint=None):
    """Save one spot: locate pads above the table, capture at table Z."""
    pixel = None
    used_z = None
    prior_pairs = list((http_get("/api/state").get("wizard") or {}).get("pairs") or [])
    predicted = _predict_uv(x, y, prior_pairs)
    if predicted is not None:
        print(label, "predict", round(predicted[0], 1), round(predicted[1], 1))
    for sight_z in (min(z_table + 22.0, 88.0), min(z_table + 30.0, 95.0), 100.0):
        move_coords(x, y, sight_z, speed=6)
        for attempt in range(4):
            time.sleep(0.45 + 0.25 * attempt)
            image = camera_bgr()
            height, width = image.shape[:2]
            cand = None
            source = None
            # Prefer server auto — local alone often locks onto false blobs.
            body = http_post("/api/calibrate/auto", {})
            print(
                label,
                "sight_z",
                sight_z,
                "try",
                attempt + 1,
                "auto",
                body.get("message"),
            )
            if body.get("success"):
                auto_pix = _pixel_from_auto(label, body)
                http_post("/api/calibrate/undo", {})
                if auto_pix is not None and _pixel_ok(
                    auto_pix[0], auto_pix[1], width, height
                ):
                    cand = auto_pix
                    source = "auto"
            if cand is None:
                local = find_pad_center(
                    image, hint=predicted if predicted is not None else hint
                )
                if local is not None and _pixel_ok(local[0], local[1], width, height):
                    cand = local
                    source = "local"
                    print(
                        label,
                        "local",
                        round(local[0], 1),
                        round(local[1], 1),
                    )
            if cand is not None and predicted is not None:
                err = math.hypot(cand[0] - predicted[0], cand[1] - predicted[1])
                # Gate hard on u (false blob ~150); allow larger v drift.
                u_err = abs(cand[0] - predicted[0])
                v_err = abs(cand[1] - predicted[1])
                bad = u_err > 70.0 or (prior_pairs and (err > 110.0 or v_err > 100.0))
                if bad:
                    print(
                        label,
                        "reject",
                        source,
                        round(cand[0], 1),
                        round(cand[1], 1),
                        "pred_err",
                        round(err, 1),
                        "u_err",
                        round(u_err, 1),
                    )
                    cand = None
            if cand is not None:
                pixel = cand
                used_z = sight_z
                print(label, "use", source, round(pixel[0], 1), round(pixel[1], 1))
                break
        if pixel is not None:
            break
    if pixel is None:
        raise RuntimeError("%s: pads not visible at sight height" % label)
    print(label, "pad pixel", round(pixel[0], 1), round(pixel[1], 1), "from_z", used_z)
    move_coords(x, y, z_table, speed=5)
    wait_pose_near(x, y, z_table, tol_xy=15.0, tol_z=12.0)
    time.sleep(0.5)
    # Only adopt table-height re-detect when it stays near the sight pad.
    image = camera_bgr()
    table_pixel = find_pad_center(image, hint=pixel)
    if table_pixel is not None:
        height, width = image.shape[:2]
        shift = math.hypot(table_pixel[0] - pixel[0], table_pixel[1] - pixel[1])
        if (
            _pixel_ok(table_pixel[0], table_pixel[1], width, height)
            and shift <= 40.0
        ):
            print(
                label,
                "table pad",
                round(table_pixel[0], 1),
                round(table_pixel[1], 1),
                "shift",
                round(shift, 1),
            )
            pixel = table_pixel
        elif table_pixel is not None:
            print(
                label,
                "ignore table pad",
                round(table_pixel[0], 1),
                round(table_pixel[1], 1),
                "shift",
                round(shift, 1),
            )
    http_post("/api/calibrate/hold", {})
    click = http_post(
        "/api/calibrate/click",
        {"u": float(pixel[0]), "v": float(pixel[1])},
    )
    print(label, "click", click.get("message"))
    body = http_post("/api/calibrate/capture", {})
    print(label, "capture", body.get("message"))
    if not body.get("success"):
        raise RuntimeError("%s capture failed: %s" % (label, body.get("message")))
    return body, pixel


def go_table_point(x, y, z_table):
    # Always lift before sideways travel.
    state = http_get("/api/state")
    pose = (state.get("wizard") or {}).get("tool_pose_mm") or [x, y, TRAVEL_Z]
    if pose[2] < TRAVEL_Z - 5:
        move_coords(pose[0], pose[1], TRAVEL_Z, speed=12)
    move_coords(x, y, TRAVEL_Z, speed=12)
    move_coords(x, y, HOVER_Z, speed=8)
    wait_pose_near(x, y, HOVER_Z, tol_xy=18.0, tol_z=20.0)


def camera_is_live():
    import hashlib

    first = urllib.request.urlopen(BASE + "/api/camera.jpg", timeout=20).read()
    time.sleep(0.35)
    second = urllib.request.urlopen(BASE + "/api/camera.jpg", timeout=20).read()
    return hashlib.md5(first).digest() != hashlib.md5(second).digest()


def wizard_pairs():
    return list((http_get("/api/state").get("wizard") or {}).get("pairs") or [])


def main():
    if not camera_is_live():
        raise RuntimeError("Overview camera is frozen — restart pick_place before calibrating")
    print("camera live OK")
    print("Clear + hold + open gripper")
    print(http_post("/api/calibrate/clear", {}).get("message"))
    print(http_post("/api/calibrate/clearances", {"table_z_mm": TABLE_Z, "hover_mm": 40}).get("message"))
    focus()
    http_post("/api/calibrate/hold", {})
    # Opening the gripper also clears a latched Stop so move_coords works.
    set_gripper(0, 15)
    time.sleep(1.5)

    z_touch = TABLE_Z + 1.5
    hint = None
    fit_alts = {
        4: [(210.0, -50.0), (185.0, -70.0), (220.0, -90.0)],
        5: [(155.0, -90.0), (130.0, -115.0), (160.0, -75.0)],
    }
    for index, (x, y) in enumerate(FIT, start=1):
        if not camera_is_live():
            raise RuntimeError("Overview camera froze before fit %d" % index)
        print("\n=== FIT %d/5 at %.1f, %.1f ===" % (index, x, y))
        candidates = [(x, y)] + fit_alts.get(index, [])
        last_error = None
        for cx, cy in candidates:
            try:
                go_table_point(cx, cy, z_touch)
                _body, hint = capture_at_pose(
                    "fit%d" % index, cx, cy, z_touch, hint=hint
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                print("fit try failed", cx, cy, error)
        if last_error is not None:
            raise last_error
        pairs = wizard_pairs()
        print(
            "pairs after",
            len(pairs),
            "last",
            {k: pairs[-1].get(k) for k in ("u", "v", "x", "y", "z")},
        )
        if len(pairs) != index:
            raise RuntimeError(
                "pair count mismatch: have %d want %d" % (len(pairs), index)
            )

    print("\n=== CHECK ===")
    check_ok = False
    for cx, cy in [CHECK, (240.0, -100.0), (215.0, -85.0), (190.0, -95.0)]:
        try:
            go_table_point(cx, cy, z_touch)
            capture_at_pose("check", cx, cy, z_touch, hint=hint)
            wiz = http_get("/api/state").get("wizard") or {}
            if wiz.get("check"):
                print("check saved", wiz.get("check"))
                check_ok = True
                break
            print("check not stored yet; pairs=", len(wiz.get("pairs") or []))
        except Exception as error:
            print("check try failed", cx, cy, error)
    if not check_ok:
        raise RuntimeError("check spot was not saved")

    for name, (x, y) in DROPS.items():
        label = name or "shared"
        print("\n=== DROP %s at %.1f, %.1f ===" % (label, x, y))
        go_table_point(x, y, z_touch)
        http_post("/api/calibrate/hold", {})
        body = {"class_name": name} if name else {}
        resp = http_post("/api/calibrate/drop", body)
        print("drop", label, resp.get("message"), resp.get("wizard", {}).get("drop_x_mm"))

    # Lift clear before accept
    state = http_get("/api/state")
    pose = (state.get("wizard") or {}).get("tool_pose_mm") or [200.0, -50.0, TRAVEL_Z]
    move_coords(pose[0], pose[1], TRAVEL_Z, speed=10)

    print("\n=== ACCEPT ===")
    resp = http_post("/api/calibrate/accept", {})
    print(json.dumps(resp, indent=2)[:4000])
    if not resp.get("success"):
        return 1
    print("READY", http_get("/api/state").get("calibration_ready"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("FAILED:", error, file=sys.stderr)
        try:
            http_post("/api/stop", {})
        except Exception:
            pass
        sys.exit(1)
