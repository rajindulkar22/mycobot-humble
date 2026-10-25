#!/usr/bin/env python3
"""Guided ArUco table calibration via the pick UI on :8092.

Print DICT_4X4_50 id 0 from http://127.0.0.1:8092/aruco (or --export-marker).
For each fit/check spot: tape the marker flat, put the open pad centre on the
marker centre at table z ≈ 64 mm, then this script Hold + POST /api/calibrate/aruco.

Default mode moves the arm near each target XY and waits for Enter.
Use --prompt-only to skip motion (hand-guide only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

# Reuse motion helpers from the pad-based driver.
sys.path.insert(0, "/home/er/mycobot_humble/test_runs")
import auto_calibrate as ac  # noqa: E402

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_ID = 0
MIN_SPAN_U = 100.0
MIN_SPAN_V = 80.0
MAX_DROP_X = 275.0


def _aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)


def _aruco_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def detect_aruco_uv(image, marker_id=ARUCO_ID):
    """Return (u, v) of marker centre, or None."""
    if image is None or image.size == 0:
        return None
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    dictionary = _aruco_dictionary()
    params = _aruco_parameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params
        )
    if ids is None:
        return None
    for index, found in enumerate(ids.flatten().tolist()):
        if int(found) != int(marker_id):
            continue
        points = corners[index][0]
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    return None


def _draw_marker_image(size_px):
    """Return a square marker image (OpenCV 4.5 drawMarker or 4.7+ generate)."""
    dictionary = _aruco_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, ARUCO_ID, size_px)
    if hasattr(cv2.aruco, "drawMarker"):
        return cv2.aruco.drawMarker(dictionary, ARUCO_ID, size_px)
    raise RuntimeError("This OpenCV build cannot draw ArUco markers")


def export_marker_png(path, size_px=400, border_px=60):
    """Write a printable ArUco id-0 PNG with white margin."""
    try:
        marker = _draw_marker_image(size_px)
        canvas = (
            np.ones(
                (size_px + 2 * border_px, size_px + 2 * border_px),
                dtype=np.uint8,
            )
            * 255
        )
        canvas[
            border_px : border_px + size_px,
            border_px : border_px + size_px,
        ] = marker
        if not cv2.imwrite(path, canvas):
            raise RuntimeError("Could not write %s" % path)
    except Exception as error:
        # Fall back to the pick UI endpoint (same marker the robot uses).
        print("local draw failed (%s); fetching from %s" % (error, ac.BASE))
        data = urllib.request.urlopen(
            ac.BASE + "/api/aruco/marker.png", timeout=20
        ).read()
        with open(path, "wb") as handle:
            handle.write(data)
    print("Wrote", path)
    return path


def wait_enter(prompt):
    try:
        return input(prompt)
    except EOFError:
        print("\n(no TTY — continuing after 2 s)")
        time.sleep(2.0)
        return ""


def aruco_visible(timeout_s=2.0):
    """Poll overview JPEG until marker is seen or timeout."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            img = ac.camera_bgr()
            last = detect_aruco_uv(img)
            if last is not None:
                return last
        except Exception as error:
            print("camera err", error)
        time.sleep(0.25)
    return last


def capture_aruco_spot(label, require_visible=True, retries=6):
    """Hold motors and POST /api/calibrate/aruco until success."""
    for attempt in range(1, retries + 1):
        uv = aruco_visible(timeout_s=1.5)
        print(
            label,
            "attempt",
            attempt,
            "aruco",
            None if uv is None else (round(uv[0], 1), round(uv[1], 1)),
        )
        if require_visible and uv is None:
            wait_enter(
                "%s: marker not visible. Fix lighting/framing, then Enter: "
                % label
            )
            continue
        ac.http_post("/api/calibrate/hold", {})
        time.sleep(0.25)
        resp = ac.http_post("/api/calibrate/aruco", {})
        print(label, "aruco-capture", resp.get("message"))
        if resp.get("success"):
            wiz = ac.http_get("/api/state").get("wizard") or {}
            spot = (
                wiz.get("check")
                if label.startswith("check")
                else (wiz.get("pairs") or [None])[-1]
            )
            print(label, "OK", {k: spot.get(k) for k in ("u", "v", "x", "y", "z", "source")} if spot else spot)
            return spot
        wait_enter("%s failed (%s). Adjust, then Enter: " % (label, resp.get("message")))
    raise RuntimeError("%s: ArUco capture failed after %d tries" % (label, retries))


def span_ok(pairs):
    us = [float(p["u"]) for p in pairs]
    vs = [float(p["v"]) for p in pairs]
    su, sv = max(us) - min(us), max(vs) - min(vs)
    print("SPAN", round(su, 1), "x", round(sv, 1), "px (need ≥%.0f×%.0f)" % (MIN_SPAN_U, MIN_SPAN_V))
    return su >= MIN_SPAN_U and sv >= MIN_SPAN_V


def print_report():
    st = ac.http_get("/api/state")
    wiz = st.get("wizard") or {}
    report = wiz.get("report") or {}
    print("READY", st.get("calibration_ready"), st.get("calibration_message"))
    print("check_error", report.get("error_mm"))
    print("residuals", report.get("fit_residuals"))
    for i, pair in enumerate(wiz.get("pairs") or [], 1):
        print(
            "pair",
            i,
            {k: pair.get(k) for k in ("u", "v", "x", "y", "z", "source")},
        )
    print("check", wiz.get("check"))


def approach(x, y, z_touch):
    ac.go_table_point(x, y, z_touch)
    ac.move_coords(x, y, z_touch, speed=5)
    try:
        ac.wait_pose_near(x, y, z_touch, tol_xy=18.0, tol_z=14.0)
    except Exception as error:
        print("pose warn", error)


OFFSET_DETECT_MM = 48.0  # sideways so gripper does not cover the marker
WAIT_ARUCO_S = 90.0


def wait_for_aruco(timeout_s=WAIT_ARUCO_S, expect_uv=None, max_err_px=70.0):
    """Block until DICT_4X4_50 id 0 is visible under the target.

    If expect_uv is set, ignore markers farther than max_err_px (stops
    capturing a marker left elsewhere on the table).
    """
    deadline = time.time() + timeout_s
    print(
        "Waiting up to %.0fs for ArUco id 0 UNDER the gripper"
        % timeout_s
        + (
            " (near u,v ≈ %.0f,%.0f)…" % expect_uv
            if expect_uv
            else "…"
        )
    )
    last = None
    while time.time() < deadline:
        uv = aruco_visible(timeout_s=0.8)
        last = uv
        if uv is None:
            time.sleep(0.3)
            continue
        if expect_uv is not None:
            err = math.hypot(uv[0] - expect_uv[0], uv[1] - expect_uv[1])
            if err > max_err_px:
                print(
                    "ignore far marker",
                    (round(uv[0], 1), round(uv[1], 1)),
                    "err_px",
                    round(err, 1),
                    "- slide paper under the arm tip",
                )
                time.sleep(0.5)
                continue
        print("seen", (round(uv[0], 1), round(uv[1], 1)))
        return uv
    raise RuntimeError(
        "ArUco id 0 not under target within %.0fs (last=%s)"
        % (timeout_s, last)
    )


def capture_offset_then_center(label, x, y, z_touch):
    """Detect marker with arm offset, then save pose with pads on centre.

    Live /api/calibrate/aruco fails when open pads cover the marker; this
    records the marker UV from an offset view, then click+capture at the
    table pose with the gripper centred on the mark.
    """
    ac.set_gripper(0, 15)
    pairs = ac.wizard_pairs()
    # Never wait with the tip ON the marker — pads kill ArUco detect.
    # Hover offset so the full marker stays visible, then centre for pose.
    expect = ac._predict_uv(x, y, pairs) if len(pairs) >= 2 else None
    max_err = 100.0 if len(pairs) >= 2 else 999.0
    ox = min(x + OFFSET_DETECT_MM, 275.0)
    ac.move_coords(x, y, ac.TRAVEL_Z, speed=12)
    ac.move_coords(ox, y, ac.TRAVEL_Z, speed=10)
    ac.move_coords(ox, y, 120.0, speed=8)
    print(
        label,
        "offset hover — leave marker under target (%.0f, %.0f), not under tip"
        % (x, y),
    )
    wait_for_aruco(expect_uv=expect, max_err_px=max_err)
    time.sleep(0.35)
    expect_off = (
        ac._predict_uv(ox, y, pairs) if expect is not None else None
    )
    uv = None
    for _ in range(12):
        cand = aruco_visible(timeout_s=1.0)
        if cand is None:
            time.sleep(0.25)
            continue
        if expect_off is not None:
            err = math.hypot(
                cand[0] - expect_off[0], cand[1] - expect_off[1]
            )
            if err > 120.0:
                print(
                    "offset ignore far",
                    (round(cand[0], 1), round(cand[1], 1)),
                )
                continue
        uv = cand
        break
    if uv is None:
        raise RuntimeError("%s: ArUco not visible at offset pose" % label)
    print(label, "offset UV", (round(uv[0], 1), round(uv[1], 1)))
    # Centre pads on marker at table z (open fingers).
    ac.move_coords(x, y, ac.TRAVEL_Z, speed=10)
    ac.move_coords(x, y, 110.0, speed=8)
    ac.move_coords(x, y, z_touch, speed=5)
    try:
        ac.wait_pose_near(x, y, z_touch, tol_xy=16.0, tol_z=12.0)
    except Exception as error:
        print("pose warn", error)
    time.sleep(0.45)
    ac.http_post("/api/calibrate/hold", {})
    click = ac.http_post(
        "/api/calibrate/click", {"u": float(uv[0]), "v": float(uv[1])}
    )
    print(label, "click", click.get("message"))
    resp = ac.http_post("/api/calibrate/capture", {})
    print(label, "capture", resp.get("message"))
    if not resp.get("success"):
        raise RuntimeError("%s capture failed: %s" % (label, resp.get("message")))
    wiz = ac.http_get("/api/state").get("wizard") or {}
    spot = (
        wiz.get("check")
        if label.startswith("check")
        else (wiz.get("pairs") or [None])[-1]
    )
    print(
        label,
        "OK",
        {k: spot.get(k) for k in ("u", "v", "x", "y", "z", "source")}
        if spot
        else spot,
    )
    # Lift clear so user can slide the paper to the next spot.
    ac.move_coords(x, y, ac.TRAVEL_Z, speed=10)
    return spot


def run_pipeline(
    prompt_only=False,
    skip_drops=False,
    skip_accept=False,
    offset_mode=False,
):
    if not ac.camera_is_live():
        raise RuntimeError(
            "Overview camera is frozen — restart pick_place before calibrating"
        )
    print("camera live OK")
    print("Clear wizard + set clearances + open gripper")
    print(ac.http_post("/api/calibrate/clear", {}).get("message"))
    print(
        ac.http_post(
            "/api/calibrate/clearances",
            {"table_z_mm": ac.TABLE_Z, "hover_mm": 40},
        ).get("message")
    )
    ac.focus()
    ac.set_gripper(0, 15)
    time.sleep(0.8)

    z_touch = ac.TABLE_Z + 1.0
    if offset_mode:
        print(
            "\nOFFSET ArUco pipeline (hands-light).\n"
            "Print DICT_4X4_50 id 0, keep it FLAT with white margin.\n"
            "OPEN gripper. For each spot: when prompted by the wait,\n"
            "slide the paper under the hovering arm, then leave it.\n"
            "Between spots, move the paper to a new table location.\n"
            "Clear coloured caps off the table.\n"
        )
    else:
        print(
            "\nArUco pipeline. Marker: DICT_4X4_50 id 0.\n"
            "Print: http://127.0.0.1:8092/aruco\n"
            "Clear coloured caps off the table.\n"
            "For each spot: tape marker flat, centre OPEN pads on marker centre.\n"
        )
        wait_enter("Ready to start fit 1? Enter: ")

    for index, (x, y) in enumerate(ac.FIT, start=1):
        label = "fit%d" % index
        print("\n=== FIT %d/5 target XY %.1f, %.1f ===" % (index, x, y))
        if offset_mode:
            print(
                ">>> Slide marker under (%.0f, %.0f) now — OPEN pads, flat paper"
                % (x, y)
            )
            capture_offset_then_center(label, x, y, z_touch)
        else:
            if not prompt_only:
                print("Moving near target — place marker under the pads.")
                approach(x, y, z_touch)
            wait_enter(
                "%s: pads on marker centre at z≈%.0f, then Enter: "
                % (label, ac.TABLE_Z)
            )
            capture_aruco_spot(label)
        pairs = ac.wizard_pairs()
        if len(pairs) != index:
            raise RuntimeError(
                "pair count mismatch: have %d want %d" % (len(pairs), index)
            )

    pairs = ac.wizard_pairs()
    if not span_ok(pairs):
        print(
            "WARNING: pixel span too small — Accept may reject. "
            "Undo and remake with more spread."
        )

    print("\n=== CHECK ===")
    cx, cy = ac.CHECK
    if offset_mode:
        print(">>> Slide marker to a NEW place under (%.0f, %.0f)" % (cx, cy))
        capture_offset_then_center("check", cx, cy, z_touch)
    else:
        if not prompt_only:
            approach(cx, cy, z_touch)
        wait_enter("check: pads on marker at a NEW location, then Enter: ")
        capture_aruco_spot("check")
    wiz = ac.http_get("/api/state").get("wizard") or {}
    if not wiz.get("check"):
        raise RuntimeError("check spot was not saved")

    if not skip_drops:
        for name, (x, y) in ac.DROPS.items():
            label = name or "shared"
            if x > MAX_DROP_X:
                raise RuntimeError(
                    "drop %s X=%.1f exceeds %.0f" % (label, x, MAX_DROP_X)
                )
            print("\n=== DROP %s at %.1f, %.1f ===" % (label, x, y))
            if not prompt_only or offset_mode:
                approach(x, y, z_touch)
            else:
                wait_enter(
                    "Hand-guide above %s drop (X≤%.0f), Enter: "
                    % (label, MAX_DROP_X)
                )
            ac.http_post("/api/calibrate/hold", {})
            body = {"class_name": name} if name else {}
            resp = ac.http_post("/api/calibrate/drop", body)
            print("drop", label, resp.get("message"))

    pose = (ac.http_get("/api/state").get("wizard") or {}).get("tool_pose_mm")
    if pose and pose[2] < ac.TRAVEL_Z - 5:
        ac.move_coords(pose[0], pose[1], ac.TRAVEL_Z, speed=10)

    if skip_accept:
        print("Skipping Accept (--skip-accept). Review on /calibrate.")
        print_report()
        return 0

    print("\n=== ACCEPT ===")
    resp = ac.http_post("/api/calibrate/accept", {})
    print(json.dumps(resp, indent=2)[:5000])
    print_report()
    return 0 if resp.get("success") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-marker",
        metavar="PATH",
        nargs="?",
        const="/home/er/mycobot_humble/test_runs/aruco_id0.png",
        help="Write printable ArUco id-0 PNG and exit",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Do not move the arm; only prompt + ArUco-capture",
    )
    parser.add_argument(
        "--offset",
        action="store_true",
        help=(
            "Hands-light: wait for marker in view, detect UV with arm "
            "offset (avoids pad occlusion), then capture at table centre"
        ),
    )
    parser.add_argument(
        "--skip-drops",
        action="store_true",
        help="Skip drop captures (keep previous drop XY in wizard if any)",
    )
    parser.add_argument(
        "--skip-accept",
        action="store_true",
        help="Stop after spots (and optional drops); do not Accept",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Pick UI base URL (default http://127.0.0.1:8092)",
    )
    args = parser.parse_args(argv)
    if args.base:
        ac.BASE = args.base.rstrip("/")
    if args.export_marker is not None:
        export_marker_png(args.export_marker)
        return 0
    try:
        return run_pipeline(
            prompt_only=args.prompt_only,
            skip_drops=args.skip_drops,
            skip_accept=args.skip_accept,
            offset_mode=args.offset,
        )
    except Exception as error:
        print("FAILED:", error, file=sys.stderr)
        try:
            ac.http_post("/api/stop", {})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
