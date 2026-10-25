#!/usr/bin/env python3
"""Write enhanced, gridded zoom crops around predicted fingertip pixels."""

import json
import sys

import cv2
import numpy as np

OUT = "/home/er/mycobot_humble/test_runs/"
SEED = {  # pose xy -> rough tip uv read by eye
    "200_-80": (280, 230),
    "150_-130": (205, 255),
    "150_-10": (245, 150),
}
HALF = 50
ZOOM = 6


def load_points():
    return [json.loads(line) for line in open(OUT + "recal_points.jsonl")]


def predictor(points, seeds):
    rows = [p for p in points if p["tag"] in seeds]
    xy = np.array([[p["pose"][0], p["pose"][1], 1.0] for p in rows])
    uv = np.array([seeds[p["tag"]] for p in rows], dtype=float)
    coeff, *_ = np.linalg.lstsq(xy, uv, rcond=None)
    return lambda x, y: np.array([x, y, 1.0]) @ coeff


def zoom(tag, cu, cv, half=HALF, scale=ZOOM):
    img = cv2.imread(OUT + "recal_pt_%s.jpg" % tag)
    h, w = img.shape[:2]
    u0 = int(max(0, min(w - 2 * half, round(cu) - half)))
    v0 = int(max(0, min(h - 2 * half, round(cv) - half)))
    crop = img[v0:v0 + 2 * half, u0:u0 + 2 * half].copy()
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(3.0, (4, 4)).apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    big = cv2.resize(crop, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    for i in range(0, 2 * half + 1, 5):
        u = u0 + i
        color = (0, 0, 255) if u % 10 == 0 else (0, 150, 255)
        cv2.line(big, (i * scale, 0), (i * scale, big.shape[0]), color, 1)
        if u % 10 == 0:
            cv2.putText(big, str(u), (i * scale + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        v = v0 + i
        color = (0, 0, 255) if v % 10 == 0 else (0, 150, 255)
        cv2.line(big, (0, i * scale), (big.shape[1], i * scale), color, 1)
        if v % 10 == 0:
            cv2.putText(big, str(v), (2, i * scale + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.imwrite(OUT + "recal_zoom_%s.jpg" % tag, big)
    return u0, v0


def main():
    points = load_points()
    if len(sys.argv) > 1:
        tag, cu, cv = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
        half = int(sys.argv[4]) if len(sys.argv) > 4 else HALF
        print(tag, zoom(tag, cu, cv, half=half))
        return
    predict = predictor(points, SEED)
    for p in points:
        cu, cv = predict(p["pose"][0], p["pose"][1])
        print(p["tag"], round(cu), round(cv), zoom(p["tag"], cu, cv))


if __name__ == "__main__":
    main()
