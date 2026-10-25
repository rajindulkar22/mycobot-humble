#!/usr/bin/env python3
"""Failure-aware grasp candidates, verification, and multi-cap task state.

Used by pick_place.py. No ROS imports so offline unit tests stay light.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- Safety / autonomy bounds (confirm before unsupervised runs) ---
SAFETY_PARAMS = {
    "grasp_clearance_mm": 2.0,
    "travel_z_mm": 150.0,
    "max_attempts_per_cap": 3,
    "max_auto_duration_seconds": 600.0,
    "max_total_attempts": 12,
    "stale_detection_seconds": 6.0,
    "failed_xy_repeat_tol_mm": 2.0,
    "reobserve_pixel_tol": 15.0,
    "coord_min_x_mm": 50.0,
    # Firmware rejects X beyond ~281.45 mm; keep software box inside that.
    "coord_max_x_mm": 280.0,
    "coord_min_y_mm": -200.0,
    "coord_max_y_mm": 200.0,
}

MAX_ATTEMPTS_PER_CAP = int(SAFETY_PARAMS["max_attempts_per_cap"])
MAX_AUTO_DURATION_SECONDS = float(SAFETY_PARAMS["max_auto_duration_seconds"])
MAX_TOTAL_ATTEMPTS = int(SAFETY_PARAMS["max_total_attempts"])

# Jaw bands match pick_place / state_publisher.
JAW_HOLD_MIN = 12
JAW_HOLD_MAX = 86
JAW_LID_MIN = 90
JAW_EMPTY_MAX = 10

# Ordered XY deltas (mm) after FAILURE_LID / FAILURE_EMPTY.
# First entry is "retry with fresh vision only" (0,0 skipped if used).
CANDIDATES_LID = (
    (0.0, 4.0),
    (0.0, 8.0),
    (4.0, 4.0),
    (-4.0, 4.0),
    (0.0, -4.0),
)
CANDIDATES_EMPTY = (
    (0.0, -4.0),
    (4.0, 0.0),
    (-4.0, 0.0),
    (0.0, 4.0),
    (8.0, 0.0),
)
CANDIDATES_UNKNOWN = (
    (0.0, 4.0),
    (4.0, 0.0),
    (-4.0, 0.0),
    (0.0, -4.0),
)

CAP_STATUSES = (
    "PENDING",
    "DETECTED",
    "APPROACHING",
    "ALIGNING",
    "GRASPING",
    "VERIFYING",
    "PLACING",
    "COMPLETED",
    "RETRY",
    "SKIPPED",
    "FAILED",
)


@dataclass
class CapTrack:
    """One colour's live track and attempt history."""

    class_name: str
    status: str = "PENDING"
    u: Optional[float] = None
    v: Optional[float] = None
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None
    score: float = 0.0
    attempts: int = 0
    last_failure: Optional[str] = None
    failed_xy: List[Tuple[float, float]] = field(default_factory=list)
    used_candidates: List[Tuple[float, float]] = field(default_factory=list)
    last_message: str = ""


def classify_grasp(final_value) -> str:
    """Return SUCCESS, FAILURE_LID, FAILURE_EMPTY, or UNKNOWN."""
    if not isinstance(final_value, (int, float)):
        return "UNKNOWN"
    value = int(final_value)
    if JAW_HOLD_MIN <= value <= JAW_HOLD_MAX:
        return "SUCCESS"
    if value >= JAW_LID_MIN:
        return "FAILURE_LID"
    if value <= JAW_EMPTY_MAX:
        return "FAILURE_EMPTY"
    return "UNKNOWN"


def candidates_for_failure(failure: str) -> Tuple[Tuple[float, float], ...]:
    """Return ordered XY deltas for the classified failure."""
    if failure == "FAILURE_LID":
        return CANDIDATES_LID
    if failure == "FAILURE_EMPTY":
        return CANDIDATES_EMPTY
    return CANDIDATES_UNKNOWN


def next_candidate(
    track: CapTrack,
    failure: str,
) -> Optional[Tuple[float, float]]:
    """Pick the next unused XY delta for this track."""
    for delta in candidates_for_failure(failure):
        rounded = (round(delta[0], 1), round(delta[1], 1))
        if rounded in track.used_candidates:
            continue
        track.used_candidates.append(rounded)
        return rounded
    return None


def xy_already_failed(
    track: CapTrack,
    x_mm: float,
    y_mm: float,
    tol_mm: float = None,
) -> bool:
    """True when this XY was already tried and failed."""
    if tol_mm is None:
        tol_mm = float(SAFETY_PARAMS["failed_xy_repeat_tol_mm"])
    for past_x, past_y in track.failed_xy:
        if math.hypot(past_x - x_mm, past_y - y_mm) <= tol_mm:
            return True
    return False


def remember_failed_xy(track: CapTrack, x_mm: float, y_mm: float) -> None:
    """Record a failed grasp XY so it is not repeated."""
    track.failed_xy.append((round(float(x_mm), 1), round(float(y_mm), 1)))


def in_workspace(x_mm: float, y_mm: float) -> bool:
    """True when the point sits inside the pick-place coord box."""
    return (
        SAFETY_PARAMS["coord_min_x_mm"]
        <= float(x_mm)
        <= SAFETY_PARAMS["coord_max_x_mm"]
        and SAFETY_PARAMS["coord_min_y_mm"]
        <= float(y_mm)
        <= SAFETY_PARAMS["coord_max_y_mm"]
    )


def pixel_near(
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    tol_px: float = None,
) -> bool:
    """True when two overview pixels are close."""
    if tol_px is None:
        tol_px = float(SAFETY_PARAMS["reobserve_pixel_tol"])
    return math.hypot(float(u0) - float(u1), float(v0) - float(v1)) <= tol_px


class AttemptLogger:
    """Append one JSON object per grasp attempt."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.environ.get(
                "MYCOBOT_GRASP_LOG",
                "/tmp/mycobot_grasp_attempts.jsonl",
            )
        self.path = path
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log(self, record: dict) -> None:
        payload = dict(record)
        payload.setdefault("unix_time", time.time())
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class TaskManager:
    """Bounded multi-cap pick queue."""

    def __init__(self, class_names=None):
        names = list(class_names or ("red_cap", "white_cap", "black_cap"))
        self.tracks: Dict[str, CapTrack] = {
            name: CapTrack(class_name=name) for name in names
        }
        self.started_at = time.monotonic()
        self.total_attempts = 0
        self.round_pass = 0
        self.finished = False
        self.message = ""

    def update_from_detections(self, detections: dict) -> None:
        """Refresh DETECTed tracks from locked overview detections."""
        for name, track in self.tracks.items():
            if track.status in ("COMPLETED", "FAILED"):
                continue
            det = detections.get(name)
            if det is None:
                if track.status == "DETECTED":
                    track.status = "PENDING"
                continue
            track.u = float(det["u"])
            track.v = float(det["v"])
            track.x_mm = (
                float(det["x_mm"]) if det.get("x_mm") is not None else None
            )
            track.y_mm = (
                float(det["y_mm"]) if det.get("y_mm") is not None else None
            )
            track.score = float(det.get("score") or 0.0)
            if track.status in ("PENDING", "RETRY", "SKIPPED"):
                track.status = "DETECTED"

    def limits_hit(self) -> Optional[str]:
        """Return a stop reason when attempt or time limits are hit."""
        elapsed = time.monotonic() - self.started_at
        if elapsed >= MAX_AUTO_DURATION_SECONDS:
            return "Auto pick stopped: time limit reached."
        if self.total_attempts >= MAX_TOTAL_ATTEMPTS:
            return "Auto pick stopped: total attempt limit reached."
        return None

    def select_next(self) -> Optional[CapTrack]:
        """Choose the next colour to attempt."""
        limit = self.limits_hit()
        if limit:
            self.finished = True
            self.message = limit
            return None
        # Only freshly DETECTED caps. RETRY stays out of the queue until
        # update_from_detections sees the colour again (avoids a busy-loop
        # on stale RETRY coords when the cap is out of view).
        candidates = [
            track
            for track in self.tracks.values()
            if track.status == "DETECTED"
            and track.attempts < MAX_ATTEMPTS_PER_CAP
            and track.x_mm is not None
            and track.y_mm is not None
            and in_workspace(track.x_mm, track.y_mm)
        ]
        if candidates:
            candidates.sort(
                key=lambda item: (item.attempts, -item.score)
            )
            return candidates[0]
        # Revive SKIPPED once for a second pass.
        skipped = [
            track
            for track in self.tracks.values()
            if track.status == "SKIPPED"
            and track.attempts < MAX_ATTEMPTS_PER_CAP
        ]
        if skipped and self.round_pass < 1:
            self.round_pass += 1
            for track in skipped:
                track.status = "RETRY"
            return self.select_next()
        pending = [
            track
            for track in self.tracks.values()
            if track.status not in ("COMPLETED", "FAILED")
        ]
        if not pending:
            self.finished = True
            self.message = "All caps completed or failed."
            return None
        # Still pending but not detected / out of workspace.
        self.finished = True
        unresolved = [
            track.class_name
            for track in self.tracks.values()
            if track.status not in ("COMPLETED",)
        ]
        self.message = (
            "Auto pick stopped. Unresolved: " + ", ".join(unresolved)
        )
        return None

    def mark_attempt(self, track: CapTrack) -> None:
        track.attempts += 1
        self.total_attempts += 1

    def mark_success(self, track: CapTrack, message: str = "") -> None:
        track.status = "COMPLETED"
        track.last_failure = None
        track.last_message = message or "Completed"

    def mark_failure(
        self,
        track: CapTrack,
        failure: str,
        message: str = "",
    ) -> None:
        track.last_failure = failure
        track.last_message = message
        if track.attempts >= MAX_ATTEMPTS_PER_CAP:
            track.status = "SKIPPED"
        else:
            track.status = "RETRY"

    def summary(self) -> dict:
        return {
            "finished": self.finished,
            "message": self.message,
            "total_attempts": self.total_attempts,
            "elapsed_seconds": round(
                time.monotonic() - self.started_at, 1
            ),
            "tracks": {
                name: {
                    "status": track.status,
                    "attempts": track.attempts,
                    "score": round(track.score, 3),
                    "x_mm": track.x_mm,
                    "y_mm": track.y_mm,
                    "last_failure": track.last_failure,
                    "last_message": track.last_message,
                }
                for name, track in self.tracks.items()
            },
        }


def pair_fit_residuals(pairs, project_fn) -> List[dict]:
    """Return per-spot residual mm for calibration diagnostics."""
    rows = []
    for index, pair in enumerate(pairs or []):
        pred_x, pred_y = project_fn(float(pair["u"]), float(pair["v"]))
        err = math.hypot(pred_x - float(pair["x"]), pred_y - float(pair["y"]))
        rows.append(
            {
                "index": index + 1,
                "error_mm": round(err, 2),
                "z_mm": pair.get("z"),
            }
        )
    return rows
