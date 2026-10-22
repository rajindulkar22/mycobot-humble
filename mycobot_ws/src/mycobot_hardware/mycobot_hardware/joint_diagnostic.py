"""Judgments for one measured joint check.

These functions do not open a serial port or call ROS.
"""

import math

DIAGNOSTIC_TOLERANCE_DEGREES = 0.5
COMPANION_SHIFT_DEGREES = 0.5
NO_MOTION_DEGREES = 0.2
OPPOSITE_MOTION_DEGREES = 0.3
INITIAL_POSITION_DEGREES = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def abort_reason(success, message):
    """Return why a failed move stops the sequence, or None on success."""
    if success:
        return None
    lowered = "" if message is None else str(message).lower()
    if "timed out" in lowered:
        return "timeout"
    if "controller reported error" in lowered or "error code" in lowered:
        return "controller"
    if (
        "invalid joint" in lowered
        or "not available yet" in lowered
        or "not finite" in lowered
        or "acknowledgement failed" in lowered
    ):
        return "feedback"
    return "movement"


def assess_move(
    joint_id,
    start,
    target,
    final,
    success,
    message,
    companion_shifts,
    tolerance=DIAGNOSTIC_TOLERANCE_DEGREES,
):
    """Return a verdict and the notes for one commanded joint move."""
    text = "" if message is None else str(message)
    lowered = text.lower()
    notes = []
    command_delta = _delta(start, target)
    measured_delta = _delta(start, final)
    error = _error(target, final)

    if "motion_enabled is false" in lowered:
        return _result(
            "disabled",
            notes + [
                "Motion is disabled, so this joint was not commanded. "
                "Start joint_diagnostic.launch.py and stop the other "
                "hardware node first."
            ],
            command_delta,
            measured_delta,
            error,
        )

    if (
        "invalid joint feedback" in lowered
        or "invalid joint response" in lowered
        or "current joint angle is invalid" in lowered
        or "not available yet" in lowered
    ):
        return _result(
            "feedback",
            notes + [
                "The angle sample was unusable. A reading of -1 is a "
                "failed read, not an angle of -1 degree."
            ],
            command_delta,
            measured_delta,
            error,
        )

    if "controller reported error" in lowered or "error code" in lowered:
        return _result(
            "controller",
            notes + [
                "The controller reported an error during the move. "
                "Leave the 0.5° tolerance unchanged and record the "
                "error text before moving the joint again."
            ],
            command_delta,
            measured_delta,
            error,
        )

    if "acknowledgement failed" in lowered:
        return _result(
            "feedback",
            notes + [
                "The controller did not accept the command. Reply 1 "
                "would only mean the packet was accepted, not that the "
                "joint arrived."
            ],
            command_delta,
            measured_delta,
            error,
        )

    if _rejected(lowered):
        return _result(
            "rejected",
            notes + [
                "The command was rejected before motion. The step must "
                "stay within 2° and inside the joint limit."
            ],
            command_delta,
            measured_delta,
            error,
        )

    if (
        command_delta is not None
        and measured_delta is not None
        and abs(command_delta) >= 1.0
        and measured_delta * command_delta < 0
        and abs(measured_delta) >= OPPOSITE_MOTION_DEGREES
    ):
        notes.append(
            f"J{joint_id} moved {measured_delta:+.2f}° while the command "
            f"was {command_delta:+.2f}°. The measured direction is "
            "opposite the command. Repeat this joint once before "
            "deciding the cause. Do not invert the command from one "
            "sample, and do not widen the tolerance."
        )
        notes.extend(_companion_notes(companion_shifts))
        return _result(
            "opposite",
            notes,
            command_delta,
            measured_delta,
            error,
        )

    if (
        command_delta is not None
        and measured_delta is not None
        and abs(command_delta) >= 1.0
        and abs(measured_delta) < NO_MOTION_DEGREES
    ):
        notes.append(
            f"J{joint_id} was commanded {command_delta:+.2f}° and the "
            "measured angle stayed put. Acceptance of the command is "
            "not arrival."
        )
        notes.extend(_companion_notes(companion_shifts))
        return _result(
            "still",
            notes,
            command_delta,
            measured_delta,
            error,
        )

    if (
        success
        and error is not None
        and error <= tolerance
    ):
        notes.append(
            f"J{joint_id} reached the target within {tolerance:.1f}°."
        )
        notes.extend(_companion_notes(companion_shifts))
        return _result(
            "tracked",
            notes,
            command_delta,
            measured_delta,
            error,
        )

    if (
        command_delta is not None
        and measured_delta is not None
        and error is not None
        and measured_delta * command_delta > 0
        and error > tolerance
    ):
        past_target = (final - target) * command_delta > 0
        if past_target:
            notes.append(
                f"J{joint_id} moved the commanded way and passed the "
                f"target by {error:.2f}°. Keep the tolerance at "
                f"{tolerance:.1f}°."
            )
            verdict = "overshoot"
        else:
            notes.append(
                f"J{joint_id} moved the commanded way but finished "
                f"{error:.2f}° short of the target. Keep the tolerance "
                f"at {tolerance:.1f}°."
            )
            verdict = "short"
        if "timed out" in lowered:
            notes.append(
                "The driver stopped the wait because the final error "
                "was outside tolerance."
            )
        notes.extend(_companion_notes(companion_shifts))
        return _result(
            verdict,
            notes,
            command_delta,
            measured_delta,
            error,
        )

    if "timed out" in lowered:
        notes.append(
            "The wait timed out. Use the start, target, and final "
            "angles in this report rather than the command "
            "acknowledgement."
        )
    else:
        notes.append(
            "This sample does not fit the tracked, opposite, or "
            "no-motion patterns. Keep the start, target, final, and "
            "service message together."
        )
    notes.extend(_companion_notes(companion_shifts))
    return _result(
        "unknown",
        notes,
        command_delta,
        measured_delta,
        error,
    )


def step_toward(measured, goal, step_limit):
    """Return the next target, never more than step_limit from measured."""
    if not _finite(measured) or not _finite(goal) or not _finite(step_limit):
        return None
    if step_limit <= 0:
        return None
    delta = float(goal) - float(measured)
    if abs(delta) <= float(step_limit):
        return float(goal)
    return float(measured) + math.copysign(float(step_limit), delta)


def summarize_reports(reports):
    """Return one paragraph for a full six-joint run."""
    counts = {}
    for report in reports:
        verdict = report.get("verdict", "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1

    if counts.get("tracked", 0) == len(reports) and reports:
        return (
            "Every checked joint reached its 2° target within 0.5° "
            "on the outward move."
        )

    parts = []
    labels = (
        ("opposite", "moved opposite the command"),
        ("still", "did not move"),
        ("short", "stopped short"),
        ("overshoot", "passed the target"),
        ("feedback", "had unusable feedback"),
        ("controller", "reported a controller error"),
        ("disabled", "were not commanded because motion is disabled"),
        ("rejected", "were rejected before motion"),
        ("unknown", "need the raw start, target, and final angles"),
    )
    for verdict, description in labels:
        joints = [
            f"J{report['joint_id']}"
            for report in reports
            if report.get("verdict") == verdict
        ]
        if joints:
            parts.append(f"{', '.join(joints)} {description}")

    if not parts:
        return "No joint reports were recorded."
    return (
        "; ".join(parts) + ". Leave the 0.5° tolerance unchanged. "
        "This check is one joint at a time, not the dance stream."
    )


def _rejected(lowered):
    return (
        "exceeds" in lowered
        or "outside safe range" in lowered
        or "must be between" in lowered
        or "dry run" in lowered
    )


def _companion_notes(companion_shifts):
    moved = []
    for joint_id, shift in companion_shifts:
        if shift is None:
            continue
        if abs(shift) > COMPANION_SHIFT_DEGREES:
            moved.append(f"J{joint_id} {shift:+.2f}°")
    if not moved:
        return []
    return [
        "These other joints also changed by more than "
        f"{COMPANION_SHIFT_DEGREES:.1f}°: {', '.join(moved)}."
    ]


def _delta(start, end):
    if not _finite(start) or not _finite(end):
        return None
    return float(end) - float(start)


def _error(target, final):
    if not _finite(target) or not _finite(final):
        return None
    return abs(float(final) - float(target))


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and abs(value) != float("inf")


def _result(verdict, notes, command_delta, measured_delta, error):
    return {
        "verdict": verdict,
        "notes": notes,
        "command_delta": command_delta,
        "measured_delta": measured_delta,
        "error_degrees": error,
    }
