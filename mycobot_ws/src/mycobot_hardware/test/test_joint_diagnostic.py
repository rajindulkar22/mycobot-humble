"""Offline checks for joint-diagnostic judgments."""

from mycobot_hardware.joint_diagnostic import (
    abort_reason,
    assess_move,
    step_toward,
    summarize_reports,
)


def test_historical_joint4_opposite_sample():
    report = assess_move(
        4,
        -7.82,
        -6.02,
        -8.34,
        False,
        "Movement timed out after 10.0s; last angle -8.34°. "
        "Stop command sent.",
        [(1, 0.0), (2, 0.02), (3, -0.1), (5, 0.0), (6, 0.0)],
    )
    assert report["verdict"] == "opposite"
    assert report["error_degrees"] > 0.5
    assert any("opposite" in note for note in report["notes"])
    assert any("tolerance" in note for note in report["notes"])


def test_tracked_joint_passes():
    report = assess_move(
        2,
        10.0,
        12.0,
        11.8,
        True,
        "Movement completed",
        [],
    )
    assert report["verdict"] == "tracked"


def test_short_and_overshoot_keep_tolerance():
    short = assess_move(1, 10.0, 12.0, 11.0, False, "timed out", [])
    overshoot = assess_move(3, 10.0, 12.0, 13.2, False, "timed out", [])
    assert short["verdict"] == "short"
    assert overshoot["verdict"] == "overshoot"
    assert any("0.5" in note for note in short["notes"])


def test_still_joint_is_not_arrival():
    report = assess_move(5, 1.0, 3.0, 1.05, False, "timed out", [])
    assert report["verdict"] == "still"


def test_unusable_feedback_and_disabled_motion():
    feedback = assess_move(
        4,
        -7.82,
        -6.02,
        None,
        False,
        "Invalid joint feedback during movement: -1; stop command sent",
        [],
    )
    disabled = assess_move(
        4,
        None,
        None,
        None,
        False,
        "Validation passed, but motion_enabled is false",
        [],
    )
    assert feedback["verdict"] == "feedback"
    assert "-1" in feedback["notes"][0]
    assert disabled["verdict"] == "disabled"


def test_companion_motion_is_reported():
    report = assess_move(
        4,
        0.0,
        2.0,
        1.9,
        True,
        "Movement completed",
        [(2, 1.4)],
    )
    assert report["verdict"] == "tracked"
    assert any("J2" in note for note in report["notes"])


def test_abort_reason_stops_timeout_controller_and_bad_feedback():
    assert abort_reason(False, "Movement timed out after 10.0s") == "timeout"
    assert abort_reason(
        False,
        "Controller reported error 4; stop command sent",
    ) == "controller"
    assert abort_reason(
        False,
        "Invalid joint feedback during movement: -1; stop command sent",
    ) == "feedback"
    assert abort_reason(True, "Movement completed") is None
    assert abort_reason(False, "Requested movement exceeds the limit") == "movement"


def test_initial_position_steps_stay_inside_two_degrees():
    assert step_toward(12.4, 0.0, 2.0) == 10.4
    assert step_toward(-1.2, 0.0, 2.0) == 0.0
    assert step_toward(0.2, 0.0, 2.0) == 0.0
    assert abs(step_toward(-90.0, 0.0, 2.0) - (-88.0)) < 1e-9


def test_summary_names_the_failed_joints():
    text = summarize_reports(
        [
            {"joint_id": 4, "verdict": "opposite"},
            {"joint_id": 2, "verdict": "tracked"},
            {"joint_id": 6, "verdict": "still"},
        ]
    )
    assert "J4" in text
    assert "J6" in text
    assert "0.5" in text
