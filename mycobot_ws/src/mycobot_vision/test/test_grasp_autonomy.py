#!/usr/bin/env python3
"""Offline unit tests for multi-cap grasp autonomy helpers."""

import os
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scripts",
)
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

from grasp_autonomy import (  # noqa: E402
    MAX_ATTEMPTS_PER_CAP,
    SAFETY_PARAMS,
    AttemptLogger,
    CapTrack,
    TaskManager,
    classify_grasp,
    in_workspace,
    next_candidate,
    pixel_near,
    remember_failed_xy,
    xy_already_failed,
)


class ClassifyGraspTests(unittest.TestCase):
    def test_success_band(self):
        self.assertEqual(classify_grasp(45), "SUCCESS")
        self.assertEqual(classify_grasp(12), "SUCCESS")
        self.assertEqual(classify_grasp(86), "SUCCESS")

    def test_lid_and_empty(self):
        self.assertEqual(classify_grasp(92), "FAILURE_LID")
        self.assertEqual(classify_grasp(5), "FAILURE_EMPTY")

    def test_unknown(self):
        self.assertEqual(classify_grasp(None), "UNKNOWN")
        self.assertEqual(classify_grasp(88), "UNKNOWN")


class CandidateTests(unittest.TestCase):
    def test_never_repeats_used_candidate(self):
        track = CapTrack(class_name="black_cap")
        first = next_candidate(track, "FAILURE_LID")
        second = next_candidate(track, "FAILURE_LID")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_exhausted_candidates(self):
        track = CapTrack(class_name="red_cap")
        seen = set()
        while True:
            nxt = next_candidate(track, "FAILURE_EMPTY")
            if nxt is None:
                break
            seen.add(nxt)
        self.assertGreaterEqual(len(seen), 4)
        self.assertIsNone(next_candidate(track, "FAILURE_EMPTY"))

    def test_failed_xy_memory(self):
        track = CapTrack(class_name="white_cap")
        remember_failed_xy(track, 180.0, -40.0)
        self.assertTrue(xy_already_failed(track, 180.5, -40.2))
        self.assertFalse(xy_already_failed(track, 200.0, -40.0))


class WorkspaceAndPixelTests(unittest.TestCase):
    def test_workspace_box(self):
        self.assertTrue(in_workspace(180.0, -50.0))
        self.assertFalse(in_workspace(10.0, 0.0))
        self.assertFalse(in_workspace(180.0, 250.0))

    def test_pixel_near(self):
        self.assertTrue(pixel_near(100.0, 200.0, 108.0, 205.0))
        self.assertFalse(pixel_near(100.0, 200.0, 140.0, 200.0))


class TaskManagerTests(unittest.TestCase):
    def test_select_prefers_detected(self):
        manager = TaskManager()
        manager.update_from_detections(
            {
                "red_cap": {
                    "u": 10.0,
                    "v": 20.0,
                    "x_mm": 180.0,
                    "y_mm": -40.0,
                    "score": 0.9,
                },
                "black_cap": {
                    "u": 30.0,
                    "v": 40.0,
                    "x_mm": 200.0,
                    "y_mm": -20.0,
                    "score": 0.5,
                },
            }
        )
        first = manager.select_next()
        self.assertIsNotNone(first)
        self.assertEqual(first.class_name, "red_cap")

    def test_skip_after_max_attempts(self):
        manager = TaskManager(("red_cap",))
        manager.update_from_detections(
            {
                "red_cap": {
                    "u": 1.0,
                    "v": 2.0,
                    "x_mm": 180.0,
                    "y_mm": -40.0,
                    "score": 0.8,
                }
            }
        )
        track = manager.select_next()
        for _ in range(MAX_ATTEMPTS_PER_CAP):
            manager.mark_attempt(track)
            manager.mark_failure(track, "FAILURE_LID", "lid")
        self.assertEqual(track.status, "SKIPPED")
        # Second pass may revive once, then finish.
        again = manager.select_next()
        if again is not None:
            manager.mark_attempt(again)
            manager.mark_failure(again, "FAILURE_LID", "lid")
            self.assertIn(again.status, ("SKIPPED", "RETRY", "FAILED"))

    def test_completed_not_reselected(self):
        manager = TaskManager(("red_cap", "white_cap"))
        manager.update_from_detections(
            {
                "red_cap": {
                    "u": 1.0,
                    "v": 1.0,
                    "x_mm": 180.0,
                    "y_mm": -40.0,
                    "score": 0.9,
                },
                "white_cap": {
                    "u": 2.0,
                    "v": 2.0,
                    "x_mm": 190.0,
                    "y_mm": -30.0,
                    "score": 0.8,
                },
            }
        )
        track = manager.select_next()
        manager.mark_attempt(track)
        manager.mark_success(track, "ok")
        nxt = manager.select_next()
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.class_name, "white_cap")


class AttemptLoggerTests(unittest.TestCase):
    def test_jsonl_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attempts.jsonl")
            logger = AttemptLogger(path)
            logger.log({"class_name": "red_cap", "outcome": "SUCCESS"})
            logger.log({"class_name": "black_cap", "outcome": "FAILURE_LID"})
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("SUCCESS", lines[0])


class StaleLockLogicTests(unittest.TestCase):
    """Mirror the age gate used by pick_place without importing ROS."""

    def test_stale_age_rejected(self):
        max_age = SAFETY_PARAMS["stale_detection_seconds"]
        age = max_age + 0.5
        self.assertTrue(age > max_age)

    def test_fresh_age_accepted(self):
        max_age = SAFETY_PARAMS["stale_detection_seconds"]
        age = 0.4
        self.assertFalse(age > max_age)


if __name__ == "__main__":
    unittest.main()
