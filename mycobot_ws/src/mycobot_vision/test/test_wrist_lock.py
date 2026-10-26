"""The wrist ring stays on the middle of the chosen cap."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_sticky_notes import load_pick_place


def _table(width=320, height=240):
    return np.full((height, width, 3), (235, 235, 235), np.uint8)


def test_white_logo_stays_on_the_red_cap_centre():
    module = load_pick_place()
    image = _table()
    center = (160, 120)
    cv2.circle(image, center, 40, (30, 30, 210), -1)
    cv2.circle(image, center, 16, (245, 245, 245), -1)
    found = module._red_center(image)
    assert found is not None
    assert abs(found[0] - center[0]) < 12
    assert abs(found[1] - center[1]) < 12


def test_full_red_disc_stays_on_its_centre():
    module = load_pick_place()
    image = _table()
    cv2.circle(image, (150, 110), 36, (20, 20, 200), -1)
    cv2.circle(image, (150, 110), 12, (240, 240, 240), -1)
    found = module._red_center(image)
    assert found is not None
    assert abs(found[0] - 150) < 8
    assert abs(found[1] - 110) < 8


def test_red_ring_and_floor_disc_are_not_caps():
    module = load_pick_place()
    image = _table()
    cv2.circle(image, (180, 120), 48, (30, 30, 220), 4)
    assert module.wrist_cap_center(image, "red_cap") is None

    floor = np.full_like(image, (40, 80, 150))
    cv2.circle(floor, (170, 120), 40, (15, 15, 15), -1)
    assert module.wrist_cap_center(floor, "black_cap") is None
    cv2.circle(floor, (170, 120), 40, (30, 30, 210), -1)
    assert module.wrist_cap_center(floor, "red_cap") is None


def test_wood_floor_is_not_a_red_cap():
    module = load_pick_place()
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "wrist_floor_false_red.jpg"))
    assert image is not None
    assert module.wrist_cap_center(image, "red_cap") is None
    assert module.wrist_cap_center(image, "black_cap") is None
    assert module.wrist_cap_center(image, "white_cap") is None


def test_saved_floor_views_have_no_cap_mark():
    module = load_pick_place()
    root = Path(__file__).resolve().parents[4] / "test_runs"
    for name in ("wrist_after_fix.jpg", "wrist_now.jpg"):
        image = cv2.imread(str(root / name))
        assert image is not None
        for class_name in ("red_cap", "black_cap", "white_cap"):
            assert module.wrist_cap_center(image, class_name) is None


def test_saved_caps_still_lock_on_the_cap_only():
    module = load_pick_place()
    root = Path(__file__).resolve().parents[4] / "test_runs"
    red = cv2.imread(str(root / "cap_wrist.jpg"))
    found = module.wrist_cap_center(red, "red_cap")
    assert found is not None
    assert abs(found[0] - 279) < 20
    assert module.wrist_cap_center(red, "black_cap") is None
    assert module.wrist_cap_center(red, "white_cap") is None

    both = cv2.imread(str(root / "wrist_white_live2.jpg"))
    red_found = module.wrist_cap_center(both, "red_cap")
    white_found = module.wrist_cap_center(both, "white_cap")
    assert red_found is not None and abs(red_found[0] - 493) < 25
    assert white_found is not None and abs(white_found[0] - 314) < 30
    assert module.wrist_cap_center(both, "black_cap") is None


def test_lock_stays_off_the_cap_beside_it():
    module = load_pick_place()
    image = _table()
    cv2.circle(image, (100, 120), 36, (20, 20, 200), -1)
    cv2.circle(image, (155, 120), 36, (20, 20, 200), -1)
    found = module.wrist_cap_center(image, "red_cap", hint=(100, 120))
    assert found is not None
    assert abs(found[0] - 100) < 15
    assert abs(found[1] - 120) < 15


def test_hint_keeps_the_lock_off_the_other_cap():
    module = load_pick_place()
    image = _table()
    cv2.circle(image, (100, 80), 36, (20, 20, 200), -1)
    cv2.circle(image, (230, 170), 36, (20, 20, 200), -1)
    found = module.wrist_cap_center(image, "red_cap", hint=(100, 80))
    assert found is not None
    assert abs(found[0] - 100) < 12
    assert abs(found[1] - 80) < 12
