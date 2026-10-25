"""Offline checks for the overview cap lock and circle centre."""

import importlib.util
import math
import sys
import types
from pathlib import Path


def _install_import_stubs():
    """Load yolo_detector without ROS, Ultralytics, or a trained model."""
    if "cv2" not in sys.modules:
        import cv2  # noqa: F401
    if "numpy" not in sys.modules:
        import numpy  # noqa: F401

    class _Node:
        pass

    class _QoSProfile:
        def __init__(self, **_kwargs):
            pass

    class _Policy:
        KEEP_LAST = 1
        BEST_EFFORT = 1
        RELIABLE = 1
        VOLATILE = 1
        TRANSIENT_LOCAL = 1

    class _Image:
        pass

    class _Bool:
        def __init__(self):
            self.data = False

    class _YOLO:
        def __init__(self, _path):
            raise RuntimeError("tests do not load weights")

    class _Detection2D:
        pass

    class _Detection2DArray:
        pass

    class _Hypothesis:
        pass

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.QoSProfile = _QoSProfile
    qos_mod.DurabilityPolicy = _Policy
    qos_mod.HistoryPolicy = _Policy
    qos_mod.ReliabilityPolicy = _Policy
    rclpy.node = node_mod
    rclpy.qos = qos_mod

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object

    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.Image = _Image
    sensor.msg = sensor_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Bool = _Bool
    std_msgs.msg = std_msg

    ament = types.ModuleType("ament_index_python")
    ament_packages = types.ModuleType("ament_index_python.packages")
    ament_packages.PackageNotFoundError = RuntimeError
    ament_packages.get_package_share_directory = lambda _name: ""
    ament.packages = ament_packages

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = _YOLO

    vision = types.ModuleType("vision_msgs")
    vision_msg = types.ModuleType("vision_msgs.msg")
    vision_msg.Detection2D = _Detection2D
    vision_msg.Detection2DArray = _Detection2DArray
    vision_msg.ObjectHypothesisWithPose = _Hypothesis
    vision.msg = vision_msg

    modules = {
        "rclpy": rclpy,
        "rclpy.node": node_mod,
        "rclpy.qos": qos_mod,
        "cv_bridge": cv_bridge,
        "sensor_msgs": sensor,
        "sensor_msgs.msg": sensor_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msg,
        "ament_index_python": ament,
        "ament_index_python.packages": ament_packages,
        "ultralytics": ultralytics,
        "vision_msgs": vision,
        "vision_msgs.msg": vision_msg,
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


def load_detector():
    """Import the detector module without constructing the ROS node."""
    _install_import_stubs()
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "yolo_detector.py"
    )
    spec = importlib.util.spec_from_file_location(
        "yolo_detector_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(name, u, v, score=0.8):
    return {
        "class_name": name,
        "u": u,
        "v": v,
        "size_x": 32.0,
        "size_y": 32.0,
        "score": score,
        "x_min": u - 16.0,
        "y_min": v - 16.0,
        "x_max": u + 16.0,
        "y_max": v + 16.0,
    }


def _establish(module, name="red_cap", u=100.0, v=80.0):
    locks = module.new_cap_locks()
    first = module.update_cap_locks(locks, [_sample(name, u, v, 0.4)])
    assert first == []
    visible = module.update_cap_locks(locks, [_sample(name, u, v, 0.45)])
    assert len(visible) == 1
    assert visible[0]["class_name"] == name
    return locks, visible[0]


def test_sure_lock_publishes_on_first_hit():
    module = load_detector()
    locks = module.new_cap_locks()
    visible = module.update_cap_locks(
        locks,
        [_sample("black_cap", 320.0, 258.0, 0.96)],
    )
    assert len(visible) == 1
    assert visible[0]["class_name"] == "black_cap"


def test_lock_damps_jitter():
    module = load_detector()
    locks, locked = _establish(module)
    raw_u = locked["u"] + 8.0
    visible = module.update_cap_locks(
        locks,
        [_sample("red_cap", raw_u, locked["v"])],
    )
    assert len(visible) == 1
    moved = abs(visible[0]["u"] - locked["u"])
    raw_move = abs(raw_u - locked["u"])
    assert moved < raw_move
    assert math.isclose(moved, module.LOCK_NEW * raw_move, rel_tol=1e-9, abs_tol=1e-9)


def test_lock_survives_one_missed_frame():
    module = load_detector()
    locks, locked = _establish(module)
    visible = module.update_cap_locks(locks, [])
    assert len(visible) == 1
    assert visible[0]["u"] == locked["u"]
    assert visible[0]["v"] == locked["v"]
    assert visible[0]["missed"] == 1


def test_lock_rejects_teleport():
    module = load_detector()
    locks, locked = _establish(module, u=80.0, v=90.0)
    visible = module.update_cap_locks(
        locks,
        [_sample("red_cap", 300.0, 90.0)],
    )
    assert len(visible) == 1
    assert visible[0]["u"] == locked["u"]
    assert visible[0]["v"] == locked["v"]
    assert math.hypot(300.0 - visible[0]["u"], 90.0 - visible[0]["v"]) > (
        module.LOCK_RADIUS_PX
    )


def test_lock_drops_after_miss_limit():
    module = load_detector()
    locks, _locked = _establish(module)
    for _step in range(module.LOCK_MISS_LIMIT):
        assert module.update_cap_locks(locks, [])
    assert module.update_cap_locks(locks, []) == []


def _logo_image():
    import cv2
    import numpy as np

    image = np.full((200, 200, 3), 250, dtype=np.uint8)
    cv2.circle(image, (100, 100), 16, (110, 110, 110), -1)
    cv2.circle(image, (108, 100), 6, (30, 30, 30), -1)
    cv2.circle(image, (114, 96), 5, (250, 250, 250), -1)
    return image


def _moment_center(image, class_name, pixel_u, pixel_v, size_x, size_y):
    """Reproduce the old mass-centroid centre for the logo check."""
    import cv2
    import numpy as np

    span = max(float(size_x), float(size_y), 28.0)
    half = max(span * 0.8, 26.0)
    height, width = image.shape[:2]
    x0 = max(0, int(round(pixel_u - half)))
    y0 = max(0, int(round(pixel_v - half)))
    x1 = min(width, int(round(pixel_u + half)))
    y1 = min(height, int(round(pixel_v + half)))
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yy, xx = np.ogrid[:height, :width]
    dist = np.hypot(xx - float(pixel_u), yy - float(pixel_v))
    ring = gray_full[(dist >= span * 0.6) & (dist <= span * 1.0)]
    table = float(np.percentile(ring, 80))
    gray = gray_full[y0:y1, x0:x1]
    mask = (
        (gray < table - 20.0) & (gray > 45.0)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 180.0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        center = (
            x0 + moments["m10"] / moments["m00"],
            y0 + moments["m01"] / moments["m00"],
        )
        if best is None or area > best[0]:
            best = (area, center)
    assert best is not None
    return best[1]


def test_circle_centre_ignores_logo_better_than_moments():
    module = load_detector()
    image = _logo_image()
    true = (100.0, 100.0)
    circle = module._cap_disc_center(
        image,
        "white_cap",
        100.0,
        100.0,
        34.0,
        34.0,
    )
    assert circle is not None
    moment = _moment_center(image, "white_cap", 100.0, 100.0, 34.0, 34.0)
    circle_error = math.hypot(circle[0] - true[0], circle[1] - true[1])
    moment_error = math.hypot(moment[0] - true[0], moment[1] - true[1])
    assert circle_error < moment_error


def main():
    tests = (
        test_lock_damps_jitter,
        test_lock_survives_one_missed_frame,
        test_lock_rejects_teleport,
        test_lock_drops_after_miss_limit,
        test_circle_centre_ignores_logo_better_than_moments,
    )
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:
            failed += 1
            print(f"FAIL {test.__name__}: {error}")
    print(f"failed {failed}, total {len(tests)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
