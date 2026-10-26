"""Offline checks for sticky-note drops. No arm motion."""

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def _install_import_stubs():
    """Load pick_place without ROS or the robot interfaces."""

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

    class _Service:
        pass

    class _Error(Exception):
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
    callbacks = types.ModuleType("rclpy.callback_groups")
    callbacks.MutuallyExclusiveCallbackGroup = _Node
    executors = types.ModuleType("rclpy.executors")
    executors.ExternalShutdownException = _Error
    executors.MultiThreadedExecutor = _Node
    rclpy.node = node_mod
    rclpy.qos = qos_mod
    rclpy.callback_groups = callbacks
    rclpy.executors = executors

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object

    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.Image = _Service
    sensor_msg.CameraInfo = _Service
    sensor_msg.JointState = _Service
    sensor.msg = sensor_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Bool = _Service
    std_msg.Float64MultiArray = _Service
    std_msgs.msg = std_msg

    std_srvs = types.ModuleType("std_srvs")
    std_srv = types.ModuleType("std_srvs.srv")
    std_srv.Trigger = _Service
    std_srvs.srv = std_srv

    ament = types.ModuleType("ament_index_python")
    ament_packages = types.ModuleType("ament_index_python.packages")
    ament_packages.PackageNotFoundError = RuntimeError
    ament_packages.get_package_share_directory = lambda _name: ""
    ament.packages = ament_packages

    interfaces = types.ModuleType("mycobot_interfaces")
    interface_srv = types.ModuleType("mycobot_interfaces.srv")
    interface_srv.MoveCoords = _Service
    interface_srv.MoveJoints = _Service
    interface_srv.SetGripper = _Service
    interfaces.srv = interface_srv

    vision = types.ModuleType("vision_msgs")
    vision_msg = types.ModuleType("vision_msgs.msg")
    vision_msg.Detection2DArray = _Service
    vision.msg = vision_msg

    modules = {
        "rclpy": rclpy,
        "rclpy.node": node_mod,
        "rclpy.qos": qos_mod,
        "rclpy.callback_groups": callbacks,
        "rclpy.executors": executors,
        "cv_bridge": cv_bridge,
        "sensor_msgs": sensor,
        "sensor_msgs.msg": sensor_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msg,
        "std_srvs": std_srvs,
        "std_srvs.srv": std_srv,
        "ament_index_python": ament,
        "ament_index_python.packages": ament_packages,
        "mycobot_interfaces": interfaces,
        "mycobot_interfaces.srv": interface_srv,
        "vision_msgs": vision,
        "vision_msgs.msg": vision_msg,
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


def load_pick_place():
    """Import pick_place without constructing the ROS node."""
    _install_import_stubs()
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pick_place.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pick_place_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yellow_scene():
    """Three separated yellow squares on a white table."""
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:, :] = (0, 0, 230)
    boxes = (
        (169, 243, 96, 95),
        (278, 242, 94, 94),
        (381, 241, 102, 92),
    )
    for x_px, y_px, width, height in boxes:
        hsv[y_px:y_px + height, x_px:x_px + width] = (28, 80, 210)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_live_frame_finds_three_notes_left_to_right():
    module = load_pick_place()
    image = cv2.imread(
        str(Path(__file__).parent / "fixtures" / "overview_notes.jpg")
    )
    assert image is not None
    notes = module.find_yellow_notes(image)
    assert len(notes) == 3
    centers = [note["u"] for note in notes]
    assert centers == sorted(centers)
    assert abs(notes[0]["u"] - 217) < 20
    assert abs(notes[1]["u"] - 325) < 20
    assert abs(notes[2]["u"] - 432) < 20


def test_red_cap_matches_the_red_note():
    module = load_pick_place()
    image = _yellow_scene()
    words = iter(("WHITE", "RED", "BLACK"))

    def reader(_crop, _prepared):
        return next(words)

    point, error = module.match_cap_note(
        image, "red_cap", reader=reader,
    )
    assert error is None
    assert abs(point[0] - 325) < 15
    assert abs(point[1] - 289) < 15


def test_tesseract_suffix_still_reads_red():
    module = load_pick_place()
    assert module.classify_note_word("RED\n\f") == "red"


def test_unreadable_laptop_reply_is_not_a_color():
    module = load_pick_place()
    assert module.classify_note_word("UNREADABLE") is None
    assert module.classify_note_word("Red") == "red"


def test_laptop_reader_failure_refuses_the_drop():
    module = load_pick_place()
    image = _yellow_scene()

    def reader(_crop, _prepared):
        raise module.NoteReaderError(
            "The laptop note reader is not reachable"
        )

    point, error = module.match_cap_note(
        image, "red_cap", reader=reader,
    )
    assert point is None
    assert "laptop" in error.casefold()
    assert "not dropped" in error.casefold()


def test_missing_label_refuses_the_drop():
    module = load_pick_place()
    image = _yellow_scene()
    words = iter(("WHITE", "hidden", "BLACK"))

    def reader(_crop, _prepared):
        return next(words)

    point, error = module.match_cap_note(
        image, "red_cap", reader=reader,
    )
    assert point is None
    assert "red" in error.casefold()
    assert "not dropped" in error.casefold()
