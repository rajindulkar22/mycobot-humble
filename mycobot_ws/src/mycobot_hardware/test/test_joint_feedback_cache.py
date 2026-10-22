"""Offline fake-SDK checks for measured-feedback caching.

The container pymycobot install and the historical failure logs were
not read. This file does not open a serial port.
"""

import importlib.util
import math
import struct
import sys
import threading
import time
import types
from pathlib import Path


def usable_joint_degrees(angles):
    """Return six finite degree values, or None when feedback is unusable."""
    if not isinstance(angles, list) or len(angles) != 6:
        return None
    values = []
    for angle in angles:
        try:
            value = float(angle)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


class FeedbackCache:
    """Publish policy proposed for the hardware node."""

    def __init__(self):
        self.now = 0
        self.measured_degrees = None
        self.measured_stamp = None
        self.joint_states = []
        self.commanded_states = []
        self.serial_calls = []

    def store_measurement(self, angles):
        usable = usable_joint_degrees(angles)
        if usable is None:
            return None
        self.measured_degrees = list(usable)
        self.measured_stamp = self.now
        self.publish_measured(self.measured_degrees, self.measured_stamp)
        return self.measured_degrees

    def publish_measured(self, angles, stamp):
        self.joint_states.append((stamp, list(angles)))

    def publish_commanded(self, angles):
        cached_degrees = (
            None if self.measured_degrees is None
            else list(self.measured_degrees)
        )
        cached_stamp = self.measured_stamp
        self.commanded_states.append((self.now, list(angles)))
        if (
            self.measured_degrees != cached_degrees
            or self.measured_stamp != cached_stamp
        ):
            raise AssertionError(
                "command publication changed the measured cache"
            )

    def publish_hot_stream(self, commanded):
        self.publish_commanded(commanded)
        if self.measured_degrees is not None:
            self.publish_measured(
                self.measured_degrees,
                self.measured_stamp,
            )


class FakeSDK:
    def __init__(self, angles, moving=0, error=0, reply=1):
        self.angles = angles
        self.moving = moving
        self.error = error
        self.reply = reply
        self.calls = []
        self.stopped = 0

    def get_angles(self):
        self.calls.append("get_angles")
        return self.angles

    def is_moving(self):
        self.calls.append("is_moving")
        return self.moving

    def get_error_information(self):
        self.calls.append("get_error")
        return self.error

    def send_angle(self, joint_id, target, speed, _async=False):
        self.calls.append(("send_angle", joint_id, target, speed, _async))
        return None if _async else self.reply

    def stop(self):
        self.stopped += 1


def completes(measured, target, tolerance, moving):
    return abs(measured - target) <= tolerance and moving == 0


def poll_once(sdk, joint_id, target, tolerance, stop_requested, cache):
    if stop_requested:
        return True, False, "Movement stopped through /mycobot/stop"
    angles = sdk.get_angles()
    moving = sdk.is_moving()
    error = sdk.get_error_information()
    if error != 0:
        sdk.stop()
        return (
            True,
            False,
            f"Controller reported error {error}; stop command sent",
        )
    usable = cache.store_measurement(angles)
    if usable is None:
        sdk.stop()
        return (
            True,
            False,
            "Invalid joint feedback during movement: "
            f"{angles}; stop command sent",
        )
    final = usable[joint_id - 1]
    if completes(final, target, tolerance, moving):
        return True, True, "completed"
    return False, False, "waiting"


def test_feedback_cache():
    results = []

    def check(name, cond):
        results.append(cond)
        print(("PASS " if cond else "FAIL ") + name)

    target, final = -6.02, -8.34
    check(
        "reported sample misses 0.5 degree tolerance",
        not completes(final, target, 0.5, 0),
    )
    check(
        "reported sample misses 1.5 degree tolerance",
        not completes(final, target, 1.5, 0),
    )

    sdk = FakeSDK(-1, reply=1)
    cache = FeedbackCache()
    sent = sdk.send_angle(4, target, 5, _async=False)
    done, success, message = poll_once(sdk, 4, target, 0.5, False, cache)
    check("sync reply 1 is acknowledgement only", sent == 1 and not success)
    check(
        "get_angles -1 stops the wait",
        done and sdk.stopped == 1 and "-1" in message,
    )
    check("-1 does not create a measurement stamp", cache.measured_stamp is None)

    sdk = FakeSDK([-7.82, 0, 0, float("nan"), 0, 0])
    cache = FeedbackCache()
    done, success, _message = poll_once(sdk, 4, target, 0.5, False, cache)
    check(
        "nonfinite sample stops the wait",
        done and not success and sdk.stopped == 1,
    )

    sdk = FakeSDK([-7.82, 0, 0, -8.34, 0, 0], error=4)
    cache = FeedbackCache()
    _done, success, message = poll_once(sdk, 4, target, 0.5, False, cache)
    check(
        "controller error stops the wait",
        "error 4" in message and not success,
    )

    sdk = FakeSDK([-7.82, 1, 2, -8.34, 3, 4])
    cache = FeedbackCache()
    cache.now = 100
    done, success, _message = poll_once(sdk, 4, target, 0.5, False, cache)
    check("out-of-tolerance sample does not succeed", not done and not success)
    check("measurement stamp is the read time", cache.measured_stamp == 100)

    _done, _success, message = poll_once(sdk, 4, target, 0.5, True, cache)
    check(
        "stop request returns before another read",
        message.startswith("Movement stopped")
        and sdk.calls.count("get_angles") == 1,
    )

    measured = [-10.0, -20.0, -30.0, -7.82, -50.0, -60.0]
    commanded = [-10.0, -20.0, -30.0, -6.02, -50.0, -60.0]
    cache = FeedbackCache()
    cache.now = 100
    cache.store_measurement(measured)
    cache.now = 250
    cache.publish_hot_stream(commanded)
    cache.now = 400
    cache.publish_hot_stream(commanded)

    check(
        "cached feedback keeps the original stamp",
        cache.joint_states == [
            (100, measured),
            (100, measured),
            (100, measured),
        ],
    )
    check(
        "repeated cache copies are not newer than the measurement",
        all(stamp == 100 for stamp, _sample in cache.joint_states),
    )
    check(
        "hot-stream cached feedback cannot appear newer than the last measurement",
        max(stamp for stamp, _sample in cache.joint_states) == 100
        and min(stamp for stamp, _sample in cache.commanded_states) > 100,
    )
    check(
        "command stamps stay off the measured cache",
        cache.measured_stamp == 100,
    )
    check(
        "command publications use their own later stamps",
        [stamp for stamp, _sample in cache.commanded_states] == [250, 400],
    )
    check(
        "hot-stream joint_states stay on the measured angle",
        all(sample[3] == -7.82 for _stamp, sample in cache.joint_states),
    )
    check(
        "commanded J4 is not written into the measured cache",
        cache.measured_degrees[3] == -7.82,
    )
    check("hot stream performs no serial read", cache.serial_calls == [])

    raw = int(target * 100)
    packet = bytes([0xFE, 0xFE, 6, 0x21, 4, *struct.pack(">h", raw), 5, 0xFA])
    check(
        "constructed send_angle packet uses joint 4 and -6.02 degrees",
        packet == bytes.fromhex("FEFE062104FDA605FA"),
    )

    print(f"failed {len(results) - sum(results)}")
    print(f"total {len(results)}")
    assert all(results)


class _Header:
    def __init__(self):
        self.stamp = None


class _JointState:
    def __init__(self):
        self.header = _Header()
        self.name = []
        self.position = []


class _Float64MultiArray:
    def __init__(self):
        self.data = []


class _Publisher:
    def __init__(self, sink):
        self.sink = sink

    def publish(self, message):
        self.sink.append(message)


class _Clock:
    def __init__(self):
        self.sec = 0

    def now(self):
        sec = self.sec

        class _Reading:
            def to_msg(self):
                return ("stamp", sec)

        return _Reading()


class DriverSDK:
    """Fake controller used by the modified driver methods."""

    def __init__(self, angles, moving=0, errors=None, reply=1):
        self.angles = list(angles)
        self.moving = moving
        self.errors = [0] if errors is None else list(errors)
        self.reply = reply
        self.calls = []
        self.stopped = 0
        self.fresh_mode = 0

    def _take(self, values):
        if len(values) > 1:
            return values.pop(0)
        return values[0]

    def is_controller_connected(self):
        self.calls.append("is_controller_connected")
        return 1

    def is_power_on(self):
        self.calls.append("is_power_on")
        return 1

    def get_error_information(self):
        self.calls.append("get_error")
        return self._take(self.errors)

    def is_moving(self):
        self.calls.append("is_moving")
        return self.moving

    def get_angles(self):
        self.calls.append("get_angles")
        value = self._take(self.angles)
        return value

    def set_fresh_mode(self, mode):
        self.calls.append(("set_fresh_mode", mode))
        self.fresh_mode = mode

    def get_fresh_mode(self):
        self.calls.append("get_fresh_mode")
        return self.fresh_mode

    def send_angle(self, joint_id, target, speed, _async=False):
        self.calls.append(("send_angle", joint_id, target, speed, _async))
        return None if _async else self.reply

    def get_coords(self):
        self.calls.append("get_coords")
        return [0, 0, 0, 0, 0, 0]

    def get_gripper_value(self, gripper_type=1):
        self.calls.append("get_gripper_value")
        return None

    def stop(self):
        self.calls.append("stop")
        self.stopped += 1


def _install_import_stubs():
    class _Node:
        def __init__(self, *args, **kwargs):
            pass

    class _Group:
        def __init__(self, *args, **kwargs):
            pass

    class _Executor:
        def __init__(self, *args, **kwargs):
            pass

    class _Service:
        pass

    class _MyCobot280:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("offline test must not open the serial port")

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None
    rclpy.ok = lambda: False
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    callback_mod = types.ModuleType("rclpy.callback_groups")
    callback_mod.MutuallyExclusiveCallbackGroup = _Group
    callback_mod.ReentrantCallbackGroup = _Group
    executor_mod = types.ModuleType("rclpy.executors")
    executor_mod.ExternalShutdownException = type(
        "ExternalShutdownException",
        (Exception,),
        {},
    )
    executor_mod.MultiThreadedExecutor = _Executor
    sensor_mod = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.JointState = _JointState
    sensor_mod.msg = sensor_msg
    std_msgs = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Float64MultiArray = _Float64MultiArray
    std_msgs.msg = std_msg
    std_srvs = types.ModuleType("std_srvs")
    std_srv = types.ModuleType("std_srvs.srv")
    std_srv.Trigger = _Service
    std_srvs.srv = std_srv
    interfaces = types.ModuleType("mycobot_interfaces")
    interfaces_srv = types.ModuleType("mycobot_interfaces.srv")
    interfaces_srv.MoveJoint = _Service
    interfaces_srv.MoveJoints = _Service
    interfaces_srv.SetGripper = _Service
    interfaces_srv.MoveCoords = _Service
    interfaces.srv = interfaces_srv
    pymycobot = types.ModuleType("pymycobot")
    pymycobot.MyCobot280 = _MyCobot280
    modules = {
        "rclpy": rclpy,
        "rclpy.node": node_mod,
        "rclpy.callback_groups": callback_mod,
        "rclpy.executors": executor_mod,
        "sensor_msgs": sensor_mod,
        "sensor_msgs.msg": sensor_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msg,
        "std_srvs": std_srvs,
        "std_srvs.srv": std_srv,
        "mycobot_interfaces": interfaces,
        "mycobot_interfaces.srv": interfaces_srv,
        "pymycobot": pymycobot,
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


def load_driver():
    """Import the saved driver without constructing the ROS node."""
    _install_import_stubs()
    driver_path = (
        Path(__file__).resolve().parents[1]
        / "mycobot_hardware"
        / "state_publisher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mycobot_state_publisher_under_test",
        driver_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bare_node(module):
    """Build a driver instance without __init__, serial, or parameters."""
    node = module.MyCobotStatePublisher.__new__(module.MyCobotStatePublisher)
    node.logs = []

    class _Logger:
        def info(self, text):
            node.logs.append(text)

        def warning(self, text):
            node.logs.append(text)

        def error(self, text):
            node.logs.append(text)

    node.get_logger = lambda: _Logger()
    node._clock = _Clock()
    node.get_clock = lambda: node._clock
    node.joint_messages = []
    node.command_messages = []
    node.pose_messages = []
    node.joint_publisher = _Publisher(node.joint_messages)
    node.commanded_joint_publisher = _Publisher(node.command_messages)
    node.pose_publisher = _Publisher(node.pose_messages)
    node.gripper_state_mutex = threading.Lock()
    node.last_gripper_value = None
    node.gripper_open_angle = 0.15
    node.gripper_closed_angle = -0.74
    node.serial_mutex = threading.Lock()
    node.stop_requested = threading.Event()
    node._guarded_motion_active = threading.Event()
    node._commanded_degrees = None
    node._measured_degrees = None
    node._measured_stamp = None
    node._fresh_mode_on = None
    node._stream_sent = 0.0
    node.robot = None
    node.max_speed = 10
    node.joint_limit_margin = 2.0
    node.max_joint_step = 5.0
    node.motion_enabled = True
    node.position_tolerance = 0.5
    node.motion_timeout = 10.0
    node.motion_poll_period = 0.0
    return node


class _Request:
    def __init__(self):
        self.joint_id = 4
        self.target_degrees = -6.02
        self.speed = 5
        self.execute = True


class _Response:
    def __init__(self):
        self.success = None
        self.message = ""
        self.start_degrees = None
        self.final_degrees = None


def _move(node, robot):
    node.robot = robot
    response = _Response()
    node.handle_move_joint(_Request(), response)
    return response


def test_modified_source():
    """Call the saved driver methods with a fake SDK and no serial port."""
    results = []

    def check(name, cond):
        results.append(cond)
        print(("PASS " if cond else "FAIL ") + name)

    module = load_driver()
    measured = [-10.0, -20.0, -30.0, -7.82, -50.0, -60.0]
    commanded = [-10.0, -20.0, -30.0, -6.02, -50.0, -60.0]
    node = bare_node(module)
    node._clock.sec = 100
    node._store_measurement(measured)
    stored_stamp = node._measured_stamp
    node._clock.sec = 250
    node._commanded_degrees = list(commanded)
    node._stream_sent = time.monotonic()

    class _HotRobot:
        def get_angles(self):
            raise AssertionError("hot stream read the serial port")

        def get_coords(self):
            raise AssertionError("hot stream read the serial port")

        def get_gripper_value(self, gripper_type=1):
            raise AssertionError("hot stream read the serial port")

    node.robot = _HotRobot()
    cached_degrees = list(node._measured_degrees)
    node.publish_state()
    node._clock.sec = 400
    node.publish_state()
    node._publish_commanded_degrees(commanded)

    check(
        "source republish keeps the measurement stamp object",
        all(
            message.header.stamp is stored_stamp
            for message in node.joint_messages
        ),
    )
    check(
        "source cached feedback cannot appear newer than the measurement",
        all(message.header.stamp == ("stamp", 100) for message in node.joint_messages)
        and [message.header.stamp[1] for message in node.command_messages] == [
            250,
            400,
            400,
        ],
    )
    check(
        "source command publication leaves the measured cache unchanged",
        node._measured_degrees == cached_degrees
        and node._measured_stamp is stored_stamp,
    )
    check(
        "source joint_states stay on the measured Joint 4 angle",
        all(
            abs(message.position[3] - math.radians(-7.82)) < 1e-12
            for message in node.joint_messages
        ),
    )
    check(
        "source commanded topic carries the commanded Joint 4 angle",
        all(
            abs(message.position[3] - math.radians(-6.02)) < 1e-12
            for message in node.command_messages
        ),
    )

    cold = bare_node(module)
    cold.robot = DriverSDK([measured])
    cold._clock.sec = 100
    cold.publish_state()
    cold._clock.sec = 250
    cold.publish_state()
    check(
        "source cold reads store a new measurement stamp",
        [message.header.stamp for message in cold.joint_messages]
        == [("stamp", 100), ("stamp", 250)],
    )

    check(
        "source rejects integer -1 feedback",
        module.MyCobotStatePublisher._usable_joint_degrees(node, -1) is None,
    )
    check(
        "source rejects nonfinite feedback",
        module.MyCobotStatePublisher._usable_joint_degrees(
            node,
            [-7.82, 0, 0, float("nan"), 0, 0],
        )
        is None,
    )

    start = [-7.82, 0, 0, -7.82, 0, 0]
    missed = [-7.82, 0, 0, -8.34, 0, 0]
    miss = bare_node(module)
    original_monotonic = time.monotonic
    deadline_start = original_monotonic()
    monotonic_values = iter((deadline_start, deadline_start, deadline_start + 1000))

    def _limited_monotonic():
        try:
            return next(monotonic_values)
        except StopIteration:
            return deadline_start + 1000

    time.monotonic = _limited_monotonic
    try:
        missed_response = _move(miss, DriverSDK([start, missed], reply=1))
    finally:
        time.monotonic = original_monotonic
    check(
        "source reply 1 does not complete the missed Joint 4 move",
        missed_response.success is False
        and "Movement completed" not in missed_response.message
        and any(
            call[:4] == ("send_angle", 4, -6.02, 5)
            for call in miss.robot.calls
            if isinstance(call, tuple)
        ),
    )
    check(
        "source poll log records measured, moving, and controller error",
        any("J4 poll:" in line and "controller_error=" in line for line in miss.logs),
    )

    invalid = bare_node(module)
    invalid_response = _move(invalid, DriverSDK([start, -1], reply=1))
    check(
        "source get_angles -1 stops the single-joint wait",
        invalid.robot.stopped == 1
        and "-1" in invalid_response.message
        and invalid._measured_stamp is None,
    )

    nonfinite = bare_node(module)
    nonfinite_response = _move(
        nonfinite,
        DriverSDK([start, [-7.82, 0, 0, float("nan"), 0, 0]], reply=1),
    )
    check(
        "source nonfinite sample stops the single-joint wait",
        nonfinite.robot.stopped == 1
        and nonfinite_response.success is False
        and "Invalid joint feedback" in nonfinite_response.message,
    )

    errored = bare_node(module)
    errored_response = _move(
        errored,
        DriverSDK([start, missed], errors=[0, 4], reply=1),
    )
    check(
        "source controller error 4 stops the single-joint wait",
        errored.robot.stopped == 1
        and "error 4" in errored_response.message
        and errored_response.success is False,
    )

    stopped = bare_node(module)

    class _StopAfterCommand(DriverSDK):
        def send_angle(self, joint_id, target, speed, _async=False):
            result = super().send_angle(joint_id, target, speed, _async)
            stopped.stop_requested.set()
            return result

    robot = _StopAfterCommand([start], reply=1)
    stopped.robot = robot
    stopped_response = stopped.handle_move_joint(_Request(), _Response())
    check(
        "source stop request returns before another angle read",
        stopped_response.message.startswith("Movement stopped")
        and robot.calls.count("get_angles") == 1,
    )

    print(f"source failed {len(results) - sum(results)}")
    print(f"source total {len(results)}")
    assert all(results)


if __name__ == "__main__":
    test_feedback_cache()
    test_modified_source()
