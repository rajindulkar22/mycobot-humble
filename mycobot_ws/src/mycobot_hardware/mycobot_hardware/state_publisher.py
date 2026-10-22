#!/usr/bin/env python3
"""Safe ROS 2 Humble hardware node for the myCobot 280."""

import fcntl
import math
import os
import threading
import time

import rclpy
from mycobot_interfaces.srv import (
    MoveCoords,
    MoveJoint,
    MoveJoints,
    SetGripper,
)
from pymycobot import MyCobot280
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import (
    ExternalShutdownException,
    MultiThreadedExecutor,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class MyCobotStatePublisher(Node):
    """Publish robot state and provide guarded control services."""

    JOINT_NAMES = [
        "joint2_to_joint1",
        "joint3_to_joint2",
        "joint4_to_joint3",
        "joint5_to_joint4",
        "joint6_to_joint5",
        "joint6output_to_joint6",
    ]

    GRIPPER_JOINT_NAME = "gripper_controller"

    JOINT_LIMITS_DEGREES = {
        1: (-168.0, 168.0),
        2: (-140.0, 140.0),
        3: (-150.0, 150.0),
        4: (-150.0, 150.0),
        5: (-155.0, 160.0),
        6: (-180.0, 180.0),
    }

    def __init__(self):
        super().__init__("mycobot_state_publisher")

        self.declare_parameter("port", "/dev/ttyTHS1")
        self.declare_parameter("baud", 1_000_000)
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("max_joint_step_degrees", 5.0)
        self.declare_parameter("max_speed", 10)
        self.declare_parameter("joint_limit_margin_degrees", 2.0)
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter("multi_joint_motion_enabled", False)
        self.declare_parameter("replace_motion_enabled", False)
        self.declare_parameter("motion_timeout_seconds", 6.0)
        self.declare_parameter("position_tolerance_degrees", 1.5)
        self.declare_parameter("motion_poll_period_seconds", 0.2)
        self.declare_parameter("max_gripper_speed", 20)
        self.declare_parameter("gripper_motion_enabled", False)
        self.declare_parameter("gripper_timeout_seconds", 5.0)
        self.declare_parameter("gripper_settle_seconds", 2.0)
        self.declare_parameter("gripper_open_min_value", 90)
        self.declare_parameter("gripper_closed_max_value", 10)
        self.declare_parameter("gripper_initial_value", 0)
        self.declare_parameter("gripper_open_angle_radians", 0.15)
        self.declare_parameter("gripper_closed_angle_radians", -0.74)
        self.declare_parameter("coord_motion_enabled", False)
        self.declare_parameter("coord_min_x_mm", 50.0)
        self.declare_parameter("coord_max_x_mm", 280.0)
        self.declare_parameter("coord_min_y_mm", -200.0)
        self.declare_parameter("coord_max_y_mm", 200.0)
        self.declare_parameter("coord_min_z_mm", 20.0)
        self.declare_parameter("coord_max_z_mm", 250.0)
        self.declare_parameter("coord_tolerance_mm", 8.0)
        self.declare_parameter("coord_angle_tolerance_degrees", 8.0)
        self.declare_parameter("coord_tool_tolerance_degrees", 25.0)

        port = str(self.get_parameter("port").value)
        baud = int(self.get_parameter("baud").value)
        publish_rate = float(self.get_parameter("publish_rate").value)

        self.max_joint_step = float(
            self.get_parameter("max_joint_step_degrees").value
        )
        self.max_speed = int(self.get_parameter("max_speed").value)
        self.joint_limit_margin = float(
            self.get_parameter("joint_limit_margin_degrees").value
        )
        self.motion_enabled = bool(
            self.get_parameter("motion_enabled").value
        )
        self.multi_joint_motion_enabled = bool(
            self.get_parameter("multi_joint_motion_enabled").value
        )
        self.replace_motion_enabled = bool(
            self.get_parameter("replace_motion_enabled").value
        )
        self.motion_timeout = float(
            self.get_parameter("motion_timeout_seconds").value
        )
        self.position_tolerance = float(
            self.get_parameter("position_tolerance_degrees").value
        )
        self.motion_poll_period = float(
            self.get_parameter("motion_poll_period_seconds").value
        )
        self.max_gripper_speed = int(self.get_parameter("max_gripper_speed").value)
        self.gripper_motion_enabled = bool(self.get_parameter("gripper_motion_enabled").value)
        self.gripper_timeout = float(
            self.get_parameter("gripper_timeout_seconds").value
        )
        self.gripper_settle = float(
            self.get_parameter("gripper_settle_seconds").value
        )
        self.gripper_open_min = int(
            self.get_parameter("gripper_open_min_value").value
        )
        self.gripper_closed_max = int(
            self.get_parameter("gripper_closed_max_value").value
        )
        self.gripper_initial_value = int(
            self.get_parameter("gripper_initial_value").value
        )
        self.gripper_open_angle = float(
            self.get_parameter("gripper_open_angle_radians").value
        )
        self.gripper_closed_angle = float(
            self.get_parameter("gripper_closed_angle_radians").value
        )
        self.coord_motion_enabled = bool(
            self.get_parameter("coord_motion_enabled").value
        )
        self.coord_min_x = float(self.get_parameter("coord_min_x_mm").value)
        self.coord_max_x = float(self.get_parameter("coord_max_x_mm").value)
        self.coord_min_y = float(self.get_parameter("coord_min_y_mm").value)
        self.coord_max_y = float(self.get_parameter("coord_max_y_mm").value)
        self.coord_min_z = float(self.get_parameter("coord_min_z_mm").value)
        self.coord_max_z = float(self.get_parameter("coord_max_z_mm").value)
        self.coord_tolerance_mm = float(
            self.get_parameter("coord_tolerance_mm").value
        )
        self.coord_angle_tolerance = float(
            self.get_parameter("coord_angle_tolerance_degrees").value
        )
        self.coord_tool_tolerance = float(
            self.get_parameter("coord_tool_tolerance_degrees").value
        )

        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be greater than zero")

        if self.motion_timeout <= 0.0:
            raise ValueError("motion_timeout_seconds must be positive")

        if self.position_tolerance <= 0.0:
            raise ValueError("position_tolerance_degrees must be positive")

        if self.motion_poll_period <= 0.0:
            raise ValueError("motion_poll_period_seconds must be positive")

        if self.gripper_settle <= 0.0:
            raise ValueError("gripper_settle_seconds must be positive")

        if self.gripper_settle >= self.gripper_timeout:
            raise ValueError(
                "gripper_settle_seconds must be shorter than "
                "gripper_timeout_seconds"
            )

        if (
            self.gripper_initial_value < -1
            or self.gripper_initial_value > 100
        ):
            raise ValueError(
                "gripper_initial_value must be -1 or between 0 and 100"
            )

        if not math.isfinite(self.gripper_open_angle):
            raise ValueError(
                "gripper_open_angle_radians must be finite"
            )

        if not math.isfinite(self.gripper_closed_angle):
            raise ValueError(
                "gripper_closed_angle_radians must be finite"
            )

        if not -0.74 <= self.gripper_open_angle <= 0.15:
            raise ValueError(
                "gripper_open_angle_radians is outside the URDF limit"
            )

        if not -0.74 <= self.gripper_closed_angle <= 0.15:
            raise ValueError(
                "gripper_closed_angle_radians is outside the URDF limit"
            )

        if self.coord_min_x >= self.coord_max_x:
            raise ValueError("coord_min_x_mm must be below coord_max_x_mm")
        if self.coord_min_y >= self.coord_max_y:
            raise ValueError("coord_min_y_mm must be below coord_max_y_mm")
        if self.coord_min_z >= self.coord_max_z:
            raise ValueError("coord_min_z_mm must be below coord_max_z_mm")
        if self.coord_tolerance_mm <= 0.0:
            raise ValueError("coord_tolerance_mm must be positive")
        if self.coord_angle_tolerance <= 0.0:
            raise ValueError(
                "coord_angle_tolerance_degrees must be positive"
            )

        lock_name = f"mycobot_{os.path.basename(port)}.lock"
        lock_path = os.path.join("/tmp", lock_name)

        self.port_lock = open(lock_path, "w", encoding="utf-8")

        try:
            fcntl.flock(
                self.port_lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self.port_lock.close()
            raise RuntimeError(
                f"Serial port {port} is already owned by another "
                "mycobot_hardware process"
            ) from error

        self.port_lock.write(str(os.getpid()))
        self.port_lock.flush()

        self.serial_mutex = threading.Lock()
        self.stop_requested = threading.Event()
        self._guarded_motion_active = threading.Event()
        self._commanded_degrees = None
        self._measured_degrees = None
        self._measured_stamp = None
        self._fresh_mode_on = None
        self._stream_check = 0.0
        self._stream_sent = 0.0

        self.gripper_state_mutex = threading.Lock()
        self.last_gripper_value = (
            self.gripper_initial_value
            if self.gripper_initial_value >= 0
            else None
        )

        self.io_callback_group = ReentrantCallbackGroup()
        self.motion_callback_group = MutuallyExclusiveCallbackGroup()

        self.get_logger().info(f"Acquired serial lock: {lock_path}")

        self.robot = MyCobot280(port, baud)

        self.joint_publisher = self.create_publisher(
            JointState,
            "joint_states",
            10,
        )
        self.commanded_joint_publisher = self.create_publisher(
            JointState,
            "mycobot/commanded_joint_states",
            10,
        )
        self.pose_publisher = self.create_publisher(
            Float64MultiArray,
            "mycobot/tool_pose_raw",
            10,
        )

        self.status_service = self.create_service(
            Trigger,
            "mycobot/status",
            self.handle_status,
            callback_group=self.io_callback_group,
        )
        self.stop_service = self.create_service(
            Trigger,
            "mycobot/stop",
            self.handle_stop,
            callback_group=self.io_callback_group,
        )
        self.release_service = self.create_service(
            Trigger,
            "mycobot/release_servos",
            self.handle_release_servos,
            callback_group=self.motion_callback_group,
        )
        self.focus_service = self.create_service(
            Trigger,
            "mycobot/focus_servos",
            self.handle_focus_servos,
            callback_group=self.motion_callback_group,
        )
        self.move_joint_service = self.create_service(
            MoveJoint,
            "mycobot/move_joint",
            self.handle_move_joint,
            callback_group=self.motion_callback_group,
        )
        self.move_joints_service = self.create_service(
            MoveJoints,
            "mycobot/move_joints",
            self.handle_move_joints,
            callback_group=self.motion_callback_group,
        )
        self.gripper_service = self.create_service(
            SetGripper,
            "mycobot/set_gripper",
            self.handle_set_gripper,
            callback_group=self.motion_callback_group,
        )
        self.coords_service = self.create_service(
            MoveCoords,
            "mycobot/move_coords",
            self.handle_move_coords,
            callback_group=self.motion_callback_group,
        )

        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_state,
            callback_group=self.io_callback_group,
        )

        self.get_logger().info(
            f"Connected on {port} at {baud} baud"
        )
        self.get_logger().info(
            f"Physical motion enabled: {self.motion_enabled}"
        )
        self.get_logger().info(
            "Multi-joint motion enabled: "
            f"{self.multi_joint_motion_enabled}"
        )
        self.get_logger().info(
            "Streaming multi-joint targets: "
            f"{self.replace_motion_enabled}"
        )
        self.get_logger().info(
            "Coordinate motion enabled: "
            f"{self.coord_motion_enabled}"
        )


        # Timed runner uses this node's serial owner and motion callback group.
        from mycobot_hardware.timed_approach import TimedApproach
        self.declare_parameter("timed_descent_enabled", False)
        self.declare_parameter("timed_approach_enabled", False)
        self.declare_parameter("timed_approach_time_scale", 3.0)
        self.declare_parameter("timed_approach_plan_file", "/tmp/cap_approach_timed.json")
        self._timed_approach = TimedApproach(self)
        self._timed_validate_service = self.create_service(
            Trigger, "mycobot/validate_timed_approach",
            self._timed_approach.handle_validate,
            callback_group=self.motion_callback_group,
        )
        self._timed_execute_service = self.create_service(
            Trigger, "mycobot/execute_timed_approach",
            self._timed_approach.handle_execute,
            callback_group=self.motion_callback_group,
        )

    def cache_gripper_value(self, raw_value):
        """Cache valid adaptive-gripper feedback and reject sentinels."""
        if not isinstance(raw_value, (int, float)):
            return None

        if not math.isfinite(float(raw_value)):
            return None

        value = int(raw_value)

        if value < 0 or value > 100:
            return None

        with self.gripper_state_mutex:
            self.last_gripper_value = value

        return value

    def get_cached_gripper_value(self):
        """Return the most recent valid or configured gripper value."""
        with self.gripper_state_mutex:
            return self.last_gripper_value

    def clear_cached_gripper_value(self):
        """Mark gripper position unknown after releasing its torque."""
        with self.gripper_state_mutex:
            self.last_gripper_value = None

    def gripper_value_to_radians(self, value):
        """Map physical 0–100 feedback into the tested URDF range."""
        fraction = float(value) / 100.0
        angle_range = (
            self.gripper_open_angle
            - self.gripper_closed_angle
        )
        return self.gripper_closed_angle + fraction * angle_range

    def handle_status(self, request, response):
        """Return controller, power, error and joint status."""
        del request

        try:
            with self.serial_mutex:
                connected = self.robot.is_controller_connected()
                power_state = self.robot.is_power_on()
                error_state = self.robot.get_error_information()
                moving = self.robot.is_moving()
                angles = self.robot.get_angles()

            valid_angles = isinstance(angles, list) and len(angles) == 6

            response.success = (
                connected == 1
                and power_state == 1
                and error_state == 0
                and valid_angles
            )
            response.message = (
                f"connected={connected}, power={power_state}, "
                f"error={error_state}, moving={moving}, "
                f"angles_degrees={angles}"
            )

        except Exception as error:
            response.success = False
            response.message = f"Status read failed: {error}"

        return response

    def _request_arm_stop(self, shutdown=False):
        self.stop_requested.set()
        self._commanded_degrees = None
        if shutdown:
            self._guarded_motion_active.set()
        # Do not wait indefinitely behind a blocked SDK transaction.
        acquired = self.serial_mutex.acquire(timeout=1.0)
        if not acquired:
            return False, "STOP NOT SENT: serial lock busy; use physical power control"
        try:
            # Stop takes priority over mode changes, queries, and gripper commands.
            self.robot.stop()
            return True, "Arm stop command sent; physical stop not verified"
        except Exception as error:
            return False, f"STOP FAILED: {error}; use physical power control"
        finally:
            self.serial_mutex.release()

    def handle_stop(self, request, response):
        """Latch arm stop without releasing a held object."""
        del request
        response.success, response.message = self._request_arm_stop()
        return response

    def handle_release_servos(self, request, response):
        """Relax the joints so the arm can be placed by hand."""
        del request
        response.success, response.message = self._set_servo_hold(False)
        return response

    def handle_focus_servos(self, request, response):
        """Lock the joints at the pose where the arm was released."""
        del request
        response.success, response.message = self._set_servo_hold(True)
        return response

    def _clear_contact_error(self, error_state):
        """Clear a collision latch caused by the gripper meeting an object."""
        with self.serial_mutex:
            self.robot.clear_error_information()
            cleared = self.robot.get_error_information()
        self.get_logger().warning(
            f"Cleared contact error {error_state} after the grasp. "
            f"Controller now reports {cleared}."
        )

    def _clear_planning_error(self, error_state):
        """Clear a latched inverse-kinematics error. Hardware faults stay."""
        if error_state not in (32, 33, 34):
            return error_state
        with self.serial_mutex:
            self.robot.clear_error_information()
            cleared = self.robot.get_error_information()
        self.get_logger().warning(
            f"Ignoring planning error {error_state} so a joint move "
            f"can proceed. Controller now reports {cleared}."
        )
        if cleared in (32, 33, 34):
            return 0
        return cleared

    def _set_servo_hold(self, hold):
        """Release or hold every joint. This does not send a target pose."""
        if self._guarded_motion_active.is_set():
            return False, "Arm is moving. Wait until the motion finishes."
        acquired = self.serial_mutex.acquire(timeout=1.0)
        if not acquired:
            return False, "Robot port is busy. Try again."
        try:
            if hold:
                self.robot.focus_all_servos()
                return True, (
                    "Motors are holding. Wait for a fresh pose, "
                    "then capture."
                )
            self.robot.release_all_servos()
            return True, (
                "Motors released. Support the arm and place the "
                "pad centre on the mark."
            )
        except Exception as error:
            action = "hold" if hold else "release"
            return False, f"Could not {action} the motors: {error}"
        finally:
            self.serial_mutex.release()

    def handle_set_gripper(self, request, response):
        """Validate and optionally execute a guarded gripper command."""
        response.success = False
        response.message = ""
        response.start_value = -1
        response.final_value = -1

        state = int(request.state)
        speed = int(request.speed)

        if state not in (0, 1):
            response.message = "state must be 0 (open) or 1 (close)"
            return response

        if speed < 1 or speed > self.max_gripper_speed:
            response.message = (
                f"speed must be between 1 and {self.max_gripper_speed}"
            )
            return response

        command_name = "open" if state == 0 else "close"

        try:
            with self.serial_mutex:
                connected = self.robot.is_controller_connected()
                power_state = self.robot.is_power_on()
                error_state = self.robot.get_error_information()
                arm_moving = self.robot.is_moving()
                raw_value = self.robot.get_gripper_value(
                    gripper_type=1
                )

            if connected != 1:
                response.message = "Robot controller is not connected"
                return response

            if power_state != 1:
                response.message = "Robot is not powered on"
                return response

            error_state = self._clear_planning_error(error_state)
            if error_state != 0:
                response.message = (
                    f"Robot reports error code {error_state}"
                )
                return response

            if arm_moving != 0 and not self.replace_motion_enabled:
                response.message = "Robot arm is currently moving"
                return response

            numeric_feedback = isinstance(raw_value, (int, float))

            if numeric_feedback:
                raw_value = int(raw_value)

            feedback_available = (
                numeric_feedback and 0 <= raw_value <= 100
            )
            feedback_unavailable = (
                numeric_feedback and 101 <= raw_value <= 255
            )

            if not feedback_available and not feedback_unavailable:
                response.message = (
                    f"Invalid gripper response: {raw_value}"
                )
                return response

            if feedback_available:
                start_value = self.cache_gripper_value(raw_value)
                response.start_value = start_value
                response.final_value = start_value
                feedback_text = f"current value {start_value}"
            else:
                start_value = -1
                feedback_text = (
                    f"position feedback unavailable (raw {raw_value})"
                )

            if not request.execute:
                response.success = True
                response.message = (
                    f"DRY RUN accepted: {command_name} gripper, "
                    f"{feedback_text}, speed {speed}"
                )
                return response

            if not self.gripper_motion_enabled:
                response.message = (
                    "Validation passed, but "
                    "gripper_motion_enabled is false"
                )
                return response

            self.stop_requested.clear()

            self.get_logger().warning(
                f"Executing adaptive gripper {command_name} "
                f"at speed {speed}"
            )

            with self.serial_mutex:
                self.robot.set_gripper_state(
                    state,
                    speed,
                    _type_1=1,
                )

            command_started = time.monotonic()
            deadline = command_started + self.gripper_timeout
            final_value = start_value
            contact_error = None
            stable_value = start_value
            stable_since = command_started

            while time.monotonic() < deadline:
                if self.stop_requested.is_set():
                    response.final_value = final_value
                    response.message = (
                        "Gripper operation interrupted by a stop request"
                    )
                    return response

                time.sleep(self.motion_poll_period)

                with self.serial_mutex:
                    updated_value = self.robot.get_gripper_value(
                        gripper_type=1
                    )
                    gripper_moving = self.robot.is_gripper_moving()
                    updated_error = self.robot.get_error_information()

                if self.stop_requested.is_set():
                    response.final_value = final_value
                    response.message = (
                        "Gripper operation interrupted by a stop request"
                    )
                    return response

                if updated_error in (16, 17, 18, 19) and state == 1:
                    # The fingers meeting the cap trips collision
                    # protection. Keep the grip and clear it before
                    # the arm lifts.
                    contact_error = updated_error
                elif updated_error != 0:
                    response.final_value = final_value
                    response.message = (
                        f"Controller reported error {updated_error} "
                        "after the gripper command"
                    )
                    return response

                if isinstance(updated_value, (int, float)):
                    updated_value = int(updated_value)

                    if 0 <= updated_value <= 100:
                        final_value = self.cache_gripper_value(
                            updated_value
                        )
                        response.final_value = final_value

                        if final_value != stable_value:
                            stable_value = final_value
                            stable_since = time.monotonic()
                        stable_for = time.monotonic() - stable_since
                        opened = (
                            state == 0
                            and final_value >= self.gripper_open_min
                            and stable_for >= 0.4
                        )
                        # A bottle cap stops the jaws well before they
                        # meet. A reading near fully open is a pad on
                        # the lid, and a reading near 0 is an empty close.
                        elapsed = time.monotonic() - command_started
                        held = (
                            state == 1
                            and 12 <= final_value <= 86
                            and stable_for >= 0.8
                            and elapsed >= 1.2
                        )
                        jammed = (
                            state == 1
                            and final_value >= 90
                            and stable_for >= 2.0
                            and elapsed >= 3.0
                        )
                        empty = (
                            state == 1
                            and final_value <= self.gripper_closed_max
                            and stable_for >= 0.4
                        )
                        settled = gripper_moving == 0 or stable_for >= 1.2

                        if settled and (
                            opened or held or empty or jammed
                        ):
                            if contact_error is not None:
                                self._clear_contact_error(contact_error)
                            if empty or jammed:
                                response.success = False
                                response.message = (
                                    "Gripper closed empty: "
                                    if empty
                                    else "Gripper stopped on the cap "
                                    "instead of around it: "
                                ) + f"{start_value} -> {final_value}"
                            else:
                                response.success = True
                                if held:
                                    response.message = (
                                        "Gripper closed on an object: "
                                        f"{start_value} -> {final_value}"
                                    )
                                else:
                                    response.message = (
                                        f"Gripper {command_name} completed: "
                                        f"{start_value} -> {final_value}"
                                    )
                            return response

                    elif 101 <= updated_value <= 255:
                        elapsed = time.monotonic() - command_started

                        if (
                            elapsed >= self.gripper_settle
                            and gripper_moving == 0
                        ):
                            inferred_value = (
                                100 if state == 0 else 0
                            )
                            self.cache_gripper_value(
                                inferred_value
                            )
                            response.success = True
                            response.final_value = -1
                            response.message = (
                                f"Gripper {command_name} command sent; "
                                "controller is idle, but position "
                                "feedback is unavailable—verify physically"
                            )
                            return response

            response.final_value = final_value
            response.message = (
                f"Gripper {command_name} timed out after "
                f"{self.gripper_timeout:.1f}s"
            )
            return response

        except Exception as error:
            response.message = f"Gripper command failed: {error}"
            return response

    def handle_move_coords(self, request, response):
        """Move the tool to one table point inside the configured box."""
        response.success = False
        response.message = ""
        response.start_coords = [math.nan] * 6
        response.final_coords = [math.nan] * 6

        coords = [
            float(request.x),
            float(request.y),
            float(request.z),
            float(request.rx),
            float(request.ry),
            float(request.rz),
        ]
        speed = int(request.speed)
        rejection = self._coord_rejection(coords, speed)
        if rejection:
            response.message = rejection
            return response

        if not request.execute:
            response.success = True
            response.message = (
                "DRY RUN accepted: coordinate pose "
                f"{coords}, speed {speed}"
            )
            return response

        if not self.coord_motion_enabled:
            response.message = (
                "Validation passed, but coord_motion_enabled is false"
            )
            return response

        if self.stop_requested.is_set():
            response.message = "Stop is latched; no command sent"
            return response

        started = False
        try:
            with self.serial_mutex:
                connected = self.robot.is_controller_connected()
                power_state = self.robot.is_power_on()
                error_state = self.robot.get_error_information()
                moving = self.robot.is_moving()
                start = self._usable_coords(self.robot.get_coords())

            if connected != 1:
                response.message = "Robot controller is not connected"
                return response
            if power_state != 1:
                response.message = "Robot is not powered on"
                return response
            error_state = self._clear_planning_error(error_state)
            if error_state != 0:
                response.message = (
                    f"Robot reports error code {error_state}"
                )
                return response
            if moving != 0:
                response.message = "Robot is already moving"
                return response
            if start is None:
                response.message = "Current coordinate feedback is invalid"
                return response

            response.start_coords = start
            response.final_coords = start
            self.stop_requested.clear()
            self._guarded_motion_active.set()
            started = True
            self.get_logger().warning(
                f"Executing coordinate pose {coords} at speed {speed}"
            )
            with self.serial_mutex:
                self._set_fresh_mode_locked(False)
                # Mode 0 moves in joint space. Mode 1 is a straight
                # line, and the 280 reports error 33 when that line
                # has no adjacent solution from the current pose.
                self.robot.send_coords(coords, speed, 0, _async=True)

            deadline = time.monotonic() + self.motion_timeout
            final = start
            while time.monotonic() < deadline:
                if self.stop_requested.is_set():
                    response.final_coords = final
                    response.message = (
                        "Coordinate movement stopped through "
                        "/mycobot/stop"
                    )
                    return response
                time.sleep(self.motion_poll_period)
                with self.serial_mutex:
                    updated = self._usable_coords(self.robot.get_coords())
                    moving = self.robot.is_moving()
                    updated_error = self.robot.get_error_information()
                if updated_error != 0:
                    with self.serial_mutex:
                        self.robot.stop()
                    response.final_coords = final
                    response.message = (
                        f"Controller reported error {updated_error}; "
                        "stop command sent"
                    )
                    return response
                if updated is None:
                    continue
                final = updated
                response.final_coords = final
                if self.stop_requested.is_set():
                    response.message = (
                        "Coordinate movement stopped through "
                        "/mycobot/stop"
                    )
                    return response
                if (
                    self._coords_arrived(final, coords)
                    and moving == 0
                ):
                    response.success = True
                    response.message = "Coordinate movement completed"
                    return response

            with self.serial_mutex:
                self.robot.stop()
            response.final_coords = final
            response.message = (
                "Coordinate movement timed out after "
                f"{self.motion_timeout:.1f}s; stop command sent"
            )
            return response
        except Exception as error:
            response.message = f"Coordinate movement failed: {error}"
            return response
        finally:
            if started:
                self._guarded_motion_active.clear()

    def _coord_rejection(self, coords, speed):
        """Return an error when a coordinate target is outside the table."""
        if not all(math.isfinite(value) for value in coords):
            return "All coordinates must be finite"
        if speed < 1 or speed > self.max_speed:
            return f"speed must be between 1 and {self.max_speed}"
        x_mm, y_mm, z_mm, rx, ry, rz = coords
        if not self.coord_min_x <= x_mm <= self.coord_max_x:
            return (
                f"X {x_mm:.1f} mm is outside "
                f"[{self.coord_min_x:.1f}, {self.coord_max_x:.1f}]"
            )
        if not self.coord_min_y <= y_mm <= self.coord_max_y:
            return (
                f"Y {y_mm:.1f} mm is outside "
                f"[{self.coord_min_y:.1f}, {self.coord_max_y:.1f}]"
            )
        if not self.coord_min_z <= z_mm <= self.coord_max_z:
            return (
                f"Z {z_mm:.1f} mm is outside "
                f"[{self.coord_min_z:.1f}, {self.coord_max_z:.1f}]"
            )
        rx_error = min(abs(rx - 180.0), abs(rx + 180.0))
        if (
            rx_error > self.coord_tool_tolerance
            or abs(ry) > self.coord_tool_tolerance
            or not -180.0 <= rz <= 180.0
        ):
            return (
                "Tool orientation must point down: "
                "rx near ±180°, ry near 0°, rz within ±180°"
            )
        return ""

    def _usable_coords(self, coords):
        """Return six finite coordinates, or None when the read failed."""
        if not isinstance(coords, list) or len(coords) != 6:
            return None
        values = []
        for value in coords:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number):
                return None
            values.append(number)
        return values

    def _coords_arrived(self, current, target):
        """Return true when the tool is within the coordinate tolerance."""
        for index in range(3):
            if abs(current[index] - target[index]) > self.coord_tolerance_mm:
                return False
        for index in range(3, 6):
            delta = abs(current[index] - target[index]) % 360.0
            delta = min(delta, 360.0 - delta)
            if delta > self.coord_angle_tolerance:
                return False
        return True

    def _stream_update_is_current(self, execute):
        """Return true when a live stream can skip a full status read."""
        if not execute or not self.replace_motion_enabled:
            return False
        if self._commanded_degrees is None or self.stop_requested.is_set():
            return False
        # Stay on the fast path for the whole trajectory. A status read
        # in the middle makes the arm brake and start again.
        return time.monotonic() - self._stream_sent < 0.5

    def _target_rejection(self, reference, targets):
        """Return an error message when targets leave the safe step."""
        for joint_id, (start, target) in enumerate(
            zip(reference, targets),
            start=1,
        ):
            lower, upper = self.JOINT_LIMITS_DEGREES[joint_id]
            safe_lower = lower + self.joint_limit_margin
            safe_upper = upper - self.joint_limit_margin
            if target < safe_lower or target > safe_upper:
                return (
                    f"J{joint_id} target {target:.2f}° is outside "
                    f"safe range [{safe_lower:.2f}°, {safe_upper:.2f}°]"
                )
            movement = abs(target - start)
            if movement > self.max_joint_step:
                return (
                    f"J{joint_id} movement {movement:.2f}° exceeds "
                    f"the {self.max_joint_step:.2f}° limit"
                )
        return ""

    def _set_fresh_mode_locked(self, enabled):
        """Explicitly set and verify the controller command mode."""
        if enabled == self._fresh_mode_on:
            return

        desired = 1 if enabled else 0
        self._fresh_mode_on = None
        self.robot.set_fresh_mode(desired)
        actual = self.robot.get_fresh_mode()

        if actual != desired:
            raise RuntimeError(
                f"Command mode verification failed: "
                f"requested {desired}, received {actual}"
            )

        self._fresh_mode_on = bool(enabled)
        self.get_logger().info(
            f"Controller command mode verified: {actual}"
        )

    def _usable_joint_degrees(self, angles):
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

    def _read_live_angles(self):
        """Return the six measured joint angles, or None if unusable."""
        with self.serial_mutex:
            angles = self.robot.get_angles()
        return self._usable_joint_degrees(angles)

    def _stream_angles(self, targets, speed, response):
        """Send the next dance target without waiting for a stop."""
        if self.stop_requested.is_set():
            response.message = (
                "Multi-joint movement stopped through /mycobot/stop"
            )
            return response

        rejection = self._target_rejection(
            self._commanded_degrees,
            targets,
        )
        if rejection:
            # The arm can be far from the last command. Accept the step
            # when it is small relative to the measured pose.
            live = self._read_live_angles()
            if (
                live is None
                or self._target_rejection(live, targets)
            ):
                response.message = rejection
                return response
            self.get_logger().warning(
                "Stream step measured from the live pose"
            )

        try:
            with self.serial_mutex:
                if self.stop_requested.is_set() or not self.multi_joint_motion_enabled:
                    response.message = "Stop latched or multi-joint motion disabled"
                    return response
                self._set_fresh_mode_locked(True)
                if self.stop_requested.is_set():
                    response.message = "Stop latched; no command sent"
                    return response
                self.robot.send_angles(
                    targets,
                    speed,
                    _async=True,
                )
        except Exception as error:
            self._commanded_degrees = None
            response.message = f"Multi-joint movement failed: {error}"
            return response

        self._commanded_degrees = list(targets)
        self._stream_sent = time.monotonic()
        response.success = True
        response.start_degrees = list(targets)
        response.final_degrees = list(targets)
        response.message = "Six-joint target sent"
        return response

    def handle_move_joints(self, request, response):
        """Validate a six-joint pose command without executing it yet."""
        response.success = False
        response.message = ""
        response.start_degrees = [math.nan] * 6
        response.final_degrees = [math.nan] * 6

        targets = [float(value) for value in request.target_degrees]
        speed = int(request.speed)

        if len(targets) != 6:
            response.message = "Exactly six joint targets are required"
            return response

        if not all(math.isfinite(value) for value in targets):
            response.message = "All joint targets must be finite"
            return response

        if speed < 1 or speed > self.max_speed:
            response.message = (
                f"speed must be between 1 and {self.max_speed}"
            )
            return response

        # Enforce the motion gate before the legacy streaming fast path.
        if request.execute and not self.multi_joint_motion_enabled:
            response.message = "multi_joint_motion_enabled is false"
            return response
        if request.execute and self.stop_requested.is_set():
            response.message = "Stop is latched; no command sent"
            return response

        if self._stream_update_is_current(request.execute):
            return self._stream_angles(targets, speed, response)

        try:
            with self.serial_mutex:
                connected = self.robot.is_controller_connected()
                power_state = self.robot.is_power_on()
                error_state = self.robot.get_error_information()
                moving = self.robot.is_moving()
                angles = self.robot.get_angles()

            if connected != 1:
                response.message = "Robot controller is not connected"
                return response

            if power_state != 1:
                response.message = "Robot is not powered on"
                return response

            if error_state != 0:
                response.message = (
                    f"Robot reports error code {error_state}"
                )
                return response

            streaming = (
                self.replace_motion_enabled
                and bool(request.execute)
            )

            if moving != 0 and not streaming:
                response.message = "Robot is already moving"
                return response

            if streaming and self.stop_requested.is_set():
                response.message = (
                    "Multi-joint movement stopped through "
                    "/mycobot/stop"
                )
                return response

            if not isinstance(angles, list) or len(angles) != 6:
                response.message = f"Invalid joint feedback: {angles}"
                return response

            starts = [float(value) for value in angles]

            if not all(math.isfinite(value) for value in starts):
                response.message = "Current joint feedback is invalid"
                return response

            response.start_degrees = starts
            response.final_degrees = starts

            reference = starts
            if (
                streaming
                and self._commanded_degrees is not None
                and self._target_rejection(
                    self._commanded_degrees,
                    targets,
                ) == ""
            ):
                reference = self._commanded_degrees

            movements = []

            for joint_id, (start, target) in enumerate(
                zip(reference, targets),
                start=1,
            ):
                lower, upper = self.JOINT_LIMITS_DEGREES[joint_id]
                safe_lower = lower + self.joint_limit_margin
                safe_upper = upper - self.joint_limit_margin

                if target < safe_lower or target > safe_upper:
                    response.message = (
                        f"J{joint_id} target {target:.2f}° is outside "
                        f"safe range [{safe_lower:.2f}°, "
                        f"{safe_upper:.2f}°]"
                    )
                    return response

                movement = abs(target - start)
                movements.append(movement)

                if movement > self.max_joint_step:
                    response.message = (
                        f"J{joint_id} movement {movement:.2f}° exceeds "
                        f"the {self.max_joint_step:.2f}° limit"
                    )
                    return response

            maximum_movement = max(movements)

            if not request.execute:
                response.success = True
                response.message = (
                    "DRY RUN accepted: six-joint pose, "
                    f"maximum movement {maximum_movement:.2f}°, "
                    f"speed {speed}"
                )
                return response

            if not self.multi_joint_motion_enabled:
                response.message = (
                    "Validation passed, but "
                    "multi_joint_motion_enabled is false"
                )
                return response

            if not streaming:
                self.stop_requested.clear()
                self._guarded_motion_active.set()

            target_text = ", ".join(
                f"{value:.2f}" for value in targets
            )

            if streaming:
                self.get_logger().debug(
                    f"Streaming six-joint pose [{target_text}] "
                    f"at speed {speed}"
                )
            else:
                self.get_logger().warning(
                    f"Executing six-joint pose [{target_text}] "
                    f"at speed {speed}"
                )

            with self.serial_mutex:
                if streaming:
                    self._set_fresh_mode_locked(True)
                else:
                    self._set_fresh_mode_locked(False)
                command_reply = self.robot.send_angles(
                    targets,
                    speed,
                    _async=streaming,
                )
                if not streaming:
                    self.get_logger().info(
                        f"Six-joint command reply: {command_reply!r}"
                    )
                    if command_reply is None or command_reply == -1:
                        raise RuntimeError(
                            "Six-joint command acknowledgement failed"
                        )

            if streaming:
                self._stream_check = time.monotonic()
                self._stream_sent = self._stream_check
                self._commanded_degrees = list(targets)
                response.success = True
                response.final_degrees = starts
                response.message = "Six-joint target sent"
                return response

            deadline = time.monotonic() + self.motion_timeout
            final_angles = starts

            while time.monotonic() < deadline:
                if self.stop_requested.is_set():
                    response.final_degrees = final_angles
                    response.message = (
                        "Multi-joint movement stopped through "
                        "/mycobot/stop"
                    )
                    return response

                time.sleep(self.motion_poll_period)

                with self.serial_mutex:
                    updated_angles = self.robot.get_angles()
                    moving = self.robot.is_moving()
                    updated_error = self.robot.get_error_information()

                if updated_error != 0:
                    with self.serial_mutex:
                        self.robot.stop()

                    response.final_degrees = final_angles
                    response.message = (
                        f"Controller reported error {updated_error}; "
                        "stop command sent"
                    )
                    return response

                usable = self._usable_joint_degrees(updated_angles)
                if usable is None:
                    with self.serial_mutex:
                        self.robot.stop()

                    response.message = (
                        f"Invalid joint feedback during movement: "
                        f"{updated_angles}; stop command sent"
                    )
                    return response

                final_angles = usable
                response.final_degrees = final_angles
                self._store_measurement(final_angles)

                # Recheck after controller I/O so a concurrent stop takes
                # priority over declaring successful completion.
                if self.stop_requested.is_set():
                    response.message = (
                        "Multi-joint movement stopped through "
                        "/mycobot/stop"
                    )
                    return response

                errors = [
                    abs(final_value - target)
                    for final_value, target in zip(final_angles, targets)
                ]
                maximum_error = max(errors)

                if (
                    maximum_error <= self.position_tolerance
                    and moving == 0
                ):
                    response.success = True
                    response.message = (
                        "Six-joint movement completed; "
                        f"maximum final error {maximum_error:.2f}°"
                    )
                    return response

            with self.serial_mutex:
                self.robot.stop()

            response.final_degrees = final_angles
            response.message = (
                f"Multi-joint movement timed out after "
                f"{self.motion_timeout:.1f}s; stop command sent"
            )
            return response

        except Exception as error:
            self._commanded_degrees = None
            try:
                with self.serial_mutex:
                    self.robot.stop()
            except Exception:
                pass

            response.message = f"Multi-joint movement failed: {error}"
            return response
        finally:
            self._guarded_motion_active.clear()

    def handle_move_joint(self, request, response):
        """Validate and optionally execute one guarded joint movement."""
        response.success = False
        response.message = ""
        response.start_degrees = math.nan
        response.final_degrees = math.nan

        joint_id = int(request.joint_id)
        target = float(request.target_degrees)
        speed = int(request.speed)

        if joint_id not in self.JOINT_LIMITS_DEGREES:
            response.message = "joint_id must be between 1 and 6"
            return response

        if not math.isfinite(target):
            response.message = "target_degrees must be finite"
            return response

        if speed < 1 or speed > self.max_speed:
            response.message = (
                f"speed must be between 1 and {self.max_speed}"
            )
            return response

        try:
            with self.serial_mutex:
                connected = self.robot.is_controller_connected()
                power_state = self.robot.is_power_on()
                error_state = self.robot.get_error_information()
                moving = self.robot.is_moving()
                angles = self.robot.get_angles()

            if connected != 1:
                response.message = "Robot controller is not connected"
                return response

            if power_state != 1:
                response.message = "Robot is not powered on"
                return response

            if error_state != 0:
                response.message = (
                    f"Robot reports error code {error_state}"
                )
                return response

            if moving != 0:
                response.message = "Robot is already moving"
                return response

            if not isinstance(angles, list) or len(angles) != 6:
                response.message = f"Invalid joint feedback: {angles}"
                return response

            start = float(angles[joint_id - 1])

            if not math.isfinite(start):
                response.message = "Current joint angle is invalid"
                return response

            response.start_degrees = start
            response.final_degrees = start

            lower, upper = self.JOINT_LIMITS_DEGREES[joint_id]
            safe_lower = lower + self.joint_limit_margin
            safe_upper = upper - self.joint_limit_margin

            if target < safe_lower or target > safe_upper:
                response.message = (
                    f"Target {target:.2f}° is outside safe range "
                    f"[{safe_lower:.2f}°, {safe_upper:.2f}°]"
                )
                return response

            movement = abs(target - start)

            if movement > self.max_joint_step:
                response.message = (
                    f"Requested movement {movement:.2f}° exceeds "
                    f"the {self.max_joint_step:.2f}° limit"
                )
                return response

            if not request.execute:
                response.success = True
                response.message = (
                    f"DRY RUN accepted: J{joint_id}, "
                    f"{start:.2f}° -> {target:.2f}°, speed {speed}"
                )
                return response

            if not self.motion_enabled:
                response.message = (
                    "Validation passed, but motion_enabled is false"
                )
                return response

            self.stop_requested.clear()

            self.get_logger().warning(
                f"Executing J{joint_id}: "
                f"{start:.2f}° -> {target:.2f}°, speed {speed}"
            )

            self._guarded_motion_active.set()
            with self.serial_mutex:
                self._set_fresh_mode_locked(False)
                command_reply = self.robot.send_angle(
                    joint_id,
                    target,
                    speed,
                    _async=False,
                )
                self.get_logger().info(
                    f"Single-joint command reply: {command_reply!r}"
                )
                if command_reply is None or command_reply == -1:
                    raise RuntimeError(
                        "Single-joint command acknowledgement failed"
                    )

            deadline = time.monotonic() + self.motion_timeout
            final_angle = start

            while time.monotonic() < deadline:
                if self.stop_requested.is_set():
                    response.final_degrees = final_angle
                    response.message = (
                        "Movement stopped through /mycobot/stop"
                    )
                    return response

                time.sleep(self.motion_poll_period)

                with self.serial_mutex:
                    updated_angles = self.robot.get_angles()
                    moving = self.robot.is_moving()
                    updated_error = self.robot.get_error_information()

                self.get_logger().info(
                    f"J{joint_id} poll: measured={updated_angles}, "
                    f"moving={moving}, controller_error={updated_error}"
                )

                if updated_error != 0:
                    with self.serial_mutex:
                        self.robot.stop()
                    response.final_degrees = final_angle
                    response.message = (
                        f"Controller reported error {updated_error}; "
                        "stop command sent"
                    )
                    return response

                usable = self._usable_joint_degrees(updated_angles)
                if usable is None:
                    with self.serial_mutex:
                        self.robot.stop()
                    response.message = (
                        "Invalid joint feedback during movement: "
                        f"{updated_angles}; stop command sent"
                    )
                    return response

                final_angle = usable[joint_id - 1]
                response.final_degrees = final_angle
                self._store_measurement(usable)

                if self.stop_requested.is_set():
                    response.message = (
                        "Movement stopped through /mycobot/stop"
                    )
                    return response

                error = abs(final_angle - target)
                if error <= self.position_tolerance and moving == 0:
                    response.success = True
                    response.message = (
                        f"Movement completed: J{joint_id}, "
                        f"{start:.2f}° -> {final_angle:.2f}°, "
                        f"error {error:.2f}°"
                    )
                    return response

            with self.serial_mutex:
                self.robot.stop()

            response.message = (
                f"Movement timed out after {self.motion_timeout:.1f}s; "
                f"last angle {final_angle:.2f}°. Stop command sent."
            )
            return response

        except Exception as error:
            try:
                with self.serial_mutex:
                    self.robot.stop()
            except Exception:
                pass

            response.message = f"Movement failed: {error}"
            return response
        finally:
            self._guarded_motion_active.clear()

    def _store_measurement(self, degrees):
        """Record one serial sample and publish it with this read's stamp."""
        self._measured_degrees = list(degrees)
        self._measured_stamp = self.get_clock().now().to_msg()
        self._publish_joint_degrees(
            self._measured_degrees,
            self._measured_stamp,
        )

    def _publish_joint_degrees(self, angles, stamp):
        """Publish measured joint angles. The stamp is the measurement time."""
        joint_names = list(self.JOINT_NAMES)
        joint_positions = [math.radians(float(angle)) for angle in angles]
        gripper_value = self.get_cached_gripper_value()
        if gripper_value is not None:
            joint_names.append(self.GRIPPER_JOINT_NAME)
            joint_positions.append(
                self.gripper_value_to_radians(gripper_value)
            )
        message = JointState()
        message.header.stamp = stamp
        message.name = joint_names
        message.position = joint_positions
        self.joint_publisher.publish(message)

    def _publish_commanded_degrees(self, angles):
        """Publish commanded angles without changing measured feedback."""
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.JOINT_NAMES)
        message.position = [
            math.radians(float(angle)) for angle in angles
        ]
        self.commanded_joint_publisher.publish(message)

    def publish_state(self):
        """Publish arm, gripper and raw tool-pose feedback."""
        if self._guarded_motion_active.is_set():
            return

        commanded = self._commanded_degrees
        stream_is_hot = (
            commanded is not None
            and time.monotonic() - self._stream_sent < 0.5
        )
        if stream_is_hot:
            self._publish_commanded_degrees(commanded)
            if self._measured_degrees is not None:
                self._publish_joint_degrees(
                    self._measured_degrees,
                    self._measured_stamp,
                )
            return

        raw_gripper_value = None

        try:
            with self.serial_mutex:
                angles = self.robot.get_angles()
                coordinates = self.robot.get_coords()

                try:
                    raw_gripper_value = (
                        self.robot.get_gripper_value(
                            gripper_type=1
                        )
                    )
                except Exception:
                    # Gripper feedback is optional. A failed gripper
                    # read must not stop arm-state publication.
                    raw_gripper_value = None

            self.cache_gripper_value(raw_gripper_value)

            usable = self._usable_joint_degrees(angles)
            if usable is None:
                self.get_logger().warning(
                    f"Invalid joint response: {angles}"
                )
            else:
                self._store_measurement(usable)

            if (
                isinstance(coordinates, list)
                and len(coordinates) == 6
            ):
                pose = Float64MultiArray()
                pose.data = [
                    float(value) for value in coordinates
                ]
                self.pose_publisher.publish(pose)

        except Exception as error:
            self.get_logger().error(
                f"Robot read failed: {error}"
            )

    def destroy_node(self):
        ok, message = self._request_arm_stop(shutdown=True)
        if not ok:
            print(message, flush=True)
        # main() calls this only after executor workers have drained.
        serial = getattr(getattr(self, "robot", None), "_serial_port", None)
        if serial is not None:
            serial.close()
        if hasattr(self, "port_lock") and not self.port_lock.closed:
            fcntl.flock(self.port_lock.fileno(), fcntl.LOCK_UN)
            self.port_lock.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MyCobotStatePublisher()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Latch stop BEFORE waiting for the trajectory callback to finish.
        ok, message = node._request_arm_stop(shutdown=True)
        print(message, flush=True)
        drained = executor.shutdown(timeout_sec=2.0)
        if drained:
            node.destroy_node()
        else:
            # A worker may still own serial: do not close it or release its lock.
            print("Workers did not stop; serial ownership retained. "
                  "Use physical power control.", flush=True)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()