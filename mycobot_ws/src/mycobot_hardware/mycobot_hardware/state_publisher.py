#!/usr/bin/env python3
"""Safe ROS 2 Humble hardware node for the myCobot 280."""

import fcntl
import math
import os
import threading
import time

import rclpy
from mycobot_interfaces.srv import MoveJoint
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
        self.declare_parameter("motion_timeout_seconds", 6.0)
        self.declare_parameter("position_tolerance_degrees", 1.5)
        self.declare_parameter("motion_poll_period_seconds", 0.2)

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
        self.motion_timeout = float(
            self.get_parameter("motion_timeout_seconds").value
        )
        self.position_tolerance = float(
            self.get_parameter("position_tolerance_degrees").value
        )
        self.motion_poll_period = float(
            self.get_parameter("motion_poll_period_seconds").value
        )

        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be greater than zero")

        if self.motion_timeout <= 0.0:
            raise ValueError("motion_timeout_seconds must be positive")

        if self.position_tolerance <= 0.0:
            raise ValueError("position_tolerance_degrees must be positive")

        if self.motion_poll_period <= 0.0:
            raise ValueError("motion_poll_period_seconds must be positive")

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

        self.io_callback_group = ReentrantCallbackGroup()
        self.motion_callback_group = MutuallyExclusiveCallbackGroup()

        self.get_logger().info(f"Acquired serial lock: {lock_path}")

        self.robot = MyCobot280(port, baud)

        self.joint_publisher = self.create_publisher(
            JointState,
            "joint_states",
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
        self.move_joint_service = self.create_service(
            MoveJoint,
            "mycobot/move_joint",
            self.handle_move_joint,
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

    def handle_stop(self, request, response):
        """Request cancellation and send the controller stop command."""
        del request
        self.stop_requested.set()

        try:
            with self.serial_mutex:
                self.robot.stop()

            response.success = True
            response.message = "Stop command sent to the robot"
            self.get_logger().warning(response.message)

        except Exception as error:
            response.success = False
            response.message = f"Stop command failed: {error}"
            self.get_logger().error(response.message)

        return response

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

            with self.serial_mutex:
                self.robot.send_angle(
                    joint_id,
                    target,
                    speed,
                    _async=True,
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

                if (
                    isinstance(updated_angles, list)
                    and len(updated_angles) == 6
                ):
                    final_angle = float(updated_angles[joint_id - 1])
                    response.final_degrees = final_angle

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

    def publish_state(self):
        """Read and publish joint angles and raw tool pose."""
        try:
            with self.serial_mutex:
                angles = self.robot.get_angles()
                coordinates = self.robot.get_coords()

            if isinstance(angles, list) and len(angles) == 6:
                message = JointState()
                message.header.stamp = self.get_clock().now().to_msg()
                message.name = self.JOINT_NAMES
                message.position = [
                    math.radians(angle) for angle in angles
                ]
                self.joint_publisher.publish(message)
            else:
                self.get_logger().warning(
                    f"Invalid joint response: {angles}"
                )

            if isinstance(coordinates, list) and len(coordinates) == 6:
                pose = Float64MultiArray()
                pose.data = [float(value) for value in coordinates]
                self.pose_publisher.publish(pose)

        except Exception as error:
            self.get_logger().error(f"Robot read failed: {error}")

    def destroy_node(self):
        self.stop_requested.set()

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
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()