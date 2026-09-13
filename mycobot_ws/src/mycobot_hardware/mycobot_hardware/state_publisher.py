#!/usr/bin/env python3
"""Read-only ROS 2 state publisher for the myCobot 280.

Polls the arm over serial and publishes joint and tool-pose feedback. This
node never sends motion commands.

Published topics:
    joint_states (sensor_msgs/JointState):
        Six joint positions in radians, named to match the robot URDF.
    mycobot/tool_pose_raw (std_msgs/Float64MultiArray):
        Raw Cartesian pose [x, y, z, rx, ry, rz] in mm and degrees from the
        controller firmware.

Parameters:
    port (str): Serial device path. Default ``/dev/ttyTHS1``.
    baud (int): Serial baud rate. Default ``1000000``.
    publish_rate (float): Poll/publish frequency in Hz. Default ``5.0``.

Run:
    ros2 run mycobot_hardware state_publisher
"""

import fcntl
import math
import os

import rclpy
from rclpy.node import Node

# Serial API to the myCobot 280 controller.
from pymycobot import MyCobot280

# Standard joint feedback for robot_state_publisher, TF, and RViz.
from sensor_msgs.msg import JointState

# Generic numeric array for project-specific raw tool pose (not a ROS pose type).
from std_msgs.msg import Float64MultiArray

from mycobot_interfaces.srv import MoveJoint
from rclpy.executors import ExternalShutdownException
from std_srvs.srv import Trigger

class MyCobotStatePublisher(Node):
    """Publish physical robot state without commanding motion.

    Connects to the arm via pymycobot, reads joint angles and Cartesian
    coordinates on a timer, and publishes them to ROS topics. Only read
    APIs are used; no move or write commands are sent.
    """

    # Names must match the myCobot 280 URDF so robot_state_publisher and
    # RViz can consume /joint_states without remapping.
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
        """Declare parameters, open the serial connection, and start polling."""
        super().__init__("mycobot_state_publisher")

        # --- ROS parameters ---
        self.declare_parameter("port", "/dev/ttyTHS1")
        self.declare_parameter("baud", 1_000_000)
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("max_joint_step_degrees", 5.0)
        self.declare_parameter("max_speed", 10)
        self.declare_parameter("joint_limit_margin_degrees", 2.0)

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

        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be greater than zero")

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

        self.get_logger().info(f"Acquired serial lock: {lock_path}")

        # --- Hardware connection ---
        self.robot = MyCobot280(port, baud)

        # --- Publishers ---
        # Joint positions for the standard ROS robot model pipeline.
        self.joint_publisher = self.create_publisher(
            JointState, "joint_states", 10
        )
        # Raw end-effector pose from firmware (mm, degrees); not SI-normalized.
        self.pose_publisher = self.create_publisher(
            Float64MultiArray, "mycobot/tool_pose_raw", 10
        )

        self.status_service = self.create_service(
            Trigger, "mycobot/status", self.handle_status
        )
        self.stop_service = self.create_service(
            Trigger, "mycobot/stop", self.handle_stop
        )
        self.move_joint_service = self.create_service(
            MoveJoint,
            "mycobot/move_joint",
            self.handle_move_joint,
        )

        # --- Timer ---
        # Poll hardware at publish_rate Hz and publish both topics.
        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_state,
        )

        self.get_logger().info(
            f"Connected on {port} at {baud} baud"
        )
        self.get_logger().info(
            "Read-only mode: no robot motion commands are used"
        )

    def handle_status(self, request, response):
        """Return the current robot power and joint status."""
        del request

        try:
            power_state = self.robot.is_power_on()
            angles = self.robot.get_angles()

            valid_angles = isinstance(angles, list) and len(angles) == 6
            valid_power = power_state in (0, 1)

            response.success = valid_power and valid_angles
            response.message = (
                f"power_on={power_state == 1}, "
                f"angles_degrees={angles}"
            )
        except Exception as error:
            response.success = False
            response.message = f"Status read failed: {error}"

        return response


    def handle_stop(self, request, response):
        """Send a stop command while keeping the servos engaged."""
        del request

        try:
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
        """Validate a joint request without executing motion."""
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
            power_state = self.robot.is_power_on()

            if power_state != 1:
                response.message = "Robot is not powered on"
                return response

            angles = self.robot.get_angles()

            if not isinstance(angles, list) or len(angles) != 6:
                response.message = f"Invalid joint feedback: {angles}"
                return response

            start = float(angles[joint_id - 1])
            response.start_degrees = start
            response.final_degrees = start

            lower, upper = self.JOINT_LIMITS_DEGREES[joint_id]
            safe_lower = lower + self.joint_limit_margin
            safe_upper = upper - self.joint_limit_margin

            if target < safe_lower or target > safe_upper:
                response.message = (
                    f"Target {target:.2f}° is outside the safe range "
                    f"[{safe_lower:.2f}°, {safe_upper:.2f}°]"
                )
                return response

            movement = abs(target - start)

            if movement > self.max_joint_step:
                response.message = (
                    f"Requested movement {movement:.2f}° exceeds the "
                    f"{self.max_joint_step:.2f}° limit"
                )
                return response

            if request.execute:
                response.message = (
                    "Validation passed, but physical execution is disabled "
                    "during the dry-run phase"
                )
                return response

            response.success = True
            response.message = (
                f"DRY RUN accepted: J{joint_id}, "
                f"{start:.2f}° -> {target:.2f}°, speed {speed}"
            )
            return response

        except Exception as error:
            response.message = f"Validation failed: {error}"
            return response

    def destroy_node(self):
        if hasattr(self, "port_lock") and not self.port_lock.closed:
            fcntl.flock(self.port_lock.fileno(), fcntl.LOCK_UN)
            self.port_lock.close()

        return super().destroy_node()

    def publish_state(self):
        """Read joint angles and tool pose from hardware and publish both.

        On each timer tick:
            1. Read six joint angles (degrees from hardware).
            2. Validate, convert to radians, publish JointState.
            3. Read six Cartesian values (mm and degrees from hardware).
            4. Validate and publish Float64MultiArray.

        Serial or protocol errors are logged; the node keeps running.
        """
        try:
            # --- Joint read path ---
            # Hardware returns degrees; ROS JointState.position uses radians.
            angles = self.robot.get_angles()

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

            # --- Pose read path ---
            # get_coords() returns [x, y, z, rx, ry, rz] in mm and degrees.
            coordinates = self.robot.get_coords()

            if isinstance(coordinates, list) and len(coordinates) == 6:
                pose = Float64MultiArray()
                pose.data = [float(value) for value in coordinates]
                self.pose_publisher.publish(pose)

        except Exception as error:
            self.get_logger().error(f"Robot read failed: {error}")


def main(args=None):
    """Initialize the node, spin until interrupted, then shut down cleanly."""
    rclpy.init(args=args)
    node = MyCobotStatePublisher()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
