"""Launch one-joint diagnostics and the measured-feedback page."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """Start guarded single-joint motion and the diagnostic page."""
    hardware_share = get_package_share_directory("mycobot_hardware")
    description_share = get_package_share_directory("mycobot_description")
    hardware_config = os.path.join(hardware_share, "config", "hardware.yaml")
    urdf_file = os.path.join(
        description_share,
        "urdf",
        "mycobot_280_jn",
        "mycobot_280_jn_adaptive_gripper.urdf",
    )
    with open(urdf_file, "r", encoding="utf-8") as file:
        robot_description = file.read()

    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    "Joint diagnostic checks every joint by 2 degrees at speed 5. "
                    "A failed joint is recorded and the next joint is still checked. "
                    "Initial position walks all six joints to 0 degrees in 2 degree steps "
                    "and stops that walk if a step fails. Stop any other "
                    "mycobot_state_publisher first. "
                    "Open http://<this-computer>:8090"
                )
            ),
            Node(
                package="mycobot_hardware",
                executable="state_publisher",
                name="mycobot_state_publisher",
                output="screen",
                parameters=[
                    hardware_config,
                    {
                        "motion_enabled": True,
                        "multi_joint_motion_enabled": False,
                        "replace_motion_enabled": False,
                        "max_joint_step_degrees": 2.0,
                        "max_speed": 5,
                        "position_tolerance_degrees": 0.5,
                        "motion_timeout_seconds": 10.0,
                        "gripper_motion_enabled": False,
                    },
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="mycobot_hardware",
                executable="demo_ui",
                name="mycobot_joint_diagnostic_ui",
                output="screen",
                additional_env={"MYCOBOT_UI_MODE": "diagnostic"},
                parameters=[{"ui_port": 8090}],
            ),
        ]
    )
