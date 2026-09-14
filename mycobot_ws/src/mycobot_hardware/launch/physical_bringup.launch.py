"""Launch physical feedback and the myCobot robot model."""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    hardware_share = get_package_share_directory(
        "mycobot_hardware"
    )
    description_share = get_package_share_directory(
        "mycobot_description"
    )

    hardware_config = os.path.join(
        hardware_share,
        "config",
        "hardware.yaml",
    )
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
            Node(
                package="mycobot_hardware",
                executable="state_publisher",
                name="mycobot_state_publisher",
                output="screen",
                parameters=[
                    hardware_config,
                    {
                        "motion_enabled": False,
                        "multi_joint_motion_enabled": False,
                        "gripper_motion_enabled": False,
                    },
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": robot_description,
                    }
                ],
            ),
        ]
    )
