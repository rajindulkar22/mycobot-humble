import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("mycobot_hardware"),
        "config",
        "hardware.yaml",
    )

    description_file = os.path.join(
        get_package_share_directory("mycobot_description"),
        "urdf",
        "mycobot_280_jn",
        "mycobot_280_jn_adaptive_gripper.urdf",
    )

    with open(description_file, encoding="utf-8") as description_stream:
        robot_description = description_stream.read()

    return LaunchDescription(
        [
            Node(
                package="mycobot_hardware",
                executable="state_publisher",
                name="mycobot_state_publisher",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
        ]
    )
