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

    return LaunchDescription(
        [
            Node(
                package="mycobot_hardware",
                executable="state_publisher",
                name="mycobot_state_publisher",
                output="screen",
                parameters=[config_file],
            )
        ]
    )