"""Launch the joint offset plot without opening the serial port."""

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """Start the commanded-versus-measured page."""
    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    "Joint offset plot does not move the arm. "
                    "Open http://<this-computer>:8091"
                )
            ),
            Node(
                package="mycobot_hardware",
                executable="joint_plot",
                name="mycobot_joint_plot",
                output="screen",
                parameters=[{"ui_port": 8091}],
            ),
        ]
    )
