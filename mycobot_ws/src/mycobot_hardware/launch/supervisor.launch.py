"""Launch the mode supervisor without opening the serial port."""

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """Start the page that starts and stops the other myCobot launches."""
    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    "Supervisor does not move the arm by itself. "
                    "Open http://<this-computer>:8093. "
                    "Start only one mode that opens the serial port. "
                    "Stopping a mode is a software stop, not the power switch."
                )
            ),
            Node(
                package="mycobot_hardware",
                executable="supervisor_ui",
                name="mycobot_supervisor",
                output="screen",
                parameters=[{"ui_port": 8093}],
            ),
        ]
    )
