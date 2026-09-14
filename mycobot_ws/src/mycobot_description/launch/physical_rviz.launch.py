"""Display the remotely published physical myCobot state in RViz."""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    description_share = get_package_share_directory(
        "mycobot_description"
    )
    rviz_config = os.path.join(
        description_share,
        "rviz",
        "physical_mycobot.rviz",
    )

    return LaunchDescription(
        [
            Node(
                package="rviz2",
                executable="rviz2",
                name="physical_mycobot_rviz",
                output="screen",
                arguments=["-d", rviz_config],
            )
        ]
    )