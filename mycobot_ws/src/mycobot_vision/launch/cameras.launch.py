"""Launch the physical overview and wrist cameras."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("mycobot_vision")

    overview_config = os.path.join(
        package_share,
        "config",
        "overview_camera.yaml",
    )
    wrist_config = os.path.join(
        package_share,
        "config",
        "wrist_camera.yaml",
    )

    enable_overview = LaunchConfiguration("enable_overview")
    enable_wrist = LaunchConfiguration("enable_wrist")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_overview",
                default_value="true",
                description="Start the fixed HP overview camera.",
            ),
            DeclareLaunchArgument(
                "enable_wrist",
                default_value="false",
                description="Start the wrist-mounted myCobot camera.",
            ),
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                namespace="overview_camera",
                name="camera",
                output="screen",
                parameters=[overview_config],
                condition=IfCondition(enable_overview),
            ),
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                namespace="wrist_camera",
                name="camera",
                output="screen",
                parameters=[wrist_config],
                condition=IfCondition(enable_wrist),
            ),
        ]
    )
