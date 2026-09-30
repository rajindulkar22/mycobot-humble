"""Publish provisional table TF, card/cap markers and open RViz."""

from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vision = Path(get_package_share_directory("mycobot_vision"))
    description = Path(get_package_share_directory("mycobot_description"))
    config = yaml.safe_load(
        (vision / "config" / "table_transform.yaml").read_text()
    )
    translation = config["translation"]
    rotation = config["rotation"]
    arguments = []
    for axis in ("x", "y", "z"):
        arguments.extend([f"--{axis}", str(translation[axis])])
    for axis in ("x", "y", "z", "w"):
        arguments.extend([f"--q{axis}", str(rotation[axis])])
    arguments.extend([
        "--frame-id", config["parent_frame"],
        "--child-frame-id", config["child_frame"],
    ])

    return LaunchDescription([
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="table_transform_publisher",
            arguments=arguments,
            output="screen",
        ),
        Node(
            package="mycobot_vision",
            executable="table_preview.py",
            name="table_preview",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="table_preview_rviz",
            arguments=["-d", str(description / "rviz" / "table_preview.rviz")],
            condition=IfCondition(LaunchConfiguration("launch_rviz")),
            output="screen",
        ),
    ])
