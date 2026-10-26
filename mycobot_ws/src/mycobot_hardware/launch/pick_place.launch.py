"""Launch a guarded table pick-and-place with the overview camera."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Start the arm gates, detection, and the pick page."""
    hardware_share = get_package_share_directory("mycobot_hardware")
    vision_share = get_package_share_directory("mycobot_vision")
    hardware_config = os.path.join(
        hardware_share,
        "config",
        "hardware.yaml",
    )
    calibration_file = os.path.join(
        vision_share,
        "config",
        "table_calibration.yaml",
    )
    homography_file = os.path.join(
        vision_share,
        "config",
        "table_homography.yaml",
    )
    transform_file = os.path.join(
        vision_share,
        "config",
        "table_transform.yaml",
    )
    enable_vision = LaunchConfiguration("enable_vision")
    use_wrist_guidance = LaunchConfiguration("use_wrist_guidance")
    note_reader_url = LaunchConfiguration("note_reader_url")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_vision",
                default_value="true",
                description=(
                    "Start the overview camera, optional wrist camera, "
                    "and the trained detector."
                ),
            ),
            DeclareLaunchArgument(
                "note_reader_url",
                default_value="http://192.168.178.187:11434",
                description=(
                    "Laptop address for the sticky-note reader. "
                    "Empty uses the local letter reader."
                ),
            ),
            DeclareLaunchArgument(
                "use_wrist_guidance",
                default_value="true",
                description=(
                    "When true, the wrist camera nudges during the "
                    "vertical approach. Set false to use overview "
                    "XY only."
                ),
            ),
            LogInfo(
                msg=(
                    "Pick and place moves the real arm. "
                    "Stop any other mycobot_state_publisher first. "
                    "Use enable_vision:=false if the cameras are "
                    "already running. "
                    "Open http://<this-computer>:8092. "
                    "Overview finds the cap. "
                    "Wrist guidance nudges during the descent. "
                    "Set use_wrist_guidance:=false for overview XY only."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        vision_share,
                        "launch",
                        "cameras.launch.py",
                    )
                ),
                condition=IfCondition(enable_vision),
                launch_arguments={
                    "enable_overview": "true",
                    "enable_wrist": use_wrist_guidance,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        vision_share,
                        "launch",
                        "yolo_detection.launch.py",
                    )
                ),
                condition=IfCondition(enable_vision),
            ),
            Node(
                package="mycobot_hardware",
                executable="state_publisher",
                name="mycobot_state_publisher",
                output="screen",
                parameters=[
                    hardware_config,
                    {
                        "coord_motion_enabled": True,
                        "gripper_motion_enabled": True,
                        "multi_joint_motion_enabled": True,
                        "replace_motion_enabled": True,
                        "max_joint_step_degrees": 10.0,
                        "max_speed": 30,
                        "motion_timeout_seconds": 20.0,
                        "coord_max_x_mm": 325.0,
                        "coord_tolerance_mm": 3.0,
                        "gripper_timeout_seconds": 34.0,
                    },
                ],
            ),
            Node(
                package="mycobot_vision",
                executable="pick_place.py",
                name="mycobot_pick_place",
                output="screen",
                parameters=[
                    {
                        "ui_port": 8092,
                        "calibration_file": calibration_file,
                        "homography_file": homography_file,
                        "transform_file": transform_file,
                        "target_class": "red_cap",
                        "use_wrist_guidance": ParameterValue(
                            use_wrist_guidance,
                            value_type=bool,
                        ),
                        "note_reader_url": note_reader_url,
                    }
                ],
            ),
        ]
    )
