"""Launch MoveIt in planning-only mode using live physical feedback."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    package_name = "mycobot_280jn_physical_moveit_config"

    moveit_config = (
        MoveItConfigsBuilder(
            "mycobot_280_jn_adaptive_gripper",
            package_name=package_name,
        )
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl"],
        )
        .trajectory_execution(
            file_path="config/planning_only_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
    )

    rviz_config = os.path.join(
        get_package_share_directory(package_name),
        "config",
        "moveit.rviz",
    )

    planning_only_parameters = {
        "allow_trajectory_execution": False,
        "moveit_manage_controllers": False,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            planning_only_parameters,
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="moveit_rviz",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([move_group, rviz])
