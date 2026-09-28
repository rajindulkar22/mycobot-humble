import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory(
        "mycobot_vision"
    )

    detector_config = os.path.join(
        package_share,
        "config",
        "yolo_detector.yaml",
    )
    localizer_config = os.path.join(
        package_share,
        "config",
        "table_localizer.yaml",
    )
    homography_config = os.path.join(
        package_share,
        "config",
        "table_homography.yaml",
    )

    rectify_node = Node(
        package="image_proc",
        executable="rectify_node",
        namespace="overview_camera",
        name="rectify",
        output="screen",
        remappings=[
            ("image", "image_raw"),
            ("camera_info", "camera_info"),
            ("image_rect", "image_rect"),
        ],
    )

    detector_node = Node(
        package="mycobot_vision",
        executable="yolo_detector.py",
        name="yolo_detector",
        output="screen",
        parameters=[detector_config],
    )

    localizer_node = Node(
        package="mycobot_vision",
        executable="table_localizer.py",
        name="table_localizer",
        output="screen",
        parameters=[
            localizer_config,
            {
                "homography_file": homography_config,
            },
        ],
    )

    return LaunchDescription(
        [
            rectify_node,
            detector_node,
            localizer_node,
        ]
    )