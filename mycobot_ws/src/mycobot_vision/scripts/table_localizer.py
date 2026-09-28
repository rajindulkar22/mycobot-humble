#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rclpy
import yaml

from rclpy.node import Node
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)


class TableLocalizer(Node):
    def __init__(self):
        super().__init__("table_localizer")

        self.declare_parameter(
            "homography_file",
            "",
        )
        self.declare_parameter(
            "input_topic",
            "/overview_camera/yolo/detections",
        )
        self.declare_parameter(
            "output_topic",
            "/overview_camera/table_detections",
        )
        self.declare_parameter(
            "table_frame",
            "table_frame",
        )

        homography_file = str(
            self.get_parameter("homography_file").value
        )
        input_topic = str(
            self.get_parameter("input_topic").value
        )
        output_topic = str(
            self.get_parameter("output_topic").value
        )
        self.table_frame = str(
            self.get_parameter("table_frame").value
        )

        if not homography_file:
            raise ValueError(
                "homography_file parameter must not be empty"
            )

        self.homography_file = Path(homography_file)
        self.pixel_to_table = self.load_homography(
            self.homography_file
        )

        self.subscription = self.create_subscription(
            Detection2DArray,
            input_topic,
            self.detections_callback,
            10,
        )

        self.publisher = self.create_publisher(
            Detection3DArray,
            output_topic,
            10,
        )

        self.message_count = 0

        self.get_logger().info(
            f"Loaded homography from {self.homography_file}"
        )
        self.get_logger().info(
            f"Converting {input_topic} to {output_topic}"
        )
        self.get_logger().info(
            f"Output frame: {self.table_frame}"
        )

    @staticmethod
    def load_homography(homography_file):
        if not homography_file.is_file():
            raise FileNotFoundError(
                f"Homography file does not exist: "
                f"{homography_file}"
            )

        with homography_file.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            calibration = yaml.safe_load(file_handle)

        if not isinstance(calibration, dict):
            raise ValueError(
                "Homography file must contain a YAML mapping"
            )

        matrix_section = calibration.get(
            "pixel_to_table_homography"
        )

        if not isinstance(matrix_section, dict):
            raise ValueError(
                "pixel_to_table_homography section is missing"
            )

        rows = int(matrix_section.get("rows", 0))
        columns = int(matrix_section.get("cols", 0))
        matrix_data = matrix_section.get("data")

        if rows != 3 or columns != 3:
            raise ValueError(
                "Pixel-to-table homography must be 3x3"
            )

        if (
            not isinstance(matrix_data, list)
            or len(matrix_data) != 9
        ):
            raise ValueError(
                "Homography data must contain nine values"
            )

        matrix = np.array(
            matrix_data,
            dtype=np.float64,
        ).reshape(3, 3)

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Homography contains non-finite values"
            )

        if abs(np.linalg.det(matrix)) < 1.0e-15:
            raise ValueError(
                "Homography matrix is singular"
            )

        return matrix

    def pixel_to_table_point(self, pixel_x, pixel_y):
        homogeneous_pixel = np.array(
            [
                float(pixel_x),
                float(pixel_y),
                1.0,
            ],
            dtype=np.float64,
        )

        homogeneous_table = (
            self.pixel_to_table @ homogeneous_pixel
        )

        denominator = float(homogeneous_table[2])

        if abs(denominator) < 1.0e-12:
            raise ValueError(
                "Homography produced a point at infinity"
            )

        table_x = float(
            homogeneous_table[0] / denominator
        )
        table_y = float(
            homogeneous_table[1] / denominator
        )

        return table_x, table_y

    def create_3d_detection(
        self,
        detection_2d,
        table_x,
        table_y,
        header,
    ):
        detection_3d = Detection3D()
        detection_3d.header = header
        detection_3d.id = detection_2d.id

        detection_3d.bbox.center.position.x = table_x
        detection_3d.bbox.center.position.y = table_y
        detection_3d.bbox.center.position.z = 0.0
        detection_3d.bbox.center.orientation.w = 1.0

        # Physical object dimensions are intentionally left at zero.
        # They will be configured after the real cubes are available.
        detection_3d.bbox.size.x = 0.0
        detection_3d.bbox.size.y = 0.0
        detection_3d.bbox.size.z = 0.0

        for result_2d in detection_2d.results:
            result_3d = ObjectHypothesisWithPose()

            result_3d.hypothesis.class_id = (
                result_2d.hypothesis.class_id
            )
            result_3d.hypothesis.score = (
                result_2d.hypothesis.score
            )

            result_3d.pose.pose.position.x = table_x
            result_3d.pose.pose.position.y = table_y
            result_3d.pose.pose.position.z = 0.0
            result_3d.pose.pose.orientation.w = 1.0

            detection_3d.results.append(result_3d)

        return detection_3d

    def detections_callback(self, message):
        output_message = Detection3DArray()
        output_message.header = message.header
        output_message.header.frame_id = self.table_frame

        summaries = []

        for detection_2d in message.detections:
            try:
                pixel_x = (
                    detection_2d.bbox.center.position.x
                )
                pixel_y = (
                    detection_2d.bbox.center.position.y
                )

                table_x, table_y = (
                    self.pixel_to_table_point(
                        pixel_x,
                        pixel_y,
                    )
                )

                detection_3d = self.create_3d_detection(
                    detection_2d,
                    table_x,
                    table_y,
                    output_message.header,
                )
                output_message.detections.append(
                    detection_3d
                )

                if detection_2d.results:
                    class_name = (
                        detection_2d
                        .results[0]
                        .hypothesis
                        .class_id
                    )
                else:
                    class_name = "unknown"

                summaries.append(
                    f"{class_name}: "
                    f"x={table_x:.4f} m, "
                    f"y={table_y:.4f} m"
                )

            except Exception as exception:
                self.get_logger().error(
                    "Unable to localize detection "
                    f"{detection_2d.id}: {exception}"
                )

        self.publisher.publish(output_message)

        self.message_count += 1

        if summaries and self.message_count % 20 == 0:
            self.get_logger().info(
                "Table detections: "
                + "; ".join(summaries)
            )


def main(args=None):
    rclpy.init(args=args)
    node = TableLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()