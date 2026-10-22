#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


class TableHomographyCalibrator(Node):
    CORNER_NAMES = (
        "top-left",
        "top-right",
        "bottom-right",
        "bottom-left",
    )

    def __init__(self):
        super().__init__("table_homography_calibrator")

        self.declare_parameter(
            "input_topic",
            "/overview_camera/image_rect",
        )
        self.declare_parameter(
            "output_file",
            "/tmp/table_homography.yaml",
        )
        self.declare_parameter("reference_width_m", 0.08560)
        self.declare_parameter("reference_height_m", 0.05398)

        self.input_topic = str(
            self.get_parameter("input_topic").value
        )
        self.output_file = Path(
            str(self.get_parameter("output_file").value)
        )
        self.reference_width_m = float(
            self.get_parameter("reference_width_m").value
        )
        self.reference_height_m = float(
            self.get_parameter("reference_height_m").value
        )

        if self.reference_width_m <= 0.0:
            raise ValueError(
                "reference_width_m must be positive"
            )

        if self.reference_height_m <= 0.0:
            raise ValueError(
                "reference_height_m must be positive"
            )

        self.bridge = CvBridge()
        self.latest_image = None
        self.frozen_image = None
        self.clicked_points = []
        self.saved = False

        self.window_name = "Table Homography Calibration"
        self.window_initialized = False

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            image_qos,
        )

        self.display_timer = self.create_timer(
            0.03,
            self.update_display,
        )

        self.get_logger().info(
            f"Waiting for rectified images on {self.input_topic}"
        )
        self.get_logger().info(
            "Controls: SPACE=freeze/resume, "
            "left-click=corners, R=reset, S=save, Q=quit"
        )

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
            self.latest_image = image.copy()
        except Exception as exception:
            self.get_logger().error(
                f"Unable to convert image: {exception}"
            )

    def initialize_window(self):
        if self.window_initialized:
            return

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_AUTOSIZE,
        )
        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback,
        )
        self.window_initialized = True

    def mouse_callback(
        self,
        event,
        x_coordinate,
        y_coordinate,
        flags,
        parameter,
    ):
        del flags
        del parameter

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.frozen_image is None:
            self.get_logger().warning(
                "Press SPACE to freeze the image before "
                "selecting corners"
            )
            return

        if len(self.clicked_points) >= 4:
            self.get_logger().warning(
                "Four corners are already selected. "
                "Press R to reset or S to save."
            )
            return

        point = (
            float(x_coordinate),
            float(y_coordinate),
        )
        self.clicked_points.append(point)
        self.saved = False

        corner_index = len(self.clicked_points) - 1
        corner_name = self.CORNER_NAMES[corner_index]

        self.get_logger().info(
            f"Selected {corner_name}: "
            f"({point[0]:.1f}, {point[1]:.1f})"
        )

        if len(self.clicked_points) < 4:
            next_corner = self.CORNER_NAMES[
                len(self.clicked_points)
            ]
            self.get_logger().info(
                f"Next corner: {next_corner}"
            )
        else:
            self.get_logger().info(
                "All corners selected. Press S to save."
            )

    def reset_selection(self):
        self.clicked_points.clear()
        self.saved = False

        self.get_logger().info(
            "Corner selection reset. "
            "Next corner: top-left"
        )

    def draw_interface(self, image):
        display_image = image.copy()

        for point_index, point in enumerate(
            self.clicked_points
        ):
            x_coordinate = int(round(point[0]))
            y_coordinate = int(round(point[1]))

            cv2.circle(
                display_image,
                (x_coordinate, y_coordinate),
                6,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                display_image,
                str(point_index + 1),
                (x_coordinate + 8, y_coordinate - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if len(self.clicked_points) >= 2:
            point_array = np.array(
                self.clicked_points,
                dtype=np.int32,
            )

            cv2.polylines(
                display_image,
                [point_array],
                False,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if len(self.clicked_points) == 4:
            point_array = np.array(
                self.clicked_points,
                dtype=np.int32,
            )

            cv2.polylines(
                display_image,
                [point_array],
                True,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.frozen_image is None:
            status = "LIVE - Press SPACE to freeze"
            status_colour = (0, 255, 255)
        elif len(self.clicked_points) < 4:
            next_corner = self.CORNER_NAMES[
                len(self.clicked_points)
            ]
            status = (
                f"FROZEN - Click {next_corner} "
                f"({len(self.clicked_points) + 1}/4)"
            )
            status_colour = (0, 165, 255)
        elif self.saved:
            status = "CALIBRATION SAVED - Press Q to quit"
            status_colour = (0, 255, 0)
        else:
            status = "Four corners selected - Press S to save"
            status_colour = (0, 255, 0)

        cv2.rectangle(
            display_image,
            (0, 0),
            (display_image.shape[1], 34),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            display_image,
            status,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_colour,
            2,
            cv2.LINE_AA,
        )

        return display_image

    def calculate_homography(self):
        image_points = np.array(
            self.clicked_points,
            dtype=np.float32,
        )

        table_points = np.array(
            [
                [0.0, 0.0],
                [self.reference_width_m, 0.0],
                [
                    self.reference_width_m,
                    self.reference_height_m,
                ],
                [0.0, self.reference_height_m],
            ],
            dtype=np.float32,
        )

        pixel_to_table = cv2.getPerspectiveTransform(
            image_points,
            table_points,
        )

        table_to_pixel = np.linalg.inv(pixel_to_table)

        transformed_points = cv2.perspectiveTransform(
            image_points.reshape(1, -1, 2),
            pixel_to_table,
        ).reshape(-1, 2)

        width_top = float(
            np.linalg.norm(
                transformed_points[1]
                - transformed_points[0]
            )
        )
        width_bottom = float(
            np.linalg.norm(
                transformed_points[2]
                - transformed_points[3]
            )
        )
        height_right = float(
            np.linalg.norm(
                transformed_points[2]
                - transformed_points[1]
            )
        )
        height_left = float(
            np.linalg.norm(
                transformed_points[3]
                - transformed_points[0]
            )
        )

        return (
            image_points,
            pixel_to_table,
            table_to_pixel,
            width_top,
            width_bottom,
            height_right,
            height_left,
        )

    @staticmethod
    def matrix_data(matrix):
        return ", ".join(
            f"{float(value):.16g}"
            for value in matrix.reshape(-1)
        )

    def save_calibration(self):
        if self.frozen_image is None:
            self.get_logger().error(
                "No frozen calibration image is available"
            )
            return

        if len(self.clicked_points) != 4:
            self.get_logger().error(
                "Exactly four corners must be selected"
            )
            return

        (
            image_points,
            pixel_to_table,
            table_to_pixel,
            width_top,
            width_bottom,
            height_right,
            height_left,
        ) = self.calculate_homography()

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        yaml_lines = [
            "format_version: 1",
            f'image_topic: "{self.input_topic}"',
            'reference_type: "ISO_ID_1_card"',
            f"reference_width_m: {self.reference_width_m:.8f}",
            (
                "reference_height_m: "
                f"{self.reference_height_m:.8f}"
            ),
            (
                "image_width_px: "
                f"{self.frozen_image.shape[1]}"
            ),
            (
                "image_height_px: "
                f"{self.frozen_image.shape[0]}"
            ),
            "image_points_px:",
        ]

        for point, corner_name in zip(
            image_points,
            self.CORNER_NAMES,
        ):
            yaml_lines.append(
                f'  - name: "{corner_name}"'
            )
            yaml_lines.append(
                f"    x: {float(point[0]):.6f}"
            )
            yaml_lines.append(
                f"    y: {float(point[1]):.6f}"
            )

        yaml_lines.extend(
            [
                "pixel_to_table_homography:",
                "  rows: 3",
                "  cols: 3",
                (
                    "  data: ["
                    + self.matrix_data(pixel_to_table)
                    + "]"
                ),
                "table_to_pixel_homography:",
                "  rows: 3",
                "  cols: 3",
                (
                    "  data: ["
                    + self.matrix_data(table_to_pixel)
                    + "]"
                ),
                "validation:",
                f"  width_top_m: {width_top:.8f}",
                f"  width_bottom_m: {width_bottom:.8f}",
                f"  height_right_m: {height_right:.8f}",
                f"  height_left_m: {height_left:.8f}",
                "",
            ]
        )

        self.output_file.write_text(
            "\n".join(yaml_lines),
            encoding="utf-8",
        )

        self.saved = True

        self.get_logger().info(
            f"Calibration saved to {self.output_file}"
        )
        self.get_logger().info(
            "Validation dimensions: "
            f"top={width_top:.5f} m, "
            f"bottom={width_bottom:.5f} m, "
            f"right={height_right:.5f} m, "
            f"left={height_left:.5f} m"
        )

    def update_display(self):
        if self.latest_image is None:
            return

        self.initialize_window()

        if self.frozen_image is None:
            source_image = self.latest_image
        else:
            source_image = self.frozen_image

        display_image = self.draw_interface(source_image)

        cv2.imshow(
            self.window_name,
            display_image,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if self.frozen_image is None:
                self.frozen_image = self.latest_image.copy()
                self.reset_selection()
                self.get_logger().info(
                    "Image frozen. Click the top-left corner."
                )
            else:
                self.frozen_image = None
                self.reset_selection()
                self.get_logger().info(
                    "Live view resumed"
                )

        elif key in (ord("r"), ord("R")):
            if self.frozen_image is not None:
                self.reset_selection()

        elif key in (ord("s"), ord("S")):
            self.save_calibration()

        elif key in (ord("q"), ord("Q"), 27):
            self.get_logger().info(
                "Closing calibration tool"
            )
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TableHomographyCalibrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()