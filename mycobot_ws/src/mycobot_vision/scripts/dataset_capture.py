#!/usr/bin/env python3

import csv
from pathlib import Path

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class DatasetCapture(Node):
    def __init__(self):
        super().__init__("dataset_capture")

        self.declare_parameter(
            "input_topic",
            "/overview_camera/image_rect",
        )
        self.declare_parameter(
            "output_directory",
            "/root/mycobot_humble/datasets/"
            "bottle_caps/raw/images",
        )
        self.declare_parameter("filename_prefix", "cap")
        self.declare_parameter("jpeg_quality", 95)

        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )
        self.output_directory = Path(
            self.get_parameter("output_directory")
            .get_parameter_value()
            .string_value
        )
        self.filename_prefix = (
            self.get_parameter("filename_prefix")
            .get_parameter_value()
            .string_value
        )
        self.jpeg_quality = int(
            self.get_parameter("jpeg_quality")
            .get_parameter_value()
            .integer_value
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest_path = (
            self.output_directory.parent / "capture_manifest.csv"
        )

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_message = None
        self.saved_count = self._find_next_index() - 1
        self.window_name = "Bottle-cap dataset capture"

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.display_timer = self.create_timer(
            1.0 / 30.0,
            self.display_callback,
        )

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        self.get_logger().info(
            f"Waiting for rectified images on {self.input_topic}"
        )
        self.get_logger().info(
            f"Images will be saved in {self.output_directory}"
        )
        self.get_logger().info(
            "Controls: SPACE=save image, Q=quit"
        )

    def _find_next_index(self):
        highest_index = 0

        pattern = f"{self.filename_prefix}_*.jpg"

        for image_path in self.output_directory.glob(pattern):
            suffix = image_path.stem.removeprefix(
                f"{self.filename_prefix}_"
            )

            if suffix.isdigit():
                highest_index = max(
                    highest_index,
                    int(suffix),
                )

        return highest_index + 1

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as exception:
            self.get_logger().error(
                f"Image conversion failed: {exception}"
            )
            return

        self.latest_image = image
        self.latest_message = message

    def display_callback(self):
        if self.latest_image is None:
            waiting_image = self._create_waiting_image()
            cv2.imshow(self.window_name, waiting_image)
        else:
            display_image = self.latest_image.copy()
            self._draw_overlay(display_image)
            cv2.imshow(self.window_name, display_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            self.save_current_image()
        elif key in (ord("q"), ord("Q"), 27):
            self.get_logger().info(
                "Dataset capture stopped by user"
            )
            rclpy.shutdown()

    def _create_waiting_image(self):
        image = 30 * (
            cv2.UMat(480, 640, cv2.CV_8UC3).get()
        )

        cv2.putText(
            image,
            f"Waiting for {self.input_topic}",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return image

    def _draw_overlay(self, image):
        height, width = image.shape[:2]

        cv2.rectangle(
            image,
            (0, 0),
            (width, 72),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            image,
            "SPACE: save clean frame    Q: quit",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"Saved images: {self.saved_count}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{width}x{height} rectified",
            (width - 220, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def save_current_image(self):
        if self.latest_image is None:
            self.get_logger().warning(
                "No image is available to save"
            )
            return

        next_index = self._find_next_index()

        filename = (
            f"{self.filename_prefix}_{next_index:06d}.jpg"
        )
        output_path = self.output_directory / filename

        success = cv2.imwrite(
            str(output_path),
            self.latest_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not success:
            self.get_logger().error(
                f"Failed to save {output_path}"
            )
            return

        self.saved_count = next_index
        self._append_manifest(filename)

        self.get_logger().info(
            f"Saved {output_path}"
        )

    def _append_manifest(self, filename):
        manifest_exists = self.manifest_path.exists()

        message = self.latest_message
        stamp = message.header.stamp

        with self.manifest_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as manifest_file:
            writer = csv.writer(manifest_file)

            if not manifest_exists:
                writer.writerow(
                    [
                        "filename",
                        "ros_time_sec",
                        "ros_time_nanosec",
                        "frame_id",
                        "source_topic",
                    ]
                )

            writer.writerow(
                [
                    filename,
                    stamp.sec,
                    stamp.nanosec,
                    message.header.frame_id,
                    self.input_topic,
                ]
            )


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCapture()

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