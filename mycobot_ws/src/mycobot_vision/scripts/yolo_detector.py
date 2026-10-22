#!/usr/bin/env python3

import math
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

# Silver caps score near 0.10. Red and black keep the yaml threshold.
WHITE_CAP_CONFIDENCE = 0.08


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("model_path", "best.pt")
        self.declare_parameter("target_class", "")
        self.declare_parameter(
            "input_topic",
            "/overview_camera/image_rect",
        )
        self.declare_parameter(
            "annotated_topic",
            "/overview_camera/yolo/annotated",
        )
        self.declare_parameter(
            "detections_topic",
            "/overview_camera/yolo/detections",
        )
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("iou_threshold", 0.50)
        self.declare_parameter("image_size", 480)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("process_every_n_frames", 3)

        model_path = _resolve_model_path(
            self.get_parameter("model_path")
            .get_parameter_value()
            .string_value
        )
        self.target_class = (
            self.get_parameter("target_class")
            .get_parameter_value()
            .string_value
            .strip()
        )
        input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )
        annotated_topic = (
            self.get_parameter("annotated_topic")
            .get_parameter_value()
            .string_value
        )
        detections_topic = (
            self.get_parameter("detections_topic")
            .get_parameter_value()
            .string_value
        )

        self.confidence_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.iou_threshold = (
            self.get_parameter("iou_threshold")
            .get_parameter_value()
            .double_value
        )
        self.image_size = (
            self.get_parameter("image_size")
            .get_parameter_value()
            .integer_value
        )
        self.device = (
            self.get_parameter("device")
            .get_parameter_value()
            .string_value
        )
        self.process_every_n_frames = max(
            1,
            self.get_parameter("process_every_n_frames")
            .get_parameter_value()
            .integer_value,
        )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.processed_count = 0

        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.get_logger().info(
            f"Loading YOLO model: {model_path}"
        )
        if self.target_class:
            self.get_logger().info(
                f"Publishing only class: {self.target_class}"
            )
        try:
            self.model = YOLO(model_path)
        except Exception as error:
            raise RuntimeError(
                "Could not load trained weights "
                f"'{model_path}': {error}. "
                "Picking stays disabled."
            ) from error

        self.image_subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            self.image_qos,
        )

        self.annotated_publisher = self.create_publisher(
            Image,
            annotated_topic,
            self.image_qos,
        )

        self.detections_publisher = self.create_publisher(
            Detection2DArray,
            detections_topic,
            10,
        )
        self.ready_publisher = self.create_publisher(
            Bool,
            "/overview_camera/yolo/model_ready",
            _ready_qos(),
        )
        ready = Bool()
        ready.data = True
        self.ready_publisher.publish(ready)

        self.get_logger().info(
            f"Listening for images on {input_topic}"
        )

    def image_callback(self, image_message):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding="bgr8",
            )

            # The silver cap scores about 0.10. Red and black stay at
            # the configured threshold so the cable is not a black cap.
            result = self.model.predict(
                source=image,
                conf=min(self.confidence_threshold, WHITE_CAP_CONFIDENCE),
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]

            detections_message = Detection2DArray()
            detections_message.header = image_message.header
            verified = []

            if result.boxes is not None:
                coordinates = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for box, confidence, class_id_value in zip(
                    coordinates,
                    confidences,
                    class_ids,
                ):
                    x_min, y_min, x_max, y_max = [
                        float(value) for value in box
                    ]
                    class_id = int(class_id_value)

                    if isinstance(result.names, dict):
                        class_name = str(
                            result.names.get(class_id, class_id)
                        )
                    else:
                        class_name = str(result.names[class_id])

                    if (
                        self.target_class
                        and class_name.casefold()
                        != self.target_class.casefold()
                    ):
                        continue
                    minimum = (
                        WHITE_CAP_CONFIDENCE
                        if class_name.casefold() == "white_cap"
                        else self.confidence_threshold
                    )
                    if float(confidence) < minimum:
                        continue

                    box_u = (x_min + x_max) / 2.0
                    box_v = (y_min + y_max) / 2.0
                    size_x = max(0.0, x_max - x_min)
                    size_y = max(0.0, y_max - y_min)
                    disc = _cap_disc_center(
                        image,
                        class_name,
                        box_u,
                        box_v,
                        size_x,
                        size_y,
                    )
                    if disc is None:
                        continue
                    center_u, center_v = disc

                    detection = Detection2D()
                    detection.header = image_message.header

                    detection.bbox.center.position.x = float(center_u)
                    detection.bbox.center.position.y = float(center_v)
                    detection.bbox.center.theta = 0.0
                    detection.bbox.size_x = size_x
                    detection.bbox.size_y = size_y

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = class_name
                    hypothesis.hypothesis.score = float(confidence)

                    detection.results.append(hypothesis)
                    detections_message.detections.append(detection)
                    verified.append(
                        (
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                            center_u,
                            center_v,
                            class_name,
                            float(confidence),
                        )
                    )

            self.detections_publisher.publish(
                detections_message
            )

            annotated_image = image.copy()
            _draw_verified_caps(annotated_image, verified)
            annotated_message = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding="bgr8",
            )
            annotated_message.header = image_message.header

            self.annotated_publisher.publish(
                annotated_message
            )

            self.processed_count += 1
            if self.processed_count == 1:
                self.get_logger().info(
                    "Detection centers use the "
                    f"{image.shape[1]}x{image.shape[0]} source image, "
                    f"not inference size {self.image_size}."
                )

            if self.processed_count % 30 == 0:
                self.get_logger().info(
                    "Processed "
                    f"{self.processed_count} frames; "
                    f"latest detections: "
                    f"{len(detections_message.detections)}"
                )

        except Exception as exception:
            self.get_logger().error(
                f"YOLO inference failed: {exception}"
            )


def _cap_disc_center(image, class_name, pixel_u, pixel_v, size_x, size_y):
    """Return the disc centre for a real round cap, or None for cables."""
    if image is None or image.size == 0:
        return None
    span = max(float(size_x), float(size_y), 28.0)
    if span < 22.0 or span > 55.0:
        return None
    half = max(span * 0.8, 26.0)
    height, width = image.shape[:2]
    x0 = max(0, int(round(pixel_u - half)))
    y0 = max(0, int(round(pixel_v - half)))
    x1 = min(width, int(round(pixel_u + half)))
    y1 = min(height, int(round(pixel_v + half)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yy, xx = np.ogrid[:height, :width]
    dist = np.hypot(xx - float(pixel_u), yy - float(pixel_v))
    ring = gray_full[(dist >= span * 0.6) & (dist <= span * 1.0)]
    if ring.size < 40 or float(np.mean(ring < 90)) > 0.2:
        return None
    table = float(np.percentile(ring, 80))
    roi = image[y0:y1, x0:x1]
    gray = gray_full[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    name = str(class_name or "").casefold()
    if name == "red_cap":
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 80, 40), (12, 255, 255)),
            cv2.inRange(hsv, (168, 80, 40), (180, 255, 255)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
    elif name == "black_cap":
        mask = (gray < min(80.0, table - 55.0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
    elif name == "white_cap":
        mask = (
            (gray < table - 20.0) & (gray > 45.0)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
    else:
        return float(pixel_u), float(pixel_v)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 180.0 or area > 2200.0:
            continue
        _center, (short_w, short_h), _angle = cv2.minAreaRect(contour)
        short_side = min(float(short_w), float(short_h))
        long_side = max(float(short_w), float(short_h))
        if short_side < 14.0 or long_side / short_side > 1.45:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        center_u = x0 + moments["m10"] / moments["m00"]
        center_v = y0 + moments["m01"] / moments["m00"]
        if math.hypot(center_u - pixel_u, center_v - pixel_v) > span * 0.5:
            continue
        if best is None or area > best[0]:
            best = (area, center_u, center_v)
    if best is None:
        return None
    if name == "black_cap":
        # A dark logo on the silver cap is not a black cap.
        # The whole disc has to be dark.
        circle = np.zeros(gray_full.shape, np.uint8)
        cv2.circle(
            circle,
            (int(round(best[1])), int(round(best[2]))),
            12,
            255,
            -1,
        )
        pixels = gray_full[circle > 0]
        if pixels.size < 20 or float(np.median(pixels)) > 25.0:
            return None
    return best[1], best[2]


def _draw_verified_caps(image, verified):
    """Draw only the verified cap boxes and their yellow disc centres."""
    colors = {
        "red_cap": (255, 0, 0),
        "black_cap": (255, 255, 0),
        "white_cap": (255, 255, 255),
    }
    for (
        x_min,
        y_min,
        x_max,
        y_max,
        center_u,
        center_v,
        class_name,
        score,
    ) in verified:
        color = colors.get(class_name.casefold(), (0, 255, 0))
        p0 = (int(round(x_min)), int(round(y_min)))
        p1 = (int(round(x_max)), int(round(y_max)))
        cv2.rectangle(image, p0, p1, color, 2)
        label = f"{class_name} {score:.2f}"
        cv2.putText(
            image,
            label,
            (p0[0], max(16, p0[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            image,
            (int(round(center_u)), int(round(center_v))),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
            line_type=cv2.LINE_AA,
        )


def _ready_qos():
    """Keep the loaded-weights flag for subscribers that start later."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _resolve_model_path(model_path):
    """Return the trained weights path. Never substitute another model."""
    path = model_path
    if not os.path.isabs(path):
        if os.path.isfile(path):
            path = os.path.abspath(path)
        else:
            try:
                share = get_package_share_directory("mycobot_vision")
            except PackageNotFoundError:
                share = ""
            installed = (
                os.path.join(share, "models", path) if share else ""
            )
            if installed and os.path.isfile(installed):
                path = installed
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RuntimeError(
            f"Trained weights were not found at '{model_path}'. "
            "Picking stays disabled."
        )
    return path


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()