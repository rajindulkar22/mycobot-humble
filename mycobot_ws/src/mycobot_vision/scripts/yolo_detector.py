#!/usr/bin/env python3

import cv2
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
from ultralytics import YOLO
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("model_path", "yolo11n.pt")
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

        self.declare_parameter("allowed_classes", ["cup"])

        self.declare_parameter("roi_enabled", True)
        self.declare_parameter("roi_x_min", 80)
        self.declare_parameter("roi_y_min", 100)
        self.declare_parameter("roi_x_max", 600)
        self.declare_parameter("roi_y_max", 450)

        model_path = self.get_parameter("model_path").value
        input_topic = self.get_parameter("input_topic").value
        annotated_topic = self.get_parameter(
            "annotated_topic"
        ).value
        detections_topic = self.get_parameter(
            "detections_topic"
        ).value

        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self.iou_threshold = float(
            self.get_parameter("iou_threshold").value
        )
        self.image_size = int(
            self.get_parameter("image_size").value
        )
        self.device = str(
            self.get_parameter("device").value
        )
        self.process_every_n_frames = max(
            1,
            int(
                self.get_parameter(
                    "process_every_n_frames"
                ).value
            ),
        )

        self.allowed_classes = {
            str(class_name).strip().lower()
            for class_name in self.get_parameter(
                "allowed_classes"
            ).value
            if str(class_name).strip()
        }

        self.roi_enabled = bool(
            self.get_parameter("roi_enabled").value
        )
        self.roi_x_min = int(
            self.get_parameter("roi_x_min").value
        )
        self.roi_y_min = int(
            self.get_parameter("roi_y_min").value
        )
        self.roi_x_max = int(
            self.get_parameter("roi_x_max").value
        )
        self.roi_y_max = int(
            self.get_parameter("roi_y_max").value
        )

        if (
            self.roi_x_max <= self.roi_x_min
            or self.roi_y_max <= self.roi_y_min
        ):
            raise ValueError(
                "ROI maximum values must be greater than "
                "ROI minimum values"
            )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.processed_count = 0
        self.invalid_roi_warning_sent = False

        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.get_logger().info(
            f"Loading YOLO model: {model_path}"
        )
        self.model = YOLO(model_path)

        self.allowed_class_ids = self.resolve_allowed_classes()

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

        self.get_logger().info(
            f"Listening for images on {input_topic}"
        )

        if self.allowed_classes:
            self.get_logger().info(
                "Allowed classes: "
                + ", ".join(sorted(self.allowed_classes))
            )
        else:
            self.get_logger().info(
                "Allowed-class filtering is disabled"
            )

        if self.roi_enabled:
            self.get_logger().info(
                "ROI enabled: "
                f"({self.roi_x_min}, {self.roi_y_min}) to "
                f"({self.roi_x_max}, {self.roi_y_max})"
            )
        else:
            self.get_logger().info("ROI filtering is disabled")

    def resolve_allowed_classes(self):
        if not self.allowed_classes:
            return None

        if isinstance(self.model.names, dict):
            model_classes = self.model.names.items()
        else:
            model_classes = enumerate(self.model.names)

        allowed_class_ids = []
        available_class_names = set()

        for class_id, class_name in model_classes:
            normalized_name = str(class_name).strip().lower()
            available_class_names.add(normalized_name)

            if normalized_name in self.allowed_classes:
                allowed_class_ids.append(int(class_id))

        unknown_classes = (
            self.allowed_classes - available_class_names
        )

        if unknown_classes:
            self.get_logger().warning(
                "Configured classes are not present in the model: "
                + ", ".join(sorted(unknown_classes))
            )

        if not allowed_class_ids:
            raise ValueError(
                "None of the configured allowed_classes are "
                "present in the YOLO model"
            )

        return allowed_class_ids

    def get_roi(self, image_width, image_height):
        if not self.roi_enabled:
            return 0, 0, image_width, image_height

        x_min = max(0, min(self.roi_x_min, image_width - 1))
        y_min = max(0, min(self.roi_y_min, image_height - 1))
        x_max = max(1, min(self.roi_x_max, image_width))
        y_max = max(1, min(self.roi_y_max, image_height))

        if x_max <= x_min or y_max <= y_min:
            if not self.invalid_roi_warning_sent:
                self.get_logger().warning(
                    "Configured ROI is invalid for the current "
                    "image size; using the complete image"
                )
                self.invalid_roi_warning_sent = True

            return 0, 0, image_width, image_height

        return x_min, y_min, x_max, y_max

    @staticmethod
    def class_name_from_result(result, class_id):
        if isinstance(result.names, dict):
            return str(result.names.get(class_id, class_id))

        return str(result.names[class_id])

    def draw_detection(
        self,
        image,
        x_min,
        y_min,
        x_max,
        y_max,
        class_name,
        confidence,
    ):
        colour = (0, 255, 0)

        top_left = (
            int(round(x_min)),
            int(round(y_min)),
        )
        bottom_right = (
            int(round(x_max)),
            int(round(y_max)),
        )

        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            colour,
            2,
        )

        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_x = max(0, top_left[0])
        label_y = max(
            text_size[1] + baseline + 4,
            top_left[1],
        )

        cv2.rectangle(
            image,
            (
                label_x,
                label_y - text_size[1] - baseline - 4,
            ),
            (
                label_x + text_size[0] + 6,
                label_y + 2,
            ),
            colour,
            -1,
        )

        cv2.putText(
            image,
            label,
            (label_x + 3, label_y - baseline),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
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

            image_height, image_width = image.shape[:2]

            roi_x_min, roi_y_min, roi_x_max, roi_y_max = (
                self.get_roi(image_width, image_height)
            )

            result = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                classes=self.allowed_class_ids,
                verbose=False,
            )[0]

            detections_message = Detection2DArray()
            detections_message.header = image_message.header

            annotated_image = image.copy()

            if self.roi_enabled:
                cv2.rectangle(
                    annotated_image,
                    (roi_x_min, roi_y_min),
                    (roi_x_max, roi_y_max),
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated_image,
                    "Manipulation ROI",
                    (roi_x_min + 5, max(20, roi_y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

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

                    class_name = self.class_name_from_result(
                        result,
                        class_id,
                    )
                    normalized_class_name = (
                        class_name.strip().lower()
                    )

                    if (
                        self.allowed_classes
                        and normalized_class_name
                        not in self.allowed_classes
                    ):
                        continue

                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0

                    center_is_inside_roi = (
                        roi_x_min <= center_x <= roi_x_max
                        and roi_y_min <= center_y <= roi_y_max
                    )

                    if (
                        self.roi_enabled
                        and not center_is_inside_roi
                    ):
                        continue

                    detection = Detection2D()
                    detection.header = image_message.header

                    detection.bbox.center.position.x = center_x
                    detection.bbox.center.position.y = center_y
                    detection.bbox.center.theta = 0.0
                    detection.bbox.size_x = max(
                        0.0,
                        x_max - x_min,
                    )
                    detection.bbox.size_y = max(
                        0.0,
                        y_max - y_min,
                    )

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = class_name
                    hypothesis.hypothesis.score = float(
                        confidence
                    )

                    detection.results.append(hypothesis)
                    detections_message.detections.append(
                        detection
                    )

                    self.draw_detection(
                        annotated_image,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        class_name,
                        float(confidence),
                    )

            self.detections_publisher.publish(
                detections_message
            )

            annotated_message = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding="bgr8",
            )
            annotated_message.header = image_message.header

            self.annotated_publisher.publish(
                annotated_message
            )

            self.processed_count += 1

            if self.processed_count % 30 == 0:
                self.get_logger().info(
                    "Processed "
                    f"{self.processed_count} frames; "
                    "accepted detections: "
                    f"{len(detections_message.detections)}"
                )

        except Exception as exception:
            self.get_logger().error(
                f"YOLO inference failed: {exception}"
            )


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