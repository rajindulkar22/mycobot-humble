#!/usr/bin/env python3

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

        model_path = (
            self.get_parameter("model_path")
            .get_parameter_value()
            .string_value
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
        self.model = YOLO(model_path)

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

    def image_callback(self, image_message):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding="bgr8",
            )

            result = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]

            detections_message = Detection2DArray()
            detections_message.header = image_message.header

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

                    detection = Detection2D()
                    detection.header = image_message.header

                    detection.bbox.center.position.x = (
                        x_min + x_max
                    ) / 2.0
                    detection.bbox.center.position.y = (
                        y_min + y_max
                    ) / 2.0
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
                    hypothesis.hypothesis.score = float(confidence)

                    detection.results.append(hypothesis)
                    detections_message.detections.append(detection)

            self.detections_publisher.publish(
                detections_message
            )

            annotated_image = result.plot()
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
                    f"latest detections: "
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