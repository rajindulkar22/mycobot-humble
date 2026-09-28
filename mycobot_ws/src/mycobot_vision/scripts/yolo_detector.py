#!/usr/bin/env python3

from collections import deque
import math

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


class DetectionCandidate:
    def __init__(
        self,
        class_name,
        confidence,
        x_min,
        y_min,
        x_max,
        y_max,
    ):
        self.class_name = class_name
        self.confidence = confidence
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self):
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self):
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self):
        return max(0.0, self.y_max - self.y_min)


class DetectionTrack:
    def __init__(self, track_id, history_size):
        self.track_id = track_id
        self.history = deque(maxlen=history_size)
        self.last_candidate = None
        self.consecutive_hits = 0
        self.missed_frames = 0
        self.matched_this_frame = False

    @property
    def class_name(self):
        if self.last_candidate is None:
            return ""

        return self.last_candidate.class_name

    def update(self, candidate):
        self.last_candidate = candidate
        self.history.append(candidate)
        self.consecutive_hits += 1
        self.missed_frames = 0
        self.matched_this_frame = True

    def mark_missed(self):
        self.missed_frames += 1
        self.consecutive_hits = 0
        self.matched_this_frame = False
        self.history.clear()

    def maximum_jitter(self, sample_count):
        samples = list(self.history)[-sample_count:]

        if not samples:
            return float("inf")

        mean_x = sum(
            candidate.center_x for candidate in samples
        ) / len(samples)
        mean_y = sum(
            candidate.center_y for candidate in samples
        ) / len(samples)

        return max(
            math.hypot(
                candidate.center_x - mean_x,
                candidate.center_y - mean_y,
            )
            for candidate in samples
        )

    def is_stable(
        self,
        required_frames,
        maximum_jitter_pixels,
    ):
        if not self.matched_this_frame:
            return False

        if self.consecutive_hits < required_frames:
            return False

        if len(self.history) < required_frames:
            return False

        return (
            self.maximum_jitter(required_frames)
            <= maximum_jitter_pixels
        )

    def averaged_candidate(self, sample_count):
        samples = list(self.history)[-sample_count:]

        return DetectionCandidate(
            class_name=self.class_name,
            confidence=sum(
                candidate.confidence for candidate in samples
            )
            / len(samples),
            x_min=sum(
                candidate.x_min for candidate in samples
            )
            / len(samples),
            y_min=sum(
                candidate.y_min for candidate in samples
            )
            / len(samples),
            x_max=sum(
                candidate.x_max for candidate in samples
            )
            / len(samples),
            y_max=sum(
                candidate.y_max for candidate in samples
            )
            / len(samples),
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

        self.declare_parameter("stability_enabled", True)
        self.declare_parameter("stability_required_frames", 3)
        self.declare_parameter(
            "stability_max_center_distance_pixels",
            40.0,
        )
        self.declare_parameter(
            "stability_max_jitter_pixels",
            10.0,
        )
        self.declare_parameter(
            "stability_max_missed_frames",
            2,
        )
        self.declare_parameter(
            "stability_history_size",
            5,
        )

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

        self.stability_enabled = bool(
            self.get_parameter("stability_enabled").value
        )
        self.stability_required_frames = max(
            1,
            int(
                self.get_parameter(
                    "stability_required_frames"
                ).value
            ),
        )
        self.stability_max_center_distance_pixels = max(
            0.0,
            float(
                self.get_parameter(
                    "stability_max_center_distance_pixels"
                ).value
            ),
        )
        self.stability_max_jitter_pixels = max(
            0.0,
            float(
                self.get_parameter(
                    "stability_max_jitter_pixels"
                ).value
            ),
        )
        self.stability_max_missed_frames = max(
            0,
            int(
                self.get_parameter(
                    "stability_max_missed_frames"
                ).value
            ),
        )
        self.stability_history_size = max(
            self.stability_required_frames,
            int(
                self.get_parameter(
                    "stability_history_size"
                ).value
            ),
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

        self.tracks = []
        self.next_track_id = 1

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

        if self.roi_enabled:
            self.get_logger().info(
                "ROI enabled: "
                f"({self.roi_x_min}, {self.roi_y_min}) to "
                f"({self.roi_x_max}, {self.roi_y_max})"
            )

        if self.stability_enabled:
            self.get_logger().info(
                "Detection stabilization enabled: "
                f"{self.stability_required_frames} frames, "
                f"maximum jitter "
                f"{self.stability_max_jitter_pixels:.1f} px"
            )

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

    def extract_candidates(
        self,
        result,
        roi_x_min,
        roi_y_min,
        roi_x_max,
        roi_y_max,
    ):
        candidates = []

        if result.boxes is None:
            return candidates

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
            normalized_class_name = class_name.strip().lower()

            if (
                self.allowed_classes
                and normalized_class_name
                not in self.allowed_classes
            ):
                continue

            candidate = DetectionCandidate(
                class_name=class_name,
                confidence=float(confidence),
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

            center_is_inside_roi = (
                roi_x_min <= candidate.center_x <= roi_x_max
                and roi_y_min <= candidate.center_y <= roi_y_max
            )

            if self.roi_enabled and not center_is_inside_roi:
                continue

            candidates.append(candidate)

        return candidates

    def update_tracks(self, candidates):
        for track in self.tracks:
            track.matched_this_frame = False

        assignments = {}

        for candidate_index, candidate in enumerate(candidates):
            best_track = None
            best_distance = float("inf")

            for track in self.tracks:
                if track.matched_this_frame:
                    continue

                if track.class_name != candidate.class_name:
                    continue

                if track.last_candidate is None:
                    continue

                distance = math.hypot(
                    candidate.center_x
                    - track.last_candidate.center_x,
                    candidate.center_y
                    - track.last_candidate.center_y,
                )

                if (
                    distance
                    <= self.stability_max_center_distance_pixels
                    and distance < best_distance
                ):
                    best_track = track
                    best_distance = distance

            if best_track is None:
                best_track = DetectionTrack(
                    track_id=self.next_track_id,
                    history_size=self.stability_history_size,
                )
                self.next_track_id += 1
                self.tracks.append(best_track)

            best_track.update(candidate)
            assignments[candidate_index] = best_track

        for track in self.tracks:
            if not track.matched_this_frame:
                track.mark_missed()

        self.tracks = [
            track
            for track in self.tracks
            if (
                track.missed_frames
                <= self.stability_max_missed_frames
            )
        ]

        return assignments

    def create_detection_message(
        self,
        candidate,
        header,
        track_id=None,
    ):
        detection = Detection2D()
        detection.header = header

        detection.bbox.center.position.x = candidate.center_x
        detection.bbox.center.position.y = candidate.center_y
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = candidate.width
        detection.bbox.size_y = candidate.height

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = candidate.class_name
        hypothesis.hypothesis.score = candidate.confidence

        detection.results.append(hypothesis)

        if track_id is not None:
            detection.id = f"track_{track_id}"

        return detection

    @staticmethod
    def draw_box(image, candidate, label, colour):
        top_left = (
            int(round(candidate.x_min)),
            int(round(candidate.y_min)),
        )
        bottom_right = (
            int(round(candidate.x_max)),
            int(round(candidate.y_max)),
        )

        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            colour,
            2,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
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

            candidates = self.extract_candidates(
                result,
                roi_x_min,
                roi_y_min,
                roi_x_max,
                roi_y_max,
            )

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

            if self.stability_enabled:
                assignments = self.update_tracks(candidates)

                for candidate_index, candidate in enumerate(
                    candidates
                ):
                    track = assignments[candidate_index]

                    if track.is_stable(
                        self.stability_required_frames,
                        self.stability_max_jitter_pixels,
                    ):
                        averaged_candidate = (
                            track.averaged_candidate(
                                self.stability_required_frames
                            )
                        )

                        detection = (
                            self.create_detection_message(
                                averaged_candidate,
                                image_message.header,
                                track.track_id,
                            )
                        )
                        detections_message.detections.append(
                            detection
                        )

                        label = (
                            f"{averaged_candidate.class_name} "
                            f"{averaged_candidate.confidence:.2f} "
                            "STABLE"
                        )
                        self.draw_box(
                            annotated_image,
                            averaged_candidate,
                            label,
                            (0, 255, 0),
                        )

                    elif (
                        track.consecutive_hits
                        >= self.stability_required_frames
                    ):
                        label = (
                            f"{candidate.class_name} MOVING"
                        )
                        self.draw_box(
                            annotated_image,
                            candidate,
                            label,
                            (0, 165, 255),
                        )

                    else:
                        label = (
                            f"{candidate.class_name} "
                            f"{track.consecutive_hits}/"
                            f"{self.stability_required_frames}"
                        )
                        self.draw_box(
                            annotated_image,
                            candidate,
                            label,
                            (0, 165, 255),
                        )

            else:
                for candidate in candidates:
                    detection = self.create_detection_message(
                        candidate,
                        image_message.header,
                    )
                    detections_message.detections.append(
                        detection
                    )

                    label = (
                        f"{candidate.class_name} "
                        f"{candidate.confidence:.2f}"
                    )
                    self.draw_box(
                        annotated_image,
                        candidate,
                        label,
                        (0, 255, 0),
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
                    "stable detections: "
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