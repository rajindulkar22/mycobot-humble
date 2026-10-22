#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray


class TablePreview(Node):
    def __init__(self):
        super().__init__("table_preview")
        self.pub = self.create_publisher(
            MarkerArray, "/overview_camera/table_markers", 10
        )
        self.sub = self.create_subscription(
            Detection3DArray, "/overview_camera/table_detections",
            self.receive, qos_profile_sensor_data
        )
        self.latest = None
        self.received_at = 0.0
        self.timer = self.create_timer(0.2, self.draw)
        self.get_logger().info("Publishing card and live cap markers")

    def receive(self, msg):
        if msg.header.frame_id != "table_frame":
            self.get_logger().warning("Ignoring detections outside table_frame")
            return
        self.latest = msg
        self.received_at = time.monotonic()

    def marker(self, namespace, number, kind, colour):
        m = Marker()
        m.header.frame_id = "table_frame"
        m.ns, m.id, m.type = namespace, number, kind
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b = colour
        m.color.a = 1.0
        m.lifetime.sec = 1
        return m

    def draw(self):
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers = [clear]
        corners = [
            ("A", 0.0, 0.0), ("B", 0.0856, 0.0),
            ("D", 0.0856, 0.05398), ("C", 0.0, 0.05398)
        ]
        card = self.marker("card", 0, Marker.LINE_STRIP, (1.0, 0.8, 0.0))
        card.scale.x = 0.002
        card.points = [
            Point(x=x, y=y, z=-0.001)
            for _, x, y in corners + [corners[0]]
        ]
        markers.append(card)

        for i, (name, x, y) in enumerate(corners):
            label = self.marker("corners", i, Marker.TEXT_VIEW_FACING,
                                (1.0, 0.8, 0.0))
            label.pose.position = Point(x=x, y=y, z=-0.015)
            label.scale.z = 0.012
            label.text = name
            markers.append(label)

        colours = {
            "red_cap": (1.0, 0.0, 0.0),
            "black_cap": (0.15, 0.15, 0.15),
            "white_cap": (1.0, 1.0, 1.0),
        }
        if self.latest and time.monotonic() - self.received_at < 1.5:
            for i, detection in enumerate(self.latest.detections):
                if not detection.results:
                    continue
                name = detection.results[0].hypothesis.class_id
                p = detection.bbox.center.position
                cap = self.marker("caps", i, Marker.CYLINDER,
                                  colours.get(name, (0.5, 0.5, 0.5)))
                cap.pose.position = Point(x=p.x, y=p.y, z=-0.005)
                cap.scale.x = cap.scale.y = 0.030
                cap.scale.z = 0.010
                markers.append(cap)

                # table_frame +Z points downward, so heights are negative Z.
                grasp = Point(x=p.x, y=p.y, z=-0.007)
                approach = Point(x=p.x, y=p.y, z=-0.057)

                for namespace, point, colour in [
                    ("grasp_points", grasp, (0.0, 1.0, 1.0)),
                    ("approach_points", approach, (0.0, 1.0, 0.0)),
                ]:
                    marker = self.marker(
                        namespace, i, Marker.SPHERE, colour
                    )
                    marker.pose.position = point
                    marker.scale.x = 0.008
                    marker.scale.y = 0.008
                    marker.scale.z = 0.008
                    markers.append(marker)

                arrow = self.marker(
                    "approach_direction", i, Marker.ARROW,
                    (0.0, 1.0, 0.0)
                )
                arrow.points = [approach, grasp]
                arrow.scale.x = 0.002
                arrow.scale.y = 0.006
                arrow.scale.z = 0.008
                markers.append(arrow)
                label = self.marker("cap_labels", i, Marker.TEXT_VIEW_FACING,
                                    (1.0, 1.0, 1.0))
                label.pose.position = Point(x=p.x + 0.025, y=p.y, z=-0.025)
                label.scale.z = 0.012
                label.text = name
                markers.append(label)

        self.pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = TablePreview()
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
