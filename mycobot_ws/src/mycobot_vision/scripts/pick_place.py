#!/usr/bin/env python3
"""Pick with overview for coarse aim, wrist camera for the vertical approach."""

import base64
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from grasp_autonomy import (
    MAX_ATTEMPTS_PER_CAP,
    SAFETY_PARAMS,
    AttemptLogger,
    CapTrack,
    TaskManager,
    classify_grasp,
    in_workspace,
    next_candidate,
    pair_fit_residuals,
    pixel_near,
    remember_failed_xy,
    xy_already_failed,
)

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from mycobot_interfaces.srv import MoveCoords, MoveJoints, SetGripper
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import (
    ExternalShutdownException,
    MultiThreadedExecutor,
)
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray

# The overview detector publishes about every 2.5 s. A 2 s limit
# expired a lock before the next frame, so a visible cap looked missing.
DETECTION_MAX_AGE_SECONDS = 6.0
CAP_CLASSES = {
    "red_cap": "red",
    "black_cap": "black",
    "white_cap": "white",
}
POSE_MAX_AGE_SECONDS = 1.0
FIT_SPOT_COUNT = 5
MIN_PIXEL_SPACING = 50.0
MIN_ROBOT_SPACING_MM = 25.0
# Affine fit needs coverage in both image axes. The last manual
# set had three spots on one row and a 25 mm check error.
MIN_CALIB_SPAN_U_PX = 100.0
MIN_CALIB_SPAN_V_PX = 80.0
MAX_ACCEPT_CHECK_ERROR_MM = 12.0
# Remake any fit spot whose residual exceeds this (spot 4 was ~10 mm).
MAX_FIT_SPOT_RESIDUAL_MM = 8.0
# Fit-spot Z should match table_z_mm within this band.
MAX_FIT_SPOT_Z_DELTA_MM = 5.0
COLOUR_DROP_KEYS = {
    "red_cap": ("drop_red_x_mm", "drop_red_y_mm"),
    "black_cap": ("drop_black_x_mm", "drop_black_y_mm"),
    "white_cap": ("drop_white_x_mm", "drop_white_y_mm"),
}
# Table ArUco for reliable pixel centres (DICT_4X4_50 id 0).
ARUCO_DICTIONARY = "DICT_4X4_50"
ARUCO_MARKER_ID = 0
ARUCO_PRINT_SIZE_PX = 400
ARUCO_PRINT_BORDER_PX = 60
SERVICE_WAIT_SECONDS = 2.0
MOVE_TIMEOUT_SECONDS = 25.0
GRIPPER_TIMEOUT_SECONDS = 42.0
GRIPPER_SPEED = 15
CLOSE_GRIPPER_SPEED = 15
DESCEND_SPEED = 5
ALIGN_SETTLE_SECONDS = 0.8
ALIGN_TOLERANCE_MM = 8.0
ALIGN_MAX_NUDGES = 3
ALIGN_MAX_STEP_MM = 10.0
ALIGN_FRAME_WAIT_SECONDS = 1.0
# Wrist image axes are fixed while the tool angle stays at the
# calibrated orientation. One millimetre scale and two signs turn a
# pixel error into a small robot nudge.
# Robot move (dx, dy mm) for a wrist pixel error (du, dv), measured at
# TRAVEL_Z_MM by stepping the tool 10 mm over a cap. The image axes are
# rotated against the robot axes, so separate signs per axis walked the
# arm away from the cap (89 px grew to 343 px on a live red pick).
WRIST_GAIN_MM_PER_PX = ((-0.268, 0.152), (0.089, 0.313))
# Estimated lens height above the tool point. Lower poses see the cap
# larger, so the travel-height gain is scaled down while descending.
WRIST_LENS_ABOVE_TIP_MM = 50.0
# A nudge that leaves the cap this much further from the cross is
# undone, and the descent finishes on the overview aim.
WRIST_DIVERGE_MARGIN_PX = 12.0
# The gripper sits on the left of this wrist mount. Cap search
# ignores the far-left body and bright gripper plastic so pads
# are never mistaken for a lid. The yellow cross defaults to the
# open-jaw grasp point, not the image centre.
WRIST_TARGET_U_FRAC = 0.30
WRIST_TARGET_V_FRAC = 0.50
WRIST_SEARCH_LEFT_FRAC = 0.18
WRIST_MIN_CIRCULARITY = 0.72
# A real cap fills most of its enclosing circle and stays in a
# known size band. Glare holes are closed before this test.
WRIST_MIN_FILL = 0.55
WRIST_MIN_DIAMETER_PX = 36.0
WRIST_MAX_DIAMETER_PX = 220.0
# While a colour is locked, only a reading on that same disc is
# kept. 70 px was about one cap across, so the ring could sit on
# the table beside the cap and still count.
WRIST_LOCK_RADIUS_PX = 28.0
WRIST_LOCK_BLEND = 0.35
WRIST_LOCK_MISSES = 4
# During the vertical approach the overview reading is only a
# starting guess. The wrist camera recentres the cap on the
# gripper target before each short drop.
WRIST_ALIGN_TOLERANCE_PX = 10.0
WRIST_MAX_NUDGE_MM = 14.0
WRIST_NUDGES_PER_STEP = 3
WRIST_SETTLE_SECONDS = 0.25
WRIST_MISS_LIMIT = 3
WRIST_PREVIEW_PERIOD_SECONDS = 0.5
# The arm stops a few millimetres short of the commanded height.
# Closing at the bare table reading presses the pads onto the lid.
# A few millimetres higher lets the jaws close around the wall.
GRASP_CLEARANCE_MM = 3.0
# Calibration already uses the open-finger gap midpoint, so the
# global grasp offset stays zero unless tuned for a new camera view.
GRASP_OFFSET_X_MM = 0.0
GRASP_OFFSET_Y_MM = 0.0
# Seed bias before overview soft-align / grasp nudge.
# white: right finger hit disc centre at 0,0; +12/-8 overshot sticky.
# black: 0,0 closed on lid (jaw 95); +8 Y held (jaw 80). Use +6.
# red: successful pick at 0,0; soft-align cleans residual.
# Soft-align at hover + nudge at grasp close the rest each pick.
CLASS_GRASP_OFFSET_MM = {
    "red_cap": (0.0, 0.0),
    # Soft-align at travel height closes residual; these are seeds.
    "white_cap": (10.0, -6.0),
    # Black often lids at 0/+6; nudge further in +Y after run4 LID streak.
    "black_cap": (0.0, 8.0),
}
# Soft hover align: take most of the measured jaw-to-cap error.
ALIGN_CORRECTION_GAIN = 0.85
# Travel this high before moving sideways. A lower joint-space
# move bows across the table and knocks the cap off the cross.
TRAVEL_Z_MM = 150.0
DESCEND_STEP_MM = 12.0
# Park here before looking and after each pick. Over the caps or the
# drops the arm hides caps from the overview camera.
OBSERVE_X_MM = 130.0
OBSERVE_Y_MM = 130.0
OBSERVE_Z_MM = 200.0
OBSERVE_SETTLE_SECONDS = 1.5
# One settled overview frame after the arm parks. The camera runs
# faster than this; the extra time covers a late frame.
NOTE_LOOK_SECONDS = 2.0
# Pale yellow sticky notes on the white table. Tuned on the live
# overview frame: three notes, each about 95 pixels square.
NOTE_HUE_MIN = 15
NOTE_HUE_MAX = 40
NOTE_SAT_MIN = 40
NOTE_SAT_MAX = 180
NOTE_VAL_MIN = 140
NOTE_MIN_AREA_FRAC = 0.008
NOTE_MAX_AREA_FRAC = 0.08
NOTE_MIN_SIDE_PX = 40
NOTE_ASPECT_MIN = 0.70
NOTE_ASPECT_MAX = 1.45
# A colour fix may pull a loose box onto the disc, but it must not
# walk off the yellow cross onto a cable or another cap.
REFINE_MAX_PIXELS = 18.0
ZERO_POSITION_DEGREES = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ZERO_SPEED = 12
ZERO_STEP_DEGREES = 4.0
ZERO_PERIOD_SECONDS = 0.04
ZERO_LOOKAHEAD_SECONDS = 0.16
ZERO_TRAVEL_RATE = 12.0
ZERO_MIN_TRAVEL_SECONDS = 1.2
ZERO_TOLERANCE_DEGREES = 1.5
ZERO_SETTLE_SECONDS = 12.0
ARUCO_PAGE = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ArUco marker</title>
<style>
body{margin:0;background:#f6f3ec;color:#1c1915;font-family:sans-serif}
main{max-width:640px;margin:0 auto;padding:24px 20px}
img{display:block;max-width:100%;height:auto;border:1px solid #d9d2c5;background:#fff}
a{color:#1f4b3a}
p{line-height:1.45}
</style></head>
<body>
<main>
<h1>Calibration ArUco</h1>
<p>Print this marker (DICT_4X4_50, id 0). Keep a white margin. Tape it flat on the table. Put the gripper pad centre on the marker centre for each spot, then press ArUco-capture on the calibrate page.</p>
<p><a href="/calibrate">Back to calibrate</a> | <a href="/overview">Overview camera</a></p>
<img src="/api/aruco/marker.png" alt="ArUco marker id 0">
<p>Tip: print at about 5-8 cm wide so the overview camera sees it clearly.</p>
</main>
</body></html>
"""
OVERVIEW_PAGE = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Overview camera</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif}
p{margin:8px 12px}
a{color:#9cf}
img{max-width:100%;height:auto;display:block;background:#222}
</style></head>
<body>
<p>Overview camera (live). Coloured rings = caps. Orange cross = ArUco id 0.
<a href="/">Pick page</a> | <a href="/calibrate">Calibrate</a> | <a href="/aruco">Print marker</a></p>
<img id="frame" alt="waiting for overview camera">
<script>
const img = document.getElementById("frame");
function tick(){ img.src = "/api/camera.jpg?t=" + Date.now(); }
img.onload = () => setTimeout(tick, 200);
img.onerror = () => setTimeout(tick, 500);
tick();
</script>
</body></html>
"""
WRIST_PAGE = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Wrist camera</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif}
p{margin:8px 12px}
a{color:#9cf}
img{max-width:100%;height:auto;display:block}
</style></head>
<body>
<p>Wrist camera. Yellow cross = grasp point between the jaws. Green circle = coloured cap (gripper zone ignored).
<a href="/">Pick page</a> | <a href="/overview">Overview camera</a></p>
<img id="frame" alt="waiting for wrist camera">
<script>
const img = document.getElementById("frame");
function tick(){ img.src = "/api/wrist.jpg?t=" + Date.now(); }
img.onload = () => setTimeout(tick, 200);
img.onerror = () => setTimeout(tick, 500);
tick();
</script>
</body></html>
"""
JOINT_NAMES = (
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
)


class PickHttpServer(ThreadingHTTPServer):
    """Serve pick requests without blocking process shutdown."""

    daemon_threads = True
    allow_reuse_address = True


class PickPlace(Node):
    """Turn one camera box into a guarded pick-and-place."""

    def __init__(self):
        """Load the table mapping and open the pick page."""
        super().__init__("mycobot_pick_place")
        self.declare_parameter("ui_host", "0.0.0.0")
        self.declare_parameter("ui_port", 8092)
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("homography_file", "")
        self.declare_parameter("transform_file", "")
        self.declare_parameter("target_class", "red_cap")
        # Overview XY is the reliable pick path. Wrist guidance is
        # opt-in after /wrist shows a real cap lock and jaw crosshair.
        self.declare_parameter("use_wrist_guidance", False)
        self.declare_parameter("grasp_offset_x_mm", GRASP_OFFSET_X_MM)
        self.declare_parameter("grasp_offset_y_mm", GRASP_OFFSET_Y_MM)
        self.declare_parameter("wrist_target_u", -1.0)
        self.declare_parameter("wrist_target_v", -1.0)
        self.declare_parameter("observe_x_mm", OBSERVE_X_MM)
        self.declare_parameter("observe_y_mm", OBSERVE_Y_MM)
        self.declare_parameter("observe_z_mm", OBSERVE_Z_MM)
        self.declare_parameter("note_reader_url", "")

        calibration_file = str(
            self.get_parameter("calibration_file").value
        )
        if not calibration_file:
            calibration_file = _default_calibration_file()
        self._calibration_path = calibration_file
        self._wizard = _load_wizard(calibration_file)
        self._image_bridge = CvBridge()
        self._jpeg = None
        self._jpeg_encoded_at = 0.0
        self._frame = None
        self._frame_received = 0.0
        self._wrist_jpeg = None
        self._wrist_jpeg_encoded_at = 0.0
        self._wrist_frame = None
        self._wrist_frame_received = 0.0
        # One lock per colour, so red, black, and white each keep
        # their own centre. A new colour does not clear the others.
        self._wrist_locks = {}
        self._use_wrist_guidance = bool(
            self.get_parameter("use_wrist_guidance").value
        )
        self._grasp_offset_x_mm = float(
            self.get_parameter("grasp_offset_x_mm").value
        )
        self._grasp_offset_y_mm = float(
            self.get_parameter("grasp_offset_y_mm").value
        )
        self._wrist_target_u = float(
            self.get_parameter("wrist_target_u").value
        )
        self._wrist_target_v = float(
            self.get_parameter("wrist_target_v").value
        )
        self._observe_xyz = (
            float(self.get_parameter("observe_x_mm").value),
            float(self.get_parameter("observe_y_mm").value),
            float(self.get_parameter("observe_z_mm").value),
        )
        self._note_reader_url = str(
            self.get_parameter("note_reader_url").value or ""
        ).strip()
        self._tool_pose = None
        self._tool_pose_received = 0.0
        self._initial_pose = None
        self._calibration = None
        self._mapping = None
        self._mapping_error = ""
        try:
            self._calibration = _load_calibration(calibration_file)
            self._mapping = fit_table(self._calibration["pairs"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._mapping_error = str(error)

        self._preview_model = None
        self._preview_error = ""
        try:
            self._preview_model = load_table_preview(
                _configured_file(
                    self,
                    "homography_file",
                    "table_homography.yaml",
                ),
                _configured_file(
                    self,
                    "transform_file",
                    "table_transform.yaml",
                ),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._preview_error = str(error)
        self._image_width = None
        self._image_height = None
        self._target_class = str(
            self.get_parameter("target_class").value
        ).strip()
        self._lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._joint_lock = threading.Lock()
        self._joint_degrees_live = [None] * 6
        self._zero_stop = threading.Event()
        self._busy = False
        self._weights_ready = False
        self._weights_message = (
            "Trained weights are not loaded. Picking is disabled."
        )
        self._detections = {}
        self._pick_class = ""
        self._server = None
        self._calibrate_page_path = _calibrate_page_path()
        self._attempt_logger = AttemptLogger()
        self._auto_summary = None
        self.get_logger().info(
            "Autonomy safety params: "
            + ", ".join(
                f"{key}={value}" for key, value in SAFETY_PARAMS.items()
            )
        )
        self._warn_spot_residuals()

        # Marking a frame costs longer than the camera period. Keep the
        # image work off the group that carries the pose, the cap boxes
        # and the weights flag, or those callbacks never run.
        image_group = MutuallyExclusiveCallbackGroup()
        state_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Image,
            "/overview_camera/image_rect",
            self._on_image,
            _camera_qos(),
            callback_group=image_group,
        )
        self.create_subscription(
            Image,
            "/wrist_camera/image_raw",
            self._on_wrist_image,
            _camera_qos(),
            callback_group=image_group,
        )
        self.create_subscription(
            Float64MultiArray,
            "/mycobot/tool_pose_raw",
            self._on_tool_pose,
            10,
            callback_group=state_group,
        )
        self.create_subscription(
            JointState,
            "joint_states",
            self._on_joint_states,
            10,
            callback_group=state_group,
        )
        self.create_subscription(
            Bool,
            "/overview_camera/yolo/model_ready",
            self._on_model_ready,
            _ready_qos(),
            callback_group=state_group,
        )
        self.create_subscription(
            CameraInfo,
            "/overview_camera/camera_info",
            self._on_camera_info,
            _camera_qos(),
            callback_group=state_group,
        )
        self.create_subscription(
            Detection2DArray,
            "/overview_camera/yolo/detections",
            self._on_detections,
            10,
            callback_group=state_group,
        )
        self._coords_client = self.create_client(
            MoveCoords,
            "mycobot/move_coords",
        )
        self._joints_client = self.create_client(
            MoveJoints,
            "mycobot/move_joints",
        )
        self._gripper_client = self.create_client(
            SetGripper,
            "mycobot/set_gripper",
        )
        self._stop_client = self.create_client(
            Trigger,
            "mycobot/stop",
        )
        self._release_client = self.create_client(
            Trigger,
            "mycobot/release_servos",
        )
        self._focus_client = self.create_client(
            Trigger,
            "mycobot/focus_servos",
        )

        host = str(self.get_parameter("ui_host").value)
        port = int(self.get_parameter("ui_port").value)
        self._page_path = _page_path()
        try:
            self._server = PickHttpServer((host, port), PickHandler)
        except OSError as error:
            raise RuntimeError(
                f"Could not listen on {host}:{port}: {error}"
            ) from error
        self._server.pick_node = self
        threading.Thread(
            target=self._server.serve_forever,
            name="mycobot_pick_place_http",
            daemon=True,
        ).start()
        self.get_logger().info(f"Pick page: http://{host}:{port}")
        self.get_logger().info(
            "Wrist guidance: "
            + ("on" if self._use_wrist_guidance else "off (overview only)")
        )
        if self._mapping_error:
            self.get_logger().warning(self._mapping_error)
        if self._preview_error:
            self.get_logger().warning(self._preview_error)
        else:
            self.get_logger().info(
                "Table preview is display-only. "
                "It does not command the arm."
            )

    def close_server(self):
        """Stop accepting browser connections."""
        server = self._server
        self._server = None
        if server is None:
            return
        server.shutdown()
        server.server_close()

    def page_bytes(self):
        """Read the pick page from the installed HTML file."""
        with open(self._page_path, "rb") as page_file:
            return page_file.read()

    def state_payload(self):
        """Return calibration status and each cap the camera can see."""
        detections = self._visible_detections()
        detection = None
        for candidate in detections.values():
            if detection is None or candidate["score"] > detection["score"]:
                detection = candidate
        mapped = None
        if detection is not None and self._mapping is not None:
            mapped = project_table(
                self._mapping,
                detection["u"],
                detection["v"],
            )
        age = None if detection is None else detection.get("age_seconds")
        with self._lock:
            weights_ready = self._weights_ready
            weights_message = self._weights_message
        return {
            "calibration_ready": self._mapping is not None,
            "calibration_message": self._mapping_error,
            "target_class": self._target_class,
            "detections": detections,
            "detection": detection,
            "detection_age_seconds": age,
            "mapped_x_mm": None if mapped is None else round(mapped[0], 1),
            "mapped_y_mm": None if mapped is None else round(mapped[1], 1),
            "services_ready": self._services_ready(),
            "weights_ready": weights_ready,
            "weights_message": weights_message,
            "use_wrist_guidance": self._use_wrist_guidance,
            "busy": self._busy,
            "preview": self._preview_payload(detection),
            "wizard": self.wizard_payload(),
            "safety_params": dict(SAFETY_PARAMS),
            "auto_task": getattr(self, "_auto_summary", None),
        }

    def calibrate_page_bytes(self):
        """Read the guided calibration page."""
        with open(self._calibrate_page_path, "rb") as page_file:
            return page_file.read()

    def camera_jpeg(self):
        """Return the newest rectified overview frame."""
        with self._lock:
            jpeg = self._jpeg
        if not jpeg:
            return None
        return jpeg

    def wrist_jpeg(self):
        """Return the newest wrist-camera frame."""
        with self._lock:
            jpeg = self._wrist_jpeg
        if not jpeg:
            return None
        return jpeg

    def note_click(self, pixel_u, pixel_v):
        """Remember a clicked image pixel. This does not move the arm."""
        try:
            pixel_u = float(pixel_u)
            pixel_v = float(pixel_v)
        except (TypeError, ValueError):
            return _error_payload("Click the image spot first.")
        if not math.isfinite(pixel_u) or not math.isfinite(pixel_v):
            return _error_payload("Click coordinates must be finite.")
        if self._image_width and self._image_height:
            if not (
                0.0 <= pixel_u <= self._image_width
                and 0.0 <= pixel_v <= self._image_height
            ):
                return _error_payload(
                    "Click is outside the overview image."
                )
        with self._lock:
            self._wizard["pending_pixel"] = {
                "u": round(pixel_u, 1),
                "v": round(pixel_v, 1),
            }
        return 200, {
            "success": True,
            "message": (
                "Pixel saved. Place the centre between the "
                "gripping pads on that mark, then capture."
            ),
            "wizard": self.wizard_payload(),
        }

    def capture_spot(self):
        """Save the clicked pixel with a fresh tool pose. No motion."""
        with self._lock:
            pending = self._wizard.get("pending_pixel")
        if pending is None:
            return _error_payload(
                "Click the image first, or use Auto-capture gripper "
                "so you do not need to click."
            )
        return self._save_calibration_spot(
            float(pending["u"]),
            float(pending["v"]),
            source="click",
        )

    def auto_capture_spot(self):
        """Save the gripper pad centre seen by the overview camera."""
        frame = self._latest_overview_frame()
        if frame is None:
            return _error_payload(
                "No overview image yet. Wait for the camera."
            )
        center = overview_pad_center(frame)
        if center is None:
            return _error_payload(
                "Could not see the open gripper pads in the overview. "
                "Prefer ArUco capture instead."
            )
        return self._save_calibration_spot(
            float(center[0]),
            float(center[1]),
            source="auto",
        )

    def aruco_capture_spot(self):
        """Save the ArUco marker centre with the live tool pose.

        Tape DICT_4X4_50 id 0 on the table. Put the pad centre on the
        marker centre, Hold, then call this. Move the paper to a new
        place for each spot.
        """
        frame = self._latest_overview_frame()
        if frame is None:
            return _error_payload(
                "No overview image yet. Wait for the camera."
            )
        detected = detect_overview_aruco(
            frame,
            marker_id=ARUCO_MARKER_ID,
        )
        if detected is None:
            return _error_payload(
                "ArUco marker id "
                f"{ARUCO_MARKER_ID} not visible. Print it from "
                "/aruco, tape it flat on the table, keep it fully "
                "in view, and try again."
            )
        center_u, center_v, marker_id = detected
        return self._save_calibration_spot(
            float(center_u),
            float(center_v),
            source=f"aruco_{marker_id}",
        )

    def aruco_marker_png(self):
        """Return a printable PNG of the calibration ArUco marker."""
        return _aruco_marker_png(
            marker_id=ARUCO_MARKER_ID,
            size_px=ARUCO_PRINT_SIZE_PX,
            border_px=ARUCO_PRINT_BORDER_PX,
        )

    def _latest_overview_frame(self):
        """Return a fresh overview frame, or the latest cached copy."""
        frame = self._wait_for_frame()
        if frame is not None:
            return frame
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def clear_calibration_spots(self):
        """Forget fit and check spots so calibration can restart."""
        with self._lock:
            self._wizard["pairs"] = []
            self._wizard["check"] = None
            self._wizard["pending_pixel"] = None
            self._wizard["accepted"] = False
            self._store_wizard(accepted=False)
            self._calibration = None
            self._mapping = None
            self._mapping_error = (
                "Calibration has not been accepted. Picking is disabled."
            )
        return 200, {
            "success": True,
            "message": "All calibration spots cleared.",
            "wizard": self.wizard_payload(),
        }

    def _save_calibration_spot(self, pixel_u, pixel_v, source):
        """Store one fit or check spot from a pixel and live pose."""
        with self._lock:
            pair_count = len(self._wizard["pairs"])
            has_check = self._wizard.get("check") is not None
        if pair_count >= FIT_SPOT_COUNT and has_check:
            return _error_payload(
                f"{FIT_SPOT_COUNT} fit spots and the check spot "
                "are already saved. Remove one or Clear spots."
            )
        pose = self._fresh_tool_pose()
        if pose is None:
            return _error_payload(
                "No fresh /mycobot/tool_pose_raw reading. "
                "Wait until the pose age is under one second."
            )
        spot = {
            "u": round(float(pixel_u), 1),
            "v": round(float(pixel_v), 1),
            "x": round(pose[0], 2),
            "y": round(pose[1], 2),
            "z": round(pose[2], 2),
            "source": source,
        }
        with self._lock:
            existing = list(self._wizard["pairs"])
            if self._wizard.get("check"):
                existing.append(self._wizard["check"])
            too_close = _nearby_spot(existing, spot)
            if too_close is not None:
                return _error_payload(
                    "This spot is too close to spot "
                    f"{too_close}. Move farther across the table "
                    "in both left-right and forward-back."
                )
            if pair_count < FIT_SPOT_COUNT:
                self._wizard["pairs"].append(spot)
                message = (
                    f"Fit spot {pair_count + 1} of {FIT_SPOT_COUNT} "
                    f"saved ({source}). "
                    "Spread the next ones in both image directions."
                )
            else:
                self._wizard["check"] = spot
                message = (
                    f"Check spot saved ({source}). "
                    "It is not used in the fit."
                )
            self._wizard["pending_pixel"] = None
            self._wizard["accepted"] = False
            self._store_wizard(accepted=False)
            self._calibration = None
            self._mapping = None
            self._mapping_error = (
                "Calibration has not been accepted. Picking is disabled."
            )
        return 200, {
            "success": True,
            "message": message,
            "wizard": self.wizard_payload(),
        }

    def capture_drop(self, class_name=None):
        """Save the current tool XY as shared or per-colour drop. No motion."""
        pose = self._fresh_tool_pose()
        if pose is None:
            return _error_payload(
                "No fresh /mycobot/tool_pose_raw reading."
            )
        chosen = None
        if class_name:
            chosen = _cap_class(class_name)
            if chosen is None:
                return _error_payload(
                    "Drop colour must be red_cap, white_cap, or black_cap."
                )
        with self._lock:
            if chosen is None:
                self._wizard["drop_x_mm"] = round(pose[0], 2)
                self._wizard["drop_y_mm"] = round(pose[1], 2)
                message = (
                    "Shared drop position saved from the current tool pose. "
                    "The arm was not moved."
                )
            else:
                x_key, y_key = COLOUR_DROP_KEYS[chosen]
                self._wizard[x_key] = round(pose[0], 2)
                self._wizard[y_key] = round(pose[1], 2)
                label = CAP_CLASSES[chosen]
                message = (
                    f"{label.capitalize()} drop saved at "
                    f"{pose[0]:.1f}, {pose[1]:.1f} mm. "
                    "The arm was not moved."
                )
            self._wizard["accepted"] = False
            self._store_wizard(accepted=False)
            self._calibration = None
            self._mapping = None
            self._mapping_error = (
                "Calibration has not been accepted. Picking is disabled."
            )
        return 200, {
            "success": True,
            "message": message,
            "wizard": self.wizard_payload(),
        }

    def save_clearances(self, body):
        """Store the typed table height and hover clearance."""
        try:
            table_z = float(body["table_z_mm"])
            hover = float(body["hover_mm"])
        except (KeyError, TypeError, ValueError):
            return _error_payload(
                "Enter table height and hover clearance in millimetres."
            )
        if not math.isfinite(table_z) or not math.isfinite(hover):
            return _error_payload("Clearances must be finite numbers.")
        if hover <= 0.0:
            return _error_payload(
                "Hover clearance must be greater than 0 mm."
            )
        with self._lock:
            self._wizard["table_z_mm"] = round(table_z, 2)
            self._wizard["hover_mm"] = round(hover, 2)
            self._wizard["accepted"] = False
            self._store_wizard(accepted=False)
            self._calibration = None
            self._mapping = None
            self._mapping_error = (
                "Calibration has not been accepted. Picking is disabled."
            )
        return 200, {
            "success": True,
            "message": "Table height and hover clearance saved.",
            "wizard": self.wizard_payload(),
        }

    def undo_spot(self):
        """Remove the latest recorded spot. This does not move the arm."""
        with self._lock:
            if self._wizard.get("check") is not None:
                self._wizard["check"] = None
            elif self._wizard["pairs"]:
                self._wizard["pairs"].pop()
            else:
                return _error_payload("No spot to remove.")
            self._wizard["accepted"] = False
            self._store_wizard()
            self._calibration = None
            self._mapping = None
            self._mapping_error = (
                "Calibration has not been accepted. Picking is disabled."
            )
        return 200, {
            "success": True,
            "message": "Latest spot removed.",
            "wizard": self.wizard_payload(),
        }

    def accept_calibration(self):
        """Lock in the measured spots and enable picking."""
        with self._lock:
            wizard = dict(self._wizard)
        try:
            calibration = _calibration_from_wizard(wizard)
            mapping = fit_table(calibration["pairs"])
        except (ValueError, KeyError, TypeError) as error:
            return _error_payload(str(error))
        report = _check_report(wizard)
        if report is None:
            return _error_payload(
                "Need four fit spots, a check spot, table height, "
                "hover clearance, and a drop position."
            )
        span = _calibration_span(wizard.get("pairs") or [])
        if (
            span["span_u"] < MIN_CALIB_SPAN_U_PX
            or span["span_v"] < MIN_CALIB_SPAN_V_PX
        ):
            return _error_payload(
                "Fit spots are too close together "
                f"(span {span['span_u']:.0f}×{span['span_v']:.0f} px). "
                f"Need at least {MIN_CALIB_SPAN_U_PX:.0f}×"
                f"{MIN_CALIB_SPAN_V_PX:.0f} px across the table. "
                "Clear spots and spread them in both directions."
            )
        error_mm = float(report.get("error_mm") or 999.0)
        if error_mm > MAX_ACCEPT_CHECK_ERROR_MM:
            return _error_payload(
                "Check error is "
                f"{error_mm:.1f} mm. Remeasure spots before accepting "
                f"(limit {MAX_ACCEPT_CHECK_ERROR_MM:.0f} mm)."
            )
        residuals = pair_fit_residuals(
            calibration["pairs"],
            lambda u, v: project_table(mapping, u, v),
        )
        table_z = float(calibration["table_z_mm"])
        for row in residuals:
            if row["error_mm"] > MAX_FIT_SPOT_RESIDUAL_MM:
                return _error_payload(
                    f"Fit spot {row['index']} residual "
                    f"{row['error_mm']:.1f} mm exceeds "
                    f"{MAX_FIT_SPOT_RESIDUAL_MM:.0f} mm. "
                    f"Remake that spot at table z ≈ {table_z:.0f} mm "
                    "(open fingers on the mark), then Accept again."
                )
            z_val = row.get("z_mm")
            if z_val is not None and abs(float(z_val) - table_z) > MAX_FIT_SPOT_Z_DELTA_MM:
                return _error_payload(
                    f"Fit spot {row['index']} z={float(z_val):.1f} mm "
                    f"is more than {MAX_FIT_SPOT_Z_DELTA_MM:.0f} mm from "
                    f"table_z {table_z:.0f} mm. Remake that spot tip-down "
                    "on the table before accepting."
                )
        with self._lock:
            self._wizard["accepted"] = True
            self._wizard.update(
                {
                    "table_z_mm": calibration["table_z_mm"],
                    "hover_mm": calibration["hover_mm"],
                    "drop_x_mm": calibration["drop_x_mm"],
                    "drop_y_mm": calibration["drop_y_mm"],
                    "drop_red_x_mm": calibration.get("drop_red_x_mm"),
                    "drop_red_y_mm": calibration.get("drop_red_y_mm"),
                    "drop_black_x_mm": calibration.get("drop_black_x_mm"),
                    "drop_black_y_mm": calibration.get("drop_black_y_mm"),
                    "drop_white_x_mm": calibration.get("drop_white_x_mm"),
                    "drop_white_y_mm": calibration.get("drop_white_y_mm"),
                    "tool_rx": calibration["tool_rx"],
                    "tool_ry": calibration["tool_ry"],
                    "tool_rz": calibration["tool_rz"],
                    "speed": calibration["speed"],
                }
            )
            self._store_wizard(accepted=True)
            self._calibration = calibration
            self._mapping = mapping
            self._mapping_error = ""
        self.get_logger().info(
            "Calibration accepted. Check error "
            f"{error_mm:.2f} mm. Picking is enabled."
        )
        return 200, {
            "success": True,
            "message": (
                "Calibration accepted. Check error "
                f"{error_mm:.1f} mm. Picking is enabled."
            ),
            "wizard": self.wizard_payload(),
        }

    def wizard_payload(self):
        """Return the guided calibration state and the display-only target."""
        with self._lock:
            wizard = {
                "accepted": bool(self._wizard.get("accepted")),
                "pending_pixel": self._wizard.get("pending_pixel"),
                "pairs": list(self._wizard["pairs"]),
                "check": self._wizard.get("check"),
                "table_z_mm": self._wizard.get("table_z_mm"),
                "hover_mm": self._wizard.get("hover_mm"),
                "drop_x_mm": self._wizard.get("drop_x_mm"),
                "drop_y_mm": self._wizard.get("drop_y_mm"),
                "drop_red_x_mm": self._wizard.get("drop_red_x_mm"),
                "drop_red_y_mm": self._wizard.get("drop_red_y_mm"),
                "drop_black_x_mm": self._wizard.get("drop_black_x_mm"),
                "drop_black_y_mm": self._wizard.get("drop_black_y_mm"),
                "drop_white_x_mm": self._wizard.get("drop_white_x_mm"),
                "drop_white_y_mm": self._wizard.get("drop_white_y_mm"),
            }
            pose = None if self._tool_pose is None else list(self._tool_pose)
            pose_age = None
            if self._tool_pose is not None:
                pose_age = round(
                    time.monotonic() - self._tool_pose_received,
                    3,
                )
        report = _check_report(wizard)
        span = _calibration_span(wizard["pairs"])
        return {
            "accepted": wizard["accepted"],
            "motion_command": False,
            "fit_required": FIT_SPOT_COUNT,
            "pending_pixel": wizard["pending_pixel"],
            "pairs": wizard["pairs"],
            "check": wizard["check"],
            "span_u_px": round(span["span_u"], 1),
            "span_v_px": round(span["span_v"], 1),
            "table_z_mm": wizard["table_z_mm"],
            "hover_mm": wizard["hover_mm"],
            "drop_x_mm": wizard["drop_x_mm"],
            "drop_y_mm": wizard["drop_y_mm"],
            "drop_red_x_mm": wizard.get("drop_red_x_mm"),
            "drop_red_y_mm": wizard.get("drop_red_y_mm"),
            "drop_black_x_mm": wizard.get("drop_black_x_mm"),
            "drop_black_y_mm": wizard.get("drop_black_y_mm"),
            "drop_white_x_mm": wizard.get("drop_white_x_mm"),
            "drop_white_y_mm": wizard.get("drop_white_y_mm"),
            "tool_pose_mm": None if pose is None else [
                round(pose[0], 2),
                round(pose[1], 2),
                round(pose[2], 2),
            ],
            "tool_pose_age_seconds": pose_age,
            "report": report,
        }

    def pick(self, class_name):
        """Pick the cap color chosen on the page."""
        chosen = _cap_class(class_name)
        if chosen is None:
            return _error_payload("Choose the red, white, or black cap.")
        label = CAP_CLASSES[chosen]
        if not self._wizard.get("accepted"):
            return _error_payload(
                "Calibration is still a preview. Picking is disabled."
            )
        if not self._weights_ready:
            return _error_payload(self._weights_message)
        if self._mapping is None or self._calibration is None:
            return _error_payload(self._mapping_error)
        if not self._motion_lock.acquire(blocking=False):
            return _error_payload("A pick is already running", 409)
        self._busy = True
        self._auto_summary = None
        try:
            initial = self._fresh_tool_pose()
            if initial is None:
                return _error_payload(
                    "No fresh tool pose. Wait until the arm "
                    "is reporting its position."
                )
            status, body = self._go_observe()
            if status != 200 or not body.get("success"):
                return status, body
            self._initial_pose = self._observe_pose()
            if self._wait_for_detection(chosen) is None:
                return _error_payload(
                    f"The {label} cap is not in view from the parking pose."
                )
            self._pick_class = chosen
            track = CapTrack(class_name=chosen)
            status, body = self._run_pick_with_retries(track)
            if status == 200 and body.get("success"):
                self._go_observe()
                return status, body
            if _stopped_by_user(body):
                return status, body
            return self._fail_and_return(status, body)
        finally:
            self._busy = False
            self._motion_lock.release()

    def pick_all(self):
        """Pick every locked colour once (bounded retries per colour)."""
        if not self._wizard.get("accepted"):
            return _error_payload(
                "Calibration is still a preview. Picking is disabled."
            )
        if not self._weights_ready:
            return _error_payload(self._weights_message)
        if self._mapping is None or self._calibration is None:
            return _error_payload(self._mapping_error)
        if not self._motion_lock.acquire(blocking=False):
            return _error_payload("A pick is already running", 409)
        self._busy = True
        manager = TaskManager()
        self._auto_summary = manager.summary()
        try:
            initial = self._fresh_tool_pose()
            if initial is None:
                return _error_payload(
                    "No fresh tool pose. Wait until the arm "
                    "is reporting its position."
                )
            status, body = self._go_observe()
            if status != 200 or not body.get("success"):
                return status, body
            self._initial_pose = self._observe_pose()
            reports = []
            idle_passes = 0
            while True:
                detections = self._detections_with_xy()
                manager.update_from_detections(detections)
                self._auto_summary = manager.summary()
                track = manager.select_next()
                if track is None:
                    break
                self._pick_class = track.class_name
                label = CAP_CLASSES.get(track.class_name, track.class_name)
                self.get_logger().info(
                    f"Auto pick selecting {label} "
                    f"(attempt {track.attempts + 1})"
                )
                before_attempts = int(track.attempts)
                status, body = self._run_pick_with_retries(track, manager)
                if len(reports) < 40:
                    reports.append(
                        {
                            "class_name": track.class_name,
                            "success": bool(body.get("success")),
                            "message": body.get("message"),
                            "status": track.status,
                            "attempts": track.attempts,
                            "last_failure": track.last_failure,
                        }
                    )
                if _stopped_by_user(body):
                    manager.finished = True
                    manager.message = body.get("message", "Stopped.")
                    break
                self._go_observe()
                if int(track.attempts) <= before_attempts and not body.get(
                    "success"
                ):
                    idle_passes += 1
                    if idle_passes >= 6:
                        manager.finished = True
                        manager.message = (
                            "Auto pick stopped: no grasp progress "
                            "(caps not visible or unworkable)."
                        )
                        break
                else:
                    idle_passes = 0
                self._auto_summary = manager.summary()
            summary = manager.summary()
            self._auto_summary = summary
            completed = sum(
                1
                for track in manager.tracks.values()
                if track.status == "COMPLETED"
            )
            message = summary.get("message") or (
                f"Auto pick finished. Completed {completed} colour(s)."
            )
            return 200, {
                "success": completed > 0,
                "message": message,
                "completed": completed,
                "reports": reports,
                "auto_task": summary,
            }
        finally:
            self._busy = False
            self._motion_lock.release()

    def _detections_with_xy(self):
        """Return fresh detections with projected robot XY."""
        found = {}
        for name, detection in self._visible_detections().items():
            row = dict(detection)
            if self._mapping is not None:
                x_mm, y_mm = project_table(
                    self._mapping,
                    float(row["u"]),
                    float(row["v"]),
                )
                row["x_mm"] = x_mm
                row["y_mm"] = y_mm
            found[name] = row
        return found

    def _run_pick_with_retries(self, track, manager=None):
        """Bounded adaptive retries for one colour."""
        label = CAP_CLASSES.get(track.class_name, "chosen")
        last_status, last_body = _error_payload(
            f"No attempt ran for the {label} cap."
        )
        candidate_delta = (0.0, 0.0)
        while track.attempts < MAX_ATTEMPTS_PER_CAP:
            if manager is not None:
                limit = manager.limits_hit()
                if limit:
                    manager.finished = True
                    manager.message = limit
                    return _error_payload(limit)
            live = self._fresh_detection(track.class_name)
            if live is None:
                # Drop stale XY so select_next will not re-pick this colour
                # until the overview locks it again.
                track.status = "PENDING"
                track.x_mm = None
                track.y_mm = None
                track.last_message = "Cap not visible for retry."
                track.last_failure = "UNKNOWN"
                return _error_payload(
                    f"The {label} cap is not in view for another try."
                )
            age = float(live.get("age_seconds") or 0.0)
            if age > float(SAFETY_PARAMS["stale_detection_seconds"]):
                return _error_payload(
                    f"Stale {label} lock ({age:.1f} s). Refusing pick."
                )
            if manager is not None:
                manager.mark_attempt(track)
            else:
                track.attempts += 1
            track.status = "APPROACHING"
            status, body = self._run_pick_stages(
                track,
                candidate_delta,
            )
            last_status, last_body = status, body
            if _stopped_by_user(body):
                return status, body
            outcome = body.get("grasp_outcome") or (
                "SUCCESS" if body.get("success") else "UNKNOWN"
            )
            self._attempt_logger.log(
                {
                    "class_name": track.class_name,
                    "attempt": track.attempts,
                    "outcome": outcome,
                    "candidate_delta_mm": list(candidate_delta),
                    "x_mm": body.get("grasp_x_mm"),
                    "y_mm": body.get("grasp_y_mm"),
                    "jaw": body.get("final_value"),
                    "message": body.get("message"),
                    "place_verify": body.get("place_verify"),
                }
            )
            if outcome == "SUCCESS" and body.get("success"):
                if manager is not None:
                    manager.mark_success(track, body.get("message", ""))
                else:
                    track.status = "COMPLETED"
                return status, body
            # Grasp held but place failed: do not burn XY candidates.
            if body.get("grasp_held") and not body.get("success"):
                track.last_failure = "PLACE_FAILURE"
                track.last_message = str(body.get("message") or "Place failed")
                if manager is not None:
                    manager.mark_failure(
                        track, "PLACE_FAILURE", track.last_message
                    )
                else:
                    track.status = "FAILED"
                return status, body
            grasp_x = body.get("grasp_x_mm")
            grasp_y = body.get("grasp_y_mm")
            if grasp_x is not None and grasp_y is not None:
                remember_failed_xy(track, grasp_x, grasp_y)
            track.last_failure = outcome
            track.last_message = str(body.get("message") or outcome)
            if manager is not None:
                manager.mark_failure(track, outcome, track.last_message)
            else:
                if track.attempts >= MAX_ATTEMPTS_PER_CAP:
                    track.status = "SKIPPED"
                else:
                    track.status = "RETRY"
            # Retreat / open, then park so the arm does not hide the cap.
            self._retreat_after_fail()
            self._go_observe()
            fresh = self._wait_for_detection(track.class_name)
            if fresh is None:
                last_body["message"] = (
                    track.last_message
                    + f" Cap left view after attempt {track.attempts}."
                )
                break
            # Cap moved: use new detection only (no candidate shift).
            if track.u is not None and track.v is not None:
                if not pixel_near(track.u, track.v, fresh["u"], fresh["v"]):
                    candidate_delta = (0.0, 0.0)
                    track.u = float(fresh["u"])
                    track.v = float(fresh["v"])
                    self.get_logger().info(
                        f"{label} moved on the table; "
                        "retrying from the new lock."
                    )
                    continue
            track.u = float(fresh["u"])
            track.v = float(fresh["v"])
            nxt = next_candidate(track, outcome)
            if nxt is None:
                last_body["message"] = (
                    track.last_message
                    + " No unused XY candidates left."
                )
                break
            candidate_delta = nxt
            self.get_logger().info(
                f"Retry {label} with XY delta "
                f"{candidate_delta[0]:+.1f}, {candidate_delta[1]:+.1f} mm "
                f"after {outcome}"
            )
        return last_status, last_body

    def _retreat_after_fail(self):
        """Open jaws and lift before the next observe/retry."""
        if self._calibration is None:
            self._set_gripper(0)
            return
        orientation = (
            self._calibration["tool_rx"],
            self._calibration["tool_ry"],
            self._calibration["tool_rz"],
        )
        speed = self._calibration["speed"]
        self._set_gripper(0)
        pose = self._fresh_tool_pose()
        if pose is not None and pose[2] < TRAVEL_Z_MM - 5.0:
            self._move_to(
                (pose[0], pose[1], TRAVEL_Z_MM),
                orientation,
                speed,
            )

    def _run_pick_stages(self, track, candidate_delta=(0.0, 0.0)):
        """Approach, align, descend, close, lift, place for one attempt."""
        self._pick_class = track.class_name
        label = CAP_CLASSES.get(track.class_name, "chosen")
        center = self._stable_cap_center(track.class_name)
        if center is None:
            return _error_payload(
                f"No steady view of the {label} cap. "
                "Keep the arm out of the picture."
            )
        table_z = self._calibration["table_z_mm"]
        grasp_z = table_z + GRASP_CLEARANCE_MM
        orientation = (
            self._calibration["tool_rx"],
            self._calibration["tool_ry"],
            self._calibration["tool_rz"],
        )
        speed = self._calibration["speed"]
        x_mm, y_mm = center[0], center[1]
        aim_u, aim_v = float(center[2]), float(center[3])
        track.u, track.v = aim_u, aim_v
        seed = self._fresh_detection(track.class_name) or {}
        aim_w = float(seed.get("size_x") or 36.0)
        aim_h = float(seed.get("size_y") or 36.0)
        offset_x, offset_y = self._grasp_offsets(track.class_name)
        x_mm += offset_x + float(candidate_delta[0])
        y_mm += offset_y + float(candidate_delta[1])
        if not in_workspace(x_mm, y_mm):
            return _error_payload(
                f"Target {x_mm:.1f}, {y_mm:.1f} mm is outside "
                "the workspace box."
            )
        if xy_already_failed(track, x_mm, y_mm):
            return _error_payload(
                f"Refusing to repeat failed XY "
                f"{x_mm:.1f}, {y_mm:.1f} mm."
            )

        # --- approach ---
        track.status = "APPROACHING"
        status, body = self._stage_approach(orientation, speed, x_mm, y_mm)
        if status != 200 or not body.get("success"):
            body["grasp_outcome"] = "UNKNOWN"
            body["grasp_x_mm"] = round(x_mm, 1)
            body["grasp_y_mm"] = round(y_mm, 1)
            return status, body
        x_mm, y_mm = self._confirm_cap_target(
            x_mm, y_mm, TRAVEL_Z_MM, orientation, speed,
        )
        live = self._fresh_detection(track.class_name)
        if live is not None:
            age = float(live.get("age_seconds") or 0.0)
            if age > float(SAFETY_PARAMS["stale_detection_seconds"]):
                return _error_payload(
                    f"Stale lock after approach ({age:.1f} s)."
                )
            aim_u = float(live["u"])
            aim_v = float(live["v"])
            aim_w = float(live.get("size_x") or aim_w)
            aim_h = float(live.get("size_y") or aim_h)
            track.u, track.v = aim_u, aim_v

        # --- align ---
        track.status = "ALIGNING"
        x_mm, y_mm = self._soft_align_at_hover(
            x_mm, y_mm, TRAVEL_Z_MM, orientation,
            aim_u, aim_v, aim_w, aim_h,
        )
        if not in_workspace(x_mm, y_mm):
            return _error_payload(
                f"Aligned target left the workspace "
                f"({x_mm:.1f}, {y_mm:.1f} mm)."
            )

        # --- descend ---
        track.status = "GRASPING"
        wrist_guided = False
        if self._use_wrist_guidance:
            status, body, x_mm, y_mm = self._wrist_guided_descend(
                x_mm,
                y_mm,
                TRAVEL_Z_MM,
                grasp_z,
                orientation,
                track.class_name,
            )
            wrist_guided = bool(body.get("wrist_guided"))
        else:
            status, body = self._perform(
                "Lower straight around the cap",
                (x_mm, y_mm, grasp_z),
                None,
                orientation,
                DESCEND_SPEED,
            )
        if status != 200 or not body.get("success"):
            body["grasp_outcome"] = "UNKNOWN"
            body["grasp_x_mm"] = round(x_mm, 1)
            body["grasp_y_mm"] = round(y_mm, 1)
            return status, body
        if not wrist_guided:
            x_mm, y_mm = self._nudge_at_grasp(
                x_mm, y_mm, aim_u, aim_v, grasp_z, orientation, aim_w, aim_h,
            )

        # --- close ---
        status, body = self._perform(
            "Close gripper",
            None,
            1,
            orientation,
            speed,
            CLOSE_GRIPPER_SPEED,
        )
        self.get_logger().info(
            "Close finished with jaw "
            f"{body.get('final_value')}: {body.get('message')}"
        )
        outcome = classify_grasp(body.get("final_value"))
        body["grasp_outcome"] = outcome
        body["grasp_x_mm"] = round(x_mm, 1)
        body["grasp_y_mm"] = round(y_mm, 1)
        if outcome != "SUCCESS":
            body["success"] = False
            body["message"] = (
                f"Grasp {outcome} (jaw {body.get('final_value')}). "
                + str(body.get("message", ""))
            ).strip()
            return status, body

        # --- lift ---
        status, body = self._perform(
            "Lift the object",
            (x_mm, y_mm, TRAVEL_Z_MM),
            None,
            orientation,
            speed,
        )
        if status != 200 or not body.get("success"):
            body["grasp_outcome"] = "UNKNOWN"
            body["grasp_x_mm"] = round(x_mm, 1)
            body["grasp_y_mm"] = round(y_mm, 1)
            return status, body

        # Jaw can report a hold while the disc never left the table.
        # Check overview before spending a place cycle.
        left = self._verify_cap_left_grasp(
            track.class_name, aim_u, aim_v,
        )
        if left == "FAILURE":
            body["success"] = False
            body["grasp_outcome"] = "FAILURE_EMPTY"
            body["message"] = (
                "Grasp looked held but the cap is still at the pick spot."
            )
            body["grasp_x_mm"] = round(x_mm, 1)
            body["grasp_y_mm"] = round(y_mm, 1)
            body["final_value"] = body.get("final_value")
            return status, body

        # --- place + verify ---
        track.status = "PLACING"
        status, body = self._place_cap(
            track.class_name,
            grasp_z,
            orientation,
            speed,
            grasp_u=aim_u,
            grasp_v=aim_v,
        )
        body["grasp_held"] = True
        body["grasp_outcome"] = (
            "SUCCESS" if body.get("success") else "PLACE_FAILURE"
        )
        body["grasp_x_mm"] = round(x_mm, 1)
        body["grasp_y_mm"] = round(y_mm, 1)
        return status, body

    def _stage_approach(self, orientation, speed, x_mm, y_mm):
        """Raise, open, and move above the aim XY."""
        status, body = self._raise_for_travel(orientation, speed)
        if status != 200 or not body.get("success"):
            return status, body
        status, body = self._perform(
            "Open gripper", None, 0, orientation, speed,
        )
        if status != 200 or not body.get("success"):
            return status, body
        return self._perform(
            "Move above the cap",
            (x_mm, y_mm, TRAVEL_Z_MM),
            None,
            orientation,
            speed,
        )

    def _drop_xy_for_class(self, class_name):
        """Saved colour spot. Placement reads the sticky note instead."""
        keys = COLOUR_DROP_KEYS.get(class_name)
        calib = self._calibration or {}
        wizard = self._wizard or {}
        if keys is not None:
            x_key, y_key = keys
            x_val = calib.get(x_key, wizard.get(x_key))
            y_val = calib.get(y_key, wizard.get(y_key))
            if x_val is not None and y_val is not None:
                return self._clamp_drop_xy(float(x_val), float(y_val))
        return self._clamp_drop_xy(
            float(calib["drop_x_mm"]), float(calib["drop_y_mm"])
        )

    def _clamp_drop_xy(self, x_mm, y_mm):
        """Keep drop poses inside the firmware-safe workspace box."""
        max_x = float(SAFETY_PARAMS["coord_max_x_mm"])
        min_x = float(SAFETY_PARAMS["coord_min_x_mm"])
        max_y = float(SAFETY_PARAMS["coord_max_y_mm"])
        min_y = float(SAFETY_PARAMS["coord_min_y_mm"])
        return (
            min(max(float(x_mm), min_x), max_x),
            min(max(float(y_mm), min_y), max_y),
        )

    def _note_reader(self):
        """Return a laptop reader, or None to keep the local reader."""
        url = self._note_reader_url
        if not url:
            return None

        def reader(crop, _prepared):
            text = vlm_read_note(crop, url)
            self.get_logger().info(f"Laptop read a note as {text.strip()!r}.")
            return text

        return reader

    def _place_cap(
        self,
        class_name,
        grasp_z,
        orientation,
        speed,
        grasp_u=None,
        grasp_v=None,
    ):
        """Park clear, read the matching sticky note, then drop there."""
        status, body = self._go_observe()
        if status != 200 or not body.get("success"):
            return status, body
        frame = self._wait_for_frame(NOTE_LOOK_SECONDS)
        if frame is None:
            return _error_payload(
                "No overview image for the sticky notes. "
                "The cap was not dropped."
            )
        if self._mapping is None:
            return _error_payload(
                self._mapping_error or "Table mapping is not ready."
            )
        note_uv, error = match_cap_note(
            frame, class_name, reader=self._note_reader(),
        )
        if note_uv is None:
            self.get_logger().warning(error)
            return _error_payload(error)
        raw_x, raw_y = project_table(
            self._mapping, note_uv[0], note_uv[1],
        )
        label = CAP_CLASSES.get(class_name, "matching")
        if not in_workspace(raw_x, raw_y):
            return _error_payload(
                f"The {label} note at {raw_x:.0f}, {raw_y:.0f} mm "
                "is outside the arm's reach. The cap was not dropped."
            )
        drop_x, drop_y = self._clamp_drop_xy(raw_x, raw_y)
        self.get_logger().info(
            f"Sticky note for {label} at "
            f"{drop_x:.1f}, {drop_y:.1f} mm "
            f"(pixel {note_uv[0]:.0f}, {note_uv[1]:.0f})."
        )
        steps = (
            (
                "Move above the drop",
                (drop_x, drop_y, TRAVEL_Z_MM),
                None,
                speed,
            ),
            (
                "Lower to the drop",
                (drop_x, drop_y, grasp_z),
                None,
                DESCEND_SPEED,
            ),
            ("Open gripper", None, 0, speed),
            ("Lift clear", (drop_x, drop_y, TRAVEL_Z_MM), None, speed),
        )
        open_jaw = None
        last_body = {"success": True, "message": "Placed"}
        for name, point, gripper_state, step_speed in steps:
            status, body = self._perform(
                name, point, gripper_state, orientation, step_speed,
            )
            last_body = body
            if name == "Open gripper":
                open_jaw = body.get("final_value")
            if status != 200 or not body.get("success"):
                return status, body
        # Placement verify: jaws should read empty after open.
        verify = self._verify_place_empty(open_jaw)
        if verify == "FAILURE":
            return _error_payload(
                "Place verify failed: jaws still look occupied."
            )
        # Overview check: colour must not still sit at the grasp pixel.
        left = self._verify_cap_left_grasp(
            class_name, grasp_u, grasp_v,
        )
        if left == "FAILURE":
            return _error_payload(
                "Place verify failed: cap still near the pick spot."
            )
        place_verify = verify
        if left == "UNKNOWN" and verify == "EMPTY":
            place_verify = "EMPTY"
        elif left == "UNKNOWN":
            place_verify = "UNKNOWN" if verify == "UNKNOWN" else verify
        elif verify == "UNKNOWN":
            place_verify = "LEFT_PICK"
        message = "Pick and place completed"
        if place_verify == "UNKNOWN":
            message += " (place verify UNKNOWN; jaw not conclusive)."
        return 200, {
            "success": True,
            "message": message,
            "place_verify": place_verify,
            "final_value": open_jaw,
        }

    def _verify_place_empty(self, open_jaw=None):
        """Return EMPTY, FAILURE, or UNKNOWN after the drop open."""
        value = open_jaw
        if not isinstance(value, int) or value < 0:
            _status, body = self._set_gripper(0)
            value = body.get("final_value")
        if not isinstance(value, int) or value < 0:
            return "UNKNOWN"
        if value <= 10 or value >= 90:
            return "EMPTY"
        if 12 <= value <= 86:
            return "FAILURE"
        return "UNKNOWN"

    def _verify_cap_left_grasp(self, class_name, grasp_u, grasp_v):
        """Confirm the colour is no longer at the grasp pixel."""
        if grasp_u is None or grasp_v is None:
            return "UNKNOWN"
        time.sleep(0.45)
        live = self._fresh_detection(class_name)
        if live is None:
            return "LEFT"
        if pixel_near(grasp_u, grasp_v, live["u"], live["v"]):
            return "FAILURE"
        return "LEFT"

    def _warn_spot_residuals(self):
        """Log fit-spot residuals so spot 4 remakes are obvious."""
        pairs = list((self._wizard or {}).get("pairs") or [])
        if len(pairs) < 3 or self._mapping is None:
            return
        rows = pair_fit_residuals(
            pairs,
            lambda u, v: project_table(self._mapping, u, v),
        )
        table_z = self._wizard.get("table_z_mm")
        for row in rows:
            msg = (
                f"Fit spot {row['index']} residual "
                f"{row['error_mm']:.1f} mm"
            )
            z_val = row.get("z_mm")
            if table_z is not None and z_val is not None:
                delta = abs(float(z_val) - float(table_z))
                msg += f", z delta {delta:.1f} mm"
            if row["error_mm"] > MAX_FIT_SPOT_RESIDUAL_MM:
                self.get_logger().warn(
                    msg
                    + f". Remake at table z ≈ {table_z} mm "
                    f"(limit {MAX_FIT_SPOT_RESIDUAL_MM:.0f} mm)."
                )
            else:
                self.get_logger().info(msg)

    def _observe_pose(self):
        """Return the parking pose with the calibrated tool angle."""
        return [
            self._observe_xyz[0],
            self._observe_xyz[1],
            self._observe_xyz[2],
            float(self._calibration["tool_rx"]),
            float(self._calibration["tool_ry"]),
            float(self._calibration["tool_rz"]),
        ]

    def _go_observe(self):
        """Lift, then park where the overview camera sees every cap."""
        if self._calibration is None:
            return _error_payload(self._mapping_error or "Not calibrated.")
        orientation = (
            self._calibration["tool_rx"],
            self._calibration["tool_ry"],
            self._calibration["tool_rz"],
        )
        speed = self._calibration["speed"]
        status, body = self._raise_for_travel(orientation, speed)
        if status != 200 or not body.get("success"):
            return status, body
        status, body = self._perform(
            "Park clear of the overview camera",
            self._observe_xyz,
            None,
            orientation,
            speed,
        )
        if status == 200 and body.get("success"):
            time.sleep(OBSERVE_SETTLE_SECONDS)
        return status, body

    def _wait_for_detection(self, class_name, timeout_seconds=6.0):
        """Return a fresh box for one colour, waiting briefly for it."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            detection = self._fresh_detection(class_name)
            if detection is not None:
                return detection
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.1)

    def _raise_for_travel(self, orientation, speed):
        """Lift straight up before a sideways move."""
        pose = self._fresh_tool_pose()
        if pose is None or pose[2] >= TRAVEL_Z_MM - 5.0:
            return 200, {"success": True}
        return self._perform(
            "Lift clear of the table",
            (pose[0], pose[1], TRAVEL_Z_MM),
            None,
            orientation,
            speed,
        )

    def _step_down(self, x_mm, y_mm, start_z, end_z, orientation):
        """Descend in short moves so the gripper stays over the cap."""
        self.get_logger().info(
            f"Lower in short steps from {start_z:.0f} mm "
            f"to {end_z:.0f} mm"
        )
        height = float(start_z)
        while height > float(end_z) + 1.0:
            height = max(float(end_z), height - DESCEND_STEP_MM)
            status, body = self._move_to(
                (x_mm, y_mm, height),
                orientation,
                DESCEND_SPEED,
            )
            if status != 200 or not body.get("success"):
                return status, body
        pose = self._fresh_tool_pose()
        if pose is not None:
            self.get_logger().info(
                "Descent finished at "
                f"{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f} mm"
            )
        return 200, {"success": True}

    def _wrist_guided_descend(
        self,
        x_mm,
        y_mm,
        start_z,
        end_z,
        orientation,
        class_name,
    ):
        """Lower onto the cap while the wrist camera keeps it centred.

        The overview camera only sets the starting X-Y. Each short
        drop recentres the cap on the wrist gripper target when the
        wrist can see it. Missing frames keep the last aim so a brief
        occlusion does not abort the pick.
        """
        self.get_logger().info(
            "Lower with wrist guidance from "
            f"{start_z:.0f} mm to {end_z:.0f} mm"
        )
        height = float(start_z)
        misses = 0
        guided = False
        abandoned = False
        while True:
            if abandoned:
                saw_cap = None
            else:
                status, body, x_mm, y_mm, saw_cap = self._wrist_recenter(
                    x_mm,
                    y_mm,
                    height,
                    orientation,
                    class_name,
                )
                if status != 200 or not body.get("success"):
                    return status, body, x_mm, y_mm
                if body.get("wrist_diverged"):
                    abandoned = True
                    guided = False
                    saw_cap = None
            if saw_cap is None:
                pass
            elif saw_cap:
                guided = True
                misses = 0
            else:
                misses += 1
                if misses == 1:
                    self.get_logger().info(
                        "Wrist camera did not see the cap; "
                        "keeping the overview aim for this step."
                    )
                elif (
                    misses == WRIST_MISS_LIMIT
                    and not guided
                ):
                    self.get_logger().info(
                        "Wrist camera stayed blind, finishing "
                        "the descent on the overview aim."
                    )
            if height <= float(end_z) + 1.0:
                break
            height = max(float(end_z), height - DESCEND_STEP_MM)
            status, body = self._move_to(
                (x_mm, y_mm, height),
                orientation,
                DESCEND_SPEED,
            )
            if status != 200 or not body.get("success"):
                return status, body, x_mm, y_mm
            time.sleep(WRIST_SETTLE_SECONDS)
        pose = self._fresh_tool_pose()
        if pose is not None:
            self.get_logger().info(
                "Wrist descent finished at "
                f"{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f} mm"
            )
        return 200, {"success": True, "wrist_guided": guided}, x_mm, y_mm

    def _wrist_recenter(
        self,
        x_mm,
        y_mm,
        height_z,
        orientation,
        class_name,
    ):
        """Nudge X-Y until the wrist grasp point sits on the cap.

        Every nudge is checked by the next frame. A nudge that leaves the
        cap further away is undone and body["wrist_diverged"] is set.
        """
        saw_cap = False
        previous_error = None
        before_x, before_y = x_mm, y_mm
        for attempt in range(WRIST_NUDGES_PER_STEP + 1):
            frame = self._wait_for_wrist_frame()
            if frame is None:
                return 200, {"success": True}, x_mm, y_mm, saw_cap
            center = self._track_wrist_center(
                frame, class_name, blend=False, follow_motion=True,
            )
            if center is None:
                return 200, {"success": True}, x_mm, y_mm, saw_cap
            saw_cap = True
            target_u, target_v = _wrist_target(
                frame,
                self._wrist_target_u,
                self._wrist_target_v,
            )
            error_u = float(center[0]) - float(target_u)
            error_v = float(center[1]) - float(target_v)
            error_px = math.hypot(error_u, error_v)
            self.get_logger().info(
                "Wrist cap is "
                f"{error_px:.1f} px from the grasp point "
                f"({error_u:.1f}, {error_v:.1f})."
            )
            if (
                previous_error is not None
                and error_px > previous_error + WRIST_DIVERGE_MARGIN_PX
            ):
                self.get_logger().warning(
                    "Wrist nudge moved the cap away "
                    f"({previous_error:.0f} px to {error_px:.0f} px). "
                    "Undoing it and finishing on the overview aim."
                )
                status, body = self._perform(
                    "Undo the wrist nudge",
                    (before_x, before_y, height_z),
                    None,
                    orientation,
                    DESCEND_SPEED,
                )
                if status != 200 or not body.get("success"):
                    return status, body, x_mm, y_mm, True
                return (
                    200,
                    {"success": True, "wrist_diverged": True},
                    before_x,
                    before_y,
                    False,
                )
            if error_px <= WRIST_ALIGN_TOLERANCE_PX:
                return 200, {"success": True}, x_mm, y_mm, True
            if attempt == WRIST_NUDGES_PER_STEP:
                break
            previous_error = error_px
            before_x, before_y = x_mm, y_mm
            scale = _wrist_gain_scale(
                height_z, float(self._calibration["table_z_mm"]),
            )
            gain = WRIST_GAIN_MM_PER_PX
            shift_x = scale * (gain[0][0] * error_u + gain[0][1] * error_v)
            shift_y = scale * (gain[1][0] * error_u + gain[1][1] * error_v)
            shift_x, shift_y = _limited_shift(
                shift_x,
                shift_y,
                WRIST_MAX_NUDGE_MM,
            )
            moved_x = x_mm + shift_x
            moved_y = y_mm + shift_y
            status, body = self._perform(
                "Center the wrist on the cap",
                (moved_x, moved_y, height_z),
                None,
                orientation,
                DESCEND_SPEED,
            )
            if status != 200 or not body.get("success"):
                return status, body, x_mm, y_mm, True
            x_mm, y_mm = moved_x, moved_y
            time.sleep(WRIST_SETTLE_SECONDS)
        return 200, {"success": True}, x_mm, y_mm, saw_cap

    def _perform(
        self,
        name,
        point,
        gripper_state,
        orientation,
        speed,
        gripper_speed=None,
    ):
        """Run one named gripper or coordinate step."""
        self.get_logger().info(name)
        if point is None:
            return self._set_gripper(gripper_state, gripper_speed)
        return self._move_to(point, orientation, speed)

    def _stable_cap_center(self, class_name):
        """Average several views of one cap colour before the arm moves."""
        samples = []
        seen = set()
        started = time.monotonic()
        # The overview detector runs below 1 frame per second on the Jetson.
        deadline = started + 8.0
        while len(samples) < 5 and time.monotonic() < deadline:
            detection = self._fresh_detection(class_name)
            if detection is not None:
                stamp = detection.get("received")
                if stamp not in seen and stamp >= started:
                    seen.add(stamp)
                    with self._lock:
                        frame = (
                            None if self._frame is None
                            else self._frame.copy()
                        )
                    pixel_u, pixel_v = refine_cap_center(
                        frame,
                        class_name,
                        float(detection["u"]),
                        float(detection["v"]),
                        float(detection.get("size_x", 0.0)),
                        float(detection.get("size_y", 0.0)),
                    )
                    samples.append((pixel_u, pixel_v))
            time.sleep(0.15)
        if len(samples) < 3:
            return None
        pixel_u = sum(sample[0] for sample in samples) / len(samples)
        pixel_v = sum(sample[1] for sample in samples) / len(samples)
        x_mm, y_mm = project_table(self._mapping, pixel_u, pixel_v)
        self.get_logger().info(
            f"{CAP_CLASSES.get(class_name, class_name)} cap centre "
            "averaged from "
            f"{len(samples)} views at pixel "
            f"{pixel_u:.1f}, {pixel_v:.1f}"
        )
        return x_mm, y_mm, pixel_u, pixel_v

    def _confirm_cap_target(
        self,
        x_mm,
        y_mm,
        hover_z,
        orientation,
        speed,
    ):
        """Follow the yellow cross once more before lowering.

        The first reading can be pulled aside while the arm is still
        leaving the picture. From the hover the cap is clear, so a
        second reading replaces that aim. The gripper still closes
        if this look fails.
        """
        time.sleep(0.3)
        samples = []
        seen = set()
        deadline = time.monotonic() + 1.2
        while len(samples) < 3 and time.monotonic() < deadline:
            detection = self._fresh_detection(self._pick_class)
            if detection is not None:
                stamp = detection.get("received")
                if stamp not in seen:
                    seen.add(stamp)
                    with self._lock:
                        frame = (
                            None if self._frame is None
                            else self._frame.copy()
                        )
                    if not _cap_still_clear(
                        frame,
                        self._pick_class,
                        detection,
                    ):
                        self.get_logger().info(
                            "Gripper is covering the cap, "
                            "lowering on the first aim."
                        )
                        return x_mm, y_mm
                    pixel_u, pixel_v = refine_cap_center(
                        frame,
                        self._pick_class,
                        float(detection["u"]),
                        float(detection["v"]),
                        float(detection.get("size_x", 0.0)),
                        float(detection.get("size_y", 0.0)),
                    )
                    samples.append((pixel_u, pixel_v))
            time.sleep(0.12)
        if len(samples) < 2:
            self.get_logger().info(
                "Cap is hidden at the hover, lowering on the first aim."
            )
            return x_mm, y_mm
        pixel_u = _median(sample[0] for sample in samples)
        pixel_v = _median(sample[1] for sample in samples)
        new_x, new_y = _grasp_point(self._mapping, pixel_u, pixel_v)
        offset_x, offset_y = self._grasp_offsets(self._pick_class)
        new_x += offset_x
        new_y += offset_y
        shift = math.hypot(new_x - x_mm, new_y - y_mm)
        self.get_logger().info(
            "Cap from above is "
            f"{new_x:.1f}, {new_y:.1f} mm, "
            f"{shift:.1f} mm from the first aim."
        )
        if shift < 3.0 or shift > 35.0:
            return x_mm, y_mm
        status, body = self._perform(
            "Move onto the cap centre",
            (new_x, new_y, hover_z),
            None,
            orientation,
            speed,
        )
        if status != 200 or not body.get("success"):
            return x_mm, y_mm
        return new_x, new_y

    def _grasp_offsets(self, class_name):
        """Return global plus per-colour grasp bias in millimetres."""
        name = str(class_name or "").casefold()
        class_x, class_y = CLASS_GRASP_OFFSET_MM.get(name, (0.0, 0.0))
        return (
            self._grasp_offset_x_mm + class_x,
            self._grasp_offset_y_mm + class_y,
        )

    def _soft_align_at_hover(
        self,
        x_mm,
        y_mm,
        hover_z,
        orientation,
        aim_u=0.0,
        aim_v=0.0,
        aim_w=36.0,
        aim_h=36.0,
    ):
        """Nudge so the open jaws sit on the remembered cap centre.

        Uses the live detection when it is still clear. Otherwise keeps
        the pixel remembered before the arm covered the disc. Never
        aborts the pick if jaws cannot be seen.
        """
        cap_u, cap_v = float(aim_u), float(aim_v)
        box_w, box_h = float(aim_w), float(aim_h)
        seed_x, seed_y = x_mm, y_mm
        best_x, best_y = x_mm, y_mm
        best_error = None
        for attempt in range(ALIGN_MAX_NUDGES + 1):
            frame = self._wait_for_frame()
            if frame is None:
                self.get_logger().info(
                    "No overview frame at align; keeping the aim."
                )
                return best_x, best_y
            detection = self._fresh_detection(self._pick_class)
            if detection is not None and _cap_still_clear(
                frame, self._pick_class, detection
            ):
                cap_u, cap_v = refine_cap_center(
                    frame,
                    self._pick_class,
                    float(detection["u"]),
                    float(detection["v"]),
                    float(detection.get("size_x", 0.0)),
                    float(detection.get("size_y", 0.0)),
                )
                box_w = float(detection.get("size_x") or box_w)
                box_h = float(detection.get("size_y") or box_h)
            elif not (cap_u > 1.0 and cap_v > 1.0):
                self.get_logger().info(
                    "No remembered cap pixel at align; "
                    "keeping the current aim."
                )
                return best_x, best_y
            tip = _jaw_midpoint(frame, cap_u, cap_v, box_w, box_h)
            if tip is None:
                self.get_logger().info(
                    "Gripper jaws not visible at align; "
                    "keeping the current aim."
                )
                return best_x, best_y
            error_x, error_y = _pixel_offset_mm(
                self._mapping,
                cap_u,
                cap_v,
                tip[0],
                tip[1],
            )
            error = math.hypot(error_x, error_y)
            self.get_logger().info(
                "Gripper jaws are "
                f"{error:.1f} mm from the cap centre "
                f"at pixel {cap_u:.0f}, {cap_v:.0f}."
            )
            # Huge initial jaw error is usually a false tip (arm/cable);
            # trust the overview XY seed instead of walking away.
            if best_error is None and error > 35.0:
                self.get_logger().info(
                    "Jaw-to-cap error "
                    f"{error:.1f} mm looks unreliable; "
                    "keeping the overview aim."
                )
                return seed_x, seed_y
            # A noisy jaw reading that suddenly looks worse is ignored;
            # keep the best pose soft-align has already reached.
            if best_error is not None and error > best_error + 4.0:
                self.get_logger().info(
                    "Jaw reading jumped worse "
                    f"({error:.1f} mm after {best_error:.1f} mm); "
                    "keeping the better aim."
                )
                return best_x, best_y
            if best_error is not None and error > best_error - 1.5:
                self.get_logger().info(
                    "Jaw align did not improve "
                    f"({error:.1f} mm after {best_error:.1f} mm); "
                    "keeping the better aim."
                )
                return best_x, best_y
            if best_error is None or error < best_error:
                best_error = error
                best_x, best_y = x_mm, y_mm
            if error <= ALIGN_TOLERANCE_MM:
                return best_x, best_y
            if attempt == ALIGN_MAX_NUDGES:
                self.get_logger().info(
                    "Hover align stopped at "
                    f"{best_error:.1f} mm; lowering anyway."
                )
                if best_error is not None and best_error > 25.0:
                    return seed_x, seed_y
                return best_x, best_y
            shift_x, shift_y = _limited_shift(
                -error_x * ALIGN_CORRECTION_GAIN,
                -error_y * ALIGN_CORRECTION_GAIN,
                ALIGN_MAX_STEP_MM,
            )
            moved_x = x_mm + shift_x
            moved_y = y_mm + shift_y
            status, body = self._perform(
                "Nudge toward the cap centre",
                (moved_x, moved_y, hover_z),
                None,
                orientation,
                DESCEND_SPEED,
            )
            if status != 200 or not body.get("success"):
                return best_x, best_y
            x_mm, y_mm = moved_x, moved_y
            time.sleep(ALIGN_SETTLE_SECONDS)
        return best_x, best_y

    def _nudge_at_grasp(
        self,
        x_mm,
        y_mm,
        pixel_u,
        pixel_v,
        grasp_z,
        orientation,
        box_w=36.0,
        box_h=36.0,
    ):
        """One last sideways nudge if the open jaws sit beside the cap."""
        frame = self._wait_for_frame()
        if frame is None:
            return x_mm, y_mm
        detection = self._fresh_detection(self._pick_class)
        if detection is not None:
            pixel_u, pixel_v = refine_cap_center(
                frame,
                self._pick_class,
                float(detection["u"]),
                float(detection["v"]),
                float(detection.get("size_x", 0.0)),
                float(detection.get("size_y", 0.0)),
            )
            box_w = float(detection.get("size_x") or box_w)
            box_h = float(detection.get("size_y") or box_h)
        if str(self._pick_class or "").casefold() == "red_cap":
            red = red_disc_center(frame, pixel_u, pixel_v, box_w, box_h)
            if red is not None and red[2] >= 200.0:
                pixel_u, pixel_v = red[0], red[1]
        tip = _jaw_midpoint(frame, pixel_u, pixel_v, box_w, box_h)
        if tip is None:
            self.get_logger().info(
                "Jaws not visible at grasp; closing on the aim point."
            )
            return x_mm, y_mm
        error_x, error_y = _pixel_offset_mm(
            self._mapping,
            pixel_u,
            pixel_v,
            tip[0],
            tip[1],
        )
        error = math.hypot(error_x, error_y)
        self.get_logger().info(
            f"Jaws are {error:.1f} mm from the cap at grasp."
        )
        if error < 2.0 or error > 22.0:
            return x_mm, y_mm
        shift_x, shift_y = _limited_shift(
            -error_x * ALIGN_CORRECTION_GAIN,
            -error_y * ALIGN_CORRECTION_GAIN,
            10.0,
        )
        moved_x = x_mm + shift_x
        moved_y = y_mm + shift_y
        status, body = self._perform(
            "Center the jaws on the cap",
            (moved_x, moved_y, grasp_z),
            None,
            orientation,
            DESCEND_SPEED,
        )
        if status != 200 or not body.get("success"):
            return x_mm, y_mm
        return moved_x, moved_y

    def _wait_for_wrist_frame(self):
        """Return a wrist frame captured after this call starts."""
        started = time.monotonic()
        deadline = started + ALIGN_FRAME_WAIT_SECONDS
        while True:
            with self._lock:
                received = self._wrist_frame_received
                frame = self._wrist_frame
            if frame is not None and received >= started:
                return frame.copy()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _wait_for_frame(self, timeout_seconds=None):
        """Return a camera frame captured after this call starts."""
        if timeout_seconds is None:
            timeout_seconds = ALIGN_FRAME_WAIT_SECONDS
        started = time.monotonic()
        deadline = started + float(timeout_seconds)
        while True:
            with self._lock:
                received = self._frame_received
                frame = self._frame
            if frame is not None and received >= started:
                return frame.copy()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _fail_and_return(self, status, body):
        """Move back to the pose saved at the start of this pick."""
        original = body.get("message", "Pick failed")
        if self._calibration is not None:
            hover_z = (
                self._calibration["table_z_mm"]
                + self._calibration["hover_mm"]
            )
            orientation = (
                self._calibration["tool_rx"],
                self._calibration["tool_ry"],
                self._calibration["tool_rz"],
            )
            speed = self._calibration["speed"]
            self._set_gripper(0)
            pose = self._fresh_tool_pose()
            if pose is not None and pose[2] < hover_z - 2.0:
                self._move_to(
                    (pose[0], pose[1], hover_z),
                    orientation,
                    speed,
                )
        home_status, home_body = self._move_initial()
        if home_status == 200 and home_body.get("success"):
            body["message"] = (
                original + " Returned to the initial position."
            )
            return status, body
        home_message = home_body.get("message", "Could not return.")
        body["message"] = original + " Could not return: " + home_message
        return status, body

    def go_zero(self):
        """Move all six joints to 0°."""
        if not self._motion_lock.acquire(blocking=False):
            return _error_payload("A move is already running", 409)
        self._zero_stop.clear()
        self._busy = True
        try:
            return self._move_zero()
        finally:
            self._busy = False
            self._motion_lock.release()

    def _move_zero(self):
        """Stream joint targets to the zero pose. Caller holds the lock."""
        goal = list(ZERO_POSITION_DEGREES)
        try:
            commanded = self._joint_degrees()
        except ValueError as error:
            return _error_payload(str(error))
        origin = list(commanded)
        duration = _travel_duration(origin, goal)
        elapsed = 0.0
        settle_started = None
        next_tick = time.monotonic()
        self.get_logger().info("Moving all joints to 0°")
        while not self._zero_stop.is_set():
            if duration <= 0.0 or _max_abs_diff(commanded, goal) < 0.2:
                if settle_started is None:
                    settle_started = time.monotonic()
                try:
                    live = self._joint_degrees()
                except ValueError as error:
                    return _error_payload(str(error))
                if _max_abs_diff(live, goal) <= ZERO_TOLERANCE_DEGREES:
                    return 200, {
                        "success": True,
                        "message": "Zero position reached",
                    }
                if time.monotonic() - settle_started > ZERO_SETTLE_SECONDS:
                    return _error_payload(
                        "Zero position was not reached"
                    )
            else:
                settle_started = None
                elapsed += ZERO_PERIOD_SECONDS
                command_elapsed = min(
                    duration,
                    elapsed + ZERO_LOOKAHEAD_SECONDS,
                )
                desired = _eased_targets(
                    origin,
                    goal,
                    command_elapsed,
                    duration,
                )
                targets = _limit_step(
                    commanded,
                    desired,
                    ZERO_STEP_DEGREES,
                )
                status, body = self._command_joints(targets, ZERO_SPEED)
                if self._zero_stop.is_set() or _motion_halted(body):
                    return 200, {
                        "success": True,
                        "message": "Move to zero stopped",
                    }
                if status != 200 or not body.get("success"):
                    return status, body
                commanded = targets
            next_tick += ZERO_PERIOD_SECONDS
            delay = next_tick - time.monotonic()
            if delay > 0 and self._zero_stop.wait(delay):
                return 200, {
                    "success": True,
                    "message": "Move to zero stopped",
                }
        return 200, {
            "success": True,
            "message": "Move to zero stopped",
        }

    def _command_joints(self, targets, speed):
        """Send one six-joint target. Caller holds the motion lock."""
        request = MoveJoints.Request()
        request.target_degrees = [float(value) for value in targets]
        request.speed = int(speed)
        request.execute = True
        response, error = _call_service(
            self._joints_client,
            request,
            MOVE_TIMEOUT_SECONDS,
        )
        return _service_payload(response, error)

    def _on_joint_states(self, message):
        """Store the six arm joints in degrees."""
        angles = [None] * 6
        for index, name in enumerate(JOINT_NAMES):
            if name not in message.name:
                continue
            position_index = list(message.name).index(name)
            if position_index >= len(message.position):
                continue
            radians = float(message.position[position_index])
            if math.isfinite(radians):
                angles[index] = math.degrees(radians)
        with self._joint_lock:
            self._joint_degrees_live = angles

    def _joint_degrees(self):
        """Return the six live joint angles in degrees."""
        with self._joint_lock:
            angles = list(self._joint_degrees_live)
        if any(angle is None or not math.isfinite(angle) for angle in angles):
            raise ValueError("Joint feedback is not available yet")
        return [float(angle) for angle in angles]

    def return_initial(self):
        """Move to the parking pose the overview camera sees from."""
        if self._calibration is None:
            return _error_payload(self._mapping_error or "Not calibrated.")
        if not self._motion_lock.acquire(blocking=False):
            return _error_payload("A move is already running", 409)
        self._busy = True
        try:
            if self._initial_pose is None:
                self._initial_pose = self._observe_pose()
            status, body = self._move_initial()
            if status == 200 and body.get("success"):
                body["message"] = "Initial position reached"
            return status, body
        finally:
            self._busy = False
            self._motion_lock.release()

    def _move_initial(self):
        """Command the saved starting pose, or the drop hover if out of range."""
        pose = self._initial_pose
        if pose is None:
            return _error_payload("No initial position saved.")
        speed = 12
        if self._calibration is not None:
            speed = self._calibration["speed"]
            # A direct move from a low pose sweeps across the caps.
            status, body = self._raise_for_travel(
                (
                    self._calibration["tool_rx"],
                    self._calibration["tool_ry"],
                    self._calibration["tool_rz"],
                ),
                speed,
            )
            if status != 200 or not body.get("success"):
                return status, body
        target_x, target_y, target_z = (
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
        )
        orientation = (float(pose[3]), float(pose[4]), float(pose[5]))
        if (
            not 50.0 <= target_x <= 325.0
            or not -200.0 <= target_y <= 200.0
            or not 20.0 <= target_z <= 250.0
        ):
            if self._calibration is not None:
                target_x = float(self._calibration["drop_x_mm"])
                target_y = float(self._calibration["drop_y_mm"])
                target_z = float(
                    self._calibration["table_z_mm"]
                    + self._calibration["hover_mm"]
                )
                orientation = (
                    float(self._calibration["tool_rx"]),
                    float(self._calibration["tool_ry"]),
                    float(self._calibration["tool_rz"]),
                )
        return self._move_to(
            (target_x, target_y, target_z),
            orientation,
            speed,
        )

    def stop_robot(self):
        """Call the hardware stop service."""
        self._zero_stop.set()
        self.get_logger().warning("Stop requested from the pick page")
        response, error = _call_service(
            self._stop_client,
            Trigger.Request(),
            SERVICE_WAIT_SECONDS,
        )
        return _service_payload(response, error)

    def set_motor_hold(self, hold):
        """Release the arm for hand placement, or lock it in place."""
        client = self._focus_client if hold else self._release_client
        response, error = _call_service(
            client,
            Trigger.Request(),
            SERVICE_WAIT_SECONDS,
        )
        status, payload = _service_payload(response, error)
        payload["wizard"] = self.wizard_payload()
        return status, payload

    def _fresh_detection(self, class_name=None):
        """Return one recent cap detection, or the strongest if unnamed."""
        with self._lock:
            stored = {
                name: dict(detection)
                for name, detection in self._detections.items()
            }
        now = time.monotonic()
        fresh = []
        for name, detection in stored.items():
            if now - detection["received"] > DETECTION_MAX_AGE_SECONDS:
                continue
            if (
                class_name
                and name.casefold() != str(class_name).casefold()
            ):
                continue
            fresh.append(detection)
        if not fresh:
            return None
        return max(fresh, key=lambda detection: detection["score"])

    def _visible_detections(self):
        """Return each recent cap without its internal timestamp."""
        with self._lock:
            stored = {
                name: dict(detection)
                for name, detection in self._detections.items()
            }
        now = time.monotonic()
        visible = {}
        for name, detection in stored.items():
            age = now - detection["received"]
            if age > DETECTION_MAX_AGE_SECONDS:
                continue
            detection.pop("received", None)
            detection["age_seconds"] = round(age, 3)
            if self._mapping is not None:
                x_mm, y_mm = project_table(
                    self._mapping,
                    detection["u"],
                    detection["v"],
                )
                detection["x_mm"] = round(x_mm, 1)
                detection["y_mm"] = round(y_mm, 1)
            visible[name] = detection
        return visible

    def _move_to(self, point, orientation, speed):
        """Send one guarded coordinate target."""
        request = MoveCoords.Request()
        request.x = float(point[0])
        request.y = float(point[1])
        request.z = float(point[2])
        request.rx = float(orientation[0])
        request.ry = float(orientation[1])
        request.rz = float(orientation[2])
        request.speed = int(speed)
        request.execute = True
        response, error = _call_service(
            self._coords_client,
            request,
            MOVE_TIMEOUT_SECONDS,
        )
        status, body = _service_payload(response, error)
        arrived = body.get("final_coords")
        if arrived:
            self.get_logger().info(
                f"Arrived at {arrived[0]:.1f}, {arrived[1]:.1f}, "
                f"{arrived[2]:.1f} mm"
            )
        return status, body

    def _set_gripper(self, state, speed=None):
        """Open or close the gripper. Speed applies to this call only."""
        request = SetGripper.Request()
        request.state = int(state)
        if speed is None:
            request.speed = GRIPPER_SPEED
        else:
            request.speed = int(speed)
        request.execute = True
        response, error = _call_service(
            self._gripper_client,
            request,
            GRIPPER_TIMEOUT_SECONDS,
        )
        return _service_payload(response, error)

    def _services_ready(self):
        """Return true when the arm services are available."""
        return (
            self._coords_client.service_is_ready()
            and self._joints_client.service_is_ready()
            and self._gripper_client.service_is_ready()
            and self._stop_client.service_is_ready()
        )

    def _on_image(self, message):
        """Keep the latest rectified frame for the cap centre and the page."""
        now = time.monotonic()
        try:
            image = self._image_bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().warning(f"Could not read image: {error}")
            return
        with self._lock:
            self._frame = image.copy()
            self._frame_received = now
            self._image_width = int(image.shape[1])
            self._image_height = int(image.shape[0])
        if now - self._jpeg_encoded_at < 0.2:
            return
        marked = _mark_overview_detections(image, self._visible_detections())
        marked = _mark_overview_aruco(marked, ARUCO_MARKER_ID)
        success, encoded = cv2.imencode(
            ".jpg",
            marked,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )
        if not success:
            return
        with self._lock:
            self._jpeg = encoded.tobytes()
            self._jpeg_encoded_at = now

    def _same_locked_cap(self, state, measured, wide):
        """True when a new reading is still the locked disc."""
        jump = math.hypot(
            float(measured[0]) - float(state["u"]),
            float(measured[1]) - float(state["v"]),
        )
        radius = float(state.get("radius") or 0.0)
        limit = WRIST_LOCK_RADIUS_PX
        if wide and radius > 1.0:
            # A descent nudge moves the cap by more than the still-frame
            # lock, but not onto a different object.
            limit = max(limit, 1.15 * radius)
        if jump > limit:
            return False
        if radius > 1.0 and len(measured) > 2 and float(measured[2]) > 1.0:
            ratio = float(measured[2]) / radius
            if ratio < 0.65 or ratio > 1.45:
                return False
        return True

    def _track_wrist_center(
        self, image, class_name, blend=True, keep_last=False, follow_motion=False,
    ):
        """Return the wrist centre for one cap colour.

        Each colour has its own lock. A new reading has to land on that
        same disc, so the ring cannot walk onto the table. blend is off
        for the page and the descent: the drawn point is the fresh
        centre. keep_last holds that centre through a short miss, then
        drops it. follow_motion lets one nudge shift the disc without
        accepting a different size of object.
        """
        name = str(class_name or "").casefold()
        if name not in CAP_CLASSES:
            return None
        with self._lock:
            state = self._wrist_locks.get(name)
            hint = None if state is None else (state["u"], state["v"])
        measured = wrist_cap_center(image, name, hint=hint)
        if (
            measured is not None
            and state is not None
            and not self._same_locked_cap(state, measured, wide=False)
        ):
            measured = None
        if (
            measured is None
            and hint is not None
            and follow_motion
            and state is not None
        ):
            candidate = wrist_cap_center(image, name, hint=None)
            if candidate is not None and self._same_locked_cap(
                state, candidate, wide=True,
            ):
                measured = candidate
        with self._lock:
            state = self._wrist_locks.get(name)
            if measured is None:
                if state is None:
                    return None
                state["misses"] = int(state.get("misses") or 0) + 1
                if state["misses"] >= WRIST_LOCK_MISSES:
                    self._wrist_locks.pop(name, None)
                    return None
                if keep_last:
                    return (state["u"], state["v"])
                return None
            if blend and state is not None:
                kept_u = (
                    WRIST_LOCK_BLEND * float(measured[0])
                    + (1.0 - WRIST_LOCK_BLEND) * state["u"]
                )
                kept_v = (
                    WRIST_LOCK_BLEND * float(measured[1])
                    + (1.0 - WRIST_LOCK_BLEND) * state["v"]
                )
            else:
                kept_u = float(measured[0])
                kept_v = float(measured[1])
            if len(measured) > 2 and float(measured[2]) > 1.0:
                radius = float(measured[2])
            elif state is not None:
                radius = float(state.get("radius") or 0.0)
            else:
                radius = 0.0
            self._wrist_locks[name] = {
                "u": kept_u,
                "v": kept_v,
                "misses": 0,
                "radius": radius,
            }
            return (kept_u, kept_v)

    def _on_wrist_image(self, message):
        """Keep the latest wrist frame and a marked JPEG for the page."""
        now = time.monotonic()
        try:
            image = self._image_bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().warning(
                f"Could not read wrist image: {error}"
            )
            return
        with self._lock:
            self._wrist_frame = image
            self._wrist_frame_received = now
        # The page only needs a slow preview. Running the colour
        # finders on every frame starves the other callbacks.
        if now - self._wrist_jpeg_encoded_at < WRIST_PREVIEW_PERIOD_SECONDS:
            return
        marked = image.copy()
        target_u, target_v = _wrist_target(
            marked,
            self._wrist_target_u,
            self._wrist_target_v,
        )
        cv2.drawMarker(
            marked,
            (int(round(target_u)), int(round(target_v))),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        # Show the ignored gripper strip so false locks are obvious.
        left = int(round(marked.shape[1] * WRIST_SEARCH_LEFT_FRAC))
        cv2.line(
            marked,
            (left, 0),
            (left, marked.shape[0] - 1),
            (80, 80, 80),
            1,
        )
        # Each colour keeps its own centre. The arm still follows
        # only the cap being picked; that ring is drawn thicker.
        active = str(self._pick_class or self._target_class or "").casefold()
        for class_name in CAP_CLASSES:
            center = self._track_wrist_center(
                image, class_name, blend=False, keep_last=True,
            )
            if center is None:
                continue
            chosen = class_name == active
            point = (int(round(center[0])), int(round(center[1])))
            cv2.circle(
                marked,
                point,
                10 if chosen else 8,
                (0, 255, 0),
                2 if chosen else 1,
            )
            cv2.putText(
                marked,
                CAP_CLASSES[class_name],
                (point[0] + 10, max(16, point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        success, encoded = cv2.imencode(
            ".jpg",
            marked,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )
        if not success:
            return
        with self._lock:
            self._wrist_jpeg = encoded.tobytes()
            self._wrist_jpeg_encoded_at = now

    def _on_tool_pose(self, message):
        """Remember the controller pose. This callback sends no command."""
        if len(message.data) < 6:
            return
        values = []
        for value in message.data[:6]:
            number = float(value)
            if not math.isfinite(number):
                return
            values.append(number)
        with self._lock:
            self._tool_pose = values
            self._tool_pose_received = time.monotonic()

    def _fresh_tool_pose(self):
        """Return a tool pose published within the last second."""
        with self._lock:
            pose = self._tool_pose
            received = self._tool_pose_received
        if pose is None:
            return None
        if time.monotonic() - received > POSE_MAX_AGE_SECONDS:
            return None
        return list(pose)

    def _store_wizard(self, accepted=None):
        """Write the wizard. Passing accepted=True enables picking."""
        if accepted is None:
            accepted = bool(self._wizard.get("accepted"))
        self._wizard["accepted"] = bool(accepted)
        _write_wizard(
            self._calibration_path,
            self._wizard,
            accepted=bool(accepted),
        )

    def _on_camera_info(self, message):
        """Remember the overview image size used by the homography."""
        self._image_width = int(message.width)
        self._image_height = int(message.height)

    def _preview_payload(self, detection):
        """Return the display-only table and joint1 target."""
        empty = {
            "ready": False,
            "motion_command": False,
            "message": self._preview_error,
            "pixel_u": None,
            "pixel_v": None,
            "table_x_m": None,
            "table_y_m": None,
            "joint1_x_m": None,
            "joint1_y_m": None,
            "joint1_z_m": None,
        }
        if self._preview_model is None:
            return empty
        if detection is None:
            empty["message"] = (
                "No fresh cap centre to preview."
            )
            return empty
        try:
            preview = project_table_preview(
                self._preview_model,
                detection["u"],
                detection["v"],
                self._image_width,
                self._image_height,
            )
        except ValueError as error:
            empty["message"] = str(error)
            empty["pixel_u"] = round(detection["u"], 1)
            empty["pixel_v"] = round(detection["v"], 1)
            return empty
        preview["ready"] = True
        preview["motion_command"] = False
        preview["message"] = (
            "Display only. joint1 is not tool_pose_raw, "
            "and this does not move the arm."
        )
        return preview

    def _on_model_ready(self, message):
        """Enable picking only after the trained weights load."""
        ready = bool(message.data)
        if ready:
            text = ""
        else:
            text = (
                "Trained weights failed to load. "
                "Picking is disabled."
            )
        with self._lock:
            self._weights_ready = ready
            self._weights_message = text

    def _on_detections(self, message):
        """Keep the newest box of each cap color."""
        found = {}
        for detection in message.detections:
            if not detection.results:
                continue
            hypothesis = detection.results[0].hypothesis
            class_name = str(hypothesis.class_id)
            chosen = _cap_class(class_name)
            if chosen is None:
                continue
            score = float(hypothesis.score)
            current = found.get(chosen)
            if current is not None and score <= current["score"]:
                continue
            found[chosen] = {
                "class_name": chosen,
                "score": round(score, 3),
                "u": float(detection.bbox.center.position.x),
                "v": float(detection.bbox.center.position.y),
                "size_x": float(detection.bbox.size_x),
                "size_y": float(detection.bbox.size_y),
                "received": time.monotonic(),
            }
        with self._lock:
            self._detections = found


class PickHandler(BaseHTTPRequestHandler):
    """HTTP handlers for the pick page."""

    def do_GET(self):
        """Serve the page or the live detection JSON."""
        path = self.path.split("?", 1)[0]
        node = self.server.pick_node
        if path in ("/", "/pick"):
            body = node.page_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/calibrate":
            body = node.calibrate_page_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/state":
            body = json.dumps(node.state_payload()).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if path in ("/overview", "/camera"):
            self._send(200, OVERVIEW_PAGE, "text/html; charset=utf-8")
            return
        if path == "/aruco":
            self._send(200, ARUCO_PAGE, "text/html; charset=utf-8")
            return
        if path == "/api/aruco/marker.png":
            png = node.aruco_marker_png()
            if png is None:
                self.send_error(500, "Could not draw ArUco marker")
                return
            self._send(200, png, "image/png")
            return
        if path == "/api/camera.jpg":
            jpeg = node.camera_jpeg()
            if jpeg is None:
                self.send_error(503, "Overview image is not available")
                return
            self._send(200, jpeg, "image/jpeg")
            return
        if path == "/wrist":
            self._send(200, WRIST_PAGE, "text/html; charset=utf-8")
            return
        if path == "/api/wrist.jpg":
            jpeg = node.wrist_jpeg()
            if jpeg is None:
                self.send_error(503, "Wrist image is not available")
                return
            self._send(200, jpeg, "image/jpeg")
            return
        self.send_error(404)

    def do_POST(self):
        """Start one pick or stop the arm."""
        path = self.path.split("?", 1)[0]
        node = self.server.pick_node
        if path == "/api/pick":
            body = self._read_json()
            status, payload = node.pick(body.get("class_name"))
        elif path == "/api/pick_all":
            status, payload = node.pick_all()
        elif path == "/api/stop":
            status, payload = node.stop_robot()
        elif path == "/api/return":
            status, payload = node.return_initial()
        elif path == "/api/zero":
            status, payload = node.go_zero()
        elif path == "/api/calibrate/click":
            body = self._read_json()
            status, payload = node.note_click(body.get("u"), body.get("v"))
        elif path == "/api/calibrate/capture":
            status, payload = node.capture_spot()
        elif path == "/api/calibrate/auto":
            status, payload = node.auto_capture_spot()
        elif path == "/api/calibrate/aruco":
            status, payload = node.aruco_capture_spot()
        elif path == "/api/calibrate/clear":
            status, payload = node.clear_calibration_spots()
        elif path == "/api/calibrate/drop":
            body = self._read_json()
            status, payload = node.capture_drop(body.get("class_name"))
        elif path == "/api/calibrate/clearances":
            status, payload = node.save_clearances(self._read_json())
        elif path == "/api/calibrate/undo":
            status, payload = node.undo_spot()
        elif path == "/api/calibrate/accept":
            status, payload = node.accept_calibration()
        elif path == "/api/calibrate/release":
            status, payload = node.set_motor_hold(False)
        elif path == "/api/calibrate/hold":
            status, payload = node.set_motor_hold(True)
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json")

    def _read_json(self):
        """Read one JSON request body."""
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send(self, status, body, content_type):
        """Write one HTTP response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Keep ordinary page polls out of the launch log."""
        del format, args


def fit_table(pairs):
    """Fit pixel (u, v) to robot (x, y) millimetres."""
    if len(pairs) < 3:
        raise ValueError(
            "Add at least three calibration pairs before picking."
        )
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    rhs = [[0.0, 0.0] for _ in range(3)]
    for pair in pairs:
        row = [float(pair["u"]), float(pair["v"]), 1.0]
        target = [float(pair["x"]), float(pair["y"])]
        for row_index in range(3):
            for column in range(3):
                matrix[row_index][column] += row[row_index] * row[column]
            rhs[row_index][0] += row[row_index] * target[0]
            rhs[row_index][1] += row[row_index] * target[1]
    return _solve(matrix, rhs)


def project_table(mapping, pixel_u, pixel_v):
    """Return robot X-Y millimetres for one pixel."""
    row = [float(pixel_u), float(pixel_v), 1.0]
    x_mm = sum(row[index] * mapping[index][0] for index in range(3))
    y_mm = sum(row[index] * mapping[index][1] for index in range(3))
    return x_mm, y_mm


def _grasp_point(mapping, pixel_u, pixel_v):
    """Return the robot point for the measured centre of the cap."""
    return project_table(mapping, pixel_u, pixel_v)


def _jaw_is_holding(body):
    """Return true when the jaws stopped on a cap, not on empty air."""
    value = body.get("final_value")
    if not isinstance(value, int):
        return False
    return 12 <= value <= 86


def _median(values):
    """Return the middle value of a short list of samples."""
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _near_cross(origin_u, origin_v, pixel_u, pixel_v):
    """Return true when a refined centre still sits on the yellow cross."""
    return math.hypot(
        float(pixel_u) - float(origin_u),
        float(pixel_v) - float(origin_v),
    ) <= REFINE_MAX_PIXELS


def _cap_still_clear(image, class_name, detection):
    """Return true when the hover has not hidden the chosen cap."""
    if image is None or detection is None:
        return False
    if str(class_name or "").casefold() != "red_cap":
        return True
    disc = red_disc_center(
        image,
        float(detection["u"]),
        float(detection["v"]),
        float(detection.get("size_x", 0.0)),
        float(detection.get("size_y", 0.0)),
    )
    return disc is not None and disc[2] >= 400.0


def _pixel_offset_mm(mapping, cap_u, cap_v, grip_u, grip_v):
    """Return how far the gripper pixel sits from the cap, in millimetres."""
    cap_x, cap_y = project_table(mapping, cap_u, cap_v)
    grip_x, grip_y = project_table(mapping, grip_u, grip_v)
    return grip_x - cap_x, grip_y - cap_y


def _jaw_midpoint(image, cap_u, cap_v, box_w, box_h):
    """Return the open-jaw midpoint near a remembered cap pixel.

    Prefers the two-finger pair beside the disc. Falls back to the
    global pad midpoint only when it sits near that disc, so the arm
    base is never mistaken for the gripper.
    """
    tip = finger_midpoint(image, cap_u, cap_v, box_w, box_h)
    if tip is not None:
        return tip
    tip = gripper_beside_cap(image, cap_u, cap_v)
    if tip is not None:
        return tip
    pad = overview_pad_center(image)
    if pad is None:
        return None
    if math.hypot(pad[0] - cap_u, pad[1] - cap_v) > 110.0:
        return None
    return (pad[0], pad[1])


def _limited_shift(shift_x, shift_y, limit_mm):
    """Scale a nudge so one try stays inside the step limit."""
    length = math.hypot(shift_x, shift_y)
    if length <= limit_mm or length == 0.0:
        return shift_x, shift_y
    scale = limit_mm / length
    return shift_x * scale, shift_y * scale


def _wrist_gain_scale(height_z, table_z):
    """Return the share of the travel-height wrist gain for one height."""
    reference = TRAVEL_Z_MM - table_z + WRIST_LENS_ABOVE_TIP_MM
    current = height_z - table_z + WRIST_LENS_ABOVE_TIP_MM
    if reference <= 0.0:
        return 1.0
    return min(1.0, max(0.3, current / reference))


def gripper_tip(image, cap_u, cap_v, box_width, box_height):
    """Return the open-gripper tip nearest a visible cap centre.

    The cap disc is ignored. Vertical dark parts above that disc are
    the fingers. The tip is the bottom of those fingers.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    span = max(float(box_width), float(box_height), 24.0)
    half_w = int(max(90.0, span * 3.0))
    above = int(max(140.0, span * 5.0))
    below = int(max(span, 30.0))
    x0 = max(0, int(round(cap_u)) - half_w)
    x1 = min(width, int(round(cap_u)) + half_w)
    y0 = max(0, int(round(cap_v)) - above)
    y1 = min(height, int(round(cap_v)) + below)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    table = float(np.percentile(gray, 75))
    threshold = min(table - 18.0, 165.0)
    if threshold < 40.0:
        return None
    mask = np.uint8(gray < threshold) * np.uint8(255)
    center_u = int(round(cap_u)) - x0
    center_v = int(round(cap_v)) - y0
    cv2.circle(
        mask,
        (center_u, center_v),
        int(max(span * 0.8, 14.0)),
        0,
        -1,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    fingers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 60.0:
            continue
        _center, size, _angle = cv2.minAreaRect(contour)
        short_side = min(float(size[0]), float(size[1]))
        long_side = max(float(size[0]), float(size[1]))
        if short_side < 1.0 or long_side < 18.0:
            continue
        if long_side / short_side < 1.4:
            continue
        points = contour[:, 0, :]
        lowest = int(points[:, 1].max())
        bottom_points = points[points[:, 1] >= lowest - 2]
        fingers.append(
            (
                float(bottom_points[:, 0].mean() + x0),
                float(bottom_points[:, 1].mean() + y0),
            )
        )
    if not fingers:
        return None
    fingers.sort(
        key=lambda point: abs(point[0] - cap_u) + abs(point[1] - cap_v)
    )
    if len(fingers) >= 2:
        gap = math.hypot(
            fingers[0][0] - fingers[1][0],
            fingers[0][1] - fingers[1][1],
        )
        if 8.0 <= gap <= max(80.0, span * 3.0):
            return (
                (fingers[0][0] + fingers[1][0]) / 2.0,
                (fingers[0][1] + fingers[1][1]) / 2.0,
            )
    return fingers[0]


_OVERVIEW_MARK_COLORS = {
    "red_cap": (40, 40, 220),
    "white_cap": (230, 230, 230),
    "black_cap": (40, 220, 40),
}


def _mark_overview_detections(image, detections):
    """Draw each visible cap centre on a copy of the overview frame."""
    if image is None or image.size == 0:
        return image
    marked = image.copy()
    for name, detection in (detections or {}).items():
        try:
            center_u = float(detection["u"])
            center_v = float(detection["v"])
            size_x = float(detection.get("size_x") or 28.0)
            size_y = float(detection.get("size_y") or 28.0)
            score = float(detection.get("score") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        radius = max(10, int(round(0.5 * max(size_x, size_y))))
        color = _OVERVIEW_MARK_COLORS.get(name, (0, 255, 255))
        point = (int(round(center_u)), int(round(center_v)))
        cv2.circle(marked, point, radius, color, 2)
        cv2.drawMarker(
            marked,
            point,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        label = CAP_CLASSES.get(name, name) + f" {int(round(score * 100))}%"
        cv2.putText(
            marked,
            label,
            (point[0] + 8, max(16, point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return marked


def _wrist_target(image, target_u, target_v):
    """Return the wrist pixel the cap centre should reach.

    Defaults sit on the open-jaw grasp point for this wrist mount,
    which is left of the image centre.
    """
    height, width = image.shape[:2]
    if target_u < 0.0:
        target_u = width * WRIST_TARGET_U_FRAC
    if target_v < 0.0:
        target_v = height * WRIST_TARGET_V_FRAC
    return target_u, target_v


def _wrist_search_roi(image):
    """Crop the far-left gripper body; grasp point stays searchable."""
    height, width = image.shape[:2]
    left = min(width - 16, max(0, int(round(width * WRIST_SEARCH_LEFT_FRAC))))
    return image[:, left:], left


def _gripper_free_mask(image):
    """Mask out bright gripper plastic so nearby dark pads are ignored.

    Only the jaw column is removed. A wider band also erased a white
    cap sitting just to the right of the fingers.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
    left_band = np.zeros((height, width), dtype=np.uint8)
    left_band[:, : int(width * 0.34)] = 255
    gripper = cv2.bitwise_and(white, left_band)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gripper = cv2.dilate(gripper, kernel, iterations=1)
    return cv2.bitwise_not(gripper)


def wrist_cap_center(image, class_name, hint=None):
    """Return the coloured cap centre in a wrist image, or None.

    hint is the last accepted centre for this colour. A blob far from
    it is rejected so the ring cannot jump to another object.
    """
    if image is None or image.size == 0:
        return None
    roi, left = _wrist_search_roi(image)
    if roi.size == 0:
        return None
    free = _gripper_free_mask(roi)
    name = str(class_name or "").casefold()
    local_hint = None
    if hint is not None:
        local_hint = (float(hint[0]) - left, float(hint[1]))
    if name == "black_cap":
        center = _round_dark_center(roi, free, hint=local_hint)
    elif name == "white_cap":
        center = _pale_center(roi, free, hint=local_hint)
    else:
        center = _red_center(roi, free, hint=local_hint)
    if center is None:
        return None
    return center[0] + left, center[1], float(center[2])


def _cap_disc_and_ring(image, center_u, center_v, radius):
    """Return the cap interior and the visible table ring around it."""
    height, width = image.shape[:2]
    if radius < 8.0:
        return None
    pixel_u = int(round(center_u))
    pixel_v = int(round(center_v))
    if pixel_u < 0 or pixel_v < 0 or pixel_u >= width or pixel_v >= height:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rows, cols = np.ogrid[:height, :width]
    distance = np.hypot(cols - center_u, rows - center_v)
    disc = distance <= radius * 0.85
    ring = (distance >= radius * 1.08) & (distance <= radius * 1.45)
    if int(np.count_nonzero(disc)) < 40 or int(np.count_nonzero(ring)) < 40:
        return None
    return gray, hsv, disc, ring, distance


def _black_cap_disc(image, raw_mask, center_u, center_v, radius):
    """True for a filled black bottle cap on the pale table.

    Dark letters on a silver cap, and dark objects on the wooden
    floor, are round enough to reach this test. The cap is a dark
    disc with pale plastic around it.
    """
    regions = _cap_disc_and_ring(image, center_u, center_v, radius)
    if regions is None:
        return False
    gray, hsv, disc, ring, distance = regions
    circle = distance <= radius
    circle_count = int(np.count_nonzero(circle))
    if circle_count < 40:
        return False
    raw_fill = float(np.count_nonzero(raw_mask[circle])) / float(circle_count)
    if raw_fill < 0.50:
        return False
    if float(np.median(gray[disc])) > 25.0:
        return False
    if float(np.median(gray[ring])) - float(np.median(gray[disc])) < 50.0:
        return False
    if float(np.median(hsv[:, :, 1][ring])) > 80.0:
        return False
    return True


def _red_center(image, free_mask=None, hint=None):
    """Return the centre of a red bottle cap on the pale table.

    The white logo splits a colour blob into a crescent, and a hand
    above the cap is red too. A circle counts when its inside is red
    and the plastic around it is pale, which a hand, a cable, and the
    wooden floor are not.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, _value = cv2.split(hsv)
    red = ((hue <= 15) | (hue >= 165)) & (saturation >= 70)
    if free_mask is not None:
        red = red & (free_mask > 0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1.2,
        40,
        param1=80,
        param2=18,
        minRadius=22,
        maxRadius=80,
    )
    if circles is None:
        return None
    height, width = gray.shape[:2]
    candidates = []
    for raw_u, raw_v, radius in circles[0]:
        center_u = float(raw_u)
        center_v = float(raw_v)
        radius = float(radius)
        pixel_u = int(round(center_u))
        pixel_v = int(round(center_v))
        if pixel_u < 0 or pixel_v < 0 or pixel_u >= width or pixel_v >= height:
            continue
        if free_mask is not None and free_mask[pixel_v, pixel_u] == 0:
            continue
        if hint is not None:
            hint_u, hint_v = hint
            if math.hypot(center_u - hint_u, center_v - hint_v) > WRIST_LOCK_RADIUS_PX:
                continue
        y0 = max(0, int(center_v - radius * 1.6))
        y1 = min(height, int(center_v + radius * 1.6) + 1)
        x0 = max(0, int(center_u - radius * 1.6))
        x1 = min(width, int(center_u + radius * 1.6) + 1)
        rows, cols = np.ogrid[y0:y1, x0:x1]
        distance = np.hypot(cols - center_u, rows - center_v)
        disc = distance <= radius * 0.80
        ring = (distance >= radius * 1.12) & (distance <= radius * 1.50)
        disc_count = int(np.count_nonzero(disc))
        ring_count = int(np.count_nonzero(ring))
        if disc_count < 80 or ring_count < 40:
            continue
        red_fraction = float(np.count_nonzero(red[y0:y1, x0:x1][disc])) / float(disc_count)
        if red_fraction < 0.70:
            continue
        # Wood is red enough to fill a circle, but it is duller than
        # the plastic cap. The white ledge behind the floor is pale,
        # so the table-ring test alone accepts it.
        disc_saturation = float(np.median(saturation[y0:y1, x0:x1][disc]))
        if disc_saturation < 210.0:
            continue
        ring_saturation = float(np.median(saturation[y0:y1, x0:x1][ring]))
        ring_gray = float(np.median(gray[y0:y1, x0:x1][ring]))
        disc_gray = float(np.median(gray[y0:y1, x0:x1][disc]))
        # The table around a cap is pale. Wood and skin are not.
        if ring_saturation > 45.0 or ring_gray < 120.0:
            continue
        if ring_gray - disc_gray < 30.0:
            continue
        inner = distance <= radius * 0.35
        if int(np.count_nonzero(inner)) < 15:
            continue
        inner_saturation = float(np.median(saturation[y0:y1, x0:x1][inner]))
        candidates.append((
            float(radius),
            inner_saturation,
            center_u,
            center_v,
        ))
    if not candidates:
        return None
    # The white logo sits in the middle. A circle shifted onto the
    # red plastic has a red centre, so prefer a pale centre and then
    # the largest rim. A solid red disc has no pale centre and falls
    # back to the largest circle that still passed.
    pale = [item for item in candidates if item[1] < 90.0]
    pool = pale or candidates
    pool.sort(key=lambda item: item[0], reverse=True)
    chosen = pool[0]
    return chosen[2], chosen[3], chosen[0]


def _round_dark_center(image, free_mask=None, hint=None):
    """Return a round dark bottle cap and ignore cables or shadows."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    # Stay below the cast shadow. A higher cutoff joins the shadow
    # to the lid and the roundness test then rejects the cap.
    threshold = min(median * 0.22, 40.0)
    if threshold < 18.0:
        return None
    mask = np.uint8(gray < threshold) * np.uint8(255)
    if free_mask is not None:
        mask = cv2.bitwise_and(mask, free_mask)

    def accept(center_u, center_v, radius):
        return _black_cap_disc(image, mask, center_u, center_v, radius)

    return _round_blob_center(
        mask, image.shape, minimum_area=180.0, hint=hint, accept=accept,
    )


def _pale_center(image, free_mask=None, hint=None):
    """Return a silver cap: a cap-sized disc slightly darker than the table.

    The lid is too close to the white table for a brightness mask, and the
    red print breaks the disc. A circle counts only when it holds a little
    of that print (not a solid red cap) and is darker than the ring around it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255))
    red = cv2.bitwise_or(red, cv2.inRange(hsv, (168, 90, 60), (180, 255, 255)))
    if free_mask is not None:
        red = cv2.bitwise_and(red, free_mask)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1.2,
        50,
        param1=80,
        param2=18,
        minRadius=22,
        maxRadius=95,
    )
    if circles is None:
        return None
    height, width = gray.shape[:2]
    best = None
    for raw_u, raw_v, radius in circles[0]:
        center_u = float(raw_u)
        center_v = float(raw_v)
        radius = float(radius)
        pixel_u = int(round(center_u))
        pixel_v = int(round(center_v))
        if pixel_u < 0 or pixel_v < 0 or pixel_u >= width or pixel_v >= height:
            continue
        if free_mask is not None and free_mask[pixel_v, pixel_u] == 0:
            continue
        if hint is not None:
            hint_u, hint_v = hint
            if math.hypot(center_u - hint_u, center_v - hint_v) > WRIST_LOCK_RADIUS_PX:
                continue
        # Measure in a window around the circle. A full-frame mask per
        # candidate costs more time than the camera frame interval.
        reach = int(math.ceil(radius * 1.35)) + 1
        left = max(0, pixel_u - reach)
        top = max(0, pixel_v - reach)
        right = min(width, pixel_u + reach + 1)
        bottom = min(height, pixel_v + reach + 1)
        window_gray = gray[top:bottom, left:right]
        window_red = red[top:bottom, left:right]
        window_hsv = hsv[top:bottom, left:right]
        rows, cols = np.ogrid[top:bottom, left:right]
        distance = np.hypot(cols - center_u, rows - center_v)
        disc = distance <= radius * 0.85
        disc_count = int(np.count_nonzero(disc))
        if disc_count < 80:
            continue
        print_fraction = (
            float(np.count_nonzero(window_red[disc])) / float(disc_count)
        )
        if print_fraction < 0.06 or print_fraction > 0.35:
            continue
        ring = (distance >= radius * 1.05) & (distance <= radius * 1.35)
        if int(np.count_nonzero(ring)) < 40:
            continue
        disc_median = float(np.median(window_gray[disc]))
        ring_median = float(np.median(window_gray[ring]))
        # A black cap is far darker than a silver lid in shadow.
        # A bright table circle is not a lid either.
        if disc_median < 28.0 or disc_median > 110.0:
            continue
        if ring_median - disc_median < 22.0:
            continue
        # Wood and the red mat are saturated. The lid is grey plastic.
        plain = disc & (window_red == 0)
        if int(np.count_nonzero(plain)) < 40:
            continue
        if float(np.median(window_hsv[plain][:, 1])) > 120.0:
            continue
        if float(np.median(window_hsv[ring][:, 1])) > 90.0:
            continue
        # The red print is the logo in the middle of the lid. Red at
        # the rim is a neighbouring object, not the cap.
        red_rows, red_cols = np.where(disc & (window_red > 0))
        if len(red_cols) == 0:
            continue
        logo_offset = math.hypot(
            float(np.mean(red_cols)) + left - center_u,
            float(np.mean(red_rows)) + top - center_v,
        ) / radius
        if logo_offset > 0.40:
            continue
        inner = distance <= radius * 0.45
        edge = (distance >= radius * 0.70) & disc
        inner_count = int(np.count_nonzero(inner))
        edge_count = int(np.count_nonzero(edge))
        if inner_count < 20 or edge_count < 20:
            continue
        inner_red = float(np.count_nonzero(window_red[inner])) / float(inner_count)
        edge_red = float(np.count_nonzero(window_red[edge])) / float(edge_count)
        if inner_red < 0.15 or edge_red > 0.12:
            continue
        if best is None or print_fraction > best[0]:
            best = (print_fraction, center_u, center_v, radius)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _prepare_blob_mask(mask):
    """Close glare holes, then drop single-pixel noise."""
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)


def _round_blob_center(mask, shape, minimum_area=120.0, hint=None, accept=None):
    """Return the enclosing-circle centre of the round cap blob."""
    mask = _prepare_blob_mask(mask)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates = []
    max_area = float(shape[0] * shape[1]) * 0.18
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area or area > max_area:
            continue
        # The logo bites a crescent out of the colour mask. The
        # enclosing circle of that crescent sits on the printed side.
        # The convex hull restores the middle of the round cap.
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        perimeter = cv2.arcLength(hull, True)
        if perimeter < 1.0 or hull_area < 1.0:
            continue
        circularity = 4.0 * math.pi * hull_area / (perimeter * perimeter)
        if circularity < WRIST_MIN_CIRCULARITY:
            continue
        (center_u, center_v), radius = cv2.minEnclosingCircle(hull)
        if radius < 1.0:
            continue
        diameter = 2.0 * float(radius)
        if diameter < WRIST_MIN_DIAMETER_PX or diameter > WRIST_MAX_DIAMETER_PX:
            continue
        fill = hull_area / (math.pi * float(radius) * float(radius))
        if fill < WRIST_MIN_FILL:
            continue
        _box_center, size, _angle = cv2.minAreaRect(hull)
        short_side = min(float(size[0]), float(size[1]))
        long_side = max(float(size[0]), float(size[1]))
        # A finger can hide one side of the cap. The hull is then a
        # thick segment, still far rounder than a cable.
        if short_side < 10.0 or long_side / max(short_side, 1e-6) > 1.75:
            continue
        if accept is not None and not accept(
            float(center_u), float(center_v), float(radius),
        ):
            continue
        candidates.append((
            float(center_u),
            float(center_v),
            circularity * math.sqrt(area),
            float(radius),
        ))
    if not candidates:
        return None
    if hint is not None:
        near = [
            item for item in candidates
            if math.hypot(item[0] - float(hint[0]), item[1] - float(hint[1]))
            <= WRIST_LOCK_RADIUS_PX
        ]
        if not near:
            return None
        candidates = near
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0][0], candidates[0][1], candidates[0][3]


def _largest_blob_center(mask, minimum_area):
    """Return the centroid of the largest contour in a mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area or area <= best_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        best = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        best_area = area
    return best


def finger_midpoint(image, pixel_u, pixel_v, box_width, box_height):
    """Return the midpoint of the two open fingers near a cap centre.

    Elongated dark blobs are the fingers. A round dark blob, such as
    the cap itself, is ignored. Returns None when the pair is not visible.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    span = max(float(box_width), float(box_height), 40.0)
    half = int(round(span * 1.5))
    half = min(160, max(half, 48))
    center_u = int(round(pixel_u))
    center_v = int(round(pixel_v))
    x0 = max(0, center_u - half)
    y0 = max(0, center_v - half)
    x1 = min(width, center_u + half)
    y1 = min(height, center_v + half)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    threshold = min(median - 35.0, 100.0)
    if threshold < 20.0:
        return None
    dark = np.uint8(gray < threshold) * np.uint8(255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(
        dark,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    window_area = float((x1 - x0) * (y1 - y0))
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80.0 or area > window_area * 0.35:
            continue
        if len(contour) < 3:
            continue
        _center, size, _angle = cv2.minAreaRect(contour)
        short_side = min(float(size[0]), float(size[1]))
        long_side = max(float(size[0]), float(size[1]))
        if short_side < 1.0 or long_side / short_side < 1.8:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        candidates.append(
            (
                moments["m10"] / moments["m00"] + x0,
                moments["m01"] / moments["m00"] + y0,
            )
        )
    if len(candidates) < 2:
        return None
    min_gap = max(12.0, 0.35 * span)
    best = None
    best_distance = None
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            gap = math.hypot(first[0] - second[0], first[1] - second[1])
            if gap < min_gap:
                continue
            mid_u = (first[0] + second[0]) / 2.0
            mid_v = (first[1] + second[1]) / 2.0
            distance = math.hypot(mid_u - pixel_u, mid_v - pixel_v)
            if distance > half:
                continue
            if best is None or distance < best_distance:
                best = (mid_u, mid_v)
                best_distance = distance
    return best


def _solve(matrix, rhs):
    """Solve a 3x3 system whose right-hand side has two columns."""
    rows = [matrix[index][:] + rhs[index][:] for index in range(3)]
    for column in range(3):
        pivot = max(
            range(column, 3),
            key=lambda index: abs(rows[index][column]),
        )
        if abs(rows[pivot][column]) < 1e-9:
            raise ValueError(
                "Calibration points do not define a plane mapping."
            )
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        for index in range(column, 5):
            rows[column][index] /= scale
        for row_index in range(3):
            if row_index == column:
                continue
            factor = rows[row_index][column]
            for index in range(column, 5):
                rows[row_index][index] -= factor * rows[column][index]
    return [[rows[index][3], rows[index][4]] for index in range(3)]


def _load_calibration(path):
    """Read the table calibration yaml."""
    with open(path, "r", encoding="utf-8") as calibration_file:
        loaded = yaml.safe_load(calibration_file) or {}
    data = loaded.get("table_calibration", loaded)
    if not bool(data.get("accepted")):
        raise ValueError(
            "Calibration has not been accepted. Picking is disabled."
        )
    pairs = list(data.get("pairs") or [])
    measured_keys = (
        "table_z_mm",
        "hover_mm",
        "drop_x_mm",
        "drop_y_mm",
    )
    missing = [key for key in measured_keys if data.get(key) is None]
    if missing or len(pairs) < 3:
        parts = []
        if missing:
            parts.append("Unmeasured: " + ", ".join(missing) + ".")
        if len(pairs) < 3:
            parts.append(
                "Need at least three measured pixel to "
                "tool_pose_raw pairs."
            )
        raise ValueError(" ".join(parts) + " Picking is disabled.")
    loaded_values = {
        key: _finite_float(data, key) for key in measured_keys
    }
    loaded_values.update(
        {
            "tool_rx": _finite_float(data, "tool_rx"),
            "tool_ry": _finite_float(data, "tool_ry"),
            "tool_rz": _finite_float(data, "tool_rz"),
            "speed": int(data.get("speed", 20)),
            "pairs": pairs,
        }
    )
    for key in (
        "drop_red_x_mm",
        "drop_red_y_mm",
        "drop_black_x_mm",
        "drop_black_y_mm",
        "drop_white_x_mm",
        "drop_white_y_mm",
    ):
        if data.get(key) is not None:
            loaded_values[key] = float(data[key])
        else:
            loaded_values[key] = None
    return loaded_values


def _finite_float(data, key):
    """Return one finite calibration number."""
    value = float(data[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite.")
    return value


def _camera_qos():
    """Match the overview camera's best-effort image info."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _ready_qos():
    """Match the detector's latched weights-ready flag."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _configured_file(node, parameter_name, file_name):
    """Return a configured config path, or the installed default."""
    configured = str(node.get_parameter(parameter_name).value)
    if configured:
        return configured
    share = get_package_share_directory("mycobot_vision")
    return os.path.join(share, "config", file_name)


def load_table_preview(homography_path, transform_path):
    """Load the pixel homography and the provisional joint1 transform."""
    with open(homography_path, "r", encoding="utf-8") as handle:
        homography = yaml.safe_load(handle)
    with open(transform_path, "r", encoding="utf-8") as handle:
        transform = yaml.safe_load(handle)
    if homography.get("image_topic") != "/overview_camera/image_rect":
        raise ValueError(
            "Homography is not calibrated for image_rect."
        )
    matrix = _matrix3(homography["pixel_to_table_homography"])
    width = int(homography["image_width_px"])
    height = int(homography["image_height_px"])
    if transform.get("parent_frame") != "joint1":
        raise ValueError("Table transform parent is not joint1.")
    if transform.get("child_frame") != "table_frame":
        raise ValueError("Table transform child is not table_frame.")
    translation = [
        _finite_float(transform["translation"], axis)
        for axis in ("x", "y", "z")
    ]
    quaternion = [
        _finite_float(transform["rotation"], axis)
        for axis in ("x", "y", "z", "w")
    ]
    return {
        "matrix": matrix,
        "width": width,
        "height": height,
        "translation": translation,
        "rotation": _quaternion_matrix(quaternion),
    }


def project_table_preview(
    model,
    pixel_u,
    pixel_v,
    image_width,
    image_height,
):
    """Map one source-image center to table metres and joint1 metres."""
    if image_width is None or image_height is None:
        raise ValueError(
            "Waiting for overview camera info before converting."
        )
    if (
        int(image_width) != model["width"]
        or int(image_height) != model["height"]
    ):
        raise ValueError(
            f"Camera is {image_width}x{image_height}; "
            f"homography is {model['width']}x{model['height']}."
        )
    pixel_u = float(pixel_u)
    pixel_v = float(pixel_v)
    if not (
        0.0 <= pixel_u <= model["width"]
        and 0.0 <= pixel_v <= model["height"]
    ):
        raise ValueError(
            "Detection center is outside the calibrated image."
        )
    table_x, table_y = _apply_homography(
        model["matrix"],
        pixel_u,
        pixel_v,
    )
    local = [table_x, table_y, 0.0]
    rotated = _matvec(model["rotation"], local)
    joint1 = [
        model["translation"][index] + rotated[index]
        for index in range(3)
    ]
    return {
        "pixel_u": round(pixel_u, 1),
        "pixel_v": round(pixel_v, 1),
        "table_x_m": round(table_x, 4),
        "table_y_m": round(table_y, 4),
        "joint1_x_m": round(joint1[0], 4),
        "joint1_y_m": round(joint1[1], 4),
        "joint1_z_m": round(joint1[2], 4),
    }


def _matrix3(section):
    """Return a finite 3x3 matrix stored in row-major order."""
    if int(section["rows"]) != 3 or int(section["cols"]) != 3:
        raise ValueError("Homography must be 3x3.")
    values = [float(value) for value in section["data"]]
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise ValueError("Homography data must be nine finite values.")
    return [values[0:3], values[3:6], values[6:9]]


def _apply_homography(matrix, pixel_u, pixel_v):
    """Apply a pixel-to-table homography. The result is metres."""
    point = [pixel_u, pixel_v, 1.0]
    mapped = _matvec(matrix, point)
    if abs(mapped[2]) < 1.0e-12:
        raise ValueError("Homography produced a point at infinity.")
    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def _quaternion_matrix(quaternion):
    """Return the rotation matrix for an x, y, z, w quaternion."""
    x_value, y_value, z_value, w_value = quaternion
    norm = math.sqrt(
        x_value * x_value
        + y_value * y_value
        + z_value * z_value
        + w_value * w_value
    )
    if norm < 1.0e-9:
        raise ValueError("Table quaternion is zero.")
    x_value /= norm
    y_value /= norm
    z_value /= norm
    w_value /= norm
    return [
        [
            1.0 - 2.0 * (y_value * y_value + z_value * z_value),
            2.0 * (x_value * y_value - z_value * w_value),
            2.0 * (x_value * z_value + y_value * w_value),
        ],
        [
            2.0 * (x_value * y_value + z_value * w_value),
            1.0 - 2.0 * (x_value * x_value + z_value * z_value),
            2.0 * (y_value * z_value - x_value * w_value),
        ],
        [
            2.0 * (x_value * z_value - y_value * w_value),
            2.0 * (y_value * z_value + x_value * w_value),
            1.0 - 2.0 * (x_value * x_value + y_value * y_value),
        ],
    ]


def _matvec(matrix, vector):
    """Multiply a 3x3 matrix by a 3-vector."""
    return [
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def _load_wizard(path):
    """Load saved spots. A missing file starts an empty wizard."""
    wizard = {
        "accepted": False,
        "pending_pixel": None,
        "pairs": [],
        "check": None,
        "table_z_mm": None,
        "hover_mm": None,
        "drop_x_mm": None,
        "drop_y_mm": None,
        "drop_red_x_mm": None,
        "drop_red_y_mm": None,
        "drop_black_x_mm": None,
        "drop_black_y_mm": None,
        "drop_white_x_mm": None,
        "drop_white_y_mm": None,
        "tool_rx": -180.0,
        "tool_ry": 0.0,
        "tool_rz": 0.0,
        "speed": 20,
    }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except OSError:
        return wizard
    data = loaded.get("table_calibration", loaded) or {}
    wizard["pairs"] = [
        _spot(pair) for pair in (data.get("pairs") or [])
    ][:FIT_SPOT_COUNT]
    if data.get("check"):
        wizard["check"] = _spot(data["check"])
    for key in (
        "table_z_mm",
        "hover_mm",
        "drop_x_mm",
        "drop_y_mm",
        "drop_red_x_mm",
        "drop_red_y_mm",
        "drop_black_x_mm",
        "drop_black_y_mm",
        "drop_white_x_mm",
        "drop_white_y_mm",
    ):
        if data.get(key) is not None:
            wizard[key] = float(data[key])
    for key in ("tool_rx", "tool_ry", "tool_rz"):
        if data.get(key) is not None:
            wizard[key] = float(data[key])
    if data.get("speed") is not None:
        wizard["speed"] = int(data["speed"])
    wizard["accepted"] = bool(data.get("accepted"))
    return wizard


def _spot(pair):
    """Return one stored pixel and tool-pose sample."""
    spot = {
        "u": float(pair["u"]),
        "v": float(pair["v"]),
        "x": float(pair["x"]),
        "y": float(pair["y"]),
    }
    if pair.get("z") is not None:
        spot["z"] = float(pair["z"])
    return spot


def _calibration_from_wizard(wizard):
    """Build a pick calibration dict from the guided wizard fields."""
    pairs = list(wizard.get("pairs") or [])
    if len(pairs) < FIT_SPOT_COUNT:
        raise ValueError(
            f"Need {FIT_SPOT_COUNT} fit spots before accepting."
        )
    if wizard.get("check") is None:
        raise ValueError("Save a check spot before accepting.")
    for key in ("table_z_mm", "hover_mm", "drop_x_mm", "drop_y_mm"):
        if wizard.get(key) is None:
            raise ValueError(
                "Save table height, hover clearance, and drop "
                "before accepting."
            )
    return {
        "table_z_mm": float(wizard["table_z_mm"]),
        "hover_mm": float(wizard["hover_mm"]),
        "drop_x_mm": float(wizard["drop_x_mm"]),
        "drop_y_mm": float(wizard["drop_y_mm"]),
        "drop_red_x_mm": (
            None if wizard.get("drop_red_x_mm") is None
            else float(wizard["drop_red_x_mm"])
        ),
        "drop_red_y_mm": (
            None if wizard.get("drop_red_y_mm") is None
            else float(wizard["drop_red_y_mm"])
        ),
        "drop_black_x_mm": (
            None if wizard.get("drop_black_x_mm") is None
            else float(wizard["drop_black_x_mm"])
        ),
        "drop_black_y_mm": (
            None if wizard.get("drop_black_y_mm") is None
            else float(wizard["drop_black_y_mm"])
        ),
        "drop_white_x_mm": (
            None if wizard.get("drop_white_x_mm") is None
            else float(wizard["drop_white_x_mm"])
        ),
        "drop_white_y_mm": (
            None if wizard.get("drop_white_y_mm") is None
            else float(wizard["drop_white_y_mm"])
        ),
        "tool_rx": float(wizard.get("tool_rx", -176.0)),
        "tool_ry": float(wizard.get("tool_ry", 0.0)),
        "tool_rz": float(wizard.get("tool_rz", -88.0)),
        "speed": int(wizard.get("speed", 12)),
        "pairs": pairs,
        "check": wizard.get("check"),
        "accepted": True,
    }


def _write_wizard(path, wizard, accepted=False):
    """Save the guided calibration yaml."""
    payload = {
        "table_calibration": {
            "accepted": bool(accepted),
            "table_z_mm": wizard.get("table_z_mm"),
            "hover_mm": wizard.get("hover_mm"),
            "tool_rx": wizard.get("tool_rx", -176.0),
            "tool_ry": wizard.get("tool_ry", 0.0),
            "tool_rz": wizard.get("tool_rz", -88.0),
            "drop_x_mm": wizard.get("drop_x_mm"),
            "drop_y_mm": wizard.get("drop_y_mm"),
            "drop_red_x_mm": wizard.get("drop_red_x_mm"),
            "drop_red_y_mm": wizard.get("drop_red_y_mm"),
            "drop_black_x_mm": wizard.get("drop_black_x_mm"),
            "drop_black_y_mm": wizard.get("drop_black_y_mm"),
            "drop_white_x_mm": wizard.get("drop_white_x_mm"),
            "drop_white_y_mm": wizard.get("drop_white_y_mm"),
            "speed": wizard.get("speed", 12),
            "pairs": wizard.get("pairs") or [],
            "check": wizard.get("check"),
        }
    }
    if accepted:
        text = (
            "# Accepted for picking. Restart is not required; "
            "the pick node reloads this file on Accept.\n"
        )
    else:
        text = (
            "# Guided calibration. Recording does not move the arm.\n"
            "# accepted stays false, so picking stays disabled.\n"
        )
    text += (
        "# u and v are image pixels. x, y and z are millimetres\n"
        "# from /mycobot/tool_pose_raw. The check spot is not fitted.\n"
    )
    text += yaml.safe_dump(payload, sort_keys=False)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _nearby_spot(spots, candidate):
    """Return a 1-based index when a spot repeats a saved location."""
    for index, spot in enumerate(spots, start=1):
        pixel = math.hypot(
            candidate["u"] - spot["u"],
            candidate["v"] - spot["v"],
        )
        robot = math.hypot(
            candidate["x"] - spot["x"],
            candidate["y"] - spot["y"],
        )
        if pixel < MIN_PIXEL_SPACING or robot < MIN_ROBOT_SPACING_MM:
            return index
    return None


def _calibration_span(pairs):
    """Return how far the fit spots cover in image pixels."""
    if not pairs:
        return {"span_u": 0.0, "span_v": 0.0}
    us = [float(pair["u"]) for pair in pairs]
    vs = [float(pair["v"]) for pair in pairs]
    return {
        "span_u": max(us) - min(us),
        "span_v": max(vs) - min(vs),
    }


def _aruco_dictionary():
    """Return the OpenCV ArUco dictionary used for table calibration."""
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _aruco_parameters():
    """Return detector parameters for OpenCV 4.5."""
    return cv2.aruco.DetectorParameters_create()


def detect_overview_aruco(image, marker_id=ARUCO_MARKER_ID):
    """Return (u, v, id) for the calibration marker, or None."""
    if image is None or image.size == 0:
        return None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    corners, ids, _rejected = cv2.aruco.detectMarkers(
        gray,
        _aruco_dictionary(),
        parameters=_aruco_parameters(),
    )
    if ids is None or len(ids) == 0:
        return None
    for index, found_id in enumerate(ids.flatten().tolist()):
        if int(found_id) != int(marker_id):
            continue
        points = corners[index][0]
        center_u = float(points[:, 0].mean())
        center_v = float(points[:, 1].mean())
        return center_u, center_v, int(found_id)
    return None


def _mark_overview_aruco(image, marker_id=ARUCO_MARKER_ID):
    """Draw the calibration ArUco outline when it is visible."""
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = cv2.aruco.detectMarkers(
        gray,
        _aruco_dictionary(),
        parameters=_aruco_parameters(),
    )
    if ids is None or len(ids) == 0:
        return image
    marked = image.copy()
    cv2.aruco.drawDetectedMarkers(marked, corners, ids)
    for index, found_id in enumerate(ids.flatten().tolist()):
        if int(found_id) != int(marker_id):
            continue
        points = corners[index][0]
        center = (
            int(round(float(points[:, 0].mean()))),
            int(round(float(points[:, 1].mean()))),
        )
        cv2.drawMarker(
            marked,
            center,
            (0, 165, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            marked,
            f"aruco {found_id}",
            (center[0] + 8, max(16, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    return marked


def _aruco_marker_png(marker_id=ARUCO_MARKER_ID, size_px=400, border_px=60):
    """Encode a printable ArUco marker as PNG bytes."""
    try:
        marker = cv2.aruco.drawMarker(
            _aruco_dictionary(),
            int(marker_id),
            int(size_px),
        )
    except Exception:
        return None
    canvas = np.full(
        (size_px + 2 * border_px, size_px + 2 * border_px),
        255,
        dtype=np.uint8,
    )
    canvas[
        border_px: border_px + size_px,
        border_px: border_px + size_px,
    ] = marker
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        return None
    return encoded.tobytes()


def overview_pad_center(image):
    """Return the open-gripper pad centre in an overview frame.

    Looks for a pair of dark finger tips (the adaptive pads) and
    returns their midpoint. Used by auto-calibration so the operator
    does not have to click the image.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ignore the far top where the arm base dominates.
    y0 = int(height * 0.12)
    crop = gray[y0:, :]
    median = float(np.median(crop))
    thresholds = [min(median - delta, 100.0) for delta in (20.0, 28.0, 35.0, 45.0)]
    thresholds.extend([45.0, 55.0, 70.0, 85.0, 100.0])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    x_cable = int(width * 0.14)
    best = None
    best_score = None

    def _ok(u, v):
        return (
            width * 0.16 < u < width * 0.84
            and height * 0.16 < v < height * 0.88
        )

    for threshold in thresholds:
        if threshold < 12.0:
            continue
        mask = np.uint8(crop < threshold) * np.uint8(255)
        mask[:, :x_cable] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        tips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 40.0 or area > 3500.0:
                continue
            _center, size, _angle = cv2.minAreaRect(contour)
            short_side = min(float(size[0]), float(size[1]))
            long_side = max(float(size[0]), float(size[1]))
            if short_side < 3.5:
                continue
            # Pads are compact; long cables are rejected.
            if long_side / short_side > 3.2:
                continue
            points = contour[:, 0, :]
            lowest = points[points[:, 1] >= points[:, 1].max() - 2]
            tip_u = float(lowest[:, 0].mean())
            tip_v = float(lowest[:, 1].mean()) + y0
            if not _ok(tip_u, tip_v):
                continue
            tips.append((tip_u, tip_v, area))
        if len(tips) < 2:
            continue
        for index, first in enumerate(tips):
            for second in tips[index + 1:]:
                gap = math.hypot(first[0] - second[0], first[1] - second[1])
                if gap < 14.0 or gap > 90.0:
                    continue
                # Pads sit side-by-side more than stacked.
                if abs(first[1] - second[1]) > 40.0:
                    continue
                mid_u = (first[0] + second[0]) / 2.0
                mid_v = (first[1] + second[1]) / 2.0
                if not _ok(mid_u, mid_v):
                    continue
                area_ratio = max(first[2], second[2]) / max(min(first[2], second[2]), 1.0)
                if area_ratio > 4.0:
                    continue
                score = (first[2] + second[2]) / (
                    1.0 + abs(first[1] - second[1]) + abs(gap - 35.0)
                )
                score -= area_ratio * 15.0
                if best is None or score > best_score:
                    best = (mid_u, mid_v)
                    best_score = score
        if best is not None:
            return best
    return best


def _check_report(wizard):
    """Predict the check spot and the motion target without sending it."""
    pairs = wizard.get("pairs") or []
    check = wizard.get("check")
    if (
        len(pairs) < FIT_SPOT_COUNT
        or not check
        or wizard.get("table_z_mm") is None
        or wizard.get("hover_mm") is None
        or wizard.get("drop_x_mm") is None
        or wizard.get("drop_y_mm") is None
    ):
        return None
    try:
        mapping = fit_table(pairs)
    except ValueError as error:
        return {"motion_command": False, "message": str(error)}
    predicted_x, predicted_y = project_table(
        mapping,
        check["u"],
        check["v"],
    )
    table_z = float(wizard["table_z_mm"])
    hover = float(wizard["hover_mm"])
    return {
        "motion_command": False,
        "predicted_x_mm": round(predicted_x, 2),
        "predicted_y_mm": round(predicted_y, 2),
        "measured_x_mm": round(float(check["x"]), 2),
        "measured_y_mm": round(float(check["y"]), 2),
        "error_mm": round(
            math.hypot(
                predicted_x - float(check["x"]),
                predicted_y - float(check["y"]),
            ),
            2,
        ),
        "grasp_x_mm": round(predicted_x, 2),
        "grasp_y_mm": round(predicted_y, 2),
        "grasp_z_mm": round(table_z, 2),
        "hover_z_mm": round(table_z + hover, 2),
        "drop_x_mm": round(float(wizard["drop_x_mm"]), 2),
        "drop_y_mm": round(float(wizard["drop_y_mm"]), 2),
        "drop_z_mm": round(table_z, 2),
        "fit_residuals": pair_fit_residuals(
            pairs,
            lambda u, v: project_table(mapping, u, v),
        ),
        "spot4_remake_hint": (
            "Remake fit spot 4 at table z ≈ "
            f"{table_z:.0f} mm if its residual exceeds "
            f"{MAX_FIT_SPOT_RESIDUAL_MM:.0f} mm."
        ),
    }


def _default_calibration_file():
    """Return the installed calibration file."""
    share = get_package_share_directory("mycobot_vision")
    return os.path.join(share, "config", "table_calibration.yaml")


def _calibrate_page_path():
    """Locate the calibration page."""
    return _share_or_local("calibrate_page.html")


def _page_path():
    """Locate the installed page, or the copy next to this script."""
    return _share_or_local("pick_page.html")


def _share_or_local(file_name):
    """Locate an installed page, or the copy next to this script."""
    try:
        share = get_package_share_directory("mycobot_vision")
    except PackageNotFoundError:
        share = ""
    installed = os.path.join(share, file_name) if share else ""
    if installed and os.path.isfile(installed):
        return installed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(script_dir, file_name),
        os.path.join(script_dir, "..", file_name),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[-1]


def _stopped_by_user(body):
    """Return true when the operator pressed Stop."""
    message = str(body.get("message", "")).casefold()
    # Do not match "stopped on the cap" / similar gripper text.
    markers = (
        "stop requested",
        "stopped by user",
        "user pressed stop",
        "motion stopped by operator",
    )
    return any(marker in message for marker in markers)


def _smootherstep(blend):
    """Ease a 0–1 blend so speed starts and ends at zero."""
    blend = min(1.0, max(0.0, blend))
    return blend * blend * blend * (
        blend * (blend * 6.0 - 15.0) + 10.0
    )


def _travel_duration(origin, goal):
    """Return seconds for one eased move from the largest joint change."""
    span = _max_abs_diff(origin, goal)
    if span < 0.2:
        return 0.0
    return max(ZERO_MIN_TRAVEL_SECONDS, span / ZERO_TRAVEL_RATE)


def _eased_targets(origin, goal, elapsed, duration):
    """Return eased joint targets for this moment in the move."""
    if duration <= 0.0:
        return [round(angle, 3) for angle in goal]
    blend = min(1.0, elapsed / duration)
    eased = _smootherstep(blend)
    return [
        round(start + (end - start) * eased, 3)
        for start, end in zip(origin, goal)
    ]


def _limit_step(current, desired, max_step):
    """Move each joint toward its target by at most max_step degrees."""
    limited = []
    for start, target in zip(current, desired):
        delta = target - start
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step
        limited.append(round(start + delta, 3))
    return limited


def _max_abs_diff(left, right):
    """Return the largest absolute joint difference."""
    return max(
        abs(first - second)
        for first, second in zip(left, right)
    )


def _motion_halted(body):
    """Return true when Stop ended the joint move."""
    message = str(body.get("message", "")).lower()
    return (
        "stopped" in message
        or "stop is latched" in message
        or "stop latched" in message
    )


def refine_cap_center(image, class_name, pixel_u, pixel_v, size_x, size_y):
    """Return the disc centre for a real round cap, or the box centre."""
    disc = _cap_disc_center(
        image,
        class_name,
        pixel_u,
        pixel_v,
        size_x,
        size_y,
    )
    if disc is not None:
        return disc
    return float(pixel_u), float(pixel_v)


def _cap_disc_center(image, class_name, pixel_u, pixel_v, size_x, size_y):
    """Return the disc centre for a round cap on the white table."""
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
        return None
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
        (circle_x, circle_y), _radius = cv2.minEnclosingCircle(contour)
        center_u = x0 + float(circle_x)
        center_v = y0 + float(circle_y)
        if math.hypot(center_u - pixel_u, center_v - pixel_v) > span * 0.5:
            continue
        if best is None or area > best[0]:
            best = (area, center_u, center_v)
    if best is None:
        return None
    if name == "black_cap":
        center_u, center_v = best[1], best[2]
        margin = 36.0
        if (
            center_u < margin
            or center_v < margin
            or center_u > width - margin
            or center_v > height - margin
            or center_v > height * 0.88
        ):
            return None
        circle = np.zeros(gray_full.shape, np.uint8)
        cv2.circle(
            circle,
            (int(round(center_u)), int(round(center_v))),
            12,
            255,
            -1,
        )
        pixels = gray_full[circle > 0]
        if pixels.size < 20:
            return None
        if float(np.median(pixels)) > 22.0 or float(np.mean(pixels < 40.0)) < 0.70:
            return None
    return best[1], best[2]


def _circle_matches_cap(roi, class_name, center_x, center_y, radius):
    """Return true when the circle interior looks like that cap colour."""
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(center_x)), int(round(center_y))),
        max(3, int(round(radius * 0.65))),
        255,
        -1,
    )
    if cv2.countNonZero(mask) < 20:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pixels = hsv[mask > 0]
    shades = gray[mask > 0]
    red = (
        (pixels[:, 0] < 12) | (pixels[:, 0] > 165)
    ) & (pixels[:, 1] > 80)
    red_fraction = float(np.mean(red))
    median = float(np.median(shades))
    spread = float(np.std(shades))
    name = str(class_name or "").casefold()
    if name == "red_cap":
        return red_fraction > 0.35
    if name == "black_cap":
        return median < 40.0 and red_fraction < 0.15
    if name == "white_cap":
        # The silver cap carries a printed logo, so it is brighter
        # than the black cap and less flat than the bare table.
        return (
            70.0 < median < 190.0
            and spread > 12.0
            and red_fraction < 0.5
        )
    return False


def red_disc_center(image, pixel_u, pixel_v, size_x, size_y):
    """Return the centre and area of the red cap disc, or None."""
    if image is None or image.size == 0:
        return None
    span = max(float(size_x), float(size_y), 28.0)
    half = max(span, 36.0)
    height, width = image.shape[:2]
    x0 = int(max(0, round(pixel_u - half)))
    y0 = int(max(0, round(pixel_v - half)))
    x1 = int(min(width, round(pixel_u + half)))
    y1 = int(min(height, round(pixel_v + half)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 50), (12, 255, 255))
    mask = cv2.bitwise_or(
        mask,
        cv2.inRange(hsv, (168, 90, 50), (180, 255, 255)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80.0 or area > 4500.0:
            continue
        _center, size, _angle = cv2.minAreaRect(contour)
        short_side = min(float(size[0]), float(size[1]))
        long_side = max(float(size[0]), float(size[1]))
        if short_side < 8.0 or long_side / short_side > 1.7:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        center_u = x0 + moments["m10"] / moments["m00"]
        center_v = y0 + moments["m01"] / moments["m00"]
        distance = math.hypot(center_u - pixel_u, center_v - pixel_v)
        if distance > half:
            continue
        if best is None or area > best[0]:
            best = (area, center_u, center_v)
    if best is None:
        return None
    return best[1], best[2], best[0]


def gripper_beside_cap(image, cap_u, cap_v):
    """Return the white jaw when it is already beside the cap.

    A jaw that only appears high above the cap is the arm in perspective,
    not a sideways miss, so this returns None in that case.
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x0 = max(0, int(round(cap_u)) - 70)
    x1 = min(width, int(round(cap_u)) + 70)
    y0 = max(0, int(round(cap_v)) - 80)
    y1 = min(height, int(round(cap_v)) + 30)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    mask = np.uint8(gray[y0:y1, x0:x1] > 200) * np.uint8(255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best = None
    for contour in contours:
        if cv2.contourArea(contour) < 80.0:
            continue
        points = contour[:, 0, :]
        lowest = int(points[:, 1].max())
        bottom = points[points[:, 1] >= lowest - 3]
        tip_u = x0 + float(bottom[:, 0].mean())
        tip_v = y0 + float(bottom[:, 1].mean())
        if tip_v < cap_v - 22.0:
            continue
        if abs(tip_u - cap_u) > 45.0 or abs(tip_v - cap_v) > 28.0:
            continue
        distance = math.hypot(tip_u - cap_u, tip_v - cap_v)
        if best is None or distance < best[0]:
            best = (distance, tip_u, tip_v)
    if best is None:
        return None
    return best[1], best[2]


def find_yellow_notes(image):
    """Return yellow sticky-note boxes, left to right.

    Each note is a dict with pixel centre u, v and the box x, y, w, h.
    """
    if image is None or image.size == 0:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (NOTE_HUE_MIN, NOTE_SAT_MIN, NOTE_VAL_MIN),
        (NOTE_HUE_MAX, NOTE_SAT_MAX, 255),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8),
    )
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(image.shape[0] * image.shape[1])
    notes = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < NOTE_MIN_AREA_FRAC * image_area:
            continue
        if area > NOTE_MAX_AREA_FRAC * image_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < NOTE_MIN_SIDE_PX or height < NOTE_MIN_SIDE_PX:
            continue
        aspect = float(width) / float(height)
        if aspect < NOTE_ASPECT_MIN or aspect > NOTE_ASPECT_MAX:
            continue
        if area < 0.55 * float(width * height):
            continue
        notes.append({
            "u": float(x) + float(width) / 2.0,
            "v": float(y) + float(height) / 2.0,
            "x": int(x),
            "y": int(y),
            "w": int(width),
            "h": int(height),
        })
    notes.sort(key=lambda note: note["u"])
    return notes


def prepare_note_ink(crop):
    """Return black writing on white, or None when the note has no ink."""
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    median = float(np.median(gray))
    dark = gray < (median - 35.0)
    not_colored = hsv[:, :, 1] < 140
    mask = np.where(dark & not_colored, 255, 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, 8,
    )
    limit = max(180, int(0.12 * crop.shape[0] * crop.shape[1]))
    kept = np.zeros_like(mask)
    ink_pixels = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8 or area > limit:
            continue
        kept[labels == label] = 255
        ink_pixels += area
    if ink_pixels < 12:
        return None
    ink = cv2.bitwise_not(kept)
    ink = cv2.resize(
        ink, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST,
    )
    return cv2.copyMakeBorder(
        ink, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255,
    )


def tesseract_read(image):
    """Read one line of letters. Empty when tesseract is missing."""
    if image is None or getattr(image, "size", 0) == 0:
        return ""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return ""
    try:
        completed = subprocess.run(
            [
                "tesseract",
                "stdin",
                "stdout",
                "--psm",
                "7",
                "-c",
                "tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ],
            input=encoded.tobytes(),
            capture_output=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.decode("utf-8", errors="replace")


def _edit_distance(left, right):
    """Return the letter edits between two short words."""
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[other_index] + 1,
                previous[other_index - 1] + (character != other),
            ))
        previous = current
    return previous[-1]


def classify_note_word(text):
    """Map OCR text to red, black, or white. Uncertain text is None."""
    letters = "".join(
        character for character in str(text or "").upper()
        if character.isalpha()
    )
    targets = ("RED", "BLACK", "WHITE")
    if not letters:
        return None
    if letters in targets:
        return letters.lower()
    contained = [word for word in targets if word in letters]
    if len(contained) == 1 and len(letters) <= len(contained[0]) + 2:
        return contained[0].lower()
    scores = sorted(
        (_edit_distance(letters, word), word) for word in targets
    )
    if scores[0][0] <= 1 and scores[1][0] >= scores[0][0] + 2:
        return scores[0][1].lower()
    return None


NOTE_VLM_MODEL = "hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF"
NOTE_VLM_PROMPT = (
    "Look at this sticky note. Reply with only RED, WHITE, BLACK, "
    "or UNREADABLE."
)


class NoteReaderError(Exception):
    """The laptop could not be asked what a note says."""


def vlm_read_note(crop, url, model=NOTE_VLM_MODEL, timeout=60):
    """Ask the laptop vision model what one note crop says.

    Returns the reply text. Raises NoteReaderError when the laptop
    cannot be reached. The crop's position stays with the caller.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return ""
    success, encoded = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not success:
        return ""
    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": NOTE_VLM_PROMPT,
            "images": [image],
        }],
    }).encode("utf-8")
    endpoint = str(url).rstrip("/") + "/api/chat"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise NoteReaderError(
            "The laptop note reader is not reachable"
        ) from error
    message = payload.get("message") or {}
    return str(message.get("content") or "")


def read_note_label(crop, reader=None):
    """Return red, black, or white for one note crop."""
    prepared = prepare_note_ink(crop)
    if reader is None:
        if prepared is None:
            return None
        text = tesseract_read(prepared)
    else:
        text = reader(crop, prepared)
    return classify_note_word(text)


def match_cap_note(image, class_name, reader=None):
    """Return ((u, v), None) or (None, reason the cap was not dropped)."""
    wanted = CAP_CLASSES.get(class_name)
    if wanted is None:
        return None, "Unknown cap color. The cap was not dropped."
    notes = find_yellow_notes(image)
    if not notes:
        return None, (
            "No yellow sticky notes are visible. "
            "The cap was not dropped."
        )
    by_color = {}
    for note in notes:
        y0 = max(int(note["y"]), 0)
        x0 = max(int(note["x"]), 0)
        y1 = min(y0 + int(note["h"]), image.shape[0])
        x1 = min(x0 + int(note["w"]), image.shape[1])
        crop = image[y0:y1, x0:x1]
        try:
            label = read_note_label(crop, reader=reader)
        except NoteReaderError as error:
            return None, f"{error}. The cap was not dropped."
        if label is None:
            continue
        if label in by_color:
            return None, (
                f"Two sticky notes both read as {label}. "
                "The cap was not dropped."
            )
        by_color[label] = (float(note["u"]), float(note["v"]))
    point = by_color.get(wanted)
    if point is None:
        return None, (
            f"The {wanted} sticky note is not readable. "
            "The cap was not dropped."
        )
    return point, None


def _cap_class(class_name):
    """Return the canonical cap class, or None when it is unknown."""
    name = str(class_name or "").strip().casefold()
    for class_id in CAP_CLASSES:
        if class_id.casefold() == name:
            return class_id
    return None


def _error_payload(message, status=400):
    """Return an HTTP error body."""
    return status, {"success": False, "message": message}


def _service_payload(response, error):
    """Convert a service result into an HTTP status and JSON body."""
    if error:
        return 503, {"success": False, "message": error}
    final_value = getattr(response, "final_value", None)
    if isinstance(final_value, float) and final_value.is_integer():
        final_value = int(final_value)
    final_coords = getattr(response, "final_coords", None)
    coords = None
    if final_coords is not None:
        coords = [round(float(value), 1) for value in list(final_coords)]
    return 200, {
        "success": bool(response.success),
        "message": response.message,
        "final_value": final_value if isinstance(final_value, int) else None,
        "final_coords": coords,
    }


def _call_service(client, request, timeout_seconds):
    """Call a service while the node executor is already spinning."""
    if not client.wait_for_service(timeout_sec=SERVICE_WAIT_SECONDS):
        return None, "Hardware service is not available"
    future = client.call_async(request)
    finished = threading.Event()
    future.add_done_callback(lambda _future: finished.set())
    if not finished.wait(timeout_seconds):
        return None, "Hardware service call timed out"
    exception = future.exception()
    if exception is not None:
        return None, f"Hardware service call failed: {exception}"
    result = future.result()
    if result is None:
        return None, "Hardware service returned no response"
    return result, ""


def main(args=None):
    """Run the pick page until the process is interrupted."""
    rclpy.init(args=args)
    node = PickPlace()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close_server()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
