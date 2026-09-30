# Physical cap detection and approach planning — 2026-09-30

## Completed

- Validated the bottle_caps_v1 YOLO model on 20 test images.
- Test results: precision 0.906, recall 0.914,
  mAP50 0.913 and mAP50-95 0.693.
- Integrated red_cap, black_cap and white_cap detection into ROS 2.
- Configured the physical detector for image size 640 and CPU inference.
- Confirmed ROS, cv_bridge and Ultralytics imports in cap_detection_env.
- Recalibrated the pixel-to-table homography and compared cap
  localization with ruler measurements.
- Added the provisional joint1-to-table_frame transform.
- Added card, cap, grasp and approach visualization in RViz.
- Added the gripper_tcp frame.
- Replaced the gripper_base collision DAE with a generated STL
  and registered its installation in mycobot_description.
- Confirmed a collision-free approach IK solution.
- Generated a planning-only approach trajectory and displayed it in RViz.
- Updated the terminal-only planning experiment to read a live red-cap
  detection instead of using an earlier recorded position.

## Configuration

- PC container workspace: /root/mycobot_humble/mycobot_ws
- Nano container workspace: /root/mycobot_ws
- ROS distribution: Humble
- Robot base frame: joint1
- Detection frame: table_frame
- Detector input: /overview_camera/image_rect
- Table detections: /overview_camera/table_detections
- Visualization markers: /overview_camera/table_markers
- Trajectory preview: /display_planned_path
- Local model: /root/mycobot_humble/models/bottle_caps_v1.pt
- Local Python environment: /root/cap_detection_env

Model weights, datasets and Python environments are local dependencies
and are not included in this source commit.

## Provisional geometry

- Cap diameter: 29–30 mm.
- Cap height: 9–10 mm.
- Proposed pad contact height: 7 mm above the table.
- Proposed approach contact height: 57 mm above the table.
- Table +Z points downward, so negative table Z is above the surface.
- Tabletop collision box: 2.0 x 2.0 x 0.05 m.
- Tabletop upper surface in joint1: Z = -0.033379 m.
- Table transform and pad-contact offset require physical validation.

## Runtime launches

Nano:
- ros2 launch mycobot_hardware physical_bringup.launch.py
- ros2 launch mycobot_vision cameras.launch.py

PC:
- ros2 launch mycobot_vision yolo_detection.launch.py
- ros2 launch mycobot_vision table_preview.launch.py launch_rviz:=false
- ros2 launch mycobot_280jn_physical_moveit_config physical_planning.launch.py

Activate cap_detection_env before launching the detector.

## Planning experiment

The approach-planning experiments were run as terminal Python blocks.
They are not yet installed as reusable package executables.

The planner used live joint feedback, collision-aware IK, state validity
checking and the GetMotionPlan service. DisplayTrajectory was published
once to avoid interrupting RViz playback.

An earlier plan returned success with 419 trajectory points.
The cap was subsequently moved, requiring a new plan.
A successful model-based plan does not establish physical grasp accuracy.

## Remaining work

- Save the live-target planning experiment as a reusable script.
- Confirm the latest trajectory endpoint against the cap marker.
- Validate the TCP/contact offset and grasp clearance physically.
- Include relevant physical obstacles in the collision scene.
- Plan and validate descent, grasp, lift and placement.
- Integrate controlled physical trajectory execution.

Physical motion remains disabled. No physical pick-and-place was
performed during this session.
