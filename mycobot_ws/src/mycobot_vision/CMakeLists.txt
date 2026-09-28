cmake_minimum_required(VERSION 3.8)
project(mycobot_vision)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/table_homography_calibrator.py
    scripts/yolo_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()