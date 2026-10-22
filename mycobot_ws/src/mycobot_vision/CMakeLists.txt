cmake_minimum_required(VERSION 3.8)
project(mycobot_vision)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
    models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/yolo_detector.py
    scripts/pick_place.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    pick_page.html
    calibrate_page.html
  DESTINATION share/${PROJECT_NAME}
)

ament_package()