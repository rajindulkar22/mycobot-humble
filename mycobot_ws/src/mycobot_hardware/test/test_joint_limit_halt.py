"""Coordinate moves stop before a joint limit. No serial port is opened."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_joint_feedback_cache import bare_node, load_driver


class _CoordRequest:
    def __init__(self):
        self.x = 200.0
        self.y = 0.0
        self.z = 100.0
        self.rx = -170.0
        self.ry = 0.0
        self.rz = -125.0
        self.speed = 10
        self.execute = True


class _CoordResponse:
    def __init__(self):
        self.success = None
        self.message = ""
        self.start_coords = None
        self.final_coords = None


class _Robot:
    """Fake 280 controller with a scripted joint solution and live angles."""

    def __init__(self, solution, poll_angles):
        self.solution = list(solution)
        self.poll_angles = list(poll_angles)
        self.calls = []
        self._polled = False

    def is_controller_connected(self):
        return 1

    def is_power_on(self):
        return 1

    def get_error_information(self):
        return 0

    def is_moving(self):
        return 1 if self._polled else 0

    def get_coords(self):
        return [180.0, 0.0, 80.0, -170.0, 0.0, -125.0]

    def get_angles(self):
        self.calls.append("get_angles")
        if not self._polled and self.calls.count("get_angles") == 1:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._polled = True
        return list(self.poll_angles)

    def solve_inv_kinematics(self, _coords, _angles):
        self.calls.append("solve")
        return list(self.solution)

    def set_fresh_mode(self, _mode):
        self.calls.append("fresh")

    def get_fresh_mode(self):
        return 0

    def send_coords(self, _coords, _speed, _mode, _async=True):
        self.calls.append("send_coords")

    def stop(self):
        self.calls.append("stop")


def _node():
    node = bare_node(load_driver())
    node.coord_motion_enabled = True
    node.coord_min_x = 50.0
    node.coord_max_x = 280.0
    node.coord_min_y = -200.0
    node.coord_max_y = 200.0
    node.coord_min_z = 20.0
    node.coord_max_z = 250.0
    node.coord_tolerance_mm = 8.0
    node.coord_angle_tolerance = 8.0
    node.coord_tool_tolerance = 25.0
    node.motion_timeout = 1.0
    node.motion_poll_period = 0.0
    return node


def _move(node, robot):
    node.robot = robot
    response = _CoordResponse()
    node.handle_move_coords(_CoordRequest(), response)
    return response


def test_out_of_limit_solution_is_not_sent():
    node = _node()
    robot = _Robot(
        solution=[0.0, 139.0, 0.0, 0.0, 0.0, 0.0],
        poll_angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    response = _move(node, robot)
    assert response.success is False
    assert "J2" in response.message
    assert "send_coords" not in robot.calls
    assert "stop" not in robot.calls


def test_live_limit_stops_the_arm():
    node = _node()
    robot = _Robot(
        solution=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        poll_angles=[0.0, 139.0, 0.0, 0.0, 0.0, 0.0],
    )
    response = _move(node, robot)
    assert response.success is False
    assert "J2" in response.message
    assert "stop command sent" in response.message
    assert robot.calls.count("send_coords") == 1
    assert "stop" in robot.calls
