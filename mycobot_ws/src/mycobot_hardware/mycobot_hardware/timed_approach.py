"""Approach-only local trajectory runner. Importing never opens a serial port.

This is software monitoring, not a hardware safety controller. Serial reads can
block; watchdog response cannot precede completion of an in-flight SDK call.
"""

import bisect
import json
import math
import time


def load_plan(path, names, limits, margin, now, max_age=120.0, *, allow_descent=False):
    with open(path, encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema_version") != 2:
        raise ValueError("Regenerate the plan with waypoint timing (schema 2)")
    if plan.get("joint_names") != list(names):
        raise ValueError("Unexpected joint names or order")
    age = now - float(plan["created_unix"])
    if not math.isfinite(age) or not 0 <= age <= max_age:
        raise ValueError("Plan expired or clocks differ; synchronize and replan")
    # Descent is a distinct plan kind, never disguised as an above-table approach.
    if plan.get("purpose") == "provisional_vertical_cap_descent_only":
        if allow_descent is not True:
            raise ValueError("Vertical descent is disabled; timed_descent_enabled must be true")
        initial_height = float(plan["initial_contact_height_m"])
        final_height = float(plan["final_contact_height_m"])
        descent = float(plan["descent_m"])
        if (not all(math.isfinite(v) for v in (initial_height, final_height, descent))
                or not .035 <= initial_height <= .075
                or not .007 <= final_height <= .020
                or not .005 <= descent <= .068
                or abs(initial_height-final_height-descent) > 1e-6):
            raise ValueError("Invalid bounded descent heights/travel")
        if (plan.get("planning_frame") != "joint1"
                or plan.get("table_frame") != "table_frame"
                or plan.get("collision_checked") is not True
                or plan.get("cartesian_fraction") != 1.0):
            raise ValueError("Descent lacks complete collision-checked Cartesian planning metadata")
        initial_contact = plan["start_contact_table_m"]
        final_contact = plan["goal_contact_table_m"]
        if (not isinstance(initial_contact, list) or not isinstance(final_contact, list)
                or len(initial_contact) != 3 or len(final_contact) != 3):
            raise ValueError("Invalid Cartesian contact metadata")
        initial_contact = [float(v) for v in initial_contact]
        final_contact = [float(v) for v in final_contact]
        if (not all(math.isfinite(v) for v in initial_contact+final_contact)
                or max(abs(initial_contact[i]-final_contact[i]) for i in (0,1)) > 1e-6
                or abs(initial_contact[2]+initial_height) > 1e-6
                or abs(final_contact[2]+final_height) > 1e-6):
            raise ValueError("Descent metadata must describe vertical motion only")
        if float(plan["time_from_start_seconds"][-1]) > 60.0:
            raise ValueError("Descent exceeds 60 seconds before scaling")
    else:
        height = float(plan["approach_contact_height_m"])
        if not math.isfinite(height) or height < 0.057:
            raise ValueError("Only the above-table approach is accepted")
    times = plan["time_from_start_seconds"]
    points = plan["points_degrees"]
    if not isinstance(times, list) or not isinstance(points, list):
        raise ValueError("Invalid trajectory arrays")
    if not 2 <= len(times) == len(points) <= 20000:
        raise ValueError("Invalid trajectory length")
    times = [float(t) for t in times]
    if (not all(math.isfinite(t) for t in times)
            or abs(times[0]) > 1e-9 or not 0 < times[-1] <= 120
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Timing must start at zero and strictly increase")
    start = vector(plan["start_degrees"])
    points = [vector(point) for point in points]
    if distance(start, points[0]) > 0.05:
        raise ValueError("Trajectory first point differs from declared start")
    for point in [start] + points:
        for index, value in enumerate(point, 1):
            lower, upper = limits[index]
            if not lower + margin <= value <= upper - margin:
                raise ValueError(f"J{index} outside configured limits")
    return plan, start, times, points


def vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError(f"Invalid six-joint sample: {values!r}")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"Non-finite six-joint sample: {values!r}")
    return result


def distance(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def sample(times, points, elapsed):
    if elapsed <= times[0]:
        return list(points[0])
    if elapsed >= times[-1]:
        return list(points[-1])
    index = bisect.bisect_right(times, elapsed) - 1
    fraction = (elapsed - times[index]) / (times[index + 1] - times[index])
    return [a + fraction * (b - a)
            for a, b in zip(points[index], points[index + 1])]


class TimedApproach:
    """Run a bounded local path through the existing serial owner and stop flag."""

    def __init__(self, node, clock=time.monotonic, sleep=time.sleep,
                 wall_clock=time.time):
        self.node = node
        self.clock = clock
        self.sleep = sleep
        self.wall_clock = wall_clock

    def stopped(self):
        if self.node.stop_requested.is_set():
            raise RuntimeError("Stop is latched; no further command sent")

    def parameters(self):
        node = self.node
        scale = float(node.get_parameter("timed_approach_time_scale").value)
        if not math.isfinite(scale) or not 1 <= scale <= 10:
            raise ValueError("Time scale must be between 1 and 10")
        if not (0 < node.max_speed <= 5 and 0 < node.max_joint_step <= 2
                and 0 < node.position_tolerance <= 0.5
                and 0 < node.motion_timeout <= 10):
            raise ValueError("Require speed <=5, step <=2, tolerance <=0.5, timeout <=10")
        if node.replace_motion_enabled or node.gripper_motion_enabled or node.motion_enabled:
            raise ValueError("Legacy streaming, gripper and single-joint gates must be off")
        return scale

    def read(self):
        from mycobot_hardware.bounded_feedback import feedback_call
        node = self.node
        self.stopped()
        before = self.clock()
        with node.serial_mutex:
            locked = self.clock()
            self.stopped()
            raw, angle_attempts = feedback_call(node.robot, "get_angles", self.clock)
            stamp = node.get_clock().now().to_msg()
            angles = vector(raw)
            angles_at = self.clock()
            self.stopped()
            moving, moving_attempts = feedback_call(node.robot, "is_moving", self.clock)
            moving_at = self.clock()
            self.stopped()
            error, error_attempts = feedback_call(node.robot, "get_error_information", self.clock)
            error_at = self.clock()
        self.stopped()
        self.last_feedback_transport = {
            "angles": angle_attempts, "moving": moving_attempts,
            "error": error_attempts,
        }
        elapsed = self.clock() - before
        if elapsed > 0.5:
            raise RuntimeError(
                f"Feedback transaction exceeded 0.5 s: total={elapsed:.3f}s, "
                f"lock={locked-before:.3f}s, angles={angles_at-locked:.3f}s, "
                f"moving={moving_at-angles_at:.3f}s, "
                f"error_read={error_at-moving_at:.3f}s; "
                f"receive attempts={self.last_feedback_transport!r}"
            )
        if error != 0 or moving not in (0, 1):
            raise RuntimeError(f"Controller feedback: error={error!r}, moving={moving!r}")
        node._measured_degrees = list(angles)
        node._measured_stamp = stamp
        node._publish_joint_degrees(angles, stamp)
        return angles, moving

    def validate(self):
        node = self.node
        scale = self.parameters()
        plan, start, times, points = load_plan(
            node.get_parameter("timed_approach_plan_file").value,
            node.JOINT_NAMES, node.JOINT_LIMITS_DEGREES,
            node.joint_limit_margin, self.wall_clock(),
            allow_descent=node.get_parameter("timed_descent_enabled").value,
        )
        # Slow down the complete time axis, preserving relative waypoint times.
        peak = max(distance(a, b) / (tb - ta)
                   for a, b, ta, tb in zip(points, points[1:], times, times[1:]))
        scale = max(scale, peak / 2.0)  # <=2 degrees/second per joint.
        times = [t * scale for t in times]
        if times[-1] > 180:
            raise ValueError("Scaled trajectory exceeds 180 seconds")
        live, moving = self.read()
        if moving != 0 or distance(live, start) > 0.5:
            raise ValueError("Robot moving or differs from plan start by more than 0.5 degrees")
        grip = node.get_cached_gripper_value()
        expected = float(plan["gripper_radians"])
        if (grip is None or not math.isfinite(expected)
                or abs(node.gripper_value_to_radians(grip) - expected) > 0.02):
            raise ValueError("Gripper state unavailable or differs from the planned geometry")
        return times, points, live

    def handle_validate(self, request, response):
        del request
        node = self.node
        node._guarded_motion_active.set()
        try:
            times, points, _ = self.validate()
            response.success = True
            response.message = f"Validated {len(points)} waypoints, {times[-1]:.2f} s; no motion sent"
        except Exception as error:
            response.success = False
            response.message = str(error)
        finally:
            node._guarded_motion_active.clear()
        return response

    def handle_execute(self, request, response):
        del request
        node = self.node
        response.success = False
        if not (node.get_parameter("timed_approach_enabled").value
                and node.multi_joint_motion_enabled):
            response.message = "Timed approach is disabled"
            return response
        node._guarded_motion_active.set()
        entered_motion = False
        try:
            times, points, measured = self.validate()
            self.stopped()
            with node.serial_mutex:
                self.stopped()
                entered_motion = True
                node._set_fresh_mode_locked(True)
            self.stopped()
            began = self.clock()
            previous = began
            feedback_at = began
            last_target = list(measured)
            next_tick = began
            while True:
                self.stopped()
                now = self.clock()
                if now - previous > 0.5:
                    raise RuntimeError("Command scheduling gap exceeded 0.5 s")
                previous = now
                if now - feedback_at >= 0.2:
                    measured, _ = self.read()
                    feedback_at = self.clock()
                    # Compare with the last SENT target, not a future waypoint.
                    if distance(measured, last_target) > 2.0:
                        raise RuntimeError("Measured tracking error exceeded 2 degrees")
                now = self.clock()
                if now - feedback_at > 0.5 or now - previous > 0.5:
                    raise RuntimeError("Feedback or command timing became stale")
                target = sample(times, points, now - began)
                if (distance(target, measured) > node.max_joint_step
                        or distance(target, last_target) > node.max_joint_step):
                    raise RuntimeError("Next target exceeds measured/commanded step limit")
                with node.serial_mutex:
                    self.stopped()
                    node.robot.send_angles(target, node.max_speed, _async=True)
                self.stopped()
                node._publish_commanded_degrees(target)
                last_target = target
                if now - began >= times[-1]:
                    break
                next_tick += 0.04
                if self.clock() - next_tick > 0.1:
                    raise RuntimeError("Missed command deadline; trajectory aborted")
                next_tick = max(next_tick, self.clock() + 0.04)
                self.sleep(max(0.0, next_tick - self.clock()))

            deadline = self.clock() + node.motion_timeout
            stable = 0
            while self.clock() < deadline:
                measured, moving = self.read()
                error = distance(measured, points[-1])
                stable = stable + 1 if moving == 0 and error <= node.position_tolerance else 0
                if stable >= 3:
                    response.success = True
                    response.message = f"Approach completed; three measured endpoint checks, max error {error:.3f} degrees"
                    break
                self.stopped()
                self.sleep(0.2)
            else:
                raise RuntimeError("Approach endpoint did not settle within tolerance")
        except Exception as error:
            response.message = str(error)
        finally:
            if entered_motion:
                if not response.success:
                    node.stop_requested.set()
                try:
                    with node.serial_mutex:
                        if not response.success:
                            node.robot.stop()
                        node._set_fresh_mode_locked(False)
                except Exception as error:
                    response.success = False
                    node.stop_requested.set()
                    response.message += f"; cleanup failed: {error}"
            node._commanded_degrees = None
            node._guarded_motion_active.clear()
        return response
