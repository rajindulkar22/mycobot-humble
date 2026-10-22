"""Local receive limits; caller MUST hold the driver's serial mutex."""
import time


def feedback_call(robot, method_name, clock=time.monotonic):
    if method_name not in ("get_angles", "is_moving", "get_error_information"):
        raise ValueError("Only read-only feedback methods are supported")
    serial = robot._serial_port
    original_timeout = serial.timeout
    original_read = robot._read
    had_override = "_read" in vars(robot)
    old_override = vars(robot).get("_read")
    attempts = []

    def receive(genre, *args, **kwargs):
        kwargs["timeout"] = min(float(kwargs.get("timeout") or 0.04), 0.04)
        started = clock()
        try:
            frame = original_read(genre, *args, **kwargs)
            attempts.append((round(clock() - started, 4),
                             len(frame) if frame is not None else 0,
                             frame[3] if frame and len(frame) > 3 else None))
            return frame
        except Exception:
            attempts.append((round(clock() - started, 4), -1, None))
            raise

    try:
        serial.timeout = 0.005
        robot._read = receive
        result = getattr(robot, method_name)()
        if type(result) is int and result == -1:
            raise RuntimeError("SDK returned -1")
        return result, attempts
    except Exception as error:
        raise RuntimeError(
            f"{method_name} failed: {error}; receive attempts "
            f"(seconds, bytes, reply opcode)={attempts!r}"
        ) from error
    finally:
        if had_override:
            robot._read = old_override
        else:
            del robot._read
        serial.timeout = original_timeout
