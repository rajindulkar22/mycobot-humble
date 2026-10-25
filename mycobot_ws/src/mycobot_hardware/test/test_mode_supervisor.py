"""Checks for the launch supervisor without starting ROS or the arm."""

import subprocess

from mycobot_hardware.mode_supervisor import (
    ModeSupervisor,
    conflict_reason,
)


class FakeProcess:
    """A process whose exit is driven by the test's kill hook."""

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.stdout = None
        self.signals = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)


def _supervisor(processes, launches):
    def popen(command):
        launches.append(command["args"])
        return processes.pop(0)

    def killpg(pid, signum):
        for process in started:
            if process.pid == pid:
                process.returncode = 0
                process.signals.append(signum)

    started = []

    def tracking_popen(command):
        process = popen(command)
        started.append(process)
        return process

    return ModeSupervisor(popen=tracking_popen, killpg=killpg)


def test_second_serial_mode_is_refused():
    first = FakeProcess(11)
    second = FakeProcess(12)
    launches = []
    supervisor = _supervisor([first, second], launches)

    ok, _message = supervisor.start("feedback")
    assert ok is True
    ok, message = supervisor.start("pick", confirmed=True)

    assert ok is False
    assert "serial" in message
    assert launches == [[
        "ros2", "launch", "mycobot_hardware", "physical_bringup.launch.py",
    ]]
    assert supervisor.snapshot()["modes"][0]["status"] == "running"


def test_plot_can_run_beside_feedback_but_not_beside_demo():
    feedback = FakeProcess(21)
    plot = FakeProcess(22)
    launches = []
    supervisor = _supervisor([feedback, plot], launches)

    assert supervisor.start("feedback")[0] is True
    assert supervisor.start("plot")[0] is True
    assert len(launches) == 2

    supervisor.stop_all()
    demo = FakeProcess(23)
    plot_again = FakeProcess(24)
    launches.clear()
    supervisor = _supervisor([demo, plot_again], launches)
    assert supervisor.start("demo", confirmed=True)[0] is True
    ok, message = supervisor.start("plot")
    assert ok is False
    assert "8091" in message


def test_motion_mode_requires_confirmation():
    supervisor = ModeSupervisor(popen=lambda _command: FakeProcess(1))
    ok, message = supervisor.start("diagnostic", confirmed=False)
    assert ok is False
    assert "Confirm" in message


def test_stop_interrupts_the_process_group():
    process = FakeProcess(31)
    nxt = FakeProcess(32)
    launches = []
    supervisor = _supervisor([process, nxt], launches)
    supervisor.start("feedback")

    ok, message = supervisor.stop("feedback")

    assert ok is True
    assert message.startswith("Stopped")
    assert process.signals[0] == 2
    assert process.returncode == 0
    status = [
        mode for mode in supervisor.snapshot()["modes"] if mode["id"] == "feedback"
    ][0]
    assert status["status"] == "exited"
    assert supervisor.start("pick", confirmed=True)[0] is True


def test_unknown_mode_and_idle_conflict_helper():
    supervisor = ModeSupervisor()
    ok, message = supervisor.start("not-a-mode")
    assert ok is False
    assert "Unknown" in message
    assert conflict_reason(supervisor.mode("plot"), []) is None
