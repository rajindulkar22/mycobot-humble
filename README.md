# mycobot_humble

ROS 2 Humble hardware interface for the **myCobot 280** on **Jetson Nano**.

The `state_publisher` node reads joint angles and tool pose from the robot over serial (`/dev/ttyTHS1`) and publishes them to ROS topics. It is **read-only** — no motion commands are sent.

## Key paths

| Location | Path |
|----------|------|
| Host project | `~/mycobot_humble` |
| Workspace (bind-mounted) | `mycobot_ws/` |
| ROS package | `mycobot_ws/src/mycobot_hardware/` |
| Container name | `mycobot-humble-jetson` |

## Prerequisites

- Docker and Docker Compose on the Jetson
- myCobot 280 connected on `/dev/ttyTHS1`

## First-time setup

From the host:

```bash
cd ~/mycobot_humble
docker compose build
docker compose up -d
docker compose exec mycobot bash
```

Inside the container:

```bash
cd /root/mycobot_ws
colcon build --symlink-install --packages-select mycobot_hardware
source install/setup.bash
```

New interactive shells inside the container automatically source ROS Humble and the workspace (configured in the Docker image `.bashrc`).

## Open the container

From the host (use this every session, including in a new terminal):

```bash
cd ~/mycobot_humble
docker compose up -d          # if not already running
docker compose exec mycobot bash
```

Alternative using the container name:

```bash
docker exec -it mycobot-humble-jetson bash
```

You can open **multiple shells** into the same container — useful for running the node in one terminal and inspecting topics in another.

## Rebuild after code changes

Inside the container:

```bash
cd /root/mycobot_ws
colcon build --symlink-install --packages-select mycobot_hardware
source install/setup.bash
```

Optional syntax check:

```bash
python3 -m py_compile src/mycobot_hardware/mycobot_hardware/state_publisher.py
```

## Run the state publisher

**Option A — direct run:**

```bash
ros2 run mycobot_hardware state_publisher
```

**Option B — launch file** (loads `config/hardware.yaml`):

```bash
ros2 launch mycobot_hardware state_publisher.launch.py
```

**Override parameters** (example):

```bash
ros2 run mycobot_hardware state_publisher --ros-args -p publish_rate:=10.0
```

Default parameters (`config/hardware.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | `/dev/ttyTHS1` | Serial device |
| `baud` | `1000000` | Baud rate |
| `publish_rate` | `5.0` | Poll rate (Hz) |

## Inspect topics

Run in a **second container terminal** while the node is running:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /joint_states
ros2 topic echo /mycobot/tool_pose_raw
ros2 topic hz /joint_states
```

Published topics:

| Topic | Message type | Notes |
|-------|--------------|-------|
| `/joint_states` | `sensor_msgs/JointState` | Joint positions in radians |
| `/mycobot/tool_pose_raw` | `std_msgs/Float64MultiArray` | x, y, z, rx, ry, rz in mm/deg |

## Serial port locking

Only **one** `state_publisher` may use the serial port at a time. The node acquires a flock lock at `/tmp/mycobot_ttyTHS1.lock`.

- Starting a second instance fails with: `Serial port /dev/ttyTHS1 is already owned by another mycobot_hardware process`
- Stop the running node with **Ctrl+C** before starting another
- If a stale process is left behind:

```bash
pkill -f state_publisher
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ros2: command not found` | `source /opt/ros/humble/setup.bash && source install/setup.bash` |
| Container not running | `docker compose up -d` from `~/mycobot_humble` |
| Permission errors editing `src/` on host | `sudo chown -R $USER:$USER ~/mycobot_humble/mycobot_ws/src` |
| Lock / port already in use | Stop other `state_publisher` process (see above) |
| Rebuild image after Dockerfile changes | `docker compose build --no-cache && docker compose up -d` |

## Project layout

```
mycobot_humble/
├── compose.yaml
├── Dockerfile
├── README.md
└── mycobot_ws/
    └── src/mycobot_hardware/
        ├── config/hardware.yaml
        ├── launch/state_publisher.launch.py
        └── mycobot_hardware/state_publisher.py
```
