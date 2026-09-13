FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-setuptools \
    python3-serial \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-srvs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "pymycobot==4.0.4"

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'if [ -f /root/mycobot_ws/install/setup.bash ]; then source /root/mycobot_ws/install/setup.bash; fi' >> /root/.bashrc

WORKDIR /root/mycobot_ws

CMD ["bash"]
