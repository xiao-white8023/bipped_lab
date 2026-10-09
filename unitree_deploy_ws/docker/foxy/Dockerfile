FROM osrf/ros:foxy-desktop

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-numpy \
        ros-foxy-rmw-cyclonedds-cpp \
        ros-foxy-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

RUN if getent group "${GROUP_ID}" >/dev/null; then \
        group_name="$(getent group "${GROUP_ID}" | cut -d: -f1)"; \
        useradd --create-home --uid "${USER_ID}" --gid "${group_name}" rosuser; \
    else \
        groupadd --gid "${GROUP_ID}" rosuser; \
        useradd --create-home --uid "${USER_ID}" --gid "${GROUP_ID}" rosuser; \
    fi \
    && echo 'source /opt/ros/foxy/setup.bash' >> /home/rosuser/.bashrc

USER rosuser
WORKDIR /home/cjh/TienKung-Lab/unitree_deploy_ws

CMD ["bash"]
