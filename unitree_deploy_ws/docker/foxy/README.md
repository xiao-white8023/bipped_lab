# G1 local ROS 2 Foxy container

The host remains on Ubuntu 24.04 and ROS 2 Jazzy. This container supplies an
Ubuntu 20.04 / ROS 2 Foxy desktop environment for communication with the G1.
It uses host networking so DDS multicast and the wired interface `enp4s0` are
visible inside the container.

## Host setup

Install Docker once:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out of the desktop session and log in again, then verify:

```bash
docker version
```

## Build and enter the container

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
./docker/foxy/build.sh
./docker/foxy/run.sh
```

The run script selects:

- ROS 2 Foxy
- `rmw_cyclonedds_cpp`
- `ROS_DOMAIN_ID=0`
- CycloneDDS interface `enp4s0`
- host networking and X11 display access

## Verify the LiDAR and start RViz2

Inside the container:

```bash
echo "$ROS_DISTRO"
echo "$RMW_IMPLEMENTATION"
ros2 topic hz /livox/lidar
rviz2
```

In RViz2, use `livox_frame` as the fixed frame and add the PointCloud2 topic
`/livox/lidar` with Best Effort reliability.

## Build the local workspace for Foxy

Do not reuse the host's Jazzy `build`, `install`, or `log` directories. Inside
the container, use separate output directories:

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
colcon --log-base log_foxy build \
  --build-base build_foxy \
  --install-base install_foxy \
  --symlink-install
source install_foxy/setup.bash
```

The current `lidar_point` diagnostic uses a newer `sensor_msgs_py` convenience
function and may need a small Foxy compatibility change before it is run in the
container. RViz2 and `ros2 topic hz` do not depend on that diagnostic node.
