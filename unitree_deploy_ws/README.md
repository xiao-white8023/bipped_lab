# G1 MID-360 点云采集（ROS 2 Jazzy）

本阶段只订阅、检查和录制点云，不发送机器人控制指令。G1 实测原始话题是 `/utlidar/cloud_livox_mid360`（`sensor_msgs/msg/PointCloud2`，`livox_frame`）；雷达 IMU 话题是 `/utlidar/imu_livox_mid360`。本仓库 `unitree_ros2` README 的 `/utlidar/cloud` 示例对应 Go2。

## 1. 连接与环境

把电脑的有线网卡接到 G1 网络，按机器人的网络配置给该网卡设置同网段地址。当前电脑的目标网卡是 `enp4s0`；连接后先检查：

```bash
ip -br addr show enp4s0
```

首次在这台 Jazzy 主机使用 CycloneDDS 时安装：

```bash
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
```

每个运行 ROS 2 命令的终端都使用相同环境（若网卡名改变，改 `enp4s0`）：

```bash
source /opt/ros/jazzy/setup.bash
source /home/cjh/TienKung-Lab/unitree_deploy_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enp4s0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

不要直接 source `/home/cjh/TienKung-Lab/unitree_ros2/setup.sh`：当前文件写死了 Foxy、另一个工作空间路径和 `enp3s0`。

## 2. 确认 ROS 2 真正收到 G1 点云

```bash
ros2 daemon stop
ros2 topic list -t | rg utlidar
ros2 topic info -v /utlidar/cloud_livox_mid360
ros2 topic hz /utlidar/cloud_livox_mid360
ros2 run lidar lidar_point
```

`lidar_point` 首包打印 `frame_id`、时间戳、尺寸、字段偏移和 `point_step`，之后打印有效 XYZ 点数、接收频率与距离范围。可用 `--ros-args -p report_every:=20` 降低日志频率。先看到真实首包，再继续高程图工作。

若本地包有修改，重新构建：

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select lidar --symlink-install
source install/setup.bash
```

## 3. 保存原始数据供 Point-LIO 和高程图回放

确认两个话题都在发布后，另开一个同样配置的终端，运行：

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
mkdir -p bags
ros2 bag record -o bags/g1_lidar_001 /utlidar/cloud_livox_mid360 /utlidar/imu_livox_mid360
```

录制一段静止数据，按 Ctrl+C 正常结束。后续需要检查不同姿态和移动时的近场覆盖，但不要把静止看不到的区域当成已观测地面。不要仅保存 PCD：rosbag 会保留原始消息头、`ring`、逐点 `time` 字段和雷达 IMU，供建图、时间同步和排错使用。

可先在 RViz2 中添加 PointCloud2，话题选 `/utlidar/cloud_livox_mid360`，Fixed Frame 设为首包显示的 `livox_frame`，检查真实点云形状。后续地图坐标系和机器人外参需要单独建立。
