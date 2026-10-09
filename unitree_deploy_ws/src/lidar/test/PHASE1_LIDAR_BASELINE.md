# G1 雷达部署第一阶段：单帧高程基线

更新日期：2026-09-28。

本阶段只验证雷达、FAST-LIO2、坐标变换和单帧高程观测，不加载
`policy.pt`，不启动 LP/VP，不发布任何 `LowCmd`。

## 1. 本阶段固定数据链路

```text
/livox/lidar (CustomMsg) + /livox/imu
  -> FAST-LIO2
  -> /cloud_registered (PointCloud2, camera_init)
     + /Odometry (camera_init -> body)
  -> rolling_elevation_map（只用当前一帧）
  -> 11x17 policy_heights + policy_valid + policy_cloud
  -> baseline_elevation_analyzer（只订阅、统计和落盘）
```

策略高程契约：

- 形状为 11 行 x 17 列，共 187 项；
- 行为 `y=-0.5..+0.5 m`，列为 `x=-0.8..+0.8 m`；
- 展平顺序为 y 升序、每行内 x 升序；
- 有效值范围为 `[-1, 1]`；
- 无效格必须同时满足 `valid=0` 和 `height=-1.1`；
- 当前实现 `height = clip(-z_capture - 0.5, -1, 1)`；
- 平地值不是天然等于 0。若躯干原点离地约 0.84 m，平地应约为
  `0.84-0.5=0.34`。旧交接文档中“平地应接近 0”的说法不适用。

## 2. 先在本地构建

在本地电脑执行：

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select lidar --symlink-install
source install/setup.bash

python3 -m pytest -q \
  src/lidar/test/test_baseline_elevation_analyzer.py
ros2 pkg executables lidar | sort
```

必须看到：

```text
lidar baseline_elevation_analyzer
lidar rolling_elevation_map
```

## 3. 第一次复制到机器人

建议使用独立的 `/home/unitree/cjh/phase1_ws`，不改机器人已有的控制包，
也不依赖旧交接文档中尚未确认的 `unitree_deploy` 源码位置。

本地电脑终端：

```bash
tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  -C /home/cjh/TienKung-Lab/unitree_deploy_ws/src \
  -czf /tmp/g1_phase1_lidar_src.tar.gz lidar

scp /tmp/g1_phase1_lidar_src.tar.gz \
  unitree@192.168.123.164:/tmp/

ssh unitree@192.168.123.164 \
  'mkdir -p /home/unitree/cjh/phase1_ws/src && \
   tar -xzf /tmp/g1_phase1_lidar_src.tar.gz \
   -C /home/unitree/cjh/phase1_ws/src'
```

如果当前使用无线网络，把地址改为 `192.168.3.157`。

机器人终端：

```bash
cd /home/unitree/cjh/phase1_ws
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

colcon build --packages-select lidar --symlink-install
source /home/unitree/cjh/phase1_ws/install/setup.bash

ros2 pkg executables lidar | sort
ls -l \
  /home/unitree/cjh/phase1_ws/install/lidar/share/lidar/config/phase1_baseline.yaml
```

以后代码更新后，重新打包、复制、解包并执行 `colcon build`。开始实验前，
每个机器人终端都使用下面同一组环境：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
```

## 4. 先做已有 bag 回放验证

回放时不要启动实时 Livox 驱动和 FAST-LIO2。先确认没有遗留发布者：

```bash
ps -ef | grep -E '[l]ivox_ros_driver2_node|[f]astlio_mapping'
ros2 topic info /cloud_registered --verbose
ros2 topic info /Odometry --verbose
```

如有实时节点，回到其原终端用 `Ctrl-C` 正常停止。

### 终端 A：世界方向静态 TF

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 run tf2_ros static_transform_publisher \
  0 0 0 \
  0.9996734 0 -0.0255576 0 \
  g1_world camera_init
```

这是四元数参数形式：`x y z qx qy qz qw parent child`。若 ROS 明确报告
bag 中已经有同名 `g1_world -> camera_init` 权威发布者，则停止本终端，不能
保留两个同名 TF 发布者。

### 终端 B：当前单帧高程节点

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 run lidar rolling_elevation_map --ros-args \
  --params-file \
  /home/unitree/cjh/phase1_ws/install/lidar/share/lidar/config/phase1_baseline.yaml
```

### 终端 C：基线分析节点

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

mkdir -p /home/unitree/cjh/results/g1_lidar_phase1

ros2 run lidar baseline_elevation_analyzer --ros-args \
  --params-file \
  /home/unitree/cjh/phase1_ws/install/lidar/share/lidar/config/phase1_baseline.yaml \
  -p output_root:=/home/unitree/cjh/results/g1_lidar_phase1 \
  -p run_name:=bag_fastlio_validation_03
```

参数文件配置了 3 秒输入空闲超时，因此 bag 播放结束后会自动写报告并退出。

### 终端 D：播放 bag

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 bag info /home/unitree/cjh/bags/fastlio_validation_03
ros2 bag play /home/unitree/cjh/bags/fastlio_validation_03
```

### 终端 E：在线检查

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

timeout 10s ros2 topic hz /cloud_registered
timeout 10s ros2 topic hz /Odometry
timeout 10s ros2 topic hz /g1/elevation/policy_heights
timeout 10s ros2 topic hz /g1/elevation/policy_cloud
```

另开终端分别检查：

```bash
ros2 topic echo /g1/elevation/pose_sync_error
ros2 topic echo /g1/elevation/valid_ratio
ros2 topic echo /g1/elevation/policy_heights --once
ros2 topic echo /g1/elevation/policy_valid --once
```

## 5. RViz 检查

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash
rviz2
```

先设 `Fixed Frame=capture_torso_yaw`，添加：

- PointCloud2 `/g1/elevation/policy_cloud`；
- PointCloud2 `/g1/elevation/internal_cloud`；
- PointCloud2 `/cloud_registered`；
- TF。

必须人工确认：

1. `capture_torso_yaw` 中 X 向前、Y 向左、Z 向上；
2. 机器人向前时，固定地形在该机身坐标系中向负 X 运动；
3. 机器人左转时，固定地形在该机身坐标系中向右旋转；
4. 当前扫描不能出现历史拖影；
5. 躯干附近高点是否确实来自机器人身体。分析器只把这些点标为
   `self_hit_candidate`，不会擅自删除它们。

## 6. 再做实时真机采集

确认 bag 验证通过后才运行本节。仍然不启动策略控制。

先检查时间：

```bash
chronyc tracking
chronyc sources -v
date '+ROBOT %F %T.%N epoch=%s.%N'
```

### 终端 1：Livox CustomMsg 驱动

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p user_config_path:=/home/unitree/Livox-SDK2-ros2/src/livox_ros_driver2/config/MID360_config.json \
  -p xfer_format:=1 \
  -p multi_topic:=0 \
  -p frame_id:=livox_frame
```

### 终端 2：FAST-LIO2

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 launch fast_lio mapping.launch.py \
  config_path:=/home/unitree/fastlio2_ws/src/fast_lio/config \
  config_file:=g1_mid360.yaml \
  rviz:=false
```

### 终端 3：静态 TF

执行第 4 节“终端 A”的命令。

### 终端 4：单帧高程节点

执行第 4 节“终端 B”的命令。

### 终端 5：分析节点

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

mkdir -p /home/unitree/cjh/results/g1_lidar_phase1

ros2 run lidar baseline_elevation_analyzer --ros-args \
  --params-file \
  /home/unitree/cjh/phase1_ws/install/lidar/share/lidar/config/phase1_baseline.yaml \
  -p output_root:=/home/unitree/cjh/results/g1_lidar_phase1 \
  -p run_name:=live_phase1 \
  -p idle_timeout_sec:=0.0
```

实时实验结束时在该终端按 `Ctrl-C`，节点会先保存报告再退出。

### 终端 6：同步录包

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/phase1_ws/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 bag record \
  -o /home/unitree/cjh/bags/g1_lidar_phase1_live \
  /livox/lidar \
  /livox/imu \
  /cloud_registered \
  /Odometry \
  /tf \
  /tf_static \
  /g1/elevation/policy_heights \
  /g1/elevation/policy_valid \
  /g1/elevation/valid_ratio \
  /g1/elevation/pose_sync_error \
  /g1/elevation/internal_cloud \
  /g1/elevation/policy_cloud
```

若同名 bag 已存在，换一个输出名，例如加日期时间，不能覆盖原始数据。

建议一次采集至少包括：平地静止 30 秒、手动缓慢前后移动和转向、在楼梯前
不同距离静止观察。第一阶段不要求机器人自主上楼梯。

## 7. 报告位置和判读

```bash
find /home/unitree/cjh/results/g1_lidar_phase1 \
  -maxdepth 2 -name summary.json -print
```

每次运行目录包含：

- `summary.json`：帧率、同步、契约错误、整体覆盖率、疑似自体点统计；
- `per_cell.csv`：187 格各自的覆盖率、最长连续缺失和高度统计；
- `valid_ratio_heatmap.txt`：11x17 覆盖率热图；
- `frames.csv`：逐帧有效率和高度范围；
- `clouds.csv`：逐帧点数和疑似自体点数量；
- `baseline_arrays.npz`：后续离线分析使用的原始数组。

查看报告：

```bash
python3 -m json.tool \
  /home/unitree/cjh/results/g1_lidar_phase1/bag_fastlio_validation_03/summary.json

less \
  /home/unitree/cjh/results/g1_lidar_phase1/bag_fastlio_validation_03/valid_ratio_heatmap.txt
```

若重复使用相同 `run_name`，节点会自动生成 `_01`、`_02` 等目录，应查看日志
中打印的实际路径。

第一阶段的通过条件：

1. `paired_frames` 与 bag 中预期帧数同量级，频率约 10 Hz；
2. `cloud_odom_sync_error_ms.max <= 30 ms`，且最好稳定在数毫秒；
3. `height_layout_errors`、`valid_layout_errors`、长度错误、云帧错误、时间戳
   回退和所有 `contract_errors` 均为 0；
4. RViz 中坐标方向、运动方向与地形高度正确；
5. 热图明确显示雷达近身盲区的位置和大小；低覆盖率本身不是失败；
6. `self_hit_candidate` 必须与 RViz 中机器人自体回波交叉确认后，下一阶段才能
   设计过滤盒，不能仅凭计数直接删除点。

完成以上项目后，才能进入第二阶段的时序补全/滚动融合设计。此时仍不接入
`policy.pt`。
