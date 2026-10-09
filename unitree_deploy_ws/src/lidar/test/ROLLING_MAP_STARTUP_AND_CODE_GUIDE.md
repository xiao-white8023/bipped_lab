# G1 MID360、FAST-LIO2 与滚动高程图：启动流程和代码学习指南

更新日期：2026-09-29。

## 1. 文档目的

本文档用于在新的对话中继续学习和审查下面两个文件：

```text
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/lidar/rolling_elevation_map.py
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/lidar/rolling_history.py
```

它记录当前真实运行链路、ROS 2 话题和坐标系、机器人端启动命令、两个文件的
阅读顺序、关键数学关系、已经验证的行为以及尚未实现的内容。

本文档主体描述 `phase2_rolling.yaml` 对应的 10 秒运动补偿点云历史方案。
Phase 3 持久世界高程格与协方差融合现已作为独立 A/B 节点实现，详见
`PHASE3_PERSISTENT_MAP.md`。阅读本文的 Phase 2 代码时，不要把 Phase 3 逻辑误认为
`rolling_elevation_map.py` 已有逻辑。

当前阶段只验证雷达地图，不加载 `policy.pt`，不发送任何 `LowCmd`。

## 2. 名称澄清

当前使用的是 **FAST-LIO2**，不是 FAST-LIVO。

- FAST-LIO2：LiDAR + IMU 里程计，当前实际运行模块；
- FAST-LIVO：LiDAR + IMU + Vision，是另一个包含视觉输入的系统；
- 本项目当前没有把相机图像送入里程计。

FAST-LIO2 不是雷达驱动，也不是高程图节点。它负责：

1. 融合 MID360 点云和 IMU；
2. 估计机器人六自由度位姿；
3. 对一帧扫描做运动畸变校正；
4. 把当前扫描注册到统一的 `camera_init` 世界坐标；
5. 发布 `/cloud_registered` 和 `/Odometry`。

FAST-LIO2 本身没有在当前链路中提供 LP 所需的 11x17 高程观测，也没有闭环
漂移消除。11x17 高程由 `rolling_elevation_map.py` 生成。

## 3. 当前完整运行链路

```text
MID360
  |
  v
livox_ros_driver2
  |-- /livox/lidar
  |     livox_ros_driver2/msg/CustomMsg
  |
  `-- /livox/imu
        sensor_msgs/msg/Imu
             |
             v
          FAST-LIO2
             |-- /cloud_registered
             |     sensor_msgs/msg/PointCloud2
             |     frame_id = camera_init
             |
             `-- /Odometry
                   nav_msgs/msg/Odometry
                   header.frame_id = camera_init
                   child_frame_id  = body
                         |
                         |  另有静态 TF：g1_world -> camera_init
                         v
              rolling_elevation_map
                         |
                         |-- /g1/elevation/policy_heights
                         |-- /g1/elevation/policy_valid
                         |-- /g1/elevation/policy_age
                         |-- /g1/elevation/valid_ratio
                         |-- /g1/elevation/pose_sync_error
                         |-- /g1/elevation/history_span
                         |-- /g1/elevation/internal_cloud
                         `-- /g1/elevation/policy_cloud
```

运行时必须启动的四部分是：

1. Livox MID360 驱动；
2. FAST-LIO2；
3. `g1_world -> camera_init` 静态 TF；
4. `lidar rolling_elevation_map`。

`baseline_elevation_analyzer`、RViz 和 rosbag 录制是可选诊断模块，不参与地图
计算。

## 4. 当前不属于活动链路的旧节点

以下节点不要与当前 FAST-LIO2 链路同时启动：

```text
lidar_point
cloud_preprocessor
static_elevation_map
```

它们的历史用途如下。

### 4.1 `lidar_point.py`

早期 PointCloud2 探针，只负责解析 XYZ、统计频率、点数和量程。它不使用 IMU，
不估计位姿，不融合历史点。

### 4.2 `cloud_preprocessor.py`

早期单帧预处理节点。它把 `/livox/lidar` 的 PointCloud2 通过固定安装外参转换到
`torso_link`，然后做距离过滤和 ROI 裁剪。固定外参只能描述雷达怎样安装在机器人
上，不能估计机器人从上一帧到当前帧的移动。

当前 Livox 驱动为 FAST-LIO2 使用 `xfer_format:=1`，因此 `/livox/lidar` 是
`livox_ros_driver2/msg/CustomMsg`，而旧节点订阅的是
`sensor_msgs/msg/PointCloud2`，消息类型不兼容。

`rolling_elevation_map.py` 仍从 `cloud_preprocessor.py` 导入两个纯工具函数：

```python
quaternion_matrix
xyz_view
```

这不代表需要运行 `g1_cloud_preprocessor` ROS 节点。

### 4.3 `static_elevation_map.py`

静止阶段对照节点，只能在 `torso_link` 不移动时把约 0.8 秒的单帧栅格取最大值。
机器人一旦前进、倒退或转向，旧点不会得到运动补偿，因此不能用于动态 LP。

## 5. 为什么滚动地图同时需要点云和 Odometry

`/cloud_registered` 已经在 `camera_init` 世界坐标中，但仅有世界点还不够。
策略需要的是地形相对于当前躯干的位置，因此还必须知道当前机器人位姿。

节点使用：

```text
/cloud_registered  -> 世界中的当前注册扫描
/Odometry          -> 当前 body 在 camera_init 中的位置和姿态
```

然后计算：

```text
T_g1world_torso
= T_g1world_camera_init
  x T_camera_init_body
  x T_body_torso
```

其中：

- `T_g1world_camera_init` 来自静态 TF；
- `T_camera_init_body` 来自 FAST-LIO2 `/Odometry`；
- `T_body_torso` 来自配置中的雷达/身体/躯干外参组合。

没有 `/Odometry`，节点不能把历史世界点转换到当前机身坐标，也就不能正确处理
前进、倒退和转向。

## 6. 坐标系定义

### 6.1 `livox_frame`

MID360 原始雷达坐标。雷达倒置安装在机器人头部。

### 6.2 `camera_init`

FAST-LIO2 初始化世界坐标。`/cloud_registered` 和 `/Odometry` 都以它为父坐标。

### 6.3 `g1_world`

本项目使用的 Z 向上世界坐标。通过下面的静态 TF 修正倒置安装和少量安装倾角：

```text
translation = [0, 0, 0]
quaternion xyzw = [0.9996734, 0, -0.0255576, 0]
parent = g1_world
child  = camera_init
```

### 6.4 `body`

FAST-LIO2 Odometry 的 child frame。

### 6.5 `capture_torso_yaw`

滚动地图每帧生成的当前策略坐标：

- 原点是当前 torso 位置；
- X 是 torso 前向在世界水平面上的投影；
- Y 向左；
- Z 向上；
- 只保留 yaw，去掉 roll 和 pitch，使高程图保持重力对齐。

策略 11x17 栅格最终在该坐标系表达。

## 7. 策略高程契约

当前配置文件：

```text
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/config/phase2_rolling.yaml
```

固定输出契约：

| 项目 | 值 |
|---|---:|
| 行数 | 11 |
| 列数 | 17 |
| 总数 | 187 |
| X | -0.8 到 +0.8 m |
| Y | -0.5 到 +0.5 m |
| 分辨率 | 0.1 m |
| 展平顺序 | y 递增，每行 x 递增，C order |
| 有效范围 | [-1, 1] |
| 无效高度 | -1.1 |

对有效格的编码为：

```text
policy_height = clip(-z_capture - 0.5, -1, 1)
```

如果 torso 原点离地约 0.88 m，则平地编码约为：

```text
-(-0.88) - 0.5 = 0.38
```

平地值不应被错误地假定为 0。

## 8. 机器人端工作区与包名

当前机器人端统一使用：

```text
工作区：/home/unitree/cjh
源码包：/home/unitree/cjh/src/lidar
安装包：/home/unitree/cjh/install/lidar
ROS包名：lidar
```

正确命令是：

```bash
ros2 run lidar rolling_elevation_map
```

不要使用旧命令：

```bash
ros2 run unitree_deploy rolling_elevation_map
```

如果日志节点名为 `g1_cloud_preprocessor`，或者显示：

```text
Filtering /livox/lidar -> /g1/lidar/filtered
```

说明启动了旧包中的错误入口，不是当前滚动地图。

## 9. 本地传输与机器人端构建

### 9.1 本地电脑打包

```bash
tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  -C /home/cjh/TienKung-Lab/unitree_deploy_ws/src \
  -czf /tmp/g1_lidar_phase2_src.tar.gz \
  lidar
```

### 9.2 复制到机器人

```bash
scp /tmp/g1_lidar_phase2_src.tar.gz \
  unitree@192.168.123.164:/tmp/
```

```bash
ssh unitree@192.168.123.164 \
  'mkdir -p /home/unitree/cjh/src && \
   tar -xzf /tmp/g1_lidar_phase2_src.tar.gz \
   -C /home/unitree/cjh/src'
```

无线网络地址可替换为 `192.168.3.157`。

### 9.3 机器人端构建

```bash
cd /home/unitree/cjh

source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

colcon build \
  --packages-select lidar \
  --symlink-install

source /home/unitree/cjh/install/setup.bash
```

检查：

```bash
ros2 pkg prefix lidar
ros2 pkg executables lidar | sort

ls -l \
  /home/unitree/cjh/install/lidar/share/lidar/config/phase2_rolling.yaml
```

`ros2 pkg prefix lidar` 应指向：

```text
/home/unitree/cjh/install/lidar
```

## 10. 实时启动命令

每个终端都是独立 shell，必须分别执行自己的 `source` 和 ROS 环境变量。

### 10.1 终端 1：Livox MID360 驱动

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

预期话题类型：

```bash
ros2 topic type /livox/lidar
ros2 topic type /livox/imu
```

```text
/livox/lidar = livox_ros_driver2/msg/CustomMsg
/livox/imu   = sensor_msgs/msg/Imu
```

### 10.2 终端 2：FAST-LIO2

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

等待 IMU 初始化完成。随后检查：

```bash
ros2 topic type /cloud_registered
ros2 topic type /Odometry
timeout 10s ros2 topic hz /cloud_registered
timeout 10s ros2 topic hz /Odometry
```

### 10.3 终端 3：世界方向静态 TF

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 run tf2_ros static_transform_publisher \
  0 0 0 \
  0.9996734 0 -0.0255576 0 \
  g1_world camera_init
```

不能同时存在两个相同的 `g1_world -> camera_init` 权威发布者。

### 10.4 终端 4：滚动高程地图

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source /home/unitree/cjh/install/setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 run lidar rolling_elevation_map --ros-args \
  --params-file \
  /home/unitree/cjh/install/lidar/share/lidar/config/phase2_rolling.yaml
```

正确日志节点名必须是：

```text
[g1_rolling_elevation_map]
```

日志应包含：

```text
cloud=/cloud_registered
odom=/Odometry
world=g1_world
capture=capture_torso_yaw
history=10.0s/200 frames
internal=33x33
policy=11x17
source=pgtt_full_atten
```

## 11. 启动后验证

### 11.1 节点和参数

```bash
ros2 node list

ros2 param get /g1_rolling_elevation_map cloud_topic
ros2 param get /g1_rolling_elevation_map odometry_topic
ros2 param get /g1_rolling_elevation_map history_seconds
ros2 param get /g1_rolling_elevation_map head_filter_enabled
```

预期：

```text
cloud_topic         = /cloud_registered
odometry_topic      = /Odometry
history_seconds     = 10.0
head_filter_enabled = true
```

### 11.2 输入 frame

```bash
ros2 topic echo /cloud_registered --field header --once
ros2 topic echo /Odometry --once
```

必须满足：

```text
/cloud_registered.header.frame_id = camera_init
/Odometry.header.frame_id         = camera_init
/Odometry.child_frame_id          = body
```

### 11.3 输出

```bash
timeout 10s ros2 topic hz /g1/elevation/policy_heights
timeout 10s ros2 topic hz /g1/elevation/policy_cloud

ros2 topic echo /g1/elevation/valid_ratio
ros2 topic echo /g1/elevation/pose_sync_error
ros2 topic echo /g1/elevation/history_span
ros2 topic echo /g1/elevation/policy_heights --once
ros2 topic echo /g1/elevation/policy_valid --once
ros2 topic echo /g1/elevation/policy_age --once
```

### 11.4 RViz

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cjh/install/setup.bash
rviz2
```

建议：

```text
Fixed Frame: capture_torso_yaw
PointCloud2: /g1/elevation/internal_cloud
PointCloud2: /g1/elevation/policy_cloud
Reliability: Best Effort
Style: Flat Squares
Size: 0.095 m
Decay Time: 0
```

查看 `/cloud_registered` 时使用 `g1_world` 作为 RViz Fixed Frame。若使用
`camera_init` 作为 Fixed Frame，原始注册点云仍可能看起来倒置；静态 TF 不会改写
点云消息，只会在坐标变换显示时生效。

## 12. `rolling_history.py` 阅读指南

该文件是纯 NumPy 辅助逻辑，不依赖 ROS。

### 12.1 `highest_point_per_cell(...)`

输入：

```text
N x 3 的当前帧局部点
地图尺寸
分辨率
```

处理：

1. 根据 XY 计算 0.1 m 栅格索引；
2. 删除地图外或非有限点；
3. 按格子和 Z 排序；
4. 每个格只保留一个真实的最高点。

输出仍然是实际 3D 点，不是栅格中心的人工点。这样一帧通常从约 1000 个 ROI 点
压缩为约 400 个代表点。

### 12.2 `TimedPointHistory`

内部结构：

```python
deque[(source_stamp, points_world)]
```

重要不变量：

- 保存的点必须是世界坐标；
- 使用消息源时间，不使用回调到达时间；
- 同时受 `history_seconds` 和 `max_frames` 限制；
- rosbag 时间倒退时清空全部历史；
- `points_and_ages()` 返回拼接后的世界点以及每个点的源时间年龄。

当前配置：

```text
history_seconds  = 10.0
history_max_frames = 200
```

它是“限时世界点包”，不是持久世界高程格。

## 13. `rolling_elevation_map.py` 阅读顺序

建议按下面顺序阅读，不要直接从 `_process_pair()` 跳着猜。

### 13.1 通用数学函数

```text
cell_count
multi_array_layout
stamp_seconds
pose_matrix
parameter_matrix
yaw_from_rotation
```

重点确认本文件矩阵约定：

```text
T_parent_child
```

表示把 child 坐标中的点转换到 parent 坐标。

代码使用行向量点，因此旋转常见形式为：

```python
points_parent = points_child @ rotation.T + translation
```

### 13.2 `__init__()`

阅读内容：

1. 参数声明；
2. `body_to_torso` 外参；
3. 33x33 内部地图和 11x17 输出裁剪；
4. 点云和 Odometry 队列；
5. `TimedPointHistory`；
6. 发布者和订阅者。

### 13.3 `_drain_sync_queue()`

它不是 ROS message_filters，而是节点自己的最近时间戳匹配：

1. 取最早待处理点云；
2. 在 Odometry 队列中找时间差最小的一条；
3. 若误差不超过 30 ms，调用 `_process_pair()`；
4. 若已经不可能再等到匹配 Odometry，丢弃该点云；
5. 发布 `/g1/elevation/pose_sync_error`。

### 13.4 `_world_from_lio()` 和 `_validate_frames()`

检查：

```text
cloud frame = camera_init
odom parent = camera_init
odom child  = body
```

并查询：

```text
g1_world <- camera_init
```

### 13.5 `_yaw_frame()` 和 `_points_in_capture()`

`_yaw_frame()` 从完整 torso 旋转中提取水平朝向，只建立 yaw 旋转矩阵。

世界点转换到当前策略坐标：

```python
points_local = (points_world - torso_position) @ capture_rotation
```

其中 `capture_rotation` 表示 `world <- capture_torso_yaw`。

### 13.6 当前帧过滤

```text
_capture_mask
_capture_packet
_remove_head_returns
highest_point_per_cell
```

当前内部区域：

```text
X: [-1.6, +1.6] m
Y: [-1.6, +1.6] m
Z: [-1.5, +0.5] m
```

头部过滤框：

```text
X: [-0.05, +0.15] m
Y: [-0.25, +0.25] m
Z: [+0.15, +0.50] m
```

头部回波在写入历史之前删除。

### 13.7 `_process_pair()` 主流程

逐行对应关系：

```text
读取 /cloud_registered XYZ
    |
    v
读取 /Odometry，计算 T_g1world_torso
    |
    v
提取当前 torso yaw
    |
    v
当前扫描 camera_init -> g1_world -> capture_torso_yaw
    |
    v
裁剪、删除头部点、每格保留当前帧最高点
    |
    v
代表点 capture_torso_yaw -> g1_world
    |
    v
写入 TimedPointHistory
    |
    v
全部历史世界点 -> 当前 capture_torso_yaw
    |
    v
当前 3.2 x 3.2 m 裁剪
    |
    v
_rasterize 每格取最高点
    |
    v
裁剪中心 11 x 17
    |
    v
编码 policy_height 并发布
```

## 14. 前进、倒退和转向为什么能够处理

历史点不是保存在旧机身坐标，而是保存在 `g1_world`。

每一帧都使用新的：

```text
torso_position
torso_yaw
```

重新计算：

```text
history_local =
(history_world - current_torso_position)
x current_capture_rotation
```

因此理论行为是：

- 机器人前进：固定地形在当前机身坐标中向负 X 移动；
- 机器人倒退：固定地形向正 X 移动；
- 机器人左转：固定地形在机身坐标中向右旋转；
- 机器人右转：固定地形向左旋转。

正确性依赖 FAST-LIO2 位姿、点云时间同步和外参。如果 yaw 估计有误，转向时会
出现弧形重影；如果位置估计漂移，旧点会错位。

## 15. `_rasterize()` 的当前语义

每个内部格当前计算：

```text
height       = 历史点中的最大 Z
freshest_age = 历史点中的最小年龄
hit_count    = 落入该格的点数
confidence   = clip(hit_count / confidence_observations, 0, 1)
```

这里有一个已经确认的重要语义问题：

```text
决定 height 的点
```

和：

```text
决定 freshest_age 的点
```

不一定是同一个点。

例如 9 秒前的高点仍然决定最大高度，但当前刚落入一个新地面点时，
`policy_age` 会接近 0。当前 `/policy_age` 因此表示“格中最新观测年龄”，不是
“最终高度来源年龄”。

后续至少应区分：

```text
freshest_observation_age
height_winner_age
```

## 16. 已经用实机 bag 验证的事实

数据目录：

```text
/home/cjh/TienKung-Lab/unitree_deploy_ws/bags/phase2_new_rolling
```

重要结果：

- 约 186 秒；
- `/cloud_registered` 和 `/Odometry` 各 1712 帧；
- 高程输出 1711 帧，约 9.2 Hz；
- 187 维契约错误为 0；
- 头部过滤框残留为 0；
- 输入点云和 Odometry 源时间戳误差为 0；
- 逐帧离线重建与实机有效掩码完全一致；
- 初始静止约 30 秒时只有 7/187 格有效；
- 前进后覆盖率可达到约 81%；
- 长距离运动期间可以达到 187/187；
- 停止后超过 10 秒，历史点按设计过期，地图可能重新变空；
- 当前帧单帧平均有效率约 2.4%；
- 10 秒滚动地图平均有效率约 25.8%；
- 真正决定最大高度的点中约 47.5% 超过 5 秒；
- 静止短时 FAST-LIO2 未观察到明显发散，但没有外部真值，不能证明无全局漂移。

这些结果说明：

1. 运动补偿世界点历史按当前设计工作；
2. 冷启动静止时不能靠 MID360 恢复从未看到的脚下地形；
3. 当前方案必须有 VP 冷启动或安全运动积累过程；
4. 如果 LP 停止超过 10 秒仍要继续使用旧地形，需要改变固定超时删除机制。

## 17. 当前已实现与尚未实现的边界

### 17.1 已实现

- FAST-LIO2 点云和 Odometry 同步；
- 当前躯干 yaw 坐标；
- 前进、倒退和转向的世界坐标重投影；
- 最近 10 秒历史；
- 每帧代表点压缩；
- 头部固定回波过滤；
- 33x33 内部地图；
- 11x17 策略输出；
- 有效掩码、覆盖率、年龄和调试点云。

### 17.2 已在独立 Phase 3 节点实现

- 机器人中心、世界固定且空间有界的持久高程格；
- 每格高度协方差融合和随时间增长的不确定度；
- Mahalanobis 冲突检测与连续观测替换；
- 源时间倒退和 FAST-LIO 位姿跳变清图；
- `height_age`、`freshest_age` 与每格方差诊断。

实现位于 `persistent_grid.py`、`persistent_elevation_map.py` 和
`phase3_persistent.yaml`，不在本文重点讲解的 Phase 2 两个文件中。

### 17.3 尚未实现

- FAST-LIO2 回环修正；
- 带 Header 的策略高程消息；
- LP 就绪/退出状态机；
- VP 到 LP 的自动切换；
- `policy.pt` 推理和真实控制。

“持久世界高程格”和“协方差融合”不能在阅读 Phase 2 代码时被当成已有逻辑；
它们属于可单独启动的 Phase 3 节点。

## 18. Phase 3 持久地图的概念边界

当前方案保存：

```text
deque[(timestamp, world_points)]
```

Phase 3 持久世界高程格保存类似：

```text
world_cell[i, j]:
    height_world
    variance
    last_observed_time
    height_source_time
    observation_count
    valid
```

二者区别：

- 持久世界格解决“已经看到的地形保存多久”；
- 协方差融合解决“多次测量怎样合并、哪个更可信”；
- 协方差融合本身不能解决固定 10 秒删除；
- 持久世界格不能解决冷启动从未观测过的区域；
- 完整方案通常将二者组合。

## 19. 常见故障判定

### 19.1 `Package 'lidar' not found`

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cjh/install/setup.bash
ros2 pkg prefix lidar
```

### 19.2 启动后出现 `g1_cloud_preprocessor`

启动了旧 `unitree_deploy` 包。停止后改用：

```bash
ros2 run lidar rolling_elevation_map --ros-args \
  --params-file \
  /home/unitree/cjh/install/lidar/share/lidar/config/phase2_rolling.yaml
```

### 19.3 节点等待 TF

确认：

```bash
ros2 run tf2_ros tf2_echo g1_world camera_init
```

并确保只有一个同名静态 TF 发布者。

### 19.4 没有高程输出

依次检查：

```bash
ros2 topic hz /cloud_registered
ros2 topic hz /Odometry
ros2 topic echo /cloud_registered --field header --once
ros2 topic echo /Odometry --once
ros2 run tf2_ros tf2_echo g1_world camera_init
```

### 19.5 RViz 中 `/cloud_registered` 倒置

不要把 `camera_init` 当作最终 Z 向上坐标。使用 `g1_world` 作为 RViz Fixed Frame，
并确认静态 TF 已运行。

## 20. 推荐给新对话的阅读任务

把本文档交给另一个对话后，可以使用下面的任务描述：

```text
请先完整阅读：
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/test/ROLLING_MAP_STARTUP_AND_CODE_GUIDE.md

然后逐行分析：
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/lidar/rolling_history.py
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/lidar/rolling_elevation_map.py

分析时请重点回答：
1. 每一个坐标变换的方向是否正确；
2. 行向量和旋转矩阵转置是否一致；
3. 点云和Odometry同步是否有快速转向风险；
4. 前进、倒退和转向时历史点如何变化；
5. 10秒历史、200帧上限和3.2m空间窗口分别限制什么；
6. 最大高度、最新年龄和高度来源年龄有什么区别；
7. 怎样在不改变11x17策略契约的情况下升级为持久世界高程格；
8. 协方差融合需要哪些状态、输入和失效条件；
9. 哪些逻辑已经实现，哪些只是未来设计；
10. 在提出修改前先用现有bag复现当前基线。

不要启动policy.pt，不要发送LowCmd，不要把旧unitree_deploy包当作当前实现。
```

## 21. 学习时最重要的结论

1. FAST-LIO2 提供运动估计和世界注册点云，不生成最终 LP 高程；
2. `rolling_elevation_map.py` 同时依赖 `/cloud_registered` 和 `/Odometry`；
3. 历史点保存在 `g1_world`，所以能够补偿前进、倒退和转向；
4. Phase 2 是 10 秒限时世界点历史；Phase 3 是独立的持久局部世界高程格；
5. 冷启动盲区是硬件可观测性限制，必须依赖 VP、先验或安全运动；
6. 停止 10 秒后地图消失符合当前设计，但不满足长期 LP 停止后继续行走；
7. 持久世界格负责保留已知地形，协方差融合负责管理测量可信度；
8. 当前 `policy_age` 不是决定高度的点的年龄；
9. 当前包名是 `lidar`，旧 `unitree_deploy` 入口不能继续使用；
10. 在真实策略接入前，必须先建立 LP 就绪、超时和退出保护。
