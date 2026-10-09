# G1 RENet 真机部署交接文档

更新日期：2026-09-28（Asia/Shanghai）

> **2026-09-29 补充：** 本文后面关于“当前单帧”和“尚未实现持久地图”的状态
> 已经过时。Phase 2 已是 10 秒世界点历史；Phase 3 已新增持久局部世界高程格、
> 协方差融合、冲突确认与位姿跳变清图。本轮最新依据请优先阅读
> `ROLLING_MAP_STARTUP_AND_CODE_GUIDE.md` 和 `PHASE3_PERSISTENT_MAP.md`。Phase 3
> 目前只在本地工作区实现并完成实机 bag 回放，尚未迁移到机器人，也未接入 LP
> 控制。

## 1. 当前目标与进度

目标是将以下 G1 RENet 策略部署到宇树 G1 真机：

```text
/home/cjh/TienKung-Lab/logs/g1_renet/2026-09-04_14-11-25/exported/policy.pt
```

用户确认使用的模式称为 `pgtt_full_atten`。从导出配置看，它由两部分组成：

```text
高程来源：pgtt_full
编码器：ame_point_cross_attention
```

当前已完成：

- MID360 驱动和原始雷达/IMU数据读取；
- 机载 ROS 2 Foxy 环境与时间同步修复；
- 静态点云过滤、TF、ROI与楼梯可视化；
- 11×17静态诊断高程图；
- FAST-LIO2 在机载电脑编译和运行；
- `/Odometry` 与 `/cloud_registered` 稳定输出；
- FAST-LIO运动数据 rosbag 录制和初步里程计分析；
- 本地实现严格使用当前单帧的动态高程图节点。

当前尚未完成：

- 新的单帧动态高程图节点还没有部署并在机器人端回放验证；
- 单帧真实MID360在187个策略格上的缺失分布尚未统计；
- 完整策略观测向量尚未逐项对齐；
- 尚未进行 shadow inference；
- 尚未接入深度图；
- 尚未向机器人发送策略动作或 LowCmd。

当前安全状态：**策略控制关闭，只运行传感器、定位、地图和可视化。**

## 2. 当前采用的部署架构

算法当前全部放在机器人机载电脑运行：

```text
MID360
  -> Livox ROS Driver2（CustomMsg + IMU）
  -> FAST-LIO2（去畸变、里程计、当前注册点云）
  -> rolling_elevation_map（当前单帧）
  -> 11×17、187项策略高程输入
  -> 后续 shadow inference
```

部署位置：

- Livox驱动：机器人机载电脑；
- FAST-LIO2：机器人机载电脑；
- 动态高程图：机器人机载电脑；
- RViz2：通过NoMachine在机器人机载电脑显示；
- 本地电脑：开发、代码保存、配置审查和离线分析。

采用此方案的原因：

- 机器人是Ubuntu 20.04 + ROS 2 Foxy；
- 本地电脑是Ubuntu 24.04 + ROS 2 Jazzy；
- 跨版本、跨RMW传输曾触发CycloneDDS反序列化错误；
- 机载Foxy内部运行当前最稳定。

## 3. 关键路径

### 3.1 本地电脑

工作区：

```text
/home/cjh/TienKung-Lab/unitree_deploy_ws
```

本地ROS包：

```text
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar
```

当前核心文件：

```text
lidar/lidar_point.py
lidar/cloud_preprocessor.py
lidar/static_elevation_map.py
lidar/rolling_elevation_map.py
setup.py
package.xml
```

本地静态楼梯bag：

```text
/home/cjh/TienKung-Lab/bags/stairs_static_01
```

策略与配置：

```text
/home/cjh/TienKung-Lab/logs/g1_renet/2026-09-04_14-11-25/exported/policy.pt
/home/cjh/TienKung-Lab/logs/g1_renet/2026-09-04_14-11-25/params/lp_observation_contract.yaml
/home/cjh/TienKung-Lab/logs/g1_renet/2026-09-04_14-11-25/params/env.yaml
/home/cjh/TienKung-Lab/logs/g1_renet/2026-09-04_14-11-25/params/agent.yaml
```

### 3.2 机器人机载电脑

网络地址：

```text
无线：unitree@192.168.3.157
有线：unitree@192.168.123.164
```

ROS 2：

```text
/opt/ros/foxy
```

Livox工作区：

```text
/home/unitree/Livox-SDK2-ros2
```

FAST-LIO2工作区：

```text
/home/unitree/fastlio2_ws
```

FAST-LIO2配置：

```text
/home/unitree/fastlio2_ws/src/fast_lio/config/g1_mid360.yaml
```

现有部署工作区根目录根据运行记录为：

```text
/home/unitree/cjh/src
```

机器人端Python包名是 `unitree_deploy`，本地包名是 `lidar`。因此相同节点的命令前缀不同：

```bash
# 本地
ros2 run lidar <executable>

# 机器人
ros2 run unitree_deploy <executable>
```

新的 `rolling_elevation_map.py` 目前只在本地包中，尚未复制到机器人端 `unitree_deploy` 包。

## 4. 硬件、网络和时间

硬件：

- 机器人：Unitree G1；
- 雷达：Livox MID360，倒置安装在机器人头部；
- 雷达IP：`192.168.123.120`；
- 机器人有线IP：`192.168.123.164`；
- Livox ROS Driver2版本：`1.2.4`。

Livox配置：

```text
/home/unitree/Livox-SDK2-ros2/src/livox_ros_driver2/config/MID360_config.json
host IP  = 192.168.123.164
lidar IP = 192.168.123.120
```

机器人使用 `chrony/chronyd`。曾经错误同步到偏差约578秒的 `192.168.123.161`，后已切换到公网NTP并执行：

```bash
sudo chronyc online
sudo chronyc makestep
sudo hwclock --systohc
```

每次真机实验前检查：

```bash
chronyc tracking
chronyc sources -v
date '+ROBOT %F %T.%N epoch=%s.%N'
```

要求：

```text
Leap status = Normal
System time 接近 0 seconds slow/fast of NTP time
```

## 5. 统一的机载ROS环境

新终端建议显式执行：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
```

不要同时启动两个Livox驱动。若出现：

```text
bind failed
Failed to init livox lidar sdk
```

优先检查：

```bash
ps -ef | grep '[l]ivox_ros_driver2_node'
ros2 node list
ros2 topic hz /livox/lidar
```

这通常说明已有驱动占用了Livox UDP端口。

## 6. 当前Livox驱动模式

FAST-LIO2需要Livox自定义消息，因此活动驱动命令使用：

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

活动话题：

```text
/livox/lidar  livox_ros_driver2/msg/CustomMsg  约10 Hz
/livox/imu    sensor_msgs/msg/Imu              约200 Hz
```

注意：

- `xfer_format:=1` 用于FAST-LIO2；
- 早期诊断曾用 `xfer_format:=0` 输出 `PointCloud2`；
- `cloud_preprocessor.py` 订阅的是 `PointCloud2`，因此它不在当前CustomMsg + FAST-LIO活动链路中；
- 不要同时运行xfer_format 0和1的两个驱动实例。

检查：

```bash
ros2 topic type /livox/lidar
ros2 topic type /livox/imu
ros2 topic hz /livox/lidar
ros2 topic hz /livox/imu
```

## 7. FAST-LIO2状态与启动

仓库分支：

```text
/home/unitree/fastlio2_ws/src/fast_lio
git branch = ros2
ikd-Tree submodule已初始化
```

为编译安装过：

```bash
sudo apt install -y ros-foxy-pcl-ros ros-foxy-pcl-conversions
```

当前已确认配置项：

```text
point_filter_num: 3
blind: 0.5
dense_publish_en: true
```

当前外参基础值：

```text
extrinsic_T: [-0.011, -0.02329, 0.04412]
extrinsic_R: identity
```

`extrinsic_est_en` 是否已固定为 `false`，每次修改配置后应重新核对：

```bash
grep -nE 'lidar_type|scan_line|blind|point_filter_num|dense_publish_en|extrinsic_est_en|extrinsic_T|extrinsic_R' \
  /home/unitree/fastlio2_ws/src/fast_lio/config/g1_mid360.yaml
```

启动：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

ros2 launch fast_lio mapping.launch.py \
  config_path:=/home/unitree/fastlio2_ws/src/fast_lio/config \
  config_file:=g1_mid360.yaml \
  rviz:=false
```

已确认输出：

```text
/Odometry         nav_msgs/msg/Odometry        约10 Hz
/cloud_registered sensor_msgs/msg/PointCloud2  约10 Hz
```

坐标语义：

```text
/cloud_registered.header.frame_id = camera_init
/Odometry.header.frame_id          = camera_init
/Odometry.child_frame_id           = body
```

FAST-LIO刚启动时出现少量：

```text
No point, skip this scan!
```

随后完成IMU初始化并稳定输出，不属于持续故障。

## 8. 当前坐标系与TF

### 8.1 头部雷达外参

早期静态诊断使用：

```bash
ros2 run tf2_ros static_transform_publisher \
  0.0002835 0.00003 0.428434 \
  0 0.05112069379091391 3.141592653589793 \
  torso_link livox_frame
```

对应近似关系：

```text
Translation torso<-livox = [0.0002835, 0.00003, 0.428434]
Quaternion xyzw          = [0.999673352, 0, -0.025557564, 0]
```

MID360安装在头部；`torso_link` 只是机器人躯干参考坐标。该外参已完成方向和大致位置验证，尚未完成严格标定。

### 8.2 FAST-LIO世界方向修正

FAST-LIO的 `camera_init` 点云受倒置安装影响，需要下列静态旋转得到Z向上的 `g1_world`：

```bash
ros2 run tf2_ros static_transform_publisher \
  0 0 0 \
  0.9996734 0 -0.0255576 0 \
  g1_world camera_init
```

这是四元数形式：

```text
x y z qx qy qz qw parent child
```

### 8.3 FAST-LIO body到torso

根据FAST-LIO外参与头部安装外参组合得到当前默认值：

```text
T_body_torso translation = [0.01060918, -0.02326, 0.47200879]
T_body_torso quaternion  = [0.999673352, 0, -0.025557564, 0]  # xyzw
```

动态节点使用：

```text
T_world_torso = T_world_camera_init × T_camera_init_body × T_body_torso
```

如果修改FAST-LIO的 `extrinsic_T/R` 或重新标定雷达安装位姿，必须重新计算 `T_body_torso`。

### 8.4 capture_torso_yaw

动态高程图节点生成：

```text
capture_torso_yaw
```

定义：

- 原点：当前torso位置；
- X：躯干前向投影到水平面；
- Y：机器人左方；
- Z：重力反方向，向上；
- 保留yaw，去除roll和pitch。

RViz查看策略地图时可将Fixed Frame设为 `capture_torso_yaw`。

## 9. `pgtt_full_atten`策略高程契约

配置证据：

```text
lp_observation_contract.yaml:
  source: pgtt_full
  shape: [11, 17]
  history_frames: 1
  update_period: 0.1
  frame: capture_torso_yaw
  order: single/y_ascending/x_ascending

env.yaml:
  renet.lp_height_source: pgtt_full

agent.yaml:
  lp_encoder_type: ame_point_cross_attention
  lp_height_history_frames: 1
```

因此用户所称的 `pgtt_full_atten` 应理解为：

```text
pgtt_full高程来源 + attention高程编码器
```

它不是 `lp_observation_contract.yaml` 中另一种source字符串，也不存在元数据冲突。

精确地形契约：

```text
shape          = 11 x 17
cell count     = 187
resolution     = 0.1 m
x              = [-0.8, 0.8] m，共17列
y              = [-0.5, 0.5] m，共11行
frame          = capture_torso_yaw
order          = 单帧 / Y递增 / 每行X递增 / C-order展平
update period  = 0.1 s
height history = 1帧
invalid height = -1.1
height clip    = [-1.0, 1.0]
action dim     = 23
```

高度转换：

```text
policy_height = clip(torso_z - terrain_z - 0.5, -1, 1)
```

在 `capture_torso_yaw` 中，torso原点的局部Z为0，因此等价于：

```text
policy_height = clip(-terrain_z_local - 0.5, -1, 1)
```

必须区分：

- `lp_height.history_frames=1`：当前策略的高程输入只有当前一帧；
- `lp_proprio_history_frames=5`：本体状态编码使用的历史，与高程帧数不同；
- `lidar_elevation_map.history_frames=8/history_seconds=0.8`：同一配置文件中的另一套LiDAR滚动地图实验参数，不是本次 `pgtt_full` 输入契约；
- `gait.period=0.8`：步态周期，也不是点云记忆。

## 10. 动态单帧高程图节点

本地文件：

```text
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/lidar/rolling_elevation_map.py
```

本地入口：

```bash
ros2 run lidar rolling_elevation_map
```

机器人端预期入口：

```bash
ros2 run unitree_deploy rolling_elevation_map
```

但机器人端部署尚未执行。

输入：

```text
/cloud_registered  sensor_msgs/msg/PointCloud2
/Odometry          nav_msgs/msg/Odometry
TF g1_world <- camera_init
```

处理过程：

1. 用时间戳近似同步 `/cloud_registered` 与 `/Odometry`；
2. 检查点云frame=`camera_init`、里程计关系=`camera_init -> body`；
3. 组合当前 `T_world_torso`；
4. 将当前注册点云转换到 `g1_world`；
5. 建立当前重力对齐的 `capture_torso_yaw`；
6. 只使用当前这一帧点云；
7. 将当前帧点落入0.1 m网格，每格暂取最高局部Z；
8. 内部空间范围为3.2×3.2 m、33×33，仅作为空间落格与调试范围；
9. 中心裁剪X=[-0.8,0.8]、Y=[-0.5,0.5]为11×17；
10. 按策略公式转换为187个高度值。

这里的33×33是当前帧的空间网格，不是时间历史。

输出：

```text
/g1/elevation/policy_heights   std_msgs/msg/Float32MultiArray
/g1/elevation/policy_valid     std_msgs/msg/UInt8MultiArray
/g1/elevation/valid_ratio      std_msgs/msg/Float32
/g1/elevation/pose_sync_error  std_msgs/msg/Float32，单位秒
/g1/elevation/internal_cloud   sensor_msgs/msg/PointCloud2
/g1/elevation/policy_cloud     sensor_msgs/msg/PointCloud2
/tf                            g1_world -> capture_torso_yaw
```

关键保证：

- 不保存0.8秒历史；
- 不进行多帧点云累积；
- 每次输出只由当前 `/cloud_registered` 一帧生成；
- 数组严格为11×17=187项；
- 未观测格保持 `-1.1`；
- `policy_valid` 只用于诊断，不是当前模型的额外输入。

本地验证已通过：

```text
python3 -m py_compile: PASS
ament_flake8: PASS
ament_pep257: PASS
colcon build --symlink-install --packages-select lidar: PASS
节点初始化与正常退出: PASS
合成11×17平地：187格有效且编码高度为0.0: PASS
初始组合后的torso方向Z向上: PASS
```

## 11. 单帧RayCaster与真实MID360的差异

仿真PGTT RayCaster在一个更新周期内直接对187个固定XY位置向地形投射射线，通常可以得到稠密的当前地形高度。

真实MID360当前单帧具有不同特性：

- 扫描线不会保证每个0.1 m格都有回波；
- 近身和脚下区域存在安装视场盲区；
- 踏面、立面和遮挡会导致空间采样不均；
- FAST-LIO的 `/cloud_registered` 是当前去畸变并注册到世界坐标的扫描，不是完整稠密地形图；
- 即使使用 `dense_publish_en: true`，也不等于187个策略格全部有效。

因此严格单帧实现很可能出现较多 `-1.1`。下一步必须先测量缺格比例和空间位置，再决定：

- 仅做有依据的空间插值；
- 拟合局部地面/踏面；
- 融合深度图；
- 或在仿真中加入真实传感器缺失模型后重新训练。

在完成验证前，不应把时间累积悄悄加入策略输入来掩盖缺格。

## 12. 旧静态诊断链路

以下节点仍保留，但不属于当前FAST-LIO策略链路：

```text
lidar_point.py
cloud_preprocessor.py
static_elevation_map.py
```

### 12.1 lidar_point

用途：诊断 `PointCloud2` 字段、频率、有限点、距离分布，不发布控制。

早期xfer_format=0实测格式：

```text
x@0 FLOAT32
y@4 FLOAT32
z@8 FLOAT32
intensity@12 FLOAT32
tag@16 UINT8
line@17 UINT8
timestamp@18 FLOAT64
point_step=26
```

原始点云约30%为 `(0,0,0)` 或近零无回波占位点，`is_dense=True` 不能排除它们。有效性过滤应至少满足：

```text
finite XYZ and range > 0.1 m
```

### 12.2 cloud_preprocessor

早期输入为 `/livox/lidar` 的 `PointCloud2`，完成：

```text
距离过滤 -> torso_link坐标变换 -> ROI裁剪
```

输出：

```text
/g1/lidar/filtered
/g1/lidar/elevation_input
/g1/lidar/valid_ratio
/g1/lidar/roi_ratio
```

当前Livox驱动使用CustomMsg，因此该节点不是活动链路的一部分。

### 12.3 static_elevation_map

该节点仅用于机器人静止时验证网格方向、高度公式和可视化。它使用8帧/0.8秒静态融合来缓解MID360稀疏，不能作为当前 `pgtt_full_atten` 策略输入。

已验证：

- 11×17和187项布局正确；
- Y递增、每行X递增；
- 未观测值为 `-1.1`；
- 输出约10 Hz；
- 静态楼梯bag可显示阶梯结构。

## 13. rosbag与现有验证证据

### 13.1 静态楼梯bag

本地：

```text
/home/cjh/TienKung-Lab/bags/stairs_static_01
```

信息：

```text
Duration: 86.049 s
Size: 约434.6 MiB
/livox/lidar: 860
/livox/imu: 17212
/tf_static: 1
```

用途：原始点云格式、无效点、头部TF、ROI和静态网格验证。

### 13.2 FAST-LIO运动验证bag

机器人端：

```text
/home/unitree/cjh/bags/fastlio_validation_03
```

信息：

```text
Duration: 39.743 s
Size: 240.1 MiB
/livox/lidar:      397
/livox/imu:        7928
/Odometry:         396
/cloud_registered: 396
/tf:               396
/tf_static:        1
```

初步里程计分析：

```text
Maximum displacement   : 0.7968 m
Start stationary RMS   : 0.0041 m
End stationary RMS     : 0.0046 m
Start/end position gap : 0.1050 m
Start/end rotation gap : 9.773 deg
```

该bag由人工遥控移动，机器人没有准确返回原位，因此0.105 m和9.773°不能直接解释为漂移。当前暂时接受FAST-LIO用于下一阶段；静止段RMS约4～5 mm。

本地还没有 `fastlio_validation_03`，因此新的动态节点尚未完成本地端到端bag测试。

录制CustomMsg bag或播放该bag前必须source Livox工作区，否则会出现：

```text
package 'livox_ros_driver2' not found
```

正确环境：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
```

## 14. 下一步：部署并验证单帧动态高程图

这是当前唯一的直接下一步。策略控制继续关闭。

### 14.1 先确定机器人端源码目录

在机器人端运行：

```bash
ros2 pkg prefix unitree_deploy
find /home/unitree/cjh -maxdepth 5 -path '*/unitree_deploy/setup.py' -print
find /home/unitree/cjh -maxdepth 6 -path '*/unitree_deploy/package.xml' -print
```

确认源码路径后：

1. 将本地 `rolling_elevation_map.py` 复制到机器人端Python模块；
2. 在机器人端 `setup.py` 增加：

```python
'rolling_elevation_map = unitree_deploy.rolling_elevation_map:main',
```

3. 在 `package.xml` 确认依赖：

```xml
<depend>geometry_msgs</depend>
<depend>nav_msgs</depend>
<depend>rclpy</depend>
<depend>sensor_msgs</depend>
<depend>std_msgs</depend>
<depend>tf2_ros</depend>
<exec_depend>python3-numpy</exec_depend>
```

4. 在正确工作区根目录构建并source。

### 14.2 机器人端回放验证顺序

停止实时Livox和FAST-LIO，避免与bag重复发布相同话题。然后依次运行：

终端1，环境与世界方向TF：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source <机器人部署工作区>/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  0 0 0 \
  0.9996734 0 -0.0255576 0 \
  g1_world camera_init
```

终端2，新节点：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash
source <机器人部署工作区>/install/setup.bash

ros2 run unitree_deploy rolling_elevation_map
```

终端3，回放：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/Livox-SDK2-ros2/install/local_setup.bash
source /home/unitree/fastlio2_ws/install/local_setup.bash

ros2 bag play /home/unitree/cjh/bags/fastlio_validation_03
```

如果bag中的 `/tf_static` 已经包含同名 `g1_world -> camera_init`，不要重复发布；先检查：

```bash
ros2 run tf2_ros tf2_echo g1_world camera_init
```

### 14.3 必查指标

```bash
ros2 topic hz /g1/elevation/policy_heights
ros2 topic hz /g1/elevation/policy_cloud
ros2 topic echo /g1/elevation/pose_sync_error
ros2 topic echo /g1/elevation/valid_ratio
```

节点日志应关注：

```text
current_scan_points=...
internal_valid=.../1089
policy_valid=.../187 (...%)
pose_dt=... ms
dropped=...
```

验收项：

1. `pose_dt` 接近0，且没有持续增长的 `dropped`；
2. `policy_heights` 每帧严格187项；
3. RViz Fixed Frame=`capture_torso_yaw` 时X前、Y左、Z上；
4. 机器人向前运动时，固定楼梯在机器人坐标中应向负X移动；
5. 转向时楼梯位置应按相反方向旋转，不应拖影或粘住；
6. 平地策略高度应接近0；
7. 每级台阶的高度差应与实物一致；
8. 统计单帧187格有效率和无效格位置。

RViz建议：

```text
Fixed Frame: capture_torso_yaw
PointCloud2: /g1/elevation/policy_cloud
Reliability: Best Effort
Style: Flat Squares
Size: 0.095 m
Color Transformer: AxisColor
Axis: Z
Decay Time: 0
```

`Decay Time` 必须为0，以免RViz显示历史造成“节点在累积”的错觉。

## 15. 动态高程图通过后的顺序

1. 根据单帧缺格实测结果决定空间补全方法；
2. 定义机器人本体点云mask，避免身体/手臂进入地形高度；
3. 重新检查踏面、立面和离群点下“每格最高Z”是否合适；
4. 完成严格雷达外参标定或误差评估；
5. 接入深度图，但先保持高程图为主；
6. 审查完整策略观测：本体状态、命令、相位、历史、归一化和拼接顺序；
7. 加载 `policy.pt` 做shadow inference，只记录输入和23维动作；
8. 检查动作关节顺序、默认姿态、缩放、限幅和频率；
9. 实现状态机、看门狗、急停和通信失效保护；
10. 具备悬吊/保护架后再讨论首次闭环控制。

## 16. 尚未解决的问题

- MID360头部外参尚未严格标定；
- FAST-LIO `extrinsic_est_en` 当前值需再次从配置确认；
- 单帧策略网格的真实有效率未知；
- 未观测格在策略中的实际影响尚未验证；
- 尚未定义机器人本体点云mask；
- 每格最高Z可能被立面、栏杆、腿部或离群点污染；
- FAST-LIO长时间、快速转向和楼梯运动时的漂移尚未系统验证；
- 深度图尚未接入；
- 整个策略观测向量尚未逐项审计；
- 23维动作与真实G1关节映射尚未验证；
- 尚未运行shadow inference；
- 尚未连接控制输出。

## 17. 安全边界

在以下条件全部满足前保持策略控制关闭：

- `capture_torso_yaw` 坐标方向经过动态验证；
- 单帧高程图位置、高度、顺序和无效值正确；
- 真实缺格不会被误当作平地；
- FAST-LIO在预期运动范围内稳定；
- 11×17展平顺序与训练完全一致；
- 策略完整输入的单位、顺序、缩放和时序一致；
- shadow inference输出合理；
- 23维动作与真实关节一一对应；
- 关节位置、速度和力矩限幅已验证；
- 通信看门狗、状态机和急停有效；
- 首次闭环测试具备悬吊/保护架和现场急停人员。

## 18. 新对话建议开场

```text
请先完整阅读：
/home/cjh/TienKung-Lab/unitree_deploy_ws/src/lidar/test/HANDOFF.md

这是G1 RENet真机部署的最新交接文档。当前模式是pgtt_full高程来源加ame_point_cross_attention编码器，即用户所称的pgtt_full_atten。FAST-LIO2已经在机器人端运行并输出/Odometry和/cloud_registered。本地已经实现严格只使用当前一帧、输出11×17=187项的rolling_elevation_map，但尚未部署到机器人端。下一步请从“第14节：部署并验证单帧动态高程图”继续。保持策略控制关闭，不发送LowCmd。
```
