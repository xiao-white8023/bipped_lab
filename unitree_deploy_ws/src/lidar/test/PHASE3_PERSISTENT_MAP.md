# Phase 3：持久局部世界高程格与协方差融合

更新日期：2026-09-29。

## 1. 目标和边界

本阶段只修改雷达高程图，不加载 `policy.pt`，不发送 `LowCmd`，也不实现 VP/LP
自动切换。

目标是：机器人已经通过运动观察到脚下地形并进入 LP 后，即使原地站立超过
10 秒，近场高程也不会因为固定历史队列到期而自动消失。

新增实现：

```text
lidar/persistent_grid.py
lidar/persistent_elevation_map.py
config/phase3_persistent.yaml
test/test_persistent_grid.py
```

Phase 2 的 `rolling_elevation_map.py + rolling_history.py` 保持为可复现的 A/B
基线，没有被替换。

## 2. 新链路

```text
/cloud_registered + /Odometry
             |
             v
按源时间戳最近邻同步、检查 frame、取得 g1_world <- camera_init
             |
             v
变换到当前 capture_torso_yaw，做 3.2 x 3.2 m ROI 和头部回波过滤
             |
             v
把雷达噪声和 Odometry 位姿协方差传播成每个点的世界 Z 方差
             |
             v
按世界 XY 栅格聚合，每格每帧保留最高点
             |
             v
一维 Kalman 高度融合 + Mahalanobis 冲突检测
             |
             v
只保留机器人周围 2.5 m 半径的世界固定稀疏栅格
             |
             v
世界地图只按点云频率融合；按最新/短时预测机器人位姿以 50 Hz
采样为 33 x 33，再截取 11 x 17
             |
             v
/g1/elevation/policy_heights + valid + age + variance
```

“持久”表示没有 10 秒硬删除，不表示创建无限大的全局地图。格子的键固定在
`g1_world`，但机器人走远后，超出 `retention_radius` 的格子会被删除。
地图只在节点本次运行的内存中持久；节点或 FAST-LIO 重启后不会从磁盘恢复旧图。

## 3. 单格状态

每个世界 XY 格保存：

```text
height                 融合后的世界 Z
variance               高度方差
last_seen              该格最近收到任何测量的源时间
last_update            当前主高度最近被融合或替换的源时间
observations            主高度累计观测数
candidate_*             与主高度冲突的新表面候选
```

同一格的正常测量使用一维 Kalman 更新：

```text
K = old_variance / (old_variance + measurement_variance)
height = height + K * (measurement - height)
variance = (1 - K) * old_variance
```

测量方差由三部分组成：

```text
雷达基础高度噪声
+ 随测距增加的高度噪声
+ Odometry 位置 Z 协方差
+ Odometry roll/pitch 协方差经点位置 Jacobian 传播到 Z 的部分
```

代码把非有限或非半正定的协方差块投影为半正定矩阵。Odometry 协方差为全零
时，至少仍使用雷达噪声，不会得到零方差。

## 4. 为什么不能让单帧直接改写旧地图

长期保存后，动态物体、飞点、FAST-LIO 短时误差或上下两层结构都可能与旧高度
冲突。Phase 3 用标准化残差判断：

```text
(measurement - height)^2
---------------------------------  > mahalanobis_threshold^2
old_variance + measurement_variance
```

严重冲突的点先进入候选表面。默认必须连续 3 次、相互高度差不超过 5 cm，才会
替换旧高度；一个孤立飞点不能立即覆盖地图。

## 5. 停止超过 10 秒时发生什么

默认：

```yaml
max_cell_age_seconds: 0.0
variance_growth_per_second: 0.000001
```

`max_cell_age_seconds: 0.0` 表示不按年龄硬删除。停止时格子继续存在，但方差以
很慢的速度增长。因此：

- 不会在第 10 秒突然整片消失；
- 越久没有重新观察，可信度会缓慢下降；
- 若方差最终超过 `max_valid_variance`，该格输出为无效，而不是把不可信旧值送给
  策略；
- 再次得到一致观测后，方差会下降；
- 冷启动时从未看见的盲区仍然无效，软件不能凭空恢复。

## 6. 坐标系重置保护

持久地图最危险的情况是 FAST-LIO 世界坐标发生离散跳变，而旧格仍留在旧世界系。
节点在以下情况清空地图：

- 点云源时间戳倒退，例如 rosbag 循环或传感器重启；
- 相邻已处理帧的 FAST-LIO 躯干位置跳变超过 0.5 m；
- 相邻已处理帧的 yaw 跳变超过 45 度。

清图后应视为重新冷启动，不能继续依赖 LP。清图次数发布在：

```text
/g1/elevation/map_reset_count
```

这只能处理离散跳变，不能消除 FAST-LIO 缓慢累积漂移。

## 7. 输出契约

原有策略接口不变：

```text
/g1/elevation/policy_heights       std_msgs/msg/Float32MultiArray，11 x 17
/g1/elevation/policy_valid         std_msgs/msg/UInt8MultiArray，11 x 17
/g1/elevation/policy_age           std_msgs/msg/Float32MultiArray，11 x 17
/g1/elevation/valid_ratio          std_msgs/msg/Float32
/g1/elevation/pose_sync_error      std_msgs/msg/Float32
/g1/elevation/history_span         std_msgs/msg/Float32
/g1/elevation/internal_cloud       sensor_msgs/msg/PointCloud2
/g1/elevation/policy_cloud         sensor_msgs/msg/PointCloud2
```

Phase 3 中 `policy_age` 是“当前融合高度最后一次被更新的年龄”，不是任意命中点的
最新年龄。无效格为 `-1`。

新增诊断：

```text
/g1/elevation/policy_variance      11 x 17，高度方差；无效格为 -1
/g1/elevation/policy_freshest_age  11 x 17，最近任何测量年龄；无效格为 -1
/g1/elevation/source_stamp         当前输出对应的点云源时间
/g1/elevation/map_cell_count       当前持久世界格数
/g1/elevation/map_reset_count      启动后清图次数
/g1/elevation/map_processing_time  单帧世界地图融合计算时间，单位秒
/g1/elevation/source_to_output_latency 点云源时间到策略输出时间，单位秒
/g1/elevation/pose_prediction_horizon  当前策略图使用的预测时域，单位秒
/g1/elevation/sampling_key_switches    本帧 33 x 33 查询格实际切换数量
/g1/elevation/superseded_clouds        为防止积压而覆盖的旧待同步点云数
```

`history_span` 为兼容旧分析工具而保留，在 Phase 3 中表示当前地图最老主高度的年龄，
它可以明显大于 10 秒。

## 8. 楼梯边缘防抖与延迟补偿

世界格仍使用最近邻采样，不对台阶上下表面做双线性插值。查询点接近 10 cm
世界格边界时，节点保留上一次格键，直到越过普通边界额外 2 cm；若被保留的格无效、
而当前最近格有效，则立即回退到当前最近格。地图清空时迟滞状态也会清空。

点云融合和策略采样已经解耦：

```text
约 10 Hz /cloud_registered -> 更新世界地图
约 10 Hz /Odometry -> 最近 0.25 s 位姿拟合速度
每次地图融合完成 -> 生成只读世界格快照
独立 50 Hz 发布线程 -> 最多向前预测 0.25 s -> 查询 11 x 17 策略图
```

融合线程只修改主地图；50 Hz 线程只读取上一份完整快照，因此不会读到更新一半的
栅格。实时路径只查询策略真正使用的 11 x 17 格，并发布 `policy_heights` 和
`policy_valid`。年龄、方差、延迟等诊断以 10 Hz 发布，33 x 33 RViz 点云以 2 Hz
生成。这样点云融合耗时超过 20 ms 时也不会持续吞掉策略发布周期。

现有 FAST-LIO 的 `Odometry.twist` 在实机 bag 中为全零，代码因此不使用它。平移速度
低于 0.08 m/s 时视为静止定位噪声；角速度只使用拟合值的 50%，以减轻转弯减速时
过冲。预测只补偿策略图相对机器人位置的滞后，并不会让旧点云变成新观测。

输入使用 sensor-data QoS depth 1，过载时保留最新点云，不允许旧点云排队使延迟继续
增长。

## 9. 本地构建和启动

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select lidar --symlink-install
source install/setup.bash

ros2 run lidar persistent_elevation_map --ros-args \
  --params-file \
  /home/cjh/TienKung-Lab/unitree_deploy_ws/install/lidar/share/lidar/config/phase3_persistent.yaml
```

启动后检查：

```bash
ros2 node list | grep g1_persistent_elevation_map
ros2 topic hz /g1/elevation/policy_heights
ros2 topic echo /g1/elevation/valid_ratio
ros2 topic echo /g1/elevation/map_cell_count
ros2 topic echo /g1/elevation/map_reset_count
ros2 topic echo /g1/elevation/map_processing_time
ros2 topic echo /g1/elevation/source_to_output_latency
ros2 topic echo /g1/elevation/pose_prediction_horizon
ros2 topic echo /g1/elevation/sampling_key_switches
```

Phase 2 与 Phase 3 发布同名主输出，A/B 测试时一次只能启动其中一个。

## 10. 机器人端迁移文件

本地包名是 `lidar`。把代码迁移到机器人当前工作区时，至少要同步：

```text
lidar/persistent_grid.py
lidar/persistent_elevation_map.py
lidar/pose_prediction.py
lidar/rolling_elevation_map.py
lidar/rolling_history.py
config/phase3_persistent.yaml
setup.py 中的新 console_scripts 和 config 安装项
```

如果机器人端继续使用旧包名 `unitree_deploy`，必须把新文件的相对导入路径、
`setup.py` 入口和 YAML 节点名一起保留，并将启动前缀改为：

```bash
ros2 run unitree_deploy persistent_elevation_map --ros-args \
  --params-file <机器人安装后的 phase3_persistent.yaml>
```

不要把本地 `ros2 run lidar ...` 原样当作机器人端已安装命令。

## 11. 当前验证结果

纯 NumPy 测试覆盖：

- 超过 10 秒仍可采样；
- 可选年龄上限；
- 随机器人移动做空间裁剪；
- 协方差加权融合；
- 冲突表面连续 3 次确认；
- 单帧每世界格保留最高点；
- 源时间戳倒退清图；
- 未观测时方差随时间增长；
- 发布用地图快照不会被后续融合原地修改。

2026-09-29 使用已有实机 bag
`g1_lidar_phase2_turn_reverse_20260929_01_0.db3` 回放前 60 秒输入：

- 成功处理约 560 帧；
- 点云与 Odometry 匹配误差为 0 ms；
- 丢弃点云 0；
- 坐标系重置 0；
- 地图建立后有效率曾达到约 84.5%；
- 后续部分有效格高度年龄达到 10～28 秒时仍然有效；
- 回放末段策略有效率约 61%，证明没有发生固定 10 秒整批到期。

这证明代码解决了“固定 10 秒删除”。它还不能证明长期高度绝对无漂移，也不能
直接授权接入真实 LP 控制。

对完整 `phase3` 实机 bag 重新回放本次防抖与延迟补偿代码：

- 持续 10 Hz 点云融合时，策略数组为 49.99 Hz；输出间隔中位数 20.07 ms，
  P95 29.27 ms，最大 36.32 ms；
- `policy_heights` 与 `policy_valid` 在独立契约检查中分别为 50.00 Hz 和
  49.99 Hz，消息数完全相同，所有消息均为 11 x 17；诊断数组约 9～10 Hz，
  `policy_cloud` 约 1.95 Hz；
- 同一段 10 秒回放中，地图融合为 9.99 Hz，单帧融合平均 18.3 ms、最大
  33.3 ms，地图重置和过期输入覆盖均为 0；
- 原分析中四个楼梯边缘问题格，在最后 32 秒静止段内的 `>5 cm` 和 `>10 cm`
  相邻帧跳变均由多次降为 0，最大变化约 1.08 cm；
- 持续融合测试中，点云源时间到输出平均约 162 ms；最新 Odometry 的预测时域平均
  约 142 ms。世界地图按最新预测机身位姿重新采样，所以机器人运动造成的相对错位
  得到补偿；FAST-LIO 已经产生的观测年龄仍然存在；
- 直接比较 bag 记录时间与消息头时间发现，在进入高程图节点之前，
  `/cloud_registered` 已平均落后 121.4 ms（P95 160.5 ms），`/Odometry` 已平均
  落后 118.4 ms（P95 157.3 ms）。因此剩余源时间年龄主要来自上游 FAST-LIO
  输出以及 10 Hz 扫描周期，不是 Phase 3 内部消息排队；
- 预测时域最大限制为 250 ms；超过这一范围不会继续盲目外推；
- 使用 bag 中未来 180 ms 的 Odometry 作离线参照，在移动段，平面位置误差中位数
  由不预测的 58.7 mm 降至约 28.3 mm，P95 由 95.9 mm 降至约 46.1 mm；
- 这些结果证明防抖和位姿对齐生效，但不等于 FAST-LIO 上游延迟已被物理消除。

## 12. 下一次真机只做雷达验证

真机下一步应在不启用策略的条件下：

1. VP 或人工安全控制走出冷启动盲区；
2. 记录地图已建立时的 `policy_valid`、`policy_age` 和 `policy_variance`；
3. 原地站立至少 30 秒；
4. 检查有效格是否保留、方差是否按预期缓慢增加；
5. 原地转向、倒退后检查地图是否仍固定在环境而不是跟随机身；
6. 对着台阶或木块重复观测，确认高度没有被单帧飞点改写；
7. 人为重启 FAST-LIO，确认 `map_reset_count` 增加且旧地图被清空；
8. 保存 `/cloud_registered`、`/Odometry`、`/tf_static` 和全部
   `/g1/elevation/*` 话题，再离线确定方差和冲突门限。

在这些检查通过前，保持 `policy.pt` 和 LP 控制关闭。

## 13. FAST-LIO 上游延迟的 20 Hz A/B 测试

Livox ROS Driver 2 的 `publish_freq` 默认是 10 Hz，官方支持 20 Hz、50 Hz 等值。
当前 FAST-LIO 把每条 `CustomMsg` 当成一帧，等待 IMU 覆盖到该帧结束时间后，再发布
同一结束时间戳的 `/cloud_registered` 和 `/Odometry`。因此可以先测试 20 Hz 是否在
保持定位质量的同时缩短观测年龄。

只在 VP 或吊架保护下测试，雷达驱动命令增加：

```bash
-p publish_freq:=20.0
```

完整驱动命令中的关键参数应为：

```bash
ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p user_config_path:=/home/unitree/Livox-SDK2-ros2/src/livox_ros_driver2/config/MID360_config.json \
  -p xfer_format:=1 \
  -p multi_topic:=0 \
  -p publish_freq:=20.0 \
  -p frame_id:=livox_frame
```

保持 FAST-LIO 和 Phase 3 其他配置不变，并检查：

```bash
timeout 10s ros2 topic hz /livox/lidar
timeout 10s ros2 topic hz /cloud_registered
timeout 10s ros2 topic hz /Odometry
timeout 10s ros2 topic hz /g1/elevation/policy_heights
```

同时记录 `/livox/lidar`、`/livox/imu`、`/cloud_registered`、`/Odometry`、`/tf_static`
和全部 `/g1/elevation/*`。只有在以下条件同时满足时才保留 20 Hz：

- `/cloud_registered` 与 `/Odometry` 接近 20 Hz；
- `policy_heights` 仍接近 50 Hz；
- `superseded_clouds` 不持续增加，`map_reset_count` 为 0；
- FAST-LIO 没有明显增加漂移、姿态抖动或跳变；
- 地图有效率、楼梯边缘稳定性和点云覆盖不低于 10 Hz 基线；
- 点云源时间到输出的延迟分布确实下降。

若输出仍为 10 Hz，或定位质量下降，则恢复 `publish_freq:=10.0`。50 Hz 驱动输出留到
20 Hz 验证通过后再评估。
