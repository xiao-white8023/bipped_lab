# Phase 4：D435i 到 RENet VP 输入

## 1. 这个节点解决什么问题

`depth_policy_preprocessor` 不输出供人观看的普通缩略深度图，而是输出
`policy.pt` 的 VP 分支实际训练契约：

```text
D435i /depth/image_rect_raw + /depth/camera_info
  （相机原生 spatial/temporal 滤波已按要求关闭，见 1.2）
  -> 按 CameraInfo 将训练相机射线投到真实深度图
  -> 64x36 训练视场（89.51° x 58.29°）
  -> 裁掉上 18、左 16、右 16 像素
  -> 18x32
  -> inf/nan 填 2.5 m（0/负值原样保留，见 1.1）
  -> [可选] 固定 GaussianBlur(3x3, sigma=1.0)（当前关闭，见 1.2）
  -> clip [0.1, 2.5] m
  -> 除以 2.5
  -> 10 Hz、两帧历史（旧帧在前，新帧在后）
```

节点只做策略契约所需的射线采样、有限值处理、截断与归一化，不加训练时的
任何随机深度噪声。**当前配置故意偏离训练契约两处**（见 1.2），其余步骤仍
与 `RENet_env.get_processed_deepcamera` 逐步一致。

### 1.1 无效深度（`invalid_depth_mode`）

训练侧 `RENet_env.get_processed_deepcamera` 只显式处理 `inf`/`nan`
（改成 `depth_max`），**不对 0 或负值做任何特殊处理**：它们会原样进入
GaussianBlur，随后被 `clip([0.1, 2.5])` 拉到近端。训练相机是 ray caster，
无命中时返回 `max_distance`，所以训练数据里本来不会出现 0。

D435i 的 0 表示“没有测到”（暗面、镜面、掠射角、超量程）。部署端因此提供
两种模式：

| 模式 | 0 / 负值 | inf / nan | 视场外射线 | 说明 |
|---|---|---|---|---|
| `training`（默认） | 原样进入模糊 → 最终被 clip 到 0.1 m | 2.5 m | 2.5 m | 与训练代码路径逐步一致 |
| `max` | 2.5 m | 2.5 m | 2.5 m | 旧部署行为，仅用于真机 A/B 对照 |

注意 `training` 的语义后果：较大的孔洞内部会饱和到**近端 0.1 m**（归一化
0.04）而不是远端 2.5 m。以 848x480 源图上一个 50x160 像素的孔洞为例，18x32
输出里会有约 20 个像素从 `max` 的 1.0 变成 `training` 的 0.04。两种填充都
不等于训练分布（训练在该像素上看到的是真实地面），因此建议在吊装状态下
用 `invalid_depth_mode` 做一次 A/B，比较 `/g1/policy/joint_target`。

### 1.2 两处有意的契约偏离（`blur_enable` 与 RealSense 滤波）

VP 输入目前**不是**训练观测的严格复刻，以下两项按需求关闭：

| 项 | 训练侧 | 当前部署 | 位置 |
|---|---|---|---|
| 固定高斯模糊 | `GaussianBlur(3, sigma=1.0)`，确定性，属于观测契约 | **关闭**（`blur_enable: false`） | `config/depth_policy.yaml` |
| RealSense spatial filter | 无（训练是 ray caster 真值） | **关闭**（`spatial_filter.enable: false`） | `config/realsense_d435i.yaml` |
| RealSense temporal filter | 无（每帧独立） | **关闭**（`temporal_filter.enable: false`） | `config/realsense_d435i.yaml` |

关闭 temporal filter 的收益是明确的：原先是 alpha=0.4 的 IIR，时间常数约
2.5 帧（30 fps 下约 80 ms），既给深度引入滞后，又让 10 Hz 采样出的相邻帧高
度相关，这两点训练里都不存在。

关闭高斯模糊则使观测比训练**更锐利**：策略见过的是 3x3 低通后的深度，现在
直接喂原始投影值，台阶/边缘的高频分量比训练时更强。这属于有意的分布偏移，
必须靠真机 A/B 判断收益，不能假设无影响。

回退方式：`blur_enable` 改回 `true`、两个 `*.enable` 改回 `true`，即恢复与
训练契约一致的链路。做 A/B 时只改一项，其它参数（`invalid_depth_mode`、
`sampling_mode`、相机 profile）保持不变。

## 2. 为什么不能直接 resize

D435i 实际分辨率、内参和主点来自当前 profile 的 `CameraInfo`。训练相机是
固定视场的针孔相机，而且策略只保留下半视场与中央半宽。整幅图直接 resize
会让 18x32 中每个像素对应错误的空间射线。节点因此先构造训练相机的目标
射线，再用真实内参到 D435i 的 rectified depth 中采样。

默认使用 nearest 采样，避免在台阶边缘产生双线性混合，尽量保留真实深度
不连续性。`bilinear` 仅作为可配置的对照实验。

## 3. 相机外参边界

训练配置中的相机相对 `torso_link` 位姿为：

```text
position = (0.0576235, 0.01753, 0.42987) m
quaternion_wxyz = (0.9149595, 0.0, 0.4035447, 0.0)
```

该四元数对应约 47.6° 的俯仰，与 G1 官方视场图中 42.4° 的互余关系一致。
裁剪后保留的像素中心相对相机光轴约为水平 `[-25.65°, +25.65°]`、竖直
`[+0.89°, +28.46°]`；按训练安装角计算，它主要观察前下方约
`48.49°～76.06°` 的射线，而不是整幅 D435i 视场。
但官方视场图不能替代精确外参标定。当前节点只对同一个真实光学坐标系中的
射线做内参重采样，不使用 TF 去“修正”相机安装误差；仅靠二维深度图也无法
严格补偿相机平移造成的遮挡变化。因此在 VP shadow inference 前，必须确认
真机 D435i 光学轴、安装高度和训练位姿一致。若不一致，应先修正训练/重训或
做带外参的三维重投影验证，不能把偏差藏进 crop 参数。

## 4. ROS 接口

输入：

- `/camera/depth/image_rect_raw`：`sensor_msgs/Image`，支持 `16UC1`、
  `mono16`、`32FC1`；
- `/camera/depth/camera_info`：`sensor_msgs/CameraInfo`。

输出：

- `/g1/depth/policy_frame`：`32FC1`，18x32，值域 `[0.04, 1.0]`，header
  沿用源深度图；
- `/g1/depth/policy_history`：`Float32MultiArray`，形状 `[2,18,32]`，
  C-order 展平，最旧帧在前；
- `/g1/depth/policy_valid`：相机射线覆盖、有效像素率、时间戳和 frame_id
  健康检查的合并结果；
- `/g1/depth/source_age`：源时间戳年龄，秒；
- `/g1/depth/valid_ratio`：目标 576 个射线上的有效深度比例。

控制节点应同时做自己的消息超时保护。`policy_history` 没有 header，因此最终
控制节点若要进行严格的原子快照，应订阅带 header 的 `policy_frame`，使用
`g1_depth.depth_policy.PolicyDepthHistory` 按源时间维护两帧；当前 history topic
主要用于 shadow inference 与联调核对。

## 5. 启动前先确认真机 topic

G1 原厂 bringup 通常会启动 D435i；也可单独执行：

```bash
ros2 launch g1_depth_camera realsense_d435i.launch.py
```

当前 G1 上的 RealSense ROS 2 4.51.1 使用 `/camera/depth/...`。若更换
launch 或 namespace，仍应先检查实际话题：

```bash
ros2 topic list | grep -E 'depth/(image_rect_raw|camera_info)$'
ros2 topic info -v /camera/depth/image_rect_raw
ros2 topic hz /camera/depth/image_rect_raw
ros2 topic echo /camera/depth/camera_info
```

若话题命名发生变化，只修改 `depth_policy.yaml` 的两个输入topic。不要通过
修改目标FOV或crop来迁就相机topic/profile。

## 6. 构建与启动

```bash
cd /home/cjh/TienKung-Lab/unitree_deploy_ws
colcon build --symlink-install --packages-select g1_depth
source install/setup.bash

ros2 run g1_depth depth_policy_preprocessor --ros-args \
  --params-file src/g1_depth/config/depth_policy.yaml
```

机器人端也应使用同一个 `g1_depth` 包名，避免再次出现源码包名、安装包名与
可执行程序目录不一致：

```bash
cd /home/unitree/cjh
colcon build --symlink-install --packages-select g1_depth
source install/setup.bash

ros2 run g1_depth depth_policy_preprocessor --ros-args \
  --params-file /home/unitree/cjh/src/g1_depth/config/depth_policy.yaml
```

## 7. 与 PGTT 真机链路的关系

PGTT 的真机代码没有 VP/深度分支。其可借鉴部分是部署边界：L1 LiDAR 与 IMU
经 Point-LIO 得到世界系点云和里程计，Elevation Mapping 保存带均值/方差的
世界系高程格，最后围绕机器人采样 11x9 heightmap；策略控制前还要求机器人
站稳、地图稳定并由操作者显式按键解锁。

本项目 Phase 3 的持久高程图沿用了“世界系保存、机体系取样”的核心思想，
但没有照搬 PGTT 代码里缺少 freshness/shape/NaN 防护的 DDS 直接读取方式。
本 Phase 4 节点则是 RENet 独有的独立 VP 数据链，不应与 LP 高程图互相填补
像素。未来的小模型选择器只选择 LP/VP；两套输入仍须各自有限、带健康状态。

参考：

- [Unitree G1 官方开发文档](https://support.unitree.com/home/zh/G1_developer/about_G1)
- [PGTT 仓库](https://github.com/NtagkasAlex/phase_guided_terrain_traversal)
- [RealSense ROS 2 wrapper](https://github.com/realsenseai/realsense-ros)

## 8. 接策略前的硬门槛

至少完成以下检查后才能做 VP shadow inference：

1. `policy_frame` 恒为 18x32、`32FC1`，所有值有限；
2. `policy_history` 恒为 1152 项，首帧启动时两个槽位完全相同；
3. 稳态发布频率约 10 Hz，相机原始流约 30 Hz；
4. `target-ray coverage` 接近 100%，否则 profile/CameraInfo 不匹配；
5. `/g1/depth/policy_valid` 连续为真，source age 没有跨时钟域；
6. 平地、楼梯上行、楼梯下行、强光、黑色/反光表面分别录 bag 统计
   `valid_ratio`；
7. 控制节点中 VP 的 1152 项必须位于总观测 `[967:2119]`，顺序为
   `oldest.reshape(-1)` 后接 `newest.reshape(-1)`；
8. 即使选择 LP，VP 输入也必须保持有限值。当前导出网络两条分支都会计算，
   NaN 不能靠乘以零的 mask 消除。

本阶段仍只允许传感器、预处理和 shadow inference，不发送 LowCmd。

## 9. PC2 首次采包与离线回放

先记录原始消息，不要只记录预处理输出；否则事后无法复核深度单位、内参、
畸变模型与时间戳。将下面两个输入 topic 替换成第 5 节在真机查到的实际名称：

```bash
ros2 bag record -o g1_d435i_phase4_raw \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info
```

每种场景保持机器人静止约 10 秒，至少覆盖：平地、正对上行楼梯、正对下行
楼梯、黑色表面、反光表面和强光。采集期间另开终端检查：

```bash
ros2 topic hz /g1/depth/policy_frame
ros2 topic echo --once /g1/depth/policy_valid
ros2 topic echo --once /g1/depth/source_age
ros2 topic echo --once /g1/depth/valid_ratio
```

离线回放必须统一使用 ROS 仿真时钟，否则 bag 中的历史时间戳会被误判为过期：

```bash
ros2 run g1_depth depth_policy_preprocessor --ros-args \
  -p use_sim_time:=true \
  --params-file src/g1_depth/config/depth_policy.yaml

ros2 bag play g1_d435i_phase4_raw --clock
```

首次 bag 应保留原始 `CameraInfo` 输出及节点启动日志。只有确认分辨率、`K/P`
内参、`distortion_model`、`encoding`、`step`、`frame_id`、时间戳时钟域和
`valid_ratio` 后，才能决定是否需要改输入 topic、`depth_unit_scale` 或
`expected_frame_id`；训练 FOV、crop、量程和历史顺序不属于现场调参项。
