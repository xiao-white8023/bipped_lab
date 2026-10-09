# G1 23DoF 最终动作 AMP 数据集 v2

原四方向动作的 50 维版本。2026-10-04，`g1_renet` 和 `g1_rough` 的 50 维选项均已改为 [新四段 PKL 数据](../g1_23dof_pkl_locomotion_v1/README.md)；选择 58 维使用 [v3](../g1_23dof_locomotion_v3/README.md)。详见 [切换说明](../../../../../docs/g1_amp_format_switch.md)。以下保留 v2 的格式和当时的验证记录；重建旧格式需显式指定 `--amp-joint-count 19 --output-dir legged_lab/envs/g1/datasets/g1_23dof_locomotion_v2`。

本数据集用于 `g1_renet`，来源是 2026-10-02 用户确认的四段最终机器人动作。
完整保留自然站立、起步、移动、收步和最终站立。所有动作 30 FPS，首尾各
3 秒静止站立，总计 5400 帧、180 秒播放时长。

| 文件名 | 内容与版本 | 帧数 | 播放时长 | AMP 采样概率 |
|---|---|---:|---:|---:|
| `walk_forward_turns` | 前走 v8：直行、左转 90°、右转 90°、转身 180°及继续行走 | 1800 | 60 秒 | 25% |
| `walk_backward` | 后退 v2：髋部、躯干和摆臂协调最终版 | 1200 | 40 秒 | 25% |
| `walk_left` | 向左侧走最终版 | 1200 | 40 秒 | 25% |
| `walk_right` | 向右侧走最终版 | 1200 | 40 秒 | 25% |

每段 `MotionWeight=1`，现有加载器先等概率选取片段，再均匀抽取其中有效时间。
所有文件 `LoopMode=Clamp`；当前状态与 0.02 秒后的状态来自同一片段，按
30 FPS 源帧插值，保留起停及转向处的连续性。根部前后/侧向速度由任务的
命令跟踪奖励训练；现有 AMP 特征描述关节与末端姿态。

## 文件与格式

- `sources/*.pkl`：最终动作的原始副本，保留源文件 SHA-256，可独立重新生成特征。
- `amp/*.txt`：JSON，`Frames` 为 **T×50**，供 AMP 训练使用。
- `visualization/*.txt`：JSON，`Frames` 为 **T×58**，供 `play_amp_animation.py` 回放。
- `manifest.json`：关节/连杆顺序、坐标系、版本、完整动作阶段、来源、哈希和转换指标。
- `validation.json`：全部帧、URDF FK、训练加载器和判别器检查结果。
- `isaac_feature_parity.json`：实际 IsaacLab G1 模型与环境 AMP 函数的特征核对结果。
- `rebuild_validation.json`：仅使用数据包内的原始副本重建，12 个数据文件的哈希一致。

50 维训练特征顺序与选择 50 维时的 `G1RENetEnv.get_amp_obs_for_expert_trans()` 一致：

| Python 切片 | 内容 | 单位 |
|---|---|---|
| `[0:19]` | 19 个非脚踝关节角 | rad |
| `[19:38]` | 同顺序的 19 个关节速度 | rad/s |
| `[38:41]` | 左手连杆原点在 pelvis 坐标系的位置 | m |
| `[41:44]` | 右手连杆原点在 pelvis 坐标系的位置 | m |
| `[44:47]` | 左脚踝连杆原点在 pelvis 坐标系的位置 | m |
| `[47:50]` | 右脚踝连杆原点在 pelvis 坐标系的位置 | m |

19 个关节顺序为：左髋 pitch/roll/yaw、左膝、右髋 pitch/roll/yaw、右膝、
腰 yaw、左肩 pitch/roll/yaw、左肘、左腕 roll、右肩 pitch/roll/yaw、右肘、右腕 roll。
脚踝四个关节参与末端 FK，并完整保存在原始动作和 58 维预览中。

58 维预览顺序为：root XYZ(3)、root XYZ Euler(3)、23 个关节角、root 世界线速度(3)、
root 局部角速度(3)、23 个关节速度。PKL 四元数为 XYZW，预览欧拉角单位为 rad。
转换保留源关节角与 root XYZ，速度由每段动作内的有限差分计算；没有重新
规划脚步、裁剪、改变播放速度或再次调整地面高度。

## 任务接入与使用

`legged_lab/envs/__init__.py` 的 `g1_renet` 注册对应 `G1RENETAGENTCFG` 与
`G1RENETENVCFG`。以上为原配置记录；当前 `RENet_cfg.py` 的 `G1_RENET_AMP_DIM = 50` 使用新的 PKL 四段，需在对应 cfg 的数据目录与片段列表中明确指定本目录才能使用这套旧数据。
路径基于 `LEGGED_LAB_ROOT`，迁移仓库后可重新解析。原动作文件及其余任务保留。

在已配置好的 IsaacLab 环境中开始训练：

```bash
cd /home/cjh/TienKung-Lab
python legged_lab/scripts/train.py --task g1_renet --headless
```

查看导入后的机器人动作：

```bash
python legged_lab/scripts/play_amp_animation.py --task g1_renet --num_envs 1 \
  --motion_file legged_lab/envs/g1/datasets/g1_23dof_locomotion_v2/visualization/walk_forward_turns.txt \
  --root_height_offset 0.0 --repeat 1 --realtime
```

修改 `--motion_file` 的文件名即可查看后退或左右侧走。

重新构建（GMR 环境中已有 MuJoCo、NumPy、SciPy）：

```bash
cd /home/cjh/TienKung-Lab
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/build_g1_renet_final_dataset.py --from-snapshots --amp-joint-count 19 --output-dir legged_lab/envs/g1/datasets/g1_23dof_locomotion_v2
```

首次从 Kimodo 工作目录导入时省略 `--from-snapshots`。IsaacLab 特征核对命令：

```bash
python legged_lab/scripts/test_g1_renet_amp_feature_parity.py --headless --device cuda:0
```

## 来源与检查范围

前走来自 ACCAD 捕获的起停和行走风格，经 G1 适配及转向路线规划完成。
后退和左侧走来自 Kimodo SOMA 生成的人体风格，结合机器人脚步计划、IK 和姿态
修正完成；右侧走为左侧机器人动作的镜像。四段为编排后的机器人运动学参考，
保留的长时间周期不能视为等长的独立真人动捕。详细来源在 `manifest.json` 中。

检查覆盖特征顺序、坐标系、速度计算、原动作保持、源文件与模型哈希、关节范围、
20 ms 状态对插值、20 万条 GPU 预加载以及 100 维判别器输入的前向/梯度/奖励计算。
IsaacLab 核对覆盖四段动作各阶段共 120 个姿态，含改变世界位置与朝向的核对，
最大特征误差约 3.94×10⁻⁶。核对采用精确状态写入与 FK 刷新，物理跟踪和训练后的
步态需要后续训练验证。本次没有启动长期策略训练。
