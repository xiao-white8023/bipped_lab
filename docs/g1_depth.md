`g1_depth` 是独立的 G1 23DOF 深度训练任务，提取当前 RENet 的四头 VP，从头训练。环境、配置、模型与 runner 分别独立定义；共享机器人资产、地形、MDP 函数、CNN组件、AMP/PPO及数据加载器。

策略输入为 `1932 = 10×78 + 2×18×32`，本体与深度均按旧到新排列。单帧本体为角速度、投影重力、命令、关节位置偏移、关节速度和前一动作。深度两帧以10 Hz捕获，按训练相机射线裁剪为18×32，clip到0.1～2.5米后除以2.5。Actor没有高程输入或分支mask。

本体/深度编码经过attention和64维GRU，输出速度3、脚高2、context均值32、terrain latent32。Actor输入147维，MLP为147→512→256→128→23。context随机采样只用于VAE重建；动作均值与导出使用posterior mean。GRU hidden每个观测窗口重新计算。

critic保留1037维特权观测；地形与足底射线扫描仅用于训练监督/奖励。地形重建默认监督x≥0的99格，速度、脚高和干净下一帧本体目标沿用当前VP定义。VAE排除done与非finite下一帧目标；AMP使用reset前的真实终止状态。

默认128环境、50Hz控制、24步rollout、50000轮、每1000轮保存，seed42；AMP默认58维/23关节，支持切换50维/19非踝关节。动作延迟默认关闭，深度噪声与0.5概率模糊从训练开始启用。训练参数见 [g1_depth_cfg.py](/home/cjh/TienKung-Lab/legged_lab/envs/g1/g1_depth_cfg.py)。

修改该配置文件顶部的 `G1_DEPTH_AMP_DIM = 58` 为 `50` 即可切换，环境和agent必须使用同一开关值。两个模式均使用当前后退、左移、右移和转向四段动作，目录位于 `legged_lab/envs/g1/datasets/make_g1_23dof/training/`：

| AMP维度 | 关节特征 | 数据子目录 | 加载器 | 判别器输入（相邻状态拼接） |
| --- | --- | --- | --- | --- |
| 58（默认） | 23位置 + 23速度 + 12末端位置 | `amp_full/` | `G1AnkleAMPLoader` | 116 |
| 50 | 19位置 + 19速度 + 12末端位置 | `amp_no_ankle/` | `AMPLoader` | 100 |

50维仅移除四个踝关节的位置和速度特征，末端位置仍由完整机器人计算。策略输入1932维、critic输入1037维、23维动作和可视化完整姿态保持一致。切换格式后从头训练；续训须使用checkpoint对应的AMP格式，不自动迁移判别器或优化器。`play_depth.py` 根据选中checkpoint的AMP契约自动选择50/58维回放配置，无需改动训练默认值；加载时仍严格校验全部checkpoint契约。

**训练**

在项目根目录、原有Isaac环境中运行：

```bash
source /home/cjh/isaac-sim/setup_conda_env.sh
/home/cjh/miniconda3/envs/env_isaaclab/bin/python legged_lab/scripts/train.py \
  --task g1_depth --headless --device cuda:0
```

可用 `--num_envs 16 --max_iterations 3` 做短训练。日志写入 `logs/g1_depth/<时间戳>_vp_multihead69_beta_vae/`，包括 `params/agent.yaml`、`env.yaml`、`depth_observation_contract.yaml`、TensorBoard和checkpoint。

checkpoint的 `iter` 表示已完成更新数，3轮得到 `model_3.pt`。直接指定checkpoint可继续训练：

```bash
/home/cjh/miniconda3/envs/env_isaaclab/bin/python legged_lab/scripts/train.py \
  --task g1_depth --headless --device cuda:0 \
  --checkpoint_path /absolute/path/to/model_3.pt --max_iterations 1
```

这将从iteration3再完成一次更新并保存 `model_4.pt`。恢复模型、Adam、AMP归一化统计和训练iteration，环境重新初始化。模型身份、观测、网络配置、AMP和辅助监督契约必须一致；不接受RENet checkpoint或自动迁移。

headless训练完成后先flush日志，再使用Kit的 `skip_cleanup` 退出，避免本机重现场景关闭时的长时间清理等待；此行为只用于新任务的headless完成路径。

**回放与导出**

```bash
/home/cjh/miniconda3/envs/env_isaaclab/bin/python legged_lab/scripts/play_depth.py \
  --checkpoint_path /absolute/path/to/model_4.pt --terrain flat \
  --vx 0.5 --vy 0 --wz 0
```

默认单环境，关闭本体/深度噪声和推力，禁用地形课程，使用固定命令；可用 `--depth_noise` 开启原深度噪声，`--no_gamepad`禁用手柄。手柄左摇杆控制vx/vy、右摇杆控制wz，A/Start重置。

回放默认关闭足缘奖励几何与足底支撑诊断；它们只参与奖励和日志，不参与策略输入。否则，即使 `--num_envs 1`，也会在完整的200块训练地形上提取、合并边缘，终端可能长时间停在 `Generating terrains randomly took ...`。需要检查该奖励时添加 `--foot_edge_diagnostics`，保留训练配置中的足缘开关与诊断设置，并预留边缘构建时间。此回放设置不改变训练配置、物理地形或模型接口。

默认同时导出；`--no_export`跳过导出，`--export_only`仅导出，`--export_dir`指定目录。默认目录为checkpoint所在run的 `exported/`。headless短回放可用 `--headless --max_steps 100 --no_export --no_gamepad`。

产物为完整深度编码和Actor的 `policy.pt`、`policy.onnx`、`depth_contract.json`及 `validation.json`。TorchScript支持 `[N,1932]→[N,23]`；ONNX固定 `[1,1932]→[1,23]`，opset17。导出过程重载TorchScript检查batch1/7，重载ONNX用CPU ReferenceEvaluator验证动作，要求rtol=1e-4、atol=1e-5。深度契约内嵌于两种模型，不使用RENet契约文件名。

模型与checkpoint属于新任务；现有2120维真机/selector接口不能直接加载这些产物。当前支持Isaac训练、回放、导出，以及下述独立MuJoCo sim2sim；真机部署适配需另行实现。

**MuJoCo sim2sim**

入口为 [sim2sim_depth.py](/home/cjh/TienKung-Lab/legged_lab/scripts/sim2sim_depth.py)。使用已包含完整VP、Actor和归一化器的 `exported/policy.pt`，读取模型内嵌的 `depth_contract.json`；不直接读取训练checkpoint，也不重复做观测归一化。AMP的50/58维选择不改变该推理接口。

本机 `env_tienkung` 已具备MuJoCo 3.3.2、PyTorch及窗口所需依赖。先用上述 `play_depth.py` 导出模型，再运行：

```bash
# 窗口与深度预览；有手柄时启用手柄，无手柄时使用固定命令。
/home/cjh/miniconda3/envs/env_tienkung/bin/python legged_lab/scripts/sim2sim_depth.py \
  --policy /absolute/path/to/exported/policy.pt --scene mix

# 无窗口、固定命令，保存数值摘要。
/home/cjh/miniconda3/envs/env_tienkung/bin/python legged_lab/scripts/sim2sim_depth.py \
  --policy /absolute/path/to/exported/policy.pt --scene flat \
  --headless --no-joystick --vx 0.5 --vy 0 --wz 0 --duration 20 \
  --report /tmp/g1_depth_sim2sim.json

# 无需策略模型即可列出场景、核对默认关节映射。
/home/cjh/miniconda3/envs/env_tienkung/bin/python legged_lab/scripts/sim2sim_depth.py --list-scenes
/home/cjh/miniconda3/envs/env_tienkung/bin/python legged_lab/scripts/sim2sim_depth.py --scene gaps --check-mapping
```

默认平地、100秒，固定命令 `vx=0.5、vy=0、wz=0`。省略 `--policy` 时查找 `logs/g1_depth/*/exported/policy.pt` 中修改时间最新的模型；没有模型或契约不兼容时明确报错。场景支持 `flat/mix/slope/gaps`；`--scene-file` 指定自定义XML，优先于 `--scene`，两者均未指定时使用 `--model`。脚本在临时副本中解析include和资源路径，仅移除自定义地形场景中机器人XML附带的平地，避免填平沟壑和重复floor；源XML保持原值。

物理200Hz、控制50Hz、深度10Hz。按模型契约中的关节名称映射观测和动作；PD增益、默认姿态、力矩上限及armature独立对应当前 `G1_23CFG`。每个物理步重新计算PD力矩并限幅，上一动作记录的是已裁剪、尚未乘action_scale的策略动作。reset后首张本体观测重复填满10帧，首张深度重复填满两帧。运行摘要记录跟踪误差、机体高度、动作/力矩限幅和实时率。

相机使用原生 `mj_multiRay`，不依赖XML中的渲染相机或OpenGL。像素中心射线和外参来自导出契约，保留训练指定的腿部/肘部视觉网格自回波。当前GroupedRayCasterCamera实际按torso完整姿态变换相机，即使配置字段 `camera_ray_alignment` 为 `yaw`，仍包含roll/pitch；sim2sim与该实现保持一致。射线距离转为光轴深度，裁剪至18×32后按训练范围限幅、归一化。两种仿真器的碰撞求解和地形表示仍有差异，跨仿真步态需实际回放评估。

默认关闭本体噪声、深度噪声与模糊。`--proprio-noise` 开启当前训练同款本体均匀噪声；`--depth-noise` 开启50%整帧概率的3×3、sigma=1模糊及距离相关高斯噪声（near_std=0、far_std=0.10米、exponent=2），先模糊、再按米加噪、最后限幅归一化。`--seed` 控制独立随机流，已存入历史的帧不会在读取时重复加噪。

手柄左摇杆控制前后/侧移，右摇杆水平轴控制转向，死区0.1；命令限制为vx `[-0.6,1.0]`、vy `[-0.5,0.5]`、wz `[-1.57,1.57]`。按住A停止，释放Start重置，Select退出。键盘R重置并恢复输入，空格停止，Esc退出；深度预览窗也接收这些按键。`--no-joystick` 使用固定命令，`--no-depth-view` 关闭深度预览，`--debug-depth` 输出深度统计。`--headless` 不初始化手柄或任何窗口。

**验证**

```bash
PYTHONPATH="$PWD/rsl_rl" /home/cjh/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  -q rsl_rl/tests/test_g1_depth.py rsl_rl/tests/test_g1_depth_amp_switch.py
```

数值提取一致性测试只在测试实例中对齐VP权重和Actor有效列；生产代码不提供RENet权重迁移。实际验收、回归基线比较和产物路径见 [实现报告](/home/cjh/TienKung-Lab/work/g1_depth_implementation_20261014/REPORT.md)。

MuJoCo接口、几何、噪声对齐及控制测试使用MuJoCo环境：

```bash
PYTHONPATH="$PWD/rsl_rl:$PWD" /home/cjh/miniconda3/envs/env_tienkung/bin/python -m pytest \
  -q rsl_rl/tests/test_sim2sim_depth.py
```
