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

50维仅移除四个踝关节的位置和速度特征，末端位置仍由完整机器人计算。策略输入1932维、critic输入1037维、23维动作和可视化完整姿态保持一致。切换格式后从头训练；续训与回放须使用checkpoint对应的AMP格式，不自动迁移判别器或优化器。

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

默认同时导出；`--no_export`跳过导出，`--export_only`仅导出，`--export_dir`指定目录。默认目录为checkpoint所在run的 `exported/`。headless短回放可用 `--headless --max_steps 100 --no_export --no_gamepad`。

产物为完整深度编码和Actor的 `policy.pt`、`policy.onnx`、`depth_contract.json`及 `validation.json`。TorchScript支持 `[N,1932]→[N,23]`；ONNX固定 `[1,1932]→[1,23]`，opset17。导出过程重载TorchScript检查batch1/7，重载ONNX用CPU ReferenceEvaluator验证动作，要求rtol=1e-4、atol=1e-5。深度契约内嵌于两种模型，不使用RENet契约文件名。

模型与checkpoint属于新任务；现有2120维真机/selector接口不能直接加载这些产物。此交付包含Isaac训练、回放和导出，部署适配需另行实现。

**验证**

```bash
PYTHONPATH="$PWD/rsl_rl" /home/cjh/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  -q rsl_rl/tests/test_g1_depth.py rsl_rl/tests/test_g1_depth_amp_switch.py
```

数值提取一致性测试只在测试实例中对齐VP权重和Actor有效列；生产代码不提供RENet权重迁移。实际验收、回归基线比较和产物路径见 [实现报告](/home/cjh/TienKung-Lab/work/g1_depth_implementation_20261014/REPORT.md)。
