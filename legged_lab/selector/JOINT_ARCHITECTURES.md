# 联合 selector 结构对照

训练入口默认使用 `branch_gru`，沿用已有五帧数据、未来1秒标签，以及原来的加权四分类CE＋两路边缘BCE。没有修改底层策略、数据筛选或切换规则。

| `SELECTOR_ARCHITECTURE` | 结构 | 默认GRU层数 | 默认参数量 |
|---|---|---:|---:|
| `single_gru` | 原统一编码＋GRU末状态，A基线 | 3 | 412100 |
| `single_gru_readout` | 原编码＋原GRU，末状态＋全帧注意池化＋当前帧直连，B读出对照 | 3 | 436804 |
| `branch_gru` | 公共编码＋LP/VP独立编码与时序＋联合四分类头，C主方案 | 每路1层 | 615940 |

这些是待训练比较的结构，参数量和模块检查不证明性能提升。

## C结构

每帧的原始列位置保持不变，先使用训练分区拟合的逐列标准化，然后按字段切分：

- 公共87维：本体`0:78`＋跨路`274:281`＋执行状态`281:283`。
- LP98维：latent`78:142`＋速度`206:209`＋候选动作`212:235`＋健康`258:266`。
- VP98维：latent`142:206`＋速度`209:212`＋候选动作`235:258`＋健康`266:274`。

```text
公共87 → MLP(87→128→64)，逐层LayerNorm＋ELU
                       ├→ 与LP98拼接 → LP MLP(162→256→128) → LP GRU(128→128)
                       └→ 与VP98拼接 → VP MLP(162→256→128) → VP GRU(128→128)

LP读出 = 末状态128＋全帧注意池化128＋当前帧编码128 = 384
VP读出 = 末状态128＋全帧注意池化128＋当前帧编码128 = 384

LP384＋VP384＋当前公共编码64 = 832
→ 融合MLP(832→256→64→4)
→ 四类logits
```

两路编码器、GRU、注意池化各有独立参数；公共编码器的同一输出供两路使用。全部模块由同一四分类CE和边缘BCE一起训练，没有独立风险概率头。仍有`pLP=q1+q3`、`pVP=q2+q3`。

单层GRU没有层间dropout，`SELECTOR_GRU_DROPOUT`在GRU层数大于1时生效。

注意池化仅对输入窗口中已有的时刻加权，不使用未来帧。当前帧直连连接的是MLP编码。五帧10Hz的显式跨度仍为0.4秒；读取10/20帧必须提供真实存储的对应历史，不能补零或复制帧扩充。

## 启动

在包含训练数据的服务器上，激活原有PyTorch环境后：

```bash
cd ~/TienKung-Lab
export SELECTOR_PYTHON="$(command -v python)"
export SELECTOR_GPU=1
export SELECTOR_ARCHITECTURE=branch_gru
export SELECTOR_GRU_LAYERS=auto
export SELECTOR_PROBABILITY_MODE=raw
bash legged_lab/selector/run_selector_joint_training.sh
```

默认仍使用现有清洗基础集和三个20k补采库，seeds为42、43、44。输出目录自动包含结构名及时间戳。A/B对照只修改`SELECTOR_ARCHITECTURE`，并保持数据、种子、概率模式、预算相同。

可继续设置`SELECTOR_CLEAN_DATASET`、`SELECTOR_ORIGINAL_DATASET`、`SELECTOR_SUPPLEMENTS_FILE`、`SELECTOR_JOINT_OUTPUT`、`SELECTOR_BATCH_SIZE`、`SELECTOR_NUM_WORKERS`、`SELECTOR_MAX_EPOCHS`、`SELECTOR_PATIENCE`。新增`SELECTOR_SEEDS`、`SELECTOR_GRU_LAYERS`、`SELECTOR_GRU_DROPOUT`。每次使用新输出目录，缓存会占用额外磁盘空间。

直接CLI支持`--architecture`、`--decision-probability-mode`及`--prediction-splits`。主概率模式在运行配置中预先指定，默认raw；另两种模式`temperature`和`temperature_bias`只在validation拟合并作为对照，不能用test选择。

## 输出与验收

- `joint_selector_checkpoint.pt`：V7，保存结构配置、权重、训练统计、概率模式和校准对照参数。
- `selection_freeze.json`：测试前冻结的选择及主概率模式。
- `joint_training_report.json`：包括所有模块更新审计、参数量、三种概率对照、训练/验证/测试的分失败时间指标。
- `train_predictions.npz/.json`、`validation_predictions.npz/.json`：默认保存完整逐条logits、三组四分类概率、标签、当前分支、错误mask、ID及首次失败步数。
- 各结果的`failure_timing.rows`：按`≤0.2`、`>0.2～0.5`、`>0.5～0.7`、`>0.7～1.0`秒分别报告分类召回、选安全支路率、需救援时救援率；双失败使用两路较晚的首次失败时间。

首次失败时间单独从源分片读取，用于离线审计和输出，不进入网络，也不加入训练随机访问缓存。当前验收重点为远期VP救援与双失败识别，同时检查双安全误切、有害切换及原有约束；单点预测不能替代闭环验证。

本地之前的训练入口为V2，而已分析服务器报告为V3。此次入口显式升级为V4实现标记，默认raw，并保存三种概率对照，避免将结构变化与隐式切换概率模式混在一起。

## 兼容性

`joint_selector_from_artifact`继续加载既有V6 `single_encoder_joint_outcomes`，保留原模型参数名、形状和初始化顺序。新结构写V7 `temporal_joint_outcomes`，旧加载器不能加载新结构，需同步本次模型、训练、评估等修改文件。既有数据格式、标签契约和原V5部署运行时保持原样。

本次没有运行完整训练，也未自动部署。当前机器没有可用CUDA，且本地缺少服务器上的清洗基础集；CPU回归和小型完整训练流程用于验证实现与接口。

## 验证记录

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest rsl_rl/tests/test_selector_joint_architectures.py -q
```

新增24项回归通过，覆盖列切分、三种结构和5/10/20帧、全模块梯度及更新、V6预测一致性、V7保存加载、三种主概率模式的短训练/评估/验证导出，以及启动脚本参数。禁用pytest自动插件加载用于避开当前环境中无关ROS插件缺少`lark`的问题。

还用真实训练分片的256行做了单步CPU更新，所有新模块梯度有限且参数更新；完整204750行validation的12格时间表也由新增诊断代码复现。记录见`outputs/selector_review_20261017/architecture_implementation_verification.json`。

扩展旧标签/定向补采测试结果为59通过、18失败；切回改动前模型和训练辅助模块复测，仍是同样18项失败。旧标签测试使用已不存在的`short_horizon_steps`/`long_horizon_steps`接口，另有旧collector替身缺少`query_health`。本次没有修改标签或采集器，也没有为让旧测试通过而改变标签含义。
