# G1 感知选择器：跨领域文献路线图

检索日期：2026-09-08。问题是视觉/激光雷达互为备份，以及双失效后的停止站立接管。本清单按方法与任务的关联度选择 15 篇文献，覆盖工业过程诊断、统计监测、视觉惯性里程计、多模态分类、工业视觉检测、服务器/航天遥测、选择性预测和运行时控制保证；不是系统综述或最新论文排行榜。

书目信息及方法描述来自出版社、会议官网、作者公开论文或 arXiv。会议稿与预印本按同一工作合并。学术 MCP 当前不可用；本地 CrossRef/PubMed/arXiv API 预检出现 SSL 连接失败，OpenAlex 发现检索成功，随后通过网页检索与一手来源核对。部分 OpenReview 页面触发浏览器验证，使用公开索引的会议论文内容；Page 论文核对了出版社书目，未取得完整正文。以下迁移建议是针对本项目的设计推论，不是原论文已经在 G1 上验证的结论。

## 先区分四个问题

| 问题 | 要输出什么 | 相关方法 |
|---|---|---|
| 数据链路是否正常？ | 超时、失效、过期、未初始化标记 | 硬规则、故障检测与隔离 |
| 当前输入是否异常或持续退化？ | 异常分数、变化告警 | CUSUM、Outlier Exposure、PatchCore、OmniAnomaly |
| 当前分支是否足以完成动作？ | 感知误差分布、分支任务失败风险 | 内省感知、QMF、深度集成 |
| 两路都不可靠怎么办？ | 拒绝运动、进入 SAFE、恢复条件 | SelectiveNet、校准、风险控制、Simplex |

检测异常、估计预测不确定性、选择信息源和证明控制安全，是不同问题。注意力权重、归一化融合权重或 softmax 最大值都不能未经校准就当作故障概率。两路权重之和为 1 不能表达两路同时缺乏可信信息。

## A. 健康诊断与规则切换的基础

### 1. Isermann：模型与残差诊断

Rolf Isermann. **Model-based fault-detection and diagnosis – status and applications.** *Annual Reviews in Control* 29(1), 71–85, 2005. DOI：[10.1016/j.arcontrol.2004.12.002](https://doi.org/10.1016/j.arcontrol.2004.12.002)。[出版社摘要](https://www.sciencedirect.com/science/article/pii/S1367578805000052)。类型：领域综述。

论文梳理参数估计、观测器与奇偶关系等诊断方法。核心价值是利用过程模型构建残差，再进行故障识别。对 G1，可在时间/坐标对齐后检查运动补偿的地图一致性、观测与独立本体估计的一致性。两个估计不同并不能单独定位故障源，仍需冗余证据与模型适用性判断。

### 2. Page：持续变化检测

E. S. Page. **Continuous Inspection Schemes.** *Biometrika* 41(1–2), 100–115, 1954. DOI：[10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)。类型：统计方法论文。

CUSUM 是持续偏移检测的经典来源。适合把有效率下降、残差增长等连续指标转成变化告警。工程上应区分持续退化和明确断流：前者可以积累证据，后者不应等统计窗口结束才处理。序列自相关、工作状态变化和传感器更新频率都需要在阈值标定时考虑。

### 3. Phan 等：后备控制接管与返回

Dung T. Phan, Radu Grosu, Nils Jansen, Nicola Paoletti, Scott A. Smolka, Scott D. Stoller. **Neural Simplex Architecture.** *NASA Formal Methods Symposium (NFM)*, 2020. [作者论文及会议说明](https://arxiv.org/abs/1908.00528)。

采用先进控制器、后备控制器与决策模块，并研究接管后返回先进控制器。可将 LP/VP 控制组合映射为先进控制器、停止站立策略映射为后备控制器。借鉴的是结构与切换条件设计；安全保证还依赖后备控制能力、模型和可恢复状态集合，不能仅因增加一个 learned safe policy 就宣称有形式化保证。

## B. 更强的质量判断与任务可靠性

### 4. Chen 等：学习选择哪些传感器特征

Changhao Chen et al. **Selective Sensor Fusion for Neural Visual-Inertial Odometry.** *CVPR*, 10542–10551, 2019. [会议页面、PDF、补充材料](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html)。

提出确定性软选择与随机硬选择，处理缺失、损坏和不同步的图像/惯性数据。适合借鉴 LP/VP 特征门控结构和故障实验设计。原任务是里程计，门控值不是绝对可用性概率；本项目需要独立的双路拒绝条件。现有二值 mask 策略若改成连续 latent 融合，需要重新训练并检验分布变化。

### 5. Han 等：证据与未知程度

Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou. **Trusted Multi-View Classification.** *ICLR*, 2021. [会议论文](https://openreview.net/pdf?id=OOsR8BzCnl5)。[作者代码](https://github.com/hanmenghan/TMC)。

通过 Dirichlet 分布表达各视图的类别证据，并利用 Dempster–Shafer 理论融合。可借鉴“证据不足”的显式表达。原方法针对分类，不能直接套在连续关节动作上；证据大小及冲突处理仍需在共同退化、分布外样本上验证，不能假定其天然校准。

### 6. Zhang 等：QMF，质量驱动动态融合

Qingyang Zhang et al. **Provable Dynamic Fusion for Low-Quality Multimodal Data.** *ICML*, PMLR 202:41753–41769, 2023. [会议论文与 PDF](https://proceedings.mlr.press/v202/zhang23ar.html)。

通过不确定性构造质量感知融合，分析权重与单模态误差的关系。对本项目最有价值的是：质量分数需要反映分支预测是否容易出错。原方法的分类能量分数及泛化理论不能直接解释为 G1 跌倒概率或闭环安全保证；可以借鉴质量学习目标，并保留本项目的硬切换。

### 7. Rabiee 与 Biswas：预测感知算法自己的误差

Sadegh Rabiee, Joydeep Biswas. **Introspective Perception for Mobile Robots.** *Artificial Intelligence* 324, 103999, 2023. DOI：[10.1016/j.artint.2023.103999](https://doi.org/10.1016/j.artint.2023.103999)。[作者全文](https://www.joydeepb.com/Publications/aij2023introspective.pdf)。

利用跨传感器与时空一致性生成监督，学习感知算法的误差分布，并在双目深度和视觉 SLAM 上验证。适合为 LP/VP 分别学习地形估计误差或误差超限概率。冗余监督也有可靠性前提：雷达不应始终被当作视觉真值，两路共同错误需要额外检测。

### 8. Rabiee 等：把感知错误与任务失败连接起来

Sadegh Rabiee, Connor Basich, Kyle Hollins Wray, Shlomo Zilberstein, Joydeep Biswas. **Competence-Aware Path Planning Via Introspective Perception.** *IEEE Robotics and Automation Letters* 7(2), 3218–3225, 2022. DOI：[10.1109/LRA.2022.3145517](https://doi.org/10.1109/LRA.2022.3145517)。[作者全文](https://www.joydeepb.com/Publications/ral2022_cpip.pdf)。

分别建模感知误差与上下文相关的任务失败，并用于路径规划。它启发本项目根据命令、运动状态和分支信息预测短期失败风险：相同感知误差在平地慢走和台阶边缘可能有不同后果。原论文验证的是导航规划；用于人形机器人模式选择需要重新定义失败事件、时间范围和监督数据。

### 9. Lakshminarayanan 等：Deep Ensembles

Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.** *NIPS/NeurIPS*, 2017. [会议论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)。

使用独立训练的模型估计预测分布。可将其用于多个小型分支风险预测器，观察平均风险和预测分歧，作为质量网络的不确定性基线。共享编码器多头是较便宜的变体，不能直接当作原始独立集成；集成也可能共同自信地出错。

## C. 从其他行业借鉴异常检测

### 10. Hendrycks 等：Outlier Exposure

Dan Hendrycks, Mantas Mazeika, Thomas Dietterich. **Deep Anomaly Detection with Outlier Exposure.** *ICLR*, 2019. [会议论文](https://openreview.net/pdf?id=HyxCxhRcY7)。

训练时提供辅助异常样本，改善对未见异常的识别。适合借鉴其训练思想，加入图像冻结、局部遮挡、地图漂移与共同退化。原方法不是专为传感器合成噪声设计；迁移时要保留未参与训练的整类故障评估泛化，而不只是换随机种子。

### 11. Roth 等：PatchCore

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. **Towards Total Recall in Industrial Anomaly Detection.** *CVPR*, 14318–14328, 2022. [会议论文与代码入口](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)。

通过正常局部特征库检测并定位异常。适合借鉴局部异常图，避免全图平均分数掩盖前方落脚区域的损坏。工业图像及 ImageNet 特征与深度/高程图不同，需要适配特征；正常的新地形可能产生高异常分数，不能直接等同于传感器损坏。

### 12. Su 等：OmniAnomaly

Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei. **Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network.** *KDD*, 2828–2837, 2019. DOI：[10.1145/3292500.3330672](https://doi.org/10.1145/3292500.3330672)。[会议摘要](https://www.kdd.org/kdd2019/accepted-papers/view/robust-anomaly-detection-for-multivariate-time-series-through-stochastic-re)。

针对服务器及航天遥测等多变量时间序列，学习正常时间关系并给出异常判断。可把包龄、覆盖率、地图残差和身体运动量作为时间序列输入，用于缓慢恶化与间歇异常。部署必须采用仅含当前和过去数据的窗口，评估实际检测延迟；工作状态变化需要条件建模。

## D. 拒绝决策、概率校准与风险阈值

### 13. Geifman 与 El-Yaniv：SelectiveNet

Yonatan Geifman, Ran El-Yaniv. **SelectiveNet: A Deep Neural Network with an Integrated Reject Option.** *ICML*, PMLR 97:2151–2159, 2019. [会议论文、PDF、代码](https://proceedings.mlr.press/v97/geifman19a.html)。

联合学习预测与拒绝，以风险—覆盖率权衡衡量模型。可借鉴“允许拒绝所有运动分支”的训练目标，防止强制二选一。拒绝预测并不自动产生安全关节动作，SAFE 停止站立能力仍需单独训练和验证。

### 14. Guo 等：置信度校准

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. **On Calibration of Modern Neural Networks.** *ICML*, PMLR 70:1321–1330, 2017. [会议论文](https://proceedings.mlr.press/v70/guo17a.html)。

研究现代网络概率失准，并检验温度缩放等后处理方法。适合校准两路质量/失败概率，使阈值具有可检验的经验含义。需要与训练数据分离的校准集；在原分布校准成功不代表在新地形或共同故障下也可靠。

### 15. Angelopoulos 等：Conformal Risk Control

Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. **Conformal Risk Control.** *ICLR*, 2024. [正式会议稿](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)。[预印本与修订记录](https://arxiv.org/abs/2208.02814)。

在相应假设下，使用校准数据控制有界、随保守参数单调变化的损失的期望。可用于研究“放行已知不适用分支”的阈值控制。连续机器人轨迹存在时间依赖与策略引起的分布改变；整段跌倒损失也未必随切换阈值单调。因此不能直接将每帧当作独立样本并声称得到全程跌倒概率保证。

## 对 G1 的组合建议：这是设计推论

第一阶段保持原有策略固定：每路健康指标及独立一致性残差 → 持续退化监测 → 带迟滞的 LP/VP/SAFE 状态机。明确断流直接否决该分支，不等待 CUSUM 累积。阈值按 MID-360/PGTT 输入契约、运动状态与采集频率标定。

第二阶段借鉴 QMF、内省感知和 CPIP，给每个分支增加任务风险预测器。一个候选定义是：

\[
R_m(t)=P(\text{未来 }H\text{ 内发生预定义控制失败}\mid\text{当前可用历史、命令，并执行分支 }m),\quad m\in\{VP,LP\}.
\]

`H` 需要覆盖检测、切换与停止响应的相关时间，且明确标签是在该时段固定分支还是执行某个已定义的切换方案下生成。失败可定义为跌倒、非足部碰撞、姿态越界或任务约束违反；不同事件的严重度和可预测性应分开评估。

候选训练数据生成方法：从同一仿真状态及完整历史启动 LP 和 VP 的分支 rollout，分别记录真实结局，从而获得两路标签。只从实际被选分支收集数据会产生选择偏差。仿真真值用于标签，不进入部署质量网络；测试应包含未训练过的故障类型和组合。

保留两路独立风险与不确定性，硬故障否决优先。仅在可用分支中选择；两路都不可用或风险不可接受时调用 SAFE。可以在这个结构中逐项比较规则、普通二头网络、集成、异常暴露、校准等方法，无需一次把所有模块叠加。

## 优先阅读与实验顺序

优先六篇：**Page（持续退化）→ Chen（选择结构）→ QMF（质量目标）→ CPIP（任务风险）→ SelectiveNet（拒绝）→ Neural Simplex（接管与恢复）**。随后用内省感知补充质量监督来源，用 Guo 和 CRC 研究阈值。

评估至少报告：故障漏检、误切换、检测延迟、恢复抖动、危险分支放行、停止位移、闭环跌倒率与任务完成率。轨迹级划分训练/校准/测试集，另保留未知故障、双路共同退化和正常新地形。不要用普通分类准确率或重建误差代替上述闭环指标。

本次仅进行文献检索、核对和方案关联分析，没有实现或评测上述网络。
