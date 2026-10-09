# G1 23DoF PKL 动作

2026-10-04，将最新修改版从 `refined/` 复制到当前目录，文件内容经 SHA-256 校验一致。最新修改版为：

- `stand_to_walk_leg_v1.pkl`
- `walk_turn_around_leg_v1.pkl`
- `walk_turn_left_leg_v1.pkl`
- `walk_turn_right_leg_v1.pkl`

这四份保留并排落地的站立结尾，并平滑缩小左腿内撇幅度；脚踝角度仍全程为零，未重新进行 IK 求解。对应 JSON 包含修改参数和检查结果。

不含 `_leg_v1` 后缀的 PKL 为原始文件，保持不变。修改细节见 [refined/README.md](refined/README.md)。

这四份修改版已转换为 [50 维 AMP 训练数据](../g1_23dof_pkl_locomotion_v1/README.md)，并接入 `g1_renet`；训练直接加载新目录的 `amp/*.txt`，而非 PKL。

2026-10-08，按用户指定另外转换四份原始 `stand_to_walk.pkl`、`walk_turn_around.pkl`、`walk_turn_left.pkl`、`walk_turn_right.pkl`：

- 含脚踝的 58 维 AMP：`../motion_amp_expert/`。
- 不含脚踝的 50 维 AMP：`../motion_amp_expert_no_ankle/`；`g1_rough` 默认仅使用这四份。
- 对应的完整姿态回放：`../motion_visualization/original_pkl/`，字段布局与 58 维 AMP 不同。

原始帧数分别为 517、454、336、371，原始帧率和姿态保持不变；未加入站立结尾或使用 `_v1` 修改版。两份其他 PKL 未转换。转换脚本为 `legged_lab/scripts/convert_g1_original_pkl_to_amp.py`，覆盖旧 TXT 前须指定 `--backup-dir`。

同时预览四份最新修改版：

```bash
cd /home/cjh/TienKung-Lab
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/play_g1_pkl_mujoco.py --motion \
  legged_lab/envs/g1/datasets/pkl/stand_to_walk_leg_v1.pkl \
  legged_lab/envs/g1/datasets/pkl/walk_turn_around_leg_v1.pkl \
  legged_lab/envs/g1/datasets/pkl/walk_turn_left_leg_v1.pkl \
  legged_lab/envs/g1/datasets/pkl/walk_turn_right_leg_v1.pkl
```

R 重播，S 半速，空格暂停。此预览是 MuJoCo 姿态回放。
