# 原始 G1 PKL 的 58 维 AMP 数据

仅由 `pkl/` 中原始起步、掉头、左转、右转四个 PKL 转换，保留全部源帧和帧率，不使用 `_v1` 版本。

特征：23 维关节角 + 23 维关节速度 + 12 维 pelvis 坐标系手/脚踝 link 位置。
58 维 AMP 与 58 维可视化文件的字段含义不同；可视化文件位于 `../motion_visualization/original_pkl/`。

| 片段 | 帧数 | FPS |
|---|---:|---:|
| `stand_to_walk` | 517 | 39.974277 |
| `walk_turn_around` | 454 | 39.941478 |
| `walk_turn_left` | 336 | 40.000000 |
| `walk_turn_right` | 371 | 40.000000 |

保留各段 MotionWeight=0.5，四段归一化采样概率各 25%；LoopMode=Clamp，不将片段首尾拼接。
训练加载器按源帧率插值为相隔 20 ms 的状态对。来源、文件哈希、关节顺序和模型记录在 `manifest.json`。
旧的其他 TXT 可能仍在此目录，g1_rough 仅使用 manifest 中明确列出的这四段。
