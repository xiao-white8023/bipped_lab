from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass

from dataclasses import MISSING

from legged_lab.utils.height_cammand.height_cammand_cfg import UniformHeightCommandCfg


class UniformHeightCommand(CommandTerm):
    cfg: UniformHeightCommandCfg

    def __init__(self,cfg:UniformHeightCommandCfg,env:ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.ranges.height is None:
            raise ValueError("高度范围是空的")
        self.robot: Articulation = env.scene[cfg.asset_name]
        # 初始化每一个环境的当前的命令高度
        self.height_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        # 初始化每一个环境的是否是站立模式(不受当前高度范围的控制)
        self.is_stand_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 更新每一个环境的高度误差
        self.metrics["error_height_z"] = torch.zeros(self.num_envs, device=self.device)

    @property  # 表示只读属性
    def command(self) -> torch.Tensor:
        """目标高度范围 Shape is (num_envs, 1)."""
        return self.height_command_b
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_height_z"] += (
            torch.abs(self.height_command_b[:, 0] - self.robot.data.root_pos_w[:, 2]) / max_command_step
        )
    
    def _resample_command(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.empty(len(env_ids), device=self.device) # 创建一个空的张量
        
        self.height_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        
        # 确定哪些环境需要站立 r.uniform_(0.0, 1.0) 生0-1之间的随机数
        # 决定哪些环境使用站立高度
        standing_mask = torch.rand(len(env_ids), device=self.device) <= self.cfg.rel_standing_envs
        self.is_stand_env[env_ids] = standing_mask

        # 3. 对站立环境，把目标高度改成站立高度
        standing_env_ids = env_ids[standing_mask]
        self.height_command_b[standing_env_ids, 0] = self.cfg.ranges.height[1]

    def _update_command(self):
        pass