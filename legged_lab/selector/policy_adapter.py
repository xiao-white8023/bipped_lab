# 中文导读
# 文件用途：为冻结 RENet 提供只读双分支接口，同一份观测可得到 LP 和 VP 的候选动作。
# 使用位置：frozen_policy 加载器、collector 数据采集，以及双分支 TorchScript 导出。
# 说明：两个分支共用原 actor，分别保留一侧特征并屏蔽另一侧；策略参数禁止梯度更新。
# 说明：adapter_from_runner 只是可选适配工具；独立 selector 训练脚本不需要训练 runner。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - FrozenRENetAdapter：封装冻结策略和观测归一化器，统一六个双分支输出。
# - FrozenRENetAdapter.__init__：检查 RENet 所需接口和维度，可选择复制策略，并冻结全部参数。
# - FrozenRENetAdapter.normalize：使用冻结的归一化器处理策略观测。
# - FrozenRENetAdapter.forward：计算 LP/VP 特征、速度估计，以及各自送入共享 actor 后的候选动作。
# - FrozenRENetAdapter.evaluate：在无梯度模式下调用双分支前向计算，并包装为 DualBranchOutput。
# - FrozenRENetAdapter.parity_error：比较适配器动作与原策略在强制 LP/VP mask 下的推理动作。
# - adapter_from_runner：从已有 runner 中提取策略和观测归一化器，构建冻结适配器。
# - validate_checkpoint_metadata：检查 checkpoint 参数键、张量形状及 LP 编码器约定是否匹配模型。
# 中文导读结束

"""Read-only dual-branch view of a frozen ``RENetActorCritic`` instance."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .contracts import DualBranchOutput


class FrozenRENetAdapter(nn.Module):
    """Evaluate both binary masks without changing the original policy class."""

    def __init__(self, policy: nn.Module, normalizer: nn.Module | None = None, *, copy_policy: bool = True):
        super().__init__()
        required = (
            "_split_actor_obs",
            "_embed_proprio_history",
            "_encode_lp_latent",
            "_encode_vp_latent",
            "actor",
            "lp_vel_estimator",
            "vp_vel_estimator",
        )
        missing = [name for name in required if not hasattr(policy, name)]
        if missing:
            raise TypeError(f"policy is not a compatible RENetActorCritic; missing {missing}")
        self.policy = copy.deepcopy(policy) if copy_policy else policy
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()
        self.policy.eval()
        self.normalizer.eval()
        if int(getattr(self.policy, "single_proprio_dim", -1)) != 78:
            raise ValueError("selector contract requires RENet single_proprio_dim=78")
        if int(getattr(self.policy, "estimator_latent_dim", -1)) != 64:
            raise ValueError("selector contract requires RENet estimator_latent_dim=64")
        if int(getattr(self.policy, "vel_dim", 3)) != 3:
            raise ValueError("selector contract requires RENet velocity dimension 3")
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        return self.normalizer(observations)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalize(observations)
        proprio_history, lidar_flat, depth_flat, _, current_proprio = self.policy._split_actor_obs(normalized)
        proprio_embed = self.policy._embed_proprio_history(proprio_history)
        lp_latent = self.policy._encode_lp_latent(
            proprio_history, lidar_flat, proprio_embed=proprio_embed
        )
        lp_query = getattr(self.policy, "_last_lp_query", None)
        vp_latent = self.policy._encode_vp_latent(
            proprio_history, depth_flat, proprio_embed=proprio_embed
        )
        zeros_lp = torch.zeros_like(lp_latent)
        zeros_vp = torch.zeros_like(vp_latent)
        lp_actor_input = torch.cat((current_proprio, lp_latent, zeros_vp), dim=-1)
        vp_actor_input = torch.cat((current_proprio, zeros_lp, vp_latent), dim=-1)
        lp_action = self.policy.actor(lp_actor_input)
        vp_action = self.policy.actor(vp_actor_input)
        if getattr(self.policy, "use_vel_estimation", True):
            uses_ame = bool(getattr(self.policy, "uses_ame_lp", False))
            lp_velocity_input = lp_query if uses_ame and lp_query is not None else lp_latent
            lp_velocity = self.policy.lp_vel_estimator(lp_velocity_input)
            vp_velocity = self.policy.vp_vel_estimator(vp_latent)
        else:
            velocity_dim = int(getattr(self.policy, "vel_dim", 3))
            lp_velocity = torch.zeros(observations.shape[0], velocity_dim, device=observations.device)
            vp_velocity = torch.zeros_like(lp_velocity)
        return lp_latent, vp_latent, lp_velocity, vp_velocity, lp_action, vp_action

    @torch.no_grad()
    def evaluate(self, observations: torch.Tensor) -> DualBranchOutput:
        output = self(observations)
        return DualBranchOutput(
            lp_latent=output[0],
            vp_latent=output[1],
            lp_velocity=output[2],
            vp_velocity=output[3],
            lp_action=output[4],
            vp_action=output[5],
        )

    @torch.no_grad()
    def parity_error(self, observations: torch.Tensor) -> dict[str, float]:
        """Compare both adapter actions with the untouched binary-mask policy."""

        normalized = self.normalize(observations)
        depth_end = self.policy.proprio_actor_dim + self.policy.lidar_flat_dim + self.policy.depth_flat_dim
        branches = self.evaluate(observations)
        errors: dict[str, float] = {}
        for name, mask, expected in (
            ("vp", 0.0, branches.vp_action),
            ("lp", 1.0, branches.lp_action),
        ):
            forced = normalized.clone()
            forced[:, depth_end : depth_end + 1] = mask
            actual = self.policy.act_inference(forced)
            errors[name] = float((actual - expected).abs().max().item())
        return errors


def adapter_from_runner(runner: Any, *, copy_policy: bool = True) -> FrozenRENetAdapter:
    """Construct the adapter from an already loaded RENet runner."""

    if not hasattr(runner, "alg") or not hasattr(runner.alg, "policy"):
        raise TypeError("runner does not expose alg.policy")
    normalizer = getattr(runner, "obs_normalizer", None)
    return FrozenRENetAdapter(runner.alg.policy, normalizer, copy_policy=copy_policy)


def validate_checkpoint_metadata(checkpoint: dict, policy: nn.Module) -> None:
    """Fail early when a checkpoint and instantiated frozen policy disagree."""

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    policy_state = policy.state_dict()
    missing = sorted(set(policy_state) - set(state))
    unexpected = sorted(set(state) - set(policy_state))
    mismatched = sorted(
        key for key in set(policy_state) & set(state) if policy_state[key].shape != state[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "checkpoint/policy contract mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape={mismatched[:5]}"
        )
    encoder_contract = checkpoint.get("lp_encoder_contract")
    if encoder_contract is not None and hasattr(policy, "get_lp_encoder_contract"):
        current = policy.get_lp_encoder_contract()
        if current != encoder_contract:
            raise ValueError("checkpoint LP encoder contract does not match the instantiated policy")
