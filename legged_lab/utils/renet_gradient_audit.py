"""Compare LP/VP PPO gradients at one fixed RENet parameter state.

This module needs only PyTorch. It does not collect rollouts, update parameters,
or infer historical gradients from Adam moments. Batches must contain actual
on-policy observations, sampled actions, their saved log probabilities, and GAE
advantages. Computing gradients from fabricated observations is not evidence of
training interference.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
from torch import nn


def gradient_pair_metrics(lp: torch.Tensor, vp: torch.Tensor) -> dict[str, Any]:
    """Compare two loss gradients, including the first-order effect of descent.

    ``vp_loss_change_per_unit_lp_gradient_step`` equals ``-g_lp.dot(g_vp)``:
    it is the derivative of VP loss for a step ``theta -= eta * g_lp`` at
    eta=0, before Adam or gradient clipping. It is not a measured reward change.
    """
    lp = lp.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    vp = vp.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if lp.shape != vp.shape:
        raise ValueError("Gradient vectors must have matching shapes.")
    if not torch.isfinite(lp).all() or not torch.isfinite(vp).all():
        raise ValueError("Gradient vectors must be finite.")
    lp_norm = float(torch.linalg.vector_norm(lp))
    vp_norm = float(torch.linalg.vector_norm(vp))
    dot = float(torch.dot(lp, vp))
    cosine = max(-1.0, min(1.0, dot / (lp_norm * vp_norm))) if lp_norm and vp_norm else None
    opposing_dot = max(0.0, -dot)
    return {
        "num_parameters": lp.numel(),
        "lp_norm": lp_norm,
        "vp_norm": vp_norm,
        "dot": dot,
        "cosine": cosine,
        "negative": cosine < 0 if cosine is not None else None,
        "lp_conflicting_projection_norm": opposing_dot / vp_norm if vp_norm else None,
        "vp_conflicting_projection_norm": opposing_dot / lp_norm if lp_norm else None,
        "vp_loss_change_per_unit_lp_gradient_step": -dot,
        "lp_loss_change_per_unit_vp_gradient_step": -dot,
        "vp_loss_change_per_unit_lp_unit_step": -dot / lp_norm if lp_norm else None,
        "lp_loss_change_per_unit_vp_unit_step": -dot / vp_norm if vp_norm else None,
    }


def _prepare_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    required = ("observations", "actions", "old_log_probs", "advantages")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"PPO batch is missing fields: {missing}")
    prepared = {key: batch[key].detach().to(device) for key in required}
    observations, actions = prepared["observations"], prepared["actions"]
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Observations and actions must have shapes [N, features] and [N, actions].")
    count = observations.shape[0]
    if count == 0 or actions.shape[0] != count:
        raise ValueError("PPO batches must be nonempty and have matching sample counts.")
    for key in ("old_log_probs", "advantages"):
        value = prepared[key]
        if tuple(value.shape) not in ((count,), (count, 1)):
            raise ValueError(f"{key} must have shape [N] or [N, 1].")
        prepared[key] = value.reshape(count)
    for key, value in prepared.items():
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{key} must be a finite floating-point tensor.")
    return prepared


def _mode_gradients(policy, batch, parameters, clip_param, entropy_coef, normalize_advantages):
    advantages = batch["advantages"]
    raw_mean = float(advantages.mean())
    raw_std = float(advantages.std()) if advantages.numel() > 1 else None
    if normalize_advantages:
        if advantages.numel() < 2:
            raise ValueError("Sample-standard-deviation normalization requires at least two samples per mode.")
        # Match RolloutStorage.compute_returns: one normalization per rollout.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy.update_distribution(batch["observations"])
    log_probs = policy.get_actions_log_prob(batch["actions"]).reshape(-1)
    if log_probs.shape != batch["old_log_probs"].shape:
        raise ValueError("Policy log probabilities must contain one scalar per action sample.")
    entropy = policy.entropy.reshape(-1).mean()
    ratio = torch.exp(log_probs - batch["old_log_probs"])
    surrogate = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1.0 - clip_param, 1.0 + clip_param),
    ).mean()
    combined = surrogate - entropy_coef * entropy
    if not torch.isfinite(surrogate) or not torch.isfinite(combined):
        raise ValueError("PPO loss is nonfinite; check saved log probabilities and policy compatibility.")
    gradients = {}
    for key, loss in (("surrogate_plus_entropy", combined), ("surrogate_only", surrogate)):
        values = torch.autograd.grad(loss, parameters, retain_graph=key == "surrogate_plus_entropy", allow_unused=True)
        gradients[key] = [value.detach() if value is not None else torch.zeros_like(parameter)
                          for parameter, value in zip(parameters, values)]
    losses = {
        "samples": advantages.numel(),
        "surrogate": float(surrogate.detach()),
        "entropy": float(entropy.detach()),
        "surrogate_plus_entropy": float(combined.detach()),
        "raw_advantage_mean": raw_mean,
        "raw_advantage_sample_std": raw_std,
        "used_advantage_mean": float(advantages.mean()),
        "used_advantage_sample_std": float(advantages.std()) if advantages.numel() > 1 else None,
        "ratio_mean": float(ratio.detach().mean()),
        "ratio_min": float(ratio.detach().min()),
        "ratio_max": float(ratio.detach().max()),
        "clipped_ratio_fraction": float(((ratio < 1.0 - clip_param) | (ratio > 1.0 + clip_param)).float().mean()),
    }
    return gradients, losses


def analyze_paired_ppo_gradients(
    policy: nn.Module,
    lp_batch: Mapping[str, torch.Tensor],
    vp_batch: Mapping[str, torch.Tensor],
    clip_param: float = 0.2,
    entropy_coef: float = 0.005,
    normalize_advantages: bool = True,
) -> dict[str, Any]:
    """Measure paired actor gradients without an optimizer step.

    Batch fields: observations [N,2120] and actions [N,23] for the legacy G1
    policy; old_log_probs and advantages [N] or [N,1]. Toy policies can use other
    feature/action widths. LP observations must retain their LP mask and VP
    observations their VP mask. All inputs are detached internally. Each mode
    is normalized independently using sample standard deviation when enabled.
    For direction subsets already normalized over their full source rollout,
    pass ``normalize_advantages=False`` to preserve the original PPO advantages.

    Shared actor parameters exclude the LP/VP-specific columns of its first
    Linear weight. The common proprioception columns, first-layer bias, and all
    subsequent actor parameters are included. Encoder, critic and exploration
    std parameters are excluded from actor blocks; std is reported separately.
    Parameter values, buffers, .grad, requires_grad and module training flags
    are preserved. Distribution/feature caches may be refreshed by the forward.
    """
    if not math.isfinite(clip_param) or not 0 <= clip_param < 1:
        raise ValueError("clip_param must be finite and in [0,1).")
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("entropy_coef must be finite and nonnegative.")
    actor_parameters = list(policy.actor.named_parameters())
    if not actor_parameters:
        raise ValueError("Policy actor has no parameters.")
    linear_layers = [(name, module) for name, module in policy.actor.named_modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("Policy actor must contain a Linear input layer.")
    first_name, first_linear = linear_layers[0]
    first_weight_name = f"{first_name}.weight" if first_name else "weight"
    common_columns = int(getattr(policy, "single_proprio_dim", 78))
    if not 0 < common_columns <= first_linear.in_features:
        raise ValueError("Common proprioception width does not fit the actor input layer.")
    std_parameters = [(name, parameter) for name, parameter in policy.named_parameters()
                      if name in ("std", "log_std")]
    named_parameters = actor_parameters + std_parameters
    parameters = [parameter for _, parameter in named_parameters]
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("Audited policy parameters must be on a single device.")
    lp_batch = _prepare_batch(lp_batch, device)
    vp_batch = _prepare_batch(vp_batch, device)
    training_flags = [(module, module.training) for module in policy.modules()]
    requires_grad_flags = [(parameter, parameter.requires_grad) for parameter in parameters]
    try:
        policy.eval()
        for parameter in parameters:
            parameter.requires_grad_(True)
        with torch.enable_grad():
            lp_gradients, lp_losses = _mode_gradients(policy, lp_batch, parameters, clip_param, entropy_coef, normalize_advantages)
            vp_gradients, vp_losses = _mode_gradients(policy, vp_batch, parameters, clip_param, entropy_coef, normalize_advantages)
    finally:
        for parameter, requires_grad in requires_grad_flags:
            parameter.requires_grad_(requires_grad)
        for module, training in training_flags:
            module.training = training

    def flatten(parts):
        return torch.cat([part.reshape(-1).detach().to(device="cpu", dtype=torch.float64) for part in parts])

    report = {
        "sample_counts": {"lp": lp_batch["observations"].shape[0], "vp": vp_batch["observations"].shape[0]},
        "losses": {"lp": lp_losses, "vp": vp_losses},
        "config": {"clip_param": clip_param, "entropy_coef": entropy_coef,
                   "normalize_advantages_per_mode": normalize_advantages,
                   "common_proprio_columns": common_columns,
                   "first_actor_linear": first_name},
    }
    actor_count = len(actor_parameters)
    for objective in ("surrogate_only", "surrogate_plus_entropy"):
        lp, vp = lp_gradients[objective], vp_gradients[objective]
        blocks = {"whole_actor": gradient_pair_metrics(flatten(lp[:actor_count]), flatten(vp[:actor_count]))}
        shared_lp, shared_vp = [], []
        first_index = None
        for index, (name, _) in enumerate(actor_parameters):
            if name == first_weight_name:
                first_index = index
                shared_lp.append(lp[index][:, :common_columns])
                shared_vp.append(vp[index][:, :common_columns])
            else:
                shared_lp.append(lp[index])
                shared_vp.append(vp[index])
        if first_index is None:
            raise ValueError("Cannot locate actor input layer's weight parameter.")
        blocks["shared_actor"] = gradient_pair_metrics(flatten(shared_lp), flatten(shared_vp))
        blocks["specific_columns"] = gradient_pair_metrics(lp[first_index][:, common_columns:], vp[first_index][:, common_columns:])
        latent_width = getattr(policy, "estimator_latent_dim", None)
        if latent_width is not None and common_columns + int(latent_width) <= first_linear.in_features:
            boundary = common_columns + int(latent_width)
            blocks["lp_latent_columns"] = gradient_pair_metrics(lp[first_index][:, common_columns:boundary], vp[first_index][:, common_columns:boundary])
            blocks["vp_latent_columns"] = gradient_pair_metrics(lp[first_index][:, boundary:], vp[first_index][:, boundary:])
        blocks["linear_layers"] = {}
        for name, _ in linear_layers:
            layer_names = {f"{name}.weight", f"{name}.bias"} if name else {"weight", "bias"}
            indices = [index for index, (parameter_name, _) in enumerate(actor_parameters) if parameter_name in layer_names]
            blocks["linear_layers"][name] = gradient_pair_metrics(flatten([lp[index] for index in indices]), flatten([vp[index] for index in indices]))
        blocks["std"] = gradient_pair_metrics(flatten(lp[actor_count:]), flatten(vp[actor_count:])) if std_parameters else None
        report[objective] = blocks
    return report
