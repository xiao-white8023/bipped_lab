"""TorchScript input-layout detection for the g1_squart deployment."""

from dataclasses import dataclass

import torch


CURRENT_SINGLE_OBS_DIM = 80
LEGACY_SINGLE_OBS_DIM = 81
SUPPORTED_LAYOUTS = ("auto", "current", "legacy_root_height")


@dataclass(frozen=True)
class SquatPolicyLayout:
    """Resolved raw-observation interface of an exported squat policy."""

    name: str
    single_obs_dim: int
    history_length: int
    policy_obs_dim: int
    includes_root_height: bool
    has_cenet: bool


def _has_cenet(policy: torch.nn.Module) -> bool:
    return any("cenet_module" in name for name, _ in policy.named_modules())


def _try_policy_input(
    policy: torch.nn.Module,
    obs_dim: int,
    num_actions: int,
) -> tuple[bool, str]:
    try:
        with torch.inference_mode():
            output = policy(torch.zeros(1, obs_dim, dtype=torch.float32))
    except Exception as exc:  # TorchScript reports dimension failures as RuntimeError.
        error_lines = str(exc).strip().splitlines()
        return False, error_lines[-1] if error_lines else type(exc).__name__

    if not isinstance(output, torch.Tensor):
        return False, f"policy returned {type(output).__name__}, expected torch.Tensor"
    if tuple(output.shape) != (1, num_actions):
        return False, f"policy returned shape {tuple(output.shape)}, expected (1, {num_actions})"
    if not torch.isfinite(output).all():
        return False, "policy returned NaN or Inf during startup validation"
    return True, ""


def resolve_squat_policy_layout(
    policy: torch.nn.Module,
    history_length: int,
    num_actions: int,
    requested_layout: str = "auto",
    configured_single_obs_dim: int | None = None,
) -> SquatPolicyLayout:
    """Probe an exported policy and select its 80- or 81-value frame layout.

    Current g1_squart policies, including both CENet-enabled and CENet-disabled
    variants, receive 10 x 80 raw actor observations. Older exported policies
    used an additional root-height value and therefore receive 10 x 81.
    """

    if requested_layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            f"policy_obs_layout must be one of {SUPPORTED_LAYOUTS}, got {requested_layout!r}."
        )
    if history_length <= 0:
        raise ValueError(f"obs_history_length must be positive, got {history_length}.")

    if requested_layout == "current":
        candidate_dims = [CURRENT_SINGLE_OBS_DIM]
    elif requested_layout == "legacy_root_height":
        candidate_dims = [LEGACY_SINGLE_OBS_DIM]
    else:
        candidate_dims = []
        if configured_single_obs_dim in (CURRENT_SINGLE_OBS_DIM, LEGACY_SINGLE_OBS_DIM):
            candidate_dims.append(int(configured_single_obs_dim))
        for dim in (CURRENT_SINGLE_OBS_DIM, LEGACY_SINGLE_OBS_DIM):
            if dim not in candidate_dims:
                candidate_dims.append(dim)

    failures = []
    for single_obs_dim in candidate_dims:
        policy_obs_dim = history_length * single_obs_dim
        accepted, error = _try_policy_input(policy, policy_obs_dim, num_actions)
        if not accepted:
            failures.append(f"{history_length}x{single_obs_dim}={policy_obs_dim}: {error}")
            continue

        has_cenet = _has_cenet(policy)
        if has_cenet and single_obs_dim != CURRENT_SINGLE_OBS_DIM:
            failures.append(
                f"{history_length}x{single_obs_dim}={policy_obs_dim}: "
                "CENet policy must use the current 80-value observation frame"
            )
            continue

        return SquatPolicyLayout(
            name="current" if single_obs_dim == CURRENT_SINGLE_OBS_DIM else "legacy_root_height",
            single_obs_dim=single_obs_dim,
            history_length=history_length,
            policy_obs_dim=policy_obs_dim,
            includes_root_height=single_obs_dim == LEGACY_SINGLE_OBS_DIM,
            has_cenet=has_cenet,
        )

    details = "\n  - ".join(failures)
    raise RuntimeError(
        "The exported g1_squart TorchScript policy is incompatible with the deployment "
        "observation interface. Tried:\n  - "
        f"{details}\n"
        "Current policies must accept [1, 800] and return [1, 15]. Re-export the "
        "checkpoint with legged_lab/scripts/play_squart.py."
    )
