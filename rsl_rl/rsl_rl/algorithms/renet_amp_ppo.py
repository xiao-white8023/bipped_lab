from __future__ import annotations

import inspect

from .amp_ppo import AMPPPO


class RENetAMPPPO(AMPPPO):
    """AMP PPO variant used by RENet training."""

    def __init__(self, *args, **kwargs):
        # RENet grew out of the visual/MoE configs; these keys are irrelevant
        # for the AMP optimizer and are kept here only for config compatibility.
        for key in (
            "obs_dim",
            "use_moe_balance_loss",
            "moe_balance_coef",
            "use_moe_gate_entropy_loss",
            "moe_gate_entropy_coef",
        ):
            kwargs.pop(key, None)

        renet_aux_keys = (
            "feet_height_coef",
            "feet_height_warmup_iters",
            "feet_height_dim",
            "feet_height_in_critic_offset",
        )
        renet_aux_cfg = {key: kwargs.get(key) for key in renet_aux_keys if key in kwargs}

        parent_params = inspect.signature(AMPPPO.__init__).parameters
        parent_accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parent_params.values())
        if not parent_accepts_kwargs:
            for key in renet_aux_keys:
                if key not in parent_params:
                    kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        # Keep RENet-specific attributes available even when running against an
        # older AMPPPO implementation that does not declare these parameters.
        if renet_aux_cfg:
            self.feet_height_coef = renet_aux_cfg.get("feet_height_coef", 0.0)
            self.feet_height_warmup_iters = renet_aux_cfg.get("feet_height_warmup_iters", 0)
            self.feet_height_dim = renet_aux_cfg.get("feet_height_dim", 2)
            self.feet_height_in_critic_offset = renet_aux_cfg.get("feet_height_in_critic_offset", 0)
            self.feet_height_obs_start_idx = (
                (self.critic_history_len - 1) * self.single_critic_dim + self.feet_height_in_critic_offset
            )
