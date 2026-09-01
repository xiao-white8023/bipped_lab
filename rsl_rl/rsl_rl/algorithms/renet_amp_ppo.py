from __future__ import annotations

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
        super().__init__(*args, **kwargs)
