from __future__ import annotations

from .moe_ppo import MoePPO


class RENetPPO(MoePPO):
    """PPO with the auxiliary losses needed by RENet training.

    The current implementation reuses the existing auxiliary-loss machinery
    from MoePPO: velocity estimation and terrain reconstruction.
    """

