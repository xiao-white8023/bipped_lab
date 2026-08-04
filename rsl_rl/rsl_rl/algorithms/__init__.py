# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Implementation of different RL agents."""
from .amp_ppo import AMPPPO
from .moe_ppo import MoePPO
from .ppo import PPO


class _RemovedAlgorithm:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This algorithm implementation was removed when keeping only "
            "g1_rough, g1_film, and g1_squart training tasks."
        )


MoeAmpPpO = _RemovedAlgorithm

__all__ = ["PPO", "AMPPPO", "MoeAmpPpO", "MoePPO"]
