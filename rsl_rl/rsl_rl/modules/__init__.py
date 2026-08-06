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

"""Definitions for neural-network components for RL-agents."""

from .actor_critic_rough import ActorCritic
from .discriminator import Discriminator
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .cnn_mlp import CnnMlp
from .film_cnn_mlp import FiLMCnnMlp

from .moe_actor_critic import MoeActorCritic
from .mha_actor_critic import MhaActorCritic


__all__ = [
    "ActorCritic",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "Discriminator",
    'CnnMlp',
    "FiLMCnnMlp",
    "MoeActorCritic",
    "MhaActorCritic",
]
