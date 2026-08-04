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

"""Implementation of runners for environment-agent interaction."""

from .amp_on_policy_runner import AmpOnPolicyRunner
from .on_policy_runner import OnPolicyRunner
from .film_on_policy_runner import FilmOnPolicyRunner


class _RemovedRunner:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This runner implementation was removed when keeping only "
            "g1_rough, g1_film, and g1_squart training tasks."
        )


DWAQAmpOnPolicyRunner = _RemovedRunner
MoeAmpOnPolicyRunner = _RemovedRunner
MoeOnPolicyRunner = _RemovedRunner
G1OnPolicyRunner = _RemovedRunner

__all__ = [
    "OnPolicyRunner",
    "AmpOnPolicyRunner",
    "DWAQAmpOnPolicyRunner",
    "MoeAmpOnPolicyRunner",
    "MoeOnPolicyRunner",
    "G1OnPolicyRunner",
    "FilmOnPolicyRunner"
]
