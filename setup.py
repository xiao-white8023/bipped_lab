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

from setuptools import find_packages, setup

setup(
    name="LeggedLab",
    packages=find_packages(),
    package_data={
        "legged_lab.sensors.grouped_ray_caster": [
            "data/*.npy",
            "data/*.md",
            "data/LICENSE*",
        ],
        "legged_lab.terrains.extreme_parkour": ["LICENSE", "NOTICE"],
    },
    version="0.1.0",
    install_requires=["IsaacLab", "pynput", "mujoco==3.3.2", "mujoco-python-viewer", "matplotlib"],
)
