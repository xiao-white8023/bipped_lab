from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Literal

import torch

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions, get_isaac_sim_version

# 随机初始化胳膊的默认位置，并保持一个episode不变
def reset_arm_pose_and_hold( env: ManagerBasedEnv,
                            env_ids: torch.Tensor,
                            position_ranges: dict[str,tuple[float, float]],
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                            ):
    asset: Articulation = env.scene[asset_cfg.name]
    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids
    
    joint_names = [asset.joint_names[joint_id] for joint_id in asset_cfg.joint_ids]
    missing_joints = [
        joint_name
        for joint_name in joint_names
        if joint_name not in position_ranges
    ]
    if missing_joints:
        raise ValueError(
            f"Missing position ranges for joints: {missing_joints}"
        )
    
    unused_joints = [
        joint_name
        for joint_name in position_ranges
        if joint_name not in joint_names
    ]
    if unused_joints:
        raise ValueError(
            f"position_ranges contains joints not selected by asset_cfg: "
            f"{unused_joints}"
        )
    
    # Construct each joint's lower and upper sampling bounds.
    lower_bounds = torch.tensor(
        [position_ranges[name][0] for name in joint_names],
        device=asset.data.default_joint_pos.device,
        dtype=asset.data.default_joint_pos.dtype,
    )

    upper_bounds = torch.tensor(
        [position_ranges[name][1] for name in joint_names],
        device=asset.data.default_joint_pos.device,
        dtype=asset.data.default_joint_pos.dtype,
    )
    # Validate the configured ranges.
    invalid_range_mask = lower_bounds > upper_bounds
    if torch.any(invalid_range_mask):
        invalid_names = [
            joint_names[index]
            for index in torch.nonzero(
                invalid_range_mask,
                as_tuple=False,
            ).flatten().tolist()
        ]
        raise ValueError(
            f"Lower bound is greater than upper bound for joints: "
            f"{invalid_names}"
        )
    num_envs = env_ids.numel()
    num_joints = len(asset_cfg.joint_ids)

    # Independently sample every joint in every resetting environment.
    random_values = torch.rand(
        (num_envs, num_joints),
        device=lower_bounds.device,
        dtype=lower_bounds.dtype,
    )
    joint_pos = lower_bounds.unsqueeze(0) + random_values * (
        upper_bounds - lower_bounds
    ).unsqueeze(0)

    # Clamp sampled values to the robot's physical soft joint limits.
    iter_env_ids = env_ids[:, None]

    joint_pos_limits = asset.data.soft_joint_pos_limits[
        iter_env_ids,
        asset_cfg.joint_ids,
    ]

    joint_pos.clamp_(
        joint_pos_limits[..., 0],
        joint_pos_limits[..., 1],
    )

    # Arms start stationary.
    joint_vel = torch.zeros_like(joint_pos)

    # Set the physical arm state at reset.
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )

    # Set the same pose as the controller target for this episode.
    asset.set_joint_position_target(
        joint_pos,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )

    asset.set_joint_velocity_target(
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )
