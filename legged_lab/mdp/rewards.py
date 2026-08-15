from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    
    from legged_lab.envs.g1.g1_rough_env import G1ROUGHEnv
    from legged_lab.envs.g1.g1_dwaq_env import G1Env

def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv  | G1ROUGHEnv |G1Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env: BaseEnv  | G1ROUGHEnv |G1Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def lin_vel_z_l2(env: BaseEnv  | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv  | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_vel_l2(env: BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)




def joint_pos_limits(env: BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def joint_acc_l2(env: BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv  | G1ROUGHEnv |G1Env) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def locomotion_action_rate_l2(env: BaseEnv | G1ROUGHEnv | G1Env) -> torch.Tensor:
    """Read the action-time Locomotion-only rate cost prepared by RENet."""
    return env.locomotion_action_rate_value


def recovery_action_rate_l2(env: BaseEnv | G1ROUGHEnv | G1Env) -> torch.Tensor:
    """Read the action-time Recovery-only rate cost prepared by the environment."""
    return env.recovery_action_rate_value


def enter_recovery_event(env: BaseEnv | G1ROUGHEnv | G1Env) -> torch.Tensor:
    """Indicate a NORMAL -> Recovery transition for locomotion failure penalty."""
    return env.enter_recovery_buf.float()


def undesired_contacts(env: BaseEnv  | G1ROUGHEnv |G1Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv  | G1ROUGHEnv |G1Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv  | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, 0:1])+2*torch.square(asset.data.projected_gravity_b[:,1:2]), dim=1)



def is_terminated(env: BaseEnv | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv |  G1ROUGHEnv |G1Env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def locomotion_feet_air_time_positive_biped(
    env: BaseEnv | G1ROUGHEnv | G1Env,
    threshold: float,
) -> torch.Tensor:
    """Use RENet's Locomotion-only foot timers with the existing reward math."""
    air_time = env.locomotion_feet_air_time
    contact_time = env.locomotion_feet_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0),
        dim=1,
    )[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv  | G1ROUGHEnv |G1Env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv  | G1ROUGHEnv |G1Envv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env:BaseEnv| G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv  | G1ROUGHEnv |G1Env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv  | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def joint_pos_limits_penalty(
    env: BaseEnv  | G1ROUGHEnv |G1Env, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9
) -> torch.Tensor:
    """
    惩罚超出关节位置极限的动作 (手动计算软极限，不依赖 Config)。
    对齐 GR-1 论文: 当关节位置逼近其物理极限的 90% 时开始线性惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取当前关节位置
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 2. 直接读取物理引擎里的绝对上下限
    # shape: (num_envs, num_joints, 2)
    limits = asset.data.default_joint_pos_limits[:, asset_cfg.joint_ids]
    lower_limits = limits[..., 0]
    upper_limits = limits[..., 1]
    
    # 3. 显式计算软极限区间
    # 比如 soft_ratio=0.9，就是两头各往里缩 5% 的行程
    joint_range = upper_limits - lower_limits
    margin = 0.5 * (1.0 - soft_ratio) * joint_range
    
    soft_lower = lower_limits + margin
    soft_upper = upper_limits - margin
    
    # 4. 计算超出软极限的惩罚 (clamp 对应 max(x, 0))
    out_of_lower = torch.clamp(soft_lower - joint_pos, min=0.0)
    out_of_upper = torch.clamp(joint_pos - soft_upper, min=0.0)
    
    # 5. 求和返回
    return torch.sum(out_of_lower + out_of_upper, dim=-1)

def joint_torque_limits_penalty(
    env: BaseEnv | G1ROUGHEnv |G1Env, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.8
) -> torch.Tensor:
    """
    惩罚超出软性关节扭矩(力矩)极限的动作。
    对齐 GR-1 论文公式: max(|tau| - 0.8 * tau_lim, 0)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取实际输出扭矩的绝对值
    applied_torque_abs = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    
    # 2. 获取底层设定的最大物理扭矩
    effort_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    
    # 3. 计算超出 80% (soft_ratio) 极限的部分
    out_of_limits = torch.clamp(applied_torque_abs - soft_ratio * effort_limits, min=0.0)
    
    # 4. 求和返回
    return torch.sum(out_of_limits, dim=-1)

def joint_vel_limits_penalty(
    env: BaseEnv | G1ROUGHEnv |G1Env, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9
) -> torch.Tensor:
    """
    惩罚超出软性关节速度极限的动作 (与 GR-1 论文保持一致)。
    论文逻辑: 当关节速度逼近其物理极限的 90% 时开始线性惩罚。
    """
    asset = env.scene[asset_cfg.name]
    
    # 1. 获取当前关节速度的绝对值
    # shape: (num_envs, num_joints)
    joint_vel_abs = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    
    # 2. 获取关节的物理速度极限 (通常是一个正数表示最大转速)
    # shape: (num_envs, num_joints)
    vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    
    # 3. 计算超出 90% 速度极限的偏差
    # 使用 clamp 对应论文中的 max(..., 0.0)
    out_of_limits = torch.clamp(joint_vel_abs - soft_ratio * vel_limits, min=0.0)
    
    # 4. 偏差求和
    return torch.sum(out_of_limits, dim=1)

# Regularization Reward
def ankle_torque(env: BaseEnv  | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_leg_ids[4],env.left_leg_ids[5],env.right_leg_ids[4],env.right_leg_ids[5]]]), dim=1)


def ankle_action(env: BaseEnv  | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[4],env.left_leg_ids[5],env.right_leg_ids[4],env.right_leg_ids[5]]]), dim=1)


def hip_roll_action(env: BaseEnv  | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[1], env.right_leg_ids[1]]]), dim=1)


def hip_yaw_action(env: BaseEnv  | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: BaseEnv  | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.ankle_link_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.ankle_link_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: BaseEnv  | G1ROUGHEnv |G1Env,  delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score

def feet_gait_improved(
    env:BaseEnv  | G1ROUGHEnv |G1Env, 
    sensor_cfg: SceneEntityCfg, 
    stance_threshold: float = 0.55,
    cmd_threshold: float = 0.1
) -> torch.Tensor:
    """
    融合版步态奖励：
    1. 严格使用 Z 轴接触力判断接触（防蹭地）
    2. 直接使用环境中计算好的相位（高效）
    3. 加入速度指令判断（防原地踏步）
    """
    # 1. 获取传感器和接触力数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # 2. 【优点融合】严格接触检测：Z轴受力必须大于 1.0 N
    is_contact = net_contact_forces[:, :, 2] > 1.0  # shape: (num_envs, 2)
    
    # 3. 【优点融合】直接读取你环境里维护的相位
    leg_phase = env.leg_phase  # shape: (num_envs, 2)
    is_stance = leg_phase < stance_threshold
    
    # 4. 计算基础相位匹配奖励
    # ~(is_contact ^ is_stance) 表示：该踩地时踩地，该悬空时悬空，奖励 1，否则 0
    phase_match = ~(is_contact ^ is_stance)
    reward = torch.sum(phase_match.float(), dim=-1) # shape: (num_envs,)
    
    # 5. 【优点融合】速度指令判断：如果没下达移动指令，就不强制要求步态
    # 在你的 Direct 环境中，速度指令存在 env.command_generator.command 中
    # 计算 XY 平面的线速度指令大小
    cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=1) 
    
    # 只有当速度指令大于 cmd_threshold 时，才给予步态奖励；否则乘 0
    reward *= (cmd_norm > cmd_threshold).float()
    
    return reward

def feet_gait(
    env: BaseEnv  | G1ROUGHEnv |G1Env,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=-1)
        reward *= cmd_norm > 0.1
    return reward
def idle_when_commanded(
    env: BaseEnv  | G1ROUGHEnv |G1Env,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a velocity command is given.
    
    This reward function detects "lazy standing" behavior where the robot receives
    a movement command but remains stationary. It returns 1.0 when the robot should
    be moving but is not, enabling a negative weight penalty.
    
    Args:
        env: Environment instance.
        cmd_threshold: Minimum command magnitude to be considered "commanded to move".
            Commands below this threshold are ignored (robot is allowed to stand).
        vel_threshold: Maximum velocity magnitude to be considered "idle/stationary".
            If actual velocity is below this, the robot is considered not moving.
        asset_cfg: Robot configuration.
    
    Returns:
        Tensor of shape (num_envs,) with values:
        - 1.0 if commanded to move but idle (should be penalized)
        - 0.0 otherwise (no penalty)
    
    Example:
        idle_penalty = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-2.0,
            params={"cmd_threshold": 0.2, "vel_threshold": 0.1}
        )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get velocity command (xy components)
    cmd_xy = env.command_generator.command[:, :2]
    cmd_magnitude = torch.linalg.norm(cmd_xy, dim=-1)
    
    # Get actual root velocity in yaw frame (same as track_lin_vel_xy uses)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_magnitude = torch.linalg.norm(vel_yaw[:, :2], dim=-1)
    
    # Detect "commanded but idle" condition
    is_commanded = cmd_magnitude > cmd_threshold  # Should be moving
    is_idle = vel_magnitude < vel_threshold       # But not moving
    
    return (is_commanded & is_idle).float()
def stand_still_pose(
    env,  
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """
    当没有速度指令时约束姿态。
    - 误差 > offset 时：施加惩罚。
    - 误差 < offset 时：给予正向奖励（将姿态精准吸附到默认位置）。
    """
    
    # 1. 适配环境指令读取
    lin_cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=1)
    ang_cmd_abs = torch.abs(env.command_generator.command[:, 2])
    zero_flag = (lin_cmd_norm < threshold) & (ang_cmd_abs < threshold)
    
    # 2. 获取关节位置
    asset = env.scene[asset_cfg.name]
    
    if asset_cfg.joint_ids is not None:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    else:
        joint_pos = asset.data.joint_pos
        default_joint_pos = asset.data.default_joint_pos
        
    # 3. 计算绝对误差总和
    dof_error = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)
    
    # 4. 
    # 允许产生负数输出，以便在 Config 中乘以负数 weight 后变成正向奖励
    return (dof_error - offset) * zero_flag.float()
def stand_still_joint_vel(env: BaseEnv  | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),threshold:float=0.1) -> torch.Tensor:
    """当命令为0时，平滑地惩罚所有关节的运动速度。"""
    cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    zero_flag = (cmd_norm < threshold).float()
    
    asset = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    
    # 使用 exp 将惩罚项软化并限制在 0 到 1 之间
    return (1.0 - torch.exp(-1.0 * joint_vel_sq)) * zero_flag


def gait_feet_spd_perio(env: BaseEnv  | G1ROUGHEnv |G1Env, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score

def gait_feet_frc_support_perio(env: BaseEnv  | G1ROUGHEnv |G1Env, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score
def track_ang_vel_z_exp(
    env: BaseEnv | G1ROUGHEnv |G1Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def alive(env: BaseEnv | G1ROUGHEnv |G1Env) -> torch.Tensor:
    """Reward for staying alive.
    
    A simple constant reward that encourages the robot to not terminate early.
    Reference: DreamWaQ (HumanoidDreamWaq/legged_gym/envs/g1/g1_env.py)
    """
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)

def new_gait_phase_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    stance_threshold: float = 0.55,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """只在有移动命令时启用接触相位奖励。"""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w[
        :, sensor_cfg.body_ids, :
    ]

    # 左右脚是否实际接触地面
    contact = net_contact_forces[:, :, 2] > 1.0

    # 左右脚期望支撑状态
    is_stance = env.leg_phase < stance_threshold

    # 接触状态与期望状态一致
    phase_match = ~(contact ^ is_stance)
    reward = torch.sum(phase_match.float(), dim=-1)

    command = env.command_generator.command

    command_norm = (
        torch.linalg.vector_norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    )

    move_mask = command_norm > command_threshold

    return reward * move_mask.float()

def gait_phase_contact(
    env: BaseEnv  | G1ROUGHEnv |G1Env, sensor_cfg: SceneEntityCfg, stance_threshold: float = 0.55
) -> torch.Tensor:
    """Reward for foot contact matching the expected gait phase.
    
    Rewards the robot when foot contact status matches the expected stance/swing phase.
    During stance phase (phase < stance_threshold), foot should be in contact.
    During swing phase (phase >= stance_threshold), foot should be in the air.
    
    Args:
        env: Environment with gait phase information.
        sensor_cfg: Contact sensor configuration for feet.
        stance_threshold: Phase threshold below which the foot should be in stance.
        
    Reference: DreamWaQ _reward_contact()
    
    Note: This function uses env.leg_phase which should be [num_envs, num_feet] tensor
    where leg_phase[:, 0] = phase_left and leg_phase[:, 1] = phase_right.
    The sensor_cfg.body_ids should match the same ordering (left foot first, right foot second).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # Check contact for each foot (use z-component like original DreamWaQ)
    # Original: contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    contact = net_contact_forces[:, :, 2] > 1.0  # (num_envs, num_feet), z-direction force
    
    # Use leg_phase directly from environment
    # leg_phase shape: (num_envs, 2) where [:, 0] = left, [:, 1] = right
    leg_phase = env.leg_phase
    
    # Expected stance: phase < stance_threshold
    is_stance = leg_phase < stance_threshold
    
    # Reward: 1 if contact matches expected phase, 0 otherwise
    # XOR gives True when they don't match, so we negate it
    phase_match = ~(contact ^ is_stance)  # (num_envs, num_feet)
    
    return torch.sum(phase_match.float(), dim=-1)  # Sum over feet


def joint_deviation_l1_always(
    env:BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize joint deviation from default pose at all times (not just when standing).

    Use this for limbs (e.g., arms) that should maintain a default pose even during locomotion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def foot_clearance_reward(
    env: BaseEnv | G1ROUGHEnv |G1Env, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset:Articulation  = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def feet_swing_height(
    env:BaseEnv | G1ROUGHEnv |G1Env, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08
) -> torch.Tensor:
    """Simple version: Penalize swing foot height deviation from fixed target.
    
    This is the original simple implementation that uses absolute z-coordinate.
    Use feet_swing_height() for terrain-aware version.
    
    Args:
        env: Environment.
        sensor_cfg: Contact sensor configuration for feet.
        asset_cfg: Robot configuration with body_ids for feet.
        target_height: Target height for swing foot (default 0.08m).
        
    Reference: DreamWaQ _reward_feet_swing_height()
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_contact_forces, dim=-1) > 1.0  # (num_envs, num_feet)
    
    # Get feet positions (z-coordinate)
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
    
    # Penalize height error only during swing phase (not in contact)
    pos_error = torch.square(feet_pos_z - target_height) * (~contact).float()
    
    return torch.sum(pos_error, dim=-1)

def single_foot_contact_area_penalty(
    env: BaseEnv| G1ROUGHEnv |G1Env, 
    contact_sensor_cfg: SceneEntityCfg,  # 接触传感器配置
    ray_sensor_cfg: SceneEntityCfg,      # 【新增】射线传感器配置
    threshold: float = 0.04  
) -> torch.Tensor:
    """计算单只脚踩在台阶边缘的惩罚"""
    
    # ==========================================
    # 1. 接触力检测：判断这只脚物理上是否踩到了地面
    # ==========================================
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # contact_sensor.data.net_forces_w_history 的形状是 (num_envs, history_length, num_bodies, 3)
    # 我们取 history_length 的第 0 个元素（当前最新帧）
    # 经过切片后，current_forces 的形状为 (num_envs, len(body_ids), 3)
    current_forces = contact_sensor.data.net_forces_w_history[:, 0, contact_sensor_cfg.body_ids, :]
    
    # 计算力向量的模长 (L2范数)，然后在最后一个维度上降维 (squeeze)
    # 最终 is_contact 形状为 (num_envs,)，值为 True 或 False
    '''
    squeeze()
    '''
    is_contact = torch.norm(current_forces, dim=-1).squeeze(-1) > 1.0  
    
    # ==========================================
    # 2. 射线检测：计算脚底板有多少面积是“踩实”的
    # ==========================================
    ray_sensor = env.scene.sensors[ray_sensor_cfg.name]
    
    # ray_sensor.data.pos_w 存着传感器自身的坐标，形状为 (num_envs, 3)
    # 取 Z 轴 [..., 2] 后形状变为 (num_envs,)
    # 为了后续和多根射线广播相减，我们用 unsqueeze 扩展维度为 (num_envs, 1)
    sensor_z = ray_sensor.data.pos_w[..., 2].unsqueeze(-1) 
    
    # ray_sensor.data.ray_hits_w 存着每根射线击中点的坐标，形状为 (num_envs, num_rays, 3)
    # 我们只关心 Z 轴高度，提取后形状为 (num_envs, num_rays)
    hit_z = ray_sensor.data.ray_hits_w[..., 2]
    
    # ==========================================
    # 3. 计算踩实比例与惩罚
    # ==========================================
    # 算出脚底每根射线打中点距离传感器的高度差
    # 如果射线打空（打到空气/无穷远），hit_z 是 inf，相减也是 inf
    ray_distances = torch.abs(sensor_z - hit_z)
    
    # 拿到单只脚的总射线数量 (例如 60 根)
    num_rays = ray_distances.shape[-1]
    
    # 统计高度差小于设定阈值 (0.04m) 的射线数量，也就是“踩实了”的射线
    solid_rays = torch.sum(ray_distances < threshold, dim=-1)
    
    # 计算接触比例：踩实的射线数 / 总射线数
    # contact_ratio 的形状是 (num_envs,)，值在 0.0 ~ 1.0 之间
    contact_ratio = solid_rays.float() / num_rays
    
    # 最终惩罚逻辑：
    # - 如果脚没踩地 (is_contact = False)，惩罚为 0 (乘以 0)
    # - 如果脚踩地了，且 100% 踩实 (contact_ratio = 1.0)，惩罚为 0
    # - 如果脚踩地了，但只踩了一半边缘 (contact_ratio = 0.5)，惩罚为 0.5
    penalty = (1.0 - contact_ratio) * is_contact.float()
    
    return penalty
def is_alive(env: BaseEnv ) -> torch.Tensor:
    """Reward for staying alive (not being reset)."""
    return torch.ones_like(env.episode_length_buf, dtype=torch.float32)

def dont_wait(
    env: BaseEnv , asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command.
    Checks if command is > 0.3 m/s forwards, but actual forwards speed is < 0.15 m/s.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_generator.command[:, 0]
    # Local forward velocity (using heading)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_x = vel_yaw[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: BaseEnv | G1ROUGHEnv |G1Env, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """当脚接触地面时，惩罚脚掌不水平的姿态（促使脚掌完美贴合地面）"""
    
    # 1. 获取机器人对象
    asset = env.scene[asset_cfg.name]
    
    # 2. 获取左右脚当前的旋转四元数
    # asset_cfg.body_ids 应该严格传入 [左脚索引, 右脚索引]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    
    # 3. 将世界坐标系的重力向量投影到脚底局部坐标系
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    
    # 4. 获取接触传感器的受力情况
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # 沿历史窗口(dim=1)取最大受力，防止抖动。计算出的 is_contact 形状为 (num_envs, 2)
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0

    # 5. 计算惩罚：XY平面投影的模长 * 接触掩码
    # 提取 [:2] 即 X (Pitch) 和 Y (Roll) 的分量
    left_penalty = torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
    right_penalty = torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    
    # 返回双脚惩罚之和
    return left_penalty + right_penalty

def straight_body_penalty_with_deadzone(
    env: BaseEnv | G1ROUGHEnv |G1Env, 
    asset_cfg: SceneEntityCfg= SceneEntityCfg("robot"),
    threshold: float = 0.05  # 允许一定范围内的自然倾斜
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b[:, asset_cfg.body_ids, :]
    
    # 1. 计算水平方向的倾斜程度 (L2范数，不带平方)
    tilt_magnitude = torch.norm(projected_gravity[:, :, :2], dim=-1)
    
    # 2. 引入死区：减去容忍阈值，并将小于0的部分置为0
    tilt_excess = torch.clamp(tilt_magnitude - threshold, min=0.0)
    
    # 3. 对超出阈值的部分进行平方惩罚，保持软惩罚的平滑特性
    reward = torch.sum(torch.square(tilt_excess), dim=1)
    
    return reward

def track_height_cmd(env:BaseEnv,
                     asset_cfg:SceneEntityCfg=SceneEntityCfg("robot"),
                     std: float = 0.05,
                    ):
    asset:Articulation=env.scene[asset_cfg.name]
    current_robot_height=asset.data.root_state_w[:,2]
    current_height_command=env.command_generator.command[:,0]
    height_error=current_robot_height-current_height_command
    return torch.exp(-torch.square(height_error) / std**2)
    

def com_support_margin_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.03,
    outside_scale: float = 10.0,
    rear_scale: float = 1.5,
) -> torch.Tensor:
    """惩罚整机质心接近或越过双脚支撑域边界。

    Args:
        env:
            强化学习环境。

        asset_cfg:
            机器人配置。目前函数直接读取 env 中的缓存，
            因此此参数主要用于保持奖励函数接口统一。

        std:
            边界惩罚的作用距离，单位为米。
            例如 0.03 表示距离边界约 3 cm 时开始明显惩罚。

        outside_scale:
            质心投影越过支撑域后的额外惩罚系数。

        rear_scale:
            后边界惩罚倍率。大于 1 时，更强地抑制重心后移。

    Returns:
        penalty:
            shape = (num_envs,)

            返回正惩罚值，建议在配置中使用负权重。
    """

    del asset_cfg

    # ---------------------------------------------------------
    # 1. 获取脚底支撑点和整机质心水平投影
    # ---------------------------------------------------------

    # shape: (num_envs, 2, 4, 2)
    foot_support_xy_points = (
        env.foot_support_corners_w[..., :2]
    )

    # shape: (num_envs, 2)
    com_xy_w = env.whole_body_com_pos_w[:, :2]

    # ---------------------------------------------------------
    # 2. 根据脚底前后点建立支撑坐标系
    #
    # 角点顺序：
    # 0: 前端 +y
    # 1: 前端 -y
    # 2: 后端 -y
    # 3: 后端 +y
    # ---------------------------------------------------------

    # 每只脚的前端中心
    # shape: (num_envs, 2, 2)
    foot_front_center_xy = (
        foot_support_xy_points[:, :, 0:2, :]
        .mean(dim=2)
    )

    # 每只脚的后端中心
    # shape: (num_envs, 2, 2)
    foot_rear_center_xy = (
        foot_support_xy_points[:, :, 2:4, :]
        .mean(dim=2)
    )

    # 两只脚的平均前向方向
    # shape: (num_envs, 2)
    support_x_axis_w = (
        foot_front_center_xy
        - foot_rear_center_xy
    ).mean(dim=1)

    support_x_axis_w = support_x_axis_w / (
        torch.linalg.vector_norm(
            support_x_axis_w,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
    )

    # 水平面内与 x 轴垂直的 y 轴
    support_y_axis_w = torch.stack(
        [
            -support_x_axis_w[:, 1],
            support_x_axis_w[:, 0],
        ],
        dim=-1,
    )

    # 支撑坐标系原点取 8 个脚底点的平均值
    # shape: (num_envs, 2)
    support_origin_xy_w = (
        foot_support_xy_points.mean(dim=(1, 2))
    )

    # ---------------------------------------------------------
    # 3. 把脚底点投影到支撑坐标系
    # ---------------------------------------------------------

    relative_points_xy = (
        foot_support_xy_points
        - support_origin_xy_w[:, None, None, :]
    )

    # shape: (num_envs, 2, 4)
    point_x = torch.sum(
        relative_points_xy
        * support_x_axis_w[:, None, None, :],
        dim=-1,
    )

    point_y = torch.sum(
        relative_points_xy
        * support_y_axis_w[:, None, None, :],
        dim=-1,
    )

    # 双脚支撑矩形边界
    # shape: (num_envs,)
    x_min = point_x.amin(dim=(1, 2))
    x_max = point_x.amax(dim=(1, 2))
    y_min = point_y.amin(dim=(1, 2))
    y_max = point_y.amax(dim=(1, 2))

    # ---------------------------------------------------------
    # 4. 把整机质心投影到同一个支撑坐标系
    # ---------------------------------------------------------

    relative_com_xy = (
        com_xy_w - support_origin_xy_w
    )

    # shape: (num_envs,)
    com_x = torch.sum(
        relative_com_xy * support_x_axis_w,
        dim=-1,
    )

    com_y = torch.sum(
        relative_com_xy * support_y_axis_w,
        dim=-1,
    )

    # ---------------------------------------------------------
    # 5. 计算质心到四条边界的有符号距离
    #
    # 距离 > 0：在支撑域内部
    # 距离 = 0：位于边界
    # 距离 < 0：已经越过边界
    # ---------------------------------------------------------

    rear_distance = com_x - x_min
    front_distance = x_max - com_x
    side_negative_distance = com_y - y_min
    side_positive_distance = y_max - com_y

    all_distances = torch.stack(
        [
            rear_distance,
            front_distance,
            side_negative_distance,
            side_positive_distance,
        ],
        dim=-1,
    )

    # 离质心最近的支撑域边界
    min_signed_distance = all_distances.amin(dim=-1)

    # ---------------------------------------------------------
    # 6. 支撑域边缘惩罚
    #
    # 在中心区域接近 0；
    # 靠近边界时接近 1。
    # ---------------------------------------------------------

    inside_distance = torch.clamp(
        min_signed_distance,
        min=0.0,
    )

    edge_penalty = torch.exp(
        -inside_distance / std
    )

    # ---------------------------------------------------------
    # 7. 越界惩罚
    # ---------------------------------------------------------

    outside_distance = torch.relu(
        -min_signed_distance
    )

    outside_penalty = (
        outside_distance / std
    ).square()

    # ---------------------------------------------------------
    # 8. 单独加强脚跟方向惩罚
    # ---------------------------------------------------------

    rear_inside_distance = torch.clamp(
        rear_distance,
        min=0.0,
    )

    rear_edge_penalty = torch.exp(
        -rear_inside_distance / std
    )

    rear_outside_distance = torch.relu(
        -rear_distance
    )

    rear_outside_penalty = (
        rear_outside_distance / std
    ).square()

    rear_penalty = (
        rear_edge_penalty
        + outside_scale * rear_outside_penalty
    )
    # rear_scale=1 时，不添加额外后侧惩罚
    penalty = (
        edge_penalty
        + outside_scale * outside_penalty
        + (rear_scale - 1.0) * rear_penalty
    )
    return penalty

def feet_no_contact(env: BaseEnv  | G1ROUGHEnv |G1Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_no_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] < threshold
    return torch.sum(is_no_contact, dim=1)

def feet_xy_velocity(env:BaseEnv,threshold:float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset:Articulation=env.scene[asset_cfg.name]
    feet_vel_xy=asset.data.body_link_vel_w[:,asset_cfg.body_ids,:2]
    feet_speed_squared = torch.sum(
        torch.square(feet_vel_xy),
        dim=-1,
    )
    penalty_per_foot = torch.relu(
        feet_speed_squared - threshold**2
    )

    # Sum over all selected feet.
    # Shape: (num_envs,)
    return torch.sum(penalty_per_foot, dim=-1)


def com_projection_to_ankle_center(
    env,
    std: float = 0.08,
) -> torch.Tensor:
    """奖励整机质心投影接近左右脚踝中心。

    Args:
        env:
            SquatStandEnv 环境实例。
        std:
            奖励的容许距离，单位为米。
            越小，奖励对偏移越敏感。

    Returns:
        shape = (num_envs,)
        每个并行环境的奖励，范围为 (0, 1]。
    """
    # asset:Articulation=env.scene[asset_cfg.name]
    # height_cmd = env.command_generator.command[:,0]
    # zero_flag=height_cmd<threshold

    # 整机质心在世界坐标系中的水平投影
    # shape: (num_envs, 2)
    com_xy = env.whole_body_com_pos_w[:, :2]

    # 左右脚踝 link 的世界坐标
    # shape: (num_envs, 2, 2)
    ankle_xy = env.robot.data.body_link_pos_w[
        :, env.ankle_link_ids, :2
    ]

    # 左右脚踝连线的中心
    # shape: (num_envs, 2)
    ankle_center_xy = ankle_xy.mean(dim=1)

    # 质心投影相对于脚踝中心的偏差
    error_xy = com_xy - ankle_center_xy

    # 平方距离，避免不必要的 sqrt
    squared_distance = torch.sum(
        torch.square(error_xy),
        dim=1,
    )

    # 高斯型奖励
    reward = torch.exp(
        -0.5 * squared_distance / (std**2)
    )

    return reward

def torse_penalty(env:BaseEnv,asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")):
    asset:Articulation=env.scene[asset_cfg.name]
    waist_pitch_pos=torch.relu(-asset.data.joint_pos[:,asset_cfg.joint_ids])
    return torch.sum(torch.square(waist_pitch_pos), dim=1)
    
