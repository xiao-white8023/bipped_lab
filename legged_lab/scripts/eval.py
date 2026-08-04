import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="The function 'quat_rotate_inverse' will be deprecated")

current_dir = os.path.dirname(os.path.abspath(__file__))
local_rsl_rl_path = os.path.abspath(os.path.join(current_dir, "../../rsl_rl"))
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained policy with RSL-RL.")

parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--terrain", type=str, default="rough_perlin")

# eval 参数
parser.add_argument("--num_eval_episodes", type=int, default=1000)
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)

# target distance：用于目标区域、MXD、ReachSteps/ReachTime
# 保留 success_distance 这个命令行名字，避免你原来的脚本需要改。
parser.add_argument("--success_distance", type=float, default=None)

# 目标区域的横向范围：只在“到达目标区域”那一刻判断，不要求整个过程都在范围内。
# 使用相对 env_origin 的 root_y，避免并行 env spacing 污染。
parser.add_argument("--success_y_min", type=float, default=-1.5)
parser.add_argument("--success_y_max", type=float, default=1.5)

# feet stumble 参数：足端水平力显著大于竖直力，近似脚部撞边/绊脚
parser.add_argument("--feet_stumble_ratio", type=float, default=5.0)
parser.add_argument("--feet_stumble_force_threshold", type=float, default=10.0)

# depth 噪声 eval
parser.add_argument("--eval_depth_noise", action="store_true")
parser.add_argument("--depth_noise_std", type=float, default=0.0)
parser.add_argument("--depth_failure_prob", type=float, default=0.0)

# 是否打开本体观测噪声。默认 eval 时关掉
parser.add_argument("--eval_obs_noise", action="store_true")

# 输出
parser.add_argument("--csv", type=str, default="eval_results.csv")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 你的策略用到了 camera / depth，这里建议强制打开 cameras
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import legged_lab.utils.cli_args as cli_args
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

from rsl_rl.runners import (
    AmpOnPolicyRunner,
    OnPolicyRunner,
    MoeAmpOnPolicyRunner,
    DWAQAmpOnPolicyRunner,
    MoeOnPolicyRunner,
    FilmOnPolicyRunner,
)
from rsl_rl.runners.g1_student_on_policy_runner import G1StudentOnPolicyRunner

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.terrains import (
    GRAVEL_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    ROUGH_PERLIN_TERRAINS_CFG,
)

cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_eval_env(env_cfg):
    """把训练环境改成 eval 环境。"""

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.max_episode_length_s = 20.0
    env_cfg.scene.env_spacing = 2.5

    # eval 时先关掉本体噪声，除非你显式打开
    env_cfg.noise.add_noise = bool(args_cli.eval_obs_noise)

    # eval 时先关掉随机 push
    env_cfg.domain_rand.events.push_robot = None

    # eval 不要手柄，固定速度指令
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = True
    env_cfg.commands.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
    env_cfg.commands.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)

    # 尽量固定 reset 初始姿态，避免 eval 时每次朝向随机
    reset_base = getattr(env_cfg.domain_rand.events, "reset_base", None)
    if reset_base is not None:
        reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    reset_joints = getattr(env_cfg.domain_rand.events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)

    # terrain 选择
    if args_cli.terrain == "rough":
        terrain = ROUGH_TERRAINS_CFG
    elif args_cli.terrain == "rough_perlin":
        terrain = ROUGH_PERLIN_TERRAINS_CFG
    elif args_cli.terrain == "gravel":
        terrain = GRAVEL_TERRAINS_CFG
    else:
        terrain = ROUGH_PERLIN_TERRAINS_CFG

    env_cfg.scene.terrain_generator = terrain
    env_cfg.scene.terrain_type = "generator"

    # eval 阶段先冻结 terrain curriculum，避免环境难度边跑边变
    if env_cfg.scene.terrain_generator is not None:
        if hasattr(env_cfg.scene.terrain_generator, "curriculum"):
            env_cfg.scene.terrain_generator.curriculum = False

    # depth noise 设置
    env_cfg.depth_noise_curriculum.enable = False

    if args_cli.eval_depth_noise:
        env_cfg.scene.camera.add_camera_noise = True
        env_cfg.depth_noise_curriculum.gaussian_std_range = (
            0.0,
            args_cli.depth_noise_std,
        )
        env_cfg.depth_noise_curriculum.failure_probability_range = (
            0.0,
            args_cli.depth_failure_prob,
        )
    else:
        env_cfg.scene.camera.add_camera_noise = False
        env_cfg.depth_noise_curriculum.gaussian_std_range = (0.0, 0.0)
        env_cfg.depth_noise_curriculum.failure_probability_range = (0.0, 0.0)

    # eval 不需要可视化 camera ray
    if hasattr(env_cfg.scene.camera, "camera"):
        env_cfg.scene.camera.camera.debug_vis = False

    return env_cfg


def get_forward_progress_from_start(env, episode_start_x):
    """计算每个 env 从本 episode 起点开始沿 world x 方向前进了多远。"""
    root_x = env.robot.data.root_pos_w[:, 0]
    return root_x - episode_start_x


def get_local_root_y(env):
    """返回相对 env_origin 的 root_y。"""
    root_y = env.robot.data.root_pos_w[:, 1]
    origin_y = env.scene.env_origins[:, 1]
    return root_y - origin_y


def get_body_collision(env):
    """检测 torso 等终止身体是否接触地面/障碍。"""
    net_contact_forces = env.contact_sensor.data.net_forces_w_history

    contact_force = torch.norm(
        net_contact_forces[:, :, env.termination_contact_cfg.body_ids],
        dim=-1,
    )

    max_contact_force = torch.max(contact_force, dim=1)[0]
    body_collision = torch.any(max_contact_force > 1.0, dim=1)

    return body_collision


def get_feet_stumble(env, ratio_threshold=5.0, horizontal_force_threshold=10.0):
    """检测足端 stumble / 撞边型接触。

    判据：足端水平接触力较大，且显著大于竖直接触力。
    这比直接统计足底接触更适合近似楼梯上的脚尖/足端撞边。
    """
    contact_data = env.contact_sensor.data

    if hasattr(contact_data, "net_forces_w"):
        forces = contact_data.net_forces_w[:, env.feet_cfg.body_ids, :]  # [N, feet, 3]
        f_xy = torch.norm(forces[..., :2], dim=-1)                       # [N, feet]
        f_z = torch.abs(forces[..., 2])                                  # [N, feet]
    else:
        forces = contact_data.net_forces_w_history[:, :, env.feet_cfg.body_ids, :]  # [N, hist, feet, 3]
        f_xy = torch.norm(forces[..., :2], dim=-1).amax(dim=1)                    # [N, feet]
        f_z = torch.abs(forces[..., 2]).amax(dim=1)                               # [N, feet]

    stumble_per_foot = (
        (f_xy > horizontal_force_threshold)
        & (f_xy > ratio_threshold * (f_z + 1e-6))
    )

    return torch.any(stumble_per_foot, dim=1)


def safe_mean(values, default=float("nan")):
    if len(values) == 0:
        return default
    return float(np.mean(values))


def summarize(records):
    sr = safe_mean([r["success"] for r in records], default=0.0)

    # True MXD：所有 episode 的最大前进距离占目标距离的比例。
    # 每个 episode 的 r["mxd"] 已在 record_episode() 中计算为：
    # min(max_progress / target_distance, 1.0)
    # 因此这里直接对所有 episode 求均值，更接近 parkour-style evaluation 中的 MXD。
    mxd = safe_mean([r["mxd"] for r in records], default=0.0)

    # FailedMXD：只统计失败 episode，用于分析失败时策略最多推进到哪里。
    # 注意：这个不是主指标，只用于 failure analysis。
    failed_records = [r for r in records if r["success"] < 0.5]
    failed_mxd = safe_mean([r["mxd"] for r in failed_records], default=float("nan"))

    # 原始最大 x 位移，单位为 m。用于检查归一化 MXD 是否合理。
    mean_max_x_displacement = safe_mean([r["max_progress"] for r in records], default=0.0)
    failed_mean_max_x_displacement = safe_mean(
        [r["max_progress"] for r in failed_records],
        default=float("nan"),
    )

    successful_records = [r for r in records if r["success"] > 0.5]
    mean_reach_steps = safe_mean([r["reach_steps"] for r in successful_records])
    mean_reach_time = safe_mean([r["reach_time"] for r in successful_records])

    feet_stumble_step_rate = (
        np.sum([r["feet_stumble_steps"] for r in records])
        / max(1, np.sum([r["episode_steps"] for r in records]))
    )

    # BodyCollisionRate：发生过躯干/严重身体碰撞或 early termination 的 episode 比例。
    body_collision_rate = safe_mean([r["body_collision_episode"] for r in records], default=0.0)
    body_collision_step_rate = (
        np.sum([r["body_collision_steps"] for r in records])
        / max(1, np.sum([r["episode_steps"] for r in records]))
    )

    mean_reward = safe_mean([r["reward"] for r in records])
    mean_episode_length = safe_mean([r["episode_steps"] for r in records])

    return {
        "SR": sr,
        "MXD": mxd,
        "FailedMXD": failed_mxd,
        "MeanMaxXDisplacement": mean_max_x_displacement,
        "FailedMeanMaxXDisplacement": failed_mean_max_x_displacement,
        "MeanReachSteps": mean_reach_steps,
        "MeanReachTime": mean_reach_time,
        "FeetStumbleStepRate": feet_stumble_step_rate,
        "BodyCollisionRate": body_collision_rate,
        "BodyCollisionStepRate": body_collision_step_rate,
        "MeanReward": mean_reward,
        "MeanEpisodeLength": mean_episode_length,
    }


def write_csv(records, summary, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["summary"])
        for k, v in summary.items():
            writer.writerow([k, v])

        writer.writerow([])
        writer.writerow([
            "episode_id",
            "success",
            "mxd",
            "max_progress",
            "final_root_y",
            "body_collision_episode",
            "body_collision_steps",
            "feet_stumble_steps",
            "reach_steps",
            "reach_time",
            "episode_steps",
            "reward",
            "end_reason",
        ])

        for i, r in enumerate(records):
            writer.writerow([
                i,
                r["success"],
                r["mxd"],
                r["max_progress"],
                r["final_root_y"],
                r["body_collision_episode"],
                r["body_collision_steps"],
                r["feet_stumble_steps"],
                r["reach_steps"],
                r["reach_time"],
                r["episode_steps"],
                r["reward"],
                r["end_reason"],
            ])


def eval_policy():
    set_seed(args_cli.seed)

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg = configure_eval_env(env_cfg)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.seed = args_cli.seed
    env_cfg.scene.seed = args_cli.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(
        log_root_path,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
    )
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.get_observations()

    fixed_cmd = torch.tensor(
        [args_cli.vx, args_cli.vy, args_cli.wz],
        dtype=torch.float32,
        device=env.device,
    )
    env.command_generator.command[:] = fixed_cmd.unsqueeze(0)

    # target_distance 用于目标区域、MXD、ReachSteps/ReachTime
    if args_cli.success_distance is not None:
        target_distance = float(args_cli.success_distance)
    else:
        try:
            target_distance = float(env.scene.terrain.cfg.terrain_generator.size[0]) / 2.0
        except Exception:
            target_distance = max(1.0, args_cli.vx * env.max_episode_length_s * 0.8)

    print(f"[INFO] target_distance = {target_distance:.3f} m")
    print(f"[INFO] target_y_range = ({args_cli.success_y_min:.3f}, {args_cli.success_y_max:.3f}) m")
    print(f"[INFO] feet_stumble_ratio = {args_cli.feet_stumble_ratio:.3f}")
    print(f"[INFO] feet_stumble_force_threshold = {args_cli.feet_stumble_force_threshold:.3f} N")
    print(f"[INFO] num_envs = {env.num_envs}")
    print(f"[INFO] target eval episodes = {args_cli.num_eval_episodes}")
    print(f"[INFO] eval_depth_noise = {args_cli.eval_depth_noise}")
    print(f"[INFO] depth_noise_std = {args_cli.depth_noise_std}")
    print(f"[INFO] depth_failure_prob = {args_cli.depth_failure_prob}")

    # 每个并行环境当前 episode 的起点。用于解决 reset x 可能不是 0 的问题。
    episode_start_x = env.robot.data.root_pos_w[:, 0].clone()

    # 每个并行环境当前 episode 的统计量
    cur_max_progress = torch.zeros(env.num_envs, device=env.device)
    cur_body_collision_steps = torch.zeros(env.num_envs, device=env.device)
    cur_feet_stumble_steps = torch.zeros(env.num_envs, device=env.device)
    cur_episode_steps = torch.zeros(env.num_envs, device=env.device)
    cur_reward_sum = torch.zeros(env.num_envs, device=env.device)

    cur_reach_steps = torch.full((env.num_envs,), -1.0, device=env.device)

    step_dt = float(getattr(env, "step_dt", 0.02))

    records = []
    total_steps = 0

    def record_episode(env_id, success, end_reason):
        max_progress = float(cur_max_progress[env_id].detach().cpu())
        mxd = min(max_progress / max(target_distance, 1e-6), 1.0)

        final_root_y = float(get_local_root_y(env)[env_id].detach().cpu())

        reach_steps_value = float(cur_reach_steps[env_id].detach().cpu())
        if reach_steps_value < 0 or not success:
            reach_steps_value = float("nan")
            reach_time_value = float("nan")
        else:
            reach_time_value = reach_steps_value * step_dt

        records.append({
            "success": float(success),
            "mxd": float(mxd),
            "max_progress": max_progress,
            "final_root_y": final_root_y,
            "body_collision_episode": float(
                bool(cur_body_collision_steps[env_id].detach().cpu() > 0)
            ),
            "body_collision_steps": float(cur_body_collision_steps[env_id].detach().cpu()),
            "feet_stumble_steps": float(cur_feet_stumble_steps[env_id].detach().cpu()),
            "reach_steps": reach_steps_value,
            "reach_time": reach_time_value,
            "episode_steps": float(cur_episode_steps[env_id].detach().cpu()),
            "reward": float(cur_reward_sum[env_id].detach().cpu()),
            "end_reason": end_reason,
        })

    def reset_episode_buffers(env_ids):
        episode_start_x[env_ids] = env.robot.data.root_pos_w[env_ids, 0].clone()
        cur_max_progress[env_ids] = 0.0
        cur_body_collision_steps[env_ids] = 0.0
        cur_feet_stumble_steps[env_ids] = 0.0
        cur_episode_steps[env_ids] = 0.0
        cur_reward_sum[env_ids] = 0.0
        cur_reach_steps[env_ids] = -1.0

    while simulation_app.is_running() and len(records) < args_cli.num_eval_episodes:
        env.command_generator.command[:] = fixed_cmd.unsqueeze(0)

        with torch.inference_mode():
            actions = policy(obs)

            # step 之前记录当前状态，因为 done env 会在 env.step 内部自动 reset。
            progress_before = get_forward_progress_from_start(env, episode_start_x)
            root_y_before = get_local_root_y(env)

            cur_max_progress = torch.maximum(cur_max_progress, progress_before)

            body_collision = get_body_collision(env)
            feet_stumble = get_feet_stumble(
                env,
                ratio_threshold=args_cli.feet_stumble_ratio,
                horizontal_force_threshold=args_cli.feet_stumble_force_threshold,
            )
            cur_body_collision_steps += body_collision.float()
            cur_feet_stumble_steps += feet_stumble.float()
            cur_episode_steps += 1.0

            # 如果当前状态已经处于目标区域，就记录成功并 reset。
            # 目标区域：从 episode 起点向前达到 target_distance，且当前 root_y 在指定横向范围内。
            success_now = (
                (progress_before >= target_distance)
                & (root_y_before > args_cli.success_y_min)
                & (root_y_before < args_cli.success_y_max)
            )
            success_ids_before = success_now.nonzero(as_tuple=False).flatten()

            if success_ids_before.numel() > 0:
                for env_id in success_ids_before.tolist():
                    if len(records) >= args_cli.num_eval_episodes:
                        break
                    cur_reach_steps[env_id] = cur_episode_steps[env_id]
                    record_episode(env_id, success=True, end_reason="target_region_reached")

                env.reset(success_ids_before)
                env.command_generator.command[:] = fixed_cmd.unsqueeze(0)
                obs, _ = env.get_observations()
                reset_episode_buffers(success_ids_before)

                # reset 后重新计算 action，避免用旧 obs/action 控制新 episode。
                actions = policy(obs)

            obs, rewards, dones, infos = env.step(actions)
            cur_reward_sum += rewards

            dones = dones.bool()
            timeouts = infos.get("time_outs", env.time_out_buf).bool()
            not_done = ~dones

            # step 之后只更新未 done 的 env；done env 可能已经自动 reset，不能拿新位置更新旧 episode。
            if torch.any(not_done):
                progress_after = get_forward_progress_from_start(env, episode_start_x)
                root_y_after = get_local_root_y(env)

                cur_max_progress[not_done] = torch.maximum(
                    cur_max_progress[not_done],
                    progress_after[not_done],
                )

                success_after = (
                    (progress_after >= target_distance)
                    & (root_y_after > args_cli.success_y_min)
                    & (root_y_after < args_cli.success_y_max)
                    & not_done
                )
                success_ids_after = success_after.nonzero(as_tuple=False).flatten()

                if success_ids_after.numel() > 0:
                    for env_id in success_ids_after.tolist():
                        if len(records) >= args_cli.num_eval_episodes:
                            break
                        cur_reach_steps[env_id] = cur_episode_steps[env_id]
                        record_episode(env_id, success=True, end_reason="target_region_reached")

                    env.reset(success_ids_after)
                    env.command_generator.command[:] = fixed_cmd.unsqueeze(0)
                    obs, _ = env.get_observations()
                    reset_episode_buffers(success_ids_after)

            # 环境自己结束：没在终止前到达目标区域，记为失败。
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                for env_id in done_ids.tolist():
                    if len(records) >= args_cli.num_eval_episodes:
                        break

                    # early termination 多数对应摔倒/躯干碰撞，这里也计入严重身体碰撞 episode。
                    if not bool(timeouts[env_id].detach().cpu()):
                        cur_body_collision_steps[env_id] = torch.maximum(
                            cur_body_collision_steps[env_id],
                            torch.tensor(1.0, device=env.device),
                        )
                        end_reason = "early_termination_before_target"
                    else:
                        end_reason = "timeout_before_target"

                    record_episode(env_id, success=False, end_reason=end_reason)

                # env.step 后 done env 已经自动 reset，这里记录新 episode 的起点。
                reset_episode_buffers(done_ids)

        total_steps += 1

        if total_steps % 100 == 0:
            print(
                f"[EVAL] sim_steps={total_steps}, "
                f"episodes={len(records)}/{args_cli.num_eval_episodes}"
            )

    records = records[: args_cli.num_eval_episodes]
    summary = summarize(records)

    print("\n========== EVAL SUMMARY ==========")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    print("==================================\n")

    write_csv(records, summary, args_cli.csv)
    print(f"[INFO] Saved CSV to: {args_cli.csv}")


if __name__ == "__main__":
    eval_policy()
    simulation_app.close()