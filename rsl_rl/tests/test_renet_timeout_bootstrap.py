from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from rsl_rl.algorithms.renet_amp_ppo import RENetAMPPPO
from rsl_rl.runners.renet_amp_on_policy_runner import RENetAmpOnPolicyRunner
from rsl_rl.storage import RolloutStorage


def _load_renet_env_methods(*method_names):
    """Load pure G1RENetEnv methods without importing the Isaac Sim runtime."""
    env_path = Path(__file__).parents[2] / "legged_lab" / "envs" / "g1" / "RENet_env.py"
    tree = ast.parse(env_path.read_text(encoding="utf-8"), filename=str(env_path))
    env_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv"
    )
    methods = []
    for method_name in method_names:
        method = next(
            node
            for node in env_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
        )
        method.decorator_list = []
        methods.append(method)
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(env_path), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def test_terminal_critic_builder_virtual_append_has_no_history_side_effects():
    virtual_append, build_terminal_obs = _load_renet_env_methods(
        "_virtual_append_critic_history",
        "build_terminal_critic_obs",
    )
    critic_history = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )
    current_terminal_frame = torch.tensor([[12.0, 13.0], [14.0, 15.0]])
    actor_history = torch.randn(2, 3, 4)
    depth_history = torch.randn(2, 2, 5)
    critic_before = critic_history.clone()
    actor_before = actor_history.clone()
    depth_before = depth_history.clone()

    fake_env = SimpleNamespace(
        num_envs=2,
        clip_obs=100.0,
        cfg=SimpleNamespace(
            scene=SimpleNamespace(height_scanner=SimpleNamespace(enable_height_scan=True))
        ),
        critic_obs_buffer=SimpleNamespace(buffer=critic_history),
        actor_obs_buffer=SimpleNamespace(buffer=actor_history),
        depth_buffer=depth_history,
        compute_current_observations=lambda: (torch.empty(2, 0), current_terminal_frame),
        _virtual_append_critic_history=virtual_append,
        _build_current_critic_height_scan=lambda: torch.tensor([[16.0], [17.0]]),
    )

    terminal_obs = build_terminal_obs(fake_env, torch.tensor([0, 1], dtype=torch.long))

    expected_history = torch.cat(
        [critic_before[:, 1:], current_terminal_frame.unsqueeze(1)], dim=1
    )
    expected_obs = torch.cat(
        [expected_history.reshape(2, -1), torch.tensor([[16.0], [17.0]])], dim=1
    )
    torch.testing.assert_close(terminal_obs, expected_obs)
    torch.testing.assert_close(critic_history, critic_before)
    torch.testing.assert_close(actor_history, actor_before)
    torch.testing.assert_close(depth_history, depth_before)
    assert terminal_obs.shape == (2, 7)
    assert torch.isfinite(terminal_obs).all()


def test_timeout_uses_terminal_value_and_stops_gae_trace():
    rewards = torch.tensor([[[1.0]], [[100.0]]])
    values = torch.tensor([[[2.0]], [[0.0]]])
    last_values = torch.zeros(1, 1)
    active = torch.ones(2, 1, 1, dtype=torch.bool)
    trace_end = torch.zeros_like(active)
    env_terminal = torch.zeros_like(active)
    time_outs = torch.tensor([[[True]], [[False]]])
    terminal_values = torch.tensor([[[5.0]], [[0.0]]])

    _, advantages = RolloutStorage.compute_segmented_gae(
        rewards=rewards,
        values=values,
        last_values=last_values,
        sample_mask=active,
        trace_end=trace_end,
        env_terminal=env_terminal,
        time_outs=time_outs,
        gamma=0.99,
        lam=0.95,
        timeout_bootstrap_values=terminal_values,
        normalize_advantage=False,
    )

    torch.testing.assert_close(advantages[0], torch.tensor([[3.95]]))


def test_runner_routes_terminal_values_by_action_time_mode_and_reset_row_alignment():
    class FakePolicy:
        @staticmethod
        def evaluate(observations):
            return observations.sum(dim=1, keepdim=True)

    class FakeRecoveryCritic:
        @staticmethod
        def __call__(observations):
            base = observations.sum(dim=1, keepdim=True)
            return {"task": base + 10.0, "amp": base + 20.0, "reg": base + 30.0}

    runner = RENetAmpOnPolicyRunner.__new__(RENetAmpOnPolicyRunner)
    runner.env = SimpleNamespace(num_envs=4)
    runner.device = "cpu"
    runner.num_privileged_obs = 2
    runner.privileged_obs_type = "critic"
    runner.empirical_normalization = False
    runner.privileged_obs_normalizer = torch.nn.Identity()
    runner.alg = SimpleNamespace(policy=FakePolicy(), recovery_critic=FakeRecoveryCritic())

    reset_env_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    recovery_mask_t = torch.tensor([False, False, False, True])
    infos = {
        "observations": {"critic": torch.zeros(4, 2)},
        # Rows align with reset_env_ids: env 1 is locomotion, env 2 is a
        # failure (no bootstrap), and env 3 is Recovery.
        "terminal_critic_obs": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "time_outs": torch.tensor([False, True, True, True]),
        "recovery_failed": torch.tensor([False, False, True, False]),
    }
    values = runner._compute_timeout_bootstrap_values(
        infos,
        reset_env_ids,
        recovery_mask_t,
        torch.zeros(4, 2),
    )

    torch.testing.assert_close(
        values["timeout_loco_values"].squeeze(1), torch.tensor([0.0, 3.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        values["timeout_rec_task_values"].squeeze(1), torch.tensor([0.0, 0.0, 0.0, 21.0])
    )
    torch.testing.assert_close(
        values["timeout_rec_amp_values"].squeeze(1), torch.tensor([0.0, 0.0, 0.0, 31.0])
    )
    torch.testing.assert_close(
        values["timeout_rec_reg_values"].squeeze(1), torch.tensor([0.0, 0.0, 0.0, 41.0])
    )


def test_true_termination_overrides_timeout_bootstrap():
    _, advantages = RolloutStorage.compute_segmented_gae(
        rewards=torch.tensor([[[1.0]]]),
        values=torch.tensor([[[2.0]]]),
        last_values=torch.zeros(1, 1),
        sample_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        trace_end=torch.zeros(1, 1, 1, dtype=torch.bool),
        env_terminal=torch.ones(1, 1, 1, dtype=torch.bool),
        time_outs=torch.ones(1, 1, 1, dtype=torch.bool),
        gamma=0.99,
        lam=0.95,
        timeout_bootstrap_values=torch.tensor([[[5.0]]]),
        normalize_advantage=False,
    )

    torch.testing.assert_close(advantages[0], torch.tensor([[-1.0]]))


def test_timeout_storage_fields_default_to_zero():
    storage = RolloutStorage(
        training_type="rl",
        num_envs=2,
        num_transitions_per_env=3,
        obs_shape=[4],
        privileged_obs_shape=[5],
        actions_shape=[2],
        device="cpu",
    )

    for field_name in (
        "timeout_loco_values",
        "timeout_rec_task_values",
        "timeout_rec_amp_values",
        "timeout_rec_reg_values",
    ):
        field = getattr(storage, field_name)
        assert field.shape == (3, 2, 1)
        assert torch.count_nonzero(field) == 0


def test_timeout_reward_correction_is_single_and_baseline_path_is_unchanged():
    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    algorithm.gamma = 0.99
    algorithm.transition = SimpleNamespace(
        rewards=torch.tensor([1.0, 1.0]),
        values=torch.tensor([[2.0], [2.0]]),
        time_outs=torch.tensor([True, True]),
        recovery_mask_t=torch.tensor([False, True]),
        recovery_failed=torch.tensor([False, False]),
        timeout_loco_value=torch.tensor([[5.0], [0.0]]),
    )

    # Current recovery-enabled/non-segmented phase: locomotion gets exactly
    # one correct terminal-value reward correction; Recovery gets none.
    algorithm.recovery_state_machine_enabled = True
    algorithm.enable_recovery_learning = False
    algorithm._apply_timeout_bootstrap()
    torch.testing.assert_close(algorithm.transition.rewards, torch.tensor([5.95, 1.0]))

    # Future segmented phase: reward remains untouched so segmented GAE is the
    # only place that applies terminal bootstrap.
    algorithm.transition.rewards.fill_(1.0)
    algorithm.enable_recovery_learning = True
    algorithm._apply_timeout_bootstrap()
    torch.testing.assert_close(algorithm.transition.rewards, torch.tensor([1.0, 1.0]))

    # Recovery-disabled baseline retains the historical V(s_t) correction.
    algorithm.transition.rewards.fill_(1.0)
    algorithm.recovery_state_machine_enabled = False
    algorithm.enable_recovery_learning = False
    algorithm._apply_timeout_bootstrap()
    torch.testing.assert_close(algorithm.transition.rewards, torch.tensor([2.98, 2.98]))
