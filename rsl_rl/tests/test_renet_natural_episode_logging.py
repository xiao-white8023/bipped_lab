from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runners.renet_amp_on_policy_runner import RENetAmpOnPolicyRunner


NATURAL_MASK = torch.tensor([True, True, True, True, False, False])


def _bookkeep(
    dones,
    rewards,
    lengths,
    *,
    natural_mask=NATURAL_MASK,
    initial_rewards=(),
    initial_lengths=(),
):
    cur_reward_sum = torch.tensor(rewards, dtype=torch.float)
    cur_episode_length = torch.tensor(lengths, dtype=torch.float)
    rewbuffer = deque(initial_rewards, maxlen=100)
    lenbuffer = deque(initial_lengths, maxlen=100)
    done_ids, natural_done_ids = (
        RENetAmpOnPolicyRunner._update_completed_episode_logging(
            torch.tensor(dones, dtype=torch.bool),
            natural_mask,
            cur_reward_sum,
            cur_episode_length,
            rewbuffer,
            lenbuffer,
        )
    )
    return {
        "done_ids": done_ids,
        "natural_done_ids": natural_done_ids,
        "cur_reward_sum": cur_reward_sum,
        "cur_episode_length": cur_episode_length,
        "rewbuffer": rewbuffer,
        "lenbuffer": lenbuffer,
    }


def test_mixed_done_records_only_natural_and_clears_every_done_accumulator():
    result = _bookkeep(
        [[True], [False], [False], [False], [True], [True]],
        [20, 11, 12, 13, 0, 0],
        [1000, 200, 200, 200, 150, 250],
    )

    assert result["done_ids"].tolist() == [0, 4, 5]
    assert result["done_ids"].dtype == torch.long
    assert result["done_ids"].ndim == 1
    assert result["natural_done_ids"].tolist() == [0]
    assert list(result["rewbuffer"]) == [20.0]
    assert list(result["lenbuffer"]) == [1000.0]
    torch.testing.assert_close(
        result["cur_reward_sum"],
        torch.tensor([0.0, 11.0, 12.0, 13.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        result["cur_episode_length"],
        torch.tensor([0.0, 200.0, 200.0, 200.0, 0.0, 0.0]),
    )


def test_dedicated_only_done_does_not_change_natural_history_but_clears_rows():
    result = _bookkeep(
        [False, False, False, False, True, True],
        [10, 11, 12, 13, 3, 4],
        [100, 200, 300, 400, 150, 250],
        initial_rewards=[99.0],
        initial_lengths=[999.0],
    )

    assert result["natural_done_ids"].numel() == 0
    assert list(result["rewbuffer"]) == [99.0]
    assert list(result["lenbuffer"]) == [999.0]
    assert result["cur_reward_sum"][4:].tolist() == [0.0, 0.0]
    assert result["cur_episode_length"][4:].tolist() == [0.0, 0.0]


def test_natural_only_done_records_all_completed_natural_episodes():
    result = _bookkeep(
        [True, False, True, False, False, False],
        [20, 11, 30, 13, 0, 0],
        [1000, 200, 1200, 200, 150, 250],
    )

    assert result["done_ids"].tolist() == [0, 2]
    assert result["natural_done_ids"].tolist() == [0, 2]
    assert list(result["rewbuffer"]) == [20.0, 30.0]
    assert list(result["lenbuffer"]) == [1000.0, 1200.0]


def test_no_done_preserves_buffers_and_accumulators():
    rewards = [10, 11, 12, 13, 3, 4]
    lengths = [100, 200, 300, 400, 150, 250]
    result = _bookkeep(
        [False] * 6,
        rewards,
        lengths,
        initial_rewards=[99.0],
        initial_lengths=[999.0],
    )

    assert result["done_ids"].numel() == 0
    assert result["natural_done_ids"].numel() == 0
    assert list(result["rewbuffer"]) == [99.0]
    assert list(result["lenbuffer"]) == [999.0]
    assert result["cur_reward_sum"].tolist() == rewards
    assert result["cur_episode_length"].tolist() == lengths


def test_all_natural_baseline_records_every_completed_episode():
    result = _bookkeep(
        [True, False, True, False, True, False],
        [20, 11, 30, 13, 40, 15],
        [1000, 200, 1200, 200, 900, 250],
        natural_mask=torch.ones(6, dtype=torch.bool),
    )

    assert result["done_ids"].tolist() == [0, 2, 4]
    assert result["natural_done_ids"].tolist() == [0, 2, 4]
    assert list(result["rewbuffer"]) == [20.0, 30.0, 40.0]
    assert list(result["lenbuffer"]) == [1000.0, 1200.0, 900.0]


def test_rnd_episode_buffers_are_natural_only_and_all_done_rows_are_cleared():
    cur_reward_sum = torch.tensor([20, 11, 12, 13, 4, 5], dtype=torch.float)
    cur_episode_length = torch.tensor([1000, 200, 200, 200, 150, 250], dtype=torch.float)
    cur_ereward_sum = torch.tensor([18, 10, 11, 12, 3, 4], dtype=torch.float)
    cur_ireward_sum = torch.tensor([2, 1, 1, 1, 1, 1], dtype=torch.float)
    rewbuffer, lenbuffer, erewbuffer, irewbuffer = (
        deque(maxlen=100),
        deque(maxlen=100),
        deque(maxlen=100),
        deque(maxlen=100),
    )

    RENetAmpOnPolicyRunner._update_completed_episode_logging(
        torch.tensor([True, False, False, False, True, True]),
        NATURAL_MASK,
        cur_reward_sum,
        cur_episode_length,
        rewbuffer,
        lenbuffer,
        cur_ereward_sum=cur_ereward_sum,
        cur_ireward_sum=cur_ireward_sum,
        erewbuffer=erewbuffer,
        irewbuffer=irewbuffer,
    )

    assert list(erewbuffer) == [18.0]
    assert list(irewbuffer) == [2.0]
    assert cur_ereward_sum[[0, 4, 5]].tolist() == [0.0, 0.0, 0.0]
    assert cur_ireward_sum[[0, 4, 5]].tolist() == [0.0, 0.0, 0.0]


def test_natural_mask_resolution_validates_and_has_warned_all_env_fallback():
    env = SimpleNamespace(num_envs=3)
    with pytest.warns(
        UserWarning,
        match="episode reward/length logging falls back to all environments",
    ):
        fallback = RENetAmpOnPolicyRunner._resolve_natural_env_mask_for_logging(
            env,
            "cpu",
        )
    assert torch.equal(fallback, torch.ones(3, dtype=torch.bool))

    env.natural_env_mask = torch.ones(3)
    with pytest.raises(TypeError, match="dtype bool"):
        RENetAmpOnPolicyRunner._resolve_natural_env_mask_for_logging(env, "cpu")

    env.natural_env_mask = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="must have shape"):
        RENetAmpOnPolicyRunner._resolve_natural_env_mask_for_logging(env, "cpu")


class _CaptureWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, key, value, step):
        self.scalars[key] = (value, step)


def test_train_aliases_match_legacy_natural_metrics_and_terminal_names(capsys):
    runner = RENetAmpOnPolicyRunner.__new__(RENetAmpOnPolicyRunner)
    runner.num_steps_per_env = 1
    runner.env = SimpleNamespace(num_envs=1)
    runner.gpu_world_size = 1
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    runner.device = "cpu"
    runner.logger_type = "tensorboard"
    runner.writer = _CaptureWriter()
    runner.alg = SimpleNamespace(
        rnd=False,
        policy=SimpleNamespace(action_std=torch.ones(1)),
        learning_rate=1.0e-3,
    )
    locs = {
        "collection_time": 1.0,
        "learn_time": 1.0,
        "ep_infos": [],
        "loss_dict": {},
        "rewbuffer": deque([20.0, 30.0], maxlen=100),
        "lenbuffer": deque([1000.0, 1200.0], maxlen=100),
        "it": 0,
        "tot_iter": 1,
        "start_iter": 0,
        "num_learning_iterations": 1,
    }

    runner.log(locs)

    scalars = runner.writer.scalars
    assert scalars["Train/mean_reward"] == (25.0, 0)
    assert scalars["Train/NaturalMeanLocomotionPpoReturn"] == (25.0, 0)
    assert scalars["Train/mean_episode_length"] == (1100.0, 0)
    assert scalars["Train/NaturalMeanWallEpisodeLength"] == (1100.0, 0)
    terminal = capsys.readouterr().out
    assert "Mean natural locomotion return:" in terminal
    assert "Mean natural episode length:" in terminal
