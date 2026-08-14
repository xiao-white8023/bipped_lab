from __future__ import annotations

import torch
import torch.nn.functional as F

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.renet_amp_ppo import RENetAMPPPO
from rsl_rl.storage import RolloutStorage


def test_recovery_mask_reaches_renet_minibatch_and_builds_action_time_loco_mask():
    storage = RolloutStorage(
        training_type="rl",
        num_envs=4,
        num_transitions_per_env=1,
        obs_shape=[2],
        privileged_obs_shape=[2],
        actions_shape=[1],
        device="cpu",
    )
    storage.recovery_masks[0, :, 0] = torch.tensor([False, True, False, True])
    batch = next(storage.mini_batch_generator(1, 1, include_recovery_data=True))
    rollout_data = batch[-1]

    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    loco_mask = algorithm._get_auxiliary_sample_mask(rollout_data, batch[0])

    assert loco_mask.shape == (4,)
    assert loco_mask.dtype == torch.bool
    assert loco_mask.device == batch[0].device
    assert torch.count_nonzero(loco_mask) == 2
    assert torch.equal(loco_mask, ~rollout_data["recovery_mask_t"].squeeze(-1))


def test_mixed_batch_op_vp_and_feet_losses_ignore_recovery_rows():
    loco_mask = torch.tensor([True, False, True, False])
    target = torch.tensor([[1.0, 2.0], [20.0, 30.0], [3.0, 4.0], [40.0, 50.0]])
    predictions = {
        "op": torch.tensor([[0.0, 2.0], [9.0, 9.0], [4.0, 2.0], [8.0, 8.0]]),
        "vp": torch.tensor([[2.0, 1.0], [7.0, 7.0], [2.0, 5.0], [6.0, 6.0]]),
        "feet": torch.tensor([[1.5, 1.5], [5.0, 5.0], [2.5, 4.5], [4.0, 4.0]]),
    }

    for prediction in predictions.values():
        expected = F.mse_loss(prediction[loco_mask], target[loco_mask])
        original_loss = AMPPPO._masked_auxiliary_mse(prediction, target, loco_mask)

        changed_prediction = prediction.clone()
        changed_target = target.clone()
        changed_prediction[~loco_mask] = 1.0e6
        changed_target[~loco_mask] = -1.0e6
        changed_loss = AMPPPO._masked_auxiliary_mse(
            changed_prediction, changed_target, loco_mask
        )

        torch.testing.assert_close(original_loss, expected)
        torch.testing.assert_close(changed_loss, expected)


def test_terrain_normalization_uses_only_normal_targets():
    loco_mask = torch.tensor([True, False, True, False])
    prediction = torch.tensor([[0.1, -0.2], [9.0, 9.0], [0.3, 0.4], [8.0, 8.0]])
    target = torch.tensor([[1.0, 2.0], [1000.0, 1000.0], [2.0, 3.0], [-1000.0, -1000.0]])

    normal_target = target[loco_mask]
    normalized_normal_target = (
        normal_target - normal_target.mean()
    ) / normal_target.std().clamp(min=1e-6)
    expected = F.mse_loss(prediction[loco_mask], normalized_normal_target)
    loss = AMPPPO._masked_terrain_reconstruction_loss(prediction, target, loco_mask)

    changed_prediction = prediction.clone()
    changed_target = target.clone()
    changed_prediction[~loco_mask] *= 1.0e5
    changed_target[~loco_mask] *= 1.0e5
    changed_loss = AMPPPO._masked_terrain_reconstruction_loss(
        changed_prediction, changed_target, loco_mask
    )

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(changed_loss, expected)


def test_all_normal_matches_legacy_formulas_and_plain_amp_uses_full_batch():
    prediction = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    target = torch.tensor([[1.0, 1.0], [1.0, 4.0], [5.0, 3.0], [8.0, 6.0]])
    all_normal = torch.ones(4, dtype=torch.bool)

    torch.testing.assert_close(
        AMPPPO._masked_auxiliary_mse(prediction, target, all_normal),
        F.mse_loss(prediction, target),
    )
    legacy_target_norm = (target - target.mean()) / target.std().clamp(min=1e-6)
    torch.testing.assert_close(
        AMPPPO._masked_terrain_reconstruction_loss(prediction, target, all_normal),
        F.mse_loss(prediction, legacy_target_norm),
    )

    plain_amp = AMPPPO.__new__(AMPPPO)
    assert plain_amp._get_auxiliary_sample_mask(None, prediction) is None


def test_all_recovery_auxiliary_losses_are_finite_autograd_safe_zeros():
    prediction = torch.randn(4, 3, requires_grad=True)
    target = torch.randn(4, 3)
    no_locomotion = torch.zeros(4, dtype=torch.bool)

    op_loss = AMPPPO._masked_auxiliary_mse(prediction, target, no_locomotion)
    vp_loss = AMPPPO._masked_auxiliary_mse(prediction, target, no_locomotion)
    terrain_loss = AMPPPO._masked_terrain_reconstruction_loss(
        prediction, target, no_locomotion
    )
    feet_loss = AMPPPO._masked_auxiliary_mse(prediction, target, no_locomotion)
    total_loss = op_loss + vp_loss + terrain_loss + feet_loss

    assert torch.isfinite(total_loss)
    torch.testing.assert_close(total_loss, torch.zeros_like(total_loss))
    total_loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0
