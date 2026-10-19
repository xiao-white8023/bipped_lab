"""History isolation at selector transition and episode boundaries."""

import pytest
import torch

from legged_lab.high_level.high_level_buffers import SelectorActionBuffer, SelectorHistoryBuffer


def test_history_first_frame_padding_and_partial_update_keep_stopped_rows():
    history = SelectorHistoryBuffer(3, 3, 2)
    assert history.buffer.shape == (3, 3, 2)
    assert not history.buffer.any()
    history.append(torch.tensor([[10., 11.], [20., 21.]]), [2, 0])
    torch.testing.assert_close(history.buffer[2], torch.tensor([[10., 11.]]).expand(3, 2))
    assert not history.buffer[1].any()
    stopped_row = history.buffer[2].clone()
    for value in (22., 24., 26., 28.):
        history.append(torch.tensor([[value, value + 1]]), [0])
    torch.testing.assert_close(history.buffer[0, :, 0], torch.tensor([24., 26., 28.]))
    torch.testing.assert_close(history.buffer[2], stopped_row)
    assert history.num_pushes.tolist() == [5, 0, 1]
    assert history.current_length.tolist() == [3, 0, 1]


def test_partial_reset_does_not_leak_episode_or_change_other_environments():
    history = SelectorHistoryBuffer(4, 2, 1)
    history.append(torch.tensor([[1.], [10.]]))
    history.append(torch.tensor([[2.], [20.]]))
    kept = history.buffer[1].clone()
    history.reset([0])
    assert history.num_pushes.tolist() == [0, 2]
    assert not history.buffer[0].any()
    torch.testing.assert_close(history.buffer[1], kept)
    history.append(torch.tensor([[100.]]), [0])
    assert (history.buffer[0] == 100.).all()
    torch.testing.assert_close(history.buffer[1], kept)
    history.reset()
    assert not history.buffer.any() and not history.num_pushes.any()


def test_actual_actions_are_zero_delay_with_six_slots_and_no_cross_episode_difference():
    actions = SelectorActionBuffer(5, 2)
    first = torch.stack((torch.ones(23), torch.ones(23) * 2))
    torch.testing.assert_close(actions.compute(first), first)
    assert actions._circular_buffer.buffer.shape == (2, 6, 23)
    difference = lambda: (actions._circular_buffer.buffer[:, -1] - actions._circular_buffer.buffer[:, -2]).square().sum(1)
    assert not difference().any()
    next_action = torch.ones(1, 23) * 3
    torch.testing.assert_close(actions.compute(next_action, [0]), next_action)
    torch.testing.assert_close(difference(), torch.tensor([92., 0.]))
    other_history = actions._circular_buffer.buffer[1].clone()
    actions.reset([0])
    actions.compute(torch.full((1, 23), -10.), [0])
    assert not difference().any()
    torch.testing.assert_close(actions._circular_buffer.buffer[1], other_history)
    assert actions.history_length == 5
    assert actions.min_time_lag == actions.max_time_lag == 0
    assert not actions.time_lags.any()
    actions.set_time_lag(0)
    actions.set_time_lag(torch.tensor([0]), [0])
    with pytest.raises(ValueError, match="disabled"):
        actions.set_time_lag(1)


def test_empty_updates_are_noops_and_history_does_not_retain_autograd():
    history = SelectorHistoryBuffer(1, 2, 3)
    history.append(torch.empty(0, 3), [])
    assert not history.num_pushes.any()
    history.reset([])
    history.append(torch.ones(2, 3, requires_grad=True))
    history.append(torch.ones(1, 3) * 2, [0])
    assert not history.buffer.requires_grad
    torch.testing.assert_close(history.buffer[:, 0], torch.tensor([[2., 2., 2.], [1., 1., 1.]]))
    actions = SelectorActionBuffer(0, 2, 3)
    assert actions.compute(torch.empty(0, 3), []).shape == (0, 3)


@pytest.mark.parametrize("ids", [[0, 0], [-1], [2], [0.5], [[0]], [True]])
def test_invalid_ids_do_not_mutate_history(ids):
    history = SelectorHistoryBuffer(3, 2, 2)
    before = history.buffer.clone()
    with pytest.raises(ValueError, match="Environment IDs"):
        history.append(torch.ones(1, 2), ids)
    torch.testing.assert_close(history.buffer, before)
    assert not history.num_pushes.any()


@pytest.mark.parametrize("data", [torch.ones(2, 3), torch.ones(1, 2, 1), torch.ones(1, 2, dtype=torch.long)])
def test_invalid_data_does_not_mutate_history(data):
    history = SelectorHistoryBuffer(3, 2, 2)
    with pytest.raises(ValueError):
        history.append(data, [0])
    assert not history.buffer.any() and not history.num_pushes.any()


@pytest.mark.parametrize("max_len,batch_size,feature_dim", [(0, 2, 3), (2, 0, 3), (2, 2, 0), (True, 2, 3)])
def test_history_requires_positive_integer_dimensions(max_len, batch_size, feature_dim):
    with pytest.raises(ValueError):
        SelectorHistoryBuffer(max_len, batch_size, feature_dim)


def test_device_mismatch_is_rejected_before_update():
    history = SelectorHistoryBuffer(2, 1, 2)
    with pytest.raises(ValueError, match="device"):
        history.append(torch.ones(1, 2, device="meta"))
    assert not history.num_pushes.any()
