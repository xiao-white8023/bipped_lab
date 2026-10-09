"""Verify the frozen G1 control-side contracts."""

import struct

import numpy as np

import g1_control
from g1_control.contracts import (
    DEFAULT_Q_MOTOR,
    DEFAULT_Q_POLICY,
    POLICY_INPUT_DIM,
    POLICY_TO_MOTOR,
    motor_to_policy,
    policy_to_motor,
)
from g1_control.observation import ProprioHistory, build_proprio_frame, projected_gravity_wxyz
from g1_control.remote import decode_remote, map_axis_to_command_range, rising_buttons
from g1_control.transition import ActionTransition


def test_control_package_imports():
    assert g1_control.__doc__


def test_joint_mapping_is_round_trip_and_matches_default():
    policy = np.arange(23, dtype=np.float32)
    assert np.array_equal(motor_to_policy(policy_to_motor(policy)), policy)
    assert np.array_equal(DEFAULT_Q_MOTOR[POLICY_TO_MOTOR], DEFAULT_Q_POLICY)


def test_observation_contract_and_history_order():
    gravity = projected_gravity_wxyz(np.asarray([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(gravity, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    frame = build_proprio_frame(
        np.zeros(3), gravity, np.zeros(3), DEFAULT_Q_POLICY, np.zeros(23), np.zeros(23)
    )
    assert frame.shape == (78,)
    history = ProprioHistory()
    first = history.append(frame)
    assert first.shape == (780,)
    assert np.array_equal(first[:78], frame)
    assert POLICY_INPUT_DIM == 2120


def test_remote_layout_and_rising_edge():
    data = bytearray(40)
    struct.pack_into("<H", data, 2, (1 << 1) | (1 << 3))
    struct.pack_into("<f", data, 4, 0.25)
    struct.pack_into("<f", data, 8, -0.5)
    struct.pack_into("<f", data, 12, 0.75)
    struct.pack_into("<f", data, 20, -1.0)
    state = decode_remote(data)
    assert state.pressed_names == ("L1", "select")
    assert np.allclose([state.lx, state.rx, state.ry, state.ly], [0.25, -0.5, 0.75, -1.0])
    assert rising_buttons(1 << 1, state.keys) == ("select",)


def test_remote_axis_maps_to_asymmetric_real_robot_command_ranges():
    x_range = (-0.4, 0.8)
    assert map_axis_to_command_range(-1.0, x_range) == -0.4
    assert map_axis_to_command_range(-0.5, x_range) == -0.2
    assert map_axis_to_command_range(0.0, x_range) == 0.0
    assert map_axis_to_command_range(0.5, x_range) == 0.4
    assert map_axis_to_command_range(1.0, x_range) == 0.8

    symmetric = (-0.4, 0.4)
    assert map_axis_to_command_range(1.0, symmetric, axis_sign=-1.0) == -0.4
    assert map_axis_to_command_range(-1.0, symmetric, axis_sign=-1.0) == 0.4


def test_mode_change_blends_only_final_action():
    transition = ActionTransition(blend_steps=5)
    _, progress = transition.step(np.zeros(23, dtype=np.float32), False)
    assert progress == 1.0
    output, progress = transition.step(np.ones(23, dtype=np.float32), True)
    assert np.allclose(output, 0.2)
    assert progress == 0.2
    transition.set_executed(np.full(23, 0.1, dtype=np.float32))
    output, progress = transition.step(np.ones(23, dtype=np.float32), True)
    assert np.allclose(output, 0.28)
    assert progress == 0.2
