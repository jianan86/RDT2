from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from async_inference.pose_utils import euler_xyz_to_matrix
from data.replay_pika import (
    gripper_width_to_joint_pair,
    normalize_args,
    target_tcp_from_capture_delta,
    target_tcp_sequence,
)


def make_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = euler_xyz_to_matrix(np.asarray(rpy, dtype=np.float32))
    mat[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return mat


def test_target_tcp_from_capture_delta_keeps_base_for_first_frame():
    replay_base = make_transform((0.4, -0.2, 0.3), (0.1, 0.2, -0.1))
    capture_first = make_transform((1.0, 2.0, 3.0), (-0.2, 0.1, 0.3))

    target = target_tcp_from_capture_delta(replay_base, capture_first, capture_first)

    np.testing.assert_allclose(target, replay_base, atol=1e-6)


def test_target_tcp_from_capture_delta_applies_relative_motion_on_replay_base():
    replay_base = make_transform((0.4, -0.2, 0.3), (0.0, 0.0, 0.2))
    capture_first = make_transform((1.0, 2.0, 3.0), (0.0, 0.0, -0.3))
    capture_next = capture_first @ make_transform((0.1, -0.02, 0.03), (0.0, 0.0, 0.15))

    target = target_tcp_from_capture_delta(replay_base, capture_first, capture_next)
    expected = replay_base @ np.linalg.inv(capture_first) @ capture_next

    np.testing.assert_allclose(target, expected, atol=1e-6)


def test_target_tcp_sequence_handles_empty_input():
    replay_base = np.eye(4, dtype=np.float64)

    sequence = target_tcp_sequence(replay_base, [])

    assert sequence.shape == (0, 4, 4)


def test_gripper_width_to_joint_pair_uses_symmetric_prismatic_joints():
    joints = gripper_width_to_joint_pair(0.08)

    np.testing.assert_allclose(joints, [0.04, -0.04])


def test_normalize_args_treats_non_positive_limits_as_unbounded():
    args = argparse.Namespace(max_frames=0, max_episodes=-1)

    normalized = normalize_args(args)

    assert normalized.max_frames is None
    assert normalized.max_episodes is None
