from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.replay_joint import load_joint_sequence, replay_sequence


def test_load_joint_sequence_accepts_saved_side(tmp_path):
    path = tmp_path / "joints.npz"
    expected = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.03]], dtype=np.float32)
    np.savez_compressed(path, right=expected)

    sequence = load_joint_sequence(path, "right")

    np.testing.assert_allclose(sequence, expected)


def test_load_joint_sequence_rejects_missing_side(tmp_path):
    path = tmp_path / "joints.npz"
    np.savez_compressed(path, left=np.zeros((1, 7), dtype=np.float32))

    try:
        load_joint_sequence(path, "right")
    except ValueError as exc:
        assert "does not contain side" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_joint_sequence_rejects_wrong_width(tmp_path):
    path = tmp_path / "joints.npz"
    np.savez_compressed(path, right=np.zeros((1, 6), dtype=np.float32))

    try:
        load_joint_sequence(path, "right")
    except ValueError as exc:
        assert "shape (N, 7)" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_replay_sequence_splits_arm_joints_and_gripper_width():
    class FakeArm:
        def __init__(self):
            self.calls = []

        def execute_joints(self, joints):
            self.calls.append(np.asarray(joints, dtype=np.float32).copy())

    class FakeGripper:
        def __init__(self):
            self.calls = []

        def execute_width(self, width):
            self.calls.append(float(width))

    args = argparse.Namespace(fps=1000.0, loop=False, dry_run=False, side="right")
    arm = FakeArm()
    gripper = FakeGripper()
    sequence = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.03],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.04],
        ],
        dtype=np.float32,
    )

    replay_sequence(args, arm, gripper, sequence)

    assert len(arm.calls) == 2
    np.testing.assert_allclose(arm.calls[0], sequence[0, :6])
    np.testing.assert_allclose(arm.calls[1], sequence[1, :6])
    np.testing.assert_allclose(gripper.calls, [0.03, 0.04])
