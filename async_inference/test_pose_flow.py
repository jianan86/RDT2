from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from async_inference.pose_utils import (
    IDENTITY_POSE10D,
    ee_pose14_to_tcp_pose14,
    relative_action_to_absolute_tcp,
    tcp_pose14_to_ee_pose14,
)
from async_inference.real_piper_client import ActionQueue, preprocess_fisheye


def test_ee_tcp_roundtrip_pose14():
    ee = np.array([
        0.3, -0.2, 0.4, 0.1, -0.2, 0.3, 0.04,
        -0.1, 0.2, 0.5, -0.3, 0.2, -0.1, 0.05,
    ], dtype=np.float32)

    tcp = ee_pose14_to_tcp_pose14(ee)
    recovered = tcp_pose14_to_ee_pose14(tcp)

    np.testing.assert_allclose(recovered, ee, atol=1e-5)


def test_identity_relative_action_returns_same_tcp_pose():
    tcp = np.array([
        0.1, 0.2, 0.3, 0.2, -0.1, 0.4, 0.03,
        -0.3, 0.1, 0.2, -0.2, 0.3, -0.4, 0.04,
    ], dtype=np.float32)
    raw = np.zeros((2, 20), dtype=np.float32)
    raw[:, 0:9] = IDENTITY_POSE10D
    raw[:, 9] = 0.03
    raw[:, 10:19] = IDENTITY_POSE10D
    raw[:, 19] = 0.04

    action = relative_action_to_absolute_tcp(raw, tcp)

    np.testing.assert_allclose(action, np.repeat(tcp[None, :], 2, axis=0), atol=1e-5)


def test_preprocess_fisheye_pads_640x480_to_384_square():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    out = preprocess_fisheye(image, 384)

    assert out.shape == (384, 384, 3)
    assert out.dtype == np.uint8
    assert out[0].max() == 0
    assert out[-1].max() == 0
    assert out[192, 192].min() == 255


def test_action_queue_drops_expired_and_blends_overlap_by_timestep():
    old = np.zeros((4, 14), dtype=np.float32)
    old[:, 0] = 1.0
    new = np.zeros((4, 14), dtype=np.float32)
    new[:, 0] = 3.0

    queue = ActionQueue()
    queue.add_chunk(first_timestep=10, action=old, latest_executed_timestep=9)
    queue.add_chunk(first_timestep=12, action=new, latest_executed_timestep=11)

    first = queue.pop_next(latest_executed_timestep=11)
    assert first is not None
    timestep, action = first
    assert timestep == 12
    assert 1.0 < action[0] < 3.0

    second = queue.pop_next(latest_executed_timestep=12)
    assert second is not None
    assert second[0] == 13
