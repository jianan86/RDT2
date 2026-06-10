from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from async_inference.debug_trace import StopDetector, collect_sdk_snapshot, summarize_action
from async_inference.pose_utils import (
    IDENTITY_POSE10D,
    ee_pose14_to_tcp_pose14,
    relative_action_to_absolute_tcp,
    tcp_pose14_to_ee_pose14,
)
from async_inference.proto import rdt2_async_pb2
from async_inference.real_piper_client import (
    ActionQueue,
    adapt_action_for_arm_mode,
    duplicate_single_arm_pose14,
    preprocess_fisheye,
)
from async_inference.server import RDT2AsyncService


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


def test_duplicate_single_arm_pose14_copies_right_pose_to_both_slots():
    pose7 = np.arange(7, dtype=np.float32)

    pose14 = duplicate_single_arm_pose14(pose7)

    assert pose14.shape == (14,)
    np.testing.assert_allclose(pose14[:7], pose7)
    np.testing.assert_allclose(pose14[7:], pose7)


def test_adapt_action_for_single_arm_copies_right_action_to_left_slot():
    action = np.arange(28, dtype=np.float32).reshape(2, 14)

    adapted = adapt_action_for_arm_mode(action, "single")

    assert adapted.shape == (2, 14)
    np.testing.assert_allclose(adapted[:, :7], action[:, :7])
    np.testing.assert_allclose(adapted[:, 7:], action[:, :7])


def test_adapt_action_for_dual_arm_leaves_action_unchanged():
    action = np.arange(28, dtype=np.float32).reshape(2, 14)

    adapted = adapt_action_for_arm_mode(action, "dual")

    assert adapted is action


def test_action_queue_prefix_no_merge_requires_positive_prefix_steps():
    try:
        ActionQueue(
            chunk_merge_strategy=ActionQueue.PREFIX_NO_MERGE,
            chunk_execute_prefix_steps=0,
        )
    except ValueError as exc:
        assert "chunk_execute_prefix_steps" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_action_queue_prefix_no_merge_filters_stale_and_takes_first_k():
    old = np.zeros((4, 14), dtype=np.float32)
    old[:, 0] = 1.0
    new = np.zeros((4, 14), dtype=np.float32)
    new[:, 0] = np.arange(4, dtype=np.float32) + 30.0

    queue = ActionQueue(
        chunk_merge_strategy=ActionQueue.PREFIX_NO_MERGE,
        chunk_execute_prefix_steps=2,
    )
    queue.add_chunk(chunk_latest_action=9, action=old, current_latest_action=9)
    queue.add_chunk(chunk_latest_action=12, action=new, current_latest_action=12)

    first = queue.pop_next(latest_action=12)
    assert first is not None
    timestep, action = first
    assert timestep == 13
    assert action[0] == 30.0

    second = queue.pop_next(latest_action=13)
    assert second is not None
    timestep, action = second
    assert timestep == 14
    assert action[0] == 31.0

    assert queue.pop_next(latest_action=14) is None


def test_action_queue_default_prefix_no_merge_takes_first_8_future_steps():
    action = np.zeros((12, 14), dtype=np.float32)
    action[:, 0] = np.arange(12, dtype=np.float32)

    queue = ActionQueue()
    queue.add_chunk(chunk_latest_action=9, action=action, current_latest_action=9)

    popped = [queue.pop_next(latest_action=9 + i) for i in range(8)]

    assert len(queue) == 0
    assert [item[0] for item in popped if item is not None] == list(range(10, 18))
    assert [item[1][0] for item in popped if item is not None] == list(range(8))


def test_action_queue_derives_steps_from_latest_action():
    action = np.zeros((4, 14), dtype=np.float32)
    action[:, 0] = np.arange(4, dtype=np.float32)

    queue = ActionQueue(chunk_merge_strategy=ActionQueue.LATEST_ONLY)
    queue.add_chunk(chunk_latest_action=-1, action=action, current_latest_action=-1)

    first = queue.pop_next(latest_action=-1)
    assert first is not None
    assert first[0] == 0
    assert first[1][0] == 0.0


def test_action_queue_drops_expired_and_blends_overlap_by_timestep():
    old = np.zeros((4, 14), dtype=np.float32)
    old[:, 0] = np.arange(4, dtype=np.float32) + 10.0
    new = np.zeros((4, 14), dtype=np.float32)
    new[:, 0] = np.arange(4, dtype=np.float32) + 30.0

    queue = ActionQueue(chunk_merge_strategy=ActionQueue.BLEND_OVERLAP)
    queue.add_chunk(chunk_latest_action=9, action=old, current_latest_action=9)
    queue.add_chunk(chunk_latest_action=11, action=new, current_latest_action=11)

    first = queue.pop_next(latest_action=11)
    assert first is not None
    timestep, action = first
    assert timestep == 12
    np.testing.assert_allclose(action[0], 18.0)

    second = queue.pop_next(latest_action=12)
    assert second is not None
    assert second[0] == 13
    np.testing.assert_allclose(second[1][0], 25.0)


def test_server_filters_duplicate_latest_action_unless_must_go():
    service = RDT2AsyncService(policy=None)
    try:
        with service.predicted_latest_actions_lock:
            service.predicted_latest_actions.add(7)

        duplicate = rdt2_async_pb2.ObservationRequest(
            request_id=1,
            timestamp=0.0,
            latest_action=7,
            must_go=False,
        )
        ack = service.SubmitObservation(duplicate, None)
        assert not ack.accepted

        must_go = rdt2_async_pb2.ObservationRequest(
            request_id=2,
            timestamp=0.0,
            latest_action=7,
            must_go=True,
        )
        ack = service.SubmitObservation(must_go, None)
        assert ack.accepted
    finally:
        service.shutdown()


def test_summarize_action_marks_static_pose_chunk():
    action = np.zeros((4, 14), dtype=np.float32)
    action[:, 0] = 0.001

    summary = summarize_action(action, pose_slices=(slice(0, 7), slice(7, 14)))

    assert summary["near_static"]
    assert summary["arms"][0]["total_pos_delta"] == 0.0


def test_summarize_action_marks_moving_pose_chunk():
    action = np.zeros((4, 14), dtype=np.float32)
    action[:, 0] = np.linspace(0.0, 0.02, 4, dtype=np.float32)

    summary = summarize_action(action, pose_slices=(slice(0, 7), slice(7, 14)))

    assert not summary["near_static"]
    assert summary["arms"][0]["total_pos_delta"] > 0.01


def test_stop_detector_fires_when_target_moves_but_actual_stays_put():
    detector = StopDetector(window_seconds=0.5)
    target0 = np.zeros(14, dtype=np.float32)
    actual = np.zeros(14, dtype=np.float32)
    target1 = target0.copy()
    target1[0] = 0.02

    assert detector.update(0.0, target0, actual, (slice(0, 7), slice(7, 14))) == []
    events = detector.update(0.5, target1, actual, (slice(0, 7), slice(7, 14)))

    assert len(events) == 1
    assert events[0]["arm"] == 0


def test_stop_detector_does_not_fire_when_actual_follows_target():
    detector = StopDetector(window_seconds=0.5)
    target0 = np.zeros(14, dtype=np.float32)
    target1 = target0.copy()
    target1[0] = 0.02

    detector.update(0.0, target0, target0, (slice(0, 7), slice(7, 14)))
    events = detector.update(0.5, target1, target1, (slice(0, 7), slice(7, 14)))

    assert events == []


def test_collect_sdk_snapshot_handles_missing_and_failing_methods():
    class FakeRobot:
        def GetArmStatusMsgs(self):
            raise RuntimeError("status unavailable")

        def GetArmEndPoseMsgs(self):
            class Msg:
                value = 3

            return Msg()

    snapshot = collect_sdk_snapshot(FakeRobot())

    assert "status unavailable" in snapshot["GetArmStatusMsgs"]["error"]
    assert snapshot["GetArmEndPoseMsgs"]["value"] == 3


def test_action_chunk_carries_latest_action_only():
    chunk = rdt2_async_pb2.ActionChunk(
        request_id=3,
        latest_action=11,
    )

    assert chunk.latest_action == 11
