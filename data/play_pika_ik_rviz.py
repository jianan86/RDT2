#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play exported Pika IK joint trajectories in RViz.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--failed-policy", choices=("hold", "skip"), default="hold")
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--publish-target-tcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-tcp-frame", default="target_tcp")
    parser.add_argument("--target-tcp-parent", default="world")
    return parser.parse_args()


class JointTrajectoryPlayer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pika_ik_joint_trajectory_player")
        if args.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {args.fps}")

        payload = np.load(args.trajectory, allow_pickle=False)
        self.joint_names = [str(name) for name in payload["joint_names"]]
        self.qpos = np.asarray(payload["qpos"], dtype=np.float64)
        self.success = np.asarray(payload["success"], dtype=bool)
        self.target_tcp = np.asarray(payload["target_tcp"], dtype=np.float64) if args.publish_target_tcp else None
        if self.qpos.ndim != 2 or self.qpos.shape[1] != len(self.joint_names):
            raise ValueError(
                f"qpos shape {self.qpos.shape} does not match joint_names length {len(self.joint_names)}"
            )
        if self.success.shape != (self.qpos.shape[0],):
            raise ValueError(f"success shape {self.success.shape} does not match qpos frames {self.qpos.shape[0]}")
        if self.qpos.shape[0] == 0:
            raise ValueError(f"empty trajectory: {args.trajectory}")
        if self.target_tcp is not None and self.target_tcp.shape != (self.qpos.shape[0], 4, 4):
            raise ValueError(f"target_tcp shape {self.target_tcp.shape} does not match qpos frames {self.qpos.shape[0]}")

        self.failed_policy = args.failed_policy
        self.loop = args.loop
        self.target_tcp_frame = args.target_tcp_frame
        self.target_tcp_parent = args.target_tcp_parent
        self.done = False
        self.frame_idx = 0
        self.last_valid = np.zeros((len(self.joint_names),), dtype=np.float64)
        self.publisher = self.create_publisher(JointState, args.topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.target_tcp is not None else None
        self.timer = self.create_timer(1.0 / args.fps, self.publish_next)
        self.get_logger().info(
            f"playing {args.trajectory} frames={self.qpos.shape[0]} fps={args.fps} "
            f"failed_policy={self.failed_policy} loop={self.loop} "
            f"target_tcp={self.target_tcp_frame if self.target_tcp is not None else 'disabled'}"
        )

    def publish_next(self) -> None:
        while self.frame_idx < self.qpos.shape[0]:
            frame_idx = self.frame_idx
            q = self.qpos[frame_idx]
            ok = bool(self.success[frame_idx]) and np.all(np.isfinite(q))
            self.frame_idx += 1
            if ok:
                self.last_valid = q.copy()
                self.publish_frame(q, frame_idx)
                return
            if self.failed_policy == "hold":
                self.publish_frame(self.last_valid, frame_idx)
                return

        if self.loop:
            self.frame_idx = 0
            return
        self.get_logger().info("trajectory finished")
        self.done = True
        self.timer.cancel()

    def publish_frame(self, q: np.ndarray, frame_idx: int) -> None:
        stamp = self.get_clock().now().to_msg()
        self.publish_joint_state(q, stamp)
        if self.target_tcp is not None:
            self.publish_target_tcp(frame_idx, stamp)

    def publish_joint_state(self, q: np.ndarray, stamp: object) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = self.joint_names
        msg.position = [float(value) for value in q]
        self.publisher.publish(msg)

    def publish_target_tcp(self, frame_idx: int, stamp: object) -> None:
        assert self.target_tcp is not None
        assert self.tf_broadcaster is not None
        transform = self.target_tcp[frame_idx]
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_tcp_parent
        msg.child_frame_id = self.target_tcp_frame
        msg.transform.translation.x = float(transform[0, 3])
        msg.transform.translation.y = float(transform[1, 3])
        msg.transform.translation.z = float(transform[2, 3])
        quat = matrix_to_quat(transform[:3, :3])
        msg.transform.rotation.w = float(quat[0])
        msg.transform.rotation.x = float(quat[1])
        msg.transform.rotation.y = float(quat[2])
        msg.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(msg)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ], dtype=np.float64)
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ], dtype=np.float64)
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ], dtype=np.float64)
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = JointTrajectoryPlayer(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
