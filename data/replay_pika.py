#!/usr/bin/env python
"""回放原始 Pika episode 到仿真 RViz 或真实 Piper/Pika 硬件。

作用：
    从 UMI/Pika 原始数据集中读取左右手 TCP 位姿和夹爪数据，将采集轨迹转换为 Piper 机械臂可执行的目标轨迹。--mode sim 通过 ROS2 JointState 发布到 RViz 检查 IK 轨迹；--mode real 将轨迹发送到真实 Piper 机械臂和 Pika 夹爪。

使用示例：
    python data/replay_pika.py --mode sim --input-root /home/jianan/workspace/data/0616_dex --episodes episode0

    python data/replay_pika.py --mode real --input-root /home/jianan/workspace/data/0616_dex --episodes episode0 --dry-run --no-piper --no-pika
"""


from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_inference.pose_utils import (
    LEFT_ARM_SLICE,
    RIGHT_ARM_SLICE,
    T_TCP_EE,
    mat_to_pose7d,
    tcp_pose14_to_ee_pose14,
)
from data.filter_pika_ik import (
    DEFAULT_GRIPPER_INPUT_MAX,
    DEFAULT_GRIPPER_OUTPUT_MAX,
    DEFAULT_INPUT_ROOT,
    DEFAULT_URDF_PATH,
    PiperIkSolver,
    TRAJECTORY_JOINT_NAMES,
    gripper_dir_for,
    load_gripper_width,
    load_pose_sequence,
    pose_dir_for,
    select_episode_dirs,
    selected_sides,
)
from data.convert_pika_to_rdt2_fm_webdataset import discover_episode_dirs, load_synced_files


DEFAULT_HARDWARE_CONFIG = REPO_ROOT / "configs/robots/eval_bimanual_piper_pika_config.yaml"


@dataclass(frozen=True)
class ReplaySide:
    side: str
    capture_tcp: list[np.ndarray]
    timestamps: np.ndarray
    gripper_widths: np.ndarray

    @property
    def frames(self) -> int:
        return len(self.capture_tcp)


@dataclass(frozen=True)
class ReplayEpisode:
    name: str
    sides: dict[str, ReplaySide]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay raw Pika episodes in ROS2 RViz or on real Piper hardware.")
    parser.add_argument("--mode", choices=("sim", "real"), required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--episodes", nargs="+", help="Episode names or ids to process, e.g. episode0 3 12.")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--arm-mode", choices=("single", "dual"), default="dual")
    parser.add_argument("--side", choices=("both", "right", "left"), default="both")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.02)
    parser.add_argument("--gripper-input-max", type=float, default=DEFAULT_GRIPPER_INPUT_MAX)
    parser.add_argument("--gripper-output-max", type=float, default=DEFAULT_GRIPPER_OUTPUT_MAX)
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--failed-policy", choices=("hold", "skip"), default="hold")
    parser.add_argument("--init-hold-sec", type=float, default=1.0)
    parser.add_argument("--publish-target-tcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-tcp-frame", default="target_tcp")
    parser.add_argument("--target-tcp-parent", default="world")
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--rot-tol", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--damp", type=float, default=1e-6)

    parser.add_argument("--use-init-tcp", action="store_true")
    parser.add_argument("--hardware-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--right-piper-can", default=None)
    parser.add_argument("--left-piper-can", default=None)
    parser.add_argument("--right-gripper-port", default=None)
    parser.add_argument("--left-gripper-port", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-piper", action="store_true")
    parser.add_argument("--no-pika", action="store_true")
    parser.add_argument("--real-init-hold-sec", type=float, default=1.0)
    parser.add_argument("--max-pos-step", default=0.01, type=float)
    parser.add_argument("--max-rot-step", default=0.05, type=float)
    parser.add_argument("--max-gripper-step", default=0.005, type=float)
    parser.add_argument("--min-gripper", default=0.0, type=float)
    parser.add_argument("--max-gripper", default=0.10, type=float)
    parser.add_argument("--diagnostic-stop-on-piper-status-errors", action="store_true")
    parser.add_argument("--ignore-piper-status-errors", action="store_true")
    return normalize_args(apply_hardware_config(parser.parse_args()))


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None
    if args.max_episodes is not None and args.max_episodes <= 0:
        args.max_episodes = None
    return args


def apply_hardware_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg: dict[str, Any] = {}
    if args.hardware_config:
        with args.hardware_config.expanduser().open("r") as f:
            cfg = yaml.safe_load(f) or {}

    def fill(name: str, side: str, key: str, default: str) -> None:
        if getattr(args, name) is None:
            setattr(args, name, cfg.get(side, {}).get(key, default))

    fill("right_piper_can", "right", "piper_can", "right_piper")
    fill("left_piper_can", "left", "piper_can", "left_piper")
    fill("right_gripper_port", "right", "gripper_port", "/dev/ttyUSB0")
    fill("left_gripper_port", "left", "gripper_port", "/dev/ttyUSB1")
    return args


def load_gripper_widths(
    gripper_dir: Path,
    pose_timestamps: np.ndarray,
    *,
    max_sync_delta_sec: float,
    gripper_input_max: float,
    gripper_output_max: float,
) -> np.ndarray:
    if max_sync_delta_sec < 0.0:
        raise ValueError(f"max_sync_delta_sec must be non-negative, got {max_sync_delta_sec}")

    files = load_synced_files(gripper_dir)
    gripper_timestamps = np.asarray([item.timestamp for item in files], dtype=np.float64)
    widths = np.full((len(pose_timestamps),), np.nan, dtype=np.float32)

    for frame_idx, timestamp in enumerate(pose_timestamps):
        nearest_idx = int(np.searchsorted(gripper_timestamps, timestamp))
        candidates = []
        if nearest_idx < len(files):
            candidates.append(files[nearest_idx])
        if nearest_idx > 0:
            candidates.append(files[nearest_idx - 1])
        if not candidates:
            continue

        best = min(candidates, key=lambda item: abs(item.timestamp - timestamp))
        if abs(best.timestamp - timestamp) > max_sync_delta_sec:
            continue
        widths[frame_idx] = load_gripper_width(
            best.path,
            input_max=gripper_input_max,
            output_max=gripper_output_max,
        )
    return widths


def load_replay_side(
    episode_dir: Path,
    *,
    arm_mode: str,
    side: str,
    max_frames: int | None,
    max_sync_delta_sec: float,
    gripper_input_max: float,
    gripper_output_max: float,
) -> ReplaySide:
    poses, timestamps = load_pose_sequence(pose_dir_for(episode_dir, arm_mode, side), max_frames)
    widths = load_gripper_widths(
        gripper_dir_for(episode_dir, arm_mode, side),
        timestamps,
        max_sync_delta_sec=max_sync_delta_sec,
        gripper_input_max=gripper_input_max,
        gripper_output_max=gripper_output_max,
    )
    return ReplaySide(side=side, capture_tcp=poses, timestamps=timestamps, gripper_widths=widths)


def load_replay_episode(
    episode_dir: Path,
    *,
    arm_mode: str,
    side: str,
    max_frames: int | None,
    max_sync_delta_sec: float,
    gripper_input_max: float,
    gripper_output_max: float,
) -> ReplayEpisode:
    sides = {
        side_name: load_replay_side(
            episode_dir,
            arm_mode=arm_mode,
            side=side_name,
            max_frames=max_frames,
            max_sync_delta_sec=max_sync_delta_sec,
            gripper_input_max=gripper_input_max,
            gripper_output_max=gripper_output_max,
        )
        for side_name in selected_sides(arm_mode, side)
    }
    return ReplayEpisode(name=episode_dir.name, sides=sides)


def target_tcp_from_capture_delta(
    replay_base_tcp: np.ndarray,
    capture_first_tcp: np.ndarray,
    capture_tcp: np.ndarray,
) -> np.ndarray:
    rel_tcp = np.linalg.inv(np.asarray(capture_first_tcp, dtype=np.float64)) @ np.asarray(capture_tcp, dtype=np.float64)
    return np.asarray(replay_base_tcp, dtype=np.float64) @ rel_tcp


def target_tcp_sequence(replay_base_tcp: np.ndarray, capture_tcp: list[np.ndarray]) -> np.ndarray:
    if not capture_tcp:
        return np.empty((0, 4, 4), dtype=np.float64)
    first_tcp = capture_tcp[0]
    return np.stack(
        [target_tcp_from_capture_delta(replay_base_tcp, first_tcp, tcp) for tcp in capture_tcp],
        axis=0,
    ).astype(np.float64, copy=False)


def tcp_matrix_to_pose7(tcp: np.ndarray, gripper_width: float) -> np.ndarray:
    return mat_to_pose7d(np.asarray(tcp, dtype=np.float32), float(gripper_width)).astype(np.float32, copy=False)


def target_ee_from_tcp(target_tcp: np.ndarray) -> np.ndarray:
    return np.asarray(target_tcp, dtype=np.float64) @ T_TCP_EE.astype(np.float64)


def gripper_width_to_joint_pair(width: float) -> np.ndarray:
    half_width = float(np.clip(width * 0.5, 0.0, 0.05))
    return np.asarray([half_width, -half_width], dtype=np.float64)


def log_initial_position(
    *,
    mode: str,
    episode: str,
    side: str,
    source: str,
    replay_base_tcp: np.ndarray,
    capture_first_tcp: np.ndarray,
    frames: int,
) -> None:
    logging.info(
        "%s %s %s frames=%d initial_source=%s replay_base_tcp=%s capture_first_tcp=%s",
        mode,
        episode,
        side,
        frames,
        source,
        np.round(replay_base_tcp, 5).tolist(),
        np.round(capture_first_tcp, 5).tolist(),
    )


class RosJointStateReplay:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from tf2_ros import TransformBroadcaster

        self.rclpy = rclpy
        self.TransformStamped = TransformStamped
        self.JointState = JointState
        self.rclpy.init()
        self.node: Node = self.rclpy.create_node("pika_episode_replay")
        self.publisher = self.node.create_publisher(JointState, args.topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self.node) if args.publish_target_tcp else None
        self.target_tcp_frame = args.target_tcp_frame
        self.target_tcp_parent = args.target_tcp_parent

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()

    def publish(self, q: np.ndarray, target_tcp: np.ndarray | None = None) -> None:
        stamp = self.node.get_clock().now().to_msg()
        msg = self.JointState()
        msg.header.stamp = stamp
        msg.name = list(TRAJECTORY_JOINT_NAMES)
        msg.position = [float(value) for value in q]
        self.publisher.publish(msg)
        if target_tcp is not None and self.tf_broadcaster is not None:
            self.publish_target_tcp(target_tcp, stamp)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_target_tcp(self, target_tcp: np.ndarray, stamp: object) -> None:
        msg = self.TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_tcp_parent
        msg.child_frame_id = self.target_tcp_frame
        msg.transform.translation.x = float(target_tcp[0, 3])
        msg.transform.translation.y = float(target_tcp[1, 3])
        msg.transform.translation.z = float(target_tcp[2, 3])
        quat = matrix_to_quat(target_tcp[:3, :3])
        msg.transform.rotation.w = float(quat[0])
        msg.transform.rotation.x = float(quat[1])
        msg.transform.rotation.y = float(quat[2])
        msg.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(msg)


class ReplayPikaGripper:
    def __init__(self, side: str, port: str, dry_run: bool, no_pika: bool) -> None:
        self.side = side
        self.port = port
        self.dry_run = dry_run
        self.device = None
        if no_pika:
            print(f"[pika:{side}] disabled")
            return

        from pika.gripper import Gripper

        self.device = Gripper(port)
        if not self.device.connect():
            raise RuntimeError(f"failed to connect Pika {side} gripper: {port}")
        self.device.enable()
        print(f"[pika:{side}] connected port={port}")

    def close(self) -> None:
        if self.device is not None:
            self.device.disconnect()

    def read_width(self) -> float:
        if self.device is None:
            return 0.0
        return max(float(self.device.get_gripper_distance()) / 1000.0, 0.0)

    def execute_width(self, width_m: float) -> None:
        if self.device is None or self.dry_run:
            return
        width_mm = float(np.clip(width_m * 1000.0, 0.0, 90.0))
        self.device.set_gripper_distance(width_mm)


class RealReplayHardware:
    def __init__(self, args: argparse.Namespace) -> None:
        from async_inference.real_piper_client import PiperRobot

        self.args = args
        self.arm_mode = args.arm_mode
        self.arms: dict[str, PiperRobot] = {}
        self.grippers: dict[str, ReplayPikaGripper] = {}
        for side in hardware_sides(args.arm_mode, args.side):
            self.arms[side] = PiperRobot(
                side,
                getattr(args, f"{side}_piper_can"),
                args.dry_run,
                args.no_piper,
                None,
                args.diagnostic_stop_on_piper_status_errors and not args.ignore_piper_status_errors,
            )
            self.grippers[side] = ReplayPikaGripper(
                side,
                getattr(args, f"{side}_gripper_port"),
                args.dry_run,
                args.no_pika,
            )

    def close(self) -> None:
        for gripper in self.grippers.values():
            gripper.close()

    def read_side_tcp_pose7(self, side: str) -> np.ndarray:
        ee = np.concatenate(
            [
                self.arms[side].read_pose(),
                np.asarray([self.grippers[side].read_width()], dtype=np.float32),
            ]
        )
        tcp = ee_pose7_to_tcp_pose7(ee)
        return tcp.astype(np.float32, copy=False)

    def execute_side_tcp_pose7(self, side: str, tcp_target: np.ndarray) -> None:
        pose14 = np.zeros((14,), dtype=np.float32)
        pose_slice = RIGHT_ARM_SLICE if side in ("right", "single") else LEFT_ARM_SLICE
        pose14[pose_slice] = tcp_target
        if side in ("right", "single"):
            pose14[LEFT_ARM_SLICE] = tcp_target
        else:
            pose14[RIGHT_ARM_SLICE] = tcp_target
        ee_target = tcp_pose14_to_ee_pose14(pose14)[pose_slice]
        self.arms[side].execute_pose(ee_target)
        self.grippers[side].execute_width(float(tcp_target[6]))


def hardware_sides(arm_mode: str, side: str) -> list[str]:
    if arm_mode == "single":
        return ["right"]
    if side == "both":
        return ["right", "left"]
    return [side]


def replay_side_to_hardware_side(arm_mode: str, side: str) -> str:
    return "right" if arm_mode == "single" else side


def ee_pose7_to_tcp_pose7(ee_pose7: np.ndarray) -> np.ndarray:
    from async_inference.pose_utils import ee_pose7d_to_tcp_pose7d

    return ee_pose7d_to_tcp_pose7d(np.asarray(ee_pose7, dtype=np.float32))


def run_sim(args: argparse.Namespace, episodes: list[ReplayEpisode]) -> None:
    if args.fps <= 0.0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    solver = PiperIkSolver(
        urdf_path=args.urdf,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        max_iter=args.max_iter,
        dt=args.dt,
        damp=args.damp,
    )
    player = RosJointStateReplay(args)
    period = 1.0 / args.fps
    initial_q = np.concatenate(
        [
            solver.arm_joints_from_q(solver.q0),
            gripper_width_to_joint_pair(0.0),
        ],
        axis=0,
    )
    try:
        while True:
            for episode in episodes:
                for side_name, side in episode.sides.items():
                    if side.frames == 0:
                        logging.warning("%s %s is empty", episode.name, side_name)
                        continue
                    base_tcp = solver.initial_tcp
                    log_initial_position(
                        mode="sim",
                        episode=episode.name,
                        side=side_name,
                        source="ik_default",
                        replay_base_tcp=base_tcp,
                        capture_first_tcp=side.capture_tcp[0],
                        frames=side.frames,
                    )
                    hold_until = time.time() + max(0.0, args.init_hold_sec)
                    while time.time() < hold_until:
                        started = time.time()
                        player.publish(initial_q, base_tcp if args.publish_target_tcp else None)
                        sleep_remaining(started, period)
                    replay_sim_side(args, solver, player, side, base_tcp, period)
            if not args.loop:
                break
    finally:
        player.close()


def replay_sim_side(
    args: argparse.Namespace,
    solver: PiperIkSolver,
    player: RosJointStateReplay,
    side: ReplaySide,
    base_tcp: np.ndarray,
    period: float,
) -> None:
    targets = target_tcp_sequence(base_tcp, side.capture_tcp)
    seed = solver.q0.copy()
    last_valid = np.concatenate(
        [
            solver.arm_joints_from_q(solver.q0),
            gripper_width_to_joint_pair(0.0),
        ],
        axis=0,
    )
    for frame_idx, target_tcp in enumerate(targets):
        started = time.time()
        width = float(side.gripper_widths[frame_idx])
        target_ee = target_ee_from_tcp(target_tcp)
        solved, solution = solver.solve(target_ee, seed)
        if solved and np.isfinite(width):
            seed = solution
            q = np.concatenate([solver.arm_joints_from_q(solution), gripper_width_to_joint_pair(width)], axis=0)
            last_valid = q
            player.publish(q, target_tcp if args.publish_target_tcp else None)
        elif args.failed_policy == "hold":
            player.publish(last_valid, target_tcp if args.publish_target_tcp else None)
        else:
            logging.warning("skip frame side=%s frame=%d solved=%s width=%s", side.side, frame_idx, solved, width)
        sleep_remaining(started, period)


def run_real(args: argparse.Namespace, episodes: list[ReplayEpisode]) -> None:
    if args.fps <= 0.0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    solver = PiperIkSolver(
        urdf_path=args.urdf,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        max_iter=args.max_iter,
        dt=args.dt,
        damp=args.damp,
    )
    hardware = RealReplayHardware(args)
    period = 1.0 / args.fps
    try:
        while True:
            for episode in episodes:
                replay_real_episode(args, solver, hardware, episode, period)
            if not args.loop:
                break
    finally:
        hardware.close()


def replay_real_episode(
    args: argparse.Namespace,
    solver: PiperIkSolver,
    hardware: RealReplayHardware,
    episode: ReplayEpisode,
    period: float,
) -> None:
    base_by_side: dict[str, np.ndarray] = {}
    targets_by_side: dict[str, np.ndarray] = {}
    for replay_side_name, side in episode.sides.items():
        if side.frames == 0:
            logging.warning("%s %s is empty", episode.name, replay_side_name)
            continue
        hardware_side = replay_side_to_hardware_side(args.arm_mode, replay_side_name)
        if args.use_init_tcp:
            base_tcp_mat = solver.initial_tcp
            source = "ik_default"
        else:
            current_tcp = hardware.read_side_tcp_pose7(hardware_side)
            base_tcp_mat = pose7_to_matrix(current_tcp)
            source = "current_hardware"
        base_by_side[replay_side_name] = base_tcp_mat
        targets_by_side[replay_side_name] = target_tcp_sequence(base_tcp_mat, side.capture_tcp)
        log_initial_position(
            mode="real",
            episode=episode.name,
            side=replay_side_name,
            source=source,
            replay_base_tcp=base_tcp_mat,
            capture_first_tcp=side.capture_tcp[0],
            frames=side.frames,
        )

    if args.use_init_tcp:
        for replay_side_name, side in episode.sides.items():
            if replay_side_name not in base_by_side:
                continue
            hardware_side = replay_side_to_hardware_side(args.arm_mode, replay_side_name)
            width = first_finite_width(side.gripper_widths)
            command = tcp_matrix_to_pose7(base_by_side[replay_side_name], width)
            hardware.execute_side_tcp_pose7(hardware_side, command)
        time.sleep(max(0.0, args.real_init_hold_sec))

    max_frames = max((side.frames for side in episode.sides.values()), default=0)
    last_command: dict[str, np.ndarray] = {}
    for frame_idx in range(max_frames):
        started = time.time()
        for replay_side_name, side in episode.sides.items():
            targets = targets_by_side.get(replay_side_name)
            if targets is None or frame_idx >= len(targets):
                continue
            hardware_side = replay_side_to_hardware_side(args.arm_mode, replay_side_name)
            width = float(side.gripper_widths[frame_idx])
            if not np.isfinite(width):
                logging.warning("skip real frame side=%s frame=%d missing gripper width", replay_side_name, frame_idx)
                continue
            requested = tcp_matrix_to_pose7(targets[frame_idx], width)
            current = last_command.get(hardware_side)
            if current is None:
                current = hardware.read_side_tcp_pose7(hardware_side)
            target = limit_pose_step(current, requested, args)
            if args.dry_run:
                print(f"[dry-run] episode={episode.name} side={replay_side_name} frame={frame_idx} target={target.round(5).tolist()}")
            else:
                hardware.execute_side_tcp_pose7(hardware_side, target)
            last_command[hardware_side] = target
        sleep_remaining(started, period)


def first_finite_width(widths: np.ndarray) -> float:
    finite = widths[np.isfinite(widths)]
    if finite.size == 0:
        return 0.0
    return float(finite[0])


def limit_pose_step(current: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).copy()
    current = np.asarray(current, dtype=np.float32)
    target[:3] = current[:3] + np.clip(target[:3] - current[:3], -args.max_pos_step, args.max_pos_step)
    target[3:6] = current[3:6] + np.clip(target[3:6] - current[3:6], -args.max_rot_step, args.max_rot_step)
    target[6] = current[6] + float(np.clip(target[6] - current[6], -args.max_gripper_step, args.max_gripper_step))
    target[6] = float(np.clip(target[6], args.min_gripper, args.max_gripper))
    return target


def sleep_remaining(started: float, period: float) -> None:
    remaining = period - (time.time() - started)
    if remaining > 0.0:
        time.sleep(remaining)


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    from async_inference.pose_utils import pose7d_to_mat

    return pose7d_to_mat(np.asarray(pose7, dtype=np.float32)).astype(np.float64, copy=False)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    from async_inference.pose_utils import matrix_to_quat as convert

    return convert(matrix)


def load_episodes(args: argparse.Namespace) -> list[ReplayEpisode]:
    episode_dirs = select_episode_dirs(
        discover_episode_dirs(args.input_root),
        requested=args.episodes,
        max_episodes=args.max_episodes,
    )
    return [
        load_replay_episode(
            episode_dir,
            arm_mode=args.arm_mode,
            side=args.side,
            max_frames=args.max_frames,
            max_sync_delta_sec=args.max_sync_delta_sec,
            gripper_input_max=args.gripper_input_max,
            gripper_output_max=args.gripper_output_max,
        )
        for episode_dir in episode_dirs
    ]


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    episodes = load_episodes(args)
    if args.mode == "sim":
        run_sim(args, episodes)
    else:
        run_real(args, episodes)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
