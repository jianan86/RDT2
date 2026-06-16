#!/usr/bin/env python
"""检查 Pika 采集轨迹是否能由 Piper 机械臂 IK 到达。

作用：
    读取原始 Pika/UMI episode 的 TCP 位姿和夹爪数据，将采集轨迹映射到 Piper 默认初始 TCP 坐标系下，逐帧运行 Pinocchio IK。脚本会统计每侧 IK 成功帧数，可选导出每帧关节轨迹 npz，便于后续 RViz 播放或过滤不可达数据。

使用示例：
    python data/pika_ik.py --input-root /home/jianan/workspace/data/0616_dex --report-path /tmp/pika_ik_report.json

    python data/pika_ik.py --input-root /home/jianan/workspace/data/0616_dex --episodes episode0 episode1 --include-failed-indices --trajectory-output-dir /tmp/pika_ik_traj
"""


from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_inference.pose_utils import T_EE_TCP, T_TCP_EE
from data.convert_pika_to_rdt2_fm_webdataset import (
    DUAL_GRIPPER_DIR,
    DUAL_POSE_DIR,
    SINGLE_GRIPPER_DIR,
    SINGLE_POSE_DIR,
    discover_episode_dirs,
    load_pose_matrix,
    load_synced_files,
)


DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0601_dex")
DEFAULT_URDF_PATH = REPO_ROOT / "data/piper_urdf/agx_arm_description/urdf/piper_pika.urdf"
DEFAULT_REPORT_PATH = Path("/tmp/pika_ik_report.json")
DEFAULT_GRIPPER_INPUT_MAX = 0.1
DEFAULT_GRIPPER_OUTPUT_MAX = 0.088
# DEFAULT_INITIAL_JOINTS = np.asarray([0.0, 1.70, -1.52, 0.10, 0.0, 0.0], dtype=np.float64)
# DEFAULT_INITIAL_JOINTS = np.asarray([0.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
DEFAULT_INITIAL_JOINTS = np.asarray([0.0, 0.44, -0.257, 0.0, 0.27, 0.0], dtype=np.float64)


ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER_JOINT_NAMES = ("gripper_left_joint", "gripper_right_joint")
TRAJECTORY_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
EE_FRAME_NAME = "link6"

JOINT_LIMITS_RAD = np.asarray(
    [
        [-154.0, 154.0],
        [0.0, 195.0],
        [-175.0, 0.0],
        [-100.0, 112.0],
        [-75.0, 75.0],
        [-170.0, 170.0],
    ],
    dtype=np.float64,
) * (math.pi / 180.0)


@dataclass(frozen=True)
class SideResult:
    total_frames: int
    success_frames: int
    failed_indices: list[int]
    qpos: np.ndarray
    success: np.ndarray
    timestamps: np.ndarray
    target_ee: np.ndarray
    target_tcp: np.ndarray

    @property
    def success_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.success_frames / self.total_frames

    def to_json(self, *, include_failed_indices: bool) -> dict[str, Any]:
        payload = {
            "total_frames": self.total_frames,
            "success_frames": self.success_frames,
            "success_ratio": self.success_ratio,
        }
        if include_failed_indices:
            payload["failed_indices"] = self.failed_indices
        return payload


@dataclass(frozen=True)
class EpisodeResult:
    episode: str
    sides: dict[str, SideResult]
    passed: bool

    def to_json(self, *, include_failed_indices: bool) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "passed": self.passed,
            "sides": {
                side: result.to_json(include_failed_indices=include_failed_indices)
                for side, result in self.sides.items()
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Pika raw episodes by Piper IK reachability.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--arm-mode", choices=("single", "dual"), default="dual")
    parser.add_argument("--side", choices=("both", "right", "left"), default="both")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--episodes", nargs="+", help="Episode names or ids to process, e.g. episode0 3 12.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--trajectory-output-dir", type=Path)
    parser.add_argument("--include-failed-indices", action="store_true")
    parser.add_argument("--episode-success-ratio", type=float, default=1.0)
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--rot-tol", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--damp", type=float, default=1e-6)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.02)
    parser.add_argument("--gripper-input-max", type=float, default=DEFAULT_GRIPPER_INPUT_MAX)
    parser.add_argument("--gripper-output-max", type=float, default=DEFAULT_GRIPPER_OUTPUT_MAX)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def import_pinocchio() -> Any:
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError(
            "Pinocchio is required for IK filtering. Install it with "
            "`conda install -c conda-forge pinocchio` or install the `pin` Python package."
        ) from exc
    return pin


def se3_from_mat(pin: Any, mat: np.ndarray) -> Any:
    mat = np.asarray(mat, dtype=np.float64)
    return pin.SE3(mat[:3, :3], mat[:3, 3])


def mat_from_se3(transform: Any) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.asarray(transform.rotation, dtype=np.float64)
    mat[:3, 3] = np.asarray(transform.translation, dtype=np.float64)
    return mat


class PiperIkSolver:
    def __init__(
        self,
        *,
        urdf_path: Path,
        pos_tol: float,
        rot_tol: float,
        max_iter: int,
        dt: float,
        damp: float,
    ) -> None:
        self.pin = import_pinocchio()
        self.model = self.pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(EE_FRAME_NAME)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"URDF does not contain frame/link: {EE_FRAME_NAME}")

        self.joint_ids = [self.model.getJointId(name) for name in ARM_JOINT_NAMES]
        missing = [name for name, joint_id in zip(ARM_JOINT_NAMES, self.joint_ids, strict=True) if joint_id == 0]
        if missing:
            raise ValueError(f"URDF does not contain arm joints: {missing}")

        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.max_iter = max_iter
        self.dt = dt
        self.damp = damp
        self.q_lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.q_upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self._apply_arm_limits()
        self.q0 = self.make_q(DEFAULT_INITIAL_JOINTS)
        self.initial_ee = self.forward_ee(self.q0)
        self.initial_tcp = self.initial_ee @ T_EE_TCP.astype(np.float64)

    def _apply_arm_limits(self) -> None:
        for idx, joint_id in enumerate(self.joint_ids):
            q_idx = self.model.idx_qs[joint_id]
            self.q_lower[q_idx] = JOINT_LIMITS_RAD[idx, 0]
            self.q_upper[q_idx] = JOINT_LIMITS_RAD[idx, 1]

    def make_q(self, joints: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq, dtype=np.float64)
        for value, joint_id in zip(joints, self.joint_ids, strict=True):
            q_idx = self.model.idx_qs[joint_id]
            q[q_idx] = float(value)
        return self.clamp_q(q)

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self.q_lower), self.q_upper)

    def forward_ee(self, q: np.ndarray) -> np.ndarray:
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return mat_from_se3(self.data.oMf[self.frame_id])

    def arm_joints_from_q(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(
            [q[self.model.idx_qs[joint_id]] for joint_id in self.joint_ids],
            dtype=np.float64,
        )

    def solve(self, target_ee: np.ndarray, seed: np.ndarray) -> tuple[bool, np.ndarray]:
        q = seed.copy()
        target = se3_from_mat(self.pin, target_ee)
        eye6 = np.eye(6, dtype=np.float64)

        for _ in range(self.max_iter):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.frame_id]
            error_tf = current.actInv(target)
            pos_err = float(np.linalg.norm(error_tf.translation))
            rot_err = float(np.linalg.norm(self.pin.log3(error_tf.rotation)))
            if pos_err <= self.pos_tol and rot_err <= self.rot_tol:
                return True, q

            err = self.pin.log(error_tf).vector
            jac = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                self.pin.LOCAL,
            )
            step = jac.T @ np.linalg.solve(jac @ jac.T + self.damp * eye6, err)
            q = self.pin.integrate(self.model, q, step * self.dt)
            q = self.clamp_q(q)

        return False, q


def selected_sides(arm_mode: str, side: str) -> list[str]:
    if arm_mode == "single":
        return ["single"]
    if side == "both":
        return ["right", "left"]
    return [side]


def normalize_episode_name(value: str) -> str:
    return value if value.startswith("episode") else f"episode{value}"


def select_episode_dirs(episodes: list[Path], requested: list[str] | None, max_episodes: int | None) -> list[Path]:
    if requested:
        by_name = {episode.name: episode for episode in episodes}
        selected = []
        missing = []
        for item in requested:
            name = normalize_episode_name(item)
            episode = by_name.get(name)
            if episode is None:
                missing.append(name)
            else:
                selected.append(episode)
        if missing:
            raise ValueError(f"Requested episodes not found: {missing}")
        return selected

    if max_episodes is not None:
        return episodes[:max_episodes]
    return episodes


def pose_dir_for(episode_dir: Path, arm_mode: str, side: str) -> Path:
    if arm_mode == "single":
        return episode_dir / SINGLE_POSE_DIR
    suffix = {"right": "r", "left": "l"}[side]
    return episode_dir / Path(str(DUAL_POSE_DIR).format(side=suffix))


def gripper_dir_for(episode_dir: Path, arm_mode: str, side: str) -> Path:
    if arm_mode == "single":
        return episode_dir / SINGLE_GRIPPER_DIR
    suffix = {"right": "r", "left": "l"}[side]
    return episode_dir / Path(str(DUAL_GRIPPER_DIR).format(side=suffix))


def load_pose_sequence(pose_dir: Path, max_frames: int | None) -> tuple[list[np.ndarray], np.ndarray]:
    files = load_synced_files(pose_dir)
    if max_frames is not None:
        files = files[:max_frames]
    timestamps = np.asarray([item.timestamp for item in files], dtype=np.float64)
    poses = [load_pose_matrix(item.path).astype(np.float64) for item in files]
    return poses, timestamps


def load_gripper_width(path: Path, *, input_max: float, output_max: float) -> float:
    with path.open() as f:
        payload = json.load(f)
    if input_max <= 0.0:
        raise ValueError(f"gripper_input_max must be positive, got {input_max}")
    width = float(payload["distance"]) / input_max * output_max
    return float(np.clip(width, 0.0, output_max))


def load_gripper_joints(
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
    joints = np.full((len(pose_timestamps), len(GRIPPER_JOINT_NAMES)), np.nan, dtype=np.float64)

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

        width = load_gripper_width(
            best.path,
            input_max=gripper_input_max,
            output_max=gripper_output_max,
        )
        half_width = float(np.clip(width * 0.5, 0.0, 0.05))
        joints[frame_idx] = (half_width, -half_width)

    return joints


def target_poses_from_capture_tcp(
    solver: PiperIkSolver,
    first_tcp: np.ndarray,
    tcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rel_tcp = np.linalg.inv(first_tcp) @ tcp
    target_tcp = solver.initial_tcp @ rel_tcp
    target_ee = target_tcp @ T_TCP_EE.astype(np.float64)
    return target_ee, target_tcp


def filter_side(
    solver: PiperIkSolver,
    poses: list[np.ndarray],
    timestamps: np.ndarray,
    gripper_joints: np.ndarray,
) -> SideResult:
    if not poses:
        empty_pose = np.empty((0, 4, 4), dtype=np.float64)
        return SideResult(
            total_frames=0,
            success_frames=0,
            failed_indices=[],
            qpos=np.empty((0, len(TRAJECTORY_JOINT_NAMES)), dtype=np.float64),
            success=np.empty((0,), dtype=bool),
            timestamps=timestamps,
            target_ee=empty_pose,
            target_tcp=empty_pose,
        )
    if gripper_joints.shape != (len(poses), len(GRIPPER_JOINT_NAMES)):
        raise ValueError(
            f"gripper_joints shape {gripper_joints.shape} does not match "
            f"frames={len(poses)} joints={len(GRIPPER_JOINT_NAMES)}"
        )

    first_tcp = poses[0]
    seed = solver.q0.copy()
    success_frames = 0
    failed_indices: list[int] = []
    qpos = np.full((len(poses), len(TRAJECTORY_JOINT_NAMES)), np.nan, dtype=np.float64)
    success = np.zeros((len(poses),), dtype=bool)
    target_ee = np.empty((len(poses), 4, 4), dtype=np.float64)
    target_tcp = np.empty((len(poses), 4, 4), dtype=np.float64)

    for frame_idx, tcp in enumerate(poses):
        frame_target_ee, frame_target_tcp = target_poses_from_capture_tcp(solver, first_tcp, tcp)
        target_ee[frame_idx] = frame_target_ee
        target_tcp[frame_idx] = frame_target_tcp
        solved, solution = solver.solve(frame_target_ee, seed)
        gripper = gripper_joints[frame_idx]
        if solved and np.all(np.isfinite(gripper)):
            success_frames += 1
            seed = solution
            success[frame_idx] = True
            qpos[frame_idx] = np.concatenate((solver.arm_joints_from_q(solution), gripper), axis=0)
        else:
            failed_indices.append(frame_idx)

    return SideResult(
        total_frames=len(poses),
        success_frames=success_frames,
        failed_indices=failed_indices,
        qpos=qpos,
        success=success,
        timestamps=timestamps,
        target_ee=target_ee,
        target_tcp=target_tcp,
    )


def filter_episode(
    solver: PiperIkSolver,
    episode_dir: Path,
    *,
    arm_mode: str,
    side: str,
    max_frames: int | None,
    episode_success_ratio: float,
    trajectory_output_dir: Path | None,
    max_sync_delta_sec: float,
    gripper_input_max: float,
    gripper_output_max: float,
) -> EpisodeResult:
    sides: dict[str, SideResult] = {}
    for side_name in selected_sides(arm_mode, side):
        poses, timestamps = load_pose_sequence(pose_dir_for(episode_dir, arm_mode, side_name), max_frames)
        gripper_joints = load_gripper_joints(
            gripper_dir_for(episode_dir, arm_mode, side_name),
            timestamps,
            max_sync_delta_sec=max_sync_delta_sec,
            gripper_input_max=gripper_input_max,
            gripper_output_max=gripper_output_max,
        )
        result = filter_side(solver, poses, timestamps, gripper_joints)
        sides[side_name] = result
        if trajectory_output_dir is not None:
            write_side_trajectory(trajectory_output_dir, episode_dir.name, side_name, result)
        logging.info(
            "%s %s: %d/%d IK frames succeeded (%.3f)",
            episode_dir.name,
            side_name,
            result.success_frames,
            result.total_frames,
            result.success_ratio,
        )

    passed = all(result.success_ratio >= episode_success_ratio for result in sides.values())
    return EpisodeResult(episode=episode_dir.name, sides=sides, passed=passed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_side_trajectory(output_dir: Path, episode_name: str, side_name: str, result: SideResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{episode_name}_{side_name}.npz"
    np.savez_compressed(
        path,
        joint_names=np.asarray(TRAJECTORY_JOINT_NAMES),
        qpos=result.qpos,
        success=result.success,
        timestamps=result.timestamps,
        target_ee=result.target_ee,
        target_tcp=result.target_tcp,
    )
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    solver = PiperIkSolver(
        urdf_path=args.urdf,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        max_iter=args.max_iter,
        dt=args.dt,
        damp=args.damp,
    )
    episodes = select_episode_dirs(
        discover_episode_dirs(args.input_root),
        requested=args.episodes,
        max_episodes=args.max_episodes,
    )

    results = [
        filter_episode(
            solver,
            episode_dir,
            arm_mode=args.arm_mode,
            side=args.side,
            max_frames=args.max_frames,
            episode_success_ratio=args.episode_success_ratio,
            trajectory_output_dir=args.trajectory_output_dir,
            max_sync_delta_sec=args.max_sync_delta_sec,
            gripper_input_max=args.gripper_input_max,
            gripper_output_max=args.gripper_output_max,
        )
        for episode_dir in episodes
    ]

    report = {
        "input_root": str(args.input_root),
        "urdf": str(args.urdf),
        "arm_mode": args.arm_mode,
        "side": args.side,
        "requested_episodes": args.episodes,
        "trajectory_output_dir": str(args.trajectory_output_dir) if args.trajectory_output_dir else None,
        "episode_success_ratio": args.episode_success_ratio,
        "max_sync_delta_sec": args.max_sync_delta_sec,
        "gripper_input_max": args.gripper_input_max,
        "gripper_output_max": args.gripper_output_max,
        "episodes_total": len(results),
        "episodes_passed": sum(result.passed for result in results),
        "episodes": [
            result.to_json(include_failed_indices=args.include_failed_indices)
            for result in results
        ],
    }
    write_json(args.report_path, report)
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    report = run(args)
    logging.info(
        "Passed %d/%d episodes. Report: %s",
        report["episodes_passed"],
        report["episodes_total"],
        args.report_path,
    )


if __name__ == "__main__":
    main()
