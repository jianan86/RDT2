#!/usr/bin/env python
"""批量统计 Pika episode 在 replay 仿真口径下的控制异常。

作用：
    遍历原始 Pika/UMI 数据集中的 episode，复用 replay_pika.py 的目标轨迹生成逻辑和 Piper IK 求解器，统计每个 episode 有多少帧无法形成有效控制命令。
    问题帧包括 IK 求解失败、目标超出机械臂可达范围、夹爪数据缺失或不同步等。

使用示例：
    python data/report_pika_replay_anomalies.py --input-root /home/jianan/workspace/data/0616_dex

    python data/report_pika_replay_anomalies.py --input-root /home/jianan/workspace/data/0616_dex --episodes episode0 episode1 --include-problem-indices --report-path /tmp/pika_replay_anomaly_report.json
"""


from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.convert_pika_to_rdt2_fm_webdataset import discover_episode_dirs
from data.filter_pika_ik import (
    DEFAULT_GRIPPER_INPUT_MAX,
    DEFAULT_GRIPPER_OUTPUT_MAX,
    DEFAULT_URDF_PATH,
    PiperIkSolver,
    select_episode_dirs,
)
from data.replay_pika import (
    load_replay_episode,
    target_ee_from_tcp,
    target_tcp_sequence,
)


DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0616_dex")
DEFAULT_REPORT_PATH = Path("/tmp/pika_replay_anomaly_report.json")


@dataclass(frozen=True)
class SideAnomalyResult:
    total_frames: int
    problem_indices: list[int]

    @property
    def problem_frames(self) -> int:
        return len(self.problem_indices)

    @property
    def problem_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.problem_frames / self.total_frames

    def to_json(self, *, include_problem_indices: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "problem_frames": self.problem_frames,
            "total_frames": self.total_frames,
            "problem_ratio": self.problem_ratio,
        }
        if include_problem_indices:
            payload["problem_indices"] = self.problem_indices
        return payload


@dataclass(frozen=True)
class EpisodeAnomalyResult:
    episode: str
    sides: dict[str, SideAnomalyResult]

    @property
    def problem_frames(self) -> int:
        return sum(side.problem_frames for side in self.sides.values())

    @property
    def total_frames(self) -> int:
        return sum(side.total_frames for side in self.sides.values())

    @property
    def problem_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.problem_frames / self.total_frames

    def to_json(self, *, include_problem_indices: bool) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "problem_frames": self.problem_frames,
            "total_frames": self.total_frames,
            "problem_ratio": self.problem_ratio,
            "sides": {
                name: side.to_json(include_problem_indices=include_problem_indices)
                for name, side in self.sides.items()
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report per-episode Pika replay control anomalies using sim-mode IK checks."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--arm-mode", choices=("single", "dual"), default="dual")
    parser.add_argument("--side", choices=("both", "right", "left"), default="both")
    parser.add_argument("--episodes", nargs="+", help="Episode names or ids to process, e.g. episode0 3 12.")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--include-problem-indices", action="store_true")
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--rot-tol", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--damp", type=float, default=1e-6)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.02)
    parser.add_argument("--gripper-input-max", type=float, default=DEFAULT_GRIPPER_INPUT_MAX)
    parser.add_argument("--gripper-output-max", type=float, default=DEFAULT_GRIPPER_OUTPUT_MAX)
    parser.add_argument("--log-level", default="INFO")
    return normalize_args(parser.parse_args())


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None
    if args.max_episodes is not None and args.max_episodes <= 0:
        args.max_episodes = None
    return args


def evaluate_side(solver: PiperIkSolver, side: Any) -> SideAnomalyResult:
    targets = target_tcp_sequence(solver.initial_tcp, side.capture_tcp)
    seed = solver.q0.copy()
    problem_indices: list[int] = []

    for frame_idx, target_tcp in enumerate(targets):
        width = float(side.gripper_widths[frame_idx])
        solved, solution = solver.solve(target_ee_from_tcp(target_tcp), seed)
        if solved and np.isfinite(width):
            seed = solution
        else:
            problem_indices.append(frame_idx)

    return SideAnomalyResult(total_frames=len(targets), problem_indices=problem_indices)


def evaluate_episode(solver: PiperIkSolver, episode: Any) -> EpisodeAnomalyResult:
    return EpisodeAnomalyResult(
        episode=episode.name,
        sides={side_name: evaluate_side(solver, side) for side_name, side in episode.sides.items()},
    )


def format_episode_line(result: EpisodeAnomalyResult) -> str:
    parts = [
        result.episode,
        f"total {result.problem_frames}/{result.total_frames}",
    ]
    for side_name, side in result.sides.items():
        parts.append(f"{side_name} {side.problem_frames}/{side.total_frames}")
    return " ".join(parts)


def build_report(args: argparse.Namespace, results: list[EpisodeAnomalyResult]) -> dict[str, Any]:
    problem_frames = sum(result.problem_frames for result in results)
    total_frames = sum(result.total_frames for result in results)
    problem_ratio = 0.0 if total_frames == 0 else problem_frames / total_frames
    return {
        "input_root": str(args.input_root),
        "urdf": str(args.urdf),
        "arm_mode": args.arm_mode,
        "side": args.side,
        "requested_episodes": args.episodes,
        "max_sync_delta_sec": args.max_sync_delta_sec,
        "gripper_input_max": args.gripper_input_max,
        "gripper_output_max": args.gripper_output_max,
        "episodes_total": len(results),
        "problem_frames": problem_frames,
        "total_frames": total_frames,
        "problem_ratio": problem_ratio,
        "episodes": [
            result.to_json(include_problem_indices=args.include_problem_indices)
            for result in results
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    solver = PiperIkSolver(
        urdf_path=args.urdf,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        max_iter=args.max_iter,
        dt=args.dt,
        damp=args.damp,
    )
    episode_dirs = select_episode_dirs(
        discover_episode_dirs(args.input_root),
        requested=args.episodes,
        max_episodes=args.max_episodes,
    )

    results: list[EpisodeAnomalyResult] = []
    for episode_dir in episode_dirs:
        episode = load_replay_episode(
            episode_dir,
            arm_mode=args.arm_mode,
            side=args.side,
            max_frames=args.max_frames,
            max_sync_delta_sec=args.max_sync_delta_sec,
            gripper_input_max=args.gripper_input_max,
            gripper_output_max=args.gripper_output_max,
        )
        result = evaluate_episode(solver, episode)
        results.append(result)
        print(format_episode_line(result))

    report = build_report(args, results)
    write_json(args.report_path, report)
    logging.info(
        "Problem frames %d/%d across %d episodes. Report: %s",
        report["problem_frames"],
        report["total_frames"],
        report["episodes_total"],
        args.report_path,
    )
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
