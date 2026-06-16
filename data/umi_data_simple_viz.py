"""可视化单个 Pika episode 的 TCP 轨迹和夹爪开合状态。

作用：
    读取一个 episode 中指定 Pika 手的 localization pose JSON 和 gripper encoder JSON，用 matplotlib 绘制 3D 轨迹。点颜色表示最近时间戳匹配到的夹爪距离，可选择按第 0 帧转换为相对坐标，便于观察采集轨迹形状。

使用示例：
    python data/umi_data_simple_viz.py

    python data/umi_data_simple_viz.py --relative --output /tmp/pika_r_pose_relative.png

注意：
    当前默认读取文件顶部的 POSE_DIR 和 GRIPPER_DIR；如需查看其他 episode 或左手，直接修改这两个常量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSE_DIR = Path("/home/jianan/workspace/data/0616_dex/episode29/localization/pose/pika_r")
GRIPPER_DIR = Path("/home/jianan/workspace/data/0616_dex/episode29/gripper/encoder/pika_r")
DEFAULT_OUTPUT_PATH = Path("umi_episode0_pika_r_pose_matplotlib.png")


def load_poses(pose_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(pose_dir.glob("*.json"), key=lambda path: float(path.stem))
    if not files:
        raise FileNotFoundError(f"No pose JSON files found in {pose_dir}")

    timestamps = []
    poses = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        timestamps.append(float(path.stem))
        poses.append([data[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")])

    return np.asarray(timestamps, dtype=float), np.asarray(poses, dtype=float)


def load_gripper_distances(gripper_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(gripper_dir.glob("*.json"), key=lambda path: float(path.stem))
    if not files:
        raise FileNotFoundError(f"No gripper JSON files found in {gripper_dir}")

    timestamps = []
    distances = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        timestamps.append(float(path.stem))
        distances.append(float(data["distance"]))

    return np.asarray(timestamps, dtype=float), np.asarray(distances, dtype=float)


def match_nearest_values(query_ts: np.ndarray, source_ts: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(source_ts, query_ts)
    idx = np.clip(idx, 0, len(source_ts) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_ts) - 1)

    use_prev = np.abs(query_ts - source_ts[prev_idx]) <= np.abs(query_ts - source_ts[idx])
    nearest_idx = np.where(use_prev, prev_idx, idx)
    return values[nearest_idx]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def pose_matrices(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = poses[:, :3]
    rotations = np.stack([rpy_to_matrix(*pose[3:]) for pose in poses], axis=0)
    return xyz, rotations


def relativize_poses(xyz: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start_xyz = xyz[0]
    start_rot_inv = rotations[0].T
    rel_xyz = (start_rot_inv @ (xyz - start_xyz).T).T
    rel_rotations = np.einsum("ij,njk->nik", start_rot_inv, rotations)
    return rel_xyz, rel_rotations


def draw_frame(ax, origin: np.ndarray, rotation: np.ndarray, length: float, label: str) -> np.ndarray:
    colors = ("r", "g", "b")
    axis_labels = ("x", "y", "z")
    endpoints = []

    for axis_idx, (color, axis_label) in enumerate(zip(colors, axis_labels)):
        end = origin + rotation[:, axis_idx] * length
        endpoints.append(end)
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=2,
        )
        ax.text(end[0], end[1], end[2], f"{label} {axis_label}", color=color)

    return np.vstack([origin, np.asarray(endpoints)])


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.05)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def default_output_path(relative: bool) -> Path:
    if not relative:
        return DEFAULT_OUTPUT_PATH
    return DEFAULT_OUTPUT_PATH.with_name(f"{DEFAULT_OUTPUT_PATH.stem}_relative{DEFAULT_OUTPUT_PATH.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize episode0 pika_r pose with gripper state colors.")
    parser.add_argument("--relative", action="store_true", help="Plot all poses relative to frame 0.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.relative)

    pose_ts, poses = load_poses(POSE_DIR)
    gripper_ts, gripper_distances = load_gripper_distances(GRIPPER_DIR)
    pose_distances = match_nearest_values(pose_ts, gripper_ts, gripper_distances)

    xyz, rotations = pose_matrices(poses)
    if args.relative:
        xyz, rotations = relativize_poses(xyz, rotations)

    start_xyz = xyz[0]
    start_rot = rotations[0]
    span = float(np.ptp(xyz, axis=0).max())
    frame_length = max(span * 0.25, 0.03)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    scatter = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=pose_distances,
        cmap="viridis",
        s=12,
        alpha=0.8,
        depthshade=False,
        label="pika_r xyz",
    )
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.35", linewidth=1.0, alpha=0.45)
    ax.scatter(
        start_xyz[0],
        start_xyz[1],
        start_xyz[2],
        c=[pose_distances[0]],
        cmap="viridis",
        vmin=float(pose_distances.min()),
        vmax=float(pose_distances.max()),
        s=95,
        marker="o",
        edgecolors="tab:red",
        linewidths=2.0,
        label="start",
        depthshade=False,
    )

    world_frame_points = draw_frame(ax, np.zeros(3), np.eye(3), frame_length, "world")
    start_frame_points = draw_frame(ax, start_xyz, start_rot, frame_length, "start")
    set_equal_axes(ax, np.vstack([xyz, world_frame_points, start_frame_points]))

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
    colorbar.set_label("gripper distance")

    mode = "relative to frame 0" if args.relative else "world"
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Episode 0 localization pose: pika_r ({mode})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Loaded {len(poses)} poses from {POSE_DIR}")
    print(f"Loaded {len(gripper_distances)} gripper samples from {GRIPPER_DIR}")
    print(f"First timestamp: {pose_ts[0]:.6f}")
    print(f"Gripper distance range on poses: {pose_distances.min():.6f} to {pose_distances.max():.6f}")
    print(f"Relative mode: {args.relative}")
    print(f"Saved figure to {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
