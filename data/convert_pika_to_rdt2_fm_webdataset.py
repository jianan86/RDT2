#!/usr/bin/env python

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0601_dex")
DEFAULT_OUTPUT_ROOT = Path("/home/jianan/workspace/data/0601_dex_rdt2_fm")
DEFAULT_NORMALIZER_PATH = Path(
    "/home/jianan/Downloads/rdt2/RVQActionTokenizer/umi_normalizer_wo_downsample_indentity_rot.pt"
)
DEFAULT_INSTRUCTION = (
    "Pick up the bread using the right hand. "
    "Put the bread into the bowl using the right hand."
)
DEFAULT_INSTRUCTION_KEY = "0601_dex/bread_to_bowl"

FISHEYE_CAMERA = Path("camera/color/pikaFisheyeCamera_{side}")
POSE_DIR = Path("localization/pose/pika_{side}")
GRIPPER_DIR = Path("gripper/encoder/pika_{side}")
IMAGE_SIZE = 384


@dataclass(frozen=True)
class TimedFile:
    timestamp: float
    path: Path


@dataclass(frozen=True)
class MatchedFrame:
    image: Path
    pose: Path
    gripper: Path


@dataclass(frozen=True)
class EpisodeStreams:
    right_camera: list[TimedFile]
    left_camera: list[TimedFile]
    right_pose: list[TimedFile]
    left_pose: list[TimedFile]
    right_gripper: list[TimedFile]
    left_gripper: list[TimedFile]


class ShardWriter:
    def __init__(self, output_root: Path, samples_per_shard: int) -> None:
        self.output_root = output_root
        self.samples_per_shard = samples_per_shard
        self.shard_idx = -1
        self.sample_in_shard = 0
        self.tar: tarfile.TarFile | None = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def _open_next_shard(self) -> None:
        self.close()
        self.shard_idx += 1
        self.sample_in_shard = 0
        shard_path = self.output_root / f"shard-{self.shard_idx:06d}.tar"
        self.tar = tarfile.open(shard_path, "w")

    def write_sample(self, sample_id: int, image: Image.Image, action: np.ndarray, meta: dict[str, Any]) -> None:
        if self.tar is None or self.sample_in_shard >= self.samples_per_shard:
            self._open_next_shard()

        assert self.tar is not None
        prefix = str(sample_id)
        self._add_bytes(f"{prefix}.image.jpg", encode_jpeg(image))
        self._add_bytes(f"{prefix}.action.npy", encode_npy(action))
        self._add_bytes(f"{prefix}.meta.json", json.dumps(meta).encode("utf-8"))
        self.sample_in_shard += 1

    def _add_bytes(self, name: str, payload: bytes) -> None:
        assert self.tar is not None
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pika UMI data to RDT2-FM WebDataset shards.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--instruction-key", default=DEFAULT_INSTRUCTION_KEY)
    parser.add_argument("--normalizer-path", type=Path, default=DEFAULT_NORMALIZER_PATH)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.02)
    parser.add_argument("--samples-per-shard", type=int, default=10000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_synced_files(directory: Path) -> list[TimedFile]:
    sync_path = directory / "sync.txt"
    if not sync_path.is_file():
        raise FileNotFoundError(f"Required sync file not found: {sync_path}")

    files: list[TimedFile] = []
    with sync_path.open() as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"File listed in sync.txt does not exist: {path}")
            files.append(TimedFile(timestamp_from_path(path), path))

    if not files:
        raise ValueError(f"No files listed in sync file: {sync_path}")
    return files


def timestamp_from_path(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"File name must start with a numeric timestamp: {path.name}") from exc


def discover_episode_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("episode")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    episodes = sorted((path for path in input_root.iterdir() if path.is_dir()), key=sort_key)
    if not episodes:
        raise ValueError(f"No episode directories found under: {input_root}")
    return episodes


def load_episode_streams(episode_dir: Path) -> EpisodeStreams:
    return EpisodeStreams(
        right_camera=load_synced_files(episode_dir / Path(str(FISHEYE_CAMERA).format(side="r"))),
        left_camera=load_synced_files(episode_dir / Path(str(FISHEYE_CAMERA).format(side="l"))),
        right_pose=load_synced_files(episode_dir / Path(str(POSE_DIR).format(side="r"))),
        left_pose=load_synced_files(episode_dir / Path(str(POSE_DIR).format(side="l"))),
        right_gripper=load_synced_files(episode_dir / Path(str(GRIPPER_DIR).format(side="r"))),
        left_gripper=load_synced_files(episode_dir / Path(str(GRIPPER_DIR).format(side="l"))),
    )


def nearest_file(files: list[TimedFile], timestamp: float, max_delta: float) -> TimedFile | None:
    timestamps = [item.timestamp for item in files]
    idx = int(np.searchsorted(timestamps, timestamp))
    candidates = []
    if idx < len(files):
        candidates.append(files[idx])
    if idx > 0:
        candidates.append(files[idx - 1])
    if not candidates:
        return None

    best = min(candidates, key=lambda item: abs(item.timestamp - timestamp))
    if abs(best.timestamp - timestamp) > max_delta:
        return None
    return best


def match_side(
    *,
    timestamp: float,
    camera_file: TimedFile | None,
    camera_stream: list[TimedFile],
    pose_stream: list[TimedFile],
    gripper_stream: list[TimedFile],
    max_delta: float,
) -> MatchedFrame | None:
    camera = camera_file if camera_file is not None else nearest_file(camera_stream, timestamp, max_delta)
    pose = nearest_file(pose_stream, timestamp, max_delta)
    gripper = nearest_file(gripper_stream, timestamp, max_delta)
    if camera is None or pose is None or gripper is None:
        return None
    return MatchedFrame(camera.path, pose.path, gripper.path)


def load_pose_matrix(path: Path) -> np.ndarray:
    with path.open() as f:
        payload = json.load(f)
    pos = np.asarray([payload[key] for key in ("x", "y", "z")], dtype=np.float32)
    rot = euler_xyz_to_rotation_matrix(
        float(payload["roll"]),
        float(payload["pitch"]),
        float(payload["yaw"]),
    )
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def load_gripper_width(path: Path) -> np.float32:
    with path.open() as f:
        payload = json.load(f)
    return np.float32(payload["distance"])


def euler_xyz_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    for axis, angle in zip(("x", "y", "z"), (roll, pitch, yaw), strict=True):
        matrix = matrix @ rotation_about_axis(axis, angle)
    return matrix.astype(np.float32, copy=False)


def rotation_about_axis(axis: str, angle: float) -> np.ndarray:
    c = np.float32(math.cos(angle))
    s = np.float32(math.sin(angle))
    if axis == "x":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    if axis == "z":
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported axis: {axis}")


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return np.concatenate((mat[:3, 0], mat[:3, 1]), axis=0).astype(np.float32, copy=False)


def pose_mat_to_pose9d(mat: np.ndarray) -> np.ndarray:
    return np.concatenate((mat[:3, 3], mat_to_rot6d(mat)), axis=0).astype(np.float32, copy=False)


def build_robot_action(base: MatchedFrame, future_frames: list[MatchedFrame]) -> np.ndarray:
    base_pose = load_pose_matrix(base.pose)
    inv_base_pose = np.linalg.inv(base_pose).astype(np.float32, copy=False)
    actions = []
    for frame in future_frames:
        future_pose = load_pose_matrix(frame.pose)
        relative_pose = inv_base_pose @ future_pose
        gripper = np.asarray([load_gripper_width(frame.gripper)], dtype=np.float32)
        actions.append(np.concatenate((pose_mat_to_pose9d(relative_pose), gripper), axis=0))
    return np.stack(actions, axis=0).astype(np.float32, copy=False)


def build_action(
    right_base: MatchedFrame,
    left_base: MatchedFrame,
    right_future: list[MatchedFrame],
    left_future: list[MatchedFrame],
) -> np.ndarray:
    right_action = build_robot_action(right_base, right_future)
    left_action = build_robot_action(left_base, left_future)
    return np.concatenate((right_action, left_action), axis=1).astype(np.float32, copy=False)


def preprocess_single_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width < height:
        raise ValueError(f"Expected fisheye image width >= height for vertical padding, got {image.size}: {path}")

    pad_total = width - height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    image = ImageOps.expand(image, border=(0, pad_top, 0, pad_bottom), fill=0)
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


def build_binocular_image(left_path: Path, right_path: Path) -> Image.Image:
    left = preprocess_single_image(left_path)
    right = preprocess_single_image(right_path)
    output = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE))
    output.paste(left, (0, 0))
    output.paste(right, (IMAGE_SIZE, 0))
    return output


def encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_dataset_config(output_root: Path, normalizer_path: Path) -> Path:
    config_path = output_root / "0601_dex_rdt2_fm.yaml"
    config = (
        "name: pika/0601_dex_rdt2_fm\n"
        "type: single\n"
        f"shards_dir: {output_root}\n"
        "kwargs:\n"
        f"  instruction_path: {output_root / 'instructions.json'}\n"
        f"  normalizer_path: {normalizer_path}\n"
    )
    config_path.write_text(config)
    return config_path


def convert(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_root(args.output_root, args.overwrite)
    write_json(args.output_root / "instructions.json", {args.instruction_key: args.instruction})
    config_path = write_dataset_config(args.output_root, args.normalizer_path)

    episodes = discover_episode_dirs(args.input_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    sample_id = 0
    skipped_sync = 0
    skipped_horizon = 0
    converted_by_episode: dict[str, int] = {}
    meta = {"sub_task_instruction_key": args.instruction_key}

    with ShardWriter(args.output_root, args.samples_per_shard) as writer:
        for episode_dir in episodes:
            streams = load_episode_streams(episode_dir)
            episode_samples = 0
            last_start = len(streams.right_camera) - args.horizon + 1
            if last_start <= 0:
                skipped_horizon += len(streams.right_camera)
                converted_by_episode[episode_dir.name] = 0
                continue

            for start_idx in range(last_start):
                if args.max_samples is not None and sample_id >= args.max_samples:
                    break

                right_future: list[MatchedFrame] = []
                left_future: list[MatchedFrame] = []
                for offset in range(args.horizon):
                    right_camera = streams.right_camera[start_idx + offset]
                    timestamp = right_camera.timestamp
                    right_frame = match_side(
                        timestamp=timestamp,
                        camera_file=right_camera,
                        camera_stream=streams.right_camera,
                        pose_stream=streams.right_pose,
                        gripper_stream=streams.right_gripper,
                        max_delta=args.max_sync_delta_sec,
                    )
                    left_frame = match_side(
                        timestamp=timestamp,
                        camera_file=None,
                        camera_stream=streams.left_camera,
                        pose_stream=streams.left_pose,
                        gripper_stream=streams.left_gripper,
                        max_delta=args.max_sync_delta_sec,
                    )
                    if right_frame is None or left_frame is None:
                        break
                    right_future.append(right_frame)
                    left_future.append(left_frame)

                if len(right_future) != args.horizon or len(left_future) != args.horizon:
                    skipped_sync += 1
                    continue

                image = build_binocular_image(left_future[0].image, right_future[0].image)
                action = build_action(
                    right_base=right_future[0],
                    left_base=left_future[0],
                    right_future=right_future,
                    left_future=left_future,
                )
                writer.write_sample(sample_id, image=image, action=action, meta=meta)
                sample_id += 1
                episode_samples += 1

            converted_by_episode[episode_dir.name] = episode_samples
            logging.info("Converted %s samples from %s", episode_samples, episode_dir)
            if args.max_samples is not None and sample_id >= args.max_samples:
                break

    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "dataset_config": str(config_path),
        "instruction_key": args.instruction_key,
        "instruction": args.instruction,
        "fps": args.fps,
        "horizon": args.horizon,
        "max_sync_delta_sec": args.max_sync_delta_sec,
        "samples_per_shard": args.samples_per_shard,
        "total_samples": sample_id,
        "skipped_sync_windows": skipped_sync,
        "skipped_horizon_frames": skipped_horizon,
        "converted_by_episode": converted_by_episode,
    }
    write_json(args.output_root / "conversion_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    manifest = convert(args)
    logging.info("Wrote %s samples to %s", manifest["total_samples"], args.output_root)
    logging.info("Dataset config: %s", manifest["dataset_config"])


if __name__ == "__main__":
    main()
