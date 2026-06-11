#!/usr/bin/env python

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.umi.common.pose_repr_util import convert_pose_mat_rep
from data.umi.pose_util import mat_to_pose10d


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

DUAL_FISHEYE_CAMERA = Path("camera/color/pikaFisheyeCamera_{side}")
DUAL_POSE_DIR = Path("localization/pose/pika_{side}")
DUAL_GRIPPER_DIR = Path("gripper/encoder/pika_{side}")
SINGLE_FISHEYE_CAMERA = Path("camera/color/pikaFisheyeCamera")
SINGLE_POSE_DIR = Path("localization/pose/pika")
SINGLE_GRIPPER_DIR = Path("gripper/encoder/pika")
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


@dataclass(frozen=True)
class EncodedSample:
    image: bytes
    action: bytes
    meta: bytes


@dataclass(frozen=True)
class EpisodeResult:
    episode_key: str
    samples: list[EncodedSample]
    skipped_sync: int
    skipped_sync_before_sample: list[int]
    skipped_horizon: int


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
        self.write_encoded_sample(
            sample_id,
            image=encode_jpeg(image),
            action=encode_npy(action),
            meta=json.dumps(meta).encode("utf-8"),
        )

    def write_encoded_sample(self, sample_id: int, *, image: bytes, action: bytes, meta: bytes) -> None:
        if self.tar is None or self.sample_in_shard >= self.samples_per_shard:
            self._open_next_shard()

        assert self.tar is not None
        prefix = str(sample_id)
        self._add_bytes(f"{prefix}.image.jpg", image)
        self._add_bytes(f"{prefix}.action.npy", action)
        self._add_bytes(f"{prefix}.meta.json", meta)
        self.sample_in_shard += 1

    def _add_bytes(self, name: str, payload: bytes) -> None:
        assert self.tar is not None
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pika UMI data to RDT2-FM WebDataset shards.")
    parser.add_argument("--input-root", type=Path, nargs="+", default=[DEFAULT_INPUT_ROOT])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--arm-mode", choices=("single", "dual"), default="dual")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--instruction-key", default=None)
    parser.add_argument("--normalizer-path", type=Path, default=DEFAULT_NORMALIZER_PATH)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.02)
    parser.add_argument("--gripper-input-max", type=float, default=0.1)
    parser.add_argument("--gripper-output-max", type=float, default=0.088)
    parser.add_argument("--samples-per-shard", type=int, default=10000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
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
    return sorted(files, key=lambda item: item.timestamp)


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


def load_episode_streams(episode_dir: Path, arm_mode: str) -> EpisodeStreams:
    if arm_mode == "single":
        camera = load_synced_files(episode_dir / SINGLE_FISHEYE_CAMERA)
        pose = load_synced_files(episode_dir / SINGLE_POSE_DIR)
        gripper = load_synced_files(episode_dir / SINGLE_GRIPPER_DIR)
        return EpisodeStreams(
            right_camera=camera,
            left_camera=camera,
            right_pose=pose,
            left_pose=pose,
            right_gripper=gripper,
            left_gripper=gripper,
        )

    if arm_mode != "dual":
        raise ValueError(f"Unsupported arm_mode: {arm_mode}")

    return EpisodeStreams(
        right_camera=load_synced_files(episode_dir / Path(str(DUAL_FISHEYE_CAMERA).format(side="r"))),
        left_camera=load_synced_files(episode_dir / Path(str(DUAL_FISHEYE_CAMERA).format(side="l"))),
        right_pose=load_synced_files(episode_dir / Path(str(DUAL_POSE_DIR).format(side="r"))),
        left_pose=load_synced_files(episode_dir / Path(str(DUAL_POSE_DIR).format(side="l"))),
        right_gripper=load_synced_files(episode_dir / Path(str(DUAL_GRIPPER_DIR).format(side="r"))),
        left_gripper=load_synced_files(episode_dir / Path(str(DUAL_GRIPPER_DIR).format(side="l"))),
    )


def stream_timestamps(files: list[TimedFile]) -> np.ndarray:
    return np.asarray([item.timestamp for item in files], dtype=np.float64)


def nearest_file(
    files: list[TimedFile],
    timestamp: float,
    max_delta: float,
    timestamps: np.ndarray | None = None,
) -> TimedFile | None:
    if timestamps is None:
        timestamps = stream_timestamps(files)
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
    camera_timestamps: np.ndarray | None = None,
    pose_timestamps: np.ndarray | None = None,
    gripper_timestamps: np.ndarray | None = None,
) -> MatchedFrame | None:
    camera = (
        camera_file
        if camera_file is not None
        else nearest_file(camera_stream, timestamp, max_delta, camera_timestamps)
    )
    pose = nearest_file(pose_stream, timestamp, max_delta, pose_timestamps)
    gripper = nearest_file(gripper_stream, timestamp, max_delta, gripper_timestamps)
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


def load_gripper_width(path: Path, *, input_max: float, output_max: float) -> np.float32:
    with path.open() as f:
        payload = json.load(f)
    width = float(payload["distance"])
    if input_max <= 0:
        raise ValueError(f"gripper_input_max must be positive, got {input_max}")
    width = width / input_max * output_max
    return np.float32(np.clip(width, 0.0, output_max))


def euler_xyz_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (
        rotation_about_axis("z", yaw)
        @ rotation_about_axis("y", pitch)
        @ rotation_about_axis("x", roll)
    ).astype(np.float32, copy=False)


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



def build_robot_action(
    base: MatchedFrame,
    future_frames: list[MatchedFrame],
    *,
    gripper_input_max: float,
    gripper_output_max: float,
) -> np.ndarray:
    base_pose = load_pose_matrix(base.pose)
    actions = []
    for frame in future_frames:
        future_pose = load_pose_matrix(frame.pose)
        relative_pose = convert_pose_mat_rep(
            future_pose[None],
            base_pose_mat=base_pose,
            pose_rep="relative",
            backward=False,
        )[0]
        gripper = np.asarray(
            [
                load_gripper_width(
                    frame.gripper,
                    input_max=gripper_input_max,
                    output_max=gripper_output_max,
                )
            ],
            dtype=np.float32,
        )
        pose10d = mat_to_pose10d(relative_pose).astype(np.float32, copy=False)
        actions.append(np.concatenate((pose10d, gripper), axis=0))
    return np.stack(actions, axis=0).astype(np.float32, copy=False)


def build_action(
    right_base: MatchedFrame,
    left_base: MatchedFrame,
    right_future: list[MatchedFrame],
    left_future: list[MatchedFrame],
    *,
    gripper_input_max: float,
    gripper_output_max: float,
) -> np.ndarray:
    right_action = build_robot_action(
        right_base,
        right_future,
        gripper_input_max=gripper_input_max,
        gripper_output_max=gripper_output_max,
    )
    left_action = build_robot_action(
        left_base,
        left_future,
        gripper_input_max=gripper_input_max,
        gripper_output_max=gripper_output_max,
    )
    return np.concatenate((right_action, left_action), axis=1).astype(np.float32, copy=False)


def cached_pose_matrix(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    pose = cache.get(path)
    if pose is None:
        pose = load_pose_matrix(path)
        cache[path] = pose
    return pose


def cached_gripper_width(
    path: Path,
    cache: dict[Path, np.float32],
    *,
    input_max: float,
    output_max: float,
) -> np.float32:
    width = cache.get(path)
    if width is None:
        width = load_gripper_width(path, input_max=input_max, output_max=output_max)
        cache[path] = width
    return width


def build_robot_action_cached(
    base: MatchedFrame,
    future_frames: list[MatchedFrame],
    *,
    pose_cache: dict[Path, np.ndarray],
    gripper_cache: dict[Path, np.float32],
    gripper_input_max: float,
    gripper_output_max: float,
) -> np.ndarray:
    base_pose = cached_pose_matrix(base.pose, pose_cache)
    actions = []
    for frame in future_frames:
        future_pose = cached_pose_matrix(frame.pose, pose_cache)
        relative_pose = convert_pose_mat_rep(
            future_pose[None],
            base_pose_mat=base_pose,
            pose_rep="relative",
            backward=False,
        )[0]
        gripper = np.asarray(
            [
                cached_gripper_width(
                    frame.gripper,
                    gripper_cache,
                    input_max=gripper_input_max,
                    output_max=gripper_output_max,
                )
            ],
            dtype=np.float32,
        )
        pose10d = mat_to_pose10d(relative_pose).astype(np.float32, copy=False)
        actions.append(np.concatenate((pose10d, gripper), axis=0))
    return np.stack(actions, axis=0).astype(np.float32, copy=False)


def build_action_cached(
    right_base: MatchedFrame,
    left_base: MatchedFrame,
    right_future: list[MatchedFrame],
    left_future: list[MatchedFrame],
    *,
    pose_cache: dict[Path, np.ndarray],
    gripper_cache: dict[Path, np.float32],
    gripper_input_max: float,
    gripper_output_max: float,
) -> np.ndarray:
    right_action = build_robot_action_cached(
        right_base,
        right_future,
        pose_cache=pose_cache,
        gripper_cache=gripper_cache,
        gripper_input_max=gripper_input_max,
        gripper_output_max=gripper_output_max,
    )
    left_action = build_robot_action_cached(
        left_base,
        left_future,
        pose_cache=pose_cache,
        gripper_cache=gripper_cache,
        gripper_input_max=gripper_input_max,
        gripper_output_max=gripper_output_max,
    )
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


def write_dataset_config(output_root: Path, dataset_name: str, normalizer_path: Path) -> Path:
    config_path = output_root / f"{dataset_name}.yaml"
    config = (
        f"name: pika/{dataset_name}\n"
        "type: single\n"
        f"shards_dir: {output_root}\n"
        "kwargs:\n"
        f"  instruction_path: {output_root / 'instructions.json'}\n"
        f"  normalizer_path: {normalizer_path}\n"
    )
    config_path.write_text(config)
    return config_path


def episode_key_for(input_root: Path, episode_dir: Path, single_input_root: bool) -> str:
    if single_input_root:
        return episode_dir.name
    return f"{input_root.name}/{episode_dir.name}"


def build_episode_matches(
    streams: EpisodeStreams,
    max_delta: float,
) -> tuple[list[MatchedFrame | None], list[MatchedFrame | None]]:
    left_camera_timestamps = stream_timestamps(streams.left_camera)
    right_pose_timestamps = stream_timestamps(streams.right_pose)
    left_pose_timestamps = stream_timestamps(streams.left_pose)
    right_gripper_timestamps = stream_timestamps(streams.right_gripper)
    left_gripper_timestamps = stream_timestamps(streams.left_gripper)

    right_matches: list[MatchedFrame | None] = []
    left_matches: list[MatchedFrame | None] = []
    for right_camera in streams.right_camera:
        timestamp = right_camera.timestamp
        right_matches.append(
            match_side(
                timestamp=timestamp,
                camera_file=right_camera,
                camera_stream=streams.right_camera,
                pose_stream=streams.right_pose,
                gripper_stream=streams.right_gripper,
                max_delta=max_delta,
                pose_timestamps=right_pose_timestamps,
                gripper_timestamps=right_gripper_timestamps,
            )
        )
        left_matches.append(
            match_side(
                timestamp=timestamp,
                camera_file=None,
                camera_stream=streams.left_camera,
                pose_stream=streams.left_pose,
                gripper_stream=streams.left_gripper,
                max_delta=max_delta,
                camera_timestamps=left_camera_timestamps,
                pose_timestamps=left_pose_timestamps,
                gripper_timestamps=left_gripper_timestamps,
            )
        )
    return right_matches, left_matches


def convert_episode(
    input_root: Path,
    episode_dir: Path,
    *,
    single_input_root: bool,
    arm_mode: str,
    horizon: int,
    max_sync_delta_sec: float,
    gripper_input_max: float,
    gripper_output_max: float,
    meta: dict[str, Any],
) -> EpisodeResult:
    streams = load_episode_streams(episode_dir, arm_mode)
    episode_key = episode_key_for(input_root, episode_dir, single_input_root)
    last_start = len(streams.right_camera) - horizon
    if last_start <= 0:
        return EpisodeResult(
            episode_key=episode_key,
            samples=[],
            skipped_sync=0,
            skipped_sync_before_sample=[0],
            skipped_horizon=len(streams.right_camera),
        )

    right_matches, left_matches = build_episode_matches(streams, max_sync_delta_sec)
    pose_cache: dict[Path, np.ndarray] = {}
    gripper_cache: dict[Path, np.float32] = {}
    samples: list[EncodedSample] = []
    skipped_sync = 0
    skipped_sync_before_sample = [0]
    meta_bytes = json.dumps(meta).encode("utf-8")

    for start_idx in range(last_start):
        right_base = right_matches[start_idx]
        left_base = left_matches[start_idx]
        if right_base is None or left_base is None:
            skipped_sync += 1
            continue

        right_future_raw = right_matches[start_idx + 1 : start_idx + horizon + 1]
        left_future_raw = left_matches[start_idx + 1 : start_idx + horizon + 1]
        if any(frame is None for frame in right_future_raw) or any(frame is None for frame in left_future_raw):
            skipped_sync += 1
            continue
        right_future = cast(list[MatchedFrame], right_future_raw)
        left_future = cast(list[MatchedFrame], left_future_raw)

        image = build_binocular_image(left_base.image, right_base.image)
        action = build_action_cached(
            right_base=right_base,
            left_base=left_base,
            right_future=right_future,
            left_future=left_future,
            pose_cache=pose_cache,
            gripper_cache=gripper_cache,
            gripper_input_max=gripper_input_max,
            gripper_output_max=gripper_output_max,
        )
        samples.append(
            EncodedSample(image=encode_jpeg(image), action=encode_npy(action), meta=meta_bytes)
        )
        skipped_sync_before_sample.append(skipped_sync)

    return EpisodeResult(
        episode_key=episode_key,
        samples=samples,
        skipped_sync=skipped_sync,
        skipped_sync_before_sample=skipped_sync_before_sample,
        skipped_horizon=0,
    )


def iter_episode_tasks(input_roots: list[Path], max_episodes: int | None) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    for input_root in input_roots:
        episodes = discover_episode_dirs(input_root)
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        tasks.extend((input_root, episode_dir) for episode_dir in episodes)
    return tasks


def write_episode_result(
    writer: ShardWriter,
    result: EpisodeResult,
    *,
    sample_id: int,
    max_samples: int | None,
) -> tuple[int, int, int]:
    if max_samples is None:
        write_count = len(result.samples)
    else:
        write_count = min(len(result.samples), max_samples - sample_id)

    for sample in result.samples[:write_count]:
        writer.write_encoded_sample(sample_id, image=sample.image, action=sample.action, meta=sample.meta)
        sample_id += 1

    skipped_sync = result.skipped_sync_before_sample[write_count]
    if write_count == len(result.samples):
        skipped_sync = result.skipped_sync
    return sample_id, write_count, skipped_sync


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_roots = args.input_root
    if isinstance(input_roots, Path):
        input_roots = [input_roots]
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")

    single_input_root = len(input_roots) == 1
    dataset_name = args.output_root.name
    instruction_key = args.instruction_key or DEFAULT_INSTRUCTION_KEY

    prepare_output_root(args.output_root, args.overwrite)
    write_json(args.output_root / "instructions.json", {instruction_key: args.instruction})
    config_path = write_dataset_config(args.output_root, dataset_name, args.normalizer_path)

    sample_id = 0
    skipped_sync = 0
    skipped_horizon = 0
    converted_by_episode: dict[str, int] = {}
    meta = {"sub_task_instruction_key": instruction_key}
    tasks = iter_episode_tasks(input_roots, args.max_episodes)

    def handle_result(writer: ShardWriter, result: EpisodeResult) -> bool:
        nonlocal sample_id, skipped_sync, skipped_horizon
        sample_id, episode_samples, episode_skipped_sync = write_episode_result(
            writer,
            result,
            sample_id=sample_id,
            max_samples=args.max_samples,
        )
        skipped_sync += episode_skipped_sync
        skipped_horizon += result.skipped_horizon
        converted_by_episode[result.episode_key] = episode_samples
        logging.info("Converted %s samples from %s", episode_samples, result.episode_key)
        return args.max_samples is not None and sample_id >= args.max_samples

    with ShardWriter(args.output_root, args.samples_per_shard) as writer:
        if args.num_workers <= 1:
            for input_root, episode_dir in tasks:
                result = convert_episode(
                    input_root,
                    episode_dir,
                    single_input_root=single_input_root,
                    arm_mode=args.arm_mode,
                    horizon=args.horizon,
                    max_sync_delta_sec=args.max_sync_delta_sec,
                    gripper_input_max=args.gripper_input_max,
                    gripper_output_max=args.gripper_output_max,
                    meta=meta,
                )
                if handle_result(writer, result):
                    break
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        convert_episode,
                        input_root,
                        episode_dir,
                        single_input_root=single_input_root,
                        arm_mode=args.arm_mode,
                        horizon=args.horizon,
                        max_sync_delta_sec=args.max_sync_delta_sec,
                        gripper_input_max=args.gripper_input_max,
                        gripper_output_max=args.gripper_output_max,
                        meta=meta,
                    )
                    for input_root, episode_dir in tasks
                ]
                for idx, future in enumerate(futures):
                    result = future.result()
                    futures[idx] = None  # type: ignore[list-item]
                    if handle_result(writer, result):
                        for remaining in futures[idx + 1 :]:
                            if remaining is not None:
                                remaining.cancel()
                        break

    manifest = {
        "output_root": str(args.output_root),
        "dataset_config": str(config_path),
        "dataset_name": dataset_name,
        "instruction_key": instruction_key,
        "instruction": args.instruction,
        "fps": args.fps,
        "horizon": args.horizon,
        "max_sync_delta_sec": args.max_sync_delta_sec,
        "gripper_input_max": args.gripper_input_max,
        "gripper_output_max": args.gripper_output_max,
        "samples_per_shard": args.samples_per_shard,
        "total_samples": sample_id,
        "skipped_sync_windows": skipped_sync,
        "skipped_horizon_frames": skipped_horizon,
        "converted_by_episode": converted_by_episode,
    }
    if single_input_root:
        manifest = {"input_root": str(input_roots[0]), **manifest}
    else:
        manifest = {"input_roots": [str(input_root) for input_root in input_roots], **manifest}
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
