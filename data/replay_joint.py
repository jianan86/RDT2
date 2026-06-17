#!/usr/bin/env python
"""Replay saved Piper joint trajectories on real Piper/Pika hardware."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_inference.real_piper_client import PiperRobot
from data.replay_pika import (
    DEFAULT_HARDWARE_CONFIG,
    ReplayPikaGripper,
    apply_hardware_config,
    sleep_remaining,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved Piper joint and Pika gripper trajectories.")
    parser.add_argument("--input", type=Path, required=True, help=".npz file saved by data/replay_pika.py --save-joints.")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hardware-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--right-piper-can", default=None)
    parser.add_argument("--left-piper-can", default=None)
    parser.add_argument("--right-gripper-port", default=None)
    parser.add_argument("--left-gripper-port", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-piper", action="store_true")
    parser.add_argument("--no-pika", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--diagnostic-stop-on-piper-status-errors", action="store_true")
    parser.add_argument("--ignore-piper-status-errors", action="store_true")
    return apply_hardware_config(parser.parse_args())


def load_joint_sequence(path: Path, side: str) -> np.ndarray:
    with np.load(path.expanduser()) as data:
        if side not in data.files:
            raise ValueError(f"joint file {path} does not contain side {side!r}; available sides: {sorted(data.files)}")
        sequence = np.asarray(data[side], dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != 7:
        raise ValueError(f"expected side {side!r} to have shape (N, 7), got {sequence.shape}")
    if not np.all(np.isfinite(sequence)):
        raise ValueError(f"side {side!r} contains non-finite joint or gripper values")
    return sequence


def replay_sequence(args: argparse.Namespace, arm: PiperRobot, gripper: ReplayPikaGripper, sequence: np.ndarray) -> None:
    if args.fps <= 0.0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    period = 1.0 / args.fps
    while True:
        for frame_idx, row in enumerate(sequence):
            started = time.time()
            joints = row[:6]
            gripper_width = float(row[6])
            if args.dry_run:
                print(
                    f"[dry-run] side={args.side} frame={frame_idx} "
                    f"joints={np.round(joints, 5).tolist()} gripper_width={gripper_width:.5f}"
                )
            arm.execute_joints(joints)
            gripper.execute_width(gripper_width)
            sleep_remaining(started, period)
        if not args.loop:
            break


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    sequence = load_joint_sequence(args.input, args.side)
    logging.info("loaded %d frames from %s side=%s", len(sequence), args.input, args.side)

    arm = PiperRobot(
        args.side,
        getattr(args, f"{args.side}_piper_can"),
        args.dry_run,
        args.no_piper,
        None,
        args.diagnostic_stop_on_piper_status_errors and not args.ignore_piper_status_errors,
    )
    gripper = ReplayPikaGripper(
        args.side,
        getattr(args, f"{args.side}_gripper_port"),
        args.dry_run,
        args.no_pika,
    )
    try:
        replay_sequence(args, arm, gripper, sequence)
    finally:
        gripper.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
