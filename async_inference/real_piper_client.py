from __future__ import annotations

import argparse
import signal
import sys
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import grpc
import numpy as np
import yaml
from async_inference.codec import encode_jpeg, save_rgb_png, unflatten_action
from async_inference.proto import rdt2_async_pb2, rdt2_async_pb2_grpc
from async_inference.pose_utils import (
    LEFT_ARM_SLICE,
    RIGHT_ARM_SLICE,
    ee_pose14_to_tcp_pose14,
    slerp_euler_xyz,
    tcp_pose14_to_ee_pose14,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)



@dataclass
class SensorSnapshot:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    tcp_pose_flat: np.ndarray
    timestamp: float


class LatestValue:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


class SensorWorker:
    def __init__(self, hardware: "BimanualHardware", width: int, height: int, image_size: int):
        self.hardware = hardware
        self.width = width
        self.height = height
        self.image_size = image_size
        self.latest = LatestValue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="piper-pika-sensor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                left = preprocess_fisheye(self.hardware.left_gripper.read_fisheye_rgb(self.width, self.height), self.image_size)
                right = preprocess_fisheye(self.hardware.right_gripper.read_fisheye_rgb(self.width, self.height), self.image_size)
                tcp_pose_flat = self.hardware.read_tcp_pose_flat()
                self.latest.set(SensorSnapshot(left_rgb=left, right_rgb=right, tcp_pose_flat=tcp_pose_flat, timestamp=time.time()))
            except Exception as exc:
                print(f"[sensor] read failed: {exc}")
            time.sleep(0.001)


class PiperRobot:
    def __init__(self, side: str, can_name: str, dry_run: bool, no_piper: bool):
        self.side = side
        self.can_name = can_name
        self.dry_run = dry_run
        self.robot = None
        if no_piper:
            print(f"[piper:{side}] disabled")
            return
        from piper_sdk import C_PiperInterface

        self.robot = C_PiperInterface(can_name=can_name)
        self.robot.ConnectPort()
        while not self.robot.EnablePiper():
            time.sleep(0.01)
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        print(f"[piper:{side}] connected can={can_name}")

    def read_pose(self) -> np.ndarray:
        if self.robot is None:
            return np.zeros(6, dtype=np.float32)
        pose = self.robot.GetArmEndPoseMsgs().end_pose
        xyz = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float32) / 1_000_000.0
        rpy = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float32) / 1000.0)
        return np.concatenate([xyz, rpy.astype(np.float32)])

    def execute_pose(self, target: np.ndarray) -> None:
        if self.robot is None or self.dry_run:
            return
        x, y, z, roll, pitch, yaw = target[:6].tolist()
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.robot.EndPoseCtrl(
            int(round(x * 1_000_000.0)),
            int(round(y * 1_000_000.0)),
            int(round(z * 1_000_000.0)),
            int(round(np.degrees(roll) * 1000.0)),
            int(round(np.degrees(pitch) * 1000.0)),
            int(round(np.degrees(yaw) * 1000.0)),
        )


class PikaGripper:
    def __init__(
        self,
        side: str,
        port: str,
        fisheye_device: int | str,
        width: int,
        height: int,
        fps: int,
        dry_run: bool,
        no_pika: bool,
    ):
        self.side = side
        self.port = port
        self.fisheye_device = fisheye_device
        self.dry_run = dry_run
        self.device = None
        self.fisheye = None
        if no_pika:
            print(f"[pika:{side}] disabled")
            return

        from pika.gripper import Gripper

        self.device = Gripper(port)
        if not self.device.connect():
            raise RuntimeError(f"failed to connect Pika {side} gripper: {port}")
        self.device.enable()
        self.device.set_camera_param(width, height, fps)
        self.device.set_fisheye_camera_index(fisheye_device)
        self.fisheye = self.device.get_fisheye_camera()
        self._wait_for_fisheye_frame(width, height)
        print(f"[pika:{side}] connected port={port} fisheye={fisheye_device}")

    def _wait_for_fisheye_frame(self, width: int, height: int) -> None:
        if self.fisheye is None or not getattr(self.fisheye, "is_connected", False):
            raise RuntimeError(f"failed to connect Pika {self.side} fisheye camera: {self.fisheye_device}")

        deadline = time.time() + 2.0
        last_shape = None
        while time.time() < deadline:
            ok, frame = self.fisheye.get_frame()
            if ok and frame is not None:
                last_shape = frame.shape
                if frame.ndim == 3 and frame.shape[2] == 3:
                    return
            time.sleep(0.02)
        raise RuntimeError(
            f"failed to read initial Pika {self.side} fisheye frame: "
            f"device={self.fisheye_device} expected={width}x{height} last_shape={last_shape}"
        )

    def close(self) -> None:
        if self.device is not None:
            self.device.disconnect()

    def read_width(self) -> float:
        if self.device is None:
            return 0.0
        return max(float(self.device.get_gripper_distance()) / 1000.0, 0.0)

    def read_fisheye_rgb(self, width: int, height: int) -> np.ndarray:
        if self.fisheye is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        ok, frame = self.fisheye.get_frame()
        if not ok or frame is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def execute_width(self, width_m: float) -> None:
        if self.device is None or self.dry_run:
            return
        width_mm = float(np.clip(width_m * 1000.0, 0.0, 90.0))
        self.device.set_gripper_distance(width_mm)


class BimanualHardware:
    def __init__(self, args):
        self.right_arm = PiperRobot("right", args.right_piper_can, args.dry_run, args.no_piper)
        self.left_arm = PiperRobot("left", args.left_piper_can, args.dry_run, args.no_piper)
        self.right_gripper = PikaGripper(
            "right",
            args.right_gripper_port,
            args.right_fisheye_device,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
            args.dry_run,
            args.no_pika,
        )
        self.left_gripper = PikaGripper(
            "left",
            args.left_gripper_port,
            args.left_fisheye_device,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
            args.dry_run,
            args.no_pika,
        )

    def close(self) -> None:
        self.right_gripper.close()
        self.left_gripper.close()

    def read_tcp_pose_flat(self) -> np.ndarray:
        right_ee = np.concatenate([
            self.right_arm.read_pose(),
            np.array([self.right_gripper.read_width()], dtype=np.float32),
        ])
        left_ee = np.concatenate([
            self.left_arm.read_pose(),
            np.array([self.left_gripper.read_width()], dtype=np.float32),
        ])
        return ee_pose14_to_tcp_pose14(np.concatenate([right_ee, left_ee]))

    def execute(self, tcp_target: np.ndarray) -> None:
        ee_target = tcp_pose14_to_ee_pose14(tcp_target)
        self.right_arm.execute_pose(ee_target[RIGHT_ARM_SLICE])
        self.left_arm.execute_pose(ee_target[LEFT_ARM_SLICE])
        self.right_gripper.execute_width(float(tcp_target[RIGHT_ARM_SLICE][6]))
        self.left_gripper.execute_width(float(tcp_target[LEFT_ARM_SLICE][6]))


class ActionQueue:
    BLEND_OVERLAP = "blend_overlap"
    PREFIX_NO_MERGE = "prefix_no_merge"

    def __init__(self, chunk_merge_strategy: str = BLEND_OVERLAP, chunk_execute_prefix_steps: int = 0):
        if chunk_merge_strategy not in {self.BLEND_OVERLAP, self.PREFIX_NO_MERGE}:
            raise ValueError(f"unknown chunk_merge_strategy: {chunk_merge_strategy}")
        if chunk_merge_strategy == self.PREFIX_NO_MERGE and chunk_execute_prefix_steps <= 0:
            raise ValueError("chunk_execute_prefix_steps must be positive for prefix_no_merge")
        self._lock = threading.Lock()
        self._actions: list[tuple[int, np.ndarray]] = []
        self._last_first_timestep: Optional[int] = None
        self._last_chunk: Optional[np.ndarray] = None
        self.chunk_merge_strategy = chunk_merge_strategy
        self.chunk_execute_prefix_steps = int(chunk_execute_prefix_steps)

    def add_chunk(self, first_timestep: int, action: np.ndarray, latest_executed_timestep: int) -> None:
        action = np.asarray(action, dtype=np.float32)
        with self._lock:
            if (
                self.chunk_merge_strategy == self.BLEND_OVERLAP
                and self._last_chunk is not None
                and self._last_first_timestep is not None
            ):
                action = self._merge(self._last_first_timestep, self._last_chunk, first_timestep, action)
            self._last_first_timestep = first_timestep
            self._last_chunk = action.copy()
            actions = self._future_actions(first_timestep, action, latest_executed_timestep)
            if self.chunk_merge_strategy == self.PREFIX_NO_MERGE:
                actions = actions[: self.chunk_execute_prefix_steps]
            self._actions = actions

    def pop_next(self, latest_executed_timestep: int) -> Optional[tuple[int, np.ndarray]]:
        with self._lock:
            while self._actions and self._actions[0][0] <= latest_executed_timestep:
                self._actions.pop(0)
            if not self._actions:
                return None
            return self._actions.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._actions)

    @staticmethod
    def _future_actions(
        first_timestep: int,
        action: np.ndarray,
        latest_executed_timestep: int,
    ) -> list[tuple[int, np.ndarray]]:
        return [
            (first_timestep + idx, row.copy())
            for idx, row in enumerate(action)
            if first_timestep + idx > latest_executed_timestep
        ]

    @staticmethod
    def _merge(old_first: int, old: np.ndarray, new_first: int, new: np.ndarray) -> np.ndarray:
        merged = new.copy()
        old_last = old_first + len(old) - 1
        new_last = new_first + len(new) - 1
        overlap_first = max(old_first, new_first)
        overlap_last = min(old_last, new_last)
        if overlap_first > overlap_last:
            return merged

        overlap = overlap_last - overlap_first + 1
        for timestep in range(overlap_first, overlap_last + 1):
            old_idx = timestep - old_first
            new_idx = timestep - new_first
            alpha = float(new_idx + 1) / float(overlap + 1)
            merged[new_idx] = blend_pose14(old[old_idx], new[new_idx], alpha)
        return merged


class ActionStreamWorker:
    def __init__(self, stub, action_queue: ActionQueue, latest_executed, stop_event: threading.Event):
        self.stub = stub
        self.action_queue = action_queue
        self.latest_executed = latest_executed
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self._loop, name="action-stream", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            for chunk in self.stub.StreamActions(rdt2_async_pb2.ActionStreamRequest()):
                if self.stop_event.is_set():
                    return
                if chunk.error:
                    print(f"[client] action request_id={chunk.request_id} error={chunk.error}")
                    continue
                action = unflatten_action(list(chunk.action_flat), chunk.horizon, chunk.action_dim)
                if action.shape[1] != 14:
                    print(f"[client] ignoring action with shape={list(action.shape)}")
                    continue
                latest = self.latest_executed.get()
                self.action_queue.add_chunk(chunk.first_timestep, action, latest)
                print(
                    f"[client] chunk request_id={chunk.request_id} first={chunk.first_timestep} "
                    f"shape={list(action.shape)} strategy={self.action_queue.chunk_merge_strategy} "
                    f"prefix_steps={self.action_queue.chunk_execute_prefix_steps} "
                    f"queue={len(self.action_queue)} latency_ms={chunk.inference_latency_ms:.1f}"
                )
        except grpc.RpcError as exc:
            if not self.stop_event.is_set():
                print(f"[client] action stream failed: {exc}")


class AtomicInt:
    def __init__(self, value: int):
        self._lock = threading.Lock()
        self._value = value

    def get(self) -> int:
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        with self._lock:
            self._value = value


def run(args) -> None:
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stop_event.set())

    tunnel = None
    local_port = args.local_port or find_free_port()
    if not args.no_tunnel:
        tunnel = start_tunnel(args.server_ssh, args.server_ssh_port, local_port, args.remote_host, args.remote_port)
        server = f"127.0.0.1:{local_port}"
    else:
        server = args.server

    hardware = BimanualHardware(args)
    sensors = SensorWorker(hardware, args.camera_width, args.camera_height, args.image_size)
    latest_executed = AtomicInt(-1)
    action_queue = ActionQueue(args.chunk_merge_strategy, args.chunk_execute_prefix_steps)

    channel = grpc.insecure_channel(server)
    stub = rdt2_async_pb2_grpc.RDT2AsyncInferenceStub(channel)
    sensors.start()
    stream = None

    try:
        grpc.channel_ready_future(channel).result(timeout=args.connect_timeout)
        health = stub.Health(rdt2_async_pb2.HealthRequest(), timeout=args.rpc_timeout)
        print(f"[client] health ready={health.ready} message={health.message}", flush=True)
        reset = stub.Reset(rdt2_async_pb2.ResetRequest(), timeout=args.rpc_timeout)
        print(f"[client] reset ok={reset.ok} message={reset.message}", flush=True)
        stream = ActionStreamWorker(stub, action_queue, latest_executed, stop_event)
        stream.start()

        request_thread = threading.Thread(
            target=submit_observations,
            args=(args, stub, sensors, latest_executed, stop_event),
            daemon=True,
        )
        request_thread.start()
        control_loop(args, hardware, action_queue, latest_executed, stop_event)
        request_thread.join(timeout=2.0)
    finally:
        stop_event.set()
        sensors.stop()
        hardware.close()
        if stream is not None:
            stream.join()
        channel.close()
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                tunnel.kill()


def maybe_save_debug_images(args, request_id: int, left: np.ndarray, right: np.ndarray) -> None:
    if not args.debug_image_dir:
        return
    every = max(1, int(args.debug_image_every))
    if request_id % every != 0:
        return
    if args.debug_image_limit > 0 and request_id > args.debug_image_limit * every:
        return

    base = Path(args.debug_image_dir)
    try:
        save_rgb_png(base / f"request_{request_id:06d}_left_client.png", left)
        save_rgb_png(base / f"request_{request_id:06d}_right_client.png", right)
    except Exception as exc:
        print(f"[client] warning: failed to save debug images request_id={request_id}: {exc}")


def submit_observations(args, stub, sensors, latest_executed: AtomicInt, stop_event: threading.Event) -> None:
    period = 1.0 / args.request_fps
    request_id = 0
    while not stop_event.is_set():
        snapshot = sensors.latest.get()
        if snapshot is None:
            time.sleep(0.01)
            continue
        request_id += 1
        started = time.time()
        maybe_save_debug_images(args, request_id, snapshot.left_rgb, snapshot.right_rgb)
        request = rdt2_async_pb2.ObservationRequest(
            request_id=request_id,
            timestamp=snapshot.timestamp,
            instruction=args.instruction,
            left_stereo_jpeg=encode_jpeg(snapshot.left_rgb, quality=args.jpeg_quality),
            right_stereo_jpeg=encode_jpeg(snapshot.right_rgb, quality=args.jpeg_quality),
            state=np.zeros(20, dtype=np.float32).tolist(),
            tcp_pose_flat=snapshot.tcp_pose_flat.tolist(),
            latest_executed_timestep=latest_executed.get(),
        )
        try:
            ack = stub.SubmitObservation(request, timeout=args.rpc_timeout)
            if args.verbose:
                print(
                    f"[client] submitted request_id={ack.request_id} "
                    f"latest={request.latest_executed_timestep} "
                    f"camera_shape={snapshot.left_rgb.shape}/{snapshot.right_rgb.shape} "
                    f"right_tcp={snapshot.tcp_pose_flat[RIGHT_ARM_SLICE].round(4).tolist()} "
                    f"left_tcp={snapshot.tcp_pose_flat[LEFT_ARM_SLICE].round(4).tolist()}"
                )
        except grpc.RpcError as exc:
            print(f"[client] submit failed: {exc}")
        sleep_remaining(started, period)


def control_loop(
    args,
    hardware: BimanualHardware,
    action_queue: ActionQueue,
    latest_executed: AtomicInt,
    stop_event: threading.Event,
) -> None:
    period = 1.0 / args.fps
    last_target = hardware.read_tcp_pose_flat()
    deadline = None if args.run_seconds <= 0 else time.time() + args.run_seconds
    while not stop_event.is_set():
        if deadline is not None and time.time() >= deadline:
            stop_event.set()
            break
        started = time.time()
        item = action_queue.pop_next(latest_executed.get())
        if item is not None:
            timestep, action = item
            target = last_target.copy()
            target[RIGHT_ARM_SLICE] = limit_step(last_target[RIGHT_ARM_SLICE], action[RIGHT_ARM_SLICE], args)
            target[LEFT_ARM_SLICE] = limit_step(last_target[LEFT_ARM_SLICE], action[LEFT_ARM_SLICE], args)
            if args.dry_run:
                print(
                    f"[dry-run] timestep={timestep} "
                    f"right={target[RIGHT_ARM_SLICE].round(4).tolist()} "
                    f"left={target[LEFT_ARM_SLICE].round(4).tolist()} queue={len(action_queue)}"
                )
            else:
                hardware.execute(target)
            last_target = target
            latest_executed.set(timestep)
        sleep_remaining(started, period)


def limit_step(current: np.ndarray, target: np.ndarray, args) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).copy()
    current = np.asarray(current, dtype=np.float32)
    target[:3] = current[:3] + np.clip(target[:3] - current[:3], -args.max_pos_step, args.max_pos_step)
    target[3:6] = current[3:6] + np.clip(target[3:6] - current[3:6], -args.max_rot_step, args.max_rot_step)
    target[6] = current[6] + float(np.clip(target[6] - current[6], -args.max_gripper_step, args.max_gripper_step))
    target[6] = float(np.clip(target[6], args.min_gripper, args.max_gripper))
    return target


def start_tunnel(server_ssh: str, ssh_port: int, local_port: int, remote_host: str, remote_port: int):
    cmd = [
        "ssh",
        "-N",
        "-L",
        f"{local_port}:{remote_host}:{remote_port}",
        server_ssh,
        "-p",
        str(ssh_port),
    ]
    proc = subprocess.Popen(cmd)
    wait_for_local_port(local_port, timeout=5.0)
    if proc.poll() is not None:
        raise RuntimeError(f"ssh tunnel exited with code {proc.returncode}")
    print(
        f"[client] tunnel 127.0.0.1:{local_port} -> {remote_host}:{remote_port} via {server_ssh}:{ssh_port}",
        flush=True,
    )
    return proc


def wait_for_local_port(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except OSError as exc:
                last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"ssh tunnel local port 127.0.0.1:{port} is not ready: {last_error}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sleep_remaining(started: float, period: float) -> None:
    remaining = period - (time.time() - started)
    if remaining > 0:
        time.sleep(remaining)


def preprocess_fisheye(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")
    h, w = image.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    if top or bottom or left or right:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def blend_pose14(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    out = (1.0 - alpha) * old + alpha * new
    for arm_slice in (RIGHT_ARM_SLICE, LEFT_ARM_SLICE):
        start = arm_slice.start
        out[start + 3 : start + 6] = slerp_euler_xyz(
            old[start + 3 : start + 6],
            new[start + 3 : start + 6],
            alpha,
        )
    return out.astype(np.float32, copy=False)


def apply_hardware_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = {}
    if args.hardware_config:
        config_path = Path(args.hardware_config).expanduser()
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}

    def fill(name: str, side: str, key: str, default):
        if getattr(args, name) is None:
            setattr(args, name, cfg.get(side, {}).get(key, default))

    fill("right_piper_can", "right", "piper_can", "right_piper")
    fill("left_piper_can", "left", "piper_can", "left_piper")
    fill("right_gripper_port", "right", "gripper_port", "/dev/ttyUSB0")
    fill("left_gripper_port", "left", "gripper_port", "/dev/ttyUSB1")
    fill("right_fisheye_device", "right", "fisheye_device", None)
    fill("left_fisheye_device", "left", "fisheye_device", None)
    fill("right_fisheye_index", "right", "fisheye_index", 0)
    fill("left_fisheye_index", "left", "fisheye_index", 1)
    args.right_fisheye_device = args.right_fisheye_device or int(args.right_fisheye_index)
    args.left_fisheye_device = args.left_fisheye_device or int(args.left_fisheye_index)
    return args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDT2 async Piper real robot client")
    parser.add_argument("--server", default="127.0.0.1:18080")
    parser.add_argument("--server-ssh", default="jianan@183.230.224.121")
    parser.add_argument("--server-ssh-port", default=50210, type=int)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", default=18080, type=int)
    parser.add_argument("--local-port", default=0, type=int)
    parser.add_argument("--no-tunnel", action="store_true")
    parser.add_argument("--hardware-config", default="configs/robots/eval_bimanual_piper_pika_config.yaml")
    parser.add_argument("--right-piper-can", default=None)
    parser.add_argument("--left-piper-can", default=None)
    parser.add_argument("--right-gripper-port", default=None)
    parser.add_argument("--left-gripper-port", default=None)
    parser.add_argument("--right-fisheye-device", default=None)
    parser.add_argument("--left-fisheye-device", default=None)
    parser.add_argument("--right-fisheye-index", default=None, type=int)
    parser.add_argument("--left-fisheye-index", default=None, type=int)
    parser.add_argument("--camera-width", default=640, type=int)
    parser.add_argument("--camera-height", default=480, type=int)
    parser.add_argument("--camera-fps", default=30, type=int)
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--fps", default=30.0, type=float)
    parser.add_argument("--request-fps", default=5.0, type=float)
    parser.add_argument("--instruction", default="move")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-piper", action="store_true", help="use zero arm poses and skip Piper CAN")
    parser.add_argument("--no-pika", action="store_true", help="use zero gripper widths/images and skip Pika SDK")
    parser.add_argument("--rpc-timeout", default=10.0, type=float)
    parser.add_argument("--connect-timeout", default=10.0, type=float)
    parser.add_argument("--jpeg-quality", default=90, type=int)
    parser.add_argument("--debug-image-dir", default="")
    parser.add_argument("--debug-image-limit", default=20, type=int)
    parser.add_argument("--debug-image-every", default=1, type=int)
    parser.add_argument("--max-pos-step", default=0.01, type=float)
    parser.add_argument("--max-rot-step", default=0.05, type=float)
    parser.add_argument("--max-gripper-step", default=0.005, type=float)
    parser.add_argument("--min-gripper", default=0.0, type=float)
    parser.add_argument("--max-gripper", default=0.10, type=float)
    parser.add_argument(
        "--chunk-merge-strategy",
        default=ActionQueue.BLEND_OVERLAP,
        choices=(ActionQueue.BLEND_OVERLAP, ActionQueue.PREFIX_NO_MERGE),
        help="How incoming action chunks are merged into the execution queue",
    )
    parser.add_argument(
        "--chunk-execute-prefix-steps",
        default=0,
        type=int,
        help="For prefix_no_merge, execute only the first N future actions from each received chunk",
    )
    parser.add_argument("--run-seconds", default=0.0, type=float)
    parser.add_argument("--verbose", action="store_true")
    return apply_hardware_config(parser.parse_args())


if __name__ == "__main__":
    run(parse_args())
