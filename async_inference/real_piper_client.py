from __future__ import annotations

import argparse
import math
import queue
import signal
import sys
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import grpc
import numpy as np

from async_inference.codec import encode_jpeg, unflatten_action
from async_inference.proto import rdt2_async_pb2, rdt2_async_pb2_grpc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


LEFT_ARM_SLICE = slice(0, 7)
RIGHT_ARM_SLICE = slice(7, 14)


@dataclass
class CameraFrames:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    timestamp: float


@dataclass
class RobotObservation:
    eef_pose_flat: np.ndarray
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


class CameraWorker:
    def __init__(self, left_device: str, right_device: str, width: int, height: int, fps: int):
        self.left_device = left_device
        self.right_device = right_device
        self.width = width
        self.height = height
        self.fps = fps
        self.latest = LatestValue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="pika-camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        caps = []
        try:
            left_cap = self._open(self.left_device)
            right_cap = self._open(self.right_device)
            caps = [left_cap, right_cap]
            while not self.stop_event.is_set():
                left = self._read(left_cap)
                right = self._read(right_cap)
                self.latest.set(CameraFrames(left_rgb=left, right_rgb=right, timestamp=time.time()))
                time.sleep(0.001)
        except Exception as exc:
            print(f"[camera] stopped: {exc}")
        finally:
            for cap in caps:
                cap.release()

    def _open(self, device: str):
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open camera: {device}")
        return cap

    def _read(self, cap) -> np.ndarray:
        ok, frame = cap.read()
        if not ok or frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class PiperRobot:
    def __init__(self, can_name: str, dry_run: bool, no_piper: bool):
        self.dry_run = dry_run
        self.robot = None
        if no_piper:
            return
        from piper_sdk import C_PiperInterface

        self.robot = C_PiperInterface(can_name=can_name)
        self.robot.ConnectPort()
        while not self.robot.EnablePiper():
            time.sleep(0.01)
        self.robot.GripperCtrl(0, 1000, 0x01, 0)
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)

    def read_eef(self) -> np.ndarray:
        if self.robot is None:
            return np.zeros(7, dtype=np.float32)
        pose = self.robot.GetArmEndPoseMsgs().end_pose
        gripper = self.robot.GetArmGripperMsgs().gripper_state
        xyz = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float32) / 1_000_000.0
        rpy = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float32) / 1000.0)
        rotvec = euler_xyz_to_rotvec(rpy)
        grip = np.array([float(gripper.grippers_angle) / 1_000_000.0], dtype=np.float32)
        return np.concatenate([xyz, rotvec.astype(np.float32), grip])

    def execute(self, target: np.ndarray, effort: int) -> None:
        if self.robot is None or self.dry_run:
            return
        x, y, z, rx, ry, rz, gripper = target.tolist()
        rpy = rotvec_to_euler_xyz(np.array([rx, ry, rz], dtype=np.float64))
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.robot.EndPoseCtrl(
            int(round(x * 1_000_000.0)),
            int(round(y * 1_000_000.0)),
            int(round(z * 1_000_000.0)),
            int(round(math.degrees(rpy[0]) * 1000.0)),
            int(round(math.degrees(rpy[1]) * 1000.0)),
            int(round(math.degrees(rpy[2]) * 1000.0)),
        )
        self.robot.GripperCtrl(int(round(max(gripper, 0.0) * 1_000_000.0)), effort, 0x01, 0)


class ObservationWorker:
    def __init__(self, robot: PiperRobot, fps: float):
        self.robot = robot
        self.period = 1.0 / fps
        self.latest = LatestValue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="piper-observation", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            left = np.zeros(7, dtype=np.float32)
            right = self.robot.read_eef()
            self.latest.set(RobotObservation(np.concatenate([left, right]).astype(np.float32), time.time()))
            time.sleep(self.period)


class ActionQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._actions: list[tuple[int, np.ndarray]] = []
        self._last_chunk: Optional[np.ndarray] = None

    def add_chunk(self, first_timestep: int, action: np.ndarray, latest_executed_timestep: int) -> None:
        with self._lock:
            if self._last_chunk is not None:
                action = self._merge(self._last_chunk, action)
            self._last_chunk = action.copy()
            self._actions = [
                (first_timestep + idx, row.copy())
                for idx, row in enumerate(action)
                if first_timestep + idx > latest_executed_timestep
            ]

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
    def _merge(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        merged = new.copy()
        keep = min(2, len(old), len(new))
        blend = min(6, len(old) - keep, len(new) - keep)
        if keep > 0:
            merged[:keep] = old[:keep]
        for idx in range(blend):
            alpha = float(idx + 1) / float(blend + 1)
            merged[keep + idx] = (1.0 - alpha) * old[keep + idx] + alpha * new[keep + idx]
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
                    f"shape={list(action.shape)} queue={len(self.action_queue)} latency_ms={chunk.inference_latency_ms:.1f}"
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

    robot = PiperRobot(args.can, dry_run=args.dry_run, no_piper=args.no_piper)
    cameras = CameraWorker(args.left_camera, args.right_camera, args.camera_width, args.camera_height, args.camera_fps)
    observations = ObservationWorker(robot, args.fps)
    latest_executed = AtomicInt(-1)
    action_queue = ActionQueue()

    channel = grpc.insecure_channel(server)
    stub = rdt2_async_pb2_grpc.RDT2AsyncInferenceStub(channel)
    cameras.start()
    observations.start()
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
            args=(args, stub, cameras, observations, latest_executed, stop_event),
            daemon=True,
        )
        request_thread.start()
        control_loop(args, robot, action_queue, latest_executed, stop_event)
        request_thread.join(timeout=2.0)
    finally:
        stop_event.set()
        cameras.stop()
        observations.stop()
        if stream is not None:
            stream.join()
        channel.close()
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                tunnel.kill()


def submit_observations(args, stub, cameras, observations, latest_executed: AtomicInt, stop_event: threading.Event) -> None:
    period = 1.0 / args.request_fps
    request_id = 0
    while not stop_event.is_set():
        frames = cameras.latest.get()
        obs = observations.latest.get()
        if frames is None or obs is None:
            time.sleep(0.01)
            continue
        request_id += 1
        started = time.time()
        request = rdt2_async_pb2.ObservationRequest(
            request_id=request_id,
            timestamp=started,
            instruction=args.instruction,
            left_stereo_jpeg=encode_jpeg(frames.left_rgb, quality=args.jpeg_quality),
            right_stereo_jpeg=encode_jpeg(frames.right_rgb, quality=args.jpeg_quality),
            state=np.zeros(20, dtype=np.float32).tolist(),
            eef_pose_flat=obs.eef_pose_flat.tolist(),
            latest_executed_timestep=latest_executed.get(),
        )
        try:
            ack = stub.SubmitObservation(request, timeout=args.rpc_timeout)
            if args.verbose:
                print(
                    f"[client] submitted request_id={ack.request_id} latest={request.latest_executed_timestep} "
                    f"camera_shape={frames.left_rgb.shape}/{frames.right_rgb.shape} eef={obs.eef_pose_flat[7:14].round(4).tolist()}"
                )
        except grpc.RpcError as exc:
            print(f"[client] submit failed: {exc}")
        sleep_remaining(started, period)


def control_loop(args, robot: PiperRobot, action_queue: ActionQueue, latest_executed: AtomicInt, stop_event: threading.Event) -> None:
    period = 1.0 / args.fps
    last_target = robot.read_eef()
    while not stop_event.is_set():
        started = time.time()
        item = action_queue.pop_next(latest_executed.get())
        if item is not None:
            timestep, action = item
            target = limit_step(last_target, action[RIGHT_ARM_SLICE], args)
            if args.dry_run:
                print(f"[dry-run] timestep={timestep} target={target.round(4).tolist()} queue={len(action_queue)}")
            else:
                robot.execute(target, effort=args.gripper_effort)
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


def rotvec_to_euler_xyz(rotvec: np.ndarray) -> np.ndarray:
    matrix = rotvec_to_matrix(rotvec)
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        y = math.atan2(-matrix[2, 0], sy)
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def euler_xyz_to_rotvec(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(v) for v in rpy]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    matrix = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    return matrix_to_rotvec(matrix)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    cos_theta = float((np.trace(matrix) - 1.0) * 0.5)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    denom = 2.0 * math.sin(theta)
    axis = np.array(
        [
            (matrix[2, 1] - matrix[1, 2]) / denom,
            (matrix[0, 2] - matrix[2, 0]) / denom,
            (matrix[1, 0] - matrix[0, 1]) / denom,
        ],
        dtype=np.float64,
    )
    return axis * theta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDT2 async Piper real robot client")
    parser.add_argument("--server", default="127.0.0.1:18080")
    parser.add_argument("--server-ssh", default="jianan@183.230.224.121")
    parser.add_argument("--server-ssh-port", default=50210, type=int)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", default=18080, type=int)
    parser.add_argument("--local-port", default=0, type=int)
    parser.add_argument("--no-tunnel", action="store_true")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--left-camera", default="/dev/pika_sensor_fisheye")
    parser.add_argument("--right-camera", default="/dev/pika_gripper_fisheye")
    parser.add_argument("--camera-width", default=640, type=int)
    parser.add_argument("--camera-height", default=480, type=int)
    parser.add_argument("--camera-fps", default=30, type=int)
    parser.add_argument("--fps", default=30.0, type=float)
    parser.add_argument("--request-fps", default=5.0, type=float)
    parser.add_argument("--instruction", default="move")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-piper", action="store_true", help="use zero EEF observations and skip Piper CAN")
    parser.add_argument("--rpc-timeout", default=10.0, type=float)
    parser.add_argument("--connect-timeout", default=10.0, type=float)
    parser.add_argument("--jpeg-quality", default=90, type=int)
    parser.add_argument("--max-pos-step", default=0.01, type=float)
    parser.add_argument("--max-rot-step", default=0.05, type=float)
    parser.add_argument("--max-gripper-step", default=0.005, type=float)
    parser.add_argument("--min-gripper", default=0.0, type=float)
    parser.add_argument("--max-gripper", default=0.10, type=float)
    parser.add_argument("--gripper-effort", default=1000, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
