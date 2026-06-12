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
from async_inference.debug_trace import (
    CommandExecutionConflictDetector,
    JsonlTraceWriter,
    StopDetector,
    collect_sdk_snapshot,
    diagnose_piper_status,
    diagnose_step_limit,
    find_piper_status_errors,
    find_piper_status_warnings,
    merge_diagnoses,
    summarize_action,
)
from async_inference.proto import rdt2_async_pb2, rdt2_async_pb2_grpc
from async_inference.pose_utils import (
    LEFT_ARM_SLICE,
    RIGHT_ARM_SLICE,
    SINGLE_ARM_DIM,
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
                right = preprocess_fisheye(self.hardware.right_gripper.read_fisheye_rgb(self.width, self.height), self.image_size)
                if self.hardware.arm_mode == "single":
                    left = right.copy()
                else:
                    left = preprocess_fisheye(
                        self.hardware.left_gripper.read_fisheye_rgb(self.width, self.height),
                        self.image_size,
                    )
                tcp_pose_flat = self.hardware.read_tcp_pose_flat()
                self.latest.set(SensorSnapshot(left_rgb=left, right_rgb=right, tcp_pose_flat=tcp_pose_flat, timestamp=time.time()))
            except Exception as exc:
                print(f"[sensor] read failed: {exc}")
            time.sleep(0.001)


class PiperRobot:
    def __init__(
        self,
        side: str,
        can_name: str,
        dry_run: bool,
        no_piper: bool,
        sdk_trace_writer: Optional[JsonlTraceWriter] = None,
        stop_on_piper_status_error: bool = True,
    ):
        self.side = side
        self.can_name = can_name
        self.dry_run = dry_run
        self.sdk_trace_writer = sdk_trace_writer
        self.stop_on_piper_status_error = stop_on_piper_status_error
        self.last_sdk_diagnosis = {"reason": "none", "severity": "none", "errors": [], "warnings": [], "evidence": []}
        self._last_printed_status_key = None
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
        end_pose_args = (
            int(round(x * 1_000_000.0)),
            int(round(y * 1_000_000.0)),
            int(round(z * 1_000_000.0)),
            int(round(np.degrees(roll) * 1000.0)),
            int(round(np.degrees(pitch) * 1000.0)),
            int(round(np.degrees(yaw) * 1000.0)),
        )
        started = time.time()
        event = {
            "event": "piper_execute_pose",
            "side": self.side,
            "can_name": self.can_name,
            "target": np.asarray(target[:6], dtype=np.float32).round(5).tolist(),
            "end_pose_args": list(end_pose_args),
        }
        try:
            event["motion_ctrl_result"] = self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            event["end_pose_result"] = self.robot.EndPoseCtrl(*end_pose_args)
        except Exception as exc:
            event["error"] = repr(exc)
            print(f"[piper:{self.side}] execute_pose failed: {exc}")
            raise
        finally:
            event["latency_ms"] = (time.time() - started) * 1000.0
            sdk_snapshot = collect_sdk_snapshot(self.robot)
            event["sdk_snapshot"] = sdk_snapshot
            status_errors = find_piper_status_errors(sdk_snapshot)
            status_warnings = find_piper_status_warnings(sdk_snapshot)
            sdk_diagnosis = diagnose_piper_status(sdk_snapshot)
            self.last_sdk_diagnosis = sdk_diagnosis
            if status_errors:
                event["piper_status_errors"] = status_errors
            if status_warnings:
                event["piper_status_warnings"] = status_warnings
            event["diagnosis"] = sdk_diagnosis
            self._maybe_print_sdk_diagnosis(event, sdk_diagnosis)
            if self.sdk_trace_writer is not None:
                self.sdk_trace_writer.write(event)
            if status_errors and self.stop_on_piper_status_error:
                raise RuntimeError(
                    f"Piper {self.side} status error after target {event['target']}: "
                    f"{status_errors}"
                )


    def get_status_diagnosis(self) -> dict:
        return dict(self.last_sdk_diagnosis)

    def _maybe_print_sdk_diagnosis(self, event: dict, diagnosis: dict) -> None:
        if diagnosis.get("severity") == "none":
            self._last_printed_status_key = None
            return
        key = (
            diagnosis.get("severity"),
            diagnosis.get("reason"),
            tuple(diagnosis.get("errors", [])),
            tuple(diagnosis.get("warnings", [])),
            tuple(diagnosis.get("evidence", [])),
        )
        if key == self._last_printed_status_key:
            return
        self._last_printed_status_key = key
        severity = diagnosis.get("severity")
        reason = diagnosis.get("reason")
        errors = diagnosis.get("errors")
        warnings = diagnosis.get("warnings")
        evidence = diagnosis.get("evidence")
        target = event.get("target")
        print(
            f"[piper-diagnostic] side={self.side} severity={severity} "
            f"reason={reason} errors={errors} warnings={warnings} "
            f"evidence={evidence} target={target}"
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
    def __init__(self, args, sdk_trace_writer: Optional[JsonlTraceWriter] = None):
        self.arm_mode = args.arm_mode
        self.right_arm = PiperRobot(
            "right",
            args.right_piper_can,
            args.dry_run,
            args.no_piper,
            sdk_trace_writer,
            args.diagnostic_stop_on_piper_status_errors and not args.ignore_piper_status_errors,
        )
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
        self.left_arm = None
        self.left_gripper = None
        if self.arm_mode == "dual":
            self.left_arm = PiperRobot(
                "left",
                args.left_piper_can,
                args.dry_run,
                args.no_piper,
                sdk_trace_writer,
                args.diagnostic_stop_on_piper_status_errors and not args.ignore_piper_status_errors,
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
        if self.left_gripper is not None:
            self.left_gripper.close()

    def read_tcp_pose_flat(self) -> np.ndarray:
        right_ee = np.concatenate([
            self.right_arm.read_pose(),
            np.array([self.right_gripper.read_width()], dtype=np.float32),
        ])
        if self.arm_mode == "single":
            right_tcp = ee_pose14_to_tcp_pose14(np.concatenate([right_ee, right_ee]))[RIGHT_ARM_SLICE]
            return duplicate_single_arm_pose14(right_tcp)

        assert self.left_arm is not None and self.left_gripper is not None
        left_ee = np.concatenate([
            self.left_arm.read_pose(),
            np.array([self.left_gripper.read_width()], dtype=np.float32),
        ])
        return ee_pose14_to_tcp_pose14(np.concatenate([right_ee, left_ee]))

    def execute(self, tcp_target: np.ndarray) -> None:
        if self.arm_mode == "single":
            ee_target = tcp_pose14_to_ee_pose14(duplicate_single_arm_pose14(tcp_target[RIGHT_ARM_SLICE]))
            self.right_arm.execute_pose(ee_target[RIGHT_ARM_SLICE])
            self.right_gripper.execute_width(float(tcp_target[RIGHT_ARM_SLICE][6]))
            return

        assert self.left_arm is not None and self.left_gripper is not None
        ee_target = tcp_pose14_to_ee_pose14(tcp_target)
        self.right_arm.execute_pose(ee_target[RIGHT_ARM_SLICE])
        self.left_arm.execute_pose(ee_target[LEFT_ARM_SLICE])
        self.right_gripper.execute_width(float(tcp_target[RIGHT_ARM_SLICE][6]))
        self.left_gripper.execute_width(float(tcp_target[LEFT_ARM_SLICE][6]))

    def get_arm_status_diagnoses(self) -> dict[int, dict]:
        diagnoses = {0: self.right_arm.get_status_diagnosis()}
        if self.arm_mode == "single":
            diagnoses[1] = diagnoses[0]
            return diagnoses
        if self.left_arm is not None:
            diagnoses[1] = self.left_arm.get_status_diagnosis()
        return diagnoses





class ActionItem:
    timestep: int
    action: np.ndarray
    metadata: dict

    def __init__(self, timestep: int, action: np.ndarray, metadata: dict):
        self.timestep = timestep
        self.action = action
        self.metadata = metadata

    def __iter__(self):
        yield self.timestep
        yield self.action

    def __getitem__(self, index: int):
        if index == 0:
            return self.timestep
        if index == 1:
            return self.action
        raise IndexError(index)


class ActionQueue:
    PREFIX_NO_MERGE = "prefix_no_merge"
    BLEND_OVERLAP = "blend_overlap"
    RTC_SMOOTH = "rtc_smooth"
    LATEST_ONLY = "latest_only"
    DEFAULT_PREFIX_STEPS = 8

    def __init__(
        self,
        chunk_merge_strategy: str = PREFIX_NO_MERGE,
        chunk_execute_prefix_steps: int = DEFAULT_PREFIX_STEPS,
    ):
        valid = {self.PREFIX_NO_MERGE, self.BLEND_OVERLAP, self.RTC_SMOOTH, self.LATEST_ONLY}
        if chunk_merge_strategy not in valid:
            raise ValueError(f"unknown chunk_merge_strategy: {chunk_merge_strategy}")
        if chunk_merge_strategy == self.PREFIX_NO_MERGE and chunk_execute_prefix_steps <= 0:
            raise ValueError("chunk_execute_prefix_steps must be positive for prefix_no_merge")
        self._lock = threading.Lock()
        self._actions: list[ActionItem] = []
        self.chunk_merge_strategy = chunk_merge_strategy
        self.chunk_execute_prefix_steps = int(chunk_execute_prefix_steps)
        self.action_chunk_size = 0

    def add_chunk(
        self,
        chunk_latest_action: int,
        action: np.ndarray,
        current_latest_action: Optional[int] = None,
        request_id: Optional[int] = None,
    ) -> None:
        action = np.asarray(action, dtype=np.float32)
        if current_latest_action is None:
            current_latest_action = chunk_latest_action
        base_timestep = int(chunk_latest_action) + 1
        incoming = self._future_actions(
            base_timestep,
            action,
            int(current_latest_action),
            request_id=request_id,
            chunk_latest_action=chunk_latest_action,
        )

        with self._lock:
            self.action_chunk_size = max(self.action_chunk_size, int(action.shape[0]))
            if self.chunk_merge_strategy == self.PREFIX_NO_MERGE:
                self._actions = incoming[: self.chunk_execute_prefix_steps]
            elif self.chunk_merge_strategy == self.LATEST_ONLY:
                self._actions = incoming
            else:
                self._actions = self._merge_with_existing(incoming, int(current_latest_action))

    def pop_next(self, latest_action: int) -> Optional[tuple[int, np.ndarray]]:
        with self._lock:
            while self._actions and self._actions[0].timestep <= latest_action:
                self._actions.pop(0)
            if not self._actions:
                return None
            return self._actions.pop(0)

    def queue_ratio(self) -> float:
        with self._lock:
            return len(self._actions) / max(1, self.action_chunk_size)

    def __len__(self) -> int:
        with self._lock:
            return len(self._actions)

    @staticmethod
    def _future_actions(
        base_timestep: int,
        action: np.ndarray,
        latest_action: int,
        *,
        request_id: Optional[int],
        chunk_latest_action: int,
    ) -> list[ActionItem]:
        first_step = base_timestep
        last_step = base_timestep + int(action.shape[0]) - 1
        return [
            ActionItem(
                timestep=base_timestep + idx,
                action=row.copy(),
                metadata={
                    "request_id": request_id,
                    "chunk_latest_action": int(chunk_latest_action),
                    "action_index": idx,
                    "first_step": first_step,
                    "last_step": last_step,
                    "merged": False,
                },
            )
            for idx, row in enumerate(action)
            if base_timestep + idx > latest_action
        ]

    def _merge_with_existing(
        self,
        incoming: list[ActionItem],
        latest_action: int,
    ) -> list[ActionItem]:
        old_live = [item for item in self._actions if item.timestep > latest_action]
        if not incoming:
            return old_live
        if not old_live:
            return incoming

        old_by_timestep = {item.timestep: item for item in old_live}
        incoming_by_timestep = {item.timestep: item for item in incoming}
        incoming_first = incoming[0].timestep
        overlap_timesteps = [item.timestep for item in incoming if item.timestep in old_by_timestep]
        overlap_count = max(1, len(overlap_timesteps))
        overlap_index = 0
        output: dict[int, ActionItem] = {}

        for item in old_live:
            if item.timestep < incoming_first and item.timestep not in incoming_by_timestep:
                output[item.timestep] = item

        for item in incoming:
            old = old_by_timestep.get(item.timestep)
            if old is None:
                output[item.timestep] = item
                continue
            alpha = float(overlap_index + 1) / float(overlap_count + 1)
            metadata = dict(item.metadata)
            metadata["merged"] = True
            metadata["sources"] = [old.metadata, item.metadata]
            output[item.timestep] = ActionItem(
                timestep=item.timestep,
                action=blend_pose14(old.action, item.action, alpha),
                metadata=metadata,
            )
            overlap_index += 1

        return [output[ts] for ts in sorted(output)]


class ActionStreamWorker:
    def __init__(
        self,
        stub,
        action_queue: ActionQueue,
        latest_action,
        request_in_flight,
        must_go,
        stop_event: threading.Event,
        arm_mode: str,
        chunk_trace_writer: Optional[JsonlTraceWriter] = None,
    ):
        self.stub = stub
        self.action_queue = action_queue
        self.latest_action = latest_action
        self.request_in_flight = request_in_flight
        self.must_go = must_go
        self.stop_event = stop_event
        self.arm_mode = arm_mode
        self.chunk_trace_writer = chunk_trace_writer
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
                    self.request_in_flight.set(-1)
                    self.must_go.set(True)
                    print(f"[client] action request_id={chunk.request_id} error={chunk.error}")
                    continue
                action = unflatten_action(list(chunk.action_flat), chunk.horizon, chunk.action_dim)
                if action.shape[1] != 14:
                    print(f"[client] ignoring action with shape={list(action.shape)}")
                    continue
                action = adapt_action_for_arm_mode(action, self.arm_mode)
                action_summary = summarize_action(action, pose_slices=(RIGHT_ARM_SLICE, LEFT_ARM_SLICE))
                current_latest_action = self.latest_action.get()
                self.action_queue.add_chunk(chunk.latest_action, action, current_latest_action, request_id=chunk.request_id)
                self.request_in_flight.set(-1)
                self.must_go.set(True)
                first_step = chunk.latest_action + 1
                last_step = first_step + action.shape[0] - 1
                if self.chunk_trace_writer is not None:
                    self.chunk_trace_writer.write({
                        "event": "action_chunk",
                        "request_id": chunk.request_id,
                        "chunk_latest_action": chunk.latest_action,
                        "current_latest_action": current_latest_action,
                        "first_step": first_step,
                        "last_step": last_step,
                        "queue_size": len(self.action_queue),
                        "latency_ms": chunk.inference_latency_ms,
                        "summary": action_summary,
                    })
                print(
                    f"[client] chunk request_id={chunk.request_id} latest_action={chunk.latest_action} "
                    f"action_steps={first_step}:{last_step} "
                    f"shape={list(action.shape)} strategy={self.action_queue.chunk_merge_strategy} "
                    f"prefix_steps={self.action_queue.chunk_execute_prefix_steps} "
                    f"queue={len(self.action_queue)} latency_ms={chunk.inference_latency_ms:.1f} "
                    f"action_static={action_summary['near_static']} "
                    f"right_pos_delta={action_summary['arms'][0]['total_pos_delta']:.5f} "
                    f"left_pos_delta={action_summary['arms'][1]['total_pos_delta']:.5f}"
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


class AtomicBool:
    def __init__(self, value: bool):
        self._lock = threading.Lock()
        self._value = value

    def get(self) -> bool:
        with self._lock:
            return self._value

    def set(self, value: bool) -> None:
        with self._lock:
            self._value = value


class ClientDebugTrace:
    def __init__(self, args):
        self.base_dir = Path(args.debug_trace_dir).expanduser() if args.debug_trace_dir else None
        self.control_writer = None
        self.chunk_writer = None
        self.sdk_writer = None
        self._candump_files = []
        self._candump_procs = []
        if self.base_dir is not None:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.control_writer = JsonlTraceWriter(self.base_dir / "client_control.jsonl")
            self.chunk_writer = JsonlTraceWriter(self.base_dir / "client_chunks.jsonl")
            self.sdk_writer = JsonlTraceWriter(self.base_dir / "client_sdk.jsonl")
        if args.debug_candump_dir:
            self._start_candump(args)

    def _start_candump(self, args) -> None:
        candump_dir = Path(args.debug_candump_dir).expanduser()
        candump_dir.mkdir(parents=True, exist_ok=True)
        can_names = [args.right_piper_can]
        if args.arm_mode == "dual":
            can_names.append(args.left_piper_can)
        for can_name in can_names:
            if not can_name:
                continue
            path = candump_dir / f"{can_name}.log"
            try:
                f = path.open("a", encoding="utf-8")
                proc = subprocess.Popen(["candump", "-L", can_name], stdout=f, stderr=subprocess.STDOUT)
            except Exception as exc:
                print(f"[debug] failed to start candump for {can_name}: {exc}")
                continue
            self._candump_files.append(f)
            self._candump_procs.append(proc)
            print(f"[debug] candump {can_name} -> {path}")

    def close(self) -> None:
        for proc in self._candump_procs:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        for f in self._candump_files:
            f.close()


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

    debug_trace = ClientDebugTrace(args)
    hardware = BimanualHardware(args, debug_trace.sdk_writer)
    sensors = SensorWorker(hardware, args.camera_width, args.camera_height, args.image_size)
    latest_action = AtomicInt(-1)
    request_in_flight = AtomicInt(-1)
    must_go = AtomicBool(True)
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
        stream = ActionStreamWorker(
            stub, action_queue, latest_action, request_in_flight, must_go, stop_event, args.arm_mode, debug_trace.chunk_writer
        )
        stream.start()

        request_thread = threading.Thread(
            target=submit_observations,
            args=(args, stub, sensors, latest_action, action_queue, request_in_flight, must_go, stop_event),
            daemon=True,
        )
        request_thread.start()
        control_loop(args, hardware, sensors, action_queue, latest_action, stop_event, debug_trace.control_writer)
        request_thread.join(timeout=2.0)
    finally:
        stop_event.set()
        sensors.stop()
        hardware.close()
        debug_trace.close()
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


def submit_observations(
    args,
    stub,
    sensors,
    latest_action: AtomicInt,
    action_queue: ActionQueue,
    request_in_flight: AtomicInt,
    must_go: AtomicBool,
    stop_event: threading.Event,
) -> None:
    period = 1.0 / (args.request_fps if args.observation_request_policy == "fixed_fps" else args.fps)
    request_id = 0
    while not stop_event.is_set():
        started = time.time()
        latest = latest_action.get()
        queue_size = len(action_queue)
        queue_ratio = action_queue.queue_ratio()
        force_request = must_go.get() and queue_size == 0

        if args.observation_request_policy == "threshold":
            if request_in_flight.get() >= 0:
                sleep_remaining(started, period)
                continue
            if not force_request and queue_ratio > args.chunk_size_threshold:
                sleep_remaining(started, period)
                continue

        snapshot = sensors.latest.get()
        if snapshot is None:
            time.sleep(0.01)
            continue

        request_id += 1
        maybe_save_debug_images(args, request_id, snapshot.left_rgb, snapshot.right_rgb)
        request = rdt2_async_pb2.ObservationRequest(
            request_id=request_id,
            timestamp=snapshot.timestamp,
            instruction=args.instruction,
            left_stereo_jpeg=encode_jpeg(snapshot.left_rgb, quality=args.jpeg_quality),
            right_stereo_jpeg=encode_jpeg(snapshot.right_rgb, quality=args.jpeg_quality),
            state=np.zeros(20, dtype=np.float32).tolist(),
            tcp_pose_flat=snapshot.tcp_pose_flat.tolist(),
            latest_action=latest,
            action_queue_size=queue_size,
            queue_ratio=queue_ratio,
            must_go=force_request,
        )
        request_in_flight.set(request_id)
        try:
            ack = stub.SubmitObservation(request, timeout=args.rpc_timeout)
            if not ack.accepted:
                request_in_flight.set(-1)
            elif force_request:
                must_go.set(False)
            if args.verbose:
                print(
                    f"[client] submitted request_id={ack.request_id} accepted={ack.accepted} "
                    f"latest_action={request.latest_action} "
                    f"queue={queue_size} queue_ratio={queue_ratio:.3f} must_go={force_request} "
                    f"camera_shape={snapshot.left_rgb.shape}/{snapshot.right_rgb.shape} "
                    f"right_tcp={snapshot.tcp_pose_flat[RIGHT_ARM_SLICE].round(4).tolist()} "
                    f"left_tcp={snapshot.tcp_pose_flat[LEFT_ARM_SLICE].round(4).tolist()}"
                )
        except grpc.RpcError as exc:
            request_in_flight.set(-1)
            must_go.set(True)
            print(f"[client] submit failed: {exc}")
        sleep_remaining(started, period)


def control_loop(
    args,
    hardware: BimanualHardware,
    sensors: SensorWorker,
    action_queue: ActionQueue,
    latest_action: AtomicInt,
    stop_event: threading.Event,
    control_trace_writer: Optional[JsonlTraceWriter] = None,
) -> None:
    period = 1.0 / args.fps
    last_target = hardware.read_tcp_pose_flat()
    stop_detector = StopDetector(
        window_seconds=args.stop_detector_window,
        target_pos_threshold=args.stop_target_pos_threshold,
        target_rot_threshold=args.stop_target_rot_threshold,
        actual_pos_threshold=args.stop_actual_pos_threshold,
        actual_rot_threshold=args.stop_actual_rot_threshold,
    )
    conflict_detector = CommandExecutionConflictDetector(
        window_seconds=args.stop_detector_window,
        command_pos_threshold=args.stop_target_pos_threshold,
        command_rot_threshold=args.stop_target_rot_threshold,
        target_pos_threshold=args.stop_target_pos_threshold,
        target_rot_threshold=args.stop_target_rot_threshold,
        actual_pos_threshold=args.stop_actual_pos_threshold,
        actual_rot_threshold=args.stop_actual_rot_threshold,
        lag_pos_threshold=max(args.stop_target_pos_threshold, args.stop_actual_pos_threshold * 5.0),
        lag_rot_threshold=max(args.stop_target_rot_threshold, args.stop_actual_rot_threshold * 5.0),
    )
    arm_slices = (RIGHT_ARM_SLICE, LEFT_ARM_SLICE)
    deadline = None if args.run_seconds <= 0 else time.time() + args.run_seconds
    while not stop_event.is_set():
        if deadline is not None and time.time() >= deadline:
            stop_event.set()
            break
        started = time.time()
        item = action_queue.pop_next(latest_action.get())
        if item is not None:
            timestep, action = item
            action_source = getattr(item, "metadata", {})
            target = last_target.copy()
            target[RIGHT_ARM_SLICE] = limit_step(last_target[RIGHT_ARM_SLICE], action[RIGHT_ARM_SLICE], args)
            target[LEFT_ARM_SLICE] = limit_step(last_target[LEFT_ARM_SLICE], action[LEFT_ARM_SLICE], args)
            step_diagnoses = {
                0: diagnose_step_limit(
                    last_target[RIGHT_ARM_SLICE],
                    action[RIGHT_ARM_SLICE],
                    target[RIGHT_ARM_SLICE],
                    max_pos_step=args.max_pos_step,
                    max_rot_step=args.max_rot_step,
                    max_gripper_step=args.max_gripper_step,
                ),
                1: diagnose_step_limit(
                    last_target[LEFT_ARM_SLICE],
                    action[LEFT_ARM_SLICE],
                    target[LEFT_ARM_SLICE],
                    max_pos_step=args.max_pos_step,
                    max_rot_step=args.max_rot_step,
                    max_gripper_step=args.max_gripper_step,
                ),
            }
            if args.dry_run:
                print(
                    f"[dry-run] timestep={timestep} "
                    f"right={target[RIGHT_ARM_SLICE].round(4).tolist()} "
                    f"left={target[LEFT_ARM_SLICE].round(4).tolist()} queue={len(action_queue)}"
                )
            else:
                hardware.execute(target)
            snapshot = sensors.latest.get()
            actual = snapshot.tcp_pose_flat.copy() if snapshot is not None else hardware.read_tcp_pose_flat()
            action_summary = summarize_action(action[None, :], pose_slices=arm_slices)
            target_delta = target - last_target
            actual_error = target - actual
            sdk_diagnoses = hardware.get_arm_status_diagnoses()
            command_execution_conflicts = conflict_detector.update(
                time.time(),
                action,
                target,
                actual,
                arm_slices,
                step_diagnoses,
            )
            for event in command_execution_conflicts:
                arm_idx = int(event["arm"])
                event["diagnosis"] = merge_diagnoses(
                    sdk_diagnoses.get(arm_idx),
                    step_diagnoses.get(arm_idx),
                    event.get("diagnosis"),
                )
            stop_events = stop_detector.update(time.time(), target, actual, arm_slices)
            for event in stop_events:
                arm_idx = int(event["arm"])
                event["diagnosis"] = merge_diagnoses(
                    sdk_diagnoses.get(arm_idx),
                    step_diagnoses.get(arm_idx),
                )
            if control_trace_writer is not None:
                control_trace_writer.write({
                    "event": "control_step",
                    "timestep": timestep,
                    "queue_size": len(action_queue),
                    "action": np.asarray(action, dtype=np.float32).round(5).tolist(),
                    "action_source": action_source,
                    "target": target.round(5).tolist(),
                    "last_target": last_target.round(5).tolist(),
                    "actual": actual.round(5).tolist(),
                    "target_delta": target_delta.round(5).tolist(),
                    "actual_error": actual_error.round(5).tolist(),
                    "action_summary": action_summary,
                    "step_diagnoses": step_diagnoses,
                    "sdk_diagnoses": sdk_diagnoses,
                    "command_execution_conflicts": command_execution_conflicts,
                    "stop_events": stop_events,
                })
            for event in command_execution_conflicts:
                diagnosis = event.get("diagnosis", {})
                print(
                    f"[command-exec-conflict] arm={event['arm']} timestep={timestep} "
                    f"reason={diagnosis.get('reason', event.get('reason', 'other'))} "
                    f"evidence={diagnosis.get('evidence', event.get('evidence', []))} "
                    f"window={event['window_seconds']:.2f}s "
                    f"commanded_pos_delta={event['commanded_pos_delta']:.5f} "
                    f"target_pos_delta={event['target_pos_delta']:.5f} "
                    f"actual_pos_delta={event['actual_pos_delta']:.5f} "
                    f"target_actual_pos_error={event['target_actual_pos_error']:.5f}"
                )
            for event in stop_events:
                diagnosis = event.get("diagnosis", {})
                print(
                    f"[stop-detector] arm={event['arm']} timestep={timestep} "
                    f"reason={diagnosis.get('reason', 'other')} "
                    f"evidence={diagnosis.get('evidence', [])} "
                    f"window={event['window_seconds']:.2f}s "
                    f"target_pos_delta={event['target_pos_delta']:.5f} "
                    f"actual_pos_delta={event['actual_pos_delta']:.5f} "
                    f"target_rot_delta={event['target_rot_delta']:.5f} "
                    f"actual_rot_delta={event['actual_rot_delta']:.5f}"
                )
            last_target = target
            latest_action.set(timestep)
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


def duplicate_single_arm_pose14(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape != (SINGLE_ARM_DIM,):
        raise ValueError(f"expected single arm pose shape ({SINGLE_ARM_DIM},), got {pose7.shape}")
    return np.concatenate([pose7, pose7]).astype(np.float32, copy=False)


def adapt_action_for_arm_mode(action: np.ndarray, arm_mode: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if arm_mode == "dual":
        return action
    if arm_mode == "single":
        right_action = action[:, RIGHT_ARM_SLICE]
        return np.concatenate([right_action, right_action], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"unsupported arm_mode: {arm_mode}")


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
    parser.add_argument("--arm-mode", choices=("dual", "single"), default="dual")
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
    parser.add_argument("--request-fps", default=5.0, type=float, help="Used only with --observation-request-policy=fixed_fps")
    parser.add_argument(
        "--observation-request-policy",
        default="threshold",
        choices=("threshold", "fixed_fps"),
        help="threshold matches LeRobot-style queue waterline scheduling; fixed_fps preserves periodic submit",
    )
    parser.add_argument("--chunk-size-threshold", default=0.5, type=float)
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
    parser.add_argument("--debug-trace-dir", default="", help="write JSONL control/chunk/SDK traces to this directory")
    parser.add_argument("--debug-candump-dir", default="", help="best-effort candump logs for Piper CAN interfaces")
    parser.add_argument(
        "--diagnostic-stop-on-piper-status-errors",
        action="store_true",
        help="diagnostic mode: stop the client when Piper reports fatal target-limit status",
    )
    parser.add_argument(
        "--ignore-piper-status-errors",
        action="store_true",
        help="compatibility no-op unless diagnostic stop mode is enabled",
    )
    parser.add_argument("--stop-detector-window", default=0.7, type=float)
    parser.add_argument("--stop-target-pos-threshold", default=0.01, type=float)
    parser.add_argument("--stop-target-rot-threshold", default=0.05, type=float)
    parser.add_argument("--stop-actual-pos-threshold", default=0.002, type=float)
    parser.add_argument("--stop-actual-rot-threshold", default=0.01, type=float)
    parser.add_argument("--max-pos-step", default=0.01, type=float)
    parser.add_argument("--max-rot-step", default=0.05, type=float)
    parser.add_argument("--max-gripper-step", default=0.005, type=float)
    parser.add_argument("--min-gripper", default=0.0, type=float)
    parser.add_argument("--max-gripper", default=0.10, type=float)
    parser.add_argument(
        "--chunk-merge-strategy",
        default=ActionQueue.PREFIX_NO_MERGE,
        choices=(
            ActionQueue.PREFIX_NO_MERGE,
            ActionQueue.BLEND_OVERLAP,
            ActionQueue.RTC_SMOOTH,
            ActionQueue.LATEST_ONLY,
        ),
        help="How incoming action chunks are merged into the execution queue",
    )
    parser.add_argument(
        "--chunk-execute-prefix-steps",
        default=ActionQueue.DEFAULT_PREFIX_STEPS,
        type=int,
        help="For prefix_no_merge, execute only the first N future actions from each received chunk",
    )
    parser.add_argument("--run-seconds", default=0.0, type=float)
    parser.add_argument("--verbose", action="store_true")
    return apply_hardware_config(parser.parse_args())


if __name__ == "__main__":
    run(parse_args())
