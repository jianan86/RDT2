from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from concurrent import futures
from dataclasses import dataclass

import cv2
import grpc
import numpy as np

from async_inference.codec import decode_jpeg, flatten_action
from async_inference.proto import rdt2_async_pb2, rdt2_async_pb2_grpc
from async_inference.rdt2_policy import (
    ABS_EEF_DIM,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_NORMALIZER_PATH,
    DEFAULT_PRETRAINED_PATH,
    DEFAULT_QWEN_PROCESSOR_PATH,
    DEFAULT_STATE_DIM,
    DEFAULT_VLM_PATH,
    DummyRDT2Policy,
    RDT2Policy,
)


@dataclass(frozen=True)
class QueuedObservation:
    request_id: int
    timestamp: float
    instruction: str
    left_stereo_jpeg: bytes
    right_stereo_jpeg: bytes
    state: list[float]
    tcp_pose_flat: list[float]
    latest_executed_timestep: int


class LatestImageViewer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.queue: queue.Queue[tuple[bytes, bytes, int, float]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.window_created = False
        self.thread = threading.Thread(target=self._loop, name="rdt2-input-image-viewer", daemon=True)
        self.thread.start()

    def submit(self, left_jpeg: bytes, right_jpeg: bytes, request_id: int, timestamp: float) -> None:
        if self.stop_event.is_set():
            return
        item = (left_jpeg, right_jpeg, request_id, timestamp)
        while not self.stop_event.is_set():
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            if self._is_headless_linux():
                print("[server] warning: input image viewer disabled: no DISPLAY or WAYLAND_DISPLAY")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                try:
                    left_jpeg, right_jpeg, request_id, timestamp = self.queue.get(timeout=0.2)
                except queue.Empty:
                    self._wait_key()
                    continue

                try:
                    image = self._make_display_image(left_jpeg, right_jpeg, request_id, timestamp)
                    cv2.imshow(self.window_name, image)
                    self.window_created = True
                    if self._wait_key() == ord("q"):
                        self.stop_event.set()
                except Exception as exc:
                    print(f"[server] warning: input image viewer stopped: {exc}")
                    self.stop_event.set()
        finally:
            if self.window_created:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error as exc:
                    print(f"[server] warning: input image viewer destroyWindow failed: {exc}")

    def _wait_key(self) -> int:
        try:
            return cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            print(f"[server] warning: input image viewer waitKey failed: {exc}")
            self.stop_event.set()
            return -1

    @staticmethod
    def _make_display_image(left_jpeg: bytes, right_jpeg: bytes, request_id: int, timestamp: float) -> np.ndarray:
        left = decode_jpeg(left_jpeg)
        right = decode_jpeg(right_jpeg)
        if left.shape[0] != right.shape[0]:
            scale = left.shape[0] / right.shape[0]
            right = cv2.resize(right, (max(1, int(right.shape[1] * scale)), left.shape[0]))
        image = np.concatenate([left, right], axis=1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label = f"request_id={request_id} timestamp={timestamp:.3f}"
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    @staticmethod
    def _is_headless_linux() -> bool:
        return sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class RDT2AsyncService(rdt2_async_pb2_grpc.RDT2AsyncInferenceServicer):
    def __init__(self, policy, image_viewer: LatestImageViewer | None = None):
        self.policy = policy
        self.image_viewer = image_viewer
        self.observation_queue: queue.Queue[QueuedObservation] = queue.Queue(maxsize=1)
        self.action_queue: queue.Queue[rdt2_async_pb2.ActionChunk] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, name="rdt2-inference-worker", daemon=True)
        self.worker.start()

    def Health(self, request, context):
        ready = bool(getattr(self.policy, "ready", False))
        message = "ready" if ready else getattr(self.policy, "error", "policy is not ready")
        return rdt2_async_pb2.HealthResponse(ready=ready, message=message)

    def Reset(self, request, context):
        self._drain(self.observation_queue)
        self._drain(self.action_queue)
        try:
            self.policy.reset()
        except Exception as exc:
            return rdt2_async_pb2.ResetResponse(ok=False, message=str(exc))
        return rdt2_async_pb2.ResetResponse(ok=True, message="reset")

    def SubmitObservation(self, request, context):
        item = QueuedObservation(
            request_id=request.request_id,
            timestamp=request.timestamp,
            instruction=request.instruction,
            left_stereo_jpeg=request.left_stereo_jpeg,
            right_stereo_jpeg=request.right_stereo_jpeg,
            state=list(request.state),
            tcp_pose_flat=list(request.tcp_pose_flat),
            latest_executed_timestep=request.latest_executed_timestep,
        )
        if self.image_viewer is not None:
            self.image_viewer.submit(
                item.left_stereo_jpeg,
                item.right_stereo_jpeg,
                item.request_id,
                item.timestamp,
            )
        self._put_latest(self.observation_queue, item)
        print(
            f"[server] accepted observation request_id={item.request_id} "
            f"latest_executed_timestep={item.latest_executed_timestep}"
        )
        return rdt2_async_pb2.ObservationAck(
            request_id=request.request_id,
            accepted=True,
            message="queued latest observation",
        )

    def StreamActions(self, request, context):
        print("[server] action stream connected")
        while context.is_active() and not self.stop_event.is_set():
            try:
                chunk = self.action_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            yield chunk
        print("[server] action stream disconnected")

    def shutdown(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.observation_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            started = time.time()
            try:
                left = decode_jpeg(item.left_stereo_jpeg)
                right = decode_jpeg(item.right_stereo_jpeg)
                state = np.asarray(item.state or [0.0] * DEFAULT_STATE_DIM, dtype=np.float32)
                tcp_pose_flat = np.asarray(item.tcp_pose_flat, dtype=np.float32)
                action = self.policy.step(left, right, state, item.instruction, tcp_pose_flat)
                action_flat, horizon, action_dim = flatten_action(action)
                latency_ms = (time.time() - started) * 1000.0
                first_timestep = item.latest_executed_timestep + 1
                chunk = rdt2_async_pb2.ActionChunk(
                    request_id=item.request_id,
                    created_timestamp=time.time(),
                    action_flat=action_flat,
                    horizon=horizon,
                    action_dim=action_dim,
                    inference_latency_ms=latency_ms,
                    first_timestep=first_timestep,
                )
                print(
                    f"[server] completed request_id={item.request_id} "
                    f"first_timestep={first_timestep} shape=[{horizon}, {action_dim}] latency_ms={latency_ms:.1f}"
                )
            except Exception as exc:
                chunk = rdt2_async_pb2.ActionChunk(
                    request_id=item.request_id,
                    created_timestamp=time.time(),
                    inference_latency_ms=(time.time() - started) * 1000.0,
                    error=str(exc),
                    first_timestep=item.latest_executed_timestep + 1,
                )
                print(f"[server] request_id={item.request_id} failed: {exc}")

            self.action_queue.put(chunk)

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _put_latest(q: queue.Queue, item: QueuedObservation) -> None:
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


class FailedPolicy:
    def __init__(self, error: str):
        self.ready = False
        self.error = error

    def reset(self) -> None:
        raise RuntimeError(self.error)

    def step(self, left_stereo, right_stereo, state, instruction, tcp_pose_flat):
        raise RuntimeError(self.error)


def build_policy(args):
    if args.dummy:
        return DummyRDT2Policy(
            horizon=args.expected_horizon,
            action_dim=args.expected_action_dim,
            state_dim=args.expected_state_dim,
        )
    return RDT2Policy(
        pretrained_path=args.pretrained_path,
        vlm_path=args.vlm_path,
        qwen_processor_path=args.qwen_processor_path,
        normalizer_path=args.normalizer_path,
        model_config=args.model_config,
        device=args.device,
        warmup=not args.no_warmup,
        compile_model=args.compile_model and not args.no_compile,
    )


def serve(args) -> None:
    try:
        policy = build_policy(args)
    except Exception as exc:
        policy = FailedPolicy(f"policy load failed: {exc}")
        print(f"[server] {policy.error}")
    image_viewer = None
    if args.visualize_input_images:
        image_viewer = LatestImageViewer(args.visualize_window_name)
    service = RDT2AsyncService(policy, image_viewer=image_viewer)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.grpc_workers))
    rdt2_async_pb2_grpc.add_RDT2AsyncInferenceServicer_to_server(service, server)
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    server.start()
    print(f"[server] listening on {bind_addr}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("[server] shutting down")
    finally:
        service.shutdown()
        if image_viewer is not None:
            image_viewer.close()
        server.stop(grace=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDT2 async gRPC inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18080, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dummy", action="store_true", help="return hold action chunks without loading RDT2")
    parser.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED_PATH)
    parser.add_argument("--vlm-path", default=DEFAULT_VLM_PATH)
    parser.add_argument("--qwen-processor-path", default=DEFAULT_QWEN_PROCESSOR_PATH)
    parser.add_argument("--normalizer-path", default=DEFAULT_NORMALIZER_PATH)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--expected-horizon", default=24, type=int)
    parser.add_argument("--expected-action-dim", default=ABS_EEF_DIM, type=int)
    parser.add_argument("--expected-state-dim", default=DEFAULT_STATE_DIM, type=int)
    parser.add_argument("--grpc-workers", default=8, type=int)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--compile-model", action="store_true", help="opt in to torch.compile for the RDT policy")
    parser.add_argument("--no-compile", action="store_true", help="kept for compatibility; compile is disabled by default")
    parser.add_argument("--visualize-input-images", action="store_true", help="show the latest submitted left/right input images")
    parser.add_argument("--visualize-window-name", default="RDT2 Async Input Images")
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
