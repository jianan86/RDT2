from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path
from concurrent import futures
from dataclasses import dataclass

import grpc
import numpy as np

from async_inference.codec import decode_jpeg, flatten_action, save_rgb_png
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
    latest_action: int
    action_queue_size: int
    queue_ratio: float
    must_go: bool


class RDT2AsyncService(rdt2_async_pb2_grpc.RDT2AsyncInferenceServicer):
    def __init__(
        self,
        policy,
        save_input_images: bool = False,
        debug_image_dir: str = "",
    ):
        self.policy = policy
        self.save_input_images = save_input_images
        self.debug_image_dir = debug_image_dir
        self.saved_first_input_images = False
        self.observation_queue: queue.Queue[QueuedObservation] = queue.Queue(maxsize=1)
        self.action_queue: queue.Queue[rdt2_async_pb2.ActionChunk] = queue.Queue()
        self.predicted_latest_actions: set[int] = set()
        self.predicted_latest_actions_lock = threading.Lock()
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
        with self.predicted_latest_actions_lock:
            self.predicted_latest_actions.clear()
        try:
            self.policy.reset()
        except Exception as exc:
            return rdt2_async_pb2.ResetResponse(ok=False, message=str(exc))
        return rdt2_async_pb2.ResetResponse(ok=True, message="reset")

    def SubmitObservation(self, request, context):
        latest_action = request.latest_action
        with self.predicted_latest_actions_lock:
            already_predicted = latest_action in self.predicted_latest_actions
        if already_predicted and not request.must_go:
            print(
                f"[server] filtered observation request_id={request.request_id} "
                f"latest_action={latest_action} already predicted"
            )
            return rdt2_async_pb2.ObservationAck(
                request_id=request.request_id,
                accepted=False,
                message="latest_action already predicted",
            )

        item = QueuedObservation(
            request_id=request.request_id,
            timestamp=request.timestamp,
            instruction=request.instruction,
            left_stereo_jpeg=request.left_stereo_jpeg,
            right_stereo_jpeg=request.right_stereo_jpeg,
            state=list(request.state),
            tcp_pose_flat=list(request.tcp_pose_flat),
            latest_action=latest_action,
            action_queue_size=request.action_queue_size,
            queue_ratio=request.queue_ratio,
            must_go=request.must_go,
        )
        self._save_input_images(item)
        self._put_latest(self.observation_queue, item)
        print(
            f"[server] accepted observation request_id={item.request_id} "
            f"latest_action={item.latest_action} queue_ratio={item.queue_ratio:.3f} must_go={item.must_go}"
        )
        return rdt2_async_pb2.ObservationAck(
            request_id=request.request_id,
            accepted=True,
            message="queued latest observation",
        )

    def _save_input_images(self, item: QueuedObservation) -> None:
        if not self.save_input_images:
            return
        should_save = not self.saved_first_input_images or item.request_id % 10 == 0
        if not should_save:
            return

        base = Path(self.debug_image_dir)
        try:
            save_rgb_png(base / "input_left_server.png", decode_jpeg(item.left_stereo_jpeg))
            save_rgb_png(base / "input_right_server.png", decode_jpeg(item.right_stereo_jpeg))
            self.saved_first_input_images = True
        except Exception as exc:
            print(f"[server] warning: failed to save input images request_id={item.request_id}: {exc}")


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
                with self.predicted_latest_actions_lock:
                    self.predicted_latest_actions.add(item.latest_action)
                action = self.policy.step(left, right, state, item.instruction, tcp_pose_flat)
                action_flat, horizon, action_dim = flatten_action(action)
                latency_ms = (time.time() - started) * 1000.0
                chunk = rdt2_async_pb2.ActionChunk(
                    request_id=item.request_id,
                    created_timestamp=time.time(),
                    action_flat=action_flat,
                    horizon=horizon,
                    action_dim=action_dim,
                    inference_latency_ms=latency_ms,
                    latest_action=item.latest_action,
                )
                first_action = max(item.latest_action, 0)
                last_action = first_action + horizon - 1
                print(
                    f"[server] completed request_id={item.request_id} "
                    f"latest_action={item.latest_action} action_steps={first_action}:{last_action} "
                    f"shape=[{horizon}, {action_dim}] latency_ms={latency_ms:.1f}"
                )
            except Exception as exc:
                chunk = rdt2_async_pb2.ActionChunk(
                    request_id=item.request_id,
                    created_timestamp=time.time(),
                    inference_latency_ms=(time.time() - started) * 1000.0,
                    error=str(exc),
                    latest_action=item.latest_action,
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
    debug_image_dir = args.debug_image_dir
    if args.save_input_images and not debug_image_dir:
        debug_image_dir = "debug_images"
    service = RDT2AsyncService(
        policy,
        save_input_images=args.save_input_images,
        debug_image_dir=debug_image_dir,
    )
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
    parser.add_argument("--save-input-images", action="store_true", help="save submitted left/right input images")
    parser.add_argument("--debug-image-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
