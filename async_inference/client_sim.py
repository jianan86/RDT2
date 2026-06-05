from __future__ import annotations

import argparse
import threading
import time

import grpc
import numpy as np

from async_inference.codec import encode_jpeg, unflatten_action
from async_inference.proto import rdt2_async_pb2, rdt2_async_pb2_grpc


def stream_actions(stub, stop_event: threading.Event) -> None:
    try:
        for chunk in stub.StreamActions(rdt2_async_pb2.ActionStreamRequest()):
            if stop_event.is_set():
                return
            if chunk.error:
                print(f"[client] action request_id={chunk.request_id} error={chunk.error}")
                continue
            action = unflatten_action(list(chunk.action_flat), chunk.horizon, chunk.action_dim)
            print(
                f"[client] action request_id={chunk.request_id} first_timestep={chunk.first_timestep} "
                f"shape={list(action.shape)} latency_ms={chunk.inference_latency_ms:.1f}"
            )
    except grpc.RpcError as exc:
        if not stop_event.is_set():
            print(f"[client] action stream failed: {exc}")


def random_rgb(height: int, width: int) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def run(args) -> None:
    channel = grpc.insecure_channel(args.server)
    stub = rdt2_async_pb2_grpc.RDT2AsyncInferenceStub(channel)

    health = stub.Health(rdt2_async_pb2.HealthRequest(), timeout=args.rpc_timeout)
    print(f"[client] health ready={health.ready} message={health.message}")

    stop_event = threading.Event()
    thread = threading.Thread(target=stream_actions, args=(stub, stop_event), daemon=True)
    thread.start()

    try:
        latest_executed_timestep = -1
        for request_id in range(1, args.num_requests + 1):
            left = random_rgb(args.image_size, args.image_size)
            right = random_rgb(args.image_size, args.image_size)
            request = rdt2_async_pb2.ObservationRequest(
                request_id=request_id,
                timestamp=time.time(),
                instruction=args.instruction,
                left_stereo_jpeg=encode_jpeg(left),
                right_stereo_jpeg=encode_jpeg(right),
                state=np.zeros(args.state_dim, dtype=np.float32).tolist(),
                tcp_pose_flat=np.zeros(14, dtype=np.float32).tolist(),
                latest_executed_timestep=latest_executed_timestep,
            )
            ack = stub.SubmitObservation(request, timeout=args.rpc_timeout)
            print(
                f"[client] submitted request_id={ack.request_id} "
                f"accepted={ack.accepted} message={ack.message}"
            )
            latest_executed_timestep += 1
            time.sleep(args.interval)

        time.sleep(args.wait_after_send)
    finally:
        stop_event.set()
        channel.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDT2 async inference simulated robot client")
    parser.add_argument("--server", default="127.0.0.1:18080")
    parser.add_argument("--instruction", default="move")
    parser.add_argument("--num-requests", default=3, type=int)
    parser.add_argument("--interval", default=0.5, type=float)
    parser.add_argument("--wait-after-send", default=5.0, type=float)
    parser.add_argument("--state-dim", default=20, type=int)
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--rpc-timeout", default=10.0, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
