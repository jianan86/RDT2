from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from async_inference.debug_trace import summarize_action
from async_inference.pose_utils import ABS_POSE_DIM, relative_action_to_absolute_tcp, validate_pose14


DEFAULT_MODEL_CONFIG = "configs/rdt/post_train.yaml"
DEFAULT_PRETRAINED_PATH = "/data/jianan/rdt2/RDT2-FM"
DEFAULT_VLM_PATH = "/data/jianan/rdt2/RDT2-VQ"
DEFAULT_QWEN_PROCESSOR_PATH = "/data/jianan/rdt2/Qwen2.5-VL-7B-Instruct"
DEFAULT_NORMALIZER_PATH = "/data/jianan/rdt2/RVQActionTokenizer/umi_normalizer_wo_downsample_indentity_rot.pt"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_STATE_DIM = 20
ABS_EEF_DIM = ABS_POSE_DIM


class DummyRDT2Policy:
    def __init__(self, horizon: int = 24, action_dim: int = ABS_EEF_DIM, state_dim: int = DEFAULT_STATE_DIM):
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.ready = True

    def reset(self) -> None:
        return None

    def warmup(self, instruction: str) -> None:
        return None

    def step(
        self,
        left_stereo: np.ndarray,
        right_stereo: np.ndarray,
        state: np.ndarray,
        instruction: str,
        tcp_pose_flat: np.ndarray,
    ) -> np.ndarray:
        self._validate_state(state)
        tcp_pose_flat = validate_pose14(tcp_pose_flat, "tcp_pose_flat")
        return np.repeat(tcp_pose_flat[None, :], self.horizon, axis=0).astype(np.float32)

    def _validate_state(self, state: np.ndarray) -> None:
        if np.asarray(state).shape != (self.state_dim,):
            raise ValueError(f"expected state shape ({self.state_dim},), got {np.asarray(state).shape}")


class RDT2Policy:
    def __init__(
        self,
        pretrained_path: str = DEFAULT_PRETRAINED_PATH,
        vlm_path: str = DEFAULT_VLM_PATH,
        qwen_processor_path: str = DEFAULT_QWEN_PROCESSOR_PATH,
        normalizer_path: str = DEFAULT_NORMALIZER_PATH,
        model_config: str = DEFAULT_MODEL_CONFIG,
        device: str = "cuda:0",
        warmup: bool = True,
        compile_model: bool = True,
    ):
        import torch
        from models.rdt_inferencer import RDTInferencer

        self.torch = torch
        self.device = torch.device(device)
        self.config = _load_yaml(model_config)
        self.state_dim = int(self.config["common"]["state_dim"])
        self.horizon = int(self.config["common"]["action_chunk_size"])
        self.raw_action_dim = int(self.config["common"]["action_dim"])
        self.action_dim = ABS_EEF_DIM
        self.ready = False

        with _patched_qwen_processor(qwen_processor_path), _patched_torch_compile(compile_model):
            self.model = RDTInferencer(
                config=self.config,
                pretrained_path=pretrained_path,
                normalizer_path=normalizer_path,
                pretrained_vision_language_model_name_or_path=vlm_path,
                device=self.device,
                dtype=torch.bfloat16,
            )

        self.ready = True
        if warmup:
            self.warmup("move")

    def reset(self) -> None:
        self.model.reset()

    def warmup(self, instruction: str) -> None:
        left = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
        right = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
        state = np.zeros(self.state_dim, dtype=np.float32)
        tcp_pose_flat = np.zeros(ABS_EEF_DIM, dtype=np.float32)
        self.step(left, right, state, instruction, tcp_pose_flat)

    def step(
        self,
        left_stereo: np.ndarray,
        right_stereo: np.ndarray,
        state: np.ndarray,
        instruction: str,
        tcp_pose_flat: np.ndarray,
    ) -> np.ndarray:
        tcp_pose_flat = validate_pose14(tcp_pose_flat, "tcp_pose_flat")

        left_stereo = _validate_image(left_stereo)
        right_stereo = _validate_image(right_stereo)
        state = np.zeros(self.state_dim, dtype=np.float32)

        started = time.time()
        with self.torch.no_grad():
            result = self.model.step(
                observations={
                    "images": {
                        "left_stereo": left_stereo,
                        "right_stereo": right_stereo,
                    },
                    "state": state,
                },
                instruction=instruction,
            )
        raw_action = result.detach().cpu().numpy() if hasattr(result, "detach") else np.asarray(result)
        raw_action = raw_action.astype(np.float32, copy=False)
        if raw_action.shape != (self.horizon, self.raw_action_dim):
            raise ValueError(
                f"expected raw action shape ({self.horizon}, {self.raw_action_dim}), got {raw_action.shape}"
            )

        action = relative_action_to_absolute_tcp(raw_action, tcp_pose_flat)
        action = _postprocess_gripper(action.astype(np.float32, copy=False))
        if action.shape != (self.horizon, self.action_dim):
            raise ValueError(f"expected action shape ({self.horizon}, {self.action_dim}), got {action.shape}")
        raw_summary = summarize_action(raw_action)
        action_summary = summarize_action(action, pose_slices=(slice(0, 7), slice(7, 14)))
        print(
            f"[policy] inference+postprocess latency {(time.time() - started) * 1000:.1f} ms "
            f"raw_total_l2={raw_summary['total_l2']:.5f} action_static={action_summary['near_static']} "
            f"right_pos_delta={action_summary['arms'][0]['total_pos_delta']:.5f} "
            f"left_pos_delta={action_summary['arms'][1]['total_pos_delta']:.5f}"
        )
        return action


def _validate_image(image: np.ndarray, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")
    if image.shape[:2] != (size, size):
        raise ValueError(f"expected {size}x{size} RGB image, got shape {image.shape}")
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def _postprocess_gripper(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    for robot_idx in range(action.shape[1] // 7):
        grip_col = robot_idx * 7 + 6
        action[:, grip_col] = action[:, grip_col] / 0.088 * 0.10
        if action[0, grip_col] > action[-1, grip_col]:
            closure_delta = action[0, grip_col] - action[-1, grip_col]
            action[:, grip_col] -= max(float(closure_delta) * 0.2, 0.010)
    return action


def _load_yaml(path: str) -> dict:
    with Path(path).expanduser().open("r") as f:
        return yaml.safe_load(f)


@contextmanager
def _patched_qwen_processor(qwen_processor_path: str) -> Iterator[None]:
    import models.rdt_inferencer as rdt_inferencer

    original = rdt_inferencer.AutoProcessor.from_pretrained

    def from_pretrained(name_or_path, *args, **kwargs):
        if name_or_path == "Qwen/Qwen2.5-VL-7B-Instruct":
            name_or_path = qwen_processor_path
        return original(name_or_path, *args, **kwargs)

    rdt_inferencer.AutoProcessor.from_pretrained = from_pretrained
    try:
        yield
    finally:
        rdt_inferencer.AutoProcessor.from_pretrained = original


@contextmanager
def _patched_torch_compile(enabled: bool) -> Iterator[None]:
    import torch

    if enabled:
        yield
        return

    original = torch.compile
    torch.compile = lambda model, *args, **kwargs: model
    try:
        yield
    finally:
        torch.compile = original
