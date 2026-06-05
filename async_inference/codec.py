from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode an HWC uint8 RGB image as JPEG bytes."""
    image = _as_rgb_uint8(image)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("failed to encode JPEG")
    return encoded.tobytes()


def decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode JPEG bytes into an HWC uint8 RGB image."""
    if not payload:
        raise ValueError("empty JPEG payload")
    data = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb_png(path: str | Path, image: np.ndarray) -> None:
    """Save an HWC uint8 RGB image as PNG."""
    image = _as_rgb_uint8(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise ValueError(f"failed to write image: {path}")


def flatten_action(action: np.ndarray) -> tuple[list[float], int, int]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"expected 2D action chunk, got shape {action.shape}")
    horizon, action_dim = action.shape
    return action.reshape(-1).astype(np.float32).tolist(), int(horizon), int(action_dim)


def unflatten_action(action_flat: list[float], horizon: int, action_dim: int) -> np.ndarray:
    if horizon <= 0 or action_dim <= 0:
        raise ValueError(f"invalid action shape [{horizon}, {action_dim}]")
    action = np.asarray(action_flat, dtype=np.float32)
    expected = horizon * action_dim
    if action.size != expected:
        raise ValueError(f"expected {expected} action values, got {action.size}")
    return action.reshape(horizon, action_dim)


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)
