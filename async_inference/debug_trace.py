from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def summarize_action(
    action: np.ndarray,
    pose_slices: Iterable[slice] = (),
    static_pos_threshold: float = 0.002,
    static_rot_threshold: float = 0.01,
) -> dict[str, Any]:
    action = np.asarray(action, dtype=np.float32)
    summary: dict[str, Any] = {
        "shape": list(action.shape),
        "finite": bool(np.isfinite(action).all()),
    }
    if action.ndim != 2 or action.shape[0] == 0:
        summary["empty"] = True
        return summary

    row_delta = np.diff(action, axis=0)
    row_l2 = np.linalg.norm(row_delta, axis=1) if len(row_delta) else np.zeros((0,), dtype=np.float32)
    total_delta = action[-1] - action[0]
    summary.update(
        {
            "first": _round_list(action[0]),
            "last": _round_list(action[-1]),
            "max_row_l2": _finite_float(row_l2.max()) if row_l2.size else 0.0,
            "mean_row_l2": _finite_float(row_l2.mean()) if row_l2.size else 0.0,
            "total_l2": _finite_float(np.linalg.norm(total_delta)),
            "max_abs": _finite_float(np.max(np.abs(action))),
        }
    )

    arm_summaries = []
    for arm_idx, arm_slice in enumerate(pose_slices):
        arm = action[:, arm_slice]
        if arm.shape[1] < 6:
            continue
        pos_steps = np.linalg.norm(np.diff(arm[:, :3], axis=0), axis=1) if len(arm) > 1 else np.zeros((0,))
        rot_steps = np.linalg.norm(np.diff(arm[:, 3:6], axis=0), axis=1) if len(arm) > 1 else np.zeros((0,))
        pos_total = float(np.linalg.norm(arm[-1, :3] - arm[0, :3]))
        rot_total = float(np.linalg.norm(arm[-1, 3:6] - arm[0, 3:6]))
        arm_summaries.append(
            {
                "arm": arm_idx,
                "first": _round_list(arm[0]),
                "last": _round_list(arm[-1]),
                "max_pos_step": _finite_float(pos_steps.max()) if pos_steps.size else 0.0,
                "max_rot_step": _finite_float(rot_steps.max()) if rot_steps.size else 0.0,
                "total_pos_delta": _finite_float(pos_total),
                "total_rot_delta": _finite_float(rot_total),
                "near_static": bool(pos_total < static_pos_threshold and rot_total < static_rot_threshold),
            }
        )
    if arm_summaries:
        summary["arms"] = arm_summaries
        summary["near_static"] = all(arm["near_static"] for arm in arm_summaries)
    else:
        summary["near_static"] = bool(summary["total_l2"] < static_pos_threshold)
    return summary


class JsonlTraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        payload = {
            "wall_time": time.time(),
            **event,
        }
        line = json.dumps(_jsonable(payload), sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class StopDetector:
    def __init__(
        self,
        window_seconds: float = 0.7,
        target_pos_threshold: float = 0.01,
        target_rot_threshold: float = 0.05,
        actual_pos_threshold: float = 0.002,
        actual_rot_threshold: float = 0.01,
        emit_cooldown_seconds: float = 1.0,
    ):
        self.window_seconds = float(window_seconds)
        self.target_pos_threshold = float(target_pos_threshold)
        self.target_rot_threshold = float(target_rot_threshold)
        self.actual_pos_threshold = float(actual_pos_threshold)
        self.actual_rot_threshold = float(actual_rot_threshold)
        self.emit_cooldown_seconds = float(emit_cooldown_seconds)
        self._samples: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self._last_emit: dict[int, float] = {}

    def update(
        self,
        timestamp: float,
        target: np.ndarray,
        actual: np.ndarray,
        arm_slices: Iterable[slice],
    ) -> list[dict[str, Any]]:
        target = np.asarray(target, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        self._samples.append((float(timestamp), target.copy(), actual.copy()))
        while self._samples and timestamp - self._samples[0][0] > self.window_seconds:
            self._samples.popleft()
        if len(self._samples) < 2:
            return []

        start_t, start_target, start_actual = self._samples[0]
        events = []
        for arm_idx, arm_slice in enumerate(arm_slices):
            t0 = start_target[arm_slice]
            t1 = target[arm_slice]
            a0 = start_actual[arm_slice]
            a1 = actual[arm_slice]
            target_pos = float(np.linalg.norm(t1[:3] - t0[:3]))
            target_rot = float(np.linalg.norm(t1[3:6] - t0[3:6]))
            actual_pos = float(np.linalg.norm(a1[:3] - a0[:3]))
            actual_rot = float(np.linalg.norm(a1[3:6] - a0[3:6]))
            target_moving = target_pos >= self.target_pos_threshold or target_rot >= self.target_rot_threshold
            actual_static = actual_pos <= self.actual_pos_threshold and actual_rot <= self.actual_rot_threshold
            last_emit = self._last_emit.get(arm_idx, -math.inf)
            if target_moving and actual_static and timestamp - last_emit >= self.emit_cooldown_seconds:
                self._last_emit[arm_idx] = timestamp
                events.append(
                    {
                        "arm": arm_idx,
                        "window_seconds": timestamp - start_t,
                        "target_pos_delta": target_pos,
                        "target_rot_delta": target_rot,
                        "actual_pos_delta": actual_pos,
                        "actual_rot_delta": actual_rot,
                    }
                )
        return events


def collect_sdk_snapshot(robot: Any) -> dict[str, Any]:
    if robot is None:
        return {}
    names = (
        "GetArmStatusMsgs",
        "GetArmEndPoseMsgs",
        "GetArmJointMsgs",
        "GetArmGripperMsgs",
        "GetArmLowSpdInfoMsgs",
        "GetPiperFirmwareVersion",
    )
    snapshot = {}
    for name in names:
        func = getattr(robot, name, None)
        if func is None:
            continue
        try:
            snapshot[name] = _jsonable(func())
        except Exception as exc:
            snapshot[name] = {"error": repr(exc)}
    return snapshot


def _round_list(values: np.ndarray, digits: int = 5) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(values).reshape(-1).tolist()]


def _finite_float(value: Any) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        fields = {
            key: _jsonable(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
        if fields:
            return fields
    public_fields = {}
    for key, class_value in vars(type(value)).items():
        if key.startswith("_") or callable(class_value):
            continue
        try:
            attr = getattr(value, key)
        except Exception:
            continue
        if callable(attr):
            continue
        public_fields[key] = _jsonable(attr)
    return public_fields or repr(value)
