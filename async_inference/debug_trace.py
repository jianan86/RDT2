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


class CommandExecutionConflictDetector:
    def __init__(
        self,
        window_seconds: float = 0.7,
        command_pos_threshold: float = 0.01,
        command_rot_threshold: float = 0.05,
        target_pos_threshold: float = 0.01,
        target_rot_threshold: float = 0.05,
        actual_pos_threshold: float = 0.002,
        actual_rot_threshold: float = 0.01,
        lag_pos_threshold: float = 0.02,
        lag_rot_threshold: float = 0.10,
        emit_cooldown_seconds: float = 1.0,
    ):
        self.window_seconds = float(window_seconds)
        self.command_pos_threshold = float(command_pos_threshold)
        self.command_rot_threshold = float(command_rot_threshold)
        self.target_pos_threshold = float(target_pos_threshold)
        self.target_rot_threshold = float(target_rot_threshold)
        self.actual_pos_threshold = float(actual_pos_threshold)
        self.actual_rot_threshold = float(actual_rot_threshold)
        self.lag_pos_threshold = float(lag_pos_threshold)
        self.lag_rot_threshold = float(lag_rot_threshold)
        self.emit_cooldown_seconds = float(emit_cooldown_seconds)
        self._samples: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]] = deque()
        self._last_emit: dict[tuple[int, str], float] = {}

    def update(
        self,
        timestamp: float,
        model_action: np.ndarray,
        target: np.ndarray,
        actual: np.ndarray,
        arm_slices: Iterable[slice],
        step_diagnoses: dict[int, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        model_action = np.asarray(model_action, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        step_diagnoses = step_diagnoses or {}
        self._samples.append((float(timestamp), model_action.copy(), target.copy(), actual.copy(), dict(step_diagnoses)))
        while self._samples and timestamp - self._samples[0][0] > self.window_seconds:
            self._samples.popleft()
        if len(self._samples) < 2:
            return []

        start_t, start_action, start_target, start_actual, _ = self._samples[0]
        events = []
        for arm_idx, arm_slice in enumerate(arm_slices):
            action0 = start_action[arm_slice]
            action1 = model_action[arm_slice]
            target0 = start_target[arm_slice]
            target1 = target[arm_slice]
            actual0 = start_actual[arm_slice]
            actual1 = actual[arm_slice]
            command_pos = float(np.linalg.norm(action1[:3] - action0[:3]))
            command_rot = float(np.linalg.norm(action1[3:6] - action0[3:6]))
            command_error_pos = float(np.linalg.norm(action1[:3] - actual1[:3]))
            command_error_rot = float(np.linalg.norm(action1[3:6] - actual1[3:6]))
            target_pos = float(np.linalg.norm(target1[:3] - target0[:3]))
            target_rot = float(np.linalg.norm(target1[3:6] - target0[3:6]))
            actual_pos = float(np.linalg.norm(actual1[:3] - actual0[:3]))
            actual_rot = float(np.linalg.norm(actual1[3:6] - actual0[3:6]))
            target_error_pos = float(np.linalg.norm(target1[:3] - actual1[:3]))
            target_error_rot = float(np.linalg.norm(target1[3:6] - actual1[3:6]))
            model_commands_motion = (
                command_pos >= self.command_pos_threshold
                or command_rot >= self.command_rot_threshold
                or command_error_pos >= self.command_pos_threshold
                or command_error_rot >= self.command_rot_threshold
            )
            target_moving = target_pos >= self.target_pos_threshold or target_rot >= self.target_rot_threshold
            actual_static = actual_pos <= self.actual_pos_threshold and actual_rot <= self.actual_rot_threshold
            lagging = target_error_pos >= self.lag_pos_threshold or target_error_rot >= self.lag_rot_threshold
            oscillating = self._action_oscillates(arm_idx, arm_slice)
            step_diag = step_diagnoses.get(arm_idx, {})
            limited = bool(step_diag.get("evidence"))

            reason = "none"
            evidence = []
            if model_commands_motion:
                evidence.append("model_commands_motion")
            if target_moving:
                evidence.append("target_moving")
            if actual_static:
                evidence.append("actual_static")
            if limited:
                evidence.extend(str(item) for item in step_diag.get("evidence", []))
            if lagging:
                evidence.append("target_actual_error_high")
            if oscillating:
                evidence.append("model_command_oscillation")

            if model_commands_motion and target_moving and actual_static:
                reason = "execution_stalled"
            elif model_commands_motion and lagging:
                reason = "execution_lagging"
            elif oscillating:
                reason = "model_command_oscillation"
            elif limited:
                reason = "too_fast"

            if reason == "none":
                continue
            key = (arm_idx, reason)
            last_emit = self._last_emit.get(key, -math.inf)
            if timestamp - last_emit < self.emit_cooldown_seconds:
                continue
            self._last_emit[key] = timestamp
            events.append({
                "arm": arm_idx,
                "reason": reason,
                "window_seconds": timestamp - start_t,
                "evidence": sorted(set(evidence)),
                "commanded_pos_delta": command_pos,
                "commanded_rot_delta": command_rot,
                "command_actual_pos_error": command_error_pos,
                "command_actual_rot_error": command_error_rot,
                "target_pos_delta": target_pos,
                "target_rot_delta": target_rot,
                "actual_pos_delta": actual_pos,
                "actual_rot_delta": actual_rot,
                "target_actual_pos_error": target_error_pos,
                "target_actual_rot_error": target_error_rot,
                "diagnosis": {
                    "reason": reason,
                    "severity": "warning",
                    "errors": [],
                    "warnings": [],
                    "evidence": sorted(set(evidence)),
                },
            })
        return events

    def _action_oscillates(self, arm_idx: int, arm_slice: slice) -> bool:
        if len(self._samples) < 4:
            return False
        steps = []
        for idx in range(1, len(self._samples)):
            prev_action = self._samples[idx - 1][1][arm_slice]
            curr_action = self._samples[idx][1][arm_slice]
            delta = curr_action[:3] - prev_action[:3]
            if float(np.linalg.norm(delta)) >= self.command_pos_threshold * 0.5:
                steps.append(delta)
        if len(steps) < 2:
            return False
        flips = 0
        for prev, curr in zip(steps, steps[1:], strict=False):
            if float(np.dot(prev, curr)) < 0.0:
                flips += 1
        return flips >= 1


def collect_sdk_snapshot(robot: Any) -> dict[str, Any]:
    if robot is None:
        return {}
    names = (
        "get_connect_status",
        "GetArmStatus",
        "GetArmEnableStatus",
        "GetArmModeCtrl",
        "GetArmCtrlCode151",
        "GetArmStatusMsgs",
        "GetArmEndPoseMsgs",
        "GetArmJointMsgs",
        "GetArmGripperMsgs",
        "GetArmHighSpdInfoMsgs",
        "GetArmLowSpdInfoMsgs",
        "GetCurrentEndVelAndAccParam",
        "GetCurrentMotorAngleLimitMaxVel",
        "GetPiperFirmwareVersion",
        "GetCanFps",
    )
    snapshot = {}
    for name in names:
        func = getattr(robot, name, None)
        if func is None:
            continue
        try:
            value = func()
            item = _jsonable(value)
            if isinstance(item, dict):
                item.setdefault("_repr", repr(value))
                item.setdefault("_str", str(value))
            snapshot[name] = item
        except Exception as exc:
            snapshot[name] = {"error": repr(exc)}
    return snapshot


PIPER_REASON_PRIORITY = {
    "none": 0,
    "other": 1,
    "model_command_oscillation": 2,
    "execution_lagging": 3,
    "execution_stalled": 4,
    "too_fast": 5,
    "ik_failed": 6,
    "out_of_range": 7,
}

PIPER_ERROR_REASON_TERMS = {
    "NO_SOLUTION": "ik_failed",
    "SINGULARITY_POINT": "ik_failed",
    "TARGET_POS_EXCEEDS_LIMIT": "out_of_range",
    "EMERGENCY_STOP": "other",
    "JOINT_COMMUNICATION_ERR": "other",
    "JOINT_BRAKE_NOT_RELEASED": "other",
    "COLLISION_OCCURRED": "other",
    "JOINT_STATUS_ERR": "other",
    "OTHER_ERR": "other",
    "MAIN_CONTROLLER_NTC_OVER_TEMPERATURE": "other",
    "RELEASE_RESISTOR_NTC_OVER_TEMPERATURE": "other",
}

PIPER_WARNING_REASON_TERMS = {
    "REACH_TARGET_POS_FAILED": "ik_failed",
    "OVERSPEED_DURING_TEACHING_DRAG": "too_fast",
}

PIPER_FATAL_STATUS_TERMS = tuple(PIPER_ERROR_REASON_TERMS)
PIPER_WARNING_STATUS_TERMS = tuple(PIPER_WARNING_REASON_TERMS)

PIPER_JOINT_LIMIT_FIELDS = (
    "joint_1_angle_limit",
    "joint_2_angle_limit",
    "joint_3_angle_limit",
    "joint_4_angle_limit",
    "joint_5_angle_limit",
    "joint_6_angle_limit",
)

PIPER_LOW_SPEED_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)


def find_piper_status_errors(snapshot: dict[str, Any]) -> list[str]:
    text = json.dumps(_jsonable(snapshot), sort_keys=True)
    return [term for term in PIPER_FATAL_STATUS_TERMS if term in text]


def find_piper_status_warnings(snapshot: dict[str, Any]) -> list[str]:
    text = json.dumps(_jsonable(snapshot), sort_keys=True)
    return [term for term in PIPER_WARNING_STATUS_TERMS if term in text]


def diagnose_piper_status(snapshot: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(_jsonable(snapshot), sort_keys=True)
    errors = find_piper_status_errors(snapshot)
    warnings = find_piper_status_warnings(snapshot)
    evidence = []
    reasons = []

    for term in errors:
        reasons.append(PIPER_ERROR_REASON_TERMS[term])
        evidence.append(term)
    for term in warnings:
        reasons.append(PIPER_WARNING_REASON_TERMS[term])
        evidence.append(term)

    for field in PIPER_JOINT_LIMIT_FIELDS:
        if _json_bool_field_is_true(text, field):
            reasons.append("out_of_range")
            evidence.append(field)

    for field in PIPER_LOW_SPEED_FAULT_FIELDS:
        if _json_bool_field_is_true(text, field):
            reasons.append("other")
            evidence.append(field)

    reason = _highest_priority_reason(reasons)
    severity = "error" if errors else "warning" if warnings or evidence else "none"
    return {
        "reason": reason,
        "severity": severity,
        "errors": errors,
        "warnings": warnings,
        "evidence": sorted(set(evidence)),
    }


def diagnose_step_limit(
    current: np.ndarray,
    requested: np.ndarray,
    limited: np.ndarray,
    *,
    max_pos_step: float,
    max_rot_step: float,
    max_gripper_step: float,
    atol: float = 1e-6,
) -> dict[str, Any]:
    current = np.asarray(current, dtype=np.float32)
    requested = np.asarray(requested, dtype=np.float32)
    limited = np.asarray(limited, dtype=np.float32)
    requested_delta = requested - current
    limited_delta = limited - current
    clipped = np.abs(requested_delta - limited_delta) > atol
    evidence = []
    if bool(np.any(clipped[:3])):
        evidence.append("pos_step_limited")
    if bool(np.any(clipped[3:6])):
        evidence.append("rot_step_limited")
    if requested.shape[0] > 6 and (bool(clipped[6]) or abs(float(requested_delta[6])) > max_gripper_step + atol):
        evidence.append("gripper_step_limited")

    return {
        "reason": "too_fast" if evidence else "none",
        "evidence": evidence,
        "requested_delta": _round_list(requested_delta),
        "limited_delta": _round_list(limited_delta),
        "requested_pos_step": _finite_float(np.linalg.norm(requested_delta[:3])),
        "limited_pos_step": _finite_float(np.linalg.norm(limited_delta[:3])),
        "requested_rot_step": _finite_float(np.linalg.norm(requested_delta[3:6])),
        "limited_rot_step": _finite_float(np.linalg.norm(limited_delta[3:6])),
    }


def merge_diagnoses(*diagnoses: dict[str, Any] | None) -> dict[str, Any]:
    reasons = []
    evidence = []
    errors = []
    warnings = []
    for diagnosis in diagnoses:
        if not diagnosis:
            continue
        reason = diagnosis.get("reason", "none")
        if reason and reason != "none":
            reasons.append(str(reason))
        evidence.extend(str(item) for item in diagnosis.get("evidence", []))
        errors.extend(str(item) for item in diagnosis.get("errors", []))
        warnings.extend(str(item) for item in diagnosis.get("warnings", []))
    severity = "error" if errors else "warning" if warnings or evidence else "none"
    return {
        "reason": _highest_priority_reason(reasons),
        "severity": severity,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "evidence": sorted(set(evidence)),
    }


def _highest_priority_reason(reasons: Iterable[str]) -> str:
    best = "none"
    for reason in reasons:
        if PIPER_REASON_PRIORITY.get(reason, 0) > PIPER_REASON_PRIORITY[best]:
            best = reason
    return best


def _json_bool_field_is_true(text: str, field: str) -> bool:
    return f"\"{field}\": true" in text


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
