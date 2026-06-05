from __future__ import annotations

import numpy as np

ABS_POSE_DIM = 14
SINGLE_ARM_DIM = 7
RAW_SINGLE_ARM_ACTION_DIM = 10
RAW_POSE_DIM = 9

RIGHT_ARM_SLICE = slice(0, 7)
LEFT_ARM_SLICE = slice(7, 14)

R_EE_TCP = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
T_EE_TCP = np.eye(4, dtype=np.float32)
T_EE_TCP[:3, :3] = R_EE_TCP
T_EE_TCP[:3, 3] = np.array([0.0, 0.0, 0.1943], dtype=np.float32)
T_TCP_EE = np.linalg.inv(T_EE_TCP).astype(np.float32)


def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float32,
    )


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    sy = float(np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    if sy >= 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], sy)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        pitch = np.arctan2(-matrix[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1).astype(np.float32, copy=False)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return np.concatenate((mat[..., :, 0], mat[..., :, 1]), axis=-1)


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    return np.concatenate((mat[..., :3, 3], mat_to_rot6d(mat[..., :3, :3])), axis=-1)


def pose10d_to_mat(d10: np.ndarray) -> np.ndarray:
    d10 = np.asarray(d10, dtype=np.float32)
    out = np.zeros(d10.shape[:-1] + (4, 4), dtype=np.float32)
    out[..., :3, :3] = rot6d_to_mat(d10[..., 3:])
    out[..., :3, 3] = d10[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return normalize(quat).astype(np.float32)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(np.asarray(quat, dtype=np.float64))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def slerp_euler_xyz(old_rpy: np.ndarray, new_rpy: np.ndarray, alpha: float) -> np.ndarray:
    q0 = matrix_to_quat(euler_xyz_to_matrix(old_rpy))
    q1 = matrix_to_quat(euler_xyz_to_matrix(new_rpy))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = normalize((1.0 - alpha) * q0 + alpha * q1)
    else:
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        quat = (np.sin(theta_0 - theta) / sin_theta_0) * q0 + (np.sin(theta) / sin_theta_0) * q1
    return matrix_to_euler_xyz(quat_to_matrix(quat))

IDENTITY_POSE10D = mat_to_pose10d(np.eye(4, dtype=np.float32)).astype(np.float32)


def validate_pose14(pose_flat: np.ndarray, name: str = "pose_flat") -> np.ndarray:
    pose_flat = np.asarray(pose_flat, dtype=np.float32)
    if pose_flat.shape != (ABS_POSE_DIM,):
        raise ValueError(f"expected {name} shape ({ABS_POSE_DIM},), got {pose_flat.shape}")
    return pose_flat


def pose7d_to_mat(pose7d: np.ndarray) -> np.ndarray:
    pose7d = np.asarray(pose7d, dtype=np.float32)
    if pose7d.shape != (SINGLE_ARM_DIM,):
        raise ValueError(f"expected pose7d shape ({SINGLE_ARM_DIM},), got {pose7d.shape}")
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = euler_xyz_to_matrix(pose7d[3:6])
    mat[:3, 3] = pose7d[:3]
    return mat


def mat_to_pose7d(mat: np.ndarray, gripper_width: float) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"expected homogeneous matrix shape (4, 4), got {mat.shape}")
    pose7d = np.empty((SINGLE_ARM_DIM,), dtype=np.float32)
    pose7d[:3] = mat[:3, 3]
    pose7d[3:6] = matrix_to_euler_xyz(mat[:3, :3])
    pose7d[6] = np.float32(gripper_width)
    return pose7d


def ee_pose7d_to_tcp_pose7d(ee_pose7d: np.ndarray) -> np.ndarray:
    ee_pose7d = np.asarray(ee_pose7d, dtype=np.float32)
    base_to_tcp = pose7d_to_mat(ee_pose7d) @ T_EE_TCP
    return mat_to_pose7d(base_to_tcp, float(ee_pose7d[6]))


def tcp_pose7d_to_ee_pose7d(tcp_pose7d: np.ndarray) -> np.ndarray:
    tcp_pose7d = np.asarray(tcp_pose7d, dtype=np.float32)
    base_to_ee = pose7d_to_mat(tcp_pose7d) @ T_TCP_EE
    return mat_to_pose7d(base_to_ee, float(tcp_pose7d[6]))


def ee_pose14_to_tcp_pose14(ee_pose_flat: np.ndarray) -> np.ndarray:
    ee_pose_flat = validate_pose14(ee_pose_flat, "ee_pose_flat")
    right = ee_pose7d_to_tcp_pose7d(ee_pose_flat[RIGHT_ARM_SLICE])
    left = ee_pose7d_to_tcp_pose7d(ee_pose_flat[LEFT_ARM_SLICE])
    return np.concatenate([right, left]).astype(np.float32, copy=False)


def tcp_pose14_to_ee_pose14(tcp_pose_flat: np.ndarray) -> np.ndarray:
    tcp_pose_flat = validate_pose14(tcp_pose_flat, "tcp_pose_flat")
    right = tcp_pose7d_to_ee_pose7d(tcp_pose_flat[RIGHT_ARM_SLICE])
    left = tcp_pose7d_to_ee_pose7d(tcp_pose_flat[LEFT_ARM_SLICE])
    return np.concatenate([right, left]).astype(np.float32, copy=False)


def relative_action_to_absolute_tcp(raw_action: np.ndarray, tcp_pose_flat: np.ndarray) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    tcp_pose_flat = validate_pose14(tcp_pose_flat, "tcp_pose_flat")
    if raw_action.ndim != 2 or raw_action.shape[1] % RAW_SINGLE_ARM_ACTION_DIM != 0:
        raise ValueError(f"expected raw action shape (T, 20), got {raw_action.shape}")

    n_arms = raw_action.shape[1] // RAW_SINGLE_ARM_ACTION_DIM
    if n_arms != 2:
        raise ValueError(f"expected 2 arms in raw action, got {n_arms}")

    chunks = []
    for arm_idx in range(n_arms):
        raw_start = arm_idx * RAW_SINGLE_ARM_ACTION_DIM
        pose_start = arm_idx * SINGLE_ARM_DIM
        base_pose = tcp_pose_flat[pose_start : pose_start + SINGLE_ARM_DIM]
        base_mat = pose7d_to_mat(base_pose)
        rel_pose10d = raw_action[:, raw_start : raw_start + RAW_POSE_DIM].copy()
        bad_rot = np.linalg.norm(rel_pose10d[:, 3:9], axis=1) < 1e-8
        if np.any(bad_rot):
            rel_pose10d[bad_rot, 3:9] = IDENTITY_POSE10D[3:9]
        rel_mat = pose10d_to_mat(rel_pose10d)
        abs_mat = base_mat @ rel_mat
        grip = raw_action[:, raw_start + RAW_POSE_DIM]
        pose7 = np.stack([mat_to_pose7d(abs_mat[i], float(grip[i])) for i in range(raw_action.shape[0])])
        chunks.append(pose7)
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
