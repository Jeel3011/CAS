"""SE(3) / SO(3) utilities used across the pipeline."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rvec_to_rot(rvec: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rvec.reshape(3)).as_matrix()


def rot_to_rvec(rot: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot).as_rotvec()


def make_T(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def compose(*Ts: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def se3_log(T: np.ndarray) -> np.ndarray:
    """Return 6-vector (translation, rotvec)."""
    return np.concatenate([T[:3, 3], rot_to_rvec(T[:3, :3])])


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Returns (translation error in meters, rotation error in degrees)."""
    dT = invert_T(T_gt) @ T_est
    t_err = float(np.linalg.norm(dT[:3, 3]))
    rot_err_deg = float(np.rad2deg(np.linalg.norm(rot_to_rvec(dT[:3, :3]))))
    return t_err, rot_err_deg
