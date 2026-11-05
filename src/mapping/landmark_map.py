"""Persistent landmark (ArUco) map.

Initialization strategy: pick the first-observed marker as the world anchor
(^W T_M0 = I).  Every subsequent marker is initialized from the current camera
pose and the camera->marker measurement:  ^W T_Mi = ^W T_Ck · ^C T_Mi.

A running average filters repeated observations of each landmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.utils.transforms import compose, invert_T, make_T, rot_to_rvec, rvec_to_rot


@dataclass
class Landmark:
    marker_id: int
    T_world_marker: np.ndarray
    observation_count: int = 0


@dataclass
class LandmarkMap:
    anchors: dict[int, np.ndarray] = field(default_factory=dict)   # prior known marker poses
    landmarks: dict[int, Landmark] = field(default_factory=dict)

    def set_anchor(self, marker_id: int, T_world_marker: np.ndarray) -> None:
        self.anchors[marker_id] = T_world_marker.copy()
        self.landmarks[marker_id] = Landmark(marker_id, T_world_marker.copy(), observation_count=1)

    def observe(self, marker_id: int, T_world_cam: np.ndarray, T_cam_marker: np.ndarray) -> None:
        T_wm = compose(T_world_cam, T_cam_marker)
        if marker_id in self.anchors:
            return  # anchored — world pose is fixed
        if marker_id not in self.landmarks:
            self.landmarks[marker_id] = Landmark(marker_id, T_wm, observation_count=1)
            return
        lm = self.landmarks[marker_id]
        lm.observation_count += 1
        alpha = 1.0 / lm.observation_count
        # Simple averaging — translation lerp, rotation via scaled rotvec diff.
        lm.T_world_marker[:3, 3] = (1 - alpha) * lm.T_world_marker[:3, 3] + alpha * T_wm[:3, 3]
        r_old = rot_to_rvec(lm.T_world_marker[:3, :3])
        r_new = rot_to_rvec(T_wm[:3, :3])
        lm.T_world_marker[:3, :3] = rvec_to_rot((1 - alpha) * r_old + alpha * r_new)

    def camera_pose_from_marker(self, marker_id: int, T_cam_marker: np.ndarray) -> np.ndarray | None:
        """Given a detected marker with known world pose, back out ^W T_C."""
        if marker_id not in self.landmarks:
            return None
        T_wm = self.landmarks[marker_id].T_world_marker
        return compose(T_wm, invert_T(T_cam_marker))

    def as_positions(self) -> dict[int, np.ndarray]:
        return {mid: lm.T_world_marker[:3, 3].copy() for mid, lm in self.landmarks.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "markers": {
                int(mid): {
                    "T_world_marker": lm.T_world_marker.tolist(),
                    "observations": int(lm.observation_count),
                    "anchored": mid in self.anchors,
                }
                for mid, lm in self.landmarks.items()
            }
        }
        with open(path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=True)

    @staticmethod
    def load(path: Path) -> "LandmarkMap":
        with open(path) as f:
            d = yaml.safe_load(f)
        m = LandmarkMap()
        for mid, entry in d.get("markers", {}).items():
            T = np.array(entry["T_world_marker"])
            lm = Landmark(int(mid), T, observation_count=int(entry.get("observations", 1)))
            m.landmarks[int(mid)] = lm
            if entry.get("anchored"):
                m.anchors[int(mid)] = T
        return m

    @staticmethod
    def from_marker_gt(marker_gt_yaml: dict) -> "LandmarkMap":
        """Build an anchored map from docs/ANDROID_RECORDING.md-style marker_gt.yaml."""
        m = LandmarkMap()
        for mid, entry in marker_gt_yaml.get("markers", {}).items():
            yaw = np.deg2rad(float(entry.get("yaw_deg", 0.0)))
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            t = np.array([float(entry["x"]), float(entry["y"]), float(entry["z"])])
            m.set_anchor(int(mid), make_T(R, t))
        return m
