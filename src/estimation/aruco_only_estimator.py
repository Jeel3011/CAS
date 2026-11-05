"""M3/M5 baseline: estimate camera trajectory from ArUco observations alone.

For each frame we solve for the camera-in-world pose using the current landmark
map.  Strategy:
  * If one or more markers are anchored, prefer them (lowest translation uncertainty).
  * Otherwise take the best-visible marker (smallest reprojection error).
  * If no marker is visible, hold the previous pose (will be replaced by VO in M4).

This is intentionally simple — it is the "baseline" the handout asks us to
compare the full VIO against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.detection.aruco_detector import MarkerDetection
from src.mapping.landmark_map import LandmarkMap
from src.utils.transforms import compose, invert_T


@dataclass
class PoseEstimate:
    frame_idx: int
    T_world_cam: np.ndarray
    source: str  # "anchor" | "landmark" | "hold"
    marker_id: int | None


class ArucoOnlyEstimator:
    def __init__(self, landmark_map: LandmarkMap) -> None:
        self.map = landmark_map
        self.last_pose: np.ndarray | None = None

    def step(self, frame_idx: int, detections: list[MarkerDetection]) -> PoseEstimate:
        # Prefer anchored markers first.
        anchored = [d for d in detections if d.marker_id in self.map.anchors]
        pool = anchored if anchored else [d for d in detections if d.marker_id in self.map.landmarks]
        if pool:
            best = min(pool, key=lambda d: d.reprojection_error_px)
            T_wc = self.map.camera_pose_from_marker(best.marker_id, best.T_cam_marker)
            if T_wc is not None:
                self.last_pose = T_wc
                # Also use remaining detections to initialize new landmarks.
                for d in detections:
                    if d.marker_id not in self.map.landmarks:
                        self.map.observe(d.marker_id, T_wc, d.T_cam_marker)
                    elif d.marker_id not in self.map.anchors:
                        self.map.observe(d.marker_id, T_wc, d.T_cam_marker)
                return PoseEstimate(frame_idx, T_wc, "anchor" if anchored else "landmark", best.marker_id)

        # Bootstrap: if the map is empty, anchor the first-seen marker at world origin.
        if not self.map.landmarks and detections:
            first = min(detections, key=lambda d: d.reprojection_error_px)
            self.map.set_anchor(first.marker_id, np.eye(4))
            T_wc = invert_T(first.T_cam_marker)  # ^W T_C = (^C T_M0)^{-1} since ^W T_M0 = I
            self.last_pose = T_wc
            for d in detections:
                if d.marker_id != first.marker_id:
                    self.map.observe(d.marker_id, T_wc, d.T_cam_marker)
            return PoseEstimate(frame_idx, T_wc, "anchor", first.marker_id)

        # Nothing visible — hold.
        if self.last_pose is None:
            self.last_pose = np.eye(4)
        return PoseEstimate(frame_idx, self.last_pose.copy(), "hold", None)

    def run(self, detections_per_frame: Iterable[tuple[int, list[MarkerDetection]]]) -> list[PoseEstimate]:
        return [self.step(idx, dets) for idx, dets in detections_per_frame]
