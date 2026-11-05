"""ArUco detection and per-marker pose estimation.

Produces, for each frame, a list of ``MarkerDetection`` carrying the marker id,
the four image-plane corners, and the camera-to-marker SE(3) transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.calibration.camera_calibration import Intrinsics
from src.utils.transforms import make_T, rvec_to_rot


@dataclass
class MarkerDetection:
    frame_idx: int
    marker_id: int
    corners_px: np.ndarray  # (4, 2) float
    T_cam_marker: np.ndarray  # (4, 4) SE(3)
    reprojection_error_px: float


def _build_object_points(marker_length_m: float) -> np.ndarray:
    """ArUco marker object-frame corner layout (OpenCV convention)."""
    h = marker_length_m / 2.0
    return np.array([
        [-h,  h, 0.0],
        [ h,  h, 0.0],
        [ h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float32)


class ArucoDetector:
    def __init__(
        self,
        intrinsics: Intrinsics,
        marker_length_m: float,
        dict_name: str = "DICT_4X4_50",
    ) -> None:
        self.K = intrinsics.K_array()
        self.dist = intrinsics.dist_array()
        self.marker_length_m = float(marker_length_m)
        self.obj_pts = _build_object_points(self.marker_length_m)

        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        params = aruco.DetectorParameters()
        # Robustness tweaks for phone footage:
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 4
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        self.detector = aruco.ArucoDetector(self.dictionary, params)

    def detect_frame(self, image_bgr: np.ndarray, frame_idx: int = 0) -> list[MarkerDetection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []
        detections: list[MarkerDetection] = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            # solvePnP uses (N,1,3) obj and (N,1,2) img points.
            img_pts = marker_corners.reshape(-1, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_pts,
                img_pts,
                self.K,
                self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            T = make_T(rvec_to_rot(rvec), tvec)

            # Reprojection error — a cheap quality signal.
            proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())

            detections.append(MarkerDetection(
                frame_idx=frame_idx,
                marker_id=int(marker_id),
                corners_px=img_pts,
                T_cam_marker=T,
                reprojection_error_px=err,
            ))
        return detections

    def detect_video(self, video_path: Path) -> Iterator[tuple[int, list[MarkerDetection]]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, self.detect_frame(frame, idx)
                idx += 1
        finally:
            cap.release()

    def annotate(self, image_bgr: np.ndarray, detections: list[MarkerDetection]) -> np.ndarray:
        out = image_bgr.copy()
        for d in detections:
            cv2.aruco.drawDetectedMarkers(
                out, [d.corners_px.reshape(1, 4, 2).astype(np.float32)], np.array([[d.marker_id]])
            )
            rvec = cv2.Rodrigues(d.T_cam_marker[:3, :3])[0]
            tvec = d.T_cam_marker[:3, 3]
            cv2.drawFrameAxes(out, self.K, self.dist, rvec, tvec, self.marker_length_m * 0.5)
        return out
