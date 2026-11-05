"""Chessboard-based camera-intrinsics calibration (OpenCV standard pipeline)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


@dataclass
class Intrinsics:
    image_width: int
    image_height: int
    K: list[list[float]]
    dist: list[float]
    rms_reprojection_error: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @staticmethod
    def load(path: Path) -> "Intrinsics":
        with open(path) as f:
            d = yaml.safe_load(f)
        return Intrinsics(**d)

    def K_array(self) -> np.ndarray:
        return np.array(self.K, dtype=np.float64)

    def dist_array(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64)


def calibrate_from_images(
    image_paths: Iterable[Path],
    pattern_size: tuple[int, int],
    square_mm: float,
    show_debug: bool = False,
) -> Intrinsics:
    """Run OpenCV chessboard calibration.

    pattern_size is (inner corners per row, inner corners per column) — e.g. (9, 6) for a 10x7 board.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_mm * 1e-3  # to meters

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_shape: tuple[int, int] | None = None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    used = 0
    paths = list(image_paths)
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if image_shape is None:
            image_shape = img.shape[::-1]  # (w, h)
        found, corners = cv2.findChessboardCorners(
            img,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )
        if not found:
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), crit)
        obj_points.append(objp)
        img_points.append(corners)
        used += 1
        if show_debug:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            cv2.imshow("corners", vis)
            cv2.waitKey(50)
    if show_debug:
        cv2.destroyAllWindows()

    if used < 6:
        raise RuntimeError(
            f"Only {used}/{len(paths)} images had a detectable chessboard; need >=6 (prefer >=20)."
        )
    assert image_shape is not None

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None
    )
    return Intrinsics(
        image_width=int(image_shape[0]),
        image_height=int(image_shape[1]),
        K=K.tolist(),
        dist=dist.reshape(-1).tolist(),
        rms_reprojection_error=float(rms),
    )
