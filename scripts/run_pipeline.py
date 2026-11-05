#!/usr/bin/env python3
"""End-to-end pipeline: load a run, detect ArUco markers, estimate the camera
trajectory, build/update the landmark map, and dump trajectory + plots.

Usage:
    python scripts/run_pipeline.py \
        --run data/raw/run_01 \
        --intrinsics data/calibration/intrinsics.yaml \
        --marker-size-mm 150

Outputs land under results/<run_name>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.acquisition.dataset import load_run
from src.calibration.camera_calibration import Intrinsics
from src.detection.aruco_detector import ArucoDetector
from src.estimation.aruco_only_estimator import ArucoOnlyEstimator
from src.mapping.landmark_map import LandmarkMap
from src.visualization.plot_trajectory import plot_top_down, plot_3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--marker-size-mm", type=float, default=150.0)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or (Path("results") / args.run.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run from {args.run} ...")
    run = load_run(args.run)
    intr = Intrinsics.load(args.intrinsics)

    # Seed the landmark map with any surveyed markers, else start empty.
    if run.marker_gt:
        lm_map = LandmarkMap.from_marker_gt(run.marker_gt)
        print(f"  seeded landmark map with {len(lm_map.anchors)} anchor markers")
    else:
        lm_map = LandmarkMap()
        print("  no marker_gt.yaml — first-seen marker will anchor the world")

    detector = ArucoDetector(intr, marker_length_m=args.marker_size_mm * 1e-3, dict_name=args.dict)
    estimator = ArucoOnlyEstimator(lm_map)

    poses: list[np.ndarray] = []
    sources: list[str] = []
    detections_log: list[dict] = []

    print(f"Processing {len(run.frame_timestamps_ns)} frames ...")
    for frame_idx, frame in tqdm(run.frames(), total=len(run.frame_timestamps_ns)):
        dets = detector.detect_frame(frame, frame_idx)
        est = estimator.step(frame_idx, dets)
        poses.append(est.T_world_cam)
        sources.append(est.source)
        detections_log.append({
            "frame": frame_idx,
            "markers": [int(d.marker_id) for d in dets],
            "source": est.source,
        })

    traj_xyz = np.array([T[:3, 3] for T in poses])
    np.save(out_dir / "trajectory_xyz.npy", traj_xyz)
    np.save(out_dir / "trajectory_T.npy", np.stack(poses))

    with open(out_dir / "sources.txt", "w") as f:
        for s in sources:
            f.write(s + "\n")

    lm_map.save(out_dir / "landmark_map.yaml")

    plot_top_down(traj_xyz, lm_map.as_positions(), title=f"{args.run.name} — top-down",
                  out_path=out_dir / "trajectory_topdown.png")
    plot_3d(traj_xyz, lm_map.as_positions(), title=f"{args.run.name} — 3D",
            out_path=out_dir / "trajectory_3d.png")

    n_with_marker = sum(1 for s in sources if s != "hold")
    print(f"\nFrames with marker constraint: {n_with_marker}/{len(sources)} "
          f"({100 * n_with_marker / max(len(sources), 1):.1f}%)")
    print(f"Landmarks in final map:       {len(lm_map.landmarks)}")
    print(f"Outputs written to:           {out_dir}/")


if __name__ == "__main__":
    main()
