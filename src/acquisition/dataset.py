"""Dataset loader for phone-recorded runs.

Expected layout of ``run_dir``::

    video.mp4
    imu.csv          # time_ns, ax, ay, az, gx, gy, gz
    timestamps.csv   # frame_idx, time_ns
    marker_gt.yaml   # optional
    meta.yaml        # optional
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

IMU_COLUMNS = ("time_ns", "ax", "ay", "az", "gx", "gy", "gz")


@dataclass
class Run:
    run_dir: Path
    video_path: Path
    fps: float
    width: int
    height: int
    frame_timestamps_ns: np.ndarray  # (N_frames,)
    imu: pd.DataFrame                # columns = IMU_COLUMNS
    marker_gt: dict | None
    meta: dict | None

    def frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.video_path}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()

    def imu_between(self, t0_ns: int, t1_ns: int) -> pd.DataFrame:
        s = self.imu
        return s[(s.time_ns >= t0_ns) & (s.time_ns < t1_ns)]


def _video_info(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n


def load_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    video_path = run_dir / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    fps, w, h, n_frames = _video_info(video_path)

    ts_path = run_dir / "timestamps.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        frame_ts_ns = ts["time_ns"].to_numpy(dtype=np.int64)
    else:
        print(f"[warn] {ts_path} missing — synthesizing from fps={fps:.2f}")
        frame_ts_ns = (np.arange(n_frames) * (1e9 / fps)).astype(np.int64)

    imu_path = run_dir / "imu.csv"
    if not imu_path.exists():
        raise FileNotFoundError(imu_path)
    imu = pd.read_csv(imu_path)
    missing = set(IMU_COLUMNS) - set(imu.columns)
    if missing:
        raise ValueError(f"IMU file missing columns {missing}. See docs/ANDROID_RECORDING.md")
    imu = imu[list(IMU_COLUMNS)].sort_values("time_ns").reset_index(drop=True)

    marker_gt = None
    mg_path = run_dir / "marker_gt.yaml"
    if mg_path.exists():
        with open(mg_path) as f:
            marker_gt = yaml.safe_load(f)

    meta = None
    meta_path = run_dir / "meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

    return Run(
        run_dir=run_dir,
        video_path=video_path,
        fps=fps,
        width=w,
        height=h,
        frame_timestamps_ns=frame_ts_ns,
        imu=imu,
        marker_gt=marker_gt,
        meta=meta,
    )


def _inspect(run: Run) -> None:
    dur_video = (run.frame_timestamps_ns[-1] - run.frame_timestamps_ns[0]) / 1e9
    imu_t = run.imu.time_ns.to_numpy()
    dur_imu = (imu_t[-1] - imu_t[0]) / 1e9
    overlap_start = max(run.frame_timestamps_ns[0], imu_t[0])
    overlap_end = min(run.frame_timestamps_ns[-1], imu_t[-1])
    overlap = max(0.0, (overlap_end - overlap_start) / 1e9)
    print(f"Video: {run.width}x{run.height} @ {run.fps:.2f} fps, {len(run.frame_timestamps_ns)} frames, {dur_video:.2f}s")
    print(f"IMU:   {len(run.imu)} samples, {dur_imu:.2f}s, mean rate {len(run.imu) / max(dur_imu, 1e-6):.1f} Hz")
    print(f"Overlap: {overlap:.2f}s ({overlap/max(dur_video, 1e-6)*100:.1f}% of video)")
    monotonic = bool(np.all(np.diff(run.frame_timestamps_ns) > 0)) and bool(np.all(np.diff(imu_t) >= 0))
    print(f"Timestamps monotonic: {monotonic}")
    if run.marker_gt:
        print(f"Marker ground truth: {len(run.marker_gt.get('markers', {}))} markers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=Path, required=True)
    args = ap.parse_args()
    run = load_run(args.inspect)
    _inspect(run)


if __name__ == "__main__":
    main()
