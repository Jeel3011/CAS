#!/usr/bin/env python3
"""Run camera-intrinsics calibration from a directory of chessboard frames.

Usage:
    python scripts/calibrate_camera.py \
        --images data/calibration/frames/ \
        --pattern 9x6 --square-mm 25 \
        --out data/calibration/intrinsics.yaml

Pattern is "inner corners per row x inner corners per column".
To extract frames from a chessboard video first:
    python scripts/calibrate_camera.py --video my_chess.mp4 --frames data/calibration/frames/ --every 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.camera_calibration import calibrate_from_images


def extract_frames(video_path: Path, out_dir: Path, every: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, help="Optional: extract frames from this video first.")
    ap.add_argument("--frames", type=Path, help="Where to write extracted frames (with --video).")
    ap.add_argument("--every", type=int, default=15, help="Save every Nth video frame.")
    ap.add_argument("--images", type=Path, help="Directory of chessboard images (PNG/JPG).")
    ap.add_argument("--pattern", default="9x6", help="Inner-corner count, e.g. '9x6'.")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.video:
        target = args.frames or (args.out.parent / "frames")
        n = extract_frames(args.video, target, args.every)
        print(f"Extracted {n} frames into {target}")
        if args.images is None:
            args.images = target

    if args.images is None:
        ap.error("Pass either --video (to extract first) or --images (pre-extracted frames).")

    pattern = tuple(int(x) for x in args.pattern.lower().split("x"))
    if len(pattern) != 2:
        ap.error("--pattern must look like '9x6'")

    image_paths = sorted([p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        ap.error(f"No images in {args.images}")

    print(f"Calibrating from {len(image_paths)} images (pattern {pattern}, {args.square_mm} mm squares)...")
    intr = calibrate_from_images(image_paths, pattern, args.square_mm, show_debug=args.debug)
    intr.save(args.out)
    print(f"RMS reprojection error: {intr.rms_reprojection_error:.3f} px")
    print(f"Wrote intrinsics to {args.out}")


if __name__ == "__main__":
    main()
