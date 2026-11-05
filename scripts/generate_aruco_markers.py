#!/usr/bin/env python3
"""Generate a set of ArUco markers as PNGs and an A4 printable PDF.

Usage:
    python scripts/generate_aruco_markers.py --count 10 --size-mm 150 --out data/markers.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_marker_images(count: int, dict_name: str, pixel_size: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    paths: list[Path] = []
    for marker_id in range(count):
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixel_size)
        # Add a 10%-wide white quiet zone and an ID label underneath.
        quiet = pixel_size // 10
        padded = np.full((pixel_size + 2 * quiet + 40, pixel_size + 2 * quiet), 255, dtype=np.uint8)
        padded[quiet:quiet + pixel_size, quiet:quiet + pixel_size] = img
        cv2.putText(
            padded,
            f"ID {marker_id}",
            (quiet, pixel_size + 2 * quiet + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            0,
            2,
            cv2.LINE_AA,
        )
        p = out_dir / f"marker_{marker_id:02d}.png"
        cv2.imwrite(str(p), padded)
        paths.append(p)
    return paths


def build_pdf(image_paths: list[Path], size_mm: float, pdf_path: Path) -> None:
    """One marker per page, centered, physical size preserved."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    a4_inch = (8.27, 11.69)
    size_inch = size_mm / 25.4
    with PdfPages(pdf_path) as pdf:
        for p in image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            fig = plt.figure(figsize=a4_inch)
            ax = fig.add_axes([
                (a4_inch[0] - size_inch) / 2 / a4_inch[0],
                (a4_inch[1] - size_inch) / 2 / a4_inch[1],
                size_inch / a4_inch[0],
                size_inch / a4_inch[1],
            ])
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{p.stem} — print at 100%, measured side = {size_mm:.0f} mm")
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10, help="Number of markers (IDs 0..count-1)")
    ap.add_argument("--dict", default="DICT_4X4_50", help="OpenCV ArUco dictionary name")
    ap.add_argument("--size-mm", type=float, default=150.0, help="Physical marker side length (mm)")
    ap.add_argument("--pixel-size", type=int, default=600, help="Marker rendering size in pixels")
    ap.add_argument("--out", type=Path, default=Path("data/markers.pdf"), help="Output PDF path")
    args = ap.parse_args()

    img_dir = args.out.parent / "markers"
    paths = generate_marker_images(args.count, args.dict, args.pixel_size, img_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(paths, args.size_mm, args.out)
    print(f"Wrote {len(paths)} marker PNGs to {img_dir}")
    print(f"Wrote printable PDF to {args.out}  (print at 100%, do NOT 'fit to page')")


if __name__ == "__main__":
    main()
