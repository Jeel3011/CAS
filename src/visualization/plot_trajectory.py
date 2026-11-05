"""2D top-down and 3D plots of trajectory + ArUco landmark map."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_top_down(
    trajectory_xyz: np.ndarray,
    landmarks: dict[int, np.ndarray] | None = None,
    title: str = "Trajectory (top-down)",
    out_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(trajectory_xyz[:, 0], trajectory_xyz[:, 1], "-", linewidth=1.5, label="Estimated trajectory")
    ax.plot(trajectory_xyz[0, 0], trajectory_xyz[0, 1], "go", markersize=10, label="Start")
    ax.plot(trajectory_xyz[-1, 0], trajectory_xyz[-1, 1], "rs", markersize=10, label="End")
    if landmarks:
        xs = [p[0] for p in landmarks.values()]
        ys = [p[1] for p in landmarks.values()]
        ax.scatter(xs, ys, c="orange", s=80, marker="^", edgecolors="black", label="ArUco markers", zorder=5)
        for mid, p in landmarks.items():
            ax.annotate(f"M{mid}", (p[0], p[1]), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_3d(
    trajectory_xyz: np.ndarray,
    landmarks: dict[int, np.ndarray] | None = None,
    title: str = "Trajectory (3D)",
    out_path: Path | None = None,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory_xyz[:, 0], trajectory_xyz[:, 1], trajectory_xyz[:, 2], "-", linewidth=1.5)
    ax.scatter(*trajectory_xyz[0], c="g", s=50, label="Start")
    ax.scatter(*trajectory_xyz[-1], c="r", s=50, label="End")
    if landmarks:
        pts = np.array(list(landmarks.values()))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="orange", s=80, marker="^", edgecolors="black", label="ArUco markers")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
