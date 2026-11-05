# IE415 Mid-Evaluation Form — Ready-to-Submit Answers

> **Deadline: November 5 (today).** Only the Team Leader submits. Architecture + framework choices below are **locked for the semester**.
>
> Fill in the `[BRACKETED]` placeholders before pasting.

---

## 1. Group ID
`[GROUP_ID — fill from your course sheet]`

## 2. Team Leader
`[TEAM_LEADER_NAME]` — Roll: `[ROLL]`

## 3. Team Members (Roll Numbers)
- 202301047 — Jeel Thummar
- 202301257 — Kanu Bhadarka
- 202301176 — Meet Barasara

## 4. Project Title
Visual–Inertial SLAM with Fiducial Landmark Anchors — Indoor Mapping and Localization using a Phone–Laptop Hybrid Pipeline.

## 5. Selected Architecture (LOCKED)
**Architecture C — Phone–Laptop Hybrid.**

Justification: the phone provides a synchronized camera + IMU stream with a user-facing recording interface, and the laptop performs the computationally heavy visual–inertial SLAM, ArUco pose estimation, factor-graph optimization, and visualization. This split lets us use real smartphone-grade sensors without being constrained by on-device compute, and keeps the estimation pipeline reproducible on recorded datasets.

## 6. Primary Framework (LOCKED)
**Python 3.10 + OpenCV (contrib) for vision and ArUco, NumPy / SciPy for estimation, GTSAM (Python bindings) for factor-graph optimization, Matplotlib + Open3D for visualization.** Phone-side recording uses the Android `SensorRecorder` / `IMU+` style app (CSV IMU logs + MP4 video with timestamp sidecar). Data transfer: USB (MTP) or ADB pull.

## 7. Hardware / Software Stack
- **Phone:** Android device (team leader's) — rear camera + accelerometer + gyroscope (≥100 Hz IMU, 30 fps 1080p video).
- **Laptop:** Linux/Windows laptop with Python 3.10, OpenCV 4.x (contrib for ArUco), NumPy, SciPy, GTSAM, Matplotlib, Open3D.
- **Markers:** 6–10 ArUco markers (DICT_4X4_50), A4-printed at 15 cm square, mounted on walls and fixed objects at measured positions.
- **Calibration target:** 9×6 chessboard, 25 mm squares.

## 8. Repository Link
`[REPO_URL — e.g. https://github.com/<your-user>/CAS-VISLAM]`

> We initialized the repository locally; team leader will push to GitHub and paste the URL before submission.

## 9. Current Milestone Status

| Milestone | Target | Status | Evidence |
|---|---|---|---|
| **M1** — Problem formulation, architecture, hardware/software stack, system diagram | Complete | ✅ Done | `docs/SYSTEM_DESIGN.md`, `docs/architecture.md`, repo scaffolded |
| **M2** — Camera + IMU acquisition, timestamp logging, camera calibration | Scripts ready; calibration pending real capture | 🟡 In progress | `src/calibration/camera_calibration.py`, `docs/ANDROID_RECORDING.md` |
| **M3** — ArUco detection + camera-to-marker pose estimation | Module implemented; awaiting calibration results | 🟡 In progress | `src/detection/aruco_detector.py`, `scripts/generate_aruco_markers.py` |
| **M4** — Visual odometry / VIO trajectory | Not started | ⏳ Planned (week of Nov 10) | — |
| **M5** — Marker map + landmark observation + visualization | Visualization skeleton ready | ⏳ Planned (week of Nov 17) | `src/visualization/plot_trajectory.py` |
| **M6** — Drift correction, loop closure, baseline comparison | Not started | ⏳ Planned (week of Nov 24) | — |
| **M7** — Final experiments, evaluation, report, demo | Not started | ⏳ Planned (week of Dec 1) | — |

**Current working milestone:** M2–M3 (sensor acquisition and ArUco pose estimation).

## 10. Brief Milestone Summary (for the form's "status" field)

> We have completed M1 (problem formulation, architecture selection, system design document, and repository scaffolding for a phone–laptop hybrid pipeline). Camera and IMU acquisition, camera-calibration, and ArUco detection/pose-estimation modules are implemented in Python using OpenCV; calibration capture and ArUco map construction are in progress. We are presently working on M2–M3 and are on schedule to begin M4 (VIO) in the coming week. Experimental trajectories (simple path, marker-loss, closed-loop, repeat run, difficult condition) will be collected using an Android phone and processed on the laptop.

## 11. Responsibilities

| Member | Primary responsibility |
|---|---|
| Jeel Thummar (202301047) | Visual–inertial estimation (VIO), factor-graph optimization, drift/loop-closure analysis |
| Kanu Bhadarka (202301257) | Phone-side acquisition app, IMU/video sync, camera calibration, dataset management |
| Meet Barasara (202301176) | ArUco detection, marker map construction, visualization, evaluation & report |

---

## Submission Checklist (do this before clicking Submit)
- [ ] Group ID filled in
- [ ] Team Leader name confirmed
- [ ] Repo pushed to GitHub and URL pasted
- [ ] Architecture = "Phone–Laptop Hybrid" selected in the form dropdown
- [ ] Framework = "Python + OpenCV (+ GTSAM)" entered
- [ ] Milestone status copy-pasted from Section 10 above
- [ ] Only the Team Leader submits (not multiple members)
