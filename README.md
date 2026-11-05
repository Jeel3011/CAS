# CAS — Visual–Inertial SLAM with Fiducial Landmark Anchors

**IE415: Control of Autonomous Systems — Semester Project**

Indoor mapping and localization system that fuses smartphone camera + IMU data with ArUco fiducial markers to estimate a user's trajectory and reconstruct the room map.

## Team
- 202301047 — Jeel Thummar
- 202301257 — Kanu Bhadarka
- 202301176 — Meet Barasara

## Architecture (locked)
**Phone–Laptop Hybrid.** Android phone records synchronized RGB video + IMU; laptop runs the SLAM pipeline.

```
Android phone (sensing)                Laptop (processing)
----------------------                 --------------------
camera (30 fps, 1080p)       -->       dataset loader
IMU (accel + gyro, 100 Hz)   -->       camera calibration
timestamps.csv               -->       ArUco detection + pose
MP4 video                    -->       visual-inertial estimator
                                       factor-graph optimization
                                       trajectory + map visualization
```

## Project Structure
```
CAS/
├── README.md
├── requirements.txt
├── Project.pdf                 # original handout
├── docs/
│   ├── MID_EVAL_FORM_ANSWERS.md     # form text to submit today
│   ├── SYSTEM_DESIGN.md             # M1 deliverable
│   └── ANDROID_RECORDING.md         # phone-side capture instructions
├── src/
│   ├── acquisition/            # dataset I/O, sync
│   ├── calibration/            # camera intrinsics calibration
│   ├── detection/              # ArUco detection + pose
│   ├── estimation/             # VO / VIO
│   ├── mapping/                # landmark map construction
│   ├── visualization/          # trajectory + map plotting
│   └── utils/                  # math, transforms, config
├── scripts/
│   ├── generate_aruco_markers.py    # prints a PDF of marker IDs
│   ├── calibrate_camera.py          # runs chessboard calibration
│   └── run_pipeline.py              # end-to-end pipeline
├── data/
│   ├── calibration/            # chessboard captures + intrinsics.yaml
│   ├── raw/                    # per-run folders: video.mp4, imu.csv, timestamps.csv
│   └── processed/              # intermediate outputs
├── tests/
├── notebooks/
└── results/                    # trajectories, maps, figures, metrics
```

## Setup

### Prerequisites
- Python 3.10+
- pip
- (Optional) virtualenv / conda

### Install
```bash
pip install -r requirements.txt
```

### Verify OpenCV ArUco is available
```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
```
Must print `True`. If not, install `opencv-contrib-python` (not plain `opencv-python`).

## Quickstart

### 1. Print markers
```bash
python scripts/generate_aruco_markers.py --count 10 --size-mm 150 --out data/markers.pdf
```
Print on A4, mount on walls, record each marker's measured world position in `data/raw/<run>/marker_gt.yaml`.

### 2. Calibrate the phone camera
Record a video of a chessboard from many angles, extract frames, then:
```bash
python scripts/calibrate_camera.py \
    --images data/calibration/frames/ \
    --pattern 9x6 --square-mm 25 \
    --out data/calibration/intrinsics.yaml
```
See `docs/ANDROID_RECORDING.md` for the full capture procedure.

### 3. Record a run
Use any Android app that saves MP4 + a CSV of accelerometer and gyroscope samples with timestamps (e.g., `Physics Toolbox Sensor Suite`, `SensorLogger`, or `IMU+`). Put outputs in `data/raw/<run_name>/`:
```
video.mp4
imu.csv           # ts_ns, ax, ay, az, gx, gy, gz
timestamps.csv    # frame_idx, ts_ns   (one row per video frame)
```

### 4. Run the pipeline
```bash
python scripts/run_pipeline.py --run data/raw/<run_name> --intrinsics data/calibration/intrinsics.yaml
```
Outputs land in `results/<run_name>/`.

## Milestones
| M | Outcome | Status |
|---|---|---|
| M1 | Architecture + system diagram | Done |
| M2 | Acquisition + calibration | In progress |
| M3 | ArUco detection + pose | In progress |
| M4 | VO / VIO trajectory | Planned |
| M5 | Landmark map + visualization | Planned |
| M6 | Drift correction, loop closure, baseline | Planned |
| M7 | Final experiments + report + demo | Planned |

See `docs/SYSTEM_DESIGN.md` for the full design document.

## License
Course project — IE415, Autumn 2025.
