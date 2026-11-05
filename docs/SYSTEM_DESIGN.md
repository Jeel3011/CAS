# System Design Document — Visual–Inertial SLAM with ArUco Anchors

**Course:** IE415 — Control of Autonomous Systems
**Team:** Jeel Thummar (202301047), Kanu Bhadarka (202301257), Meet Barasara (202301176)
**Architecture:** Phone–Laptop Hybrid (Architecture C)
**Framework:** Python 3.10 + OpenCV (contrib) + NumPy/SciPy, optional GTSAM for graph optimization

---

## 1. Problem Recap

A person carrying an Android phone walks through an indoor room containing 6–10 printed ArUco markers at fixed known positions. The system must estimate the phone's trajectory and visualize it together with the marker map.

The markers play four roles in our design:
1. **Landmarks** during map build (poses estimated jointly with the trajectory).
2. **Anchors** when at least one marker's world pose is pre-surveyed.
3. **Loop-closure identifiers** (marker IDs are unique, so re-detection gives a hard constraint).
4. **Relocalization references** for the "repeat run" experiment.

## 2. Coordinate Frames

| Frame | Symbol | Definition |
|---|---|---|
| World | `W` | Fixed room frame; origin at the first surveyed ArUco marker's center. X–right, Y–forward, Z–up (right-handed). |
| Body / IMU | `B` | Phone's IMU frame (as defined by the Android Sensor API). |
| Camera | `C` | Phone rear camera; Z-forward, X-right, Y-down (OpenCV convention). |
| Marker `i` | `M_i` | ArUco marker frame; origin at the marker center, Z-out of the marker's face, X-right along the marker, Y-down. |

Rigid transformations are denoted `^A T_B` ∈ SE(3), mapping points in B to A. Known/estimated transforms:
- `^C T_B` — camera-IMU extrinsic (estimated once via Kalibr-style procedure or assumed from the phone spec).
- `^W T_{B_k}` — body pose at time `k` (**state to estimate**).
- `^C T_{M_i}` — ArUco pose from `cv2.aruco.estimatePoseSingleMarkers`.
- `^W T_{M_i}` — marker-in-world pose (known for anchors, estimated otherwise).

## 3. State Representation

We use the standard VIO state per keyframe:

```
x_k = [ p_k , v_k , q_k , b_a,k , b_ω,k ]^T ∈ R^{16}
```
- `p_k` ∈ R³ — body position in world
- `v_k` ∈ R³ — body velocity in world
- `q_k` ∈ H  — body-to-world orientation (unit quaternion)
- `b_a,k` ∈ R³ — accelerometer bias
- `b_ω,k` ∈ R³ — gyroscope bias

Landmark states: each ArUco marker `i` carries `L_i = (p_i^W, q_i^W)` ∈ SE(3), initialized on first observation using `^W T_{M_i} = ^W T_{C_k} · ^C T_{M_i}`.

## 4. Data Flow

```
┌────────────┐    ┌────────────┐
│  Camera    │    │    IMU     │  (Android phone)
│  30 fps    │    │   100 Hz   │
└─────┬──────┘    └─────┬──────┘
      │ MP4             │ CSV
      │ timestamps.csv  │
      └──────┬──────────┘
             ▼
      ┌──────────────┐
      │ Dataset      │  (laptop)
      │ loader + sync│
      └──────┬───────┘
             │
   ┌─────────┼──────────┬──────────────┐
   ▼         ▼          ▼              ▼
┌──────┐ ┌───────┐ ┌──────────┐  ┌───────────┐
│ VO / │ │ IMU   │ │  ArUco   │  │ Camera    │
│feat. │ │ prop. │ │ detect + │  │ intrinsics│
│track │ │       │ │  pose    │  │  (yaml)   │
└──┬───┘ └───┬───┘ └────┬─────┘  └─────┬─────┘
   │         │          │              │
   └─────────┴────┬─────┴──────────────┘
                  ▼
         ┌─────────────────┐
         │  VIO / Graph    │
         │  Estimator      │
         │  (filter or     │
         │   smoother)     │
         └────────┬────────┘
                  │
         ┌────────┴─────────┐
         ▼                  ▼
┌─────────────────┐  ┌──────────────┐
│ Trajectory +    │  │   Metrics    │
│ Landmark Map    │  │ (ATE, drift) │
│ Visualization   │  │              │
└─────────────────┘  └──────────────┘
```

## 5. Calibration

### 5.1 Camera Intrinsics
Record a video of a 9×6 chessboard (25 mm squares) from ~20 viewpoints (tilt, roll, distance). Extract ~30 frames, run `scripts/calibrate_camera.py`. Output: `fx, fy, cx, cy, k1..k5` in `data/calibration/intrinsics.yaml`.

### 5.2 Camera–IMU Extrinsic
For M4/M5 we assume the standard Android phone layout: IMU and camera axes aligned up to a fixed rotation with a lever-arm of ~5 cm. If this proves too coarse we will capture a dedicated IMU–camera calibration sequence (slow sine motion along each axis) and solve for `^C T_B` via hand–eye calibration.

### 5.3 Time Synchronization
Both camera and IMU timestamps use the phone monotonic clock (ns). The loader resamples IMU samples onto per-frame timestamps and reports the sync residual. If a recording app does not emit camera timestamps we fall back to `frame_idx / fps`.

## 6. Estimation Pipeline

**Baseline (M4):** per-frame visual odometry using ORB features + 5-point RANSAC + scale-constrained pose propagation.

**With IMU (M4–M5):** loosely coupled fusion — VO provides pose updates, IMU propagates between frames. Implemented as an ES-EKF on the 16-dim state.

**With ArUco (M5–M6):** each ArUco detection adds a measurement
```
z_ki = log( (^W T_{M_i})^{-1} · ^W T_{C_k} · ^C T_{M_i}^obs )  ∈ R⁶
```
Landmarks whose world pose is unknown are initialized on first observation and jointly estimated.

**Graph (M6, optional):** poses and landmarks optimized in GTSAM with pre-integrated IMU factors + ArUco factors:
```
min Σ || r_VIO(X_k, X_{k+1}) ||²_Σk  +  Σ || r_marker(X_k, L_i, z_ki) ||²_Λki
```

## 7. Loop Closure & Relocalization

Loop closure comes "for free" with ArUco: when a marker that was seen at frame `k0` is re-observed at `k1 >> k0`, we add a between-pose factor linking `X_k0` and `X_k1` through the shared landmark. Relocalization (M6/M7): load the saved landmark map, run ArUco detection on a new run, and solve a PnP-style problem from the first 2+ marker detections to initialize `^W T_C`.

## 8. Experiments (planned per handout §8)
1. **Simple path** — rectangular walk, 4–6 m per side.
2. **Marker-loss path** — straight segment with no visible markers for ≥ 5 s.
3. **Closed-loop path** — return to start, re-observe marker 0.
4. **Repeat run** — load prior map, walk the same path, measure relocalization error.
5. **Difficult condition** — low light / rapid turns.

### Metrics
- Translational ATE (RMSE) vs. surveyed ground-truth waypoints.
- Endpoint drift and drift per meter.
- Loop-closure residual (pose difference at loop point before and after closure).
- Marker map error (estimated vs. surveyed marker positions).
- Processing rate (frames/sec).

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| IMU–camera time offset biases VIO | Record sync gesture (phone tap) at start of each run; refine offset by cross-correlation of angular velocity vs. optical flow. |
| ArUco detection fails under motion blur | Lower shutter time, slow turns; fall back to pure VO during gaps. |
| Phone rolling-shutter distorts ArUco corners | Document per-run; evaluate the error vs. baseline. |
| GTSAM install fails on Windows | Keep EKF estimator (NumPy only) as the primary; GTSAM graph is an M6 extension. |

## 10. Division of Responsibility
- **Jeel (202301047)** — VIO estimator, graph optimization, drift/loop-closure analysis.
- **Kanu (202301257)** — Android acquisition workflow, calibration, dataset management.
- **Meet (202301176)** — ArUco detection + mapping, visualization, evaluation and report.
