# Motion Map: Portable Hover Control

## Overview

Replace the FK / R\_cam\_base calibration / IK controller with a learned mapping from gripper position to joint angles, expressed in each arm's base frame. The base frame is anchored by AprilTags on the arm bases, detected once during a brief calibration step. This makes the system **portable**: record the motion map once, fit a polynomial, then move the entire setup to a new table and room. Re-run the 10-second calibration to locate the bases relative to the new camera position — no re-recording needed.

### Why the Current Approach Fails

The current hover controller chains: camera error → R\_cam\_base rotation → base-frame target → Newton-Raphson IK → joint commands. Errors in R (from the perturbation-based calibration) propagate directly into wrong base-frame targets. The left arm's calibration currently maps the vertical camera error to the wrong base-frame direction, so the arm drifts sideways/down when it should go up. Additionally, the decoupled position/orientation IK causes the wrist correction to fight the position correction.

### What Replaces It

1. **Base-frame AprilTags** on each arm base → camera-to-base transform computed once at calibration (replaces R\_cam\_base calibration entirely)
2. **Polynomial regression** learned from a one-time recording sweep → maps (base\_x, base\_z) to joint angles (replaces FK + IK entirely)
3. **Simple clamped joint-space interpolation** toward the polynomial's output (replaces the Newton-Raphson solver and wrist orientation controller)

---

## Base-Frame AprilTags

### Physical Setup

Attach two new AprilTags (one per arm base) to the front face of each arm's base, facing the camera:

| Tag | ID | Location |
|---|---|---|
| Left base | **0** | Front face of left SO-101 base |
| Right base | **2** | Front face of right SO-101 base |

These tags anchor the entire coordinate system, so accuracy matters. Use a **larger tag size than the gripper/tray markers** if there's room on the base — 30-40mm inner pattern is ideal (the SO-101 base face should accommodate this). Larger tags give more accurate pose estimation from `solvePnP`. If space is tight, 15mm works but with lower accuracy. Attach rigidly (tape or glue — they must not shift between sessions). The tag size for base markers can differ from the gripper/tray markers; we just need to pass the correct size to `solvePnP` for each.

Updated marker inventory:

| ID | Role |
|---|---|
| 0 | Left arm base |
| 1 | Left gripper |
| 2 | Right arm base |
| 3 | Tray right |
| 4 | Right gripper |
| 5 | Tray left |

### Why Base Frame

The mapping from joint angles to gripper position **in the arm's base frame** is a fixed physical property of the arm. It doesn't depend on the camera position, the table, or the room. It's the same in your current lab as it will be in any future setup.

What changes between setups is the camera's pose relative to the arm bases. The base AprilTags give you this transform via `solvePnP` — the same pipeline already used for the gripper and tray tags. This replaces the fragile perturbation-based R\_cam\_base calibration with a direct, full 6-DOF measurement.

The base tags are **only needed during calibration** (a ~10 second step with the arms out of the way). During recording and normal operation the arms will occlude the base tags, so everything uses the saved transform from calibration. The camera is assumed static between calibrations.

### Camera-to-Base Transform

Each detected base tag gives a 4x4 homogeneous transform `T_cam_base` (base-tag frame expressed in camera frame). To convert any camera-frame point to the base-tag frame:

```
p_base = T_cam_base⁻¹ @ p_cam    (as homogeneous coordinates)
```

**This transform is computed once during calibration, saved to disk, and reused for all subsequent operation.** If the camera is bumped, re-run calibration (takes seconds).

During calibration, the arms should be in their home position (or moved aside) so the base tags are unoccluded. The script collects the base tag's `rvec` and `tvec` over ~30 frames (1 second) and averages them separately (element-wise mean of translation vectors and element-wise mean of Rodrigues rotation vectors). Since the tag is stationary, frame-to-frame variation is just detection noise and averaging rvecs directly is valid. The averaged pose is converted to a 4x4 `T_cam_base` and saved.

The base-tag frame becomes the reference frame for all spatial reasoning. We don't need to know the exact offset between the tag and the arm's kinematic origin — the polynomial absorbs that constant offset during fitting.

---

## Recording Phase

### Setup

1. Run `calibrate.py` first if not already done (arms in home position, base tags visible)
2. Connect one arm (left or right) and the camera
3. Place the tray in the workspace. The script detects the tray marker, transforms it to base frame via the saved `T_cam_base`, and computes `target_base_y = tray_base_y - HOVER_HEIGHT_M`. This locks the exact hover height for the entire recording session.
4. Move to pre\_grasp via smooth\_move (establishes a good starting pose)
5. Disable torque on all 5 joints (gripper torque stays on, open position)
6. Verify the gripper tag is visible (base tags will be occluded by the arm — that's fine, the saved transform is used). The tray can be removed after the target Y is computed — it's not needed during sweeping.

All joints are free. The user controls the full arm — shoulder, elbow, and wrist. This is critical because the wrist orientation that keeps the gripper marker squared to the camera varies across the workspace. Each recorded point captures the complete 5-joint configuration for that position, including the wrist angles needed for marker visibility.

### Sweeping

Two windows are shown during recording:

**Window 1 — Camera feed** with overlays:
- Gripper marker position in base frame (x, y, z in mm)
- A horizontal guide line at the target hover Y (in base frame)
- Live valid-sample count and recording status

**Window 2 — Coverage grid** (updated live):
- A grid of cells (40 x 60 for 200mm across x 300mm depth at 5mm spacing)
- Each cell is **red** if no valid point has been recorded there yet, **green** if covered
- A frame counts as valid if: (a) gripper marker detected, (b) `|base_y - target_y| < tolerance`, (c) the rounded (base\_x, base\_z) cell hasn't been filled yet
- `target_y` is computed at startup: the script detects the tray marker, transforms it to base frame, and subtracts `HOVER_HEIGHT_M` from the tray's base\_y. This is the exact hover height you want, defined by a single tunable parameter.
- The grid gives immediate feedback on where you still need to sweep

The coverage check is trivial per frame (~one Y comparison and one array index lookup), so it doesn't affect recording rate.

The user sweeps the arm across the workspace while:

1. **Keeping the gripper at roughly the target hover height** — watch the Y readout / guide line
2. **Keeping the gripper marker squared to the camera** — adjust the wrist naturally as you move

Sweep pattern — systematic back-and-forth:
```
    Start: arm at one side of workspace
    ─────────────────────────────>  (sweep right at constant depth)
                                  |
    <─────────────────────────────  (sweep left, step forward slightly)
    |
    ─────────────────────────────>  (sweep right again)
    ...
```

Move slowly enough to keep the marker tracked and to adjust the wrist at each position. Frames where the marker is lost (bad angle, too fast) are simply skipped.

Record everything. The post-processing step filters by Y tolerance — staying within ~20mm of the target during sweeping means most frames are usable.

### What Gets Recorded (30 fps)

The recording script loads the saved `T_cam_base` from `calibrate.py` (run calibration first if not already done). Each frame where the gripper tag is detected produces one row:

| Field | Source | Notes |
|---|---|---|
| `cam_x, cam_y, cam_z` | Gripper tag detection | Camera frame, for visualization |
| `base_x, base_y, base_z` | Transformed via saved `T_cam_base` | `T_cam_base⁻¹ @ gripper_cam` |
| `shoulder_pan` | Motor encoder (raw) | |
| `shoulder_lift` | Motor encoder (raw) | |
| `elbow_flex` | Motor encoder (raw) | |
| `wrist_flex` | Motor encoder (raw) | |
| `wrist_roll` | Motor encoder (raw) | |
| `gripper` | Motor encoder (raw) | Constant (open) |

Each arm is recorded separately. ~3 minutes of sweeping per arm.

### How Much Data

At 30fps, 3 minutes produces ~5,400 raw samples. After Y-filtering (~50-70% kept) and grid downsampling (one sample per 5mm cell), expect 500-2,000 clean data points. The workspace per arm is 200mm across (base X) x 300mm deep (base Z) — 2,400 cells at 5mm spacing. A thorough 3-minute sweep should cover most of them.

---

## Building the Motion Map

### Step 1: Filter by Y (base frame)

Keep only rows where the gripper's base-frame Y is within tolerance of the target hover height:

```
target_base_y = tray_base_y - HOVER_HEIGHT_M   # set at recording time
tolerance = 0.005   # 5 mm
keep row if |base_y - target_base_y| < tolerance
```

The `target_base_y` is saved in the raw CSV header during recording (computed from the tray marker's base-frame position and `HOVER_HEIGHT_M`). The build script reads it from there — no re-computation or estimation needed. Since both the tray height and hover offset are fixed physical quantities relative to the arm base, this value is consistent across setups.

### Step 2: Downsample

Bin the (base\_x, base\_z) space into 5mm grid cells. For each cell, keep the sample whose base\_y is closest to the target. This removes redundancy and ensures one clean sample per spatial region.

### Step 3: Fit Polynomial

For each of the 5 joints (plus gripper if desired), fit a polynomial function of (base\_x, base\_z):

```
joint_i = f_i(base_x, base_z)
```

**Polynomial degree:** Start with degree 3. A degree-3 polynomial in 2 variables has 10 terms:

```
1, x, z, x², xz, z², x³, x²z, xz², z³
```

For 5 joints, that's **50 total coefficients** — an extremely compact model.

**Fitting method:** `sklearn.preprocessing.PolynomialFeatures` + `Ridge` regression (small alpha ~0.001 for regularization). Ridge prevents overfitting if any polynomial terms are poorly constrained by the data.

**Per-joint fitting:** Fit each joint independently. This lets you inspect residuals per joint and increase the degree for any joint that needs it (e.g., shoulder\_pan might need degree 4 if it has more curvature across the workspace).

### Step 4: Validate

- Compute per-joint residuals on a held-out validation set (10-15% of data)
- Residuals should be < 10 raw encoder counts (~0.9 degrees) for each joint
- Visualize: scatter plot of predicted vs. actual joint values for each joint
- Visualize: the polynomial surface over (base\_x, base\_z) for each joint — should be smooth with no wild extrapolation at the edges

### Step 5: Save

```json
{
  "metadata": {
    "arm": "left",
    "target_base_y": -0.042,
    "poly_degree": 3,
    "n_training_points": 1087,
    "validation_rmse_counts": [4.2, 3.8, 5.1, 6.3, 4.7],
    "base_tag_id": 0,
    "gripper_tag_id": 1,
    "recorded_at": "2026-04-13T14:30:00",
    "workspace_bounds": {
      "base_x_min": -0.15, "base_x_max": 0.12,
      "base_z_min": -0.05, "base_z_max": 0.18
    }
  },
  "coefficients": {
    "shoulder_pan": [c0, c1, ..., c9],
    "shoulder_lift": [c0, c1, ..., c9],
    "elbow_flex": [c0, c1, ..., c9],
    "wrist_flex": [c0, c1, ..., c9],
    "wrist_roll": [c0, c1, ..., c9],
    "gripper": [c0, c1, ..., c9]
  },
  "poly_powers": [[0,0], [1,0], [0,1], [2,0], [1,1], [0,2], [3,0], [2,1], [1,2], [0,3]],
  "normalization": {
    "x_mean": -0.02, "x_std": 0.08,
    "z_mean": 0.07, "z_std": 0.06
  }
}
```

The `poly_powers` list defines the monomial basis so evaluation doesn't depend on sklearn. The `normalization` values (mean/std of the training data) are used to standardize inputs before polynomial evaluation — this improves numerical conditioning.

**Total model size: ~1 KB.** Compared to a lookup table of 1,000+ points, the polynomial is tiny and evaluates in microseconds.

---

## Runtime Controller

### Initialization

1. Load the polynomial JSON for each arm
2. Reconstruct the evaluation function: given (base\_x, base\_z), compute the polynomial features, multiply by coefficients, produce 6 joint targets

### Control Loop (per arm, 15 Hz)

At startup, the saved `T_cam_base` transforms are loaded (one per arm). These are constant for the entire session — no base tag detection happens during the control loop.

Each arm tracks its own tray marker (left arm → tray\_left ID 5, right arm → tray\_right ID 3) and transforms it through that arm's own saved `T_cam_base`.

```
each step:
    1. Capture camera frame → ArUco detection (tray + gripper tags only)
    2. Detect this arm's tray marker → tray_cam (x, y, z) in camera frame
       (if not detected: hold position, continue)
    3. Transform tray to base frame using saved T_cam_base:
         tray_base = T_cam_base⁻¹ @ tray_cam
    4. Target position: (tray_base_x, tray_base_z)
       (Y is handled by construction — polynomial was fit at hover Y)
    5. Check workspace bounds: if target is outside recorded region,
         hold position, log warning
    6. Evaluate polynomial → target_joints (6 raw encoder values)
    7. Read current joint encoders → current_joints
    8. Dead-zone check:
         - If gripper marker detected: transform to base frame, compute
           XZ distance to target. If < POSITION_TOLERANCE_M, skip.
         - If gripper marker lost: fall back to joint-space check —
           if max |target_joints - current_joints| < threshold, skip.
    9. Compute delta = target_joints - current_joints
   10. Clamp delta (MAX_JOINT_STEP_DEG per joint)
   11. Command current_joints + clamped_delta
```

The polynomial evaluation (step 6) is ~10 multiplications and additions per joint — negligible compute. The per-frame coordinate transform (step 3) is a single 4x4 matrix multiply using the stored `T_cam_base`. The dominant cost is ArUco detection, same as the current controller.

### Position Tolerance

Same dead-zone as before: if the gripper's base-frame (X, Z) distance to the target is below `POSITION_TOLERANCE_M` (10mm), skip the control step. This prevents jitter when the arm is already over the tray.

---

## Portability: Moving to a New Setup

This is the core advantage of the base-frame + polynomial approach.

### Quick Calibration (expected default, ~10 seconds)

1. Place arms on new table, mount camera
2. Move arms to home position so base tags are unoccluded
3. Run calibration: `python algorithmic/calibrate.py`
   - Detects each base tag, averages pose over ~30 frames
   - Saves `T_cam_base` per arm to `algorithmic/base_transforms.json`
4. Run the hover controller — it uses the saved transforms + existing polynomial (base tags no longer need to be visible)

**Why it works:** The polynomial maps (base\_x, base\_z) → joints. This relationship is a physical property of the arm and doesn't change between setups. The calibration step computes the camera-to-base transform for the new camera position. The polynomial doesn't change — only the transform does.

### Verification (30 seconds, recommended)

After setting up on a new table, run a quick check:

1. Place the tray at 3-5 positions across the workspace
2. At each position, the controller moves the arm to hover
3. Camera measures the actual gripper position vs. the target tray position
4. Report per-position error

If all errors are < 10mm, the setup is good. This catches issues like a base tag that shifted or an arm with a loose joint.

### Fine-Tuning (if verification fails)

If there's a small systematic offset (e.g., base tag was reattached in a slightly different spot, shifting all predictions by a few mm):

1. Record 10-20 calibration points at known positions on the new table
2. Compute residuals: polynomial prediction vs. actual joint angles at each point
3. Fit a small **affine correction** on top of the polynomial:
   ```
   joint_corrected = joint_poly + (a * base_x + b * base_z + c)
   ```
   This is 3 extra coefficients per joint (15 total) and handles constant offsets plus linear drift.
4. Save the correction alongside the polynomial

This takes ~2 minutes and handles minor physical differences without re-recording.

### Full Re-Record (rare)

Only needed if the arm hardware has physically changed (replaced a servo, different link geometry, etc.). Follow the same recording procedure as the initial setup.

---

## Upgrading the Learner (Future)

Polynomial regression is the starting point. If residuals are too high in some workspace regions (e.g., the joint mapping has strong local curvature near the edges), consider:

**Small neural network** (2 → 64 → 64 → 5): ~5K parameters, trains in seconds on 1,000 points. Better at capturing local nonlinearities. Downside: less interpretable, worse extrapolation outside recorded region.

**Gaussian Process regression**: gives predictions with uncertainty estimates. The controller could use the uncertainty: high confidence → full speed, low confidence → reduce step size or hold. Downside: O(N) per prediction, ~1ms for 1,000 points, which is fine but not as fast as polynomial evaluation.

The interface stays the same regardless of learner: `f(base_x, base_z) → joint_angles`. Start with the polynomial and only upgrade if the validation residuals justify it.

---

## Implementation Plan

### New Files

| File | Purpose |
|---|---|
| `algorithmic/calibrate.py` | Detects base tags, averages pose over N frames, saves `T_cam_base` per arm to `base_transforms.json` |
| `algorithmic/record_motion_map.py` | Recording script with live visualization (uses calibrate for base transform) |
| `algorithmic/build_motion_map.py` | Post-processing: filter, fit polynomial, validate, save |
| `algorithmic/motion_map.py` | `MotionMap` class + runtime hover controller |

### Changes to Existing Files

| File | Change |
|---|---|
| `perception/aruco.py` | Add `left_base` and `right_base` tag IDs and `base_marker_size_m` to `MarkerConfig`. Add `get_base_pose()` method to `ArucoDetector` that returns the 4x4 transform (using the base-specific marker size for `solvePnP`). |
| `algorithmic/hover.py` | Replace the IK-based controller with the motion map controller. Remove FK/calibration/IK imports and code. Keep smooth\_move, waypoints, and main loop structure. |

### Step-by-Step

**Step 1: Add base tags to ArUco config**
Add `left_base: int = 0` and `right_base: int = 2` to `MarkerConfig`. Add a `get_base_pose()` method to `ArucoDetector` that returns the base tag's 4x4 transform (or None if not detected).

**Step 2: Build the calibration script**
`calibrate.py` connects the camera, detects both base tags over ~30 frames, averages the poses, saves `T_cam_base` per arm to `algorithmic/base_transforms.json`. Takes ~10 seconds. This is the only step that needs to be re-run when the camera moves.

**Step 3: Build the recording script**
`record_motion_map.py` connects one arm + camera, loads the saved base transform, moves to pre\_grasp, disables torque. Shows two windows: (1) live camera feed with base-frame position overlay and Y guide line, (2) live coverage grid (red/green cells showing which base-frame XZ regions have valid data). Records to CSV at 30fps. Ctrl+C saves and exits.

**Step 4: Record both arms**
~3 minutes sweeping per arm. Visually verify coverage on the live feed.

**Step 5: Build the motion map**
`build_motion_map.py` reads raw CSV, filters by base-frame Y, downsamples, fits polynomial per joint, validates, produces coverage plots, saves JSON.

**Step 6: Build the runtime controller**
`motion_map.py` contains the `MotionMap` class (loads JSON, evaluates polynomial) and `motion_map_hover_loop()` (loads saved base transforms, detects tray/gripper tags, transforms to base frame via stored transform, queries polynomial, clamps and commands joints).

**Step 7: Replace hover.py**
Rewrite `hover.py` main to: load base transforms + motion maps, move to pre\_grasp via waypoints, run `motion_map_hover_loop` per arm. Remove all FK/IK/calibration code.

**Step 8: Test and iterate**
Run on stationary tray, then sliding tray. Check errors. If polynomial residuals are too high for any joint, increase degree for that joint and re-fit.

---

## Coverage and Quality Checks

1. **Coverage**: The build script visualizes which 5mm grid cells have data. Gaps = regions where the polynomial extrapolates. Fill gaps by re-recording those regions (the build script merges new data with existing).

2. **Smoothness**: Plot each joint's polynomial surface over (base\_x, base\_z). Should be smooth with no sharp features. All 5 joints — including wrist — should show gradual variation. The wrist will show a clear gradient (more adjustment at workspace edges), which is expected.

3. **Y consistency**: Standard deviation of base\_y across filtered samples should be < 3mm.

4. **Polynomial residuals**: Per-joint RMSE on held-out data should be < 10 encoder counts (~0.9 degrees). If a joint exceeds this, try degree 4 for that joint.

5. **Dry run**: Hover over a stationary tray. Arm should converge to < 5mm error and hold steady with no jitter. Then slide the tray and verify smooth tracking.

---

## Limitations

- **Fixed hover height**: The polynomial is fit at one base-frame Y. For different hover heights, record and fit a separate polynomial (or extend to 3D input: base\_x, base\_y, base\_z → joints, which requires sweeping a volume instead of a plane).

- **Single wrist solution per (x, z)**: Each point stores one wrist configuration — the one chosen during recording to keep the marker facing the camera. A different wrist orientation requires a separate recording.

- **Workspace boundary**: Outside the recorded region, the polynomial extrapolates. Low-degree polynomials extrapolate more gracefully than lookup tables (which fail completely), but predictions degrade. The controller checks workspace bounds and holds position outside them.

- **Base tag visibility at calibration time**: The base tags only need to be visible during the brief calibration step (~10 seconds, arms in home position). They do not need to be visible during recording or normal operation — the saved transform is used instead.

- **Recording time**: ~3 minutes per arm for initial setup. Subsequent setups on different tables require zero re-recording if hardware hasn't changed.
