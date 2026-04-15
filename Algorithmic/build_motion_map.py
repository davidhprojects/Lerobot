"""
Build a polynomial motion map from raw recording data.

Reads the CSV produced by record_motion_map.py, filters by base-frame Y,
downsamples to one point per 5mm grid cell, fits a polynomial per joint,
validates on a held-out split, and saves the result as JSON.

Usage:
    python algorithmic/build_motion_map.py left
    python algorithmic/build_motion_map.py right
"""

import sys
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

INPUT_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent

CELL_SIZE_M = 0.005  # 5 mm grid cells for downsampling
VALIDATION_FRACTION = 0.15  # hold out 15% for validation

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]

# Gravity compensation offsets (raw encoder counts), per arm.
# Added to recorded joint values before fitting to counteract sag from
# hand-supporting the arm during recording.  Tune these empirically.
GRAVITY_OFFSETS = {
    "left": {
        "shoulder_lift": -40,   # + is forward, - is toward home
        "elbow_flex": -30,    # + is lower
        "wrist_flex": 0,     # + is toward base
    },
    "right": {
        "shoulder_lift": -40, # + is forward, - is toward home
        "elbow_flex": -40,  # + is lower
        "wrist_flex": 0,   # + is toward base
    },
}

# Gripper-to-marker offset in base frame (meters), per arm.
# Shifts where the gripper targets relative to the tray marker.
# Signs depend on each arm's base-frame orientation — tune independently.
POSITION_OFFSET = {
    "left": {
        "base_x": -0.03,
        "base_z": 0.0,
    },
    "right": {
        "base_x": 0.0,
        "base_z": 0.0,
    },
}


# ============================================================
# CSV loading
# ============================================================

def load_raw_csv(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw recording CSV.

    Returns
    -------
    metadata : dict
        Parsed from the first comment line.
    cam_positions : ndarray (N, 3)
        cam_x, cam_y, cam_z for each sample (ground truth, never changes).
    base_positions : ndarray (N, 3)
        base_x, base_y, base_z as recorded (may be stale if calibration changed).
    joints : ndarray (N, 6)
        Raw encoder values for each sample.
    """
    with open(path) as f:
        first_line = f.readline().strip()
        if not first_line.startswith("#"):
            raise ValueError("Expected metadata comment on first line of CSV.")
        metadata = json.loads(first_line[1:].strip())

        reader = csv.DictReader(f)
        cam_list = []
        pos_list = []
        joint_list = []
        for row in reader:
            cam_list.append([
                float(row["cam_x"]),
                float(row["cam_y"]),
                float(row["cam_z"]),
            ])
            pos_list.append([
                float(row["base_x"]),
                float(row["base_y"]),
                float(row["base_z"]),
            ])
            joint_list.append([int(row[name]) for name in MOTOR_NAMES])

    return metadata, np.array(cam_list), np.array(pos_list), np.array(joint_list)


# ============================================================
# Filtering and downsampling
# ============================================================

def downsample_grid(
    positions: np.ndarray,
    joints: np.ndarray,
    target_y: float,
    cell_size: float = CELL_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin (base_x, base_z) into grid cells.  For each cell, keep the sample
    whose base_y is closest to target_y.
    """
    # Compute cell indices
    cx = np.floor(positions[:, 0] / cell_size).astype(int)
    cz = np.floor(positions[:, 2] / cell_size).astype(int)
    keys = list(zip(cx, cz))

    # Group by cell, keep best Y
    best: dict[tuple[int, int], int] = {}
    y_deviation = np.abs(positions[:, 1] - target_y)
    for i, key in enumerate(keys):
        if key not in best or y_deviation[i] < y_deviation[best[key]]:
            best[key] = i

    indices = sorted(best.values())
    return positions[indices], joints[indices]


# ============================================================
# Polynomial fitting
# ============================================================

def fit_polynomial(
    xz: np.ndarray,
    joint_values: np.ndarray,
    degree: int = 3,
    alpha: float = 0.001,
) -> tuple[np.ndarray, dict, list]:
    """
    Fit a polynomial to map (base_x, base_z) -> one joint value.

    Parameters
    ----------
    xz : ndarray (N, 2)
        Input features (base_x, base_z).
    joint_values : ndarray (N,)
        Target joint encoder values.
    degree : int
        Polynomial degree.
    alpha : float
        Ridge regularization.

    Returns
    -------
    coefficients : ndarray (n_features,)
    normalization : dict with x_mean, x_std, z_mean, z_std
    powers : list of [px, pz] pairs defining the monomial basis
    """
    # Normalize inputs for numerical stability
    x_mean, x_std = xz[:, 0].mean(), max(xz[:, 0].std(), 1e-6)
    z_mean, z_std = xz[:, 1].mean(), max(xz[:, 1].std(), 1e-6)
    norm = {"x_mean": x_mean, "x_std": x_std, "z_mean": z_mean, "z_std": z_std}

    x_n = (xz[:, 0] - x_mean) / x_std
    z_n = (xz[:, 1] - z_mean) / z_std

    # Build polynomial feature matrix
    powers = []
    for total in range(degree + 1):
        for px in range(total + 1):
            pz = total - px
            powers.append([px, pz])

    # Feature matrix: each column is x^px * z^pz
    X = np.column_stack([
        x_n**px * z_n**pz for px, pz in powers
    ])

    # Ridge regression: (X^T X + alpha I) coeff = X^T y
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ joint_values
    coefficients = np.linalg.solve(XtX, Xty)

    return coefficients, norm, powers


def evaluate_polynomial(
    xz: np.ndarray,
    coefficients: np.ndarray,
    normalization: dict,
    powers: list,
) -> np.ndarray:
    """Evaluate a polynomial at the given (base_x, base_z) points."""
    x_n = (xz[:, 0] - normalization["x_mean"]) / normalization["x_std"]
    z_n = (xz[:, 1] - normalization["z_mean"]) / normalization["z_std"]
    X = np.column_stack([x_n**px * z_n**pz for px, pz in powers])
    return X @ coefficients


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("left", "right"):
        print("Usage: python build_motion_map.py [left | right] [--degree N]")
        sys.exit(1)
    side = sys.argv[1]

    degree = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--degree" and i + 1 < len(sys.argv):
            degree = int(sys.argv[i + 1])

    csv_path = INPUT_DIR / f"motion_map_raw_{side}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run record_motion_map.py {side} first.")
        sys.exit(1)

    # ── Load ────────────────────────────────────────────────
    print(f"Loading {csv_path}...")
    metadata, cam_positions, _, joints = load_raw_csv(csv_path)
    print(f"  {len(cam_positions)} samples (pre-filtered by Y during recording)")

    # ── Re-transform cam→base using current calibration ─────
    # The CSV stores base_x/y/z from recording time, but the calibration
    # may have changed since then.  Re-transform from camera-frame
    # (ground truth) using the current base_transforms.json.
    from algorithmic.calibrate import load_base_transforms
    base_transforms = load_base_transforms()
    if side not in base_transforms:
        print(f"  ERROR: No transform for {side}. Run calibrate.py first.")
        sys.exit(1)
    T_base_cam = np.linalg.inv(base_transforms[side])

    # Transform all cam positions to base frame
    ones = np.ones((len(cam_positions), 1))
    cam_h = np.hstack([cam_positions, ones])  # (N, 4)
    base_h = (T_base_cam @ cam_h.T).T         # (N, 4)
    positions = base_h[:, :3]                  # (N, 3)

    target_y = float(np.median(positions[:, 1]))
    print(f"  Re-transformed to current calibration")
    print(f"  target_base_y = {target_y*1000:.1f} mm (median of data)")

    if len(positions) < 20:
        print("ERROR: Too few samples. Record more data.")
        sys.exit(1)

    # ── Apply gravity compensation offsets ──────────────────
    grav = GRAVITY_OFFSETS.get(side, {})
    any_grav = any(v != 0 for v in grav.values())
    if any_grav:
        print("  Applying gravity offsets:")
        for name, offset in grav.items():
            if offset != 0:
                col = MOTOR_NAMES.index(name)
                joints[:, col] += offset
                print(f"    {name}: {offset:+d} counts")

    # ── Apply position offset ───────────────────────────────
    # Shifts the recorded gripper positions so the polynomial learns to
    # target an offset location relative to the tray marker.
    pos_off = POSITION_OFFSET.get(side, {})
    any_pos = any(v != 0 for v in pos_off.values())
    if any_pos:
        print("  Applying position offsets:")
        if pos_off.get("base_x", 0) != 0:
            positions[:, 0] += pos_off["base_x"]
            print(f"    base_x: {pos_off['base_x']*1000:+.1f} mm")
        if pos_off.get("base_z", 0) != 0:
            positions[:, 2] += pos_off["base_z"]
            print(f"    base_z: {pos_off['base_z']*1000:+.1f} mm")

    # ── Downsample ──────────────────────────────────────────
    positions, joints = downsample_grid(positions, joints, target_y)
    print(f"  {len(positions)} after grid downsample ({CELL_SIZE_M*1000:.0f} mm cells)")

    # ── Train/validation split ──────────────────────────────
    n = len(positions)
    n_val = max(1, int(n * VALIDATION_FRACTION))
    perm = np.random.default_rng(42).permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    xz_train = positions[train_idx][:, [0, 2]]  # base_x, base_z
    xz_val = positions[val_idx][:, [0, 2]]
    j_train = joints[train_idx]
    j_val = joints[val_idx]

    print(f"\n  Train: {len(train_idx)},  Validation: {len(val_idx)}")

    # ── Fit polynomial per joint ────────────────────────────
    print(f"\nFitting degree-{degree} polynomial per joint...")
    all_coeffs = {}
    normalization = None
    powers = None
    val_rmses = {}

    for i, name in enumerate(MOTOR_NAMES):
        coeffs, norm, pwr = fit_polynomial(xz_train, j_train[:, i], degree)

        if normalization is None:
            normalization = norm
            powers = pwr

        pred_val = evaluate_polynomial(xz_val, coeffs, norm, pwr)
        rmse = np.sqrt(np.mean((pred_val - j_val[:, i]) ** 2))
        val_rmses[name] = round(float(rmse), 2)

        pred_train = evaluate_polynomial(xz_train, coeffs, norm, pwr)
        train_rmse = np.sqrt(np.mean((pred_train - j_train[:, i]) ** 2))

        all_coeffs[name] = coeffs.tolist()

        deg_rmse = rmse / (4096 / 360)  # convert counts to degrees
        print(f"  {name:16s}  train_rmse={train_rmse:.1f}  val_rmse={rmse:.1f} counts ({deg_rmse:.2f} deg)")

    # ── Workspace bounds ────────────────────────────────────
    xz_all = positions[:, [0, 2]]
    bounds = {
        "base_x_min": float(xz_all[:, 0].min()),
        "base_x_max": float(xz_all[:, 0].max()),
        "base_z_min": float(xz_all[:, 1].min()),
        "base_z_max": float(xz_all[:, 1].max()),
    }

    # ── Save ────────────────────────────────────────────────
    output = {
        "metadata": {
            "arm": side,
            "target_base_y": target_y,
            "hover_height_m": metadata.get("hover_height_m", CELL_SIZE_M),
            "poly_degree": degree,
            "n_training_points": len(train_idx),
            "validation_rmse_counts": val_rmses,
            "recorded_at": metadata.get("recorded_at", ""),
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "workspace_bounds": bounds,
        },
        "coefficients": all_coeffs,
        "poly_powers": powers,
        "normalization": {k: round(float(v), 8) for k, v in normalization.items()},
    }

    out_path = OUTPUT_DIR / f"motion_map_{side}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # ── Summary ─────────────────────────────────────────────
    n_terms = len(powers)
    n_coeffs = n_terms * len(MOTOR_NAMES)
    print(f"\n  Polynomial: degree {degree}, {n_terms} terms per joint, {n_coeffs} total coefficients")
    print(f"  Workspace: X=[{bounds['base_x_min']*1000:.0f}, {bounds['base_x_max']*1000:.0f}] mm"
          f"  Z=[{bounds['base_z_min']*1000:.0f}, {bounds['base_z_max']*1000:.0f}] mm")
    print("  Done.")


if __name__ == "__main__":
    main()
