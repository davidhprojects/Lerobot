"""
Motion map runtime controller.

Loads a polynomial motion map (from build_motion_map.py) and provides:
  - MotionMap class: evaluates polynomial for (base_x, base_z) → joint targets
  - motion_map_hover_loop(): continuous servoing loop for one arm
"""

import json
import threading
import time
import numpy as np
from pathlib import Path


MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]

# Default tuning (can be overridden via kwargs)
MAX_JOINT_STEP_RAW = 6.0 * (4096.0 / 360.0)  # 10 degrees in raw encoder counts
POSITION_TOLERANCE_M = 0.012  # 10 mm dead-zone
CONTROL_HZ = 30

# Raise-on-traverse: lift the arm proportionally to XZ error so it
# approaches the target from above instead of swooping through the tray.
# The offset is applied per-joint in raw encoder counts, scaled by
# (xz_dist / RAISE_DIST_SCALE).  At xz_dist == RAISE_DIST_SCALE the
# full offset is applied; at xz_dist == 0 the offset is zero.
RAISE_DIST_SCALE_M = 0.050  # 100mm of xz error = 1x the raise offset
RAISE_OFFSETS = {
    "shoulder_lift": 40,
    "elbow_flex": -120,
    "wrist_flex": 10,
}


class MotionMap:
    """
    Evaluates a polynomial motion map: (base_x, base_z) → joint angles.

    The polynomial coefficients, normalization, and workspace bounds are
    loaded from the JSON produced by build_motion_map.py.
    """

    def __init__(self, path: Path | str):
        with open(path) as f:
            data = json.load(f)

        self.metadata = data["metadata"]
        self.arm = self.metadata["arm"]
        self.target_base_y = self.metadata["target_base_y"]
        self.powers = data["poly_powers"]  # list of [px, pz]
        self.norm = data["normalization"]
        self.bounds = self.metadata["workspace_bounds"]

        # Build coefficient matrix: (n_joints, n_terms)
        self._coeffs = np.array([
            data["coefficients"][name] for name in MOTOR_NAMES
        ])  # (6, n_terms)

    def evaluate(self, base_x: float, base_z: float) -> dict[str, float]:
        """
        Evaluate the polynomial at one (base_x, base_z) point.

        Returns a dict of {motor_name: raw_encoder_value}.
        """
        x_n = (base_x - self.norm["x_mean"]) / self.norm["x_std"]
        z_n = (base_z - self.norm["z_mean"]) / self.norm["z_std"]

        features = np.array([
            x_n ** px * z_n ** pz for px, pz in self.powers
        ])

        values = self._coeffs @ features  # (6,)
        return {name: float(values[i]) for i, name in enumerate(MOTOR_NAMES)}

    def in_bounds(self, base_x: float, base_z: float, margin: float = 0.01) -> bool:
        """Check if (base_x, base_z) is within the recorded workspace bounds."""
        b = self.bounds
        return (b["base_x_min"] - margin <= base_x <= b["base_x_max"] + margin
                and b["base_z_min"] - margin <= base_z <= b["base_z_max"] + margin)


def cam_to_base(p_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Transform a 3D point from camera frame to base-tag frame."""
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return (T_base_cam @ p_h)[:3]


def motion_map_hover_loop(
    robot,
    get_frame,
    detector,
    side: str,
    motion_map: MotionMap,
    T_base_cam: np.ndarray,
    max_step_raw: float = MAX_JOINT_STEP_RAW,
    position_tolerance: float = POSITION_TOLERANCE_M,
    control_hz: float = CONTROL_HZ,
    stop_event: threading.Event | None = None,
):
    """
    Continuously servo one arm to hover above its tray marker using the
    motion map polynomial.

    Parameters
    ----------
    robot : SOFollower
        Connected arm.
    get_frame : callable
        Returns the latest camera frame (BGR) or None.
    detector : ArucoDetector
        Marker detector.
    side : 'left' or 'right'
    motion_map : MotionMap
        Loaded polynomial model for this arm.
    T_base_cam : ndarray (4, 4)
        Inverse of T_cam_base (transforms camera → base-tag frame).
    stop_event : threading.Event, optional
        Set to signal the loop to exit.
    """
    interval = 1.0 / control_hz
    step_count = 0
    miss_streak = 0
    smooth_bx = None  # EMA-smoothed tray base position
    smooth_bz = None
    alpha = 0.2  # EMA smoothing factor (0 = ignore new, 1 = no smoothing)

    tray_marker_id = (detector.config.tray_left if side == "left"
                      else detector.config.tray_right)

    print(f"  [{side}] Motion map hover loop started (tray marker ID {tray_marker_id}).")

    while stop_event is None or not stop_event.is_set():
        t0 = time.perf_counter()

        frame = get_frame()
        if frame is None:
            time.sleep(interval)
            continue

        poses = detector.detect(frame)

        # ── Detect tray marker ──────────────────────────────
        tray_cam = detector.get_tray_marker_pose(poses, side)
        if tray_cam is None:
            miss_streak += 1
            if miss_streak == 30:
                print(f"  [{side}] Lost tray marker (ID {tray_marker_id}), holding position.")
            time.sleep(interval)
            continue

        if miss_streak >= 30:
            print(f"  [{side}] Tray marker recovered.")
        miss_streak = 0

        # ── Transform tray to base frame ────────────────────
        tray_base = cam_to_base(tray_cam, T_base_cam)

        # Smooth the tray position to reduce marker detection noise
        if smooth_bx is None:
            smooth_bx = tray_base[0]
            smooth_bz = tray_base[2]
        else:
            smooth_bx = alpha * tray_base[0] + (1 - alpha) * smooth_bx
            smooth_bz = alpha * tray_base[2] + (1 - alpha) * smooth_bz

        target_bx = smooth_bx
        target_bz = smooth_bz

        # ── Bounds check ────────────────────────────────────
        if not motion_map.in_bounds(target_bx, target_bz):
            step_count += 1
            if step_count % (int(control_hz) * 2) == 0:
                print(f"  [{side}] Target outside workspace, holding position.")
            time.sleep(interval)
            continue

        # ── Evaluate polynomial ─────────────────────────────
        target_joints = motion_map.evaluate(target_bx, target_bz)

        # ── Measure XZ distance (gripper to target) ─────────
        grip_cam = detector.get_gripper_pose(poses, side)
        xz_dist = None
        if grip_cam is not None:
            grip_base = cam_to_base(grip_cam, T_base_cam)
            xz_dist = np.sqrt(
                (grip_base[0] - target_bx) ** 2 + (grip_base[2] - target_bz) ** 2
            )

        # ── Raise-on-traverse: lift proportional to XZ error ──
        if xz_dist is not None:
            raise_scale = xz_dist / RAISE_DIST_SCALE_M
            for name, offset in RAISE_OFFSETS.items():
                target_joints[name] += offset * raise_scale

        # ── Read current joints ─────────────────────────────
        current_joints = robot.bus.sync_read("Present_Position", normalize=False)

        # ── Dead-zone check ─────────────────────────────────
        skip = False
        if xz_dist is not None:
            if xz_dist < position_tolerance:
                skip = True
        else:
            # Fallback: joint-space proximity
            max_delta = max(
                abs(target_joints[name] - current_joints[name])
                for name in MOTOR_NAMES
            )
            if max_delta < max_step_raw * 0.5:
                skip = True

        if not skip:
            # ── Proportional clamp (preserves joint-space direction) ──
            deltas = {name: target_joints[name] - current_joints[name]
                      for name in MOTOR_NAMES}
            max_abs = max(abs(d) for d in deltas.values())
            if max_abs > max_step_raw:
                scale = max_step_raw / max_abs
                deltas = {name: d * scale for name, d in deltas.items()}

            cmd = {name: current_joints[name] + deltas[name]
                   for name in MOTOR_NAMES}
            robot.bus.sync_write("Goal_Position", cmd, normalize=False)

        # ── Periodic logging ────────────────────────────────
        step_count += 1
        if step_count % (int(control_hz) * 2) == 0:
            tag = "HOLD" if skip else "ok"
            err_str = f"{xz_dist*1000:.1f}mm" if grip_cam is not None else "no marker"
            print(
                f"  [{side}] err={err_str} [{tag}]  "
                f"tray_base=({target_bx*1000:.0f},{target_bz*1000:.0f})"
            )

        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print(f"  [{side}] Motion map hover loop stopped.")
