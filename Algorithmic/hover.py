"""
Decentralized tray-hover controller.

Each SO-101 arm independently aligns its gripper with its tray marker and
hovers a configurable distance above it using Jacobian-based visual servoing.
The arms share a single camera feed but make fully independent control
decisions -- no communication between them.

When run directly, both arms continuously track their respective tray markers
even as the tray is slid around. Press Ctrl+C to stop.

Reusable functions:
    calibrate_camera_to_base()  -- estimate camera-to-robot-base rotation
    compute_hover_target()      -- target position from tray marker pose
    hover_step()                -- one Jacobian-based IK iteration
    hover_loop()                -- continuous servoing loop for one arm
"""

import sys
import time
import json
import threading
import numpy as np
from pathlib import Path

# Allow importing from sibling directories
sys.path.insert(0, str(Path(__file__).parent.parent / "GNN Pipeline"))

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from forward_kinematics import (
    forward_kinematics, ee_position, JOINT_NAMES as FK_JOINT_NAMES, CALIBRATION_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"
WAYPOINTS_FILE = Path(__file__).parent.parent / "Data Collection" / "waypoints.json"
CALIBRATION_FILE = Path(__file__).parent / "camera_to_base_calibration.json"

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Hover target
HOVER_HEIGHT_M = 0.14           # 14 cm above tray marker (camera Y axis)

# Controller tuning
MAX_JOINT_STEP_DEG = 3.0        # per-joint clamp per control step (degrees)
CONTROL_HZ = 15                 # control loop frequency
ORIENTATION_GAIN = 0.3          # wrist orientation correction gain
ORIENTATION_DAMPING = 0.01      # DLS damping for 2x2 wrist solver
IK_MAX_ITERS = 10               # Newton-Raphson iterations for position IK
IK_TOLERANCE = 5e-4             # convergence threshold (0.5 mm)

# Calibration
CALIBRATION_PERTURB_DEG = 8.0   # how far to wiggle each joint
SETTLE_TIME_S = 0.8             # pause after each calibration perturbation

# Smooth motion
APPROACH_DURATION_S = 4.0       # home -> pre_grasp transition time
STARTUP_DURATION_S = 4.0        # current position -> home transition time
RECORD_FPS = 30                 # frame rate for interpolation steps

# Derived constants
COUNTS_PER_DEG = 4096.0 / 360.0


# ============================================================
# HARDWARE HELPERS
# ============================================================

def load_ports() -> dict:
    """Load COM port assignments from Setup/ports.json."""
    if not PORTS_FILE.exists():
        print(f"ERROR: {PORTS_FILE} not found. Run Setup/find_ports.py first.")
        sys.exit(1)
    with open(PORTS_FILE) as f:
        return json.load(f)


def connect_arm(arm_name: str, port: str) -> SOFollower:
    """Connect to one SO-101 arm and configure its motors."""
    config = SOFollowerRobotConfig(
        id=arm_name,
        port=port,
        use_degrees=True,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SOFollower(config)
    robot.connect()
    return robot


def read_joints_raw(robot: SOFollower) -> dict[str, float]:
    """Read current joint positions as raw encoder values."""
    return robot.bus.sync_read("Present_Position", normalize=False)


def write_joints_raw(robot: SOFollower, target: dict[str, float]):
    """Command joint positions using raw encoder values."""
    robot.bus.sync_write("Goal_Position", target, normalize=False)


def read_joints_deg(robot: SOFollower) -> dict[str, float]:
    """Read current joint positions in calibrated degrees."""
    return robot.bus.sync_read("Present_Position", normalize=True)


def get_fk_angles_rad(robot: SOFollower) -> np.ndarray:
    """Read the 5 FK joint angles in radians from motor encoders."""
    deg = read_joints_deg(robot)
    return np.deg2rad(np.array([deg[n] for n in FK_JOINT_NAMES]))


# ============================================================
# WAYPOINTS AND SMOOTH MOTION
# ============================================================

def load_waypoints() -> dict:
    """Load waypoints from Data Collection/waypoints.json."""
    if not WAYPOINTS_FILE.exists():
        print(f"ERROR: {WAYPOINTS_FILE} not found. Run centralized_collect.py --teach first.")
        sys.exit(1)
    with open(WAYPOINTS_FILE) as f:
        return json.load(f)


def minimum_jerk(t: float, duration: float) -> float:
    """Minimum-jerk trajectory scalar: smooth 0->1 with zero velocity at endpoints."""
    tau = np.clip(t / duration, 0.0, 1.0)
    return float(10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def interpolate_joints(
    start: dict[str, float],
    end: dict[str, float],
    progress: float,
) -> dict[str, float]:
    """Linearly interpolate between two raw joint-angle dicts at progress [0,1]."""
    return {
        name: start[name] + (end[name] - start[name]) * progress
        for name in start
    }


def smooth_move(
    left_robot: SOFollower,
    right_robot: SOFollower,
    left_start: dict, left_end: dict,
    right_start: dict, right_end: dict,
    duration: float,
    label: str = "",
):
    """
    Smoothly interpolate both arms from start to end waypoints using
    minimum-jerk timing.  Blocks until the motion completes.
    """
    fps = RECORD_FPS
    interval = 1.0 / fps
    n_steps = max(1, int(duration * fps))

    if label:
        print(f"  [{label}] {duration:.1f}s, {n_steps} steps")

    for step in range(n_steps):
        t_start = time.perf_counter()

        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)

        left_target = interpolate_joints(left_start, left_end, progress)
        right_target = interpolate_joints(right_start, right_end, progress)

        write_joints_raw(left_robot, left_target)
        write_joints_raw(right_robot, right_target)

        elapsed = time.perf_counter() - t_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================
# DECOUPLED IK: POSITION (JOINTS 0-2) + ORIENTATION (JOINTS 3-4)
# ============================================================

def orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """
    Compute orientation error as an axis-angle 3-vector.

    Returns the rotation (in radians) that would bring R_current to R_target.
    Zero vector means aligned.
    """
    R_err = R_target @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * np.sin(angle))

    return angle * axis


def solve_position_ik(
    q_arm: np.ndarray,
    q_wrist: np.ndarray,
    target_base: np.ndarray,
    max_iters: int = IK_MAX_ITERS,
    tol: float = IK_TOLERANCE,
) -> tuple[np.ndarray, bool]:
    """
    Newton-Raphson IK for the 3 arm joints (shoulder_pan, shoulder_lift,
    elbow_flex) to place the end-effector at target_base.

    Wrist joints are held constant at q_wrist.

    Returns (q_arm_solution, converged).
    """
    q3 = q_arm.copy()
    eps = 1e-4

    for _ in range(max_iters):
        q_full = np.concatenate([q3, q_wrist])
        p = ee_position(q_full)
        error = target_base - p
        if np.linalg.norm(error) < tol:
            return q3, True

        # 3x3 position Jacobian for arm joints only
        J = np.zeros((3, 3))
        for i in range(3):
            q_pert = q_full.copy()
            q_pert[i] += eps
            J[:, i] = (ee_position(q_pert) - p) / eps

        try:
            dq = np.linalg.solve(J, error)
        except np.linalg.LinAlgError:
            return q3, False

        # Limit Newton step to prevent divergence
        max_newton_step = np.deg2rad(10.0)
        scale = np.max(np.abs(dq)) / max_newton_step
        if scale > 1.0:
            dq /= scale

        q3 = q3 + dq

    # Didn't converge within tolerance — return best effort
    return q3, False


def compute_wrist_correction(
    q_full_rad: np.ndarray,
    R_cam_base: np.ndarray,
    target_rot_cam: np.ndarray,
    gain: float = ORIENTATION_GAIN,
    damping: float = ORIENTATION_DAMPING,
) -> np.ndarray:
    """
    Compute wrist_flex and wrist_roll deltas for 2D orientation correction.

    Only corrects the two orientation axes visible in the camera image
    (camera Y and Z).  The camera X axis (tilt toward/away from camera)
    is left unconstrained.

    Returns dq_wrist as a (2,) array in radians.
    """
    eps = 1e-4

    # Current orientation in camera frame
    R_ee_cam = R_cam_base @ forward_kinematics(q_full_rad)[:3, :3]

    # Full 3D error, then keep only Y and Z components
    ori_err_2d = orientation_error(R_ee_cam, target_rot_cam)[1:]  # (2,)

    # 2-joint angular Jacobian for wrist_flex (joint 3) and wrist_roll (joint 4)
    R0 = forward_kinematics(q_full_rad)[:3, :3]
    J_ori_base = np.zeros((3, 2))
    for i in range(2):
        q = q_full_rad.copy()
        q[3 + i] += eps
        R_delta = forward_kinematics(q)[:3, :3] @ R0.T
        J_ori_base[0, i] = (R_delta[2, 1] - R_delta[1, 2]) / (2 * eps)
        J_ori_base[1, i] = (R_delta[0, 2] - R_delta[2, 0]) / (2 * eps)
        J_ori_base[2, i] = (R_delta[1, 0] - R_delta[0, 1]) / (2 * eps)

    # Rotate to camera frame and take Y, Z rows → (2, 2)
    J_2d = (R_cam_base @ J_ori_base)[1:, :]

    # Damped least-squares on the 2x2 system
    JJt = J_2d @ J_2d.T + damping * np.eye(2)
    return J_2d.T @ np.linalg.solve(JJt, gain * ori_err_2d)


# ============================================================
# CAMERA-TO-BASE CALIBRATION
# ============================================================

def calibrate_camera_to_base(
    robot: SOFollower,
    camera: RealSenseCamera,
    detector: ArucoDetector,
    side: str,
    perturb_deg: float = CALIBRATION_PERTURB_DEG,
    settle_s: float = SETTLE_TIME_S,
) -> np.ndarray:
    """
    Estimate the 3x3 rotation from robot base frame to camera frame.

    Perturbs shoulder_pan, shoulder_lift, and elbow_flex one at a time by
    a small angle, observing the resulting position change in both the
    camera frame (ArUco) and the base frame (FK).  Solves for R via
    Procrustes alignment (SVD projection onto SO(3)).

    Parameters
    ----------
    robot : SOFollower
        Connected arm with torque enabled.
    camera : RealSenseCamera
        Connected camera.
    detector : ArucoDetector
        Marker detector.
    side : 'left' or 'right'
        Which gripper marker to track.
    perturb_deg : float
        Perturbation magnitude in degrees.
    settle_s : float
        Time to wait after commanding a joint move.

    Returns
    -------
    R : ndarray (3, 3)
        Rotation such that  dp_cam ~ R @ dp_base.
    """
    delta_raw = perturb_deg * COUNTS_PER_DEG

    # Baseline observation
    joints_0 = read_joints_raw(robot)
    q_0 = get_fk_angles_rad(robot)
    p_base_0 = ee_position(q_0)

    frame = camera.get_frame()
    poses = detector.detect(frame)
    p_cam_0 = detector.get_gripper_pose(poses, side)

    if p_cam_0 is None:
        raise RuntimeError(
            f"Cannot see {side} gripper marker. "
            f"Position the arm so the marker faces the camera."
        )

    cam_deltas = []
    base_deltas = []

    for idx in [0, 1, 2]:  # shoulder_pan, shoulder_lift, elbow_flex
        name = FK_JOINT_NAMES[idx]

        # Perturb one joint
        target = dict(joints_0)
        target[name] = joints_0[name] + delta_raw
        write_joints_raw(robot, target)
        time.sleep(settle_s)

        # Observe perturbed state
        q_p = get_fk_angles_rad(robot)
        p_base_p = ee_position(q_p)
        frame = camera.get_frame()
        poses = detector.detect(frame)
        p_cam_p = detector.get_gripper_pose(poses, side)

        # Return to baseline
        write_joints_raw(robot, joints_0)
        time.sleep(settle_s)

        if p_cam_p is None:
            raise RuntimeError(
                f"Lost {side} gripper marker while perturbing {name}. "
                f"Ensure the marker stays visible during small motions."
            )

        cam_deltas.append(p_cam_p - p_cam_0)
        base_deltas.append(p_base_p - p_base_0)
        print(f"    {name}: base delta={np.linalg.norm(base_deltas[-1])*1000:.1f}mm, "
              f"cam delta={np.linalg.norm(cam_deltas[-1])*1000:.1f}mm")

    # Solve  C = R @ B  where B and C are 3x3 matrices of column deltas
    B = np.column_stack(base_deltas)
    C = np.column_stack(cam_deltas)

    if abs(np.linalg.det(B)) < 1e-12:
        raise RuntimeError(
            "Base-frame deltas are degenerate. Try a different starting pose."
        )

    R_est = C @ np.linalg.inv(B)

    # Project onto SO(3) via SVD (nearest proper rotation)
    U, _, Vt = np.linalg.svd(R_est)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def save_calibration(
    R_left: np.ndarray, R_right: np.ndarray,
    R_target_base_left: np.ndarray, R_target_base_right: np.ndarray,
    path: Path = CALIBRATION_FILE,
):
    """Save camera-to-base rotations and target orientations to JSON."""
    data = {
        "R_left": R_left.tolist(),
        "R_right": R_right.tolist(),
        "R_target_base_left": R_target_base_left.tolist(),
        "R_target_base_right": R_target_base_right.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Calibration saved to {path}")


def load_calibration(path: Path = CALIBRATION_FILE) -> dict[str, np.ndarray] | None:
    """Load saved calibration. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}


# ============================================================
# HOVER CONTROLLER
# ============================================================

def compute_hover_target(
    tray_pos_cam: np.ndarray,
    hover_height_m: float = HOVER_HEIGHT_M,
) -> np.ndarray:
    """
    Target gripper position in camera frame: aligned with the tray marker
    in X and Z, offset upward by hover_height_m in Y.

    Camera convention: +Y points down, so "above" is negative Y.
    """
    target = tray_pos_cam.copy()
    target[1] -= hover_height_m
    return target


def hover_step(
    robot: SOFollower,
    gripper_pos_cam: np.ndarray,
    target_pos_cam: np.ndarray,
    R_cam_base: np.ndarray,
    target_rot_cam: np.ndarray | None = None,
    max_step_deg: float = MAX_JOINT_STEP_DEG,
) -> float:
    """
    Decoupled position + orientation controller.

    Joints 0-2 (shoulder_pan, shoulder_lift, elbow_flex) are solved via
    exact Newton-Raphson IK for the full 3D target position.

    Joints 3-4 (wrist_flex, wrist_roll) correct the 2D gripper orientation
    visible in the camera image (ignoring toward/away-from-camera tilt).

    Returns the 3D position error magnitude (meters) before the step.
    """
    error_cam = target_pos_cam - gripper_pos_cam
    error_mag = np.linalg.norm(error_cam)

    q_rad = get_fk_angles_rad(robot)
    q_arm = q_rad[:3]
    q_wrist = q_rad[3:]

    # --- Position: exact IK for arm joints (0-2) ---
    # Convert camera-frame target to base-frame target.
    # p_base_target = p_base_current + R^T @ error_cam
    # (translation cancels in the delta)
    p_base_current = ee_position(q_rad)
    p_base_target = p_base_current + R_cam_base.T @ error_cam

    q_arm_target, converged = solve_position_ik(q_arm, q_wrist, p_base_target)

    # Proportionally clamp arm deltas (preserves direction toward target)
    dq_arm_deg = np.rad2deg(q_arm_target - q_arm)
    max_component = np.max(np.abs(dq_arm_deg))
    if max_component > max_step_deg:
        dq_arm_deg *= max_step_deg / max_component

    # --- Orientation: 2D wrist correction (joints 3-4) ---
    dq_wrist_deg = np.zeros(2)
    if target_rot_cam is not None:
        # Compute orientation at the clamped arm position we're about to command
        q_arm_clamped = q_arm + np.deg2rad(dq_arm_deg)
        q_after = np.concatenate([q_arm_clamped, q_wrist])
        dq_wrist_rad = compute_wrist_correction(q_after, R_cam_base, target_rot_cam)
        dq_wrist_deg = np.clip(np.rad2deg(dq_wrist_rad), -max_step_deg, max_step_deg)

    # --- Command all 5 joints ---
    dq_all_raw = np.concatenate([dq_arm_deg, dq_wrist_deg]) * COUNTS_PER_DEG

    joints = read_joints_raw(robot)
    cmd = dict(joints)
    for i, name in enumerate(FK_JOINT_NAMES):
        cmd[name] = joints[name] + dq_all_raw[i]
    write_joints_raw(robot, cmd)

    return error_mag


def hover_loop(
    robot: SOFollower,
    get_frame,
    detector: ArucoDetector,
    side: str,
    R_cam_base: np.ndarray,
    target_rot_cam: np.ndarray | None = None,
    hover_height_m: float = HOVER_HEIGHT_M,
    max_step_deg: float = MAX_JOINT_STEP_DEG,
    control_hz: float = CONTROL_HZ,
    stop_event: threading.Event | None = None,
):
    """
    Continuously servo one arm to hover above its tray marker.

    Parameters
    ----------
    get_frame : callable () -> ndarray | None
        Returns the latest camera frame (BGR) or None.
    side : 'left' or 'right'
        Which tray marker and gripper marker to use.
    R_cam_base : ndarray (3, 3)
        Rotation from calibrate_camera_to_base().
    target_rot_cam : ndarray (3, 3), optional
        Target gripper orientation in camera frame (captured at pre_grasp).
        If provided, wrist joints maintain vertical gripper alignment.
    stop_event : threading.Event, optional
        Set this to signal the loop to exit.
    """
    interval = 1.0 / control_hz
    tray_marker_id = (detector.config.tray_left if side == "left"
                      else detector.config.tray_right)
    miss_streak = 0
    step_count = 0

    print(f"  [{side}] Hover loop started (tracking tray marker ID {tray_marker_id}).")

    while stop_event is None or not stop_event.is_set():
        t0 = time.perf_counter()

        frame = get_frame()
        if frame is None:
            time.sleep(interval)
            continue

        poses = detector.detect(frame)
        grip = detector.get_gripper_pose(poses, side)
        tray_pos = poses[tray_marker_id][:3, 3].copy() if tray_marker_id in poses else None

        if tray_pos is None or grip is None:
            miss_streak += 1
            if miss_streak == 30:
                parts = []
                if tray_pos is None:
                    parts.append(f"tray_{side} (ID {tray_marker_id})")
                if grip is None:
                    parts.append(f"{side} gripper")
                print(f"  [{side}] Lost: {', '.join(parts)}  (holding position)")
            time.sleep(interval)
            continue

        if miss_streak >= 30:
            print(f"  [{side}] Markers recovered.")
        miss_streak = 0

        target = compute_hover_target(tray_pos, hover_height_m)
        error = hover_step(
            robot, grip, target, R_cam_base,
            target_rot_cam=target_rot_cam,
            max_step_deg=max_step_deg,
        )

        step_count += 1
        if step_count % (int(control_hz) * 2) == 0:  # every ~2 seconds
            print(
                f"  [{side}] err={error*1000:.1f}mm  "
                f"tray=({tray_pos[0]*1000:.0f},{tray_pos[1]*1000:.0f},{tray_pos[2]*1000:.0f})  "
                f"grip=({grip[0]*1000:.0f},{grip[1]*1000:.0f},{grip[2]*1000:.0f})"
            )

        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print(f"  [{side}] Hover loop stopped.")


# ============================================================
# FRAME PROVIDER
# ============================================================

class FrameProvider:
    """Thread-safe camera capture running in the background."""

    def __init__(self, camera: RealSenseCamera):
        self.camera = camera
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _run(self):
        while self._running:
            f = self.camera.get_frame()
            with self._lock:
                self._frame = f


# ============================================================
# MAIN -- continuous tracking when run directly
# ============================================================

def main():
    ports = load_ports()

    print("Connecting arms...")
    left_robot = connect_arm("left", ports["left"])
    right_robot = connect_arm("right", ports["right"])
    print(f"  Left on {ports['left']}, Right on {ports['right']}")

    print("Connecting camera...")
    camera = RealSenseCamera()
    camera.connect()
    print(f"  Camera connected (S/N: {camera.serial_number})")

    cfg = MarkerConfig()
    detector = ArucoDetector(
        config=cfg,
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )

    provider = None
    stop = threading.Event()

    try:
        # --- Move to pre_grasp via waypoints ---
        waypoints = load_waypoints()
        home_left = waypoints["home"]["left"]
        home_right = waypoints["home"]["right"]
        pre_grasp_left = waypoints["pre_grasp"]["left"]
        pre_grasp_right = waypoints["pre_grasp"]["right"]

        # Current position -> home (smooth)
        print("\nMoving to home position...")
        current_left = read_joints_raw(left_robot)
        current_right = read_joints_raw(right_robot)
        smooth_move(
            left_robot, right_robot,
            current_left, home_left,
            current_right, home_right,
            duration=STARTUP_DURATION_S,
            label="home",
        )
        time.sleep(0.5)

        # Home -> pre_grasp (smooth)
        print("Moving to pre-grasp position...")
        smooth_move(
            left_robot, right_robot,
            home_left, pre_grasp_left,
            home_right, pre_grasp_right,
            duration=APPROACH_DURATION_S,
            label="pre_grasp",
        )
        time.sleep(0.5)

        # --- Load or run calibration ---
        saved = load_calibration()
        if saved is not None:
            print("\nLoaded saved calibration from", CALIBRATION_FILE)
            R_left = saved["R_left"]
            R_right = saved["R_right"]
            R_target_base_left = saved["R_target_base_left"]
            R_target_base_right = saved["R_target_base_right"]
        else:
            # Capture target orientation at pre_grasp (gripper is vertical).
            # Record each arm's FK orientation in base frame now; after
            # calibration we'll rotate into camera frame.
            q_left = get_fk_angles_rad(left_robot)
            R_target_base_left = forward_kinematics(q_left)[:3, :3]
            q_right = get_fk_angles_rad(right_robot)
            R_target_base_right = forward_kinematics(q_right)[:3, :3]

            # Diagnostic: show which marker IDs the camera can see
            print("\nMarker check (verifying visibility)...")
            frame = camera.get_frame()
            poses = detector.detect(frame)
            id_to_label = {
                cfg.left_gripper: "left_gripper",
                cfg.tray_left: "tray_left",
                cfg.tray_right: "tray_right",
                cfg.right_gripper: "right_gripper",
            }
            for mid, label in id_to_label.items():
                if mid in poses:
                    p = poses[mid][:3, 3]
                    print(f"  ID {mid} ({label}): ({p[0]*1000:.0f}, {p[1]*1000:.0f}, {p[2]*1000:.0f}) mm")
                else:
                    print(f"  ID {mid} ({label}): NOT DETECTED")

            # Calibrate camera-to-base rotation for each arm.
            # Arms are now near the tray in a known configuration.
            print("\nCalibrating camera-to-base rotation...")
            print("  Ensure both gripper markers face the camera.\n")

            print("  Left arm:")
            R_left = calibrate_camera_to_base(left_robot, camera, detector, "left")
            print("  Left arm calibrated.\n")

            print("  Right arm:")
            R_right = calibrate_camera_to_base(right_robot, camera, detector, "right")
            print("  Right arm calibrated.\n")

            save_calibration(R_left, R_right, R_target_base_left, R_target_base_right)

        # Convert target orientations from base frame to camera frame
        target_rot_cam_left = R_left @ R_target_base_left
        target_rot_cam_right = R_right @ R_target_base_right

        # --- Start frame provider and hover loops ---
        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.3)

        threads = []
        for robot, side, R, R_target in [
            (left_robot, "left", R_left, target_rot_cam_left),
            (right_robot, "right", R_right, target_rot_cam_right),
        ]:
            t = threading.Thread(
                target=hover_loop,
                args=(robot, provider.get_frame, detector, side, R),
                kwargs={"target_rot_cam": R_target, "stop_event": stop},
                daemon=True,
            )
            t.start()
            threads.append(t)

        print("Hover tracking active. Slide the tray around. Ctrl+C to stop.\n")

        while not stop.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop.set()
        time.sleep(0.5)
        if provider is not None:
            provider.stop()
        for r in [left_robot, right_robot]:
            r.bus.sync_write("Torque_Enable", 0, normalize=False)
            r.config.disable_torque_on_disconnect = False
            r.disconnect()
        camera.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
