"""
run.py - State machine for bimanual tray lift sequence.

Moves both SO-101 arms through a coordinated sequence:
  HOME -> PRE_GRASP -> HOVER -> APPROACH -> LIFT_WAITING -> LIFTING -> TRANSLATE -> LOWER -> RELEASE -> HOME

HOME and PRE_GRASP are synchronized (both arms move together).
From HOVER onward, each arm runs independently in its own thread.
Coordination emerges from shared visual observations (camera), not
direct communication between arms.

Usage: python run.py
"""

import sys
import time
import json
import threading
import numpy as np
from pathlib import Path

# Ensure repo root is on the path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Borrow a bunch of stuff from other modules
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from algorithmic.calibrate import load_base_transforms
from algorithmic.motion_map import (
    MotionMap,
    cam_to_base,
    MAX_JOINT_STEP_RAW,
    CONTROL_HZ,
    RAISE_DIST_SCALE_M,
    RAISE_OFFSETS,
)
from algorithmic.build_motion_map import POSITION_OFFSET


# ============================================================
# CONFIGURATION
# ============================================================

PORTS_FILE      = Path(__file__).parent / "Setup" / "ports.json"
WAYPOINTS_FILE  = Path(__file__).parent / "data_collection" / "waypoints.json"
CALIBRATION_DIR = Path(__file__).parent / "calibrations"
MOTION_MAP_DIR  = Path(__file__).parent / "algorithmic"

ARM_MOTORS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]

RECORD_FPS = 30

# ── Timing ──────────────────────────────────────────────────
HOME_DURATION_S      = 4.0   # seconds to move to home
PRE_GRASP_DURATION_S = 4.0   # seconds to move to pre-grasp
NUDGE_DURATION_S     = 2.0   # seconds for nudge lift
GRIPPER_CLOSE_WAIT_S = 1.0   # wait after closing gripper
GRIPPER_OPEN_WAIT_S  = 0.5   # wait after opening gripper
RELEASE_DURATION_S   = 3.0   # seconds to move to release waypoint
RETURN_HOME_DURATION = 4.0   # seconds to return home at end

# ── Thresholds ──────────────────────────────────────────────
CONVERGENCE_THRESHOLD_M = 0.018   # 18 mm XZ error to leave HOVER
HOVER_FEEDBACK_GAIN     = 0.5     # fraction of XZ error fed back to motion map
CONVERGENCE_READINGS    = 3       # consecutive sub-threshold readings
NUDGE_FRACTION          = 0.4    # 40% of descent delta
TAG_RISE_THRESHOLD_M    = 0.003   # 3 mm rise on opposing tag
LIFT_HEIGHT_M           = 0.060   # 60 mm total rise for lift
LIFT_STEP_FRACTION      = 0.03    # each lift step = 3% of descent delta
LIFT_STEP_INTERVAL_S    = 0.02    # seconds between lift steps
TRANSLATE_DURATION_S    = 5.0    # total translation time
TRANSLATE_STEP_INTERVAL = 0.05    # seconds between translate steps
# Per-arm height trim during translation, as fraction of descent delta.
# Negative = raise arm (subtract descent direction).
TRANSLATE_HEIGHT_TRIM   = {"left": -1.0, "right": -0.5}
TILT_PAUSE_THRESHOLD_M  = 0.005   # 5 mm height diff pauses higher side
GRASP_PROXIMITY_M       = 0.085   # 85 mm vertical distance to stop descent
LOWER_DURATION_S        = 4.0     # total time for lowering
LOWER_STEP_INTERVAL_S   = 0.02    # seconds between lower steps

# ── Baseline recording ──────────────────────────────────────
BASELINE_READINGS       = 10      # number of tray Y readings to average
BASELINE_TIMEOUT_FRAMES = 60      # max frames to try (~2 s at 30 fps)


# ============================================================
# HARDWARE HELPERS
# ============================================================

def load_ports() -> dict:
    if not PORTS_FILE.exists():
        print(f"ERROR: {PORTS_FILE} not found. Run Setup/find_ports.py first.")
        sys.exit(1)
    with open(PORTS_FILE) as f:
        return json.load(f)


def connect_arm(arm_name: str, port: str) -> SOFollower:
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
    return robot.bus.sync_read("Present_Position", normalize=False)


def write_joints_raw(robot: SOFollower, target: dict[str, float]):
    robot.bus.sync_write("Goal_Position", target, normalize=False)


def load_waypoints() -> dict:
    if not WAYPOINTS_FILE.exists():
        print(f"ERROR: {WAYPOINTS_FILE} not found.")
        sys.exit(1)
    with open(WAYPOINTS_FILE) as f:
        return json.load(f)


# ============================================================
# SMOOTH MOTION
# ============================================================

def minimum_jerk(t: float, duration: float) -> float:
    tau = np.clip(t / duration, 0.0, 1.0)
    return float(10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def interpolate_joints(
    start: dict[str, float],
    end: dict[str, float],
    progress: float,
) -> dict[str, float]:
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
):
    """Smoothly interpolate both arms in sync (minimum-jerk)."""
    interval = 1.0 / RECORD_FPS
    n_steps = max(1, int(duration * RECORD_FPS))
    for step in range(n_steps):
        t0 = time.perf_counter()
        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)
        write_joints_raw(left_robot, interpolate_joints(left_start, left_end, progress))
        write_joints_raw(right_robot, interpolate_joints(right_start, right_end, progress))
        elapsed = time.perf_counter() - t0
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)


def smooth_move_single(
    robot: SOFollower,
    start: dict, end: dict,
    duration: float,
):
    """Smoothly interpolate one arm (minimum-jerk)."""
    interval = 1.0 / RECORD_FPS
    n_steps = max(1, int(duration * RECORD_FPS))
    for step in range(n_steps):
        t0 = time.perf_counter()
        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)
        write_joints_raw(robot, interpolate_joints(start, end, progress))
        elapsed = time.perf_counter() - t0
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)


# ============================================================
# MOTION MAP INVERSE LOOKUP
# ============================================================

def find_closest_coords(
    motion_map: MotionMap,
    target_joints: dict[str, float],
    n_grid: int = 200,
) -> tuple[float, float, float]:
    """
    Find the (base_x, base_z) whose motion-map output best matches
    target_joints.  Returns (bx, bz, rmse).
    """
    b = motion_map.bounds
    bx_vals = np.linspace(b["base_x_min"], b["base_x_max"], n_grid)
    bz_vals = np.linspace(b["base_z_min"], b["base_z_max"], n_grid)

    target_arr = np.array([target_joints[m] for m in ARM_MOTORS])
    best_err = float("inf")
    best_bx, best_bz = float(bx_vals[0]), float(bz_vals[0])

    for bx in bx_vals:
        for bz in bz_vals:
            predicted = motion_map.evaluate(bx, bz)
            pred_arr = np.array([predicted[m] for m in ARM_MOTORS])
            err = float(np.sum((pred_arr - target_arr) ** 2))
            if err < best_err:
                best_err = err
                best_bx, best_bz = float(bx), float(bz)

    return best_bx, best_bz, float(np.sqrt(best_err / len(ARM_MOTORS)))


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
            try:
                f = self.camera.get_frame()
                with self._lock:
                    self._frame = f
            except RuntimeError as e:
                print(f"  [camera] Frame capture error: {e} -- retrying")
                time.sleep(0.1)


# ============================================================
# PER-ARM STATE MACHINE
# ============================================================

def run_arm(
    side: str,
    robot: SOFollower,
    get_frame,
    detector: ArucoDetector,
    motion_map: MotionMap,
    T_base_cam: np.ndarray,
    descent_delta: dict[str, float],
    gripper_open: float,
    gripper_closed: float,
    baseline_tray_y: dict[str, float],
    waypoints: dict,
    stop_event: threading.Event,
):
    """
    Per-arm state machine: HOVER -> APPROACH -> LIFT_WAITING -> LIFTING.

    Runs in its own thread. Each arm executes this independently; coordination
    emerges from shared camera observations, not direct communication.
    """
    other_side = "right" if side == "left" else "left"
    interval = 1.0 / CONTROL_HZ

    # ===========================================================
    #  HOVER - track tray marker via motion map, converge
    # ===========================================================
    print(f"  [{side}] -> HOVER")

    smooth_bx = None
    smooth_bz = None
    alpha = 0.2
    miss_streak = 0
    converged_count = 0

    tray_marker_id = (detector.config.tray_left if side == "left"
                      else detector.config.tray_right)

    while not stop_event.is_set():
        t0 = time.perf_counter()

        frame = get_frame()
        if frame is None:
            time.sleep(interval)
            continue

        poses = detector.detect(frame)

        # ---- detect tray marker ----
        tray_cam = detector.get_tray_marker_pose(poses, side)
        if tray_cam is None:
            miss_streak += 1
            if miss_streak == 30:
                print(f"  [{side}] HOVER: lost tray marker "
                      f"(ID {tray_marker_id}), holding position")
            time.sleep(interval)
            continue

        if miss_streak >= 30:
            print(f"  [{side}] HOVER: tray marker recovered")
        miss_streak = 0

        # ---- transform to base frame ----
        tray_base = cam_to_base(tray_cam, T_base_cam)

        if smooth_bx is None:
            smooth_bx = tray_base[0]
            smooth_bz = tray_base[2]
        else:
            smooth_bx = alpha * tray_base[0] + (1 - alpha) * smooth_bx
            smooth_bz = alpha * tray_base[2] + (1 - alpha) * smooth_bz

        target_bx, target_bz = smooth_bx, smooth_bz

        if not motion_map.in_bounds(target_bx, target_bz):
            time.sleep(interval)
            continue

        # ---- measure XZ error (gripper vs offset-adjusted target) ----
        # The polynomial was trained with POSITION_OFFSET baked in, so the
        # gripper's intended position is (target - offset), not raw target.
        pos_off = POSITION_OFFSET.get(side, {})
        adjusted_bx = target_bx - pos_off.get("base_x", 0.0)
        adjusted_bz = target_bz - pos_off.get("base_z", 0.0)

        grip_cam = detector.get_gripper_pose(poses, side)
        xz_dist = None
        error_bx = 0.0
        error_bz = 0.0
        if grip_cam is not None:
            grip_base = cam_to_base(grip_cam, T_base_cam)
            error_bx = adjusted_bx - grip_base[0]  # positive = gripper needs +bx
            error_bz = adjusted_bz - grip_base[2]  # positive = gripper needs +bz
            xz_dist = float(np.sqrt(error_bx ** 2 + error_bz ** 2))

        # ---- evaluate polynomial with feedback correction ----
        # Shift the query point by the observed error so the motion map
        # output drives the gripper toward the actual target, not just
        # the open-loop estimate.
        query_bx = target_bx + error_bx * HOVER_FEEDBACK_GAIN
        query_bz = target_bz + error_bz * HOVER_FEEDBACK_GAIN

        if not motion_map.in_bounds(query_bx, query_bz):
            query_bx, query_bz = target_bx, target_bz

        target_joints = motion_map.evaluate(query_bx, query_bz)

        # ---- hover raised: subtract half a descent delta so the arms
        #      track from above and swoop down as they converge ----
        for m in ARM_MOTORS:
            target_joints[m] -= descent_delta[m] * 0.5

        # ---- raise-on-traverse ----
        if xz_dist is not None:
            raise_scale = xz_dist / RAISE_DIST_SCALE_M
            for name, offset in RAISE_OFFSETS.items():
                target_joints[name] += offset * raise_scale

        # ---- height trim ----
        trim = TRANSLATE_HEIGHT_TRIM.get(side, 0.0)
        for m in ARM_MOTORS:
            target_joints[m] += descent_delta[m] * trim

        # ---- servo step (no dead-zone: keep correcting until converged) ----
        current_joints = read_joints_raw(robot)
        deltas = {
            name: target_joints[name] - current_joints[name]
            for name in ARM_MOTORS
        }
        max_abs = max(abs(d) for d in deltas.values())
        if max_abs > MAX_JOINT_STEP_RAW:
            scale = MAX_JOINT_STEP_RAW / max_abs
            deltas = {name: d * scale for name, d in deltas.items()}

        cmd = {name: current_joints[name] + deltas[name]
               for name in ARM_MOTORS}
        write_joints_raw(robot, cmd)

        # ---- convergence check ----
        if xz_dist is not None and xz_dist < CONVERGENCE_THRESHOLD_M:
            converged_count += 1
            if converged_count >= CONVERGENCE_READINGS:
                break
        else:
            converged_count = 0

        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if stop_event.is_set():
        return

    # ===========================================================
    #  APPROACH - open gripper, descend, close gripper
    # ===========================================================
    print(f"  [{side}] -> APPROACH")

    write_joints_raw(robot, {"gripper": gripper_open})
    time.sleep(GRIPPER_OPEN_WAIT_S)
    if stop_event.is_set():
        return

    descent_step = {m: descent_delta[m] * LIFT_STEP_FRACTION
                    for m in ARM_MOTORS}
    descent_target = {m: v for m, v in read_joints_raw(robot).items()
                      if m in ARM_MOTORS}

    while not stop_event.is_set():
        frame = get_frame()
        if frame is not None:
            poses = detector.detect(frame)
            tray_cam = detector.get_tray_marker_pose(poses, side)
            grip_cam = detector.get_gripper_pose(poses, side)

            if tray_cam is not None and grip_cam is not None:
                vert_dist = abs(tray_cam[1] - grip_cam[1])
                if vert_dist < GRASP_PROXIMITY_M:
                    break

        # Accumulate descent step
        for m in ARM_MOTORS:
            descent_target[m] += descent_step[m]
        write_joints_raw(robot, descent_target)

        time.sleep(LIFT_STEP_INTERVAL_S)
    if stop_event.is_set():
        return

    write_joints_raw(robot, {"gripper": gripper_closed})
    time.sleep(GRIPPER_CLOSE_WAIT_S)
    if stop_event.is_set():
        return

    # ===========================================================
    #  LIFT_WAITING - nudge lift, then watch opposing tray tag
    # ===========================================================
    print(f"  [{side}] -> LIFT_WAITING")

    current = read_joints_raw(robot)
    nudge_start = {m: current[m] for m in ARM_MOTORS}
    nudge_end = {m: current[m] - descent_delta[m] * NUDGE_FRACTION
                 for m in ARM_MOTORS}
    smooth_move_single(
        robot, nudge_start, nudge_end,
        NUDGE_DURATION_S,
    )
    if stop_event.is_set():
        return

    other_tray_key = f"tray_{other_side}"
    baseline_y = baseline_tray_y[other_tray_key]

    while not stop_event.is_set():
        frame = get_frame()
        if frame is None:
            time.sleep(0.033)
            continue

        poses = detector.detect(frame)
        tray_poses = detector.get_tray_poses(poses)
        if tray_poses is None:
            time.sleep(0.033)
            continue

        current_y = tray_poses[other_tray_key][1]
        # Camera Y points down, so a physical rise = decrease in Y
        rise = baseline_y - current_y

        if rise >= TAG_RISE_THRESHOLD_M:
            break

        time.sleep(0.033)

    if stop_event.is_set():
        return

    # ===========================================================
    #  LIFTING - slow, tilt-aware raise
    # ===========================================================
    print(f"  [{side}] -> LIFTING")

    my_tray_key = f"tray_{side}"
    other_tray_key = f"tray_{other_side}"
    my_baseline_y = baseline_tray_y[my_tray_key]
    lift_step_delta = {m: -descent_delta[m] * LIFT_STEP_FRACTION
                       for m in ARM_MOTORS}

    # Track a running goal position so tiny deltas accumulate even if
    # the servo hasn't physically moved yet (avoids dead-band trap).
    lift_target = {m: v for m, v in read_joints_raw(robot).items()
                   if m in ARM_MOTORS}

    while not stop_event.is_set():
        frame = get_frame()
        if frame is None:
            time.sleep(LIFT_STEP_INTERVAL_S)
            continue

        poses = detector.detect(frame)
        tray_poses = detector.get_tray_poses(poses)
        if tray_poses is None:
            time.sleep(LIFT_STEP_INTERVAL_S)
            continue

        my_y = tray_poses[my_tray_key][1]
        other_y = tray_poses[other_tray_key][1]
        my_rise = my_baseline_y - my_y        # camera Y down, so rise = decrease
        other_rise = baseline_tray_y[other_tray_key] - other_y

        if my_rise >= LIFT_HEIGHT_M:
            break

        # Only raise if my side is lower or equal to the other side
        # (lower physically = less rise from baseline)
        if my_rise <= other_rise:
            for m in ARM_MOTORS:
                lift_target[m] += lift_step_delta[m]
            write_joints_raw(robot, lift_target)

        time.sleep(LIFT_STEP_INTERVAL_S)

    if stop_event.is_set():
        return

    # Wait for the other side to also reach LIFT_HEIGHT_M
    while not stop_event.is_set():
        frame = get_frame()
        if frame is None:
            time.sleep(0.033)
            continue
        poses = detector.detect(frame)
        tray_poses = detector.get_tray_poses(poses)
        if tray_poses is None:
            time.sleep(0.033)
            continue
        other_rise = baseline_tray_y[other_tray_key] - tray_poses[other_tray_key][1]
        if other_rise >= LIFT_HEIGHT_M:
            break
        time.sleep(0.033)

    if stop_event.is_set():
        return

    # ===========================================================
    #  TRANSLATE — leader-follower at hover height
    #
    #  Left arm (leader): interpolates in (bx, bz) toward its
    #     destination, evaluating its own motion map each step.
    #  Right arm (follower): reads the left tray marker, offsets
    #     by tray width in base_x, evaluates its own motion map.
    #  Both arms run for TRANSLATE_DURATION_S on the same timer.
    # ===========================================================
    print(f"  [{side}] -> TRANSLATE (Phase B: "
          f"{'leader' if side == 'left' else 'follower'})")

    # Measure tray offset from markers (full XZ vector between the two
    # tray markers in this arm's base frame)
    tray_offset_bx = None
    tray_offset_bz = None
    for _ in range(30):
        frame = get_frame()
        if frame is not None:
            poses = detector.detect(frame)
            tray_poses = detector.get_tray_poses(poses)
            if tray_poses is not None:
                left_tray_base = cam_to_base(tray_poses["tray_left"], T_base_cam)
                right_tray_base = cam_to_base(tray_poses["tray_right"], T_base_cam)
                tray_offset_bx = right_tray_base[0] - left_tray_base[0]
                tray_offset_bz = right_tray_base[2] - left_tray_base[2]
                break
        time.sleep(0.033)

    n_steps_b = max(1, int(TRANSLATE_DURATION_S / TRANSLATE_STEP_INTERVAL))

    # Per-arm height trim: offset applied on top of motion map output
    trim = TRANSLATE_HEIGHT_TRIM.get(side, 0.0)
    height_trim = {m: descent_delta[m] * trim for m in ARM_MOTORS}

    if side == "left":
        # ── LEADER: interpolate in (bx, bz) toward destination ──
        target_joints = {m: waypoints["translated"][side][m] for m in ARM_MOTORS}
        dest_bx, dest_bz, _ = find_closest_coords(motion_map, target_joints)

        # Read current position
        bx_readings, bz_readings = [], []
        for _ in range(30):
            frame = get_frame()
            if frame is not None:
                poses = detector.detect(frame)
                tray_cam = detector.get_tray_marker_pose(poses, side)
                if tray_cam is not None:
                    tray_base = cam_to_base(tray_cam, T_base_cam)
                    bx_readings.append(tray_base[0])
                    bz_readings.append(tray_base[2])
            if len(bx_readings) >= 5:
                break
            time.sleep(0.033)

        if len(bx_readings) >= 3:
            start_bx = float(np.mean(bx_readings))
            start_bz = float(np.mean(bz_readings))
        else:
            # Fallback: reverse-lookup from current joint angles
            current_arm = {m: read_joints_raw(robot)[m] for m in ARM_MOTORS}
            start_bx, start_bz, _ = find_closest_coords(motion_map, current_arm)

        for step in range(n_steps_b):
            if stop_event.is_set():
                return

            t = (step + 1) / n_steps_b * TRANSLATE_DURATION_S
            progress = minimum_jerk(t, TRANSLATE_DURATION_S)

            interp_bx = start_bx + (dest_bx - start_bx) * progress
            interp_bz = start_bz + (dest_bz - start_bz) * progress
            cmd = motion_map.evaluate(interp_bx, interp_bz)
            for m in ARM_MOTORS:
                cmd[m] += height_trim[m]
            write_joints_raw(robot, cmd)

            time.sleep(TRANSLATE_STEP_INTERVAL)

    else:
        smooth_fbx = None
        smooth_fbz = None
        f_alpha = 0.1

        for step in range(n_steps_b):
            if stop_event.is_set():
                return

            frame = get_frame()
            if frame is not None:
                poses = detector.detect(frame)
                # Read the LEFT tray marker and transform to RIGHT arm's base frame
                left_tray_cam = detector.get_tray_marker_pose(poses, "left")
                if left_tray_cam is not None:
                    left_in_my_base = cam_to_base(left_tray_cam, T_base_cam)
                    # Offset by full tray vector to get where my marker should be
                    my_target_bx = left_in_my_base[0] + tray_offset_bx
                    my_target_bz = left_in_my_base[2] + tray_offset_bz

                    if smooth_fbx is None:
                        smooth_fbx = my_target_bx
                        smooth_fbz = my_target_bz
                    else:
                        smooth_fbx = f_alpha * my_target_bx + (1 - f_alpha) * smooth_fbx
                        smooth_fbz = f_alpha * my_target_bz + (1 - f_alpha) * smooth_fbz

                    if motion_map.in_bounds(smooth_fbx, smooth_fbz):
                        cmd = motion_map.evaluate(smooth_fbx, smooth_fbz)
                        for m in ARM_MOTORS:
                            cmd[m] += height_trim[m]
                        write_joints_raw(robot, cmd)

            time.sleep(TRANSLATE_STEP_INTERVAL)

    if stop_event.is_set():
        return

    # ===========================================================
    #  LOWER - descend from translated to lowered waypoint,
    #          pausing the lower side if tray tilts
    # ===========================================================
    print(f"  [{side}] -> LOWER")

    lower_dest = {m: waypoints["lowered"][side][m] for m in ARM_MOTORS}
    arm_start = {m: v for m, v in read_joints_raw(robot).items()
                 if m in ARM_MOTORS}
    n_steps = max(1, int(LOWER_DURATION_S / LOWER_STEP_INTERVAL_S))
    step = 0

    while step < n_steps and not stop_event.is_set():
        # Progress along the path (minimum-jerk)
        t = (step + 1) / n_steps * LOWER_DURATION_S
        progress = minimum_jerk(t, LOWER_DURATION_S)

        should_step = True

        # Check tray tilt — pause if my side is LOWER (other side catches up)
        frame = get_frame()
        if frame is not None:
            poses = detector.detect(frame)
            tray_poses = detector.get_tray_poses(poses)
            if tray_poses is not None:
                my_y = tray_poses[my_tray_key][1]
                other_y = tray_poses[other_tray_key][1]
                # Camera Y down: larger Y = physically lower
                # Pause if my side is lower than other by threshold
                tilt = my_y - other_y  # positive = my side is lower
                if tilt > TILT_PAUSE_THRESHOLD_M:
                    should_step = False

        if should_step:
            cmd = {m: arm_start[m] + (lower_dest[m] - arm_start[m]) * progress
                   for m in ARM_MOTORS}
            write_joints_raw(robot, cmd)
            step += 1

        time.sleep(LOWER_STEP_INTERVAL_S)

    if stop_event.is_set():
        return

    print(f"  [{side}] -> RELEASE")

    write_joints_raw(robot, {"gripper": gripper_open})
    time.sleep(GRIPPER_OPEN_WAIT_S)
    if stop_event.is_set():
        return

    current = read_joints_raw(robot)
    release_start = {m: current[m] for m in ARM_MOTORS}
    release_end = {m: waypoints["release"][side][m] for m in ARM_MOTORS}
    smooth_move_single(robot, release_start, release_end, RELEASE_DURATION_S)
    if stop_event.is_set():
        return

    print(f"  [{side}] -> HOME")

    current = read_joints_raw(robot)
    home_start = {m: current[m] for m in ARM_MOTORS}
    home_end = {m: waypoints["home"][side][m] for m in ARM_MOTORS}
    smooth_move_single(robot, home_start, home_end, RETURN_HOME_DURATION)

    print(f"  [{side}] DONE")


# ============================================================
# MAIN
# ============================================================

def main():
    sides = ["left", "right"]

    ports = load_ports()
    waypoints = load_waypoints()

    # ── Connect arms ────────────────────────────────────────
    print("Connecting arms...")
    robots: dict[str, SOFollower] = {}
    for side in sides:
        robots[side] = connect_arm(side, ports[side])
        print(f"  {side.capitalize()} on {ports[side]}")

    # ── Connect camera ──────────────────────────────────────
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
    threads = []
    gripper_vals = waypoints["_gripper"]

    try:
        # ── Load base transforms ────────────────────────────
        print("\nLoading base transforms...")
        base_transforms = load_base_transforms()
        for side in sides:
            if side not in base_transforms:
                print(f"  ERROR: No transform for {side}. "
                      f"Run calibrate.py first.")
                raise SystemExit(1)
        print("  Loaded.")

        T_base_cams = {
            side: np.linalg.inv(base_transforms[side])
            for side in sides
        }

        # ── Load motion maps ───────────────────────────────
        print("Loading motion maps...")
        motion_maps: dict[str, MotionMap] = {}
        for side in sides:
            mm_path = MOTION_MAP_DIR / f"motion_map_{side}.json"
            if not mm_path.exists():
                print(f"  ERROR: {mm_path} not found. "
                      f"Run record_motion_map.py + build_motion_map.py.")
                raise SystemExit(1)
            motion_maps[side] = MotionMap(mm_path)
            mm = motion_maps[side]
            print(f"  {side.capitalize()}: "
                  f"{mm.metadata['n_training_points']} training pts, "
                  f"degree {mm.metadata['poly_degree']}")

        # ── Compute descent deltas from waypoints ───────────
        descent_deltas: dict[str, dict[str, float]] = {}
        for side in sides:
            delta = {}
            for m in ARM_MOTORS:
                if m in ("shoulder_pan", "wrist_roll"):
                    delta[m] = 0.0  # ignore noise in pan/roll
                else:
                    delta[m] = (waypoints["grasp"][side][m]
                                - waypoints["pre_grasp"][side][m])
            descent_deltas[side] = delta
        print("\n  -> HOME")
        smooth_move(
            robots["left"], robots["right"],
            read_joints_raw(robots["left"]),  waypoints["home"]["left"],
            read_joints_raw(robots["right"]), waypoints["home"]["right"],
            duration=HOME_DURATION_S,
        )
        time.sleep(0.5)

        # Move to pre_grasp raised by one descent delta (arms start high)
        raised_pre_grasp = {}
        for side in sides:
            raised_pre_grasp[side] = dict(waypoints["pre_grasp"][side])
            for m in ARM_MOTORS:
                raised_pre_grasp[side][m] -= descent_deltas[side][m]

        print("  -> PRE_GRASP (raised)")
        smooth_move(
            robots["left"], robots["right"],
            waypoints["home"]["left"],      raised_pre_grasp["left"],
            waypoints["home"]["right"],     raised_pre_grasp["right"],
            duration=PRE_GRASP_DURATION_S,
        )
        time.sleep(0.5)

        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.5)

        # ── Record baseline tray tag positions ──────────────
        tray_left_ys = []
        tray_right_ys = []
        for _ in range(BASELINE_TIMEOUT_FRAMES):
            frame = provider.get_frame()
            if frame is not None:
                poses = detector.detect(frame)
                tray_poses = detector.get_tray_poses(poses)
                if tray_poses is not None:
                    tray_left_ys.append(tray_poses["tray_left"][1])
                    tray_right_ys.append(tray_poses["tray_right"][1])
            if len(tray_left_ys) >= BASELINE_READINGS:
                break
            time.sleep(0.033)

        if len(tray_left_ys) < 3:
            print(f"  ERROR: Only got {len(tray_left_ys)} tray tag readings. "
                  f"Need at least 3.")
            print("  Check that both tray tags are visible to the camera.")
            raise SystemExit(1)

        baseline_tray_y = {
            "tray_left":  float(np.mean(tray_left_ys)),
            "tray_right": float(np.mean(tray_right_ys)),
        }

        # ── Launch per-arm state machines ───────────────────
        print("\nLaunching arm state machines...")
        for side in sides:
            t = threading.Thread(
                target=run_arm,
                args=(
                    side,
                    robots[side],
                    provider.get_frame,
                    detector,
                    motion_maps[side],
                    T_base_cams[side],
                    descent_deltas[side],
                    gripper_vals[f"{side}_open"],
                    gripper_vals[f"{side}_closed"],
                    baseline_tray_y,
                    waypoints,
                    stop,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Wait for both arms to finish (or Ctrl+C)
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n\nCtrl+C -- stopping...")

    finally:
        stop.set()

        # Wait for arm threads to fully exit before touching the bus
        for t in threads:
            t.join(timeout=3.0)
            if t.is_alive():
                print(f"  Warning: arm thread did not exit in time")

        if provider is not None:
            provider.stop()

        # On Ctrl+C, release grippers and go home as safety fallback.
        # On normal completion the arms already did this themselves.
        if stop.is_set():
            try:
                write_joints_raw(robots["left"],
                                 {"gripper": gripper_vals["left_open"]})
                write_joints_raw(robots["right"],
                                 {"gripper": gripper_vals["right_open"]})
                time.sleep(1.0)
                smooth_move(
                    robots["left"], robots["right"],
                    read_joints_raw(robots["left"]),
                    waypoints["home"]["left"],
                    read_joints_raw(robots["right"]),
                    waypoints["home"]["right"],
                    duration=RETURN_HOME_DURATION,
                )
            except Exception as e:
                print(f"  Warning: cleanup failed: {e}")

        time.sleep(0.3)

        for side in sides:
            try:
                robots[side].bus.sync_write(
                    "Torque_Enable", 0, normalize=False,
                )
                robots[side].config.disable_torque_on_disconnect = False
                robots[side].disconnect()
            except Exception:
                pass
        camera.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
