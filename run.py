"""
run.py - State machine for bimanual tray lift sequence.

Moves both SO-101 arms through a coordinated sequence:
  HOME -> PRE_GRASP -> HOVER -> APPROACH -> LIFT_WAITING -> LIFTING

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
DESCENT_DURATION_S   = 3.0   # seconds for grasp descent
NUDGE_DURATION_S     = 2.0   # seconds for nudge lift
GRIPPER_CLOSE_WAIT_S = 1.0   # wait after closing gripper
GRIPPER_OPEN_WAIT_S  = 0.5   # wait after opening gripper
RETURN_HOME_DURATION = 4.0   # seconds to return home at end

# ── Thresholds ──────────────────────────────────────────────
CONVERGENCE_THRESHOLD_M = 0.010   # 10 mm XZ error to leave HOVER
CONVERGENCE_READINGS    = 3       # consecutive sub-threshold readings
NUDGE_FRACTION          = 0.30    # 40% of descent delta
TAG_RISE_THRESHOLD_M    = 0.004   # 4 mm rise on opposing tag

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
    label: str = "",
):
    """Smoothly interpolate both arms in sync (minimum-jerk)."""
    interval = 1.0 / RECORD_FPS
    n_steps = max(1, int(duration * RECORD_FPS))
    if label:
        print(f"  [{label}] {duration:.1f}s, {n_steps} steps")
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
    label: str = "",
):
    """Smoothly interpolate one arm (minimum-jerk)."""
    interval = 1.0 / RECORD_FPS
    n_steps = max(1, int(duration * RECORD_FPS))
    if label:
        print(f"  [{label}] {duration:.1f}s, {n_steps} steps")
    for step in range(n_steps):
        t0 = time.perf_counter()
        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)
        write_joints_raw(robot, interpolate_joints(start, end, progress))
        elapsed = time.perf_counter() - t0
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)


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
    print(f"  [{side}] -> HOVER  (tray marker tracking, "
          f"threshold: {CONVERGENCE_THRESHOLD_M * 1000:.0f}mm)")

    smooth_bx = None
    smooth_bz = None
    alpha = 0.2
    miss_streak = 0
    converged_count = 0
    last_log_time = time.time()

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

        # ---- evaluate polynomial -> joint targets ----
        target_joints = motion_map.evaluate(target_bx, target_bz)

        # ---- measure XZ distance (gripper to offset-adjusted target) ----
        # The polynomial was trained with POSITION_OFFSET baked in, so the
        # gripper's intended position is (target - offset), not raw target.
        pos_off = POSITION_OFFSET.get(side, {})
        adjusted_bx = target_bx - pos_off.get("base_x", 0.0)
        adjusted_bz = target_bz - pos_off.get("base_z", 0.0)

        grip_cam = detector.get_gripper_pose(poses, side)
        xz_dist = None
        if grip_cam is not None:
            grip_base = cam_to_base(grip_cam, T_base_cam)
            xz_dist = float(np.sqrt(
                (grip_base[0] - adjusted_bx) ** 2
                + (grip_base[2] - adjusted_bz) ** 2
            ))

        # ---- raise-on-traverse ----
        if xz_dist is not None:
            raise_scale = xz_dist / RAISE_DIST_SCALE_M
            for name, offset in RAISE_OFFSETS.items():
                target_joints[name] += offset * raise_scale

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

        # ---- log every 1 second ----
        now = time.time()
        if now - last_log_time >= 1.0:
            if xz_dist is not None:
                print(f"  [{side}] HOVER: XZ error = {xz_dist * 1000:.1f}mm  "
                      f"(need < {CONVERGENCE_THRESHOLD_M * 1000:.0f}mm)")
            else:
                print(f"  [{side}] HOVER: gripper marker not visible")
            last_log_time = now

        # ---- convergence check ----
        if xz_dist is not None and xz_dist < CONVERGENCE_THRESHOLD_M:
            converged_count += 1
            if converged_count >= CONVERGENCE_READINGS:
                print(f"  [{side}] HOVER: CONVERGED  "
                      f"(XZ error {xz_dist * 1000:.1f}mm < "
                      f"{CONVERGENCE_THRESHOLD_M * 1000:.0f}mm, "
                      f"{converged_count} consecutive readings)")
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

    # ---- open gripper to max ----
    print(f"  [{side}] APPROACH: opening gripper to max ({gripper_open:.0f})")
    write_joints_raw(robot, {"gripper": gripper_open})
    time.sleep(GRIPPER_OPEN_WAIT_S)
    if stop_event.is_set():
        return

    # ---- descend using hardcoded delta (arm motors only) ----
    print(f"  [{side}] APPROACH: descending to grasp position "
          f"({DESCENT_DURATION_S:.1f}s)")
    current = read_joints_raw(robot)
    descent_start = {m: current[m] for m in ARM_MOTORS}
    descent_end = {m: current[m] + descent_delta[m] for m in ARM_MOTORS}
    smooth_move_single(
        robot, descent_start, descent_end,
        DESCENT_DURATION_S, label=f"{side} descent",
    )
    if stop_event.is_set():
        return

    # ---- close gripper ----
    print(f"  [{side}] APPROACH: closing gripper ({gripper_closed:.0f})")
    write_joints_raw(robot, {"gripper": gripper_closed})
    print(f"  [{side}] APPROACH: waiting {GRIPPER_CLOSE_WAIT_S:.1f}s "
          f"for gripper to close")
    time.sleep(GRIPPER_CLOSE_WAIT_S)
    if stop_event.is_set():
        return

    # ===========================================================
    #  LIFT_WAITING - nudge lift, then watch opposing tray tag
    # ===========================================================
    print(f"  [{side}] -> LIFT_WAITING")

    # ---- nudge lift (reverse of descent, scaled by NUDGE_FRACTION) ----
    print(f"  [{side}] LIFT_WAITING: nudge lift "
          f"({NUDGE_FRACTION * 100:.0f}% of descent delta, "
          f"{NUDGE_DURATION_S:.1f}s)")
    current = read_joints_raw(robot)
    nudge_start = {m: current[m] for m in ARM_MOTORS}
    nudge_end = {m: current[m] - descent_delta[m] * NUDGE_FRACTION
                 for m in ARM_MOTORS}
    smooth_move_single(
        robot, nudge_start, nudge_end,
        NUDGE_DURATION_S, label=f"{side} nudge lift",
    )
    if stop_event.is_set():
        return

    print(f"  [{side}] LIFT_WAITING: nudge complete, holding still")
    print(f"  [{side}] LIFT_WAITING: watching tray_{other_side} for "
          f">= {TAG_RISE_THRESHOLD_M * 1000:.0f}mm rise from baseline")

    other_tray_key = f"tray_{other_side}"
    baseline_y = baseline_tray_y[other_tray_key]
    last_log_time = time.time()

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

        now = time.time()
        if now - last_log_time >= 1.0:
            print(f"  [{side}] LIFT_WAITING: tray_{other_side} rise = "
                  f"{rise * 1000:.1f}mm  "
                  f"(need >= {TAG_RISE_THRESHOLD_M * 1000:.0f}mm)")
            last_log_time = now

        if rise >= TAG_RISE_THRESHOLD_M:
            print(f"  [{side}] LIFT_WAITING: tray_{other_side} rose "
                  f"{rise * 1000:.1f}mm >= "
                  f"{TAG_RISE_THRESHOLD_M * 1000:.0f}mm threshold  "
                  f"-- other arm confirmed!")
            break

        time.sleep(0.033)

    if stop_event.is_set():
        return

    # ===========================================================
    #  LIFTING - placeholder (not implemented yet)
    # ===========================================================
    print(f"  [{side}] -> LIFTING")
    print(f"  [{side}] *** ARRIVED AT LIFTING STATE ***")
    print(f"  [{side}] (lifting not yet implemented)")


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
        print("\nDescent deltas (grasp - pre_grasp):")
        descent_deltas: dict[str, dict[str, float]] = {}
        for side in sides:
            delta = {}
            for m in ARM_MOTORS:
                delta[m] = (waypoints["grasp"][side][m]
                            - waypoints["pre_grasp"][side][m])
            descent_deltas[side] = delta
            parts = [f"{m}: {delta[m]:+.0f}" for m in ARM_MOTORS]
            print(f"  {side.capitalize()}: {', '.join(parts)}")

        # ============================================================
        # STATE: HOME (synchronized, both arms)
        # ============================================================
        print(f"\n{'=' * 60}")
        print(f"STATE: HOME  --  moving both arms to home "
              f"({HOME_DURATION_S:.0f}s)")
        print(f"{'=' * 60}")
        smooth_move(
            robots["left"], robots["right"],
            read_joints_raw(robots["left"]),  waypoints["home"]["left"],
            read_joints_raw(robots["right"]), waypoints["home"]["right"],
            duration=HOME_DURATION_S, label="home",
        )
        time.sleep(0.5)

        # ============================================================
        # STATE: PRE_GRASP (synchronized, both arms)
        # ============================================================
        print(f"\n{'=' * 60}")
        print(f"STATE: PRE_GRASP  --  moving both arms to pre-grasp "
              f"({PRE_GRASP_DURATION_S:.0f}s)")
        print(f"{'=' * 60}")
        smooth_move(
            robots["left"], robots["right"],
            waypoints["home"]["left"],      waypoints["pre_grasp"]["left"],
            waypoints["home"]["right"],     waypoints["pre_grasp"]["right"],
            duration=PRE_GRASP_DURATION_S, label="pre_grasp",
        )
        time.sleep(0.5)

        # ── Start frame provider ────────────────────────────
        print("\nStarting camera frame provider...")
        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.5)

        # ── Record baseline tray tag positions ──────────────
        print("Recording baseline tray tag positions...")
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
        print(f"  Baseline tray_left  Y (cam): "
              f"{baseline_tray_y['tray_left'] * 1000:.1f}mm")
        print(f"  Baseline tray_right Y (cam): "
              f"{baseline_tray_y['tray_right'] * 1000:.1f}mm")
        print(f"  ({len(tray_left_ys)} readings averaged)")

        # ============================================================
        # STATES: HOVER -> APPROACH -> LIFT_WAITING -> LIFTING
        # (each arm runs independently in its own thread)
        # ============================================================
        print(f"\n{'=' * 60}")
        print("Launching per-arm state machines")
        print("  HOVER -> APPROACH -> LIFT_WAITING -> LIFTING")
        print(f"{'=' * 60}\n")

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
                    stop,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)
            print(f"  {side.capitalize()} arm thread started")

        print()

        # Wait for both arms to finish (or Ctrl+C)
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)

        if not stop.is_set():
            print(f"\n{'=' * 60}")
            print("Both arms reached LIFTING state!")
            print("Inspect the tray, then press Enter to release and return home.")
            print(f"{'=' * 60}")
            input()

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

        # ── Release grippers before moving home ─────────────
        print("\nOpening grippers to release tray...")
        try:
            write_joints_raw(robots["left"],
                             {"gripper": gripper_vals["left_open"]})
            write_joints_raw(robots["right"],
                             {"gripper": gripper_vals["right_open"]})
            time.sleep(2.0)
        except Exception as e:
            print(f"  Warning: could not open grippers: {e}")

        # ── Return to home ──────────────────────────────────
        print("Returning to home...")
        try:
            smooth_move(
                robots["left"], robots["right"],
                read_joints_raw(robots["left"]),
                waypoints["home"]["left"],
                read_joints_raw(robots["right"]),
                waypoints["home"]["right"],
                duration=RETURN_HOME_DURATION, label="home",
            )
        except Exception as e:
            print(f"  Warning: could not return home cleanly: {e}")

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
        print("Disconnected. Done.")


if __name__ == "__main__":
    main()
