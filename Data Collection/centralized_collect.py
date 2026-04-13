#!python
"""
Centralized deterministic data collection for bimanual tray lifting.

A single process controls both SO-101 arms through a scripted sequence of
coordinated moves: approach -> grasp -> lift -> translate -> lower -> release.
Both arms move in lockstep with smooth minimum-jerk interpolation.

The algorithm uses joint-space waypoints that are recorded interactively
via --teach mode (you physically position both arms at each phase, press
ENTER, and the joint angles are saved). Then --collect replays those
waypoints with smooth interpolation while recording RGB frames and joint
data for GNN training.

Usage:
  python centralized_collect.py --teach              Record waypoints
  python centralized_collect.py --collect 20         Collect 20 episodes
  python centralized_collect.py --collect 1 --dry-run  Test run, no saving
"""

import sys
import os
import time
import json
import csv
import threading
import argparse
import enum
import numpy as np
from pathlib import Path

# Allow importing from sibling directories
sys.path.insert(0, str(Path(__file__).parent.parent / "GNN Pipeline"))

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig


# ============================================================
# TUNABLE PARAMETERS
# ============================================================
# All distances in meters, angles in degrees, times in seconds.
# Adjust these based on your physical setup and desired behavior.

# --- Pre-grasp positioning ---
# Vertical offset: how far ABOVE the tray tag each gripper hovers
# before descending to grasp. Positive = above.
PRE_GRASP_VERTICAL_OFFSET_M = 0.050        # 50 mm

# Horizontal offset: how far OUTWARD from the tray tag each gripper
# starts before moving in to grasp. Left arm moves right (+X in camera
# frame), right arm moves left (-X).
PRE_GRASP_HORIZONTAL_OFFSET_M = 0.030      # 30 mm

# Depth alignment: both gripper AprilTags should be at the same depth
# (Z in camera frame) as the tray tags before attempting the grasp.
# This tolerance defines "close enough" for the depth check.
DEPTH_ALIGNMENT_TOLERANCE_M = 0.010         # 10 mm

# --- Gripper positions (degrees) ---
# These will need tuning for your specific gripper/tray combination.
GRIPPER_OPEN_DEG = 45.0                     # wide open
GRIPPER_CLOSED_DEG = -30.0                  # clamped on tray edge

# --- Post-grasp vertical lift ---
# How far to raise the tray after both grippers close.
LIFT_HEIGHT_M = 0.050                       # 50 mm

# --- Horizontal translation ---
# How far to move the tray forward (toward/away from camera) after lift.
TRANSLATE_DISTANCE_M = 0.100                # 100 mm

# --- Lower (set down) ---
# How far to lower the tray when placing it down. Usually matches lift.
LOWER_HEIGHT_M = 0.050                      # 50 mm

# --- Phase durations (control speed of each movement) ---
# Longer duration = slower, smoother motion. Shorter = faster.
APPROACH_DURATION_S = 5.0                   # home -> pre-grasp
DESCEND_DURATION_S = 2.0                    # pre-grasp -> grasp position
GRASP_CLOSE_DURATION_S = 0.5               # time to close grippers
GRASP_SETTLE_S = 0.5                        # pause after close for firm grip
LIFT_DURATION_S = 2.0                       # vertical raise
TRANSLATE_DURATION_S = 4.0                  # horizontal translation
LOWER_DURATION_S = 2.0                      # lower to table
RELEASE_OPEN_DURATION_S = 0.5              # time to open grippers
RETREAT_DURATION_S = 2.0                    # grasp position -> home

# --- Pauses ---
INTER_PHASE_PAUSE_S = 1.0                  # brief pause between phases

# --- Recording ---
RECORD_FPS = 30
OUTPUT_DIR = Path(__file__).parent / "episodes"

# --- Safety ---
MAX_TILT_DEG = 15.0                         # emergency stop if tray exceeds
MAX_JOINT_SPEED_DEG_S = 60.0                # clamp joint velocity
MAX_CONSECUTIVE_TILT_MISREADS = 150         # stop after this many unreadable frames (~5s at 30Hz) - Ignoring any issues for now!

# --- Randomization (for data diversity across episodes) ---
# Small random perturbations make the dataset richer.
SPEED_JITTER_FRACTION = 0.10                # +/-10% variation on durations
GRIPPER_CLOSE_JITTER_DEG = 2.0              # +/-2 deg on grip tightness

# --- Hardware paths ---
PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"
WAYPOINTS_FILE = Path(__file__).parent / "waypoints.json"

# Motor names on each arm (6 motors: 5 joints + gripper)
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
FK_JOINT_NAMES = MOTOR_NAMES[:5]            # gripper excluded from FK


# ============================================================
# WAYPOINT DEFINITIONS
# ============================================================
# These are the phases the arms move through, in order.
# Joint angles are recorded via --teach mode and saved to waypoints.json.

PHASE_SEQUENCE = [
    "home",         # neutral resting position, arms clear of tray
    "pre_grasp",    # hovering above/outside tray edges, grippers open
    "grasp",        # grippers aligned with tray edges, ready to close
    # (gripper close happens here — not a movement phase)
    "lifted",       # tray raised vertically
    "translated",   # tray moved horizontally (forward)
    "lowered",      # tray lowered back to table height
    # (gripper open happens here — not a movement phase)
    "release",      # arms pulled back slightly after releasing
]

# Maps each movement phase to its duration parameter.
PHASE_DURATIONS = {
    ("home", "pre_grasp"):          APPROACH_DURATION_S,
    ("pre_grasp", "grasp"):         DESCEND_DURATION_S,
    ("grasp", "lifted"):            LIFT_DURATION_S,
    ("lifted", "translated"):       TRANSLATE_DURATION_S,
    ("translated", "lowered"):      LOWER_DURATION_S,
    ("lowered", "release"):         RETREAT_DURATION_S,
}


# ============================================================
# MOTION HELPERS
# ============================================================

def minimum_jerk(t: float, duration: float) -> float:
    """
    Minimum-jerk trajectory scalar: maps time t in [0, duration] to
    a smooth progress value in [0, 1] with zero velocity and acceleration
    at both endpoints. Produces natural-looking motion.
    """
    tau = np.clip(t / duration, 0.0, 1.0)
    return float(10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def interpolate_joints(
    start: dict[str, float],
    end: dict[str, float],
    progress: float,
) -> dict[str, float]:
    """Linearly interpolate between two joint-angle dicts at a given progress [0,1]."""
    return {
        name: start[name] + (end[name] - start[name]) * progress
        for name in start
    }


def jitter_duration(base_duration: float) -> float:
    """Apply small random perturbation to a phase duration for data diversity."""
    jitter = 1.0 + np.random.uniform(-SPEED_JITTER_FRACTION, SPEED_JITTER_FRACTION)
    return base_duration * jitter


# ============================================================
# HARDWARE CONNECTION
# ============================================================

def load_ports() -> dict:
    """Load COM port assignments from ports.json."""
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


def read_joints(robot: SOFollower) -> dict[str, float]:
    """Read current joint positions (raw encoder values) from all 6 motors."""
    return robot.bus.sync_read("Present_Position", normalize=False)


def write_joints(robot: SOFollower, target: dict[str, float]):
    """Command all 6 motors to target positions (raw encoder values)."""
    robot.bus.sync_write("Goal_Position", target, normalize=False)


def read_joints_deg(robot: SOFollower) -> dict[str, float]:
    """Read current joint positions in degrees (normalized)."""
    return robot.bus.sync_read("Present_Position", normalize=True)


# ============================================================
# WAYPOINT I/O
# ============================================================

def save_waypoints(waypoints: dict):
    """Save waypoint dict to JSON."""
    with open(WAYPOINTS_FILE, "w") as f:
        json.dump(waypoints, f, indent=2)
    print(f"Waypoints saved to {WAYPOINTS_FILE}")


def load_waypoints() -> dict:
    """Load waypoints from JSON. Exits if file missing."""
    if not WAYPOINTS_FILE.exists():
        print(f"ERROR: {WAYPOINTS_FILE} not found. Run --teach first.")
        sys.exit(1)
    with open(WAYPOINTS_FILE) as f:
        return json.load(f)


# ============================================================
# DATA RECORDING
# ============================================================

class EpisodeRecorder:
    """
    Records RGB frames and joint angles for both arms at RECORD_FPS.

    Data is written to:
        episodes/episode_NNN/
            rgb/000000.png, 000001.png, ...
            joints_left.csv
            joints_right.csv
            metadata.json
    """

    def __init__(self, episode_dir: Path, camera: RealSenseCamera,
                 left_robot: SOFollower, right_robot: SOFollower):
        self.episode_dir = episode_dir
        self.rgb_dir = episode_dir / "rgb"
        self.camera = camera
        self.left_robot = left_robot
        self.right_robot = right_robot
        self.frame_index = 0
        self.start_time = None
        self._left_rows: list[list] = []
        self._right_rows: list[list] = []

    def setup(self):
        """Create output directories."""
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.frame_index = 0

    def record_frame(self):
        """Capture one synchronized snapshot: RGB + both arms' joints."""
        timestamp = time.time() - self.start_time

        # RGB frame
        frame = self.camera.get_frame()
        frame_path = self.rgb_dir / f"{self.frame_index:06d}.png"
        import cv2
        cv2.imwrite(str(frame_path), frame)

        # Joint angles (raw encoder values for exact replay)
        left_joints = read_joints(self.left_robot)
        right_joints = read_joints(self.right_robot)

        self._left_rows.append(
            [timestamp] + [left_joints[m] for m in MOTOR_NAMES]
        )
        self._right_rows.append(
            [timestamp] + [right_joints[m] for m in MOTOR_NAMES]
        )

        self.frame_index += 1
        return frame  # for optional live display

    def save(self, metadata: dict | None = None):
        """Write CSV files and metadata."""
        header = ["timestamp"] + MOTOR_NAMES

        for filename, rows in [("joints_left.csv", self._left_rows),
                                ("joints_right.csv", self._right_rows)]:
            with open(self.episode_dir / filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

        meta = {
            "fps": RECORD_FPS,
            "n_frames": self.frame_index,
            "duration_s": round(self._left_rows[-1][0], 3) if self._left_rows else 0,
            "motor_names": MOTOR_NAMES,
            "format": "raw_encoder",
        }
        if metadata:
            meta.update(metadata)

        with open(self.episode_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved {self.frame_index} frames to {self.episode_dir}")


# ============================================================
# PERCEPTION VALIDATION
# ============================================================

def check_depth_alignment(detector: ArucoDetector, frame: np.ndarray) -> bool:
    """
    Verify both gripper markers are at approximately the same depth (Z)
    as the tray markers. Returns True if aligned within tolerance.
    """
    poses = detector.detect(frame)
    tray = detector.get_tray_poses(poses)
    left_grip = detector.get_gripper_pose(poses, "left")
    right_grip = detector.get_gripper_pose(poses, "right")

    if tray is None or left_grip is None or right_grip is None:
        print("  WARNING: Not all markers visible for depth check.")
        return False

    left_ok = abs(left_grip[2] - tray["tray_left"][2]) < DEPTH_ALIGNMENT_TOLERANCE_M
    right_ok = abs(right_grip[2] - tray["tray_right"][2]) < DEPTH_ALIGNMENT_TOLERANCE_M

    if not left_ok or not right_ok:
        print(f"  Depth bad: "
              f"tray_L={tray['tray_left'][2]*1000:.1f}mm  grip_L={left_grip[2]*1000:.1f}mm  |  "
              f"tray_R={tray['tray_right'][2]*1000:.1f}mm  grip_R={right_grip[2]*1000:.1f}mm  "
              f"(tolerance={DEPTH_ALIGNMENT_TOLERANCE_M*1000:.1f}mm)")
    return left_ok and right_ok # return true if both are true :)


class TiltStatus(enum.Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


def check_tray_tilt(detector: ArucoDetector, frame: np.ndarray) -> TiltStatus:
    """Check tray tilt against safety limits.

    Returns SAFE, UNSAFE, or UNKNOWN (markers not visible).
    """
    poses = detector.detect(frame)
    tray = detector.get_tray_poses(poses)
    if tray is None:
        return TiltStatus.UNKNOWN

    dz = abs(tray["tray_right"][2] - tray["tray_left"][2])
    spacing = detector.config.tray_marker_spacing_m
    tilt_deg = np.degrees(np.arctan2(dz, spacing))

    if tilt_deg > MAX_TILT_DEG:
        print(f"  SAFETY: Tray tilt {tilt_deg:.1f} deg exceeds limit {MAX_TILT_DEG} deg!")
        return TiltStatus.UNSAFE
    return TiltStatus.SAFE


# ============================================================
# PHASE EXECUTION
# ============================================================

def execute_movement(
    left_robot: SOFollower,
    right_robot: SOFollower,
    left_start: dict, left_end: dict,
    right_start: dict, right_end: dict,
    duration: float,
    recorder: EpisodeRecorder | None = None,
    detector: ArucoDetector | None = None,
    camera: RealSenseCamera | None = None,
    phase_name: str = "",
) -> bool:
    """
    Smoothly interpolate both arms from start to end waypoints over
    the given duration using minimum-jerk timing. Records data at
    RECORD_FPS if a recorder is provided.

    Returns False if a safety check triggers (tilt exceeded).
    """
    fps = RECORD_FPS
    interval = 1.0 / fps
    n_steps = max(1, int(duration * fps))

    print(f"  [{phase_name}] {duration:.1f}s, {n_steps} steps")

    consecutive_misreads = 0

    for step in range(n_steps):
        t_start = time.perf_counter()

        # Smooth progress 0 -> 1
        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)

        # Interpolate joint targets
        left_target = interpolate_joints(left_start, left_end, progress)
        right_target = interpolate_joints(right_start, right_end, progress)

        # Command both arms (as close together as possible for sync)
        write_joints(left_robot, left_target)
        write_joints(right_robot, right_target)

        # Record if active
        if recorder is not None:
            recorder.record_frame()

        # Safety: check tray tilt during critical phases
        if detector is not None and camera is not None:
            if phase_name in ("lift", "translate"):
                frame = camera.get_frame()
                status = check_tray_tilt(detector, frame)

                if status == TiltStatus.UNSAFE:
                    print("  STOP: tilt limit exceeded.")
                    return False
                elif status == TiltStatus.UNKNOWN:
                    consecutive_misreads += 1
                    if consecutive_misreads >= MAX_CONSECUTIVE_TILT_MISREADS:
                        print(f"  STOP: lost tray tracking for "
                              f"{consecutive_misreads} consecutive frames.")
                        return False
                else:
                    consecutive_misreads = 0

        # Maintain target frame rate
        elapsed = time.perf_counter() - t_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    return True


def execute_gripper(
    left_robot: SOFollower,
    right_robot: SOFollower,
    left_current: dict, right_current: dict,
    target_deg: float,
    duration: float,
    recorder: EpisodeRecorder | None = None,
    phase_name: str = "",
):
    """
    Move only the grippers (both arms simultaneously) to a target
    position while keeping all other joints stationary.
    """
    # Build end waypoints: same as current but with gripper changed
    left_end = dict(left_current)
    right_end = dict(right_current)

    # Convert gripper target from degrees to raw encoder value.
    # For now, we store the target directly — the raw value will need
    # calibration. TODO: use the calibration formula to convert degrees -> raw.
    # Placeholder: treat the target as a raw offset from current.
    # In practice, you'll tune GRIPPER_OPEN_DEG and GRIPPER_CLOSED_DEG
    # as raw encoder values during --teach mode.
    left_end["gripper"] = target_deg
    right_end["gripper"] = target_deg

    execute_movement(
        left_robot, right_robot,
        left_current, left_end,
        right_current, right_end,
        duration=duration,
        recorder=recorder,
        phase_name=phase_name,
    )


def pause(duration: float, recorder: EpisodeRecorder | None = None,
          left_robot: SOFollower | None = None,
          right_robot: SOFollower | None = None):
    """Hold position for a duration, still recording frames if active."""
    if recorder is None or duration <= 0:
        time.sleep(duration)
        return

    fps = RECORD_FPS
    interval = 1.0 / fps
    n_steps = max(1, int(duration * fps))

    for _ in range(n_steps):
        t_start = time.perf_counter()
        recorder.record_frame()
        elapsed = time.perf_counter() - t_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================
# TEACH MODE
# ============================================================

def teach(left_robot: SOFollower, right_robot: SOFollower):
    """
    Interactively record waypoints by physically positioning both arms.

    For each phase in PHASE_SEQUENCE, the operator moves both arms to the
    desired pose and presses ENTER. The joint angles are read and saved.
    """
    print("\n=== TEACH MODE ===")
    print("For each phase, position BOTH arms and press ENTER to record.")
    print("Motors are free to move by hand.\n")

    # Disable torque so arms move freely
    left_robot.bus.disable_torque()
    right_robot.bus.disable_torque()

    waypoints = {}

    for phase in PHASE_SEQUENCE:
        print(f"--- Phase: {phase} ---")
        print(f"  Position both arms for '{phase}', then press ENTER.")

        if phase == "grasp":
            print("  (Position grippers right at the tray edges, as if about to close.)")
        elif phase == "pre_grasp":
            print(f"  (Hover ~{PRE_GRASP_VERTICAL_OFFSET_M*1000:.0f}mm above and "
                  f"~{PRE_GRASP_HORIZONTAL_OFFSET_M*1000:.0f}mm outward from tray edges.)")
        elif phase == "lifted":
            print(f"  (Tray grasped and raised ~{LIFT_HEIGHT_M*1000:.0f}mm.)")
        elif phase == "translated":
            print(f"  (Tray raised and moved forward ~{TRANSLATE_DISTANCE_M*1000:.0f}mm.)")
        elif phase == "lowered":
            print(f"  (Tray lowered back to table at the translated position.)")

        input("  > ")

        left_joints = read_joints(left_robot)
        right_joints = read_joints(right_robot)

        # Also read degrees for display
        left_deg = read_joints_deg(left_robot)
        right_deg = read_joints_deg(right_robot)

        waypoints[phase] = {
            "left": left_joints,
            "right": right_joints,
        }

        print(f"  Recorded. Left (deg):  {_format_joints_deg(left_deg)}")
        print(f"            Right (deg): {_format_joints_deg(right_deg)}")
        print()

    # Also record the gripper open/closed raw values
    print("--- Gripper calibration ---")

    print("  Open both grippers fully, then press ENTER.")
    input("  > ")
    left_grip_open = read_joints(left_robot)["gripper"]
    right_grip_open = read_joints(right_robot)["gripper"]
    print(f"  Open: left={left_grip_open}, right={right_grip_open}")

    print("  Close both grippers onto the tray edge, then press ENTER.")
    input("  > ")
    left_grip_closed = read_joints(left_robot)["gripper"]
    right_grip_closed = read_joints(right_robot)["gripper"]
    print(f"  Closed: left={left_grip_closed}, right={right_grip_closed}")

    waypoints["_gripper"] = {
        "left_open": left_grip_open,
        "left_closed": left_grip_closed,
        "right_open": right_grip_open,
        "right_closed": right_grip_closed,
    }

    save_waypoints(waypoints)
    print("\nTeach mode complete. You can now run --collect.")


def _format_joints_deg(joints_deg: dict) -> str:
    """Pretty-print joint angles in degrees."""
    parts = [f"{name}={joints_deg.get(name, 0):+.1f}" for name in MOTOR_NAMES]
    return "  ".join(parts)


# ============================================================
# COLLECT MODE
# ============================================================

def collect(
    n_episodes: int,
    left_robot: SOFollower,
    right_robot: SOFollower,
    camera: RealSenseCamera | None,
    detector: ArucoDetector | None,
    dry_run: bool = False,
):
    """
    Execute the scripted task sequence and record data for N episodes.
    """
    waypoints = load_waypoints()

    # Extract gripper raw values
    grip = waypoints["_gripper"]
    left_grip_open = grip["left_open"]
    left_grip_closed = grip["left_closed"]
    right_grip_open = grip["right_open"]
    right_grip_closed = grip["right_closed"]

    # Find the next episode number
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(d.name.split("_")[1]) for d in OUTPUT_DIR.iterdir()
                if d.is_dir() and d.name.startswith("episode_")]
    next_ep = max(existing, default=-1) + 1

    print(f"\n=== COLLECT MODE: {n_episodes} episode(s) ===")
    if dry_run:
        print("  (dry run — no data will be saved)\n")

    for ep in range(n_episodes):
        ep_num = next_ep + ep
        ep_dir = OUTPUT_DIR / f"episode_{ep_num:03d}"
        print(f"\n--- Episode {ep_num} ---")

        # Set up recorder (or None for dry run)
        recorder = None
        if not dry_run and camera is not None:
            recorder = EpisodeRecorder(ep_dir, camera, left_robot, right_robot)
            recorder.setup()

        # Apply per-episode randomization to durations
        durations = {
            key: jitter_duration(dur)
            for key, dur in PHASE_DURATIONS.items()
        }

        success = True

        # ---- Phase 1: Move to home ----
        # Command both arms to the home waypoint to start from a known state.
        home_left = waypoints["home"]["left"]
        home_right = waypoints["home"]["right"]

        # Set grippers open at home
        home_left_full = dict(home_left)
        home_left_full["gripper"] = left_grip_open
        home_right_full = dict(home_right)
        home_right_full["gripper"] = right_grip_open

        print("  Moving to home...")
        current_left = read_joints(left_robot)
        current_right = read_joints(right_robot)
        execute_movement(
            left_robot, right_robot,
            current_left, home_left_full,
            current_right, home_right_full,
            duration=APPROACH_DURATION_S,
            recorder=recorder,
            phase_name="go_home",
        )
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # ---- Phase 2: Home -> Pre-grasp (grippers open) ----
        pre_left = dict(waypoints["pre_grasp"]["left"])
        pre_left["gripper"] = left_grip_open
        pre_right = dict(waypoints["pre_grasp"]["right"])
        pre_right["gripper"] = right_grip_open

        success = execute_movement(
            left_robot, right_robot,
            home_left_full, pre_left,
            home_right_full, pre_right,
            duration=durations[("home", "pre_grasp")],
            recorder=recorder,
            phase_name="approach",
        )
        if not success:
            continue
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # Optional: verify depth alignment at pre-grasp
        if detector is not None and camera is not None:
            frame = camera.get_frame()
            check_depth_alignment(detector, frame)

        # ---- Phase 3: Pre-grasp -> Grasp position (descend, close gap) ----
        grasp_left = dict(waypoints["grasp"]["left"])
        grasp_left["gripper"] = left_grip_open  # still open during descent
        grasp_right = dict(waypoints["grasp"]["right"])
        grasp_right["gripper"] = right_grip_open

        success = execute_movement(
            left_robot, right_robot,
            pre_left, grasp_left,
            pre_right, grasp_right,
            duration=durations[("pre_grasp", "grasp")],
            recorder=recorder,
            phase_name="descend",
        )
        if not success:
            continue
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # ---- Phase 4: Close grippers ----
        grasp_left_closed = dict(grasp_left)
        grasp_left_closed["gripper"] = left_grip_closed
        grasp_right_closed = dict(grasp_right)
        grasp_right_closed["gripper"] = right_grip_closed

        execute_movement(
            left_robot, right_robot,
            grasp_left, grasp_left_closed,
            grasp_right, grasp_right_closed,
            duration=GRASP_CLOSE_DURATION_S,
            recorder=recorder,
            phase_name="grasp_close",
        )
        pause(GRASP_SETTLE_S, recorder, left_robot, right_robot)

        # ---- Phase 5: Lift ----
        lift_left = dict(waypoints["lifted"]["left"])
        lift_left["gripper"] = left_grip_closed
        lift_right = dict(waypoints["lifted"]["right"])
        lift_right["gripper"] = right_grip_closed

        success = execute_movement(
            left_robot, right_robot,
            grasp_left_closed, lift_left,
            grasp_right_closed, lift_right,
            duration=durations[("grasp", "lifted")],
            recorder=recorder,
            detector=detector,
            camera=camera,
            phase_name="lift",
        )
        if not success:
            continue
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # ---- Phase 6: Translate ----
        trans_left = dict(waypoints["translated"]["left"])
        trans_left["gripper"] = left_grip_closed
        trans_right = dict(waypoints["translated"]["right"])
        trans_right["gripper"] = right_grip_closed

        success = execute_movement(
            left_robot, right_robot,
            lift_left, trans_left,
            lift_right, trans_right,
            duration=durations[("lifted", "translated")],
            recorder=recorder,
            detector=detector,
            camera=camera,
            phase_name="translate",
        )
        if not success:
            continue
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # ---- Phase 7: Lower ----
        lower_left = dict(waypoints["lowered"]["left"])
        lower_left["gripper"] = left_grip_closed
        lower_right = dict(waypoints["lowered"]["right"])
        lower_right["gripper"] = right_grip_closed

        success = execute_movement(
            left_robot, right_robot,
            trans_left, lower_left,
            trans_right, lower_right,
            duration=durations[("translated", "lowered")],
            recorder=recorder,
            phase_name="lower",
        )
        if not success:
            continue
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # ---- Phase 8: Release grippers ----
        lower_left_open = dict(lower_left)
        lower_left_open["gripper"] = left_grip_open
        lower_right_open = dict(lower_right)
        lower_right_open["gripper"] = right_grip_open

        execute_movement(
            left_robot, right_robot,
            lower_left, lower_left_open,
            lower_right, lower_right_open,
            duration=RELEASE_OPEN_DURATION_S,
            recorder=recorder,
            phase_name="release",
        )
        pause(INTER_PHASE_PAUSE_S, recorder, left_robot, right_robot)

        # ---- Phase 9: Retreat to release waypoint ----
        release_left = dict(waypoints["release"]["left"])
        release_left["gripper"] = left_grip_open
        release_right = dict(waypoints["release"]["right"])
        release_right["gripper"] = right_grip_open

        execute_movement(
            left_robot, right_robot,
            lower_left_open, release_left,
            lower_right_open, release_right,
            duration=durations[("lowered", "release")],
            recorder=recorder,
            phase_name="retreat",
        )

        # ---- Save episode ----
        if recorder is not None:
            recorder.save(metadata={
                "episode": ep_num,
                "success": success,
                "durations": {f"{a}->{b}": d for (a, b), d in durations.items()},
            })

        print(f"  Episode {ep_num} complete. {'SUCCESS' if success else 'FAILED'}")

        # Prompt user to reset the tray before the next episode
        if ep < n_episodes - 1:
            input("\n  Reposition the tray to the starting location, then press ENTER...")

    print(f"\nCollection finished. {n_episodes} episode(s) processed.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Centralized bimanual data collection for tray lifting."
    )
    parser.add_argument("--teach", action="store_true",
                        help="Interactively record waypoints by positioning arms.")
    parser.add_argument("--collect", type=int, metavar="N",
                        help="Collect N episodes using saved waypoints.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one episode without saving data.")

    args = parser.parse_args()

    if not args.teach and args.collect is None and not args.dry_run:
        parser.print_help()
        sys.exit(1)

    # Load port assignments
    ports = load_ports()

    # Connect both arms
    print("Connecting arms...")
    left_robot = connect_arm("left", ports["left"])
    right_robot = connect_arm("right", ports["right"])
    print(f"  Left arm on {ports['left']}, Right arm on {ports['right']}")

    try:
        if args.teach:
            teach(left_robot, right_robot)

        elif args.collect is not None or args.dry_run:
            n = args.collect if args.collect is not None else 1

            # Connect camera for recording and perception
            camera = None
            detector = None
            if not args.dry_run:
                print("Connecting camera...")
                camera = RealSenseCamera()
                camera.connect()
                print(f"  Camera connected (S/N: {camera.serial_number})")

                marker_config = MarkerConfig()
                detector = ArucoDetector(
                    config=marker_config,
                    camera_matrix=camera.get_camera_matrix(),
                    dist_coeffs=camera.get_dist_coeffs(),
                )

            try:
                collect(n, left_robot, right_robot, camera, detector,
                        dry_run=args.dry_run)
            finally:
                if camera is not None:
                    camera.disconnect()
                    print("Camera disconnected.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        for r in [left_robot, right_robot]:
            r.bus.sync_write("Torque_Enable", 0, normalize=False)
            r.config.disable_torque_on_disconnect = False
            r.disconnect()
        print("Arms disconnected.")


if __name__ == "__main__":
    main()
