"""
Motion-map hover controller.

Both SO-101 arms independently track their respective tray markers using
polynomial motion maps.  Each arm's motion map was recorded once and maps
(base_x, base_z) → joint angles at a fixed hover height.  The camera-to-
base transform (from base AprilTags) converts tray positions to each arm's
base frame.

When run directly, both arms continuously hover above their tray markers
even as the tray is slid around.  Press Ctrl+C to stop.

Prerequisites:
    1. calibrate.py  → base_transforms.json
    2. record_motion_map.py left/right  → motion_map_raw_{side}.csv
    3. build_motion_map.py left/right   → motion_map_{side}.json
"""

import sys
import time
import json
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from algorithmic.calibrate import load_base_transforms
from algorithmic.motion_map import MotionMap, motion_map_hover_loop


# ============================================================
# CONFIGURATION
# ============================================================

PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"
WAYPOINTS_FILE = Path(__file__).parent.parent / "data_collection" / "waypoints.json"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"
MOTION_MAP_DIR = Path(__file__).parent

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Smooth motion
APPROACH_DURATION_S = 4.0
STARTUP_DURATION_S = 4.0
RECORD_FPS = 30


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


# ============================================================
# WAYPOINTS AND SMOOTH MOTION
# ============================================================

def load_waypoints() -> dict:
    if not WAYPOINTS_FILE.exists():
        print(f"ERROR: {WAYPOINTS_FILE} not found.")
        sys.exit(1)
    with open(WAYPOINTS_FILE) as f:
        return json.load(f)


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


def smooth_move_single(
    robot: SOFollower,
    start: dict, end: dict,
    duration: float,
    label: str = "",
):
    """Smoothly interpolate one arm using minimum-jerk timing."""
    fps = RECORD_FPS
    interval = 1.0 / fps
    n_steps = max(1, int(duration * fps))

    if label:
        print(f"  [{label}] {duration:.1f}s, {n_steps} steps")

    for step in range(n_steps):
        t_start = time.perf_counter()
        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)
        write_joints_raw(robot, interpolate_joints(start, end, progress))
        elapsed = time.perf_counter() - t_start
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)


def smooth_move(
    left_robot: SOFollower,
    right_robot: SOFollower,
    left_start: dict, left_end: dict,
    right_start: dict, right_end: dict,
    duration: float,
    label: str = "",
):
    """Smoothly interpolate both arms using minimum-jerk timing."""
    fps = RECORD_FPS
    interval = 1.0 / fps
    n_steps = max(1, int(duration * fps))

    if label:
        print(f"  [{label}] {duration:.1f}s, {n_steps} steps")

    for step in range(n_steps):
        t_start = time.perf_counter()
        t = (step + 1) / n_steps * duration
        progress = minimum_jerk(t, duration)
        write_joints_raw(left_robot, interpolate_joints(left_start, left_end, progress))
        write_joints_raw(right_robot, interpolate_joints(right_start, right_end, progress))
        elapsed = time.perf_counter() - t_start
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
            f = self.camera.get_frame()
            with self._lock:
                self._frame = f


# ============================================================
# MAIN
# ============================================================

def main():
    # ── Parse optional arm argument ─────────────────────────
    sides = ["left", "right"]
    if len(sys.argv) > 1:
        if sys.argv[1] in ("left", "right"):
            sides = [sys.argv[1]]
        else:
            print("Usage: python hover.py [left | right]")
            print("  No argument = both arms")
            sys.exit(1)

    ports = load_ports()
    robots: dict[str, SOFollower] = {}

    print("Connecting arms...")
    for side in sides:
        robots[side] = connect_arm(side, ports[side])
        print(f"  {side.capitalize()} on {ports[side]}")

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
        # ── Load base transforms ────────────────────────────
        print("\nLoading base transforms...")
        base_transforms = load_base_transforms()
        for side in sides:
            if side not in base_transforms:
                print(f"  ERROR: No transform for {side}. Run calibrate.py first.")
                raise SystemExit(1)
        print("  Loaded.")

        T_base_cams = {side: np.linalg.inv(base_transforms[side]) for side in sides}

        # ── Load motion maps ────────────────────────────────
        print("Loading motion maps...")
        motion_maps: dict[str, MotionMap] = {}
        for side in sides:
            mm_path = MOTION_MAP_DIR / f"motion_map_{side}.json"
            if not mm_path.exists():
                print(f"  ERROR: {mm_path} not found.")
                print(f"         Run record_motion_map.py + build_motion_map.py first.")
                raise SystemExit(1)
            motion_maps[side] = MotionMap(mm_path)
            mm = motion_maps[side]
            print(f"  {side.capitalize()}: {mm.metadata['n_training_points']} training points, "
                  f"degree {mm.metadata['poly_degree']}")

        # ── Move to pre_grasp via waypoints ─────────────────
        waypoints = load_waypoints()

        if len(sides) == 2:
            print("\nMoving to home position...")
            smooth_move(
                robots["left"], robots["right"],
                read_joints_raw(robots["left"]), waypoints["home"]["left"],
                read_joints_raw(robots["right"]), waypoints["home"]["right"],
                duration=STARTUP_DURATION_S, label="home",
            )
            time.sleep(0.5)

            print("Moving to pre-grasp position...")
            smooth_move(
                robots["left"], robots["right"],
                waypoints["home"]["left"], waypoints["pre_grasp"]["left"],
                waypoints["home"]["right"], waypoints["pre_grasp"]["right"],
                duration=APPROACH_DURATION_S, label="pre_grasp",
            )
            time.sleep(0.5)
        else:
            side = sides[0]
            print("\nMoving to home position...")
            smooth_move_single(
                robots[side],
                read_joints_raw(robots[side]), waypoints["home"][side],
                duration=STARTUP_DURATION_S, label="home",
            )
            time.sleep(0.5)

            print("Moving to pre-grasp position...")
            smooth_move_single(
                robots[side],
                waypoints["home"][side], waypoints["pre_grasp"][side],
                duration=APPROACH_DURATION_S, label="pre_grasp",
            )
            time.sleep(0.5)

        # ── Start frame provider and hover loops ────────────
        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.3)

        threads = []
        for side in sides:
            t = threading.Thread(
                target=motion_map_hover_loop,
                args=(robots[side], provider.get_frame, detector, side,
                      motion_maps[side], T_base_cams[side]),
                kwargs={"stop_event": stop},
                daemon=True,
            )
            t.start()
            threads.append(t)

        label = sides[0] if len(sides) == 1 else "both arms"
        print(f"Hover tracking active ({label}). Slide the tray around. Ctrl+C to stop.\n")

        while not stop.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop.set()
        time.sleep(0.5)
        if provider is not None:
            provider.stop()

        # ── Return to home ──────────────────────────────────
        print("Returning to home...")
        if len(sides) == 2:
            smooth_move(
                robots["left"], robots["right"],
                read_joints_raw(robots["left"]), waypoints["home"]["left"],
                read_joints_raw(robots["right"]), waypoints["home"]["right"],
                duration=STARTUP_DURATION_S, label="home",
            )
        else:
            side = sides[0]
            smooth_move_single(
                robots[side],
                read_joints_raw(robots[side]), waypoints["home"][side],
                duration=STARTUP_DURATION_S, label="home",
            )
        time.sleep(0.3)

        for r in robots.values():
            r.bus.sync_write("Torque_Enable", 0, normalize=False)
            r.config.disable_torque_on_disconnect = False
            r.disconnect()
        camera.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
