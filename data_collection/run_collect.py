"""
run_collect.py - Record data while executing the full run.py tray-lift sequence.

Re-uses run.py's per-arm state machines (HOME -> PRE_GRASP -> HOVER ->
APPROACH -> LIFT_WAITING -> LIFTING -> TRANSLATE -> LOWER -> RELEASE ->
HOME) and a background recorder thread samples synchronized data at
RECORD_FPS:

  episodes/episode_NNN/
    rgb/000000.png, ...     RGB frames (same format as centralized_collect)
    joints_left.csv         timestamp + 6 motors per row
    joints_right.csv        timestamp + 6 motors per row
    aruco.jsonl             per-frame ArUco marker poses (camera frame)
    metadata.json           episode metadata

joints_*.csv and metadata.json match centralized_collect.py --collect
exactly. aruco.jsonl is an extension specific to this recorder.

Usage:
  python data_collection/run_collect.py                   # 1 episode
  python data_collection/run_collect.py -n 5              # 5 episodes
  python data_collection/run_collect.py -n 1 --dry-run    # no saving
"""

import sys
import time
import json
import csv
import shutil
import threading
import argparse
from pathlib import Path

import numpy as np
import cv2

# Make the repo root importable so `import run` and friends resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run
from run import (
    ARM_MOTORS,
    RECORD_FPS,
    HOME_DURATION_S,
    PRE_GRASP_DURATION_S,
    RETURN_HOME_DURATION,
    BASELINE_READINGS,
    BASELINE_TIMEOUT_FRAMES,
    MOTION_MAP_DIR,
    FrameProvider,
    connect_arm,
    load_ports,
    load_waypoints,
    run_arm,
    smooth_move,
)

# ─── Bus serialization ──────────────────────────────────────
# The recorder samples joints at 30 Hz on the same Feetech bus the arm
# control threads read/write at 30 Hz. Without serialization both sides
# hit "[TxRxResult] Port is in use!" and the control thread crashes. We
# wrap run.read_joints_raw / run.write_joints_raw in a per-robot lock
# before any thread starts; run_arm resolves these names via module
# lookup and automatically picks up the wrapped versions.
_BUS_LOCKS: dict[int, threading.Lock] = {}
_ORIG_READ  = run.read_joints_raw
_ORIG_WRITE = run.write_joints_raw


def _locked_read(robot):
    lock = _BUS_LOCKS.get(id(robot))
    if lock is None:
        return _ORIG_READ(robot)
    with lock:
        return _ORIG_READ(robot)


def _locked_write(robot, target):
    lock = _BUS_LOCKS.get(id(robot))
    if lock is None:
        return _ORIG_WRITE(robot, target)
    with lock:
        return _ORIG_WRITE(robot, target)


run.read_joints_raw  = _locked_read
run.write_joints_raw = _locked_write

from lerobot.robots.so_follower import SOFollower
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from algorithmic.calibrate import load_base_transforms
from algorithmic.motion_map import MotionMap


OUTPUT_DIR = Path(__file__).parent / "episodes"

# Match centralized_collect.py so downstream code can treat the two
# collectors interchangeably.
MOTOR_NAMES = ARM_MOTORS + ["gripper"]


# ============================================================
# RECORDER
# ============================================================

class EpisodeRecorder:
    """Background thread that samples RGB + joints + ArUco at RECORD_FPS.

    Runs alongside run.py's per-arm control threads. Uses the main
    FrameProvider (started in _run_one_episode) to avoid contending with
    the RealSense driver. Joint reads go through the same motor buses as
    the control threads — the feetech sync_read call is atomic per
    packet, so concurrent reads from another thread return valid data.
    """

    def __init__(
        self,
        episode_dir: Path,
        detector: ArucoDetector,
        left_robot: SOFollower,
        right_robot: SOFollower,
    ):
        self.episode_dir = episode_dir
        self.rgb_dir = episode_dir / "rgb"
        self.detector = detector
        self.left_robot = left_robot
        self.right_robot = right_robot

        self.frame_provider: FrameProvider | None = None
        self.frame_index = 0
        self.start_time: float | None = None
        self._left_rows: list[list] = []
        self._right_rows: list[list] = []
        self._aruco_rows: list[dict] = []
        self._state_rows: list[list] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def setup(self):
        self.rgb_dir.mkdir(parents=True, exist_ok=True)

    def start(self, frame_provider: FrameProvider):
        if self.start_time is not None:
            raise RuntimeError("EpisodeRecorder already started")
        self.frame_provider = frame_provider
        self.start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self):
        interval = 1.0 / RECORD_FPS
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._sample()
            except Exception as e:
                print(f"  [recorder] sample error: {e}")
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _sample(self):
        ts = time.perf_counter() - self.start_time

        frame = self.frame_provider.get_frame() if self.frame_provider else None
        if frame is None:
            return  # camera not ready yet

        # Write PNG first; if anything below fails we already have the frame.
        frame_path = self.rgb_dir / f"{self.frame_index:06d}.png"
        cv2.imwrite(str(frame_path), frame)

        left_joints  = run.read_joints_raw(self.left_robot)
        right_joints = run.read_joints_raw(self.right_robot)

        self._left_rows.append(
            [ts] + [left_joints.get(m, 0.0) for m in MOTOR_NAMES]
        )
        self._right_rows.append(
            [ts] + [right_joints.get(m, 0.0) for m in MOTOR_NAMES]
        )

        poses = self.detector.detect(frame)
        self._aruco_rows.append({
            "frame": self.frame_index,
            "t": round(ts, 4),
            "markers": {str(mid): T.tolist() for mid, T in poses.items()},
        })

        # Per-frame phase labels straight from run.STATE_TRACKER — set
        # by main() during HOME/PRE_GRASP and by run.py's run_arm thread
        # for everything after PRE_GRASP. If STATE_TRACKER is None the
        # cell is left empty so downstream tools know phase info is
        # unavailable for this episode.
        tracker = run.STATE_TRACKER
        left_state  = (tracker or {}).get("left",  "")
        right_state = (tracker or {}).get("right", "")
        self._state_rows.append(
            [self.frame_index, round(ts, 4), left_state, right_state]
        )

        self.frame_index += 1

    def save(self, metadata: dict | None = None):
        header = ["timestamp"] + MOTOR_NAMES

        for fname, rows in [
            ("joints_left.csv",  self._left_rows),
            ("joints_right.csv", self._right_rows),
        ]:
            with open(self.episode_dir / fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

        with open(self.episode_dir / "aruco.jsonl", "w") as f:
            for entry in self._aruco_rows:
                f.write(json.dumps(entry) + "\n")

        with open(self.episode_dir / "states.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "timestamp", "left_state", "right_state"])
            w.writerows(self._state_rows)

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
# EPISODE EXECUTION (mirrors run.main())
# ============================================================

def _run_one_episode(
    robots: dict[str, SOFollower],
    camera: RealSenseCamera,
    detector: ArucoDetector,
    waypoints: dict,
    recorder: EpisodeRecorder | None,
) -> bool:
    """Execute one full tray-lift episode, optionally recording data.

    Mirrors run.main()'s flow so behavior is identical to run.py. Returns
    True if both arm threads completed normally.
    """
    sides = ["left", "right"]
    gripper_vals = waypoints["_gripper"]
    stop = threading.Event()
    threads: list[threading.Thread] = []
    provider: FrameProvider | None = None
    success = False

    try:
        # Base transforms + motion maps
        base_transforms = load_base_transforms()
        for side in sides:
            if side not in base_transforms:
                raise SystemExit(
                    f"No base transform for {side}. Run calibrate.py first."
                )
        T_base_cams = {
            side: np.linalg.inv(base_transforms[side]) for side in sides
        }

        motion_maps: dict[str, MotionMap] = {}
        for side in sides:
            mm_path = MOTION_MAP_DIR / f"motion_map_{side}.json"
            if not mm_path.exists():
                raise SystemExit(f"{mm_path} not found.")
            motion_maps[side] = MotionMap(mm_path)

        # Descent deltas (grasp - pre_grasp), zeroed for pan/roll
        descent_deltas: dict[str, dict[str, float]] = {}
        for side in sides:
            delta = {}
            for m in ARM_MOTORS:
                if m in ("shoulder_pan", "wrist_roll"):
                    delta[m] = 0.0
                else:
                    delta[m] = (waypoints["grasp"][side][m]
                                - waypoints["pre_grasp"][side][m])
            descent_deltas[side] = delta

        # Enable phase tracking for this episode. run.py's _set_phase
        # helper updates run.STATE_TRACKER whenever an arm thread enters
        # a new phase; we set the initial values for the centralized
        # HOME and PRE_GRASP moves here (both arms are in the same phase
        # during those moves).
        run.STATE_TRACKER = {"left": "HOME", "right": "HOME"}

        # HOME
        print("  -> HOME")
        smooth_move(
            robots["left"], robots["right"],
            run.read_joints_raw(robots["left"]),  waypoints["home"]["left"],
            run.read_joints_raw(robots["right"]), waypoints["home"]["right"],
            duration=HOME_DURATION_S,
        )
        time.sleep(0.5)

        # PRE_GRASP raised by one descent delta
        raised_pre_grasp = {}
        for side in sides:
            raised_pre_grasp[side] = dict(waypoints["pre_grasp"][side])
            for m in ARM_MOTORS:
                raised_pre_grasp[side][m] -= descent_deltas[side][m]

        run.STATE_TRACKER = {"left": "PRE_GRASP", "right": "PRE_GRASP"}
        print("  -> PRE_GRASP (raised)")
        smooth_move(
            robots["left"], robots["right"],
            waypoints["home"]["left"],  raised_pre_grasp["left"],
            waypoints["home"]["right"], raised_pre_grasp["right"],
            duration=PRE_GRASP_DURATION_S,
        )
        time.sleep(0.5)

        # Camera capture thread
        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.5)

        # Start recording now that frames are streaming. Everything from
        # here on — the baseline read, the HOVER->...-> HOME sequence —
        # gets captured.
        if recorder is not None:
            recorder.start(provider)

        # Baseline tray Y
        tray_left_ys, tray_right_ys = [], []
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
            raise SystemExit(
                f"Only {len(tray_left_ys)} tray tag readings — need 3+. "
                "Check that both tray tags are visible."
            )

        baseline_tray_y = {
            "tray_left":  float(np.mean(tray_left_ys)),
            "tray_right": float(np.mean(tray_right_ys)),
        }

        # Per-arm state machines
        print("\nLaunching arm state machines...")
        for side in sides:
            t = threading.Thread(
                target=run_arm,
                args=(
                    side, robots[side], provider.get_frame,
                    detector, motion_maps[side], T_base_cams[side],
                    descent_deltas[side],
                    gripper_vals[f"{side}_open"],
                    gripper_vals[f"{side}_closed"],
                    baseline_tray_y, waypoints, stop,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        while any(t.is_alive() for t in threads):
            time.sleep(0.2)

        success = not stop.is_set()

    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3.0)
        if recorder is not None:
            recorder.stop()
        if provider is not None:
            provider.stop()

    return success


# ============================================================
# MAIN
# ============================================================

def _prompt_keep(ep_num: int) -> bool:
    """Ask whether to save or discard the episode just recorded."""
    while True:
        resp = input(f"  Keep episode {ep_num}? [y/n]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Record the run.py tray-lift sequence for GNN training.",
    )
    parser.add_argument(
        "-n", "--episodes", type=int, default=1,
        help="Number of episodes to record.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the sequence without saving anything.",
    )
    args = parser.parse_args()

    sides = ["left", "right"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name.split("_")[1])
        for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    ]
    next_ep = max(existing, default=-1) + 1

    ports = load_ports()
    waypoints = load_waypoints()

    print("Connecting arms...")
    robots: dict[str, SOFollower] = {}
    for side in sides:
        robots[side] = connect_arm(side, ports[side])
        _BUS_LOCKS[id(robots[side])] = threading.Lock()
        print(f"  {side.capitalize()} on {ports[side]}")

    print("Connecting camera...")
    camera = RealSenseCamera()
    camera.connect()
    print(f"  Camera connected (S/N: {camera.serial_number})")

    detector = ArucoDetector(
        config=MarkerConfig(),
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )

    try:
        for ep_offset in range(args.episodes):
            ep_num = next_ep + ep_offset
            ep_dir = OUTPUT_DIR / f"episode_{ep_num:03d}"
            print(f"\n=== Episode {ep_num} ===")

            recorder = None
            if not args.dry_run:
                recorder = EpisodeRecorder(
                    ep_dir, detector, robots["left"], robots["right"],
                )
                recorder.setup()

            try:
                success = _run_one_episode(
                    robots, camera, detector, waypoints, recorder,
                )
            except KeyboardInterrupt:
                print("\n\nCtrl+C -- stopping this episode.")
                success = False

            print(f"  Episode {ep_num} complete. "
                  f"{'SUCCESS' if success else 'FAILED'}")

            if recorder is not None:
                if _prompt_keep(ep_num):
                    recorder.save(metadata={
                        "episode": ep_num,
                        "success": success,
                        "source": "run_collect.py",
                    })
                else:
                    shutil.rmtree(ep_dir, ignore_errors=True)
                    print(f"  Episode {ep_num} discarded.")

            # Prompt to reset tray before the next episode (to match
            # centralized_collect.py's interactive flow — the user varies
            # the tray pose between episodes for data diversity).
            if ep_offset < args.episodes - 1:
                input(
                    "\n  Reposition the tray for the next episode, "
                    "then press ENTER..."
                )

    except KeyboardInterrupt:
        print("\n\nCtrl+C -- stopping...")

    finally:
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
