"""
GNN/run_hybrid.py — run the bimanual tray-lift sequence with a modular,
swappable list of phase controllers.

The default phase list is entirely deterministic and replicates
run.py's behavior. Swap any entry for a learned (GNN) controller to
hand that phase off to a trained model:

    from GNN.phases import deterministic as det, learned as lrn
    phases_left  = det.default_phases()
    phases_right = det.default_phases()
    phases_left[2]  = lrn.LearnedLift.load(
        "GNN/checkpoints/lift.pt", cal_left, cal_right,
    )
    phases_right[2] = lrn.LearnedLift.load(
        "GNN/checkpoints/lift.pt", cal_left, cal_right,
    )

See ``--swap LIFTING=GNN/checkpoints/lift.pt`` for the CLI version.

Usage (all-deterministic, sanity-check baseline):
    python GNN/run_hybrid.py

Usage (swap LIFTING for a GNN):
    python GNN/run_hybrid.py --swap LIFTING=GNN/checkpoints/lift.pt \\
                             --max-step-deg 4 --smooth-window 5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run
from run import (
    ARM_MOTORS,
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
    smooth_move,
)
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from algorithmic.calibrate import load_base_transforms
from algorithmic.motion_map import MotionMap

from GNN.phases import deterministic as det
from GNN.phases import learned as lrn
from GNN.phases.base import PhaseContext, PhaseController
from GNN.scene_observer import SceneObserver


REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = REPO_ROOT / "calibrations"


# ------------------------------------------------------------------
# Bus-lock monkey-patch (same pattern as run_collect.py)
# ------------------------------------------------------------------

_BUS_LOCKS: dict[int, threading.Lock] = {}
_ORIG_READ = run.read_joints_raw
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


run.read_joints_raw = _locked_read
run.write_joints_raw = _locked_write


# ------------------------------------------------------------------
# Shared joint snapshot (used by learned phases that need both arms'
# joint states without cross-bus contention)
# ------------------------------------------------------------------

class JointSnapshot:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def update(self, side: str, joints: dict):
        with self._lock:
            self._data[side] = dict(joints)

    def get_both(self) -> tuple[dict | None, dict | None]:
        with self._lock:
            return self._data.get("left"), self._data.get("right")


# ------------------------------------------------------------------
# Calibration loader (used by LearnedLift.load)
# ------------------------------------------------------------------

def _load_calibration(side: str) -> dict:
    with open(CALIB_DIR / f"{side}.json") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Phase composition helpers
# ------------------------------------------------------------------

# Maps phase name -> factory that returns a learned controller.
# Extend as more learned phases are implemented.
_LEARNED_FACTORIES = {
    "LIFTING": lambda ckpt, cal_L, cal_R, max_step_deg, smooth_window:
        lrn.LearnedLift.load(ckpt, cal_L, cal_R, max_step_deg, smooth_window),
}


def _build_phases(
    cal_left: dict,
    cal_right: dict,
    swaps: list[tuple[str, str]],
    max_step_deg: float,
    smooth_window: int,
) -> list[PhaseController]:
    """Build one arm's phase list, applying any --swap overrides.

    ``swaps`` is a list of (phase_name, ckpt_path). Each phase_name must
    match one of the classes in deterministic.default_phases(); the
    matching controller is replaced by its learned counterpart.
    """
    phases = det.default_phases()
    if not swaps:
        return phases

    by_name = {p.name: i for i, p in enumerate(phases)}
    for phase_name, ckpt in swaps:
        if phase_name not in _LEARNED_FACTORIES:
            raise SystemExit(
                f"--swap {phase_name}=...: no learned controller registered "
                f"for {phase_name}. Available: {sorted(_LEARNED_FACTORIES)}"
            )
        if phase_name not in by_name:
            raise SystemExit(
                f"--swap {phase_name}=...: no deterministic phase with that "
                f"name in the default list. Names: {sorted(by_name)}"
            )
        factory = _LEARNED_FACTORIES[phase_name]
        phases[by_name[phase_name]] = factory(
            ckpt, cal_left, cal_right, max_step_deg, smooth_window,
        )
        print(f"  Swapped {phase_name}: learned controller from {ckpt}")
    return phases


# ------------------------------------------------------------------
# Per-arm runner: iterate the phase list on a shared PhaseContext
# ------------------------------------------------------------------

def _run_arm(ctx: PhaseContext, phases: list[PhaseController]):
    for phase in phases:
        if ctx.stop_event.is_set():
            return
        result = phase.run(ctx)
        ctx.carryover.update(result.carryover)
        if not result.success:
            return


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _parse_swap(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--swap expects PHASE=path/to/ckpt.pt, got: {s!r}"
        )
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--swap", action="append", type=_parse_swap, default=[],
        metavar="PHASE=ckpt.pt",
        help="Replace a deterministic phase with a learned controller. "
             "May be passed multiple times.",
    )
    parser.add_argument("--max-step-deg", type=float, default=4.0,
                        help="Per-tick joint step cap for learned phases.")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Velocity-smoothing window for learned phases.")
    parser.add_argument("--skip-init", action="store_true",
                        help="Skip initial HOME + PRE_GRASP moves.")
    args = parser.parse_args()

    sides = ["left", "right"]
    ports = load_ports()
    waypoints = load_waypoints()

    # --- hardware setup (same as run_collect.py) ---
    print("Connecting arms...")
    robots = {}
    for side in sides:
        robots[side] = connect_arm(side, ports[side])
        _BUS_LOCKS[id(robots[side])] = threading.Lock()
        print(f"  {side.capitalize()} on {ports[side]}")

    print("Connecting camera...")
    camera = RealSenseCamera()
    camera.connect()
    detector = ArucoDetector(
        config=MarkerConfig(),
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )
    print(f"  Camera connected (S/N: {camera.serial_number})")

    cal_left = _load_calibration("left")
    cal_right = _load_calibration("right")
    base_transforms = load_base_transforms()
    T_base_cams = {s: np.linalg.inv(base_transforms[s]) for s in sides}

    # --- motion maps ---
    motion_maps: dict[str, MotionMap] = {}
    for side in sides:
        mm_path = MOTION_MAP_DIR / f"motion_map_{side}.json"
        if not mm_path.exists():
            raise SystemExit(f"{mm_path} not found.")
        motion_maps[side] = MotionMap(mm_path)

    # --- descent deltas (matches run.py) ---
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

    gripper_vals = waypoints["_gripper"]

    # --- scene observers (only used by learned phases; cheap to always build) ---
    observers = {
        s: SceneObserver(s, base_transforms, dt=1.0 / 30.0)
        for s in sides
    }
    snapshot = JointSnapshot()

    # --- build per-arm phase lists ---
    phases_by_side = {
        s: _build_phases(
            cal_left, cal_right, args.swap,
            args.max_step_deg, args.smooth_window,
        )
        for s in sides
    }
    print("Phases (left):  " + " -> ".join(
        f"{type(p).__name__}:{p.name}" for p in phases_by_side["left"]
    ))
    print("Phases (right): " + " -> ".join(
        f"{type(p).__name__}:{p.name}" for p in phases_by_side["right"]
    ))

    stop = threading.Event()
    threads: list[threading.Thread] = []
    provider: FrameProvider | None = None
    run.STATE_TRACKER = {"left": "INIT", "right": "INIT"}

    try:
        # --- Initial HOME + raised PRE_GRASP (centralized, same as run.py) ---
        if not args.skip_init:
            run.STATE_TRACKER = {"left": "HOME", "right": "HOME"}
            print("  -> HOME")
            smooth_move(
                robots["left"], robots["right"],
                run.read_joints_raw(robots["left"]),  waypoints["home"]["left"],
                run.read_joints_raw(robots["right"]), waypoints["home"]["right"],
                duration=HOME_DURATION_S,
            )
            time.sleep(0.5)

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

        # --- camera stream ---
        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.5)

        # --- baseline tray Y (same as run.py) ---
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
                f"Only {len(tray_left_ys)} tray readings — need 3. "
                "Check that both tray tags are visible."
            )
        baseline_tray_y = {
            "tray_left":  float(np.mean(tray_left_ys)),
            "tray_right": float(np.mean(tray_right_ys)),
        }

        # --- per-arm threads ---
        print("\nLaunching phase runners...")
        for side in sides:
            ctx = PhaseContext(
                side=side,
                robot=robots[side],
                get_frame=provider.get_frame,
                detector=detector,
                motion_map=motion_maps[side],
                T_base_cam=T_base_cams[side],
                descent_delta=descent_deltas[side],
                gripper_open=gripper_vals[f"{side}_open"],
                gripper_closed=gripper_vals[f"{side}_closed"],
                baseline_tray_y=baseline_tray_y,
                waypoints=waypoints,
                stop_event=stop,
                observer=observers[side],
                joint_snapshot=snapshot,
            )
            t = threading.Thread(
                target=_run_arm,
                args=(ctx, phases_by_side[side]),
                daemon=True,
            )
            t.start()
            threads.append(t)

        while any(t.is_alive() for t in threads):
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n\nCtrl+C -- stopping...")

    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3.0)
        if provider is not None:
            provider.stop()

        # Safety cleanup: release grippers and return HOME if a phase crashed.
        try:
            for side in sides:
                run.write_joints_raw(
                    robots[side], {"gripper": gripper_vals[f"{side}_open"]},
                )
            time.sleep(0.3)
            smooth_move(
                robots["left"], robots["right"],
                run.read_joints_raw(robots["left"]),  waypoints["home"]["left"],
                run.read_joints_raw(robots["right"]), waypoints["home"]["right"],
                duration=RETURN_HOME_DURATION,
            )
        except Exception as e:
            print(f"  Warning: cleanup move failed: {e}")

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
