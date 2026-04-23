"""
GNN/run_hybrid.py — run the bimanual tray-lift sequence with a modular,
swappable list of phase controllers.

Which phases are deterministic vs learned (GNN) is configured via the
USE_GNN_* flags and CKPT_* paths at the top of this file — flip a
flag to True, point its checkpoint at the right .pt, and run.

Usage:
    python GNN/run_hybrid.py
    python GNN/run_hybrid.py --max-step-deg 4 --smooth-window 5
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


# ==================================================================
# PHASE CONFIGURATION — flip a flag to True to use the learned (GNN)
# controller for that phase, False to keep the hand-coded
# deterministic version.
#
# Only phases with a registered learned controller in
# ``_LEARNED_FACTORIES`` (below) will actually be swapped. The rest
# fall back to deterministic with a warning on startup.
#
# CKPT_* paths are only read when the matching USE_GNN_* is True.
# ==================================================================

USE_GNN_HOVER          = False
USE_GNN_APPROACH       = False
USE_GNN_LIFTING        = False
USE_GNN_TRANSLATE      = True
USE_GNN_LOWER          = False
USE_GNN_RELEASE        = False

# HOME is intentionally not configurable — the final return-to-home
# motion is out of camera frame, so there is nothing a GNN could
# observe, and it stays deterministic forever.
#
# TRANSLATE_PREP is also not configurable: its motion is effectively a
# shorter prefix of TRANSLATE, so when USE_GNN_TRANSLATE is True we
# fold prep frames into the translate model at training time (both
# sources of data labeled "TRANSLATE"). The deterministic prep phase
# still runs on the hardware to bridge the lift-accumulated pose into
# a motion-map-compatible starting pose for the translate controller.

CKPT_HOVER          = str(REPO_ROOT / "GNN" / "checkpoints" / "hover.pt")
CKPT_APPROACH       = str(REPO_ROOT / "GNN" / "checkpoints" / "approach.pt")
CKPT_LIFTING        = str(REPO_ROOT / "GNN" / "checkpoints" / "lifting.pt")
CKPT_TRANSLATE      = str(REPO_ROOT / "GNN" / "checkpoints" / "translate.pt")
CKPT_LOWER          = str(REPO_ROOT / "GNN" / "checkpoints" / "lower.pt")
CKPT_RELEASE        = str(REPO_ROOT / "GNN" / "checkpoints" / "release.pt")

# Internal mapping: phase-name -> (use_gnn_flag, ckpt_path). Keep in
# sync with the constants above. Order doesn't matter here — the
# default phase list defines execution order.
_PHASE_CONFIG: dict[str, tuple[bool, str]] = {
    "HOVER":     (USE_GNN_HOVER,     CKPT_HOVER),
    "APPROACH":  (USE_GNN_APPROACH,  CKPT_APPROACH),
    "LIFTING":   (USE_GNN_LIFTING,   CKPT_LIFTING),
    "TRANSLATE": (USE_GNN_TRANSLATE, CKPT_TRANSLATE),
    "LOWER":     (USE_GNN_LOWER,     CKPT_LOWER),
    "RELEASE":   (USE_GNN_RELEASE,   CKPT_RELEASE),
}


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
_LEARNED_FACTORIES = {
    "HOVER":          lambda ckpt, cal_L, cal_R, ms, sw:
        lrn.LearnedHover.load(ckpt, cal_L, cal_R, ms, sw),
    "APPROACH":       lambda ckpt, cal_L, cal_R, ms, sw:
        lrn.LearnedApproach.load(ckpt, cal_L, cal_R, ms, sw),
    "LIFTING":        lambda ckpt, cal_L, cal_R, ms, sw:
        lrn.LearnedLift.load(ckpt, cal_L, cal_R, ms, sw),
    "TRANSLATE":      lambda ckpt, cal_L, cal_R, ms, sw:
        lrn.LearnedTranslate.load(ckpt, cal_L, cal_R, ms, sw),
    "LOWER":          lambda ckpt, cal_L, cal_R, ms, sw:
        lrn.LearnedLower.load(ckpt, cal_L, cal_R, ms, sw),
    "RELEASE":        lambda ckpt, cal_L, cal_R, ms, sw:
        lrn.LearnedRelease.load(ckpt, cal_L, cal_R, ms, sw),
}


def _build_phases(
    cal_left: dict,
    cal_right: dict,
    max_step_deg: float,
    smooth_window: int,
) -> list[PhaseController]:
    """Build one arm's phase list, honoring the USE_GNN_* config flags
    at the top of this file.

    For each default (deterministic) phase, if its USE_GNN_* flag is
    True AND a learned controller is registered in
    ``_LEARNED_FACTORIES``, the deterministic entry is replaced. If the
    flag is True but no learned controller exists yet (the common case
    for phases we haven't implemented), we warn once and keep the
    deterministic entry.
    """
    phases = det.default_phases()
    for i, p in enumerate(phases):
        use_gnn, ckpt = _PHASE_CONFIG.get(p.name, (False, ""))
        if not use_gnn:
            continue
        factory = _LEARNED_FACTORIES.get(p.name)
        if factory is None:
            print(
                f"  Warning: USE_GNN_{p.name} is True but no learned "
                f"controller is registered for this phase. Keeping "
                f"deterministic."
            )
            continue
        phases[i] = factory(ckpt, cal_left, cal_right, max_step_deg, smooth_window)
        print(f"  Swapped {p.name}: learned controller from {ckpt}")
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

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
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
            cal_left, cal_right,
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
