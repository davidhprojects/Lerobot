"""
GNN/run_policy.py — deploy the trained tray-lift GNN on real hardware.

Replaces run.py's state machine with a single GNN call at each control
tick. Each arm runs its own thread, its own SceneObserver in its own
base frame, and its own decoder head. Coordination is fully emergent:
the two threads only share the camera feed and the latest joint
snapshots — no phase flags, no planner.

  per-arm loop (30 Hz)
      frame  -> ArUco detect
      own joints -> shared snapshot
      SceneObserver.observe(aruco, left_joints, right_joints)
      build_graph(ents)
      model.forward_side(x, edge_attr, side)
          -> velocity [5]  (normalized -> raw/sec via checkpoint stats)
          -> gripper_logit -> binary open/closed
      per-step delta = velocity / fps, clamped to --max-step-deg
      write_joints_raw(own_robot, target)

Safety:
  * Per-tick step is clamped to --max-step-deg (default 4°, well under
    MAX_JOINT_STEP_RAW).
  * On exit (Ctrl+C, duration hit, or thread crash) both arms are moved
    back to the HOME waypoint with grippers open, then torque is
    released.
  * Missing tray markers = hold position (no command issued that tick).

Usage:
  python GNN/run_policy.py
  python GNN/run_policy.py --checkpoint GNN/checkpoints/best.pt --duration 45
  python GNN/run_policy.py --max-step-deg 3 --fps 30 --skip-home
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run
from run import (
    ARM_MOTORS,
    HOME_DURATION_S,
    PRE_GRASP_DURATION_S,
    RETURN_HOME_DURATION,
    FrameProvider,
    connect_arm,
    load_ports,
    load_waypoints,
    smooth_move,
)
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from algorithmic.calibrate import load_base_transforms
from GNN.entities import LEFT_EE, RIGHT_EE
from GNN.graph_builder import build_graph
from GNN.policy import TrayLiftGNN, ARM_JOINTS
from GNN.scene_observer import SceneObserver


REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = REPO_ROOT / "calibrations"
CKPT_PATH = REPO_ROOT / "GNN" / "checkpoints" / "best.pt"

FK_MOTORS = ARM_MOTORS                   # 5 arm joints, gripper handled separately
MOTOR_RESOLUTION = 4096                  # STS3215 12-bit


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_calibration(side: str) -> dict:
    with open(CALIB_DIR / f"{side}.json") as f:
        return json.load(f)


def _raw_to_rad(raw: float, cal_entry: dict) -> float:
    mid = 0.5 * (cal_entry["range_min"] + cal_entry["range_max"])
    deg = (raw - mid) * 360.0 / MOTOR_RESOLUTION
    return float(np.deg2rad(deg))


def _joints_to_rad_array(joints_raw: dict, cal: dict) -> np.ndarray:
    return np.array(
        [_raw_to_rad(joints_raw[m], cal[m]) for m in FK_MOTORS],
        dtype=np.float64,
    )


def _load_model(ckpt_path: Path) -> tuple[TrayLiftGNN, dict[str, torch.Tensor]]:
    """Load checkpoint and return (model, per-arm normalization stats).

    Supports both the newer per-arm checkpoint format
    (``mean_left``/``std_left``/``mean_right``/``std_right``) and the
    older single-stat format for backward compatibility.
    """
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}. Train first with "
            f"`python GNN/train.py`."
        )
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    train_args = ckpt.get("args", {})
    hidden = int(train_args.get("hidden", 64))
    rounds = int(train_args.get("rounds", 3))

    model = TrayLiftGNN(hidden=hidden, n_rounds=rounds)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = ckpt["normalization"]
    if "mean_left" in n:
        norm = {
            "mean_left":  torch.tensor(n["mean_left"],  dtype=torch.float32),
            "std_left":   torch.tensor(n["std_left"],   dtype=torch.float32),
            "mean_right": torch.tensor(n["mean_right"], dtype=torch.float32),
            "std_right":  torch.tensor(n["std_right"],  dtype=torch.float32),
        }
    else:
        m = torch.tensor(n["mean"], dtype=torch.float32)
        s = torch.tensor(n["std"],  dtype=torch.float32)
        norm = {"mean_left": m, "std_left": s, "mean_right": m, "std_right": s}
    return model, norm


# ------------------------------------------------------------------
# Shared joint snapshot (enables each arm to see both sides' joints
# without cross-bus contention)
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
# Per-arm policy loop
# ------------------------------------------------------------------

def run_arm_policy(
    side: str,
    robot,
    detector: ArucoDetector,
    get_frame,
    model: TrayLiftGNN,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    observer: SceneObserver,
    cal_left: dict,
    cal_right: dict,
    gripper_open_val: float,
    gripper_closed_val: float,
    max_step_raw: float,
    snapshot: JointSnapshot,
    stop_event: threading.Event,
    fps: float,
    smooth_window: int = 1,
):
    interval = 1.0 / fps
    n_steps = 0
    n_marker_loss = 0
    n_other_wait = 0
    last_log = time.perf_counter()

    # Rolling buffer of recent per-tick velocity predictions (raw/sec).
    # Commanded velocity is the uniform mean over the window. Smoothing
    # helps absorb the per-frame noise visible in the diagnostic
    # (prediction std ~60-80% of target std on some motors) without
    # adding significant latency when window is small.
    vel_history: list[np.ndarray] = []

    while not stop_event.is_set():
        t0 = time.perf_counter()

        frame = get_frame()
        if frame is None:
            time.sleep(interval)
            continue

        # Detect markers on this tick.
        aruco = detector.detect(frame)

        # Read own joints and publish to the shared snapshot so the
        # other arm's graph construction sees them.
        current = run.read_joints_raw(robot)
        snapshot.update(side, current)

        joints_L, joints_R = snapshot.get_both()
        if joints_L is None or joints_R is None:
            n_other_wait += 1
            time.sleep(interval)
            continue

        q_L = _joints_to_rad_array(joints_L, cal_left)
        q_R = _joints_to_rad_array(joints_R, cal_right)

        ents = observer.observe(aruco, q_L, q_R)
        if ents is None:
            n_marker_loss += 1
            # Hold position — no command this tick.
            elapsed = time.perf_counter() - t0
            if interval - elapsed > 0:
                time.sleep(interval - elapsed)
            continue

        graph = build_graph(ents)

        with torch.no_grad():
            x = graph.x.unsqueeze(0)
            e = graph.edge_attr.unsqueeze(0)
            out = model.forward_side(x, e, side)

        vel_norm = out["velocity"][0]
        vel_raw_per_sec = (vel_norm * norm_std + norm_mean).numpy()
        grip_closed = out["gripper_logit"].item() > 0.0

        # Temporal smoothing: mean of the last `smooth_window` predictions.
        # Absorbs single-frame noise (pred std was ~60-80% of target std
        # on some motors) without adding significant latency.
        vel_history.append(vel_raw_per_sec)
        if len(vel_history) > smooth_window:
            vel_history.pop(0)
        vel_smoothed = np.mean(np.stack(vel_history, axis=0), axis=0)

        # Per-step delta, clamped uniformly to preserve direction.
        step = vel_smoothed / fps
        max_abs = float(np.abs(step).max())
        if max_abs > max_step_raw:
            step = step * (max_step_raw / max_abs)

        target = {}
        for i, motor in enumerate(FK_MOTORS):
            target[motor] = float(current[motor]) + float(step[i])
        target["gripper"] = float(
            gripper_closed_val if grip_closed else gripper_open_val
        )

        run.write_joints_raw(robot, target)
        n_steps += 1

        now = time.perf_counter()
        if now - last_log >= 2.0:
            last_log = now
            vel_str = ",".join(f"{v:+5.0f}" for v in vel_smoothed)
            print(
                f"  [{side}] step={n_steps:4d}  "
                f"marker_loss={n_marker_loss}  "
                f"grip={'C' if grip_closed else 'O'}  "
                f"vel=[{vel_str}]"
            )

        elapsed = time.perf_counter() - t0
        if interval - elapsed > 0:
            time.sleep(interval - elapsed)

    print(
        f"  [{side}] policy stopped — {n_steps} steps, "
        f"{n_marker_loss} marker-loss holds, {n_other_wait} startup waits."
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", default=str(CKPT_PATH))
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Max runtime in seconds before auto-stop.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-step-deg", type=float, default=4.0,
                        help="Per-tick joint-velocity clamp (degrees). Start "
                             "conservative until you trust the policy.")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Number of recent velocity predictions to average "
                             "before commanding motors (1 = no smoothing).")
    parser.add_argument("--skip-init", action="store_true",
                        help="Skip the initial HOME -> PRE_GRASP moves "
                             "(only use if arms are already at raised "
                             "pre-grasp).")
    args = parser.parse_args()

    max_step_raw = args.max_step_deg * (MOTOR_RESOLUTION / 360.0)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm = _load_model(Path(args.checkpoint))
    print(f"  model loaded, {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  left  mean/std: {norm['mean_left'].tolist()} / {norm['std_left'].tolist()}")
    print(f"  right mean/std: {norm['mean_right'].tolist()} / {norm['std_right'].tolist()}")

    sides = ["left", "right"]
    ports = load_ports()
    waypoints = load_waypoints()

    print("Connecting arms...")
    robots: dict = {}
    for side in sides:
        robots[side] = connect_arm(side, ports[side])
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

    cal_left  = _load_calibration("left")
    cal_right = _load_calibration("right")
    base_transforms = load_base_transforms()

    observers = {
        s: SceneObserver(s, base_transforms, dt=1.0 / args.fps)
        for s in sides
    }

    grip = waypoints["_gripper"]
    grip_open   = {s: grip[f"{s}_open"]   for s in sides}
    grip_closed = {s: grip[f"{s}_closed"] for s in sides}

    provider: FrameProvider | None = None
    stop = threading.Event()
    threads: list[threading.Thread] = []

    try:
        # Gripper markers aren't visible from HOME (arms curled up), so
        # match run_collect.py's init exactly: HOME -> raised PRE_GRASP,
        # where "raised" = pre-grasp minus one descent delta so the
        # grippers hover above the tray with their markers facing the
        # camera. Training data began from this pose, so the GNN needs
        # to start here too.
        if not args.skip_init:
            print("  -> HOME")
            smooth_move(
                robots["left"], robots["right"],
                run.read_joints_raw(robots["left"]),  waypoints["home"]["left"],
                run.read_joints_raw(robots["right"]), waypoints["home"]["right"],
                duration=HOME_DURATION_S,
            )
            time.sleep(0.5)

            # Descent delta: how far grasp is below pre-grasp. Subtracting
            # this from pre-grasp raises the arm by the same amount.
            # shoulder_pan and wrist_roll are excluded (direction-invariant).
            raised_pre_grasp = {}
            for side in sides:
                raised_pre_grasp[side] = dict(waypoints["pre_grasp"][side])
                for m in ARM_MOTORS:
                    if m in ("shoulder_pan", "wrist_roll"):
                        continue
                    delta = (waypoints["grasp"][side][m]
                             - waypoints["pre_grasp"][side][m])
                    raised_pre_grasp[side][m] -= delta

            print("  -> PRE_GRASP (raised)")
            smooth_move(
                robots["left"], robots["right"],
                waypoints["home"]["left"],  raised_pre_grasp["left"],
                waypoints["home"]["right"], raised_pre_grasp["right"],
                duration=PRE_GRASP_DURATION_S,
            )
            time.sleep(0.5)

        provider = FrameProvider(camera)
        provider.start()
        time.sleep(0.5)

        snapshot = JointSnapshot()

        print(f"\nLaunching GNN policy (max_step={args.max_step_deg:.1f}°, "
              f"fps={args.fps:.0f}, duration={args.duration:.0f}s)")
        for side in sides:
            t = threading.Thread(
                target=run_arm_policy,
                kwargs={
                    "side": side,
                    "robot": robots[side],
                    "detector": detector,
                    "get_frame": provider.get_frame,
                    "model": model,
                    "norm_mean": norm[f"mean_{side}"],
                    "norm_std":  norm[f"std_{side}"],
                    "observer": observers[side],
                    "cal_left": cal_left,
                    "cal_right": cal_right,
                    "gripper_open_val": grip_open[side],
                    "gripper_closed_val": grip_closed[side],
                    "max_step_raw": max_step_raw,
                    "snapshot": snapshot,
                    "stop_event": stop,
                    "fps": args.fps,
                    "smooth_window": max(1, args.smooth_window),
                },
                daemon=True,
            )
            t.start()
            threads.append(t)

        t_end = time.time() + args.duration
        while time.time() < t_end and any(t.is_alive() for t in threads):
            time.sleep(0.2)
        print(f"\nDuration elapsed ({args.duration:.0f}s) — stopping.")

    except KeyboardInterrupt:
        print("\n\nCtrl+C -- stopping...")

    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3.0)
        if provider is not None:
            provider.stop()

        # Safety: open grippers and return to HOME.
        try:
            print("Opening grippers and returning to HOME...")
            for side in sides:
                run.write_joints_raw(
                    robots[side], {"gripper": grip_open[side]},
                )
            time.sleep(0.5)
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
                robots[side].bus.sync_write("Torque_Enable", 0, normalize=False)
                robots[side].config.disable_torque_on_disconnect = False
                robots[side].disconnect()
            except Exception:
                pass
        camera.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
