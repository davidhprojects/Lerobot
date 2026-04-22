"""
Learned phase controllers — GNN-backed drop-in replacements for any
deterministic phase.

Each controller runs a model that was trained on just that phase's
frames (use ``train.py --phase LIFTING`` etc.). The controller's
termination condition stays hand-coded (same as the deterministic
equivalent): the model only decides actions, not when to transition.

This module currently ships one learned phase as an example. Copy its
pattern to add more. To swap into a hybrid:

    import GNN.phases.deterministic as det
    import GNN.phases.learned as lrn
    phases = [
        det.Hover(), det.Approach(),
        lrn.LearnedLift.load("GNN/checkpoints/lift.pt"),   # <-- swapped
        det.TranslatePrep(), det.Translate(),
        det.Lower(), det.Release(), det.Home(),
    ]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import run
from run import ARM_MOTORS, LIFT_STEP_INTERVAL_S, LIFT_HEIGHT_M

from GNN.phases.base import PhaseController, PhaseContext, PhaseResult
from GNN.policy import TrayLiftGNN, ARM_JOINTS
from GNN.graph_builder import build_graph


MOTOR_RESOLUTION = 4096
FK_MOTORS = ARM_MOTORS          # 5 arm joints (gripper handled separately)


def _raw_to_rad(raw: float, cal_entry: dict) -> float:
    mid = 0.5 * (cal_entry["range_min"] + cal_entry["range_max"])
    deg = (raw - mid) * 360.0 / MOTOR_RESOLUTION
    return float(np.deg2rad(deg))


def _joints_to_rad_array(joints_raw: dict, cal: dict) -> np.ndarray:
    return np.array(
        [_raw_to_rad(joints_raw[m], cal[m]) for m in FK_MOTORS],
        dtype=np.float64,
    )


class LearnedLift(PhaseController):
    """GNN-driven bimanual lift.

    Uses the same tray-rise termination condition as ``deterministic.Lift``
    (my_rise >= LIFT_HEIGHT_M, then wait for other side) so the phase
    still hands off cleanly to TRANSLATE_PREP. Only the inner motor
    command is the GNN's decision.

    The model is expected to have been trained on frames labeled
    LIFTING / LIFT_WAITING. Instantiate via ``LearnedLift.load(...)``.
    """

    name = "LIFTING"

    def __init__(
        self,
        model: TrayLiftGNN,
        norm: dict[str, torch.Tensor],
        cal_left: dict,
        cal_right: dict,
        max_step_raw: float,
        smooth_window: int = 5,
    ):
        self.model = model
        self.norm = norm
        self.cal_left = cal_left
        self.cal_right = cal_right
        self.max_step_raw = max_step_raw
        self.smooth_window = max(1, smooth_window)

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path,
        cal_left: dict,
        cal_right: dict,
        max_step_deg: float = 4.0,
        smooth_window: int = 5,
    ) -> "LearnedLift":
        ckpt = torch.load(Path(ckpt_path), weights_only=False, map_location="cpu")
        args = ckpt.get("args", {})
        model = TrayLiftGNN(
            hidden=int(args.get("hidden", 64)),
            n_rounds=int(args.get("rounds", 3)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        n = ckpt["normalization"]
        norm = {
            "mean_left":  torch.tensor(n["mean_left"],  dtype=torch.float32),
            "std_left":   torch.tensor(n["std_left"],   dtype=torch.float32),
            "mean_right": torch.tensor(n["mean_right"], dtype=torch.float32),
            "std_right":  torch.tensor(n["std_right"],  dtype=torch.float32),
        }
        max_step_raw = max_step_deg * (MOTOR_RESOLUTION / 360.0)
        return cls(model, norm, cal_left, cal_right, max_step_raw, smooth_window)

    def run(self, ctx: PhaseContext) -> PhaseResult:
        if ctx.observer is None or ctx.joint_snapshot is None:
            raise RuntimeError(
                "LearnedLift requires ctx.observer and ctx.joint_snapshot; "
                "run_hybrid.py should have attached them before dispatch."
            )
        run._set_phase(ctx.side, "LIFTING")

        fps = 1.0 / LIFT_STEP_INTERVAL_S
        my_key = f"tray_{ctx.side}"
        other_key = f"tray_{('right' if ctx.side == 'left' else 'left')}"
        my_baseline_y = ctx.baseline_tray_y[my_key]
        vel_history: list[np.ndarray] = []

        while not ctx.stop_event.is_set():
            t0 = __import__("time").perf_counter()

            frame = ctx.get_frame()
            if frame is None:
                __import__("time").sleep(LIFT_STEP_INTERVAL_S)
                continue

            aruco = ctx.detector.detect(frame)
            tray_poses = ctx.detector.get_tray_poses(aruco)
            if tray_poses is not None:
                my_rise = my_baseline_y - tray_poses[my_key][1]
                if my_rise >= LIFT_HEIGHT_M:
                    break

            # Publish own joints to the shared snapshot, read both.
            current = run.read_joints_raw(ctx.robot)
            ctx.joint_snapshot.update(ctx.side, current)
            joints_L, joints_R = ctx.joint_snapshot.get_both()
            if joints_L is None or joints_R is None:
                __import__("time").sleep(LIFT_STEP_INTERVAL_S)
                continue

            q_L = _joints_to_rad_array(joints_L, self.cal_left)
            q_R = _joints_to_rad_array(joints_R, self.cal_right)

            ents = ctx.observer.observe(aruco, q_L, q_R)
            if ents is None:
                # Hold position: don't issue a new command this tick.
                __import__("time").sleep(LIFT_STEP_INTERVAL_S)
                continue

            graph = build_graph(ents)

            with torch.no_grad():
                x = graph.x.unsqueeze(0)
                e = graph.edge_attr.unsqueeze(0)
                out = self.model.forward_side(x, e, ctx.side)

            vel_norm = out["velocity"][0]
            vel_raw = (
                vel_norm * self.norm[f"std_{ctx.side}"]
                + self.norm[f"mean_{ctx.side}"]
            ).numpy()

            vel_history.append(vel_raw)
            if len(vel_history) > self.smooth_window:
                vel_history.pop(0)
            vel_smoothed = np.mean(np.stack(vel_history, axis=0), axis=0)

            step = vel_smoothed / fps
            max_abs = float(np.abs(step).max())
            if max_abs > self.max_step_raw:
                step = step * (self.max_step_raw / max_abs)

            target = {}
            for i, motor in enumerate(FK_MOTORS):
                target[motor] = float(current[motor]) + float(step[i])
            target["gripper"] = float(ctx.gripper_closed)
            run.write_joints_raw(ctx.robot, target)

            elapsed = __import__("time").perf_counter() - t0
            if LIFT_STEP_INTERVAL_S - elapsed > 0:
                __import__("time").sleep(LIFT_STEP_INTERVAL_S - elapsed)

        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # Wait for the other side to reach LIFT_HEIGHT_M as well, same
        # as the deterministic Lift.
        import time as _time
        while not ctx.stop_event.is_set():
            frame = ctx.get_frame()
            if frame is None:
                _time.sleep(0.033)
                continue
            poses = ctx.detector.detect(frame)
            tray_poses = ctx.detector.get_tray_poses(poses)
            if tray_poses is None:
                _time.sleep(0.033)
                continue
            other_rise = ctx.baseline_tray_y[other_key] - tray_poses[other_key][1]
            if other_rise >= LIFT_HEIGHT_M:
                break
            _time.sleep(0.033)

        return PhaseResult(success=not ctx.stop_event.is_set())
