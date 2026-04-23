"""
Learned phase controllers — GNN-backed drop-in replacements for any
deterministic phase except Home.

A learned phase is a ``PhaseController`` whose inner per-tick command
comes from a ``TrayLiftGNN`` trained on frames from that phase alone
(see ``GNN/train.py --phase LIFTING``, ``--phase TRANSLATE``, ...).
The *scaffolding* — termination conditions, gripper open/close
transitions, optional deterministic primers like the lift nudge —
stays hand-coded because those are phase-level invariants that don't
need to be learned.

Each concrete controller derives from ``_LearnedPhase``, which owns
the common machinery:

* Loading from a checkpoint (``load()``)
* Per-tick observation + forward pass + step computation
  (``_predict_step``)
* Cumulative-target accumulation loop with clamping, smoothing, and
  2 s logging (``_gnn_loop``)

Subclasses only supply the phase-specific bits in ``run()``:
pre-loop action, termination callback or duration, gripper command
during the loop, post-loop action.

HOME is excluded — the arms return to a resting pose out of camera
frame and there is nothing for a GNN to observe.

Typical swap into a hybrid:

    USE_GNN_LIFTING    = True
    CKPT_LIFTING       = "GNN/checkpoints/lift.pt"
    USE_GNN_TRANSLATE  = True
    CKPT_TRANSLATE     = "GNN/checkpoints/translate.pt"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

import run
from run import (
    ARM_MOTORS,
    CONVERGENCE_THRESHOLD_M,
    CONVERGENCE_READINGS,
    GRASP_PROXIMITY_M,
    GRIPPER_CLOSE_WAIT_S,
    GRIPPER_OPEN_WAIT_S,
    LIFT_HEIGHT_M,
    LIFT_STEP_INTERVAL_S,
    LOWER_DURATION_S,
    LOWER_STEP_INTERVAL_S,
    NUDGE_DURATION_S,
    NUDGE_FRACTION,
    RELEASE_DURATION_S,
    TAG_RISE_THRESHOLD_M,
    TRANSLATE_DURATION_S,
    TRANSLATE_STEP_INTERVAL,
    smooth_move_single,
)
from algorithmic.motion_map import CONTROL_HZ, cam_to_base

from GNN.phases.base import PhaseController, PhaseContext, PhaseResult
from GNN.policy import TrayLiftGNN
from GNN.graph_builder import build_graph


MOTOR_RESOLUTION = 4096
FK_MOTORS = ARM_MOTORS  # 5 arm joints (gripper handled separately)


def _raw_to_rad(raw: float, cal_entry: dict) -> float:
    mid = 0.5 * (cal_entry["range_min"] + cal_entry["range_max"])
    deg = (raw - mid) * 360.0 / MOTOR_RESOLUTION
    return float(np.deg2rad(deg))


def _joints_to_rad_array(joints_raw: dict, cal: dict) -> np.ndarray:
    return np.array(
        [_raw_to_rad(joints_raw[m], cal[m]) for m in FK_MOTORS],
        dtype=np.float64,
    )


# ===========================================================================
# Base class
# ===========================================================================

class _LearnedPhase(PhaseController):
    """Common plumbing shared by every GNN-driven phase controller.

    Subclasses implement ``run()`` and usually end up calling
    ``self._gnn_loop(...)`` after any deterministic preludes and
    before any deterministic wrap-up.
    """

    name = "UNNAMED"

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

    # -- Checkpoint loading ------------------------------------------------

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path,
        cal_left: dict,
        cal_right: dict,
        max_step_deg: float = 4.0,
        smooth_window: int = 5,
    ) -> "_LearnedPhase":
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

    # -- Shared prerequisites ---------------------------------------------

    def _require_ctx(self, ctx: PhaseContext) -> None:
        if ctx.observer is None or ctx.joint_snapshot is None:
            raise RuntimeError(
                f"{type(self).__name__} requires ctx.observer and "
                f"ctx.joint_snapshot; run_hybrid.py should have attached "
                f"them before dispatch."
            )

    # -- One-tick inference ------------------------------------------------

    def _predict_step(
        self,
        ctx: PhaseContext,
        aruco: dict,
        vel_history: list[np.ndarray],
        interval_s: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run observer + model for one tick.

        Returns ``(step, vel_smoothed)``, both ``np.ndarray`` shape (5,),
        or ``(None, None)`` if the tick must be skipped (missing
        joint snapshot or scene graph).
        """
        current = run.read_joints_raw(ctx.robot)
        ctx.joint_snapshot.update(ctx.side, current)
        joints_L, joints_R = ctx.joint_snapshot.get_both()
        if joints_L is None or joints_R is None:
            return None, None

        q_L = _joints_to_rad_array(joints_L, self.cal_left)
        q_R = _joints_to_rad_array(joints_R, self.cal_right)
        ents = ctx.observer.observe(aruco, q_L, q_R)
        if ents is None:
            return None, None

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

        step = vel_smoothed * interval_s
        max_abs = float(np.abs(step).max())
        if max_abs > self.max_step_raw:
            step = step * (self.max_step_raw / max_abs)
        return step, vel_smoothed

    # -- Generic GNN-driven loop ------------------------------------------

    def _gnn_loop(
        self,
        ctx: PhaseContext,
        *,
        interval_s: float,
        gripper_cmd: float,
        should_terminate: Optional[Callable[[dict], bool]] = None,
        max_duration_s: Optional[float] = None,
        log_prefix: str = "LearnedPhase",
    ) -> tuple[int, int]:
        """Run the GNN-driven accumulation loop until termination.

        Exactly one of ``should_terminate`` or ``max_duration_s`` should
        be specified (both are fine too, whichever fires first wins).
        Returns ``(n_steps_commanded, n_ticks_skipped)``.
        """
        lift_target = {
            m: v for m, v in run.read_joints_raw(ctx.robot).items()
            if m in ARM_MOTORS
        }
        vel_history: list[np.ndarray] = []
        phase_start = time.perf_counter()
        last_log = phase_start
        n_steps = 0
        n_skipped = 0

        while not ctx.stop_event.is_set():
            t0 = time.perf_counter()

            if max_duration_s is not None and t0 - phase_start >= max_duration_s:
                break

            frame = ctx.get_frame()
            if frame is None:
                time.sleep(interval_s)
                continue

            aruco = ctx.detector.detect(frame)
            if should_terminate is not None and should_terminate(aruco):
                break

            step, vel_smoothed = self._predict_step(
                ctx, aruco, vel_history, interval_s,
            )
            if step is None:
                n_skipped += 1
                time.sleep(interval_s)
                continue

            for i, m in enumerate(ARM_MOTORS):
                lift_target[m] += float(step[i])
            lift_target["gripper"] = float(gripper_cmd)
            run.write_joints_raw(ctx.robot, lift_target)
            n_steps += 1

            if t0 - last_log >= 2.0:
                last_log = t0
                step_str = ",".join(f"{v:+5.2f}" for v in step)
                vel_str  = ",".join(f"{v:+5.0f}" for v in vel_smoothed)
                print(
                    f"  [{ctx.side}] {log_prefix}  step={n_steps}  "
                    f"skipped={n_skipped}  vel=[{vel_str}]  "
                    f"step_raw=[{step_str}]"
                )

            elapsed = time.perf_counter() - t0
            remaining = interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

        return n_steps, n_skipped


# ===========================================================================
# Concrete controllers
# ===========================================================================

class LearnedHover(_LearnedPhase):
    """Servo the gripper over the tray marker until converged.

    Termination matches deterministic ``Hover``: XZ distance between
    the gripper and tray marker (this arm's base frame) must drop below
    ``CONVERGENCE_THRESHOLD_M`` for ``CONVERGENCE_READINGS`` consecutive
    ticks. Gripper stays open throughout.
    """

    name = "HOVER"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        self._require_ctx(ctx)
        run._set_phase(ctx.side, "HOVER")

        converged = [0]  # boxed so the closure can mutate

        def _terminate(aruco: dict) -> bool:
            tray_cam = ctx.detector.get_tray_marker_pose(aruco, ctx.side)
            grip_cam = ctx.detector.get_gripper_pose(aruco, ctx.side)
            if tray_cam is None or grip_cam is None:
                converged[0] = 0
                return False
            tray_base = cam_to_base(tray_cam, ctx.T_base_cam)
            grip_base = cam_to_base(grip_cam, ctx.T_base_cam)
            err_bx = tray_base[0] - grip_base[0]
            err_bz = tray_base[2] - grip_base[2]
            xz_dist = float(np.sqrt(err_bx ** 2 + err_bz ** 2))
            if xz_dist < CONVERGENCE_THRESHOLD_M:
                converged[0] += 1
                return converged[0] >= CONVERGENCE_READINGS
            converged[0] = 0
            return False

        self._gnn_loop(
            ctx,
            interval_s=1.0 / CONTROL_HZ,
            gripper_cmd=ctx.gripper_open,
            should_terminate=_terminate,
            log_prefix="LearnedHover",
        )
        return PhaseResult(success=not ctx.stop_event.is_set())


class LearnedApproach(_LearnedPhase):
    """Descend with gripper open until gripper/tray proximity, then close.

    The open/close gripper commands stay deterministic — the GNN only
    shapes the descent velocities in between.
    """

    name = "APPROACH"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        self._require_ctx(ctx)
        run._set_phase(ctx.side, "APPROACH")

        # Pre-loop: ensure gripper is open.
        run.write_joints_raw(ctx.robot, {"gripper": ctx.gripper_open})
        time.sleep(GRIPPER_OPEN_WAIT_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        def _terminate(aruco: dict) -> bool:
            tray_cam = ctx.detector.get_tray_marker_pose(aruco, ctx.side)
            grip_cam = ctx.detector.get_gripper_pose(aruco, ctx.side)
            if tray_cam is None or grip_cam is None:
                return False
            vert_dist = abs(tray_cam[1] - grip_cam[1])
            return vert_dist < GRASP_PROXIMITY_M

        self._gnn_loop(
            ctx,
            interval_s=LIFT_STEP_INTERVAL_S,
            gripper_cmd=ctx.gripper_open,
            should_terminate=_terminate,
            log_prefix="LearnedApproach",
        )

        # Post-loop: close the gripper.
        if not ctx.stop_event.is_set():
            run.write_joints_raw(ctx.robot, {"gripper": ctx.gripper_closed})
            time.sleep(GRIPPER_CLOSE_WAIT_S)
        return PhaseResult(success=not ctx.stop_event.is_set())


class LearnedLift(_LearnedPhase):
    """GNN-driven bimanual lift with deterministic scaffolding.

    The deterministic nudge + wait-for-other-tag-rise prevent deadlock
    when the GNN's initial velocity is very small. The actual raise
    is fully GNN-driven with no tilt-aware alternate-step gating —
    coordination must come from the model.
    """

    name = "LIFTING"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        self._require_ctx(ctx)
        other = "right" if ctx.side == "left" else "left"
        my_key = f"tray_{ctx.side}"
        other_key = f"tray_{other}"
        my_baseline_y = ctx.baseline_tray_y[my_key]
        other_baseline_y = ctx.baseline_tray_y[other_key]

        # -- Deterministic nudge to break stiction and raise enough
        # for the opposing tag to register.
        run._set_phase(ctx.side, "LIFT_WAITING")
        current = run.read_joints_raw(ctx.robot)
        nudge_start = {m: current[m] for m in ARM_MOTORS}
        nudge_end = {m: current[m] - ctx.descent_delta[m] * NUDGE_FRACTION
                     for m in ARM_MOTORS}
        smooth_move_single(ctx.robot, nudge_start, nudge_end, NUDGE_DURATION_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # -- Wait for the opposing tag to rise.
        while not ctx.stop_event.is_set():
            frame = ctx.get_frame()
            if frame is None:
                time.sleep(0.033)
                continue
            poses = ctx.detector.detect(frame)
            tray_poses = ctx.detector.get_tray_poses(poses)
            if tray_poses is None:
                time.sleep(0.033)
                continue
            if other_baseline_y - tray_poses[other_key][1] >= TAG_RISE_THRESHOLD_M:
                break
            time.sleep(0.033)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # -- GNN-driven raise: terminate when this side reaches target.
        run._set_phase(ctx.side, "LIFTING")

        def _terminate(aruco: dict) -> bool:
            tray_poses = ctx.detector.get_tray_poses(aruco)
            if tray_poses is None:
                return False
            return (my_baseline_y - tray_poses[my_key][1]) >= LIFT_HEIGHT_M

        self._gnn_loop(
            ctx,
            interval_s=LIFT_STEP_INTERVAL_S,
            gripper_cmd=ctx.gripper_closed,
            should_terminate=_terminate,
            log_prefix="LearnedLift",
        )
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # -- Wait for the other side to also reach target height.
        while not ctx.stop_event.is_set():
            frame = ctx.get_frame()
            if frame is None:
                time.sleep(0.033)
                continue
            poses = ctx.detector.detect(frame)
            tray_poses = ctx.detector.get_tray_poses(poses)
            if tray_poses is None:
                time.sleep(0.033)
                continue
            if (other_baseline_y - tray_poses[other_key][1]) >= LIFT_HEIGHT_M:
                break
            time.sleep(0.033)

        return PhaseResult(success=not ctx.stop_event.is_set())


class LearnedTranslate(_LearnedPhase):
    """GNN-driven horizontal translation at lift height.

    The GNN's two decoder heads must implicitly pick up the leader/
    follower asymmetry seen in training (left leads, right follows).
    """

    name = "TRANSLATE"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        self._require_ctx(ctx)
        run._set_phase(
            ctx.side, "TRANSLATE",
            detail="leader" if ctx.side == "left" else "follower",
        )
        self._gnn_loop(
            ctx,
            interval_s=TRANSLATE_STEP_INTERVAL,
            gripper_cmd=ctx.gripper_closed,
            max_duration_s=TRANSLATE_DURATION_S,
            log_prefix="LearnedTranslate",
        )
        return PhaseResult(success=not ctx.stop_event.is_set())


class LearnedLower(_LearnedPhase):
    """GNN-driven tray lowering.

    Fixed duration; gripper stays closed. The deterministic ``Lower``
    has a tilt-aware pausing mechanism — it's intentionally omitted
    here so the GNN is responsible for level-descent coordination.
    """

    name = "LOWER"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        self._require_ctx(ctx)
        run._set_phase(ctx.side, "LOWER")
        self._gnn_loop(
            ctx,
            interval_s=LOWER_STEP_INTERVAL_S,
            gripper_cmd=ctx.gripper_closed,
            max_duration_s=LOWER_DURATION_S,
            log_prefix="LearnedLower",
        )
        return PhaseResult(success=not ctx.stop_event.is_set())


class LearnedRelease(_LearnedPhase):
    """Open the gripper, then run the GNN for RELEASE_DURATION_S.

    The deterministic ``Release`` is a single smooth_move to the
    release waypoint. Here the gripper_open is still deterministic,
    but the pull-back motion is the GNN's responsibility.
    """

    name = "RELEASE"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        self._require_ctx(ctx)
        run._set_phase(ctx.side, "RELEASE")

        run.write_joints_raw(ctx.robot, {"gripper": ctx.gripper_open})
        time.sleep(GRIPPER_OPEN_WAIT_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        self._gnn_loop(
            ctx,
            interval_s=1.0 / 30.0,
            gripper_cmd=ctx.gripper_open,
            max_duration_s=RELEASE_DURATION_S,
            log_prefix="LearnedRelease",
        )
        return PhaseResult(success=not ctx.stop_event.is_set())
