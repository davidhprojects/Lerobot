"""
Deterministic phase controllers that replicate run.py's hand-coded logic.

Each class mirrors one phase of run.py's ``run_arm`` state machine so
run_hybrid.py can compose them (and later swap any one for a
``learned`` counterpart). Behavior is intentionally identical to
run.py — any divergence is a bug.

We import constants + helpers from ``run`` rather than re-defining
them, so tuning stays in a single place. Motor reads / writes go
through ``run.read_joints_raw`` / ``run.write_joints_raw`` so that
run_hybrid.py's bus-lock monkey-patches apply automatically.
"""

from __future__ import annotations

import time

import numpy as np

import run
from run import (
    ARM_MOTORS,
    CONVERGENCE_THRESHOLD_M,
    CONVERGENCE_READINGS,
    HOVER_FEEDBACK_GAIN,
    NUDGE_DURATION_S,
    NUDGE_FRACTION,
    GRIPPER_CLOSE_WAIT_S,
    GRIPPER_OPEN_WAIT_S,
    RELEASE_DURATION_S,
    RETURN_HOME_DURATION,
    TAG_RISE_THRESHOLD_M,
    LIFT_HEIGHT_M,
    LIFT_STEP_FRACTION,
    LIFT_STEP_INTERVAL_S,
    TRANSLATE_DURATION_S,
    TRANSLATE_STEP_INTERVAL,
    TRANSLATE_TRANSITION_DURATION_S,
    TRANSLATE_HEIGHT_TRIM,
    TILT_PAUSE_THRESHOLD_M,
    GRASP_PROXIMITY_M,
    LOWER_DURATION_S,
    LOWER_STEP_INTERVAL_S,
    find_closest_coords,
    minimum_jerk,
    smooth_move_single,
)
from algorithmic.motion_map import (
    cam_to_base,
    MAX_JOINT_STEP_RAW,
    CONTROL_HZ,
    RAISE_DIST_SCALE_M,
    RAISE_OFFSETS,
)
from algorithmic.build_motion_map import POSITION_OFFSET

from GNN.phases.base import PhaseController, PhaseContext, PhaseResult


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


# ---------------------------------------------------------------------------
# HOVER
# ---------------------------------------------------------------------------

class Hover(PhaseController):
    """Servo the gripper above the tray marker via the motion map
    until the XZ error is below CONVERGENCE_THRESHOLD_M for
    CONVERGENCE_READINGS ticks in a row."""

    name = "HOVER"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "HOVER")
        interval = 1.0 / CONTROL_HZ

        smooth_bx: float | None = None
        smooth_bz: float | None = None
        alpha = 0.2
        miss_streak = 0
        converged_count = 0

        tray_marker_id = (ctx.detector.config.tray_left if ctx.side == "left"
                          else ctx.detector.config.tray_right)

        while not ctx.stop_event.is_set():
            t0 = time.perf_counter()

            frame = ctx.get_frame()
            if frame is None:
                time.sleep(interval)
                continue

            poses = ctx.detector.detect(frame)
            tray_cam = ctx.detector.get_tray_marker_pose(poses, ctx.side)
            if tray_cam is None:
                miss_streak += 1
                if miss_streak == 30:
                    print(f"  [{ctx.side}] HOVER: lost tray marker "
                          f"(ID {tray_marker_id}), holding position")
                time.sleep(interval)
                continue
            if miss_streak >= 30:
                print(f"  [{ctx.side}] HOVER: tray marker recovered")
            miss_streak = 0

            tray_base = cam_to_base(tray_cam, ctx.T_base_cam)

            if smooth_bx is None:
                smooth_bx = tray_base[0]
                smooth_bz = tray_base[2]
            else:
                smooth_bx = alpha * tray_base[0] + (1 - alpha) * smooth_bx
                smooth_bz = alpha * tray_base[2] + (1 - alpha) * smooth_bz

            target_bx, target_bz = smooth_bx, smooth_bz
            if not ctx.motion_map.in_bounds(target_bx, target_bz):
                time.sleep(interval)
                continue

            pos_off = POSITION_OFFSET.get(ctx.side, {})
            adjusted_bx = target_bx - pos_off.get("base_x", 0.0)
            adjusted_bz = target_bz - pos_off.get("base_z", 0.0)

            grip_cam = ctx.detector.get_gripper_pose(poses, ctx.side)
            xz_dist = None
            error_bx = 0.0
            error_bz = 0.0
            if grip_cam is not None:
                grip_base = cam_to_base(grip_cam, ctx.T_base_cam)
                error_bx = adjusted_bx - grip_base[0]
                error_bz = adjusted_bz - grip_base[2]
                xz_dist = float(np.sqrt(error_bx ** 2 + error_bz ** 2))

            query_bx = target_bx + error_bx * HOVER_FEEDBACK_GAIN
            query_bz = target_bz + error_bz * HOVER_FEEDBACK_GAIN
            if not ctx.motion_map.in_bounds(query_bx, query_bz):
                query_bx, query_bz = target_bx, target_bz

            target_joints = ctx.motion_map.evaluate(query_bx, query_bz)

            # Hover raised: subtract 30% of a descent delta so the arms
            # approach from above.
            for m in ARM_MOTORS:
                target_joints[m] -= ctx.descent_delta[m] * 0.3

            if xz_dist is not None:
                raise_scale = xz_dist / RAISE_DIST_SCALE_M
                for name, offset in RAISE_OFFSETS.items():
                    target_joints[name] += offset * raise_scale

            trim = TRANSLATE_HEIGHT_TRIM.get(ctx.side, 0.0)
            for m in ARM_MOTORS:
                target_joints[m] += ctx.descent_delta[m] * trim

            current_joints = run.read_joints_raw(ctx.robot)
            deltas = {
                name: target_joints[name] - current_joints[name]
                for name in ARM_MOTORS
            }
            max_abs = max(abs(d) for d in deltas.values())
            if max_abs > MAX_JOINT_STEP_RAW:
                scale = MAX_JOINT_STEP_RAW / max_abs
                deltas = {n: d * scale for n, d in deltas.items()}
            cmd = {n: current_joints[n] + deltas[n] for n in ARM_MOTORS}
            run.write_joints_raw(ctx.robot, cmd)

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

        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# APPROACH — open gripper, descend until gripper/tray proximity, close
# ---------------------------------------------------------------------------

class Approach(PhaseController):
    name = "APPROACH"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "APPROACH")

        run.write_joints_raw(ctx.robot, {"gripper": ctx.gripper_open})
        time.sleep(GRIPPER_OPEN_WAIT_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        descent_step = {m: ctx.descent_delta[m] * LIFT_STEP_FRACTION
                        for m in ARM_MOTORS}
        descent_target = {
            m: v for m, v in run.read_joints_raw(ctx.robot).items()
            if m in ARM_MOTORS
        }

        while not ctx.stop_event.is_set():
            frame = ctx.get_frame()
            if frame is not None:
                poses = ctx.detector.detect(frame)
                tray_cam = ctx.detector.get_tray_marker_pose(poses, ctx.side)
                grip_cam = ctx.detector.get_gripper_pose(poses, ctx.side)
                if tray_cam is not None and grip_cam is not None:
                    vert_dist = abs(tray_cam[1] - grip_cam[1])
                    if vert_dist < GRASP_PROXIMITY_M:
                        break

            for m in ARM_MOTORS:
                descent_target[m] += descent_step[m]
            run.write_joints_raw(ctx.robot, descent_target)
            time.sleep(LIFT_STEP_INTERVAL_S)

        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        run.write_joints_raw(ctx.robot, {"gripper": ctx.gripper_closed})
        time.sleep(GRIPPER_CLOSE_WAIT_S)
        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# LIFT — nudge, wait for the other side's tag to rise, alternate-step
# lift until this side reaches LIFT_HEIGHT, then wait for the other side
# to reach LIFT_HEIGHT as well.
# ---------------------------------------------------------------------------

class Lift(PhaseController):
    """Wraps the LIFT_WAITING + LIFTING + wait-for-other sub-states."""

    name = "LIFTING"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "LIFT_WAITING")
        other = _other(ctx.side)

        # -- Nudge lift --
        current = run.read_joints_raw(ctx.robot)
        nudge_start = {m: current[m] for m in ARM_MOTORS}
        nudge_end = {m: current[m] - ctx.descent_delta[m] * NUDGE_FRACTION
                     for m in ARM_MOTORS}
        smooth_move_single(ctx.robot, nudge_start, nudge_end, NUDGE_DURATION_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # -- Wait for opposing tray tag to rise --
        other_key = f"tray_{other}"
        baseline_y = ctx.baseline_tray_y[other_key]
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
            current_y = tray_poses[other_key][1]
            rise = baseline_y - current_y
            if rise >= TAG_RISE_THRESHOLD_M:
                break
            time.sleep(0.033)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # -- LIFTING: tilt-aware alternate-step raise --
        run._set_phase(ctx.side, "LIFTING")
        my_key = f"tray_{ctx.side}"
        my_baseline_y = ctx.baseline_tray_y[my_key]
        lift_step_delta = {m: -ctx.descent_delta[m] * LIFT_STEP_FRACTION
                           for m in ARM_MOTORS}
        lift_target = {
            m: v for m, v in run.read_joints_raw(ctx.robot).items()
            if m in ARM_MOTORS
        }

        while not ctx.stop_event.is_set():
            frame = ctx.get_frame()
            if frame is None:
                time.sleep(LIFT_STEP_INTERVAL_S)
                continue
            poses = ctx.detector.detect(frame)
            tray_poses = ctx.detector.get_tray_poses(poses)
            if tray_poses is None:
                time.sleep(LIFT_STEP_INTERVAL_S)
                continue

            my_y = tray_poses[my_key][1]
            other_y = tray_poses[other_key][1]
            my_rise = my_baseline_y - my_y
            other_rise = ctx.baseline_tray_y[other_key] - other_y

            if my_rise >= LIFT_HEIGHT_M:
                break
            if my_rise <= other_rise:
                for m in ARM_MOTORS:
                    lift_target[m] += lift_step_delta[m]
                run.write_joints_raw(ctx.robot, lift_target)

            time.sleep(LIFT_STEP_INTERVAL_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        # -- Wait for the other side to also reach LIFT_HEIGHT --
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
            other_rise = ctx.baseline_tray_y[other_key] - tray_poses[other_key][1]
            if other_rise >= LIFT_HEIGHT_M:
                break
            time.sleep(0.033)

        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# TRANSLATE_PREP — smooth joint-space move from the delta-accumulated
# lifted pose to the motion-map-based pose TRANSLATE starts from.
# ---------------------------------------------------------------------------

class TranslatePrep(PhaseController):
    name = "TRANSLATE_PREP"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "TRANSLATE_PREP")

        trim = TRANSLATE_HEIGHT_TRIM.get(ctx.side, 0.0)
        height_trim = {m: ctx.descent_delta[m] * trim for m in ARM_MOTORS}

        bx_vals, bz_vals = [], []
        for _ in range(30):
            if ctx.stop_event.is_set():
                return PhaseResult(success=False)
            frame = ctx.get_frame()
            if frame is not None:
                poses = ctx.detector.detect(frame)
                tray_cam = ctx.detector.get_tray_marker_pose(poses, ctx.side)
                if tray_cam is not None:
                    tb = cam_to_base(tray_cam, ctx.T_base_cam)
                    bx_vals.append(tb[0])
                    bz_vals.append(tb[2])
            if len(bx_vals) >= 5:
                break
            time.sleep(0.033)

        if len(bx_vals) >= 3:
            prep_bx = float(np.mean(bx_vals))
            prep_bz = float(np.mean(bz_vals))
            if ctx.motion_map.in_bounds(prep_bx, prep_bz):
                prep_target = ctx.motion_map.evaluate(prep_bx, prep_bz)
                for m in ARM_MOTORS:
                    prep_target[m] += height_trim[m]
                current = run.read_joints_raw(ctx.robot)
                prep_start = {m: current[m] for m in ARM_MOTORS}
                smooth_move_single(
                    ctx.robot, prep_start, prep_target,
                    TRANSLATE_TRANSITION_DURATION_S,
                )

        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# TRANSLATE — leader (left) interpolates in (bx, bz); follower (right)
# offsets by the measured tray vector and evaluates its own motion map.
# ---------------------------------------------------------------------------

class Translate(PhaseController):
    name = "TRANSLATE"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(
            ctx.side, "TRANSLATE",
            detail="leader" if ctx.side == "left" else "follower",
        )

        trim = TRANSLATE_HEIGHT_TRIM.get(ctx.side, 0.0)
        height_trim = {m: ctx.descent_delta[m] * trim for m in ARM_MOTORS}

        # Measure the tray offset vector (right - left) once.
        tray_offset_bx: float | None = None
        tray_offset_bz: float | None = None
        for _ in range(30):
            frame = ctx.get_frame()
            if frame is not None:
                poses = ctx.detector.detect(frame)
                tray_poses = ctx.detector.get_tray_poses(poses)
                if tray_poses is not None:
                    left_tray_base = cam_to_base(
                        tray_poses["tray_left"], ctx.T_base_cam
                    )
                    right_tray_base = cam_to_base(
                        tray_poses["tray_right"], ctx.T_base_cam
                    )
                    tray_offset_bx = right_tray_base[0] - left_tray_base[0]
                    tray_offset_bz = right_tray_base[2] - left_tray_base[2]
                    break
            time.sleep(0.033)

        n_steps = max(1, int(TRANSLATE_DURATION_S / TRANSLATE_STEP_INTERVAL))

        if ctx.side == "left":
            return self._run_leader(ctx, n_steps, height_trim)
        return self._run_follower(
            ctx, n_steps, height_trim, tray_offset_bx, tray_offset_bz,
        )

    # -- LEADER --

    def _run_leader(self, ctx, n_steps, height_trim) -> PhaseResult:
        target_joints = {m: ctx.waypoints["translated"][ctx.side][m]
                         for m in ARM_MOTORS}
        dest_bx, dest_bz, _ = find_closest_coords(ctx.motion_map, target_joints)

        bx_readings, bz_readings = [], []
        for _ in range(30):
            frame = ctx.get_frame()
            if frame is not None:
                poses = ctx.detector.detect(frame)
                tray_cam = ctx.detector.get_tray_marker_pose(poses, ctx.side)
                if tray_cam is not None:
                    tray_base = cam_to_base(tray_cam, ctx.T_base_cam)
                    bx_readings.append(tray_base[0])
                    bz_readings.append(tray_base[2])
            if len(bx_readings) >= 5:
                break
            time.sleep(0.033)

        if len(bx_readings) >= 3:
            start_bx = float(np.mean(bx_readings))
            start_bz = float(np.mean(bz_readings))
        else:
            current_arm = {m: run.read_joints_raw(ctx.robot)[m] for m in ARM_MOTORS}
            start_bx, start_bz, _ = find_closest_coords(ctx.motion_map, current_arm)

        for step in range(n_steps):
            if ctx.stop_event.is_set():
                return PhaseResult(success=False)
            t = (step + 1) / n_steps * TRANSLATE_DURATION_S
            progress = minimum_jerk(t, TRANSLATE_DURATION_S)
            interp_bx = start_bx + (dest_bx - start_bx) * progress
            interp_bz = start_bz + (dest_bz - start_bz) * progress
            cmd = ctx.motion_map.evaluate(interp_bx, interp_bz)
            for m in ARM_MOTORS:
                cmd[m] += height_trim[m]
            run.write_joints_raw(ctx.robot, cmd)
            time.sleep(TRANSLATE_STEP_INTERVAL)

        return PhaseResult(success=True)

    # -- FOLLOWER --

    def _run_follower(
        self, ctx, n_steps, height_trim, tray_offset_bx, tray_offset_bz,
    ) -> PhaseResult:
        if tray_offset_bx is None:
            return PhaseResult(success=not ctx.stop_event.is_set())

        smooth_fbx: float | None = None
        smooth_fbz: float | None = None
        f_alpha = 0.1

        for step in range(n_steps):
            if ctx.stop_event.is_set():
                return PhaseResult(success=False)

            frame = ctx.get_frame()
            if frame is not None:
                poses = ctx.detector.detect(frame)
                left_tray_cam = ctx.detector.get_tray_marker_pose(poses, "left")
                if left_tray_cam is not None:
                    left_in_my_base = cam_to_base(left_tray_cam, ctx.T_base_cam)
                    my_target_bx = left_in_my_base[0] + tray_offset_bx
                    my_target_bz = left_in_my_base[2] + tray_offset_bz

                    if smooth_fbx is None:
                        smooth_fbx, smooth_fbz = my_target_bx, my_target_bz
                    else:
                        smooth_fbx = f_alpha * my_target_bx + (1 - f_alpha) * smooth_fbx
                        smooth_fbz = f_alpha * my_target_bz + (1 - f_alpha) * smooth_fbz

                    if ctx.motion_map.in_bounds(smooth_fbx, smooth_fbz):
                        cmd = ctx.motion_map.evaluate(smooth_fbx, smooth_fbz)
                        for m in ARM_MOTORS:
                            cmd[m] += height_trim[m]
                        run.write_joints_raw(ctx.robot, cmd)

            time.sleep(TRANSLATE_STEP_INTERVAL)

        return PhaseResult(success=True)


# ---------------------------------------------------------------------------
# LOWER — descend from translated pose toward lowered waypoint, pausing
# when my side is lower than the other to let it catch up.
# ---------------------------------------------------------------------------

class Lower(PhaseController):
    name = "LOWER"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "LOWER")
        other = _other(ctx.side)
        my_key = f"tray_{ctx.side}"
        other_key = f"tray_{other}"

        lower_dest = {m: ctx.waypoints["lowered"][ctx.side][m] for m in ARM_MOTORS}
        arm_start = {
            m: v for m, v in run.read_joints_raw(ctx.robot).items()
            if m in ARM_MOTORS
        }
        n_steps = max(1, int(LOWER_DURATION_S / LOWER_STEP_INTERVAL_S))
        step = 0

        while step < n_steps and not ctx.stop_event.is_set():
            t = (step + 1) / n_steps * LOWER_DURATION_S
            progress = minimum_jerk(t, LOWER_DURATION_S)

            should_step = True
            frame = ctx.get_frame()
            if frame is not None:
                poses = ctx.detector.detect(frame)
                tray_poses = ctx.detector.get_tray_poses(poses)
                if tray_poses is not None:
                    tilt = tray_poses[my_key][1] - tray_poses[other_key][1]
                    if tilt > TILT_PAUSE_THRESHOLD_M:
                        should_step = False

            if should_step:
                cmd = {m: arm_start[m] + (lower_dest[m] - arm_start[m]) * progress
                       for m in ARM_MOTORS}
                run.write_joints_raw(ctx.robot, cmd)
                step += 1

            time.sleep(LOWER_STEP_INTERVAL_S)

        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# RELEASE — open gripper + smooth move to the release waypoint
# ---------------------------------------------------------------------------

class Release(PhaseController):
    name = "RELEASE"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "RELEASE")
        run.write_joints_raw(ctx.robot, {"gripper": ctx.gripper_open})
        time.sleep(GRIPPER_OPEN_WAIT_S)
        if ctx.stop_event.is_set():
            return PhaseResult(success=False)

        current = run.read_joints_raw(ctx.robot)
        release_start = {m: current[m] for m in ARM_MOTORS}
        release_end = {m: ctx.waypoints["release"][ctx.side][m] for m in ARM_MOTORS}
        smooth_move_single(ctx.robot, release_start, release_end, RELEASE_DURATION_S)

        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# HOME — final return-to-home smooth move
# ---------------------------------------------------------------------------

class Home(PhaseController):
    name = "HOME"

    def run(self, ctx: PhaseContext) -> PhaseResult:
        run._set_phase(ctx.side, "HOME")
        current = run.read_joints_raw(ctx.robot)
        home_start = {m: current[m] for m in ARM_MOTORS}
        home_end = {m: ctx.waypoints["home"][ctx.side][m] for m in ARM_MOTORS}
        smooth_move_single(ctx.robot, home_start, home_end, RETURN_HOME_DURATION)
        run._set_phase(ctx.side, "DONE")
        return PhaseResult(success=not ctx.stop_event.is_set())


# ---------------------------------------------------------------------------
# Canonical ordering — matches run.py's run_arm exactly.
# ---------------------------------------------------------------------------

def default_phases() -> list[PhaseController]:
    return [
        Hover(),
        Approach(),
        Lift(),
        TranslatePrep(),
        Translate(),
        Lower(),
        Release(),
        Home(),
    ]
