"""
Glue layer between Phase 1 perception and Phase 2 graph construction.

Given one tick's observations (ArUco poses, per-arm joint angles, saved
base transforms), SceneObserver produces the canonical
[left_ee, right_ee, tray_left, tray_right] list[EntityObs] that
build_graph consumes.

All positions in the output are expressed in this observer's arm's base
frame (specified by ``side`` at construction).  Each physical arm should
run its own SceneObserver — the left arm reasons in left-base
coordinates, the right arm in right-base.  This matches run.py, which
already uses per-arm base frames via ``cam_to_base`` + ``T_base_cam``.

Sources per entity (see CLAUDE.md Phase 1b):

    own-arm EE           - forward kinematics of own joint encoders
                           (already in own-base frame)
    other-arm EE         - FK of other arm's joints, bridged through the
                           camera frame into this arm's base frame
    tray_left (ID 5)     - ArUco marker, camera → this arm's base
    tray_right (ID 3)    - ArUco marker, camera → this arm's base

Velocity is estimated internally with a single VelocityTracker.  Call
reset() between episodes so state from one demo doesn't leak into the
next.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from GNN.entities import (
    EntityObs, TYPE_ARM, TYPE_TRAY,
    rotation_matrix_to_quaternion,
)
from GNN.graph_builder import VelocityTracker
from perception.aruco import MarkerConfig
from algorithmic.forward_kinematics import forward_kinematics


def _cam_to_base(p_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Transform a 3D point from camera frame to an arm's base frame."""
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return (T_base_cam @ p_h)[:3]


class SceneObserver:
    """
    Converts per-tick raw observations into a list[EntityObs] in one arm's
    base frame.

    Parameters
    ----------
    side : {'left', 'right'}
        The base frame the produced graph is expressed in.  Each physical
        arm should construct its own SceneObserver with its own side.
    base_transforms : dict[str, ndarray]
        ``{'left': T_cam_base_left, 'right': T_cam_base_right}``, each a
        4x4 homogeneous transform mapping base-frame points to camera-
        frame points.  Load with
        ``algorithmic.calibrate.load_base_transforms()``.
    marker_config : MarkerConfig, optional
        ArUco ID-to-role mapping.  Defaults to the repo-standard config.
    dt : float, optional
        Seconds between consecutive observe() calls.  Used for finite-
        difference velocity.  Defaults to 1/30 (30 Hz camera).
    ema_alpha : float, optional
        Exponential-moving-average weight for velocity smoothing.
        Defaults to 0.3 (CLAUDE.md Phase 1c).
    """

    def __init__(
        self,
        side: str,
        base_transforms: dict[str, np.ndarray],
        marker_config: Optional[MarkerConfig] = None,
        dt: float = 1.0 / 30.0,
        ema_alpha: float = 0.3,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if "left" not in base_transforms or "right" not in base_transforms:
            raise KeyError(
                "base_transforms must contain both 'left' and 'right'. "
                "Run algorithmic/calibrate.py and reload."
            )

        self.side = side
        # T_cam_base[s]: side-s-base frame → camera frame
        self.T_cam_base = {
            "left":  np.asarray(base_transforms["left"],  dtype=np.float64),
            "right": np.asarray(base_transforms["right"], dtype=np.float64),
        }
        # camera → this arm's base
        self.T_base_cam = np.linalg.inv(self.T_cam_base[side])
        # other-arm's base → this arm's base (used to place the other EE
        # into my frame without depending on its gripper marker)
        other = "right" if side == "left" else "left"
        self.T_base_otherbase = self.T_base_cam @ self.T_cam_base[other]

        self.cfg = marker_config or MarkerConfig()
        self.dt = dt
        self.velocity = VelocityTracker(alpha=ema_alpha)

    def reset(self):
        """Clear per-entity velocity state — call between episodes."""
        self.velocity.reset()

    def observe(
        self,
        aruco_poses: dict[int, np.ndarray],
        left_joints_rad: np.ndarray,
        right_joints_rad: np.ndarray,
    ) -> Optional[list[EntityObs]]:
        """
        Build the canonical 4-entity observation list for one tick, in
        this observer's base frame.

        Parameters
        ----------
        aruco_poses : dict[int, ndarray]
            Output of ``ArucoDetector.detect()``.  Maps marker_id → 4x4
            ``T_cam_marker``.
        left_joints_rad, right_joints_rad : ndarray, shape (5,)
            SO-101 joint angles in radians:
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].

        Returns
        -------
        entities : list[EntityObs] or None
            None if either tray marker is missing this frame — caller
            decides whether to drop the frame, hold last, or treat as
            dropout for training.
        """
        tl_id = self.cfg.tray_left
        tr_id = self.cfg.tray_right
        if tl_id not in aruco_poses or tr_id not in aruco_poses:
            return None

        # FK produces each arm's EE in its OWN base frame. For the arm
        # matching self.side that IS our frame; the other one needs the
        # base-to-base bridge.
        T_own_base_ee_L = forward_kinematics(left_joints_rad)
        T_own_base_ee_R = forward_kinematics(right_joints_rad)

        if self.side == "left":
            T_my_ee_L = T_own_base_ee_L
            T_my_ee_R = self.T_base_otherbase @ T_own_base_ee_R
        else:
            T_my_ee_L = self.T_base_otherbase @ T_own_base_ee_L
            T_my_ee_R = T_own_base_ee_R

        left_ee_pos  = T_my_ee_L[:3, 3].astype(np.float32)
        right_ee_pos = T_my_ee_R[:3, 3].astype(np.float32)
        left_ee_quat  = rotation_matrix_to_quaternion(T_my_ee_L[:3, :3])
        right_ee_quat = rotation_matrix_to_quaternion(T_my_ee_R[:3, :3])

        # Tray markers: camera frame → this arm's base frame.
        tray_left_pos  = _cam_to_base(
            aruco_poses[tl_id][:3, 3], self.T_base_cam
        ).astype(np.float32)
        tray_right_pos = _cam_to_base(
            aruco_poses[tr_id][:3, 3], self.T_base_cam
        ).astype(np.float32)

        v_le = self.velocity.update("left_ee",    left_ee_pos,    self.dt)
        v_re = self.velocity.update("right_ee",   right_ee_pos,   self.dt)
        v_tl = self.velocity.update("tray_left",  tray_left_pos,  self.dt)
        v_tr = self.velocity.update("tray_right", tray_right_pos, self.dt)

        return [
            EntityObs("left_ee",    left_ee_pos,    TYPE_ARM,  v_le, left_ee_quat),
            EntityObs("right_ee",   right_ee_pos,   TYPE_ARM,  v_re, right_ee_quat),
            EntityObs("tray_left",  tray_left_pos,  TYPE_TRAY, v_tl, None),
            EntityObs("tray_right", tray_right_pos, TYPE_TRAY, v_tr, None),
        ]
