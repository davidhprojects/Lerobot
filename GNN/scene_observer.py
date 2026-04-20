"""
Glue layer between Phase 1 perception and Phase 2 graph construction.

Given one tick's observations (ArUco poses, per-arm joint angles, saved
base transforms), SceneObserver produces the canonical
[left_ee, right_ee, tray_left, tray_right] list[EntityObs] that
build_graph consumes.

All positions in the output are in the D435i camera frame (meters).

Sources per entity (see CLAUDE.md Phase 1b):

    left_ee, right_ee   - forward kinematics of own joint encoders,
                          lifted to camera frame via T_cam_base
    tray_left           - marker ID 5 (camera frame directly)
    tray_right          - marker ID 3 (camera frame directly)

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
from Algorithmic.forward_kinematics import forward_kinematics


class SceneObserver:
    """
    Converts per-tick raw observations into a list[EntityObs].

    Parameters
    ----------
    base_transforms : dict[str, ndarray]
        {'left': T_cam_base_left, 'right': T_cam_base_right}, each a 4x4
        homogeneous transform.  Load with
        ``Algorithmic.calibrate.load_base_transforms()``.
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
        base_transforms: dict[str, np.ndarray],
        marker_config: Optional[MarkerConfig] = None,
        dt: float = 1.0 / 30.0,
        ema_alpha: float = 0.3,
    ):
        if "left" not in base_transforms or "right" not in base_transforms:
            raise KeyError(
                "base_transforms must contain both 'left' and 'right'. "
                "Run Algorithmic/calibrate.py and reload."
            )
        self.T_cam_base = {
            "left":  np.asarray(base_transforms["left"],  dtype=np.float64),
            "right": np.asarray(base_transforms["right"], dtype=np.float64),
        }
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
        Build the canonical 4-entity observation list for one tick.

        Parameters
        ----------
        aruco_poses : dict[int, ndarray]
            Output of ``ArucoDetector.detect()``.
            Maps marker_id -> 4x4 ``T_cam_marker``.
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

        # Own-arm EE poses via FK, lifted to camera frame.
        T_cam_ee_L = self.T_cam_base["left"]  @ forward_kinematics(left_joints_rad)
        T_cam_ee_R = self.T_cam_base["right"] @ forward_kinematics(right_joints_rad)

        left_ee_pos  = T_cam_ee_L[:3, 3].astype(np.float32)
        right_ee_pos = T_cam_ee_R[:3, 3].astype(np.float32)
        left_ee_quat  = rotation_matrix_to_quaternion(T_cam_ee_L[:3, :3])
        right_ee_quat = rotation_matrix_to_quaternion(T_cam_ee_R[:3, :3])

        # Tray markers are already in camera frame — just pull translations.
        tray_left_pos  = aruco_poses[tl_id][:3, 3].astype(np.float32)
        tray_right_pos = aruco_poses[tr_id][:3, 3].astype(np.float32)

        v_le = self.velocity.update("left_ee",   left_ee_pos,  self.dt)
        v_re = self.velocity.update("right_ee",  right_ee_pos, self.dt)
        v_tl = self.velocity.update("tray_left", tray_left_pos, self.dt)
        v_tr = self.velocity.update("tray_right", tray_right_pos, self.dt)

        return [
            EntityObs("left_ee",    left_ee_pos,    TYPE_ARM,  v_le, left_ee_quat),
            EntityObs("right_ee",   right_ee_pos,   TYPE_ARM,  v_re, right_ee_quat),
            EntityObs("tray_left",  tray_left_pos,  TYPE_TRAY, v_tl, None),
            EntityObs("tray_right", tray_right_pos, TYPE_TRAY, v_tr, None),
        ]
