"""
Node definitions, feature layout, and per-entity observation type for the
bimanual tray-lifting scene graph.

All positions are expressed in an arm's base frame (meters) — each arm's
SceneObserver produces a graph in its own base coordinates, matching
run.py's per-arm ``T_base_cam`` / ``cam_to_base`` convention.
Quaternions use scipy convention: (x, y, z, w).

Node ordering is fixed so the GNN's decoder can read a specific arm's
embedding by index after message-passing:

    0: left  end-effector   (arm)
    1: right end-effector   (arm)
    2: tray_left  marker    (tray)
    3: tray_right marker    (tray)

Node feature layout (NODE_FEATURE_DIM = 12):

    [0:3]   position xyz
    [3:7]   orientation quaternion (xyzw); zero-padded when unavailable
    [7:10]  linear velocity xyz (m/s)
    [10:12] one-hot type: [1,0] = arm, [0,1] = tray
"""

from dataclasses import dataclass

from typing import Optional

import numpy as np


# --- Node ordering ---
LEFT_EE = 0
RIGHT_EE = 1
TRAY_LEFT = 2
TRAY_RIGHT = 3
NUM_NODES = 4

# --- Feature dimensions ---
NODE_FEATURE_DIM = 12  # pos(3) + quat(4) + vel(3) + type(2)
EDGE_FEATURE_DIM = 8   # rel_pos(3) + dist(1) + contact(1) + rel_vel(3)

# --- Thresholds ---
CONTACT_THRESHOLD_M = 0.03  # 3 cm (per CLAUDE.md Phase 2b)

# --- Type one-hot vectors ---
TYPE_ARM = np.array([1.0, 0.0], dtype=np.float32)
TYPE_TRAY = np.array([0.0, 1.0], dtype=np.float32)


@dataclass
class EntityObs:
    """One entity's observation at a single timestep.

    Parameters
    ----------
    name : str
        Stable key used by VelocityTracker to persist previous-frame state.
    position : ndarray, shape (3,)
        XYZ position in the observer's arm base frame, meters.
    node_type : ndarray, shape (2,)
        One-hot type flag — use TYPE_ARM or TYPE_TRAY.
    velocity : ndarray, shape (3,)
        Linear velocity in the observer's arm base frame, meters/second. Typically produced
        by VelocityTracker before calling build_graph.
    orientation : ndarray, shape (4,), optional
        Quaternion (x, y, z, w). Set to None when orientation is not
        observed (e.g., tray markers in the current spec).
    """
    name: str
    position: np.ndarray
    node_type: np.ndarray
    velocity: np.ndarray
    orientation: Optional[np.ndarray] = None


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion.

    Pure-numpy implementation of Shepperd's method; matches scipy's
    ``Rotation.from_matrix(R).as_quat()`` xyzw convention so no SciPy
    dependency is needed.
    """
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)
