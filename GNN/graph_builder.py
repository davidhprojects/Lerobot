"""
Scene graph construction for the bimanual tray-lifting GNN.

Builds a fully-connected 4-node directed graph (left_ee, right_ee,
tray_left, tray_right) with the feature layout documented in entities.py.
The output is a lightweight GraphData container; wrap it in a
torch_geometric.data.Data at call sites that need it.

The builder itself is pure — velocity estimation is factored into
VelocityTracker so callers keep explicit control over temporal state.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from GNN.entities import (
    EntityObs,
    NUM_NODES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM,
    CONTACT_THRESHOLD_M,
)


class VelocityTracker:
    """EMA-smoothed finite-difference velocity estimator, keyed by name.

    v_instant(t) = (pos(t) - pos(t-1)) / dt
    v_smooth(t)  = alpha * v_instant(t) + (1 - alpha) * v_smooth(t-1)

    The first observation for a given name yields zero velocity (no prior
    frame to diff against).
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._prev_pos: dict[str, np.ndarray] = {}
        self._smooth_vel: dict[str, np.ndarray] = {}

    def update(self, name: str, position: np.ndarray, dt: float) -> np.ndarray:
        """Record a new observation and return smoothed velocity (3,)."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        pos = np.asarray(position, dtype=np.float32)
        prev = self._prev_pos.get(name)
        if prev is None:
            v_instant = np.zeros(3, dtype=np.float32)
        else:
            v_instant = ((pos - prev) / dt).astype(np.float32)
        prev_smooth = self._smooth_vel.get(name, np.zeros(3, dtype=np.float32))
        v_smooth = (self.alpha * v_instant + (1.0 - self.alpha) * prev_smooth).astype(np.float32)
        self._prev_pos[name] = pos.copy()
        self._smooth_vel[name] = v_smooth
        return v_smooth

    def reset(self):
        self._prev_pos.clear()
        self._smooth_vel.clear()


@dataclass
class GraphData:
    """Lightweight graph container — convert to torch_geometric.Data when needed."""
    x: torch.Tensor           # (NUM_NODES, NODE_FEATURE_DIM)
    edge_index: torch.Tensor  # (2, num_edges), dtype long
    edge_attr: torch.Tensor   # (num_edges, EDGE_FEATURE_DIM)


def build_graph(entities: Sequence[EntityObs]) -> GraphData:
    """Pack 4 entity observations into the scene graph.

    Entities must be in the canonical order:
        [left_ee, right_ee, tray_left, tray_right]

    The quaternion slot (features 3:7) is zero-padded for any entity whose
    orientation is None. Edges are fully-connected directed with no
    self-loops, yielding NUM_NODES * (NUM_NODES - 1) = 12 edges.
    """
    if len(entities) != NUM_NODES:
        raise ValueError(f"Expected {NUM_NODES} entities, got {len(entities)}")

    x = torch.zeros((NUM_NODES, NODE_FEATURE_DIM), dtype=torch.float32)
    for i, e in enumerate(entities):
        x[i, 0:3] = torch.from_numpy(np.asarray(e.position, dtype=np.float32))
        if e.orientation is not None:
            x[i, 3:7] = torch.from_numpy(np.asarray(e.orientation, dtype=np.float32))
        x[i, 7:10] = torch.from_numpy(np.asarray(e.velocity, dtype=np.float32))
        x[i, 10:12] = torch.from_numpy(np.asarray(e.node_type, dtype=np.float32))

    src_dst = [(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j]
    edge_index = torch.tensor(src_dst, dtype=torch.long).t().contiguous()

    positions = x[:, 0:3]
    velocities = x[:, 7:10]
    num_edges = edge_index.shape[1]
    edge_attr = torch.zeros((num_edges, EDGE_FEATURE_DIM), dtype=torch.float32)
    for k, (i, j) in enumerate(src_dst):
        rel_pos = positions[j] - positions[i]
        dist = torch.linalg.norm(rel_pos)
        contact = (dist < CONTACT_THRESHOLD_M).float()
        rel_vel = velocities[j] - velocities[i]
        edge_attr[k, 0:3] = rel_pos
        edge_attr[k, 3] = dist
        edge_attr[k, 4] = contact
        edge_attr[k, 5:8] = rel_vel

    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
