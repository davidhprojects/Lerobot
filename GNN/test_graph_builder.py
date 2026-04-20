"""
Unit tests for GNN/graph_builder.py.

Pure-Python tests — no hardware, no torch_geometric. Run directly:

    python GNN/test_graph_builder.py

or with pytest:

    pytest GNN/test_graph_builder.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from GNN.entities import (
    EntityObs, TYPE_ARM, TYPE_TRAY,
    NUM_NODES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, CONTACT_THRESHOLD_M,
    rotation_matrix_to_quaternion,
)
from GNN.graph_builder import VelocityTracker, build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_entities(near_contact: bool = False) -> list[EntityObs]:
    """Four entities with an optional left_ee <-> tray_left contact."""
    left_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    right_pos = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    if near_contact:
        tray_left_pos = left_pos + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    else:
        tray_left_pos = np.array([0.1, 0.1, 0.5], dtype=np.float32)
    tray_right_pos = np.array([0.2, 0.1, 0.5], dtype=np.float32)

    quat_identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    zero_vel = np.zeros(3, dtype=np.float32)

    return [
        EntityObs("left_ee", left_pos, TYPE_ARM, zero_vel, quat_identity),
        EntityObs("right_ee", right_pos, TYPE_ARM, zero_vel, quat_identity),
        EntityObs("tray_left", tray_left_pos, TYPE_TRAY, zero_vel, None),
        EntityObs("tray_right", tray_right_pos, TYPE_TRAY, zero_vel, None),
    ]


def _edge_index_lookup(edge_index: torch.Tensor, src: int, dst: int) -> int:
    srcs, dsts = edge_index[0].tolist(), edge_index[1].tolist()
    return next(k for k, (s, d) in enumerate(zip(srcs, dsts)) if (s, d) == (src, dst))


# ---------------------------------------------------------------------------
# Shape / structure
# ---------------------------------------------------------------------------

def test_tensor_shapes():
    g = build_graph(_dummy_entities())
    assert g.x.shape == (NUM_NODES, NODE_FEATURE_DIM)
    assert g.edge_index.shape == (2, NUM_NODES * (NUM_NODES - 1))
    assert g.edge_attr.shape == (NUM_NODES * (NUM_NODES - 1), EDGE_FEATURE_DIM)


def test_edge_index_dtype_long():
    g = build_graph(_dummy_entities())
    assert g.edge_index.dtype == torch.long


def test_no_self_loops():
    g = build_graph(_dummy_entities())
    assert (g.edge_index[0] != g.edge_index[1]).all()


def test_fully_connected():
    g = build_graph(_dummy_entities())
    pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    expected = {(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j}
    assert pairs == expected


def test_wrong_entity_count_raises():
    try:
        build_graph(_dummy_entities()[:3])
    except ValueError:
        return
    raise AssertionError("expected ValueError for len(entities) != 4")


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def test_type_one_hot_matches_spec():
    g = build_graph(_dummy_entities())
    assert torch.allclose(g.x[0, 10:12], torch.tensor([1.0, 0.0]))  # left_ee = arm
    assert torch.allclose(g.x[1, 10:12], torch.tensor([1.0, 0.0]))  # right_ee = arm
    assert torch.allclose(g.x[2, 10:12], torch.tensor([0.0, 1.0]))  # tray_left
    assert torch.allclose(g.x[3, 10:12], torch.tensor([0.0, 1.0]))  # tray_right


def test_tray_orientation_is_zero_padded():
    g = build_graph(_dummy_entities())
    assert torch.all(g.x[2, 3:7] == 0.0)
    assert torch.all(g.x[3, 3:7] == 0.0)


def test_arm_orientation_preserved():
    g = build_graph(_dummy_entities())
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(g.x[0, 3:7], expected)
    assert torch.allclose(g.x[1, 3:7], expected)


def test_node_positions_round_trip():
    ents = _dummy_entities()
    g = build_graph(ents)
    for i, e in enumerate(ents):
        assert torch.allclose(g.x[i, 0:3], torch.from_numpy(e.position))


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def test_contact_indicator_fires_within_threshold():
    g = build_graph(_dummy_entities(near_contact=True))
    # Edge 0 -> 2 has separation ~1cm (< 3cm threshold)
    idx = _edge_index_lookup(g.edge_index, 0, 2)
    assert g.edge_attr[idx, 4].item() == 1.0


def test_contact_indicator_off_when_far():
    g = build_graph(_dummy_entities(near_contact=False))
    idx = _edge_index_lookup(g.edge_index, 0, 2)
    assert g.edge_attr[idx, 4].item() == 0.0


def test_rel_pos_antisymmetric():
    g = build_graph(_dummy_entities())
    ij = _edge_index_lookup(g.edge_index, 0, 1)
    ji = _edge_index_lookup(g.edge_index, 1, 0)
    assert torch.allclose(g.edge_attr[ij, 0:3], -g.edge_attr[ji, 0:3])


def test_distance_nonnegative():
    g = build_graph(_dummy_entities())
    assert (g.edge_attr[:, 3] >= 0).all()


def test_distance_matches_rel_pos_norm():
    g = build_graph(_dummy_entities())
    computed = torch.linalg.norm(g.edge_attr[:, 0:3], dim=1)
    assert torch.allclose(g.edge_attr[:, 3], computed)


# ---------------------------------------------------------------------------
# Velocity tracker
# ---------------------------------------------------------------------------

def test_velocity_tracker_first_update_returns_zero():
    vt = VelocityTracker()
    v = vt.update("x", np.array([0.1, 0.0, 0.0], dtype=np.float32), dt=1/30)
    assert np.allclose(v, 0.0)


def test_velocity_tracker_ema_first_step():
    vt = VelocityTracker(alpha=0.3)
    vt.update("x", np.array([0.0, 0.0, 0.0], dtype=np.float32), dt=1/30)
    # 10 cm in one 30 Hz frame -> v_instant = 3 m/s; EMA = 0.3*3 + 0.7*0 = 0.9
    v = vt.update("x", np.array([0.1, 0.0, 0.0], dtype=np.float32), dt=1/30)
    assert np.allclose(v, [0.9, 0.0, 0.0], atol=1e-5)


def test_velocity_tracker_stationary_converges_to_zero():
    vt = VelocityTracker(alpha=0.3)
    pos = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    for _ in range(30):
        v = vt.update("x", pos, dt=1/30)
    assert np.allclose(v, 0.0, atol=1e-5)


def test_velocity_tracker_reset_clears_state():
    vt = VelocityTracker()
    vt.update("x", np.zeros(3, dtype=np.float32), dt=1/30)
    vt.reset()
    v = vt.update("x", np.array([0.1, 0.0, 0.0], dtype=np.float32), dt=1/30)
    assert np.allclose(v, 0.0)  # first-update-after-reset behavior


def test_velocity_tracker_rejects_nonpositive_dt():
    vt = VelocityTracker()
    try:
        vt.update("x", np.zeros(3, dtype=np.float32), dt=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for dt <= 0")


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def test_rotation_matrix_to_quaternion_identity():
    q = rotation_matrix_to_quaternion(np.eye(3))
    # scipy returns xyzw; identity rotation -> (0, 0, 0, 1)
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
