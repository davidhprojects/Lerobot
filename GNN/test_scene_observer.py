"""
Unit tests for GNN/scene_observer.py — no camera, no motors.

Run directly:
    python GNN/test_scene_observer.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from GNN.entities import TYPE_ARM, TYPE_TRAY
from GNN.scene_observer import SceneObserver
from Algorithmic.forward_kinematics import forward_kinematics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_transforms():
    return {"left": np.eye(4), "right": np.eye(4)}


def _aruco_poses(tray_left_xyz=(0.1, 0.1, 0.5), tray_right_xyz=(0.2, 0.1, 0.5),
                 include_tray_left=True, include_tray_right=True):
    """MarkerConfig defaults: tray_right=3, tray_left=5."""
    poses = {}
    if include_tray_left:
        T = np.eye(4); T[:3, 3] = tray_left_xyz
        poses[5] = T
    if include_tray_right:
        T = np.eye(4); T[:3, 3] = tray_right_xyz
        poses[3] = T
    return poses


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_returns_four_entities():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    assert len(ents) == 4


def test_canonical_entity_order():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    assert [e.name for e in ents] == ["left_ee", "right_ee", "tray_left", "tray_right"]


def test_type_one_hots():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    assert np.array_equal(ents[0].node_type, TYPE_ARM)
    assert np.array_equal(ents[1].node_type, TYPE_ARM)
    assert np.array_equal(ents[2].node_type, TYPE_TRAY)
    assert np.array_equal(ents[3].node_type, TYPE_TRAY)


def test_arm_orientation_is_quat():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    assert ents[0].orientation.shape == (4,)
    assert ents[1].orientation.shape == (4,)


def test_tray_orientation_is_none():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    assert ents[2].orientation is None
    assert ents[3].orientation is None


# ---------------------------------------------------------------------------
# Missing markers
# ---------------------------------------------------------------------------

def test_returns_none_when_tray_left_missing():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(include_tray_left=False), np.zeros(5), np.zeros(5))
    assert ents is None


def test_returns_none_when_tray_right_missing():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(include_tray_right=False), np.zeros(5), np.zeros(5))
    assert ents is None


def test_missing_base_transform_raises():
    try:
        SceneObserver({"left": np.eye(4)})
    except KeyError:
        return
    raise AssertionError("expected KeyError when 'right' is missing")


# ---------------------------------------------------------------------------
# Position / transform math
# ---------------------------------------------------------------------------

def test_tray_positions_match_marker_translations():
    obs = SceneObserver(_identity_transforms())
    poses = _aruco_poses(tray_left_xyz=(0.11, 0.12, 0.53),
                         tray_right_xyz=(0.22, 0.13, 0.54))
    ents = obs.observe(poses, np.zeros(5), np.zeros(5))
    np.testing.assert_allclose(ents[2].position, poses[5][:3, 3].astype(np.float32))
    np.testing.assert_allclose(ents[3].position, poses[3][:3, 3].astype(np.float32))


def test_ee_position_with_identity_base_matches_fk():
    """When T_cam_base = I, left_ee camera-frame pos == FK(joints) translation."""
    obs = SceneObserver(_identity_transforms())
    q = np.array([0.1, -0.2, 0.3, -0.1, 0.05])  # arbitrary valid angles
    fk_translation = forward_kinematics(q)[:3, 3].astype(np.float32)
    ents = obs.observe(_aruco_poses(), q, np.zeros(5))
    np.testing.assert_allclose(ents[0].position, fk_translation, atol=1e-5)


def test_base_transform_translation_applied():
    """Translating T_cam_base_left by (1,2,3) shifts left_ee by same amount."""
    T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]
    base = {"left": T, "right": np.eye(4)}
    obs = SceneObserver(base)
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    fk_t = forward_kinematics(np.zeros(5))[:3, 3]
    expected = (fk_t + np.array([1.0, 2.0, 3.0])).astype(np.float32)
    np.testing.assert_allclose(ents[0].position, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

def test_velocities_zero_on_first_frame():
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    for e in ents:
        np.testing.assert_allclose(e.velocity, 0.0)


def test_tray_velocity_after_motion():
    obs = SceneObserver(_identity_transforms(), dt=1/30)
    obs.observe(_aruco_poses(tray_left_xyz=(0.1, 0.1, 0.5)),
                np.zeros(5), np.zeros(5))
    ents = obs.observe(_aruco_poses(tray_left_xyz=(0.2, 0.1, 0.5)),
                       np.zeros(5), np.zeros(5))
    # +10cm in x over 1/30 s -> instantaneous 3 m/s -> EMA 0.3*3 = 0.9
    np.testing.assert_allclose(
        ents[2].velocity, np.array([0.9, 0.0, 0.0], dtype=np.float32), atol=1e-4
    )


def test_reset_clears_velocity_history():
    obs = SceneObserver(_identity_transforms(), dt=1/30)
    obs.observe(_aruco_poses(tray_left_xyz=(0.1, 0.1, 0.5)),
                np.zeros(5), np.zeros(5))
    obs.observe(_aruco_poses(tray_left_xyz=(0.2, 0.1, 0.5)),
                np.zeros(5), np.zeros(5))
    obs.reset()
    ents = obs.observe(_aruco_poses(tray_left_xyz=(0.3, 0.1, 0.5)),
                       np.zeros(5), np.zeros(5))
    for e in ents:
        np.testing.assert_allclose(e.velocity, 0.0)


# ---------------------------------------------------------------------------
# End-to-end sanity: observer output feeds build_graph cleanly
# ---------------------------------------------------------------------------

def test_output_is_valid_for_build_graph():
    from GNN.graph_builder import build_graph
    from GNN.entities import NUM_NODES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
    obs = SceneObserver(_identity_transforms())
    ents = obs.observe(_aruco_poses(), np.zeros(5), np.zeros(5))
    g = build_graph(ents)
    assert g.x.shape == (NUM_NODES, NODE_FEATURE_DIM)
    assert g.edge_index.shape == (2, NUM_NODES * (NUM_NODES - 1))
    assert g.edge_attr.shape == (NUM_NODES * (NUM_NODES - 1), EDGE_FEATURE_DIM)


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
