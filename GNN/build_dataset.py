"""
GNN/build_dataset.py — turn recorded episodes into training tensors.

Reads ``data_collection/episodes/episode_NNN/`` (produced by
run_collect.py) and writes one ``.pt`` file per episode under
``GNN/dataset/``.

Each file contains the tensors needed for behavioral cloning:

    x_left, x_right            [N, 4, 12]  node features (base frames)
    edge_attr_left, ...        [N, 12, 8]  edge features per frame
    edge_index                 [2, 12]     shared graph topology
    action_left, action_right  [N, 6]      joint velocity labels
                                           (raw encoder units / second)
    gripper_target_left,       [N]         binary gripper target at t+1
    gripper_target_right                   (1=closed, 0=open)
    metadata                   dict        fps, frame counts, drops,
                                           gripper midpoints

Each frame is processed from TWO perspectives — once with
``SceneObserver("left", ...)`` and once with ``SceneObserver("right",
...)`` — so each decoder head can be trained on graphs in the base
frame it'll actually see at inference.

Frames with a missing tray marker (either ``SceneObserver`` returns
``None``) are skipped and counted in ``metadata["n_skipped"]``.  The
last frame of each episode is dropped because it has no next frame for
the finite-difference action label.

Usage:
    python GNN/build_dataset.py
    python GNN/build_dataset.py --episodes-dir data_collection/episodes \
                                --output-dir GNN/dataset
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GNN.graph_builder import build_graph
from GNN.scene_observer import SceneObserver
from algorithmic.calibrate import load_base_transforms
from perception.aruco import MarkerConfig
from run import GRASP_PROXIMITY_M as _GRASP_PROXIMITY_M


MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper",
]
FK_MOTORS = MOTOR_NAMES[:5]            # gripper excluded from FK
MOTOR_RESOLUTION = 4096                # STS3215 12-bit

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = REPO_ROOT / "calibrations"


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------

def _load_calibration(side: str) -> dict:
    with open(CALIB_DIR / f"{side}.json") as f:
        return json.load(f)


def _raw_to_rad(raw: float, cal_entry: dict) -> float:
    """Raw encoder value → radians, matching lerobot's normalize='degrees'."""
    mid = (cal_entry["range_min"] + cal_entry["range_max"]) / 2.0
    deg = (raw - mid) * 360.0 / MOTOR_RESOLUTION
    return float(np.deg2rad(deg))


def _read_joints_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps [N], joints [N, 6] in raw encoder units)."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        ts_col  = header.index("timestamp")
        cols    = [header.index(name) for name in MOTOR_NAMES]
        rows    = list(reader)

    n = len(rows)
    ts     = np.empty(n, dtype=np.float64)
    joints = np.empty((n, 6), dtype=np.float64)
    for i, row in enumerate(rows):
        ts[i] = float(row[ts_col])
        for j, c in enumerate(cols):
            joints[i, j] = float(row[c])
    return ts, joints


def _load_aruco_jsonl(path: Path) -> list[dict[int, np.ndarray]]:
    frames: list[dict[int, np.ndarray]] = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            markers = {
                int(mid): np.asarray(T, dtype=np.float64)
                for mid, T in entry["markers"].items()
            }
            frames.append(markers)
    return frames


def _load_states_csv(path: Path) -> tuple[list[str], list[str]]:
    """Load per-frame phase labels for both arms.

    Returns (left_states, right_states). If the file is missing (old
    episodes recorded before phase tracking was added) returns two
    empty lists — downstream code treats that as "phase unknown" and
    disables the gripper-based tray fallback.

    Phase names in the file are canonicalized through ``_PHASE_ALIASES``
    (see below) so legacy TRANSLATE_PREP frames are returned as
    TRANSLATE here — they're the same phase as far as the GNN is
    concerned.
    """
    if not path.exists():
        return [], []
    left_states, right_states = [], []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        l_idx = header.index("left_state")
        r_idx = header.index("right_state")
        for row in reader:
            left_states.append(_PHASE_ALIASES.get(row[l_idx], row[l_idx]))
            right_states.append(_PHASE_ALIASES.get(row[r_idx], row[r_idx]))
    return left_states, right_states


# Phases in which the gripper is physically clamped to the tray. Any
# tray-marker loss during these phases can be filled in from the
# gripper marker (which is larger and essentially never lost).
_GRIPPER_CLOSED_PHASES = {
    "LIFT_WAITING", "LIFTING",
    "TRANSLATE",
    "LOWER",
}

# Known phase vocabulary, in canonical order. Used for integer
# encoding so train.py can filter by phase index.
PHASE_NAMES = [
    "INIT", "HOME", "PRE_GRASP",
    "HOVER", "APPROACH",
    "LIFT_WAITING", "LIFTING",
    "TRANSLATE",
    "LOWER", "RELEASE", "DONE",
]
PHASE_TO_IDX = {name: i for i, name in enumerate(PHASE_NAMES)}
PHASE_UNKNOWN = -1

# Legacy / bundled phase-name aliases. When a states.csv row contains a
# name in this dict, it is treated as the mapped canonical phase.
# TRANSLATE_PREP is folded into TRANSLATE because the two motions are
# functionally identical and the GNN handles them as one phase.
_PHASE_ALIASES: dict[str, str] = {
    "TRANSLATE_PREP": "TRANSLATE",
}


def _phase_idx(name: str) -> int:
    name = _PHASE_ALIASES.get(name, name)
    return PHASE_TO_IDX.get(name, PHASE_UNKNOWN)


def _approx_tray_from_gripper(
    gripper_T: np.ndarray, grasp_proximity_m: float,
) -> np.ndarray:
    """Fabricate a tray-marker 4x4 from the gripper marker.

    Follows the user's heuristic: during closed-gripper phases the tray
    tag sits roughly ``GRASP_PROXIMITY_M`` offset from the gripper tag
    in the camera Y axis. The rotation is left at identity since we
    only use the translation portion downstream.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = gripper_T[:3, 3]
    T[1, 3] = T[1, 3] - grasp_proximity_m
    return T


def _fill_tray_from_gripper(
    aruco: dict[int, np.ndarray],
    marker_cfg,
    phase_left: str,
    phase_right: str,
    grasp_proximity_m: float,
) -> dict[int, np.ndarray]:
    """Return a possibly-augmented copy of ``aruco`` with synthetic
    tray markers filled in from the gripper markers when the phase
    label indicates the gripper is closed on the tray.

    Leaves the aruco dict untouched if phase is not gripper-closed,
    if either gripper marker is itself missing, or if phase labels
    are unavailable for this frame.
    """
    if (phase_left not in _GRIPPER_CLOSED_PHASES
            or phase_right not in _GRIPPER_CLOSED_PHASES):
        return aruco

    out = dict(aruco)
    if (marker_cfg.tray_left not in out
            and marker_cfg.left_gripper in out):
        out[marker_cfg.tray_left] = _approx_tray_from_gripper(
            out[marker_cfg.left_gripper], grasp_proximity_m,
        )
    if (marker_cfg.tray_right not in out
            and marker_cfg.right_gripper in out):
        out[marker_cfg.tray_right] = _approx_tray_from_gripper(
            out[marker_cfg.right_gripper], grasp_proximity_m,
        )
    return out


# ------------------------------------------------------------------
# Per-episode builder
# ------------------------------------------------------------------

def build_episode(
    ep_dir: Path,
    base_transforms: dict[str, np.ndarray],
    cal_left: dict,
    cal_right: dict,
) -> dict | None:
    """Process one episode into a dict of tensors, or None if empty."""
    required = ["metadata.json", "aruco.jsonl", "joints_left.csv", "joints_right.csv"]
    for name in required:
        if not (ep_dir / name).exists():
            print(f"  [{ep_dir.name}] missing {name} — skipping episode")
            return None

    with open(ep_dir / "metadata.json") as f:
        ep_meta = json.load(f)
    fps = float(ep_meta["fps"])
    dt  = 1.0 / fps

    aruco_frames       = _load_aruco_jsonl(ep_dir / "aruco.jsonl")
    _ts_L,   joints_L  = _read_joints_csv(ep_dir / "joints_left.csv")
    _ts_R,   joints_R  = _read_joints_csv(ep_dir / "joints_right.csv")
    phases_L, phases_R = _load_states_csv(ep_dir / "states.csv")
    phase_labels_available = bool(phases_L) and bool(phases_R)

    # All streams are written in lockstep, but guard against truncation.
    lengths = [len(aruco_frames), len(joints_L), len(joints_R)]
    if phase_labels_available:
        lengths.append(len(phases_L))
        lengths.append(len(phases_R))
    n = min(lengths)
    if n < 2:
        print(f"  [{ep_dir.name}] only {n} frames available — skipping")
        return None

    aruco_frames = aruco_frames[:n]
    joints_L     = joints_L[:n]
    joints_R     = joints_R[:n]
    if phase_labels_available:
        phases_L = phases_L[:n]
        phases_R = phases_R[:n]

    marker_cfg = MarkerConfig()
    grasp_proximity_m = _GRASP_PROXIMITY_M

    # Fresh observers per episode so VelocityTracker state from other
    # episodes doesn't leak into the first frames.
    obs_left  = SceneObserver("left",  base_transforms, dt=dt)
    obs_right = SceneObserver("right", base_transforms, dt=dt)

    x_left_frames:       list[torch.Tensor] = []
    x_right_frames:      list[torch.Tensor] = []
    edge_attr_left:      list[torch.Tensor] = []
    edge_attr_right:     list[torch.Tensor] = []
    action_left_frames:  list[torch.Tensor] = []
    action_right_frames: list[torch.Tensor] = []
    grip_target_left:    list[float] = []
    grip_target_right:   list[float] = []
    phase_idx_left:      list[int] = []
    phase_idx_right:     list[int] = []
    edge_index: torch.Tensor | None = None
    n_tray_filled = 0

    # Midpoint between each arm's full gripper range — anything below
    # this at t+1 is treated as "closed" (label 1), above as "open" (0).
    grip_mid_L = 0.5 * (
        cal_left["gripper"]["range_min"] + cal_left["gripper"]["range_max"]
    )
    grip_mid_R = 0.5 * (
        cal_right["gripper"]["range_min"] + cal_right["gripper"]["range_max"]
    )

    skipped = 0

    for i in range(n - 1):          # last frame has no action label
        aruco = aruco_frames[i]
        phase_L = phases_L[i] if phase_labels_available else ""
        phase_R = phases_R[i] if phase_labels_available else ""

        # If tray markers are missing, try filling them in from the
        # gripper markers — only during phases where the gripper is
        # known to be clamped on the tray (see _GRIPPER_CLOSED_PHASES).
        if phase_labels_available:
            tl_present = marker_cfg.tray_left in aruco
            tr_present = marker_cfg.tray_right in aruco
            if not (tl_present and tr_present):
                aruco_before_fill = aruco
                aruco = _fill_tray_from_gripper(
                    aruco, marker_cfg, phase_L, phase_R, grasp_proximity_m,
                )
                if aruco is not aruco_before_fill:
                    n_tray_filled += 1

        q_L = np.array(
            [_raw_to_rad(joints_L[i, k], cal_left[FK_MOTORS[k]]) for k in range(5)]
        )
        q_R = np.array(
            [_raw_to_rad(joints_R[i, k], cal_right[FK_MOTORS[k]]) for k in range(5)]
        )

        ents_L = obs_left.observe(aruco,  q_L, q_R)
        ents_R = obs_right.observe(aruco, q_L, q_R)
        if ents_L is None or ents_R is None:
            skipped += 1
            continue

        g_L = build_graph(ents_L)
        g_R = build_graph(ents_R)
        if edge_index is None:
            edge_index = g_L.edge_index          # static across frames

        # Raw-encoder velocity per motor. Training code can rescale per-
        # motor if MSE needs re-weighting.
        action_L = (joints_L[i + 1] - joints_L[i]) * fps
        action_R = (joints_R[i + 1] - joints_R[i]) * fps

        x_left_frames.append(g_L.x)
        x_right_frames.append(g_R.x)
        edge_attr_left.append(g_L.edge_attr)
        edge_attr_right.append(g_R.edge_attr)
        action_left_frames.append(torch.from_numpy(action_L.astype(np.float32)))
        action_right_frames.append(torch.from_numpy(action_R.astype(np.float32)))

        # Binary gripper target from raw encoder value at t+1. Feetech
        # encoders decrease toward range_min when closed, so raw < mid
        # <=> closed (label 1).
        grip_target_left.append(1.0 if joints_L[i + 1, 5] < grip_mid_L else 0.0)
        grip_target_right.append(1.0 if joints_R[i + 1, 5] < grip_mid_R else 0.0)

        phase_idx_left.append(_phase_idx(phase_L))
        phase_idx_right.append(_phase_idx(phase_R))

    if not x_left_frames:
        print(f"  [{ep_dir.name}] no usable frames after marker-drop filter")
        return None

    return {
        "x_left":               torch.stack(x_left_frames),
        "x_right":              torch.stack(x_right_frames),
        "edge_attr_left":       torch.stack(edge_attr_left),
        "edge_attr_right":      torch.stack(edge_attr_right),
        "edge_index":           edge_index,
        "action_left":          torch.stack(action_left_frames),
        "action_right":         torch.stack(action_right_frames),
        "gripper_target_left":  torch.tensor(grip_target_left,  dtype=torch.float32),
        "gripper_target_right": torch.tensor(grip_target_right, dtype=torch.float32),
        "phase_idx_left":       torch.tensor(phase_idx_left,  dtype=torch.long),
        "phase_idx_right":      torch.tensor(phase_idx_right, dtype=torch.long),
        "metadata": {
            "fps": fps,
            "n_frames_raw": n,
            "n_frames_used": len(x_left_frames),
            "n_skipped": skipped,
            "n_tray_filled_from_gripper": n_tray_filled,
            "phase_labels_available": phase_labels_available,
            "episode_dir": ep_dir.name,
            "source_episode_num": ep_meta.get("episode"),
            "source_success": ep_meta.get("success"),
            "action_units": "raw_encoder_per_second",
            "motor_names": MOTOR_NAMES,
            "gripper_midpoint_left":  grip_mid_L,
            "gripper_midpoint_right": grip_mid_R,
            "phase_names": PHASE_NAMES,
        },
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build GNN training tensors from recorded episodes.",
    )
    parser.add_argument(
        "--episodes-dir",
        default=str(REPO_ROOT / "data_collection" / "episodes"),
        help="Directory containing episode_NNN/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "GNN" / "dataset"),
        help="Where to write the per-episode .pt files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild episodes even if the output .pt already exists.",
    )
    args = parser.parse_args()

    ep_root = Path(args.episodes_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ep_root.exists():
        raise SystemExit(f"Episodes dir not found: {ep_root}")

    base_transforms = load_base_transforms()
    if "left" not in base_transforms or "right" not in base_transforms:
        raise SystemExit(
            "base_transforms.json missing a side. Re-run algorithmic/calibrate.py."
        )
    cal_left  = _load_calibration("left")
    cal_right = _load_calibration("right")

    episode_dirs = sorted(
        d for d in ep_root.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    print(f"Found {len(episode_dirs)} episode directories under {ep_root}")

    n_written = 0
    total_used = 0
    total_skipped = 0
    total_filled = 0

    for ep_dir in episode_dirs:
        out_path = out_dir / f"{ep_dir.name}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"  {ep_dir.name}: already built, skipping ({out_path.name})")
            continue

        print(f"Processing {ep_dir.name}...")
        data = build_episode(ep_dir, base_transforms, cal_left, cal_right)
        if data is None:
            continue

        torch.save(data, out_path)
        meta = data["metadata"]
        n_written  += 1
        total_used += meta["n_frames_used"]
        total_skipped += meta["n_skipped"]
        total_filled += meta.get("n_tray_filled_from_gripper", 0)
        print(
            f"  -> {meta['n_frames_used']}/{meta['n_frames_raw']} frames "
            f"({meta['n_skipped']} dropped, "
            f"{meta.get('n_tray_filled_from_gripper', 0)} tray-fills) "
            f"-> {out_path.name}"
        )

    print(
        f"\nDone. Wrote {n_written} episode(s), {total_used} total training "
        f"frames, {total_skipped} dropped to marker loss, "
        f"{total_filled} frames with tray filled from gripper marker."
    )


if __name__ == "__main__":
    main()
