"""
GNN/train.py — behavioral cloning training loop for the tray-lift GNN.

Loads the per-episode ``.pt`` files produced by build_dataset.py, splits
off the last few episodes as a validation set, normalizes arm-joint
velocities per motor, and trains TrayLiftGNN on the joint MSE (arm
velocities) + BCE (gripper binary) objective from CLAUDE.md Phase 4c.

Usage:
    python GNN/train.py
    python GNN/train.py --epochs 300 --batch-size 128 --val-episodes 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GNN.policy import TrayLiftGNN, ARM_JOINTS


REPO_ROOT  = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "GNN" / "dataset"
CKPT_DIR    = REPO_ROOT / "GNN" / "checkpoints"


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

@dataclass
class Normalization:
    """Per-arm, per-motor mean/std for the 5 arm-joint velocities.

    Computed separately because left and right arms produce systematically
    opposite velocities on direction-dependent motors (pan, wrist_roll) —
    combining them zeros out the signal the decoder needs. Diagnostic
    (GNN/diagnose.py) on the previous single-stat version showed the
    right-arm predictions collapsing ~2x in magnitude.
    """
    mean_left:  torch.Tensor   # [5]
    std_left:   torch.Tensor   # [5]
    mean_right: torch.Tensor   # [5]
    std_right:  torch.Tensor   # [5]

    def apply_left(self, vel_5: torch.Tensor) -> torch.Tensor:
        return (vel_5 - self.mean_left) / self.std_left

    def apply_right(self, vel_5: torch.Tensor) -> torch.Tensor:
        return (vel_5 - self.mean_right) / self.std_right

    def to_dict(self) -> dict:
        return {
            "mean_left":  self.mean_left.tolist(),
            "std_left":   self.std_left.tolist(),
            "mean_right": self.mean_right.tolist(),
            "std_right":  self.std_right.tolist(),
        }


# Mirror augmentation: reflection across the YZ plane between the two
# arms. Under that mirror:
#   - left arm and right arm swap roles
#   - tray_left marker and tray_right marker swap
#   - x-components of positions, velocities, and edge rel-position/vel flip
#   - two motor axes are parallel to the mirror plane and their rotation
#     direction flips: shoulder_pan (vertical axis) and wrist_roll
#     (forearm axis). The other three joints rotate around axes
#     perpendicular to the mirror plane and stay unchanged.
# If deployment shows these flips guessed wrong, the fix is a single-line
# change here.
JOINT_MIRROR_SIGNS = torch.tensor(
    [-1.0, +1.0, +1.0, +1.0, -1.0], dtype=torch.float32
)

# Node feature layout indices (must match GNN/entities.py)
_POS_SLICE  = slice(0, 3)       # position xyz
_QUAT_SLICE = slice(3, 7)       # quaternion xyzw (arm nodes only)
_VEL_SLICE  = slice(7, 10)      # velocity xyz

# Edge feature layout indices (must match GNN/graph_builder.py)
_EDGE_RELPOS  = slice(0, 3)
_EDGE_RELVEL  = slice(5, 8)


def _mirror_node_features(x: torch.Tensor) -> torch.Tensor:
    """Swap arm nodes (0<->1) and tray nodes (2<->3), flip x of pos/vel,
    and flip the sign of the quaternion's x and w components so the
    reflected orientation stays a valid rotation in the mirrored frame.
    x shape: [..., 4, 12]."""
    y = x.clone()
    y = y[..., [1, 0, 3, 2], :]                      # role swap
    y[..., _POS_SLICE.start]     *= -1.0             # pos.x
    y[..., _VEL_SLICE.start]     *= -1.0             # vel.x
    # Mirror a quaternion (x, y, z, w) across the YZ plane: (-x, y, z, -w)
    y[..., _QUAT_SLICE.start]         *= -1.0        # qx
    y[..., _QUAT_SLICE.start + 3]     *= -1.0        # qw
    return y


def _mirror_edge_features(edge_attr: torch.Tensor) -> torch.Tensor:
    """Flip x-components of relative position and relative velocity in
    every edge. Edge reordering to preserve src/dst after the node
    swap is handled by _mirror_edge_reindex.
    edge_attr shape: [..., 12, 8]."""
    y = edge_attr.clone()
    y[..., _EDGE_RELPOS.start] *= -1.0
    y[..., _EDGE_RELVEL.start] *= -1.0
    return y


def _mirror_edge_reindex(num_nodes: int = 4) -> torch.Tensor:
    """Permutation of edge indices induced by the node swap [1,0,3,2].

    The canonical edge order in build_graph is all (i, j) with i != j in
    lexicographic order. After swapping nodes via perm p, edge (i, j)
    becomes edge (p[i], p[j]). We compute the permutation that maps
    each old edge index to its new edge index.
    """
    perm = [1, 0, 3, 2]
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_to_pos = {e: k for k, e in enumerate(edges)}
    new_order = [edge_to_pos[(perm[i], perm[j])] for (i, j) in edges]
    return torch.tensor(new_order, dtype=torch.long)


_EDGE_REINDEX = _mirror_edge_reindex()


def _mirror_sample(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the physical mirror to a single (already-batched-by-loader)
    sample dict. Swaps sides, transforms node/edge features, flips the
    direction-dependent joint velocities.
    """
    out: dict[str, torch.Tensor] = {}
    # Swap views AND mirror them. The mirrored left-view graph is
    # structurally a right-view graph (in right-base frame), so it goes
    # to the right-decoder input; and vice versa.
    x_L = _mirror_node_features(sample["x_right"])
    x_R = _mirror_node_features(sample["x_left"])
    e_L = _mirror_edge_features(sample["edge_attr_right"]).index_select(-2, _EDGE_REINDEX)
    e_R = _mirror_edge_features(sample["edge_attr_left"]).index_select(-2, _EDGE_REINDEX)
    out["x_left"], out["x_right"] = x_L, x_R
    out["edge_attr_left"], out["edge_attr_right"] = e_L, e_R

    # The mirrored left action is the right arm's action with direction-
    # dependent motors flipped. Grip state is unchanged under mirror.
    signs_full = torch.cat([JOINT_MIRROR_SIGNS, torch.tensor([1.0])])  # gripper pass-through
    out["action_left"]  = sample["action_right"] * signs_full
    out["action_right"] = sample["action_left"]  * signs_full
    out["grip_left"]    = sample["grip_right"]
    out["grip_right"]   = sample["grip_left"]
    return out


class TrayLiftDataset(Dataset):
    """Flat dataset: each item is one frame with both views + both actions.

    Supports two training-time augmentations:

    * ``mirror_prob`` — probability of applying the physical mirror
      (swap sides, flip x on positions/velocities/quaternions, flip
      pan/roll action signs, permute edges). Teaches genuine symmetry.
    * ``pos_jitter_m`` — Gaussian noise std applied to node-position
      slots (and corresponding edge rel_pos/distance/contact) so the
      model learns to tolerate small perception errors. CLAUDE.md
      Phase 4d recommends ~2 mm.
    """

    def __init__(
        self,
        episode_data: list[dict],
        mirror_prob: float = 0.0,
        pos_jitter_m: float = 0.0,
    ):
        cat = lambda key: torch.cat([d[key] for d in episode_data], dim=0)
        self.x_left           = cat("x_left")
        self.x_right          = cat("x_right")
        self.edge_attr_left   = cat("edge_attr_left")
        self.edge_attr_right  = cat("edge_attr_right")
        self.action_left      = cat("action_left")
        self.action_right     = cat("action_right")
        self.grip_left        = cat("gripper_target_left")
        self.grip_right       = cat("gripper_target_right")
        self.mirror_prob      = mirror_prob
        self.pos_jitter_m     = pos_jitter_m

    def __len__(self) -> int:
        return self.x_left.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = {
            "x_left":          self.x_left[i],
            "x_right":         self.x_right[i],
            "edge_attr_left":  self.edge_attr_left[i],
            "edge_attr_right": self.edge_attr_right[i],
            "action_left":     self.action_left[i],
            "action_right":    self.action_right[i],
            "grip_left":       self.grip_left[i],
            "grip_right":      self.grip_right[i],
        }
        if self.mirror_prob > 0.0 and torch.rand(1).item() < self.mirror_prob:
            sample = _mirror_sample(sample)
        if self.pos_jitter_m > 0.0:
            sample = _jitter_positions(sample, self.pos_jitter_m)
        return sample


# Fixed edge order (matches GNN/graph_builder.py): all (i, j) with i != j
# in lexicographic order.
_EDGES_IJ: list[tuple[int, int]] = [
    (i, j)
    for i in range(4)
    for j in range(4)
    if i != j
]
_SRC_IDX = torch.tensor([e[0] for e in _EDGES_IJ], dtype=torch.long)
_DST_IDX = torch.tensor([e[1] for e in _EDGES_IJ], dtype=torch.long)
_CONTACT_THRESHOLD_M = 0.03   # must match GNN/entities.py


def _jitter_positions(
    sample: dict[str, torch.Tensor], sigma_m: float,
) -> dict[str, torch.Tensor]:
    """Add independent Gaussian noise to every node's position and
    propagate the change into each edge's rel_pos, distance, and
    contact indicator so the features stay self-consistent.

    Velocities and rel_vel are NOT jittered — the velocity tracker's
    noise is a separate phenomenon and jittering it would bias the
    per-tick action labels we're trying to predict.
    """
    out = dict(sample)
    out["x_left"]          = sample["x_left"].clone()
    out["x_right"]         = sample["x_right"].clone()
    out["edge_attr_left"]  = sample["edge_attr_left"].clone()
    out["edge_attr_right"] = sample["edge_attr_right"].clone()

    for view_key, edge_key in (
        ("x_left",  "edge_attr_left"),
        ("x_right", "edge_attr_right"),
    ):
        x = out[view_key]                              # [4, 12]
        e = out[edge_key]                              # [12, 8]
        x[:, 0:3] = x[:, 0:3] + torch.randn(4, 3) * sigma_m

        # Vectorized edge rebuild from jittered positions.
        pos = x[:, 0:3]
        rel = pos.index_select(0, _DST_IDX) - pos.index_select(0, _SRC_IDX)  # [12, 3]
        d   = torch.linalg.norm(rel, dim=-1)                                 # [12]
        e[:, 0:3] = rel
        e[:, 3]   = d
        e[:, 4]   = (d < _CONTACT_THRESHOLD_M).float()

    return out


def _load_episode_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("episode_*.pt"))


def _load_episode(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def _filter_by_phase(eps: list[dict], phase: str) -> list[dict]:
    """Return copies of each episode restricted to frames where BOTH
    arms are in the named phase.

    Requires per-phase dataset labels (see build_dataset.py). Episodes
    that predate phase tracking are dropped with a warning.
    """
    from GNN.build_dataset import PHASE_TO_IDX
    if phase not in PHASE_TO_IDX:
        raise SystemExit(
            f"--phase {phase!r}: unknown phase. Options: "
            f"{sorted(PHASE_TO_IDX)}"
        )
    phase_idx = PHASE_TO_IDX[phase]

    out: list[dict] = []
    for ep in eps:
        if "phase_idx_left" not in ep or "phase_idx_right" not in ep:
            print(
                f"  [{ep['metadata']['episode_dir']}] no phase labels "
                f"in .pt — skipping (re-run build_dataset.py on "
                f"episodes recorded with phase tracking)"
            )
            continue
        mask = (
            (ep["phase_idx_left"]  == phase_idx)
            & (ep["phase_idx_right"] == phase_idx)
        )
        n_kept = int(mask.sum().item())
        if n_kept == 0:
            continue
        kept: dict = {
            k: (v[mask] if isinstance(v, torch.Tensor)
                and v.ndim >= 1 and v.shape[0] == mask.shape[0]
                else v)
            for k, v in ep.items()
            if k != "metadata"
        }
        kept["metadata"] = {
            **ep["metadata"],
            "phase_filter": phase,
            "n_frames_used": n_kept,
        }
        out.append(kept)
    return out


def _compute_normalization(
    train_eps: list[dict], subtract_mean: bool = False,
) -> Normalization:
    """Compute per-arm std (and optionally mean) for action normalization.

    By default, mean is zeroed out. This makes the decoder's lazy
    default ("predict 0 in normalized space") equal to "predict 0 raw
    velocity" — which is the correct answer at stationary states such
    as raised pre-grasp. With mean subtraction, the lazy default was
    "predict the per-arm mean", which meant the model had to actively
    learn to cancel the mean on every stationary frame and only saw
    a weak gradient signal to do so. Std-only normalization shifts
    that gradient pressure to the much more common non-stationary
    frames where the model actually does need to learn to produce
    motion.
    """
    act_L = torch.cat([d["action_left"]  for d in train_eps], dim=0)[:, :ARM_JOINTS]
    act_R = torch.cat([d["action_right"] for d in train_eps], dim=0)[:, :ARM_JOINTS]
    zeros = torch.zeros(ARM_JOINTS)
    return Normalization(
        mean_left  = act_L.mean(dim=0) if subtract_mean else zeros.clone(),
        std_left   = act_L.std(dim=0).clamp(min=1.0),
        mean_right = act_R.mean(dim=0) if subtract_mean else zeros.clone(),
        std_right  = act_R.std(dim=0).clamp(min=1.0),
    )


# ------------------------------------------------------------------
# Losses / metrics
# ------------------------------------------------------------------

def compute_losses(
    model: TrayLiftGNN,
    batch: dict[str, torch.Tensor],
    norm: Normalization,
    grip_weight: float,
    bce: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    for k, v in batch.items():
        batch[k] = v.to(device)

    out = model(
        batch["x_left"],  batch["edge_attr_left"],
        batch["x_right"], batch["edge_attr_right"],
    )

    act_L_norm = norm.apply_left(batch["action_left"][:, :ARM_JOINTS])
    act_R_norm = norm.apply_right(batch["action_right"][:, :ARM_JOINTS])

    mse_L = torch.mean((out["left"]["velocity"]  - act_L_norm) ** 2)
    mse_R = torch.mean((out["right"]["velocity"] - act_R_norm) ** 2)

    bce_L = bce(out["left"]["gripper_logit"],  batch["grip_left"])
    bce_R = bce(out["right"]["gripper_logit"], batch["grip_right"])

    arm_loss  = 0.5 * (mse_L + mse_R)
    grip_loss = 0.5 * (bce_L + bce_R)
    total     = arm_loss + grip_weight * grip_loss

    # Gripper accuracy for diagnostics
    with torch.no_grad():
        g_pred_L = (out["left"]["gripper_logit"]  > 0.0).float()
        g_pred_R = (out["right"]["gripper_logit"] > 0.0).float()
        grip_acc = 0.5 * (
            (g_pred_L == batch["grip_left"]).float().mean() +
            (g_pred_R == batch["grip_right"]).float().mean()
        )

    return {
        "total":     total,
        "arm_mse":   arm_loss.detach(),
        "grip_bce":  grip_loss.detach(),
        "grip_acc":  grip_acc,
    }


# ------------------------------------------------------------------
# Train / eval loops
# ------------------------------------------------------------------

def train_one_epoch(
    model: TrayLiftGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm: Normalization,
    grip_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    bce = nn.BCEWithLogitsLoss()

    agg = {"total": 0.0, "arm_mse": 0.0, "grip_bce": 0.0, "grip_acc": 0.0, "n": 0}
    for batch in loader:
        losses = compute_losses(model, batch, norm, grip_weight, bce, device)
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        b = batch["x_left"].shape[0]
        agg["n"]        += b
        agg["total"]    += losses["total"].item()   * b
        agg["arm_mse"]  += losses["arm_mse"].item() * b
        agg["grip_bce"] += losses["grip_bce"].item()* b
        agg["grip_acc"] += losses["grip_acc"].item()* b

    n = agg.pop("n")
    return {k: v / n for k, v in agg.items()}


@torch.no_grad()
def evaluate(
    model: TrayLiftGNN,
    loader: DataLoader,
    norm: Normalization,
    grip_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    agg = {"total": 0.0, "arm_mse": 0.0, "grip_bce": 0.0, "grip_acc": 0.0, "n": 0}
    for batch in loader:
        losses = compute_losses(model, batch, norm, grip_weight, bce, device)
        b = batch["x_left"].shape[0]
        agg["n"]        += b
        agg["total"]    += losses["total"].item()   * b
        agg["arm_mse"]  += losses["arm_mse"].item() * b
        agg["grip_bce"] += losses["grip_bce"].item()* b
        agg["grip_acc"] += losses["grip_acc"].item()* b

    n = agg.pop("n")
    return {k: v / n for k, v in agg.items()}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--ckpt-dir",    default=str(CKPT_DIR))
    parser.add_argument("--val-episodes", type=int, default=3,
                        help="Number of latest episodes to hold out for validation.")
    parser.add_argument("--epochs",      type=int, default=200)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grip-weight", type=float, default=1.0,
                        help="Weight of BCE gripper loss relative to arm MSE.")
    parser.add_argument("--mirror-prob", type=float, default=0.5,
                        help="Probability of applying physical mirror "
                             "augmentation per training sample (0 = off).")
    parser.add_argument("--pos-jitter-m", type=float, default=0.002,
                        help="Std of Gaussian noise added to node "
                             "positions at training time, in meters "
                             "(0 = off). CLAUDE.md Phase 4d recommends ~2mm.")
    parser.add_argument("--subtract-mean", action="store_true",
                        help="Subtract per-arm mean in action "
                             "normalization (legacy). Default is "
                             "std-only so stationary frames predict 0.")
    parser.add_argument("--phase", default=None,
                        help="If set, restrict training to frames where "
                             "both arms are in this phase (e.g. LIFTING). "
                             "Requires dataset built from episodes with "
                             "phase tracking.")
    parser.add_argument("--hidden",      type=int, default=64)
    parser.add_argument("--rounds",      type=int, default=3)
    parser.add_argument("--patience",    type=int, default=40,
                        help="Early-stop patience (epochs without val improvement).")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    ep_paths = _load_episode_files(Path(args.dataset_dir))
    if not ep_paths:
        raise SystemExit(f"No episode .pt files in {args.dataset_dir}.")
    if len(ep_paths) <= args.val_episodes:
        raise SystemExit(
            f"Only {len(ep_paths)} episodes — need more than "
            f"{args.val_episodes} (val) + some for training."
        )

    train_paths = ep_paths[:-args.val_episodes]
    val_paths   = ep_paths[-args.val_episodes:]
    print(f"Episodes: {len(train_paths)} train / {len(val_paths)} val")
    print("  val episodes:", [p.stem for p in val_paths])

    train_eps = [_load_episode(p) for p in train_paths]
    val_eps   = [_load_episode(p) for p in val_paths]

    if args.phase is not None:
        n_train_before = sum(e["x_left"].shape[0] for e in train_eps)
        n_val_before   = sum(e["x_left"].shape[0] for e in val_eps)
        train_eps = _filter_by_phase(train_eps, args.phase)
        val_eps   = _filter_by_phase(val_eps,   args.phase)
        if not train_eps or not val_eps:
            raise SystemExit(
                f"Phase filter '{args.phase}' left no frames after "
                f"filtering. Check that your dataset includes phase "
                f"labels and that the phase name is correct."
            )
        n_train_after = sum(e["x_left"].shape[0] for e in train_eps)
        n_val_after   = sum(e["x_left"].shape[0] for e in val_eps)
        print(
            f"Phase filter '{args.phase}': "
            f"train {n_train_before}->{n_train_after}, "
            f"val {n_val_before}->{n_val_after}"
        )

    norm = _compute_normalization(train_eps, subtract_mean=args.subtract_mean)
    mode = "mean+std" if args.subtract_mean else "std-only"
    print(f"Normalization mode: {mode}")
    print("Per-motor left  mean/std (raw-enc/s):",
          [round(m, 1) for m in norm.mean_left.tolist()],
          [round(s, 1) for s in norm.std_left.tolist()])
    print("Per-motor right mean/std (raw-enc/s):",
          [round(m, 1) for m in norm.mean_right.tolist()],
          [round(s, 1) for s in norm.std_right.tolist()])

    train_ds = TrayLiftDataset(
        train_eps,
        mirror_prob=args.mirror_prob,
        pos_jitter_m=args.pos_jitter_m,
    )
    val_ds   = TrayLiftDataset(val_eps, mirror_prob=0.0, pos_jitter_m=0.0)
    print(f"Frames: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, drop_last=False)

    # ---- Model ----
    model = TrayLiftGNN(hidden=args.hidden, n_rounds=args.rounds).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Training loop ----
    best_val = math.inf
    best_epoch = -1
    since_improved = 0
    history: list[dict] = []

    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, norm, args.grip_weight, device,
        )
        val_metrics = evaluate(
            model, val_loader, norm, args.grip_weight, device,
        )

        history.append({
            "epoch": ep,
            "train": train_metrics,
            "val":   val_metrics,
        })

        improved = val_metrics["total"] < best_val - 1e-5
        if improved:
            best_val   = val_metrics["total"]
            best_epoch = ep
            since_improved = 0
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "normalization": norm.to_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                ckpt_dir / "best.pt",
            )
        else:
            since_improved += 1

        marker = " *" if improved else ""
        print(
            f"ep {ep:3d}  "
            f"train tot={train_metrics['total']:.4f} "
            f"arm={train_metrics['arm_mse']:.4f} "
            f"grip_acc={train_metrics['grip_acc']:.3f}  |  "
            f"val tot={val_metrics['total']:.4f} "
            f"arm={val_metrics['arm_mse']:.4f} "
            f"grip_acc={val_metrics['grip_acc']:.3f}"
            f"{marker}"
        )

        if since_improved >= args.patience:
            print(f"Early stopping (no improvement in {args.patience} epochs).")
            break

    dt = time.time() - t_start
    print(f"\nBest val total: {best_val:.4f} at epoch {best_epoch} "
          f"({dt:.1f}s total, {dt/max(1,ep):.2f}s/epoch)")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {ckpt_dir/'best.pt'} and {ckpt_dir/'history.json'}")


if __name__ == "__main__":
    main()
