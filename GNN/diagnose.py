"""
GNN/diagnose.py — sanity-check the trained policy against its own data.

Loads a checkpoint and runs the model on every .pt file in the dataset
(both train AND val). For each arm, reports:

  - per-motor RMSE of predicted velocity vs ground truth
  - per-motor mean of predictions and targets (detects bias)
  - gripper accuracy (vs binary label)
  - worst-offending frames so you can eyeball them

Purpose: decide whether poor policy behavior is a TRAINING problem
(model can't reproduce its own training labels) or a DEPLOYMENT problem
(model tracks training fine but diverges at inference).

Usage:
    python GNN/diagnose.py
    python GNN/diagnose.py --checkpoint GNN/checkpoints/best.pt --n-episodes 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GNN.policy import TrayLiftGNN, ARM_JOINTS


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "GNN" / "dataset"
CKPT_PATH   = REPO_ROOT / "GNN" / "checkpoints" / "best.pt"

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",
]


def _load_model(ckpt_path: Path) -> tuple[TrayLiftGNN, dict]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    args = ckpt.get("args", {})
    model = TrayLiftGNN(
        hidden=int(args.get("hidden", 64)),
        n_rounds=int(args.get("rounds", 3)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def _predict_episode(
    model: TrayLiftGNN,
    ep_data: dict,
    norm: dict[str, torch.Tensor],
) -> dict:
    """Run both decoder heads across one episode; return predictions in raw units."""
    x_L = ep_data["x_left"]
    e_L = ep_data["edge_attr_left"]
    x_R = ep_data["x_right"]
    e_R = ep_data["edge_attr_right"]

    out_L = model.forward_side(x_L, e_L, "left")
    out_R = model.forward_side(x_R, e_R, "right")

    # Denormalize arm velocities back to raw-encoder-per-second using
    # each arm's own stats.
    pred_vel_L = out_L["velocity"] * norm["std_left"]  + norm["mean_left"]
    pred_vel_R = out_R["velocity"] * norm["std_right"] + norm["mean_right"]
    pred_grip_L = (out_L["gripper_logit"] > 0.0).float()
    pred_grip_R = (out_R["gripper_logit"] > 0.0).float()

    return {
        "pred_vel_L":  pred_vel_L.cpu().numpy(),
        "pred_vel_R":  pred_vel_R.cpu().numpy(),
        "pred_grip_L": pred_grip_L.cpu().numpy(),
        "pred_grip_R": pred_grip_R.cpu().numpy(),
    }


def _rmse_per_motor(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RMSE along axis 0. Shapes: [N, K] → [K]."""
    return np.sqrt(np.mean((pred - target) ** 2, axis=0))


def _format_row(label: str, values: list[float], width: int = 9) -> str:
    cells = "".join(f"{v:{width}.1f}" for v in values)
    return f"  {label:<22s}{cells}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT_PATH))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--n-episodes", type=int, default=0,
                        help="Limit to first N episodes (0 = all).")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    model, ckpt = _load_model(ckpt_path)
    n = ckpt["normalization"]
    if "mean_left" in n:
        norm = {
            "mean_left":  torch.tensor(n["mean_left"],  dtype=torch.float32),
            "std_left":   torch.tensor(n["std_left"],   dtype=torch.float32),
            "mean_right": torch.tensor(n["mean_right"], dtype=torch.float32),
            "std_right":  torch.tensor(n["std_right"],  dtype=torch.float32),
        }
    else:
        # Legacy single-stat checkpoint.
        m = torch.tensor(n["mean"], dtype=torch.float32)
        s = torch.tensor(n["std"],  dtype=torch.float32)
        norm = {"mean_left": m, "std_left": s, "mean_right": m, "std_right": s}
    print(f"  mean_left :  {norm['mean_left'].tolist()}")
    print(f"  mean_right:  {norm['mean_right'].tolist()}")
    print(f"  std_left  :  {norm['std_left'].tolist()}")
    print(f"  std_right :  {norm['std_right'].tolist()}")

    ds_dir = Path(args.dataset_dir)
    ep_paths = sorted(ds_dir.glob("episode_*.pt"))
    if args.n_episodes:
        ep_paths = ep_paths[: args.n_episodes]
    print(f"Evaluating {len(ep_paths)} episode(s)")

    # Accumulators for overall stats
    all_pred_L, all_pred_R = [], []
    all_targ_L, all_targ_R = [], []
    all_gpred_L, all_gpred_R = [], []
    all_gtarg_L, all_gtarg_R = [], []

    per_episode_rows: list[str] = []
    per_episode_rows.append(
        "\nPer-episode arm-velocity RMSE (raw-encoder / second):"
    )
    header = f"  {'episode':<22s}" + "".join(
        f"{m[:8]:>9s}" for m in MOTOR_NAMES
    )
    per_episode_rows.append(header)

    for p in ep_paths:
        ep = torch.load(p, weights_only=False)
        preds = _predict_episode(model, ep, norm)

        targ_L = ep["action_left"].cpu().numpy()[:, :ARM_JOINTS]
        targ_R = ep["action_right"].cpu().numpy()[:, :ARM_JOINTS]
        gtarg_L = ep["gripper_target_left"].cpu().numpy()
        gtarg_R = ep["gripper_target_right"].cpu().numpy()

        rmse_L = _rmse_per_motor(preds["pred_vel_L"], targ_L)
        rmse_R = _rmse_per_motor(preds["pred_vel_R"], targ_R)

        per_episode_rows.append(_format_row(f"{p.stem} L", rmse_L.tolist()))
        per_episode_rows.append(_format_row(f"{p.stem} R", rmse_R.tolist()))

        all_pred_L.append(preds["pred_vel_L"]);  all_targ_L.append(targ_L)
        all_pred_R.append(preds["pred_vel_R"]);  all_targ_R.append(targ_R)
        all_gpred_L.append(preds["pred_grip_L"]); all_gtarg_L.append(gtarg_L)
        all_gpred_R.append(preds["pred_grip_R"]); all_gtarg_R.append(gtarg_R)

    for row in per_episode_rows:
        print(row)

    # Aggregate across all episodes
    P_L = np.concatenate(all_pred_L); T_L = np.concatenate(all_targ_L)
    P_R = np.concatenate(all_pred_R); T_R = np.concatenate(all_targ_R)
    GP_L = np.concatenate(all_gpred_L); GT_L = np.concatenate(all_gtarg_L)
    GP_R = np.concatenate(all_gpred_R); GT_R = np.concatenate(all_gtarg_R)

    print("\n" + "=" * 80)
    print("AGGREGATE STATS (all episodes combined)")
    print("=" * 80)

    print(_format_row("RMSE left",  _rmse_per_motor(P_L, T_L).tolist()))
    print(_format_row("RMSE right", _rmse_per_motor(P_R, T_R).tolist()))

    print()
    print(_format_row("target mean left",  T_L.mean(axis=0).tolist()))
    print(_format_row("target mean right", T_R.mean(axis=0).tolist()))
    print(_format_row("pred mean left",    P_L.mean(axis=0).tolist()))
    print(_format_row("pred mean right",   P_R.mean(axis=0).tolist()))

    print()
    print(_format_row("target std left",  T_L.std(axis=0).tolist()))
    print(_format_row("target std right", T_R.std(axis=0).tolist()))
    print(_format_row("pred std left",    P_L.std(axis=0).tolist()))
    print(_format_row("pred std right",   P_R.std(axis=0).tolist()))

    # Per-motor correlation (Pearson) — tells us if pred follows target direction
    def _per_motor_corr(P: np.ndarray, T: np.ndarray) -> list[float]:
        corrs = []
        for k in range(P.shape[1]):
            p = P[:, k] - P[:, k].mean()
            t = T[:, k] - T[:, k].mean()
            denom = np.sqrt((p ** 2).sum() * (t ** 2).sum())
            corrs.append(float(p @ t / denom) if denom > 1e-8 else 0.0)
        return corrs

    print()
    print(_format_row("corr left",  _per_motor_corr(P_L, T_L)))
    print(_format_row("corr right", _per_motor_corr(P_R, T_R)))

    print()
    print(f"  gripper accuracy left:  {float(np.mean(GP_L == GT_L)):.4f}")
    print(f"  gripper accuracy right: {float(np.mean(GP_R == GT_R)):.4f}")

    # Classification-style breakdown for gripper
    print("\nGripper confusion (frame counts):")
    for side, gp, gt in [("left", GP_L, GT_L), ("right", GP_R, GT_R)]:
        tp = int(((gp == 1) & (gt == 1)).sum())
        tn = int(((gp == 0) & (gt == 0)).sum())
        fp = int(((gp == 1) & (gt == 0)).sum())
        fn = int(((gp == 0) & (gt == 1)).sum())
        print(
            f"  {side:>6s}:  "
            f"closed->closed={tp:5d}  open->open={tn:5d}  "
            f"open->closed={fp:4d}  closed->open={fn:4d}"
        )

    # ------------------------------------------------------------------
    # First-frame analysis: does the model predict reasonable actions at
    # the exact state the policy starts deployment from (raised pre-grasp,
    # fresh velocity tracker)? If predictions here are already far from
    # labels, compounding error won't be the whole story — the first
    # command is already wrong.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FIRST-FRAME ANALYSIS (state the policy starts from at deployment)")
    print("=" * 80)

    first_preds_L, first_preds_R = [], []
    first_targ_L,  first_targ_R  = [], []
    hdr = f"  {'episode':<22s}" + "".join(
        f"{m[:9]:>11s}" for m in MOTOR_NAMES
    )
    print(f"\nPer-episode (predicted minus target) on arm-velocity at frame 0:")
    print(hdr)

    for p in ep_paths:
        ep = torch.load(p, weights_only=False)
        x_L = ep["x_left"][:1]              # keep batch dim
        e_L = ep["edge_attr_left"][:1]
        x_R = ep["x_right"][:1]
        e_R = ep["edge_attr_right"][:1]

        with torch.no_grad():
            out_L = model.forward_side(x_L, e_L, "left")
            out_R = model.forward_side(x_R, e_R, "right")

        pred_vel_L = (out_L["velocity"] * norm["std_left"]  + norm["mean_left"]).numpy()[0]
        pred_vel_R = (out_R["velocity"] * norm["std_right"] + norm["mean_right"]).numpy()[0]
        targ_L = ep["action_left"].numpy()[0,  :ARM_JOINTS]
        targ_R = ep["action_right"].numpy()[0, :ARM_JOINTS]

        first_preds_L.append(pred_vel_L); first_targ_L.append(targ_L)
        first_preds_R.append(pred_vel_R); first_targ_R.append(targ_R)

        diff_L = (pred_vel_L - targ_L).tolist()
        diff_R = (pred_vel_R - targ_R).tolist()
        print(_format_row(f"{p.stem} L diff", diff_L, width=11))
        print(_format_row(f"{p.stem} R diff", diff_R, width=11))

    FP_L = np.stack(first_preds_L); FT_L = np.stack(first_targ_L)
    FP_R = np.stack(first_preds_R); FT_R = np.stack(first_targ_R)

    print("\nAggregate first-frame stats:")
    print(_format_row("first-frame pred L",  FP_L.mean(axis=0).tolist(), width=11))
    print(_format_row("first-frame tgt  L",  FT_L.mean(axis=0).tolist(), width=11))
    print(_format_row("first-frame pred R",  FP_R.mean(axis=0).tolist(), width=11))
    print(_format_row("first-frame tgt  R",  FT_R.mean(axis=0).tolist(), width=11))
    print()
    print(_format_row(
        "first-frame RMSE L",
        _rmse_per_motor(FP_L, FT_L).tolist(),
        width=11,
    ))
    print(_format_row(
        "first-frame RMSE R",
        _rmse_per_motor(FP_R, FT_R).tolist(),
        width=11,
    ))

    # How much does target itself vary across episodes at frame 0?
    # If target std >> prediction std, model is ignoring scene differences
    # (e.g. where the tray is) and predicting the mean initial action.
    print()
    print(_format_row("first-frame tgt  std L", FT_L.std(axis=0).tolist(), width=11))
    print(_format_row("first-frame pred std L", FP_L.std(axis=0).tolist(), width=11))
    print(_format_row("first-frame tgt  std R", FT_R.std(axis=0).tolist(), width=11))
    print(_format_row("first-frame pred std R", FP_R.std(axis=0).tolist(), width=11))


if __name__ == "__main__":
    main()
