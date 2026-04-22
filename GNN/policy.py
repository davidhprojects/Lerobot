"""
GNN/policy.py — encode-process-decode policy for bimanual tray lifting.

Matches CLAUDE.md Phase 3a with one deviation per current project decision:
the gripper is **binary** (open/closed), not a continuous velocity. Each
arm's decoder therefore outputs 5 joint velocities + 1 gripper logit.

The graph is fixed-topology (4 nodes, 12 directed edges) so we skip
torch_geometric entirely and vectorize message-passing with a
precomputed [N, E] aggregation matrix and cheap gather-style indexing.

Usage:
    from GNN.policy import TrayLiftGNN
    model = TrayLiftGNN()
    pred = model.forward_side(x_left, edge_attr_left, side="left")
    # pred["velocity"]: [B, 5]  pred["gripper_logit"]: [B]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GNN.entities import (
    NUM_NODES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM,
    LEFT_EE, RIGHT_EE,
)


ARM_JOINTS = 5       # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll


# ------------------------------------------------------------------
# Graph topology (precomputed, fixed)
# ------------------------------------------------------------------

def _default_edge_index() -> torch.Tensor:
    """Fully-connected directed edges (no self-loops) as [2, E]."""
    src, dst = [], []
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _mean_aggregate_matrix(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """[N, E] matrix A where (A @ edge_feats) gives per-node mean of incoming edges."""
    _, dst = edge_index[0], edge_index[1]
    n_edges = edge_index.shape[1]
    A = torch.zeros(n_nodes, n_edges, dtype=torch.float32)
    for k, d in enumerate(dst.tolist()):
        A[d, k] = 1.0
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A


# ------------------------------------------------------------------
# One round of message-passing
# ------------------------------------------------------------------

class EdgeProcessor(nn.Module):
    """Edge update then node update, with mean-aggregation over incoming edges.

    Input / output shapes:
        x         [B, N, H]     node embeddings
        edge_attr [B, E, H]     edge embeddings
        returns   (x', edge_attr')  both in the same shapes
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        agg_mat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_src = x.index_select(1, src_idx)       # [B, E, H]
        x_dst = x.index_select(1, dst_idx)       # [B, E, H]
        new_edge = self.edge_mlp(torch.cat([x_src, x_dst, edge_attr], dim=-1))

        # Per-destination mean aggregation via precomputed [N, E] matrix.
        agg = torch.einsum("ne,beh->bnh", agg_mat, new_edge)
        new_node = self.node_mlp(torch.cat([x, agg], dim=-1))
        return new_node, new_edge


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------

class TrayLiftGNN(nn.Module):
    """Encode-process-decode policy per CLAUDE.md Phase 3."""

    def __init__(
        self,
        node_in: int = NODE_FEATURE_DIM,
        edge_in: int = EDGE_FEATURE_DIM,
        hidden: int = 64,
        n_rounds: int = 3,
        arm_joints: int = ARM_JOINTS,
        decoder_hidden: int = 32,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.arm_joints = arm_joints

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden), nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(),
        )
        self.processors = nn.ModuleList(
            [EdgeProcessor(hidden) for _ in range(n_rounds)]
        )

        # Per-arm decoder heads: shared encoder/processor, separate heads
        # so each arm can specialize its output style while still sharing
        # relational reasoning.
        def _decoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden, decoder_hidden), nn.ReLU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden, arm_joints + 1),  # 5 velocity + 1 grip logit
            )
        self.left_decoder  = _decoder()
        self.right_decoder = _decoder()

        # Fixed graph topology — registered as buffers so .to(device) moves them.
        edge_index = _default_edge_index()
        self.register_buffer("src_idx", edge_index[0])
        self.register_buffer("dst_idx", edge_index[1])
        self.register_buffer("agg_mat", _mean_aggregate_matrix(edge_index, NUM_NODES))

    # -- encode + process shared across both sides --

    def _encode_process(
        self, x: torch.Tensor, edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Returns final per-node embeddings [B, N, H]."""
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for p in self.processors:
            h, e = p(h, e, self.src_idx, self.dst_idx, self.agg_mat)
        return h

    # -- per-side decode --

    def forward_side(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        side: str,
    ) -> dict[str, torch.Tensor]:
        """
        Run the GNN once and decode for ``side``.

        x, edge_attr are in ``side``'s base frame (the observer that
        produced them). The appropriate decoder reads its own EE node
        from the processed graph.
        """
        h = self._encode_process(x, edge_attr)
        if side == "left":
            own = h[:, LEFT_EE, :]
            out = self.left_decoder(own)
        elif side == "right":
            own = h[:, RIGHT_EE, :]
            out = self.right_decoder(own)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        return {
            "velocity":      out[:, : self.arm_joints],   # [B, 5]
            "gripper_logit": out[:, self.arm_joints],     # [B]
        }

    # -- convenience: run both decoders on their respective views --

    def forward(
        self,
        x_left: torch.Tensor,
        edge_attr_left: torch.Tensor,
        x_right: torch.Tensor,
        edge_attr_right: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "left":  self.forward_side(x_left,  edge_attr_left,  "left"),
            "right": self.forward_side(x_right, edge_attr_right, "right"),
        }


# ------------------------------------------------------------------
# Param counter (useful for sanity)
# ------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TrayLiftGNN()
    print(f"TrayLiftGNN parameters: {count_parameters(m):,}")
    # Smoke test
    B = 3
    xL = torch.randn(B, NUM_NODES, NODE_FEATURE_DIM)
    eL = torch.randn(B, 12, EDGE_FEATURE_DIM)
    xR = torch.randn(B, NUM_NODES, NODE_FEATURE_DIM)
    eR = torch.randn(B, 12, EDGE_FEATURE_DIM)
    out = m(xL, eL, xR, eR)
    print("left  velocity:", out["left"]["velocity"].shape,
          "  gripper:", out["left"]["gripper_logit"].shape)
    print("right velocity:", out["right"]["velocity"].shape,
          "  gripper:", out["right"]["gripper_logit"].shape)
