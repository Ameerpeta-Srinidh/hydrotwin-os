"""
HydroTwin OS — Plane 1: GNN Thermal Model

PyTorch Geometric graph neural network for thermal simulation.
Models heat propagation through the data center's physical asset graph
using Graph Attention Networks (GATv2) with edge-conditioned message passing.

The GNN predicts steady-state and transient temperature, flow rate,
and pressure at every node given boundary conditions (IT loads, ambient
temperature, cooling settings).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hydrotwin.physics.graph_models import AssetType, AssetNode, AssetEdge, EdgeType

logger = logging.getLogger(__name__)


# ─────────────────────── Asset Type Encoding ───────────────────────

_ASSET_TYPE_TO_IDX: dict[AssetType, int] = {
    AssetType.RACK: 0,
    AssetType.PIPE: 1,
    AssetType.PUMP: 2,
    AssetType.CRAH: 3,
    AssetType.SENSOR: 4,
    AssetType.COOLING_TOWER: 5,
    AssetType.COLUMN: 6,
    AssetType.WALL: 7,
    AssetType.ZONE: 8,
}
_NUM_ASSET_TYPES = len(_ASSET_TYPE_TO_IDX)

_EDGE_TYPE_TO_IDX: dict[EdgeType, int] = {
    EdgeType.THERMAL: 0,
    EdgeType.HYDRAULIC: 1,
    EdgeType.ELECTRICAL: 2,
    EdgeType.STRUCTURAL: 3,
    EdgeType.PROXIMITY: 4,
}
_NUM_EDGE_TYPES = len(_EDGE_TYPE_TO_IDX)


# ─────────────────────── Graph Conversion ───────────────────────

def nodes_to_features(nodes: list[AssetNode]) -> torch.Tensor:
    """
    Convert asset nodes to a feature tensor.

    Node features (13-dim):
        [0:9]  asset_type one-hot (9 types)
        [9]    temperature_c (normalized)
        [10]   x position (normalized)
        [11]   y position (normalized)
        [12]   capacity/load metric (type-specific, normalized)
    """
    features = []
    for node in nodes:
        # One-hot asset type
        type_vec = [0.0] * _NUM_ASSET_TYPES
        idx = _ASSET_TYPE_TO_IDX.get(node.asset_type, 0)
        type_vec[idx] = 1.0

        # Temperature (normalized: 0-50°C → 0-1)
        temp_norm = node.temperature_c / 50.0

        # Position (normalized assuming 0-100m range)
        x_norm = node.position.x / 100.0
        y_norm = node.position.y / 100.0

        # Capacity metric (type-dependent)
        capacity = 0.0
        if hasattr(node, "current_load_kw"):
            capacity = getattr(node, "current_load_kw", 0) / 20.0
        elif hasattr(node, "current_cooling_kw"):
            capacity = getattr(node, "current_cooling_kw", 0) / 200.0
        elif hasattr(node, "current_flow_lpm"):
            capacity = getattr(node, "current_flow_lpm", 0) / 1000.0
        elif hasattr(node, "current_value"):
            capacity = getattr(node, "current_value", 0) / 50.0

        feat = type_vec + [temp_norm, x_norm, y_norm, capacity]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


def edges_to_index_and_features(
    edges: list[AssetEdge],
    node_id_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert asset edges to edge index and edge feature tensors.

    Edge features (7-dim):
        [0:5]  edge_type one-hot (5 types)
        [5]    distance_m (normalized)
        [6]    thermal_conductivity (normalized)
    """
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 7, dtype=torch.float32)

    src_indices = []
    tgt_indices = []
    features = []

    for edge in edges:
        src_idx = node_id_to_idx.get(edge.source_id)
        tgt_idx = node_id_to_idx.get(edge.target_id)
        if src_idx is None or tgt_idx is None:
            continue

        # Bidirectional
        for s, t in [(src_idx, tgt_idx), (tgt_idx, src_idx)]:
            src_indices.append(s)
            tgt_indices.append(t)

            type_vec = [0.0] * _NUM_EDGE_TYPES
            eidx = _EDGE_TYPE_TO_IDX.get(edge.edge_type, 0)
            type_vec[eidx] = 1.0

            dist_norm = min(edge.distance_m / 20.0, 1.0)
            cond_norm = min(edge.thermal_conductivity / 10.0, 1.0)

            features.append(type_vec + [dist_norm, cond_norm])

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
    edge_attr = torch.tensor(features, dtype=torch.float32)

    return edge_index, edge_attr


def graph_to_tensors(
    nodes: list[AssetNode],
    edges: list[AssetEdge],
) -> dict[str, torch.Tensor | dict[str, int]]:
    """Convert full graph data to PyTorch tensors ready for the GNN."""
    node_id_to_idx = {node.id: i for i, node in enumerate(nodes)}
    x = nodes_to_features(nodes)
    edge_index, edge_attr = edges_to_index_and_features(edges, node_id_to_idx)

    # Target: current temperatures at each node
    y = torch.tensor([node.temperature_c for node in nodes], dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
        "node_id_to_idx": node_id_to_idx,
        "num_nodes": len(nodes),
        "num_edges": edge_index.shape[1] if edge_index.numel() > 0 else 0,
    }


# ─────────────────────── GNN Model ───────────────────────

class EdgeConditionedConv(nn.Module):
    """
    Edge-conditioned graph convolution layer.

    Messages are computed as: m_ij = MLP([h_j || e_ij])
    where h_j is the neighbor's features and e_ij is the edge features.
    Aggregation via attention-weighted sum.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels

        # Message MLP: transforms [neighbor_feat || edge_feat] → message
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels * heads),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels * heads, out_channels * heads),
        )

        # Attention
        self.attention = nn.Linear(out_channels * heads + in_channels, heads)

        # Update
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Updated node features [N, out_channels]
        """
        if edge_index.numel() == 0:
            # No edges — just transform node features
            return self.update_mlp(torch.cat([x, torch.zeros(x.size(0), self.out_channels, device=x.device)], dim=-1))

        src, tgt = edge_index[0], edge_index[1]

        # Compute messages
        neighbor_feats = x[src]  # [E, in_channels]
        msg_input = torch.cat([neighbor_feats, edge_attr], dim=-1)  # [E, in_channels + edge_dim]
        messages = self.message_mlp(msg_input)  # [E, out_channels * heads]

        # Compute attention weights
        attn_input = torch.cat([messages, x[tgt]], dim=-1)
        attn_weights = self.attention(attn_input)  # [E, heads]
        attn_weights = F.softmax(attn_weights, dim=0)  # normalize across edges per target

        # Weighted messages → mean over heads
        messages = messages.view(-1, self.heads, self.out_channels)  # [E, heads, out]
        attn_weights = attn_weights.unsqueeze(-1)  # [E, heads, 1]
        weighted = (messages * attn_weights).mean(dim=1)  # [E, out]

        # Aggregate via scatter (sum messages to target nodes)
        aggregated = torch.zeros(x.size(0), self.out_channels, device=x.device)
        aggregated.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.out_channels), weighted)

        # Update
        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return updated


class ThermalGNN(nn.Module):
    """
    Graph Neural Network for data center thermal simulation.

    Takes the asset graph (node features + edge features) and predicts:
        - Temperature at each node
        - Flow rate through each edge
        - Facility-level metrics (total cooling load, avg temp)

    Architecture:
        3 layers of EdgeConditionedConv → node prediction head + global pooling
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        edge_feature_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Message-passing layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                EdgeConditionedConv(hidden_dim, hidden_dim, edge_feature_dim, heads=heads)
            )

        self.dropout = nn.Dropout(dropout)

        # Node-level prediction head: temperature
        self.temp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Global pooling → facility metrics
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # [total_cooling_kw, avg_temp, max_temp, min_temp]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dict with:
                'node_temps': [N, 1] predicted temperatures
                'node_embeddings': [N, hidden] learned node representations
                'global_metrics': [4] facility-level metrics
        """
        h = self.input_proj(x)

        # Message passing
        for conv in self.conv_layers:
            h_new = conv(h, edge_index, edge_attr)
            h = h + h_new  # residual connection
            h = self.dropout(h)

        # Node temperature predictions
        node_temps = self.temp_head(h)

        # Global pooling (mean over all nodes)
        global_emb = h.mean(dim=0, keepdim=True)
        global_metrics = self.global_head(global_emb).squeeze(0)

        return {
            "node_temps": node_temps.squeeze(-1),
            "node_embeddings": h,
            "global_metrics": global_metrics,
        }

    def predict_temperatures(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> np.ndarray:
        """Convenience method returning numpy temperature predictions."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x, edge_index, edge_attr)
        return out["node_temps"].cpu().numpy()
