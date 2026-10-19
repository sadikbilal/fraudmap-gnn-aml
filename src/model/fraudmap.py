"""
FraudMap - Hybrid GraphSAGE + GAT Model

The core GNN model combining:
1. GraphSAGE for scalable neighborhood aggregation (inductive learning)
2. GAT for attention-based edge importance weighting
3. Classification head with BatchNorm and Dropout

Architecture follows the project report specification:
- 3-layer GraphSAGE (neighbor sampling: 25→10→5, max pooling)
- multi-head GAT attention block (8 heads by default)
- FC(256) → BN → Dropout(0.3) → FC(128) → BN → Dropout(0.3) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm


class FraudMapGNN(nn.Module):
    """Hybrid GraphSAGE + GAT model for fraud detection.

    The model operates in two stages:
    1. GraphSAGE captures global structural patterns via neighbor sampling
    2. GAT refines representations using learned attention weights

    The outputs of both stages are concatenated and fed through a
    classification head that produces a fraud probability score [0, 1].

    Args:
        in_channels: Input feature dimension (default: 24 per report spec).
        hidden_channels: Hidden layer dimension (default: 256).
        sage_layers: Number of GraphSAGE layers (default: 3).
        gat_heads: Number of GAT attention heads (default: 8).
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 24,
        hidden_channels: int = 256,
        sage_layers: int = 3,
        gat_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout

        # ===== Stage 1: GraphSAGE Block =====
        # 3 layers with max pooling aggregation
        self.sage_convs = nn.ModuleList()
        self.sage_bns = nn.ModuleList()

        # First SAGE layer: in_channels → hidden_channels
        self.sage_convs.append(SAGEConv(in_channels, hidden_channels, aggr="max"))
        self.sage_bns.append(BatchNorm(hidden_channels))

        # Subsequent SAGE layers: hidden_channels → hidden_channels
        for _ in range(sage_layers - 1):
            self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="max"))
            self.sage_bns.append(BatchNorm(hidden_channels))

        # ===== Stage 2: GAT Block =====
        # Multi-head attention mechanism
        # Input: hidden_channels from SAGE
        # Output: hidden_channels (gat_heads * head_dim)
        gat_head_dim = hidden_channels // gat_heads
        self.gat_conv = GATConv(
            in_channels=hidden_channels,
            out_channels=gat_head_dim,
            heads=gat_heads,
            dropout=dropout,
            concat=True,  # Concatenate heads → output = gat_heads * gat_head_dim
        )
        self.gat_bn = BatchNorm(gat_heads * gat_head_dim)

        # ===== Stage 3: Classification Head =====
        # Concatenation of SAGE + GAT outputs
        combined_dim = hidden_channels + (gat_heads * gat_head_dim)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node feature matrix [num_nodes, in_channels].
            edge_index: Edge index [2, num_edges].
            return_attention: If True, also return GAT attention weights.

        Returns:
            Fraud probability scores [num_nodes, 1] (sigmoid output).
            If return_attention=True, also returns attention weights.
        """
        # --- Stage 1: GraphSAGE ---
        h_sage = x
        for conv, bn in zip(self.sage_convs, self.sage_bns):
            h_sage = conv(h_sage, edge_index)
            h_sage = bn(h_sage)
            h_sage = F.relu(h_sage)
            h_sage = F.dropout(h_sage, p=self.dropout, training=self.training)

        # --- Stage 2: GAT ---
        if return_attention:
            h_gat, (attn_edge_index, attn_weights) = self.gat_conv(
                h_sage, edge_index, return_attention_weights=True
            )
        else:
            h_gat = self.gat_conv(h_sage, edge_index)

        h_gat = self.gat_bn(h_gat)
        h_gat = F.elu(h_gat)
        h_gat = F.dropout(h_gat, p=self.dropout, training=self.training)

        # --- Stage 3: Concatenate + Classify ---
        h_combined = torch.cat([h_sage, h_gat], dim=1)
        out = self.classifier(h_combined)
        out = torch.sigmoid(out)

        if return_attention:
            return out, attn_weights

        return out

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings before the classification head.

        Useful for visualization (t-SNE/UMAP) and downstream tasks.

        Args:
            x: Node feature matrix.
            edge_index: Edge index.

        Returns:
            Node embeddings [num_nodes, combined_dim].
        """
        self.eval()
        with torch.no_grad():
            h_sage = x
            for conv, bn in zip(self.sage_convs, self.sage_bns):
                h_sage = conv(h_sage, edge_index)
                h_sage = bn(h_sage)
                h_sage = F.relu(h_sage)

            h_gat = self.gat_conv(h_sage, edge_index)
            h_gat = self.gat_bn(h_gat)
            h_gat = F.elu(h_gat)

            return torch.cat([h_sage, h_gat], dim=1)
