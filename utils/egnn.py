#egnn.py
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from typing import Tuple

class DeepEGNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_attr_matrix: torch.Tensor, edge_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        device = x.device
        row, col = edge_indices

        edge_attr = edge_attr_matrix[:, row, col, :]
        h_row = h[:, row, :]
        h_col = h[:, col, :]

        edge_feat_input = torch.cat([h_row, h_col, edge_attr], dim=-1)
        edge_hidden = self.edge_mlp(edge_feat_input)
        attn_out, _ = self.attn(edge_hidden, edge_hidden, edge_hidden)
        edge_out = edge_hidden + attn_out

        coord_diff = x[:, row, :] - x[:, col, :]
        coord_scale = self.coord_mlp(edge_out)
        coord_update_j = coord_scale * coord_diff

        x_update = torch.zeros_like(x)
        col_expanded = col.unsqueeze(0).expand(B, -1)
        row_expanded = row.unsqueeze(0).expand(B, -1)
        x_update = x_update.scatter_add(1, col_expanded.unsqueeze(-1).expand(-1, -1, 3), coord_update_j)
        x_update = x_update.scatter_add(1, row_expanded.unsqueeze(-1).expand(-1, -1, 3), -coord_update_j)
        x_new = x + x_update

        node_agg = torch.zeros(B, N, edge_out.shape[-1], device=device)
        node_agg = node_agg.scatter_add(1, col_expanded.unsqueeze(-1).expand(-1, -1, edge_out.shape[-1]), edge_out)
        node_mlp_input = torch.cat([h, node_agg], dim=-1)
        h_update = self.node_mlp(node_mlp_input)
        h_new = h + self.residual_mlp(h_update)
        return x_new, h_new

class DeepEGNN(nn.Module):
    def __init__(self, num_layers: int = 6, node_dim: int = 256, edge_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.layers = nn.ModuleList([
            DeepEGNNLayer(node_dim, edge_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_attr_matrix: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, h = layer(x, h, edge_attr_matrix, edge_indices)
        return h