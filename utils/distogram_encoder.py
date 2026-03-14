# utils/distogram_encoder.py
import torch
import torch.nn as nn

class DistogramEncoder(nn.Module):
    """
    Processes distograms with 2D convolutions to generate edge features.
    Input distogram is expected to be (B, N, N).
    Output features can be used as edge attributes in a GNN like EGNN.
    """
    def __init__(self, num_atoms: int, output_dim: int):
        super().__init__()
        self.num_atoms = num_atoms
        # Simple example, can be made deeper or use different architectures
        self.conv_layers = nn.Sequential(
            # Input: (B, 1, N, N)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(16), # Use BatchNorm or LayerNorm
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, output_dim, kernel_size=3, padding=1),
            # Output: (B, output_dim, N, N)
        )
        # Optional: Add pooling or attention layers if needed

    def forward(self, distogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distogram (torch.Tensor): Pairwise distance matrix, shape (B, N, N).

        Returns:
            torch.Tensor: Encoded edge features, shape (B, N, N, output_dim).
        """
        if distogram.ndim != 3 or distogram.shape[-1] != distogram.shape[-2]:
             raise ValueError(f"Expected distogram shape (B, N, N), got {distogram.shape}")

        B, N, _ = distogram.shape
        x = distogram.unsqueeze(1)  # Add channel dimension: (B, 1, N, N)
        x = self.conv_layers(x)     # Shape: (B, output_dim, N, N)

        # Permute to get (B, N, N, output_dim) suitable for edge attributes
        edge_features = x.permute(0, 2, 3, 1)
        return edge_features