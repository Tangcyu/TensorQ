# utils/embedding.py
import math
import torch
import torch.nn as nn
from typing import Optional

class SinusoidalEmbedding(nn.Module):
    """ Sinusoidal time embedding module. """
    # ... (keep existing code) ...
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.

        Args:
            timesteps (Tensor): A tensor of timesteps, shape (B,).

        Returns:
            Tensor: Sinusoidal embeddings, shape (B, embedding_dim).
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        # Ensure arange is on the correct device
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        # Ensure timesteps are float for multiplication
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0) # Shape (B, half_dim)
        # Concatenate sin and cos components
        embeddings = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # Shape (B, embedding_dim)
        return embeddings


class AtomTypeEmbedding(nn.Module):
     """ Embeds atom type indices into vectors. """
     def __init__(self, num_atom_types: int, embedding_dim: int):
         super().__init__()
         self.embedding = nn.Embedding(num_atom_types, embedding_dim, padding_idx=None) # Add padding_idx if needed
         print(f"Initialized AtomTypeEmbedding with {num_atom_types} types and dim {embedding_dim}")


     def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
         """
         Args:
             atom_types (torch.Tensor): Integer atom type indices, shape (B, N).

         Returns:
             torch.Tensor: Embedded atom types, shape (B, N, embedding_dim).
         """
         return self.embedding(atom_types)