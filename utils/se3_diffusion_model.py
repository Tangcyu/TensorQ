# utils/se3_diffusion_model.py
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch.cuda.amp import autocast
import math 
from .embedding import SinusoidalEmbedding 
from typing import List

def knn_graph_pytorch(x: torch.Tensor, k: int, batch: torch.Tensor = None, loop: bool = False, flow: str = 'source_to_target'):
    """
    Computes the k-Nearest Neighbor graph based on node positions `x`.
    Replicates torch_cluster.knn_graph functionality using torch.cdist.
    Handles batches correctly.

    Args:
        x (Tensor): Node feature matrix of shape [num_nodes_total, num_dimensions].
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector of shape [num_nodes_total], which
            assigns each node to a specific example. Required for batched input.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. Defaults to :obj:`False`.
        flow (string, optional): The flow direction of edges ('source_to_target'
             or 'target_to_source'). Defaults to 'source_to_target'.

    Returns:
        LongTensor: Edge list defining the graph, shape [2, num_edges].
    """
    assert flow in ['source_to_target', 'target_to_source']

    if batch is None:
        # Single graph case
        num_nodes = x.shape[0]
        dist = torch.cdist(x, x) # Shape: [N, N]

        if not loop:
            dist.fill_diagonal_(float('inf'))

        # Ensure k is not larger than num_nodes (minus 1 if no loop)
        effective_k = min(k, num_nodes - (1 if not loop else 0))
        if effective_k <= 0:
             return torch.empty((2,0), dtype=torch.long, device=x.device)

        # Find the k nearest neighbors
        _, col = torch.topk(dist, effective_k, dim=1, largest=False) # Shape: [N, effective_k]
        row = torch.arange(num_nodes, device=x.device).view(-1, 1).repeat(1, effective_k) # Shape: [N, effective_k]

        # Flatten and create edge index
        row = row.flatten() # Shape: [N * effective_k]
        col = col.flatten() # Shape: [N * effective_k]

    else:
        # Batched graph case
        num_nodes_total = x.shape[0]

        # Create a mask to isolate intra-batch distances
        batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1) # Shape [N_total, N_total]

        # Calculate pairwise distances and mask out cross-batch distances
        dist_full = torch.cdist(x, x) # Shape [N_total, N_total]
        dist_full[~batch_mask] = float('inf') # Set cross-batch distances to infinity

        # Exclude self-loops if necessary
        if not loop:
            dist_full.fill_diagonal_(float('inf'))

        # Ensure k is not larger than the total number of nodes (minus 1 if no loops)
        # Note: Technically, k should be compared to batch_size per batch,
        # but topk handles cases where k > available neighbors gracefully by
        # potentially returning fewer than k valid indices if rows have many infs.
        # We clamp k globally for safety.
        effective_k = min(k, num_nodes_total - (1 if not loop else 0))
        if effective_k <= 0:
             return torch.empty((2,0), dtype=torch.long, device=x.device)

        # Find top-k for each node (respecting batch boundaries due to masking)
        _, col = torch.topk(dist_full, effective_k, dim=1, largest=False) # Shape: [N_total, effective_k]
        row = torch.arange(num_nodes_total, device=x.device).view(-1, 1).repeat(1, effective_k) # Shape: [N_total, effective_k]

        # Flatten and create edge index
        row = row.flatten() # Shape: [N_total * effective_k]
        col = col.flatten() # Shape: [N_total * effective_k]

        # Filter out invalid edges (where distance was inf - topk might return node 0 or self index in such cases)
        # This happens if k is larger than the number of valid neighbors within a batch.
        # A simple check: if row == col and loop is False, it's likely an invalid edge from topk on inf row.
        # Or more robustly, check if the corresponding distance was inf (requires re-fetching distances).
        # Let's keep it simple for now, assuming topk gives reasonable indices. If issues arise, add distance check.
        # If loop is False, remove self-loops that might have been picked by topk if k is small.
        if not loop:
            mask = row != col
            row = row[mask]
            col = col[mask]


    # Combine row and col based on flow direction
    if flow == 'source_to_target':
        edge_index = torch.stack([row, col], dim=0)
    else: # target_to_source
        edge_index = torch.stack([col, row], dim=0)

    # Ensure edge_index is LongTensor
    return edge_index.to(torch.long)


class SchNetLayer(nn.Module):
    """Lightweight continuous-filter convolution layer for atom-level processing"""
    # def __init__(self, node_dim: int, hidden_dim: int = 64): # hidden_dim here is for internal filter expansion
    # Corrected definition: node_dim is the main feature dim, hidden_dim is for internal MLP layers
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        # Filter network: Process distance -> Expand -> Contract to node_dim
        self.filter_net = nn.Sequential(
            nn.Linear(1, hidden_dim), # Input is distance (1 dim), expand to hidden_dim
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim) # <<< FIX: Output node_dim to match h_col
        )
        # Update network: Process [current_node_features, aggregated_messages]
        # Aggregated messages will now also have node_dim size
        self.update_net = nn.Sequential(
            # <<< FIX: Input dimension is now node_dim (from h) + node_dim (from agg_messages)
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), # Optional: LayerNorm
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim) # Output: node feature update (size node_dim)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_indices: torch.Tensor, batch_size: int):
        """
        Args:
            x (torch.Tensor): Node coordinates, shape [B, N, 3].
            h (torch.Tensor): Node features, shape [B, N, node_dim].
            edge_indices (torch.Tensor): Edge indices (global), shape [2, E_total].
            batch_size (int): Number of graphs in the batch (B).
        Returns:
            torch.Tensor: Updated node features, shape [B, N, node_dim].
        """
        B, N, node_dim_runtime = h.shape # Use runtime node_dim for safety
        device = h.device
        E_total = edge_indices.shape[1]

        row, col = edge_indices

        batch_idx_row = row // N
        node_idx_row = row % N
        batch_idx_col = col // N
        node_idx_col = col % N

        x_row = x[batch_idx_row, node_idx_row, :]
        x_col = x[batch_idx_col, node_idx_col, :]
        h_col = h[batch_idx_col, node_idx_col, :] # Shape [E_total, node_dim]

        dist = torch.norm(x_row - x_col, dim=-1, keepdim=True) # Shape [E_total, 1]

        # Calculate edge filter weights based on distances
        edge_filters = self.filter_net(dist) # <<< Now shape [E_total, node_dim]

        # Calculate messages: Source node features modulated by edge filters
        # Shapes should now match: [E_total, node_dim] * [E_total, node_dim]
        messages = h_col * edge_filters

        # Aggregate messages to target nodes using scatter_mean
        num_nodes_total = B * N
        # Ensure row index is long type for scatter
        agg_messages_flat = scatter_mean(messages, row.long(), dim=0, dim_size=num_nodes_total) # Shape [B*N, node_dim]

        # Reshape aggregated messages back to [B, N, node_dim]
        agg_messages = agg_messages_flat.view(B, N, -1)

        # Update node features: Concatenate current features with aggregated messages
        # Shape should now be [B, N, node_dim + node_dim] = [B, N, node_dim * 2]
        update_input = torch.cat([h, agg_messages], dim=-1)
        h_update = self.update_net(update_input) # Shape [B, N, node_dim]

        return h_update



class SE3DiffusionModel(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        topology, # Expects an object with n_residues, atoms list (with residue.index, name, idx)
        atom_types: List[str], # List of atom type names (strings) for all atoms
        node_feature_dim: int = 64, # Increased default
        # edge_feature_dim: int = 16, # Removed, as SchNetLayer doesn't use it now
        time_embedding_dim: int = 128, # Increased default
        hidden_dim: int = 128,         # Hidden dim for MLPs, SchNet filters, MHA
        num_schnet_layers: int = 3,    # Increased default
        num_gat_layers: int = 2,       # Increased default
        residue_attn_heads: int = 4,   # Added parameter, increased default
        k_neighbors: int = 16         # Increased default
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.k_neighbors = k_neighbors
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim

        # Process topology and atom types
        self.unique_atom_types = sorted(list(set(atom_types)))
        self.atom_type_map = {name: i for i, name in enumerate(self.unique_atom_types)}
        print(f"Unique atom types found: {len(self.unique_atom_types)}")
        self.num_atom_types_unique = len(self.unique_atom_types)

        # Store topology info needed for forward pass (indices)
        self.set_topology(topology) # Pass only topology

        # --- Embeddings ---
        # Residue ID embedding
        self.residue_embedding = nn.Embedding(self.num_residues, node_feature_dim)
        # Atom type embedding (using unique count)
        self.atom_type_embedding = nn.Embedding(self.num_atom_types_unique, node_feature_dim)
        # Coordinate encoding (simple linear layer)
        self.coord_encoder = nn.Linear(3, node_feature_dim)
        # Optional: Add LayerNorm after combining initial embeddings
        self.initial_embed_norm = nn.LayerNorm(node_feature_dim)

        # Time embedding
        self.time_embed_module = SinusoidalEmbedding(time_embedding_dim)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feature_dim) # Project to node_feature_dim for addition
        )

        # --- Atom-level Processing (SchNet-style) ---
        self.schnet_layers = nn.ModuleList([
            SchNetLayer(node_feature_dim, hidden_dim) # Pass node_dim and hidden_dim for filter net
            for _ in range(num_schnet_layers)
        ])
        # LayerNorm after each SchNet update can be beneficial
        self.schnet_norms = nn.ModuleList([nn.LayerNorm(node_feature_dim) for _ in range(num_schnet_layers)])


        # --- Residue-level Attention (Multihead Attention) ---
        self.residue_input_proj = nn.Linear(node_feature_dim, hidden_dim) # Project avg residue features
        self.residue_attn_layers = nn.ModuleList([
            # Use hidden_dim for MHA internal dimension
            nn.MultiheadAttention(hidden_dim, num_heads=residue_attn_heads, batch_first=True, dropout=0.1)
            for _ in range(num_gat_layers)
        ])
        # LayerNorm after each Attention Layer
        self.residue_attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)])
        self.residue_output_proj = nn.Linear(hidden_dim, node_feature_dim) # Project back to node_feature_dim
        self.residue_final_norm = nn.LayerNorm(node_feature_dim) # Norm after adding back to atom features


        # --- Final Output Prediction ---
        # Final processing layers before output
        self.final_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim), # Add norm
            nn.Linear(hidden_dim, node_feature_dim) # Keep node_feature_dim before final output
        )
        self.output_mlp = nn.Linear(node_feature_dim, 3) # Predicts displacement or final coords

    def set_topology(self, topology):
        """Sets topology information and precomputes indices."""
        self.topology = topology
        self.num_residues = topology.n_residues
        print(f"Topology: {self.num_residues} residues, {topology.n_atoms} atoms.")


        # Create residue indices tensor [N] - Ensure it's LongTensor
        # Ensure atoms are iterated correctly if needed, but for indices it should be fine
        self.register_buffer('residue_indices', torch.tensor([atom.residue.index for atom in topology.atoms], dtype=torch.long))

        # Create atom type indices tensor [N] using the map - Ensure it's LongTensor
        # Ensure all atom names in topology.atoms exist in the provided atom_types list during init
        try:
             # Iterate through the generator here is fine
             atom_type_indices = [self.atom_type_map[atom.name] for atom in topology.atoms]
        except KeyError as e:
             raise ValueError(f"Atom name '{e}' found in topology but not in the initial atom_types list used for mapping.")
        self.register_buffer('atom_types_mapped', torch.tensor(atom_type_indices, dtype=torch.long))

        # Precompute base covalent bond edges [2, num_bonds*2]
        # Iterating through topology.bonds (likely also a generator) is fine here
        if hasattr(topology, 'bonds') and topology.bonds:
             # Ensure indices are based on atom index (0 to N-1)
             bond_list = [[b.atom1.index, b.atom2.index] for b in topology.bonds]
             if not bond_list: # Handle case where bonds exist but list is empty
                  base_edges_tensor = torch.empty((2,0), dtype=torch.long)
             else:
                  base_edges_tensor = torch.tensor(bond_list, dtype=torch.long).t().contiguous()
                  # Add reverse edges for undirected bonds
                  base_edges_tensor = torch.cat([base_edges_tensor, base_edges_tensor.flip(0)], dim=1)
        else:
             base_edges_tensor = torch.empty((2,0), dtype=torch.long) # Handle case with no bonds attribute or empty list

        # Use topology.n_bonds if available and reliable, otherwise calculate from tensor
        # print(f"Found {base_edges_tensor.shape[1] // 2} covalent bonds.")
        # More robust way using topology attribute if it exists:
        num_bonds = topology.n_bonds if hasattr(topology, 'n_bonds') else base_edges_tensor.shape[1] // 2
        print(f"Found {num_bonds} covalent bonds.")

        self.register_buffer('base_edges', base_edges_tensor)


    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor):
        """
        Forward pass of the SE(3) diffusion model.

        Args:
            x_noisy (torch.Tensor): Noisy atom coordinates (CENTERED), shape [B, N, 3].
            t (torch.Tensor): Timesteps, shape [B].

        Returns:
            torch.Tensor: Predicted clean coordinates x0 (CENTERED), shape [B, N, 3].
        """
        B, N, _ = x_noisy.shape
        assert N == self.num_atoms, f"Input N ({N}) does not match model's num_atoms ({self.num_atoms})"
        assert x_noisy.ndim == 3 and x_noisy.shape[-1] == 3, f"Expected x_noisy shape [B, N, 3], got {x_noisy.shape}"
        assert t.ndim == 1 and t.shape[0] == B, f"Expected t shape [B], got {t.shape}"
        device = x_noisy.device

        # Ensure precomputed tensors are on the correct device (buffers should handle this, but check)
        # residue_indices = self.residue_indices.to(device)
        # atom_types_mapped = self.atom_types_mapped.to(device)
        # base_edges = self.base_edges.to(device)
        residue_indices = self.residue_indices
        atom_types_mapped = self.atom_types_mapped
        base_edges = self.base_edges

        # --- Dynamic k-NN Graph Construction ---
        x_flat = x_noisy.view(B * N, 3) # Shape [B*N, 3]
        batch_vector = torch.arange(B, device=device).repeat_interleave(N) # Shape [B*N]
        spatial_edges = knn_graph_pytorch(x_flat, k=self.k_neighbors, batch=batch_vector, loop=False)
        # spatial_edges shape: [2, num_knn_edges] (global indices 0 to B*N-1)

        # --- Combine Base Edges and Spatial Edges ---
        # Repeat base_edges for each batch item and add offset
        if base_edges.numel() > 0:
            batch_offsets = torch.arange(B, device=device) * N
            # Correctly expand offsets for addition to edges
            # Offsets shape [B], base_edges shape [2, E_base]
            # We need offset added to each element in base_edges
            # expanded_base_edges = base_edges.repeat(1, B) # Repeats edges side-by-side
            # Need to add offset correctly:
            expanded_base_edges = torch.cat([base_edges + offset for offset in batch_offsets], dim=1)
            edge_indices = torch.cat([expanded_base_edges, spatial_edges], dim=1)
        else:
            edge_indices = spatial_edges

        # Remove duplicate edges (important if base edges are also close neighbors)
        # Note: unique is computationally intensive. Consider if strictly necessary.
        # If k is large enough, k-NN might capture most bonds anyway.
        edge_indices = torch.unique(edge_indices, dim=1)

        # Enable AMP only if on CUDA and input is float
        use_amp = x_noisy.is_cuda and x_noisy.dtype == torch.float32
        with autocast(enabled=use_amp):
            # --- Initial Node Features ---
            # Expand static embeddings for the batch [B, N, node_feature_dim]
            res_emb = self.residue_embedding(residue_indices).unsqueeze(0).expand(B, -1, -1)
            atom_emb = self.atom_type_embedding(atom_types_mapped).unsqueeze(0).expand(B, -1, -1)
            coord_emb = self.coord_encoder(x_noisy) # Shape [B, N, node_feature_dim]

            # Combine initial features
            h = res_emb + atom_emb + coord_emb # Shape [B, N, node_feature_dim]
            h = self.initial_embed_norm(h)

            # --- Time Embedding ---
            t_emb = self.time_embed_module(t) # Shape [B, time_embedding_dim]
            t_proj = self.time_embed_mlp(t_emb).unsqueeze(1) # Shape [B, 1, node_feature_dim]

            # Add time embedding to node features
            h = h + t_proj # Add time influence globally to all nodes

            # --- Atom-level Processing (SchNet Layers) ---
            # Pass batched coordinates x_noisy, current features h, global edge_indices, and batch_size
            for i, layer in enumerate(self.schnet_layers):
                # Store residual for connection
                h_residual = h
                # Calculate update using the SchNet layer
                h_update = layer(x_noisy, h, edge_indices, B)
                # Apply update with residual connection and norm
                h = self.schnet_norms[i](h_residual + h_update)
                # Now h has shape [B, N, node_feature_dim]

            # --- Residue-level Processing ---
            # Average atom features within each residue for each batch item
            # Create global residue indices: [B*N]
            residue_indices_expanded = residue_indices.repeat(B) # Shape [B*N]
            batch_offsets_res = torch.arange(B, device=device).repeat_interleave(N) * self.num_residues # Offset for global residue index
            residue_indices_global = residue_indices_expanded + batch_offsets_res # Shape [B*N]

            # Scatter requires input [num_items, features], index [num_items]
            h_flat_for_scatter = h.view(B * N, -1) # Shape [B*N, node_feature_dim]
            num_residues_total = B * self.num_residues

            # Average atom features per residue globally
            residue_h_flat = scatter_mean(h_flat_for_scatter, residue_indices_global.long(), dim=0, dim_size=num_residues_total)
            # Reshape back to [B, num_residues, node_feature_dim]
            residue_h = residue_h_flat.view(B, self.num_residues, -1)

            # Project residue features to hidden_dim for MHA
            residue_h_proj = self.residue_input_proj(residue_h) # Shape [B, num_residues, hidden_dim]

            # Apply MultiheadAttention layers with residual connections and LayerNorm
            for i, attn_layer in enumerate(self.residue_attn_layers):
                residue_residual = residue_h_proj
                # MHA expects (Query, Key, Value) -> (Batch, Seq, Feature)
                attn_out, _ = attn_layer(residue_h_proj, residue_h_proj, residue_h_proj) # Shape [B, num_residues, hidden_dim]
                # Add & Norm
                residue_h_proj = self.residue_attn_norms[i](residue_residual + attn_out)

            # Project back to node_feature_dim
            residue_h_updated = self.residue_output_proj(residue_h_proj) # Shape [B, num_residues, node_feature_dim]

            # Propagate updated residue features back to atoms (Add to existing features)
            # Use global residue indices again for gathering the updated residue features
            residue_h_updated_flat = residue_h_updated.view(B * self.num_residues, -1) # Shape [B*num_res, node_dim]
            # Gather based on the global residue index for each atom
            residue_context_gathered_flat = residue_h_updated_flat[residue_indices_global.long()] # Shape [B*N, node_dim]
            residue_context_gathered = residue_context_gathered_flat.view(B, N, -1) # Shape [B, N, node_feature_dim]

            # Add residue context back to atom features and normalize
            h = self.residue_final_norm(h + residue_context_gathered)

            # --- Final Output Prediction ---
            h = self.final_mlp(h) # Final processing MLPs
            x0_pred = self.output_mlp(h) # Predict coordinates/displacement

        # The model predicts the CENTERED x0
        return x0_pred.float()