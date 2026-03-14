import torch
import torch.nn as nn 
from typing import Optional

# =========================================================
# === Loss functions (VCN + multibasin soft endpoints)
# =========================================================

@torch.compile
def JAB_vec(q0: torch.Tensor, qt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # q0, qt: (B, M); w: (B,)
    diff2 = (q0 - qt).square().sum(dim=1)  # (B,)
    return (w * diff2).sum() / (w.sum() + 1e-12)


@torch.compile
def soft_endpoint_pairwise_loss(q: torch.Tensor, pair_labels: torch.Tensor,
                                w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    q: (B, M) in [0,1]
    pair_labels: (B, M) in {-1,0,1} (int8/int64/float)
    mask where >=0, target is 0/1
    Normalize per-sample by number of constrained dims to avoid bias.
    """
    pl = pair_labels.to(q.dtype)
    mask = (pl >= 0).to(q.dtype)              # (B,M)
    target = pl.clamp(min=0.0)                # -1 -> 0 but masked out

    mse = (q - target).square() * mask        # (B,M)
    denom = mask.sum(dim=1).clamp(min=1.0)    # (B,)
    per_sample = mse.sum(dim=1) / denom       # (B,)

    if w is None:
        return per_sample.mean()
    return (w * per_sample).sum() / (w.sum() + 1e-12)


@torch.compile
def loss_vcns_multibasin_pairwise(model: nn.Module, batch, k_scale: float = 100.0,
                                  weighted_restraint: bool = False) -> torch.Tensor:
    """
    batch:
      x0: (B,D)
      xt: (B,D)
      w : (B,)
      pl0: (B,M)
      plt: (B,M)
    """
    x0, xt, w, pl0, plt = batch
    q0 = model(x0)
    qt = model(xt)

    loss_dir = JAB_vec(q0, qt, w)

    if weighted_restraint:
        r0 = soft_endpoint_pairwise_loss(q0, pl0, w=w)
        rt = soft_endpoint_pairwise_loss(qt, plt, w=w)
    else:
        r0 = soft_endpoint_pairwise_loss(q0, pl0, w=None)
        rt = soft_endpoint_pairwise_loss(qt, plt, w=None)

    return loss_dir + k_scale * (r0 + rt)