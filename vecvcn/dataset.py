from torch.utils.data import Dataset
import torch
import numpy as np
import math
from typing import Optional

# =========================================================
# === Dataset: build (x0, xt) pairs from trajectory frames
# =========================================================

class PairCommittorDataset(Dataset):
    """
    From a sequential trajectory dataset, build time-shift pairs:
      (features[t], features[t+time_shift]), with weights[t], labels at both times.

    Supports filtering:
      - if drop_intermediate=True: only keep pairs where BOTH endpoints have at least one constrained dim
        (i.e., pl has any >=0)
      - or require_meta_state endpoints != -1 (if meta_state exists and you want strict)
    """
    def __init__(
        self,
        features: torch.Tensor,          # (N,D)
        weights: torch.Tensor,           # (N,)
        pair_labels: torch.Tensor,       # (N,M)
        time_shift: int = 1,
        drop_intermediate: bool = True,
        require_both_labeled: bool = True,
        meta_state: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert features.ndim == 2
        assert weights.ndim == 1
        assert pair_labels.ndim == 2
        self.features = features
        self.weights = weights
        self.pair_labels = pair_labels
        self.time_shift = int(time_shift)

        N = features.shape[0]
        idx0 = torch.arange(0, N - self.time_shift, dtype=torch.long)
        idxt = idx0 + self.time_shift

        # determine if frame has any valid pairwise label (belongs to any endpoint basin for some pair)
        has_label0 = (pair_labels[idx0] >= 0).any(dim=1)
        has_labelt = (pair_labels[idxt] >= 0).any(dim=1)

        keep = torch.ones_like(has_label0, dtype=torch.bool)

        if drop_intermediate:
            # If user wants to remove intermediate frames entirely, this criterion works well:
            # keep frames that have at least one constrained dim.
            keep &= has_label0
            keep &= has_labelt

        if require_both_labeled:
            keep &= has_label0
            keep &= has_labelt

        if meta_state is not None:
            # strict: remove frames where meta_state == -1
            m0 = meta_state[idx0] != -1
            mt = meta_state[idxt] != -1
            keep &= m0 & mt

        self.idx0 = idx0[keep]
        self.idxt = idxt[keep]

        if len(self.idx0) == 0:
            raise RuntimeError("No training pairs left after filtering. Relax filters or check labels.")

    def __len__(self):
        return self.idx0.numel()

    def __getitem__(self, i):
        i0 = self.idx0[i]
        it = self.idxt[i]
        x0 = self.features[i0]
        xt = self.features[it]
        w = self.weights[i0]  # use weight at time 0 (common choice)
        pl0 = self.pair_labels[i0]
        plt = self.pair_labels[it]
        return x0, xt, w, pl0, plt


def split_train_val(n: int, val_ratio: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(math.ceil(val_ratio * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def subset_dataset(ds: Dataset, indices: np.ndarray) -> Dataset:
    # Wrap a dataset subset without copying tensors
    class _Subset(Dataset):
        def __init__(self, base, inds):
            self.base = base
            self.inds = torch.as_tensor(inds, dtype=torch.long)
        def __len__(self): return self.inds.numel()
        def __getitem__(self, i): return self.base[int(self.inds[i])]
    return _Subset(ds, indices)
