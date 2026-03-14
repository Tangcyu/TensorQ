#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import math
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loss import loss_vcns_multibasin_pairwise
from dataset import PairCommittorDataset, split_train_val, subset_dataset

# =========================================================
# === Config utilities
# =========================================================

def load_yaml_config(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error reading YAML: {exc}")


def setup_device(device_str: str) -> torch.device:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# =========================================================
# === Model: simple MLP encoder
# =========================================================

class Encoder(nn.Module):
    """
    MLP encoder mapping internal-coordinate features -> committor vector (pairwise).
    Output is in [0,1] via Sigmoid.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...], activation="elu", dropout=0.0):
        super().__init__()
        acts = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        Act = acts.get(activation.lower(), nn.ELU)

        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        # layers.append(nn.Sigmoid())  # pairwise committor in [0,1]
        layers.append(nn.Identity())  # pairwise committor
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





# =========================================================
# === Training loop
# =========================================================

@dataclass
class TrainState:
    best_val: float = float("inf")
    best_epoch: int = -1


def run_epoch(model, loader, optimizer, device, k_scale, weighted_restraint, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    n = 0

    for batch in loader:
        batch = [b.to(device) for b in batch]
        loss = loss_vcns_multibasin_pairwise(
            model, batch, k_scale=float(k_scale), weighted_restraint=bool(weighted_restraint)
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = batch[0].shape[0]
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(1, n)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    out_prefix: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    k_scale: float,
    weighted_restraint: bool,
    seed: int,
):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    state = TrainState()
    history = {"train": [], "val": []}

    best_path = out_prefix + "_best_model.pt"
    last_path = out_prefix + "_last_model.pt"

    for ep in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, k_scale, weighted_restraint, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, k_scale, weighted_restraint, train=False)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = val_loss < state.best_val - 1e-8
        if improved:
            state.best_val = val_loss
            state.best_epoch = ep
            # Save scripted model for portability
            scripted = torch.jit.script(model.cpu())
            scripted.save(best_path)
            model.to(device)

        # always save last
        scripted_last = torch.jit.script(model.cpu())
        scripted_last.save(last_path)
        model.to(device)

        print(f"[Epoch {ep:4d}] train={train_loss:.6e}  val={val_loss:.6e}  best={state.best_val:.6e}@{state.best_epoch}")

        if ep - state.best_epoch >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

    # Save history
    np.save(out_prefix + "_loss_train.npy", np.array(history["train"], dtype=np.float64))
    np.save(out_prefix + "_loss_val.npy", np.array(history["val"], dtype=np.float64))
    return best_path, last_path, history


# =========================================================
# === Main training pipeline
# =========================================================

def train_committor_vector(config: dict):
    """
    Expected YAML structure:
    TRAIN:
      dataset_path: "out/dataset.pt"
      out_dir: "train_out"
      device: "cuda:0"
      seed: 0
      time_shift: 1
      val_ratio: 0.1
      drop_intermediate: true
      require_both_labeled: true
      strict_meta_state: false

      # model
      hidden: [256, 256, 128]
      activation: "elu"
      dropout: 0.0

      # optimization
      batch_size: 2048
      epochs: 500
      patience: 30
      lr: 1.0e-3
      weight_decay: 1.0e-6
      k_scale: 100.0
      weighted_restraint: false
    """
    out_dir = ensure_dir(config.get("out_dir", "./train_out"))
    device = setup_device(config.get("device", "cuda:0"))
    seed = int(config.get("seed", 0))

    dataset_path = config["dataset_path"]
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    # ---- load dataset ----
    pack = torch.load(dataset_path, map_location="cpu")
    features = pack["features"].float()
    weights = pack["weights"].float()
    pair_labels = pack.get("pair_labels", None)
    meta_state = pack.get("meta_state", None)

    if pair_labels is None:
        raise RuntimeError("dataset.pt must contain 'pair_labels' for multibasin pairwise training.")

    pair_labels = pair_labels.to(torch.int8)

    N, D = features.shape
    M = pair_labels.shape[1]
    print(f"[DATA] N={N}, D={D}, M=C(n,2)={M}")

    # ---- build dataset of (x0, xt) pairs ----
    time_shift = int(config.get("time_shift", 1))
    drop_intermediate = bool(config.get("drop_intermediate", True))
    require_both_labeled = bool(config.get("require_both_labeled", True))
    strict_meta_state = bool(config.get("strict_meta_state", False))

    ds = PairCommittorDataset(
        features=features,
        weights=weights,
        pair_labels=pair_labels,
        time_shift=time_shift,
        drop_intermediate=drop_intermediate,
        require_both_labeled=require_both_labeled,
        meta_state=(meta_state if strict_meta_state else None),
    )

    # train/val split over pair indices
    val_ratio = float(config.get("val_ratio", 0.1))
    tr_idx, va_idx = split_train_val(len(ds), val_ratio=val_ratio, seed=seed)
    train_ds = subset_dataset(ds, tr_idx)
    val_ds = subset_dataset(ds, va_idx)

    batch_size = int(config.get("batch_size", 2048))
    num_workers = int(config.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # ---- model ----
    hidden = tuple(int(x) for x in config.get("hidden", [256, 256, 128]))
    activation = config.get("activation", "elu")
    dropout = float(config.get("dropout", 0.0))

    model = Encoder(in_dim=D, out_dim=M, hidden=hidden, activation=activation, dropout=dropout).to(device)

    # ---- train ----
    epochs = int(config.get("epochs", 500))
    patience = int(config.get("patience", 30))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-6))
    k_scale = float(config.get("k_scale", 100.0))
    weighted_restraint = bool(config.get("weighted_restraint", False))

    label = config.get("label", "vcn_multibasin")
    out_prefix = os.path.join(out_dir, label)

    best_path, last_path, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_prefix=out_prefix,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        k_scale=k_scale,
        weighted_restraint=weighted_restraint,
        seed=seed,
    )

    # ---- save CPU copy of best model ----
    best_model = torch.jit.load(best_path, map_location="cpu")
    cpu_path = out_prefix + "_cpu_best_model.pt"
    best_model.save(cpu_path)
    print(f"[DONE] Best model: {best_path}")
    print(f"[DONE] CPU model : {cpu_path}")

    # ---- save a tiny summary ----
    summary = {
        "best_model": os.path.abspath(best_path),
        "cpu_model": os.path.abspath(cpu_path),
        "last_model": os.path.abspath(last_path),
        "best_val": float(np.min(history["val"])) if history["val"] else None,
        "epochs_ran": len(history["val"]),
        "D": D,
        "M": M,
        "time_shift": time_shift,
        "k_scale": k_scale,
        "weighted_restraint": weighted_restraint,
    }
    with open(out_prefix + "_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    return summary


# =========================================================
# === CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train multibasin committor-vector network (pairwise)")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if "VECVCN" not in cfg:
        raise KeyError("Config must contain a top-level key: VECVCN")

    train_committor_vector(cfg["VECVCN"])


if __name__ == "__main__":
    main()
