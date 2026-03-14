#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reweight unbiased-shooting trajectories via RiteWeight on internal-coordinate
features, then cluster conformations into meta-stable states using KMeans with
automatic k from elbow selection.

The script accepts either:
- a dedicated `MultiState:` config block, or
- a legacy flat config file.

It also supports `--relabel-only`, which reloads an existing dataset and
recomputes only the meta-state labels without rerunning feature extraction or
RiteWeight.

Outputs a compact dataset (prefer torch .pt, fallback .npz) containing:
- features: (n_frames, 3N-6) float32   # for committor training
- weights:  (n_frames,) float32
- meta_state: (n_frames,) int64, in {0..k-1, -1}  # -1 = intermediate
- dist_to_centroid: (n_frames,) float32
- optional: CV matrix and headers
- optional: pairwise committor label matrix for all (i,j), i<j, with values {0,1,-1}
- optional: concatenated DCD + diagnostic figures

YAML keys (minimum):
  MultiState:
    topology_file: "xxx.psf"
    dcd_folder: "path/to/dcds"
    output_dir: "out"
    match: "shoot_"          # filename prefix match for .dcd
    sel_weights: "protein and not name H*"
    sel_output:  "protein and not name H*"
    every: 1
    colvars_mismatch: true

RiteWeight:
  MultiState:
    riteweight:
      n_clusters: 100
      n_iter: 200
      tol: 1.0e-6
      tol_window: 5
      avg_last: 20
      seed: 2026
      lag: 1

Elbow+KMeans:
  n_clusters: null         # set an integer to choose k by hand
  kmin: 2
  kmax: 12
  elbow_method: "knee"     # "knee" or "second_derivative"
  standardize_features: true
  cluster_space: "features"  # "features" or "pca_highdim"
  pca_cluster_dim: 20
  intermediate_quantile: 0.9
  kmeans_random_state: 0
  kmeans_n_init: "auto"

Dataset saving:
  save_format: "pt"        # "pt" or "npz"
  save_cv: true
  cvs_to_save: ["cv1","cv2"]  # optional; if empty, save all colvars
  periodic: false            # if true, add sin/cos for cvs_to_save (degrees)
  make_pairwise_committor: true
  pairwise_prefix: "q"       # only for naming in metadata, not needed for tensors

Optional:
  write_concat_dcd: false
  write_diag_plots: true
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eig

from MDAnalysis import Universe
from MDAnalysis.coordinates.DCD import DCDWriter
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals

# ---- optional torch ----
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# =========================================================
# === Helpers: internal coordinates (3N-6)
# =========================================================

def build_min_zmatrix_indices(n_atoms: int):
    """
    Minimal Z-matrix-like internal coordinate definition (3N-6).
    Sequential topology:
      bonds:    (i, i-1)
      angles:   (i, i-1, i-2)
      dihedral: (i, i-1, i-2, i-3)
    """
    if n_atoms < 4:
        raise ValueError("Need at least 4 atoms to form a minimal (3N-6) internal coordinate set.")

    bonds = np.array([[i, i - 1] for i in range(1, n_atoms)], dtype=np.int32)          # (N-1, 2)
    angles = np.array([[i, i - 1, i - 2] for i in range(2, n_atoms)], dtype=np.int32)  # (N-2, 3)
    diheds = np.array([[i, i - 1, i - 2, i - 3] for i in range(3, n_atoms)], dtype=np.int32)  # (N-3, 4)
    return bonds, angles, diheds

def internal_coords_min_zmatrix(positions: np.ndarray, bonds, angles, diheds):
    """
    Periodic-safe internal coordinates for one frame:
      - bond lengths: b (Å)
      - angles: sin(a), cos(a)
      - dihedrals: sin(d), cos(d)
    Output dim = (N-1) + 2*(N-2) + 2*(N-3) = 5N-11
    """
    # Bonds
    b = calc_bonds(
        positions[bonds[:, 0]],
        positions[bonds[:, 1]],
    ).astype(np.float32)

    # Angles (radians)
    a = calc_angles(
        positions[angles[:, 0]],
        positions[angles[:, 1]],
        positions[angles[:, 2]],
    ).astype(np.float32)

    # Dihedrals (radians, in [-pi, pi])
    d = calc_dihedrals(
        positions[diheds[:, 0]],
        positions[diheds[:, 1]],
        positions[diheds[:, 2]],
        positions[diheds[:, 3]],
    ).astype(np.float32)

    # Periodic encoding
    sa, ca = np.sin(a).astype(np.float32), np.cos(a).astype(np.float32)
    sd, cd = np.sin(d).astype(np.float32), np.cos(d).astype(np.float32)

    # Concatenate
    return np.concatenate([b, sa, ca, sd, cd], axis=0)


# =========================================================
# === Helpers: colvars
# =========================================================

def read_colvars(colvars_path, index_mismatch=True, skip_rows=1):
    """Read .colvars.traj and remove duplicated headers."""
    with open(colvars_path, "r") as f:
        raw_headers = None
        for line in f:
            if line.startswith("#"):
                raw_headers = line[1:].strip().split()
                break
    if raw_headers is None:
        raise ValueError(f"Cannot find header line starting with # in {colvars_path}")

    data = np.loadtxt(colvars_path, comments=["#", "@"], skiprows=skip_rows)
    if index_mismatch:
        data = data[1:]

    seen, keep_indices, headers = {}, [], []
    for i, name in enumerate(raw_headers):
        if name not in seen:
            seen[name] = True
            keep_indices.append(i)
            headers.append(name)

    data = data[:, keep_indices]
    return headers, data


# =========================================================
# === Helpers: RiteWeight
# =========================================================

def assign_clusters_random_centers(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n_frames = X.shape[0]
    if n_clusters >= n_frames:
        raise ValueError(f"riteweight.n_clusters ({n_clusters}) must be < number of frames ({n_frames}).")

    center_idx = rng.choice(n_frames, size=n_clusters, replace=False)
    centers = X[center_idx]
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ centers.T)
    return np.argmin(d2, axis=1).astype(np.int32)


def build_transition_matrix(start_labels, end_labels, weights, n_clusters, eps=1e-15):
    trans_num = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    cluster_weight = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(cluster_weight, start_labels, weights)
    np.add.at(trans_num, (start_labels, end_labels), weights)

    trans = trans_num / (cluster_weight[:, None] + eps)
    zero_rows = cluster_weight <= eps
    if np.any(zero_rows):
        trans[zero_rows, :] = 0.0
        trans[zero_rows, np.where(zero_rows)[0]] = 1.0

    trans = np.clip(trans, 0.0, 1.0)
    row_sum = trans.sum(axis=1, keepdims=True)
    trans = trans / (row_sum + eps)
    return trans, cluster_weight


def stationary_distribution(trans):
    vals, vecs = eig(trans.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    vec = np.real(vecs[:, idx])
    vec = np.abs(vec)
    norm = vec.sum()
    if norm <= 0:
        raise RuntimeError("Failed to compute a valid stationary distribution in RiteWeight.")
    return vec / norm


def riteweight(X, seg_start_idx, seg_end_idx, n_clusters, n_iter, tol, tol_window, avg_last, seed):
    rng = np.random.default_rng(seed)
    n_segments = seg_start_idx.size
    seg_weights = np.full(n_segments, 1.0 / n_segments, dtype=np.float64)
    prev_weights = seg_weights.copy()
    delta_history = []
    stable_steps = 0
    trailing_weights = []

    for it in tqdm(range(1, n_iter + 1), desc="RiteWeight"):
        labels = assign_clusters_random_centers(X, n_clusters, rng)
        start_labels = labels[seg_start_idx]
        end_labels = labels[seg_end_idx]

        trans, cluster_weight = build_transition_matrix(start_labels, end_labels, seg_weights, n_clusters)
        pi = stationary_distribution(trans)

        scale = pi / (cluster_weight + 1e-15)
        new_weights = seg_weights * scale[start_labels]
        new_weights = np.clip(new_weights, 0.0, np.inf)
        new_weights /= new_weights.sum()

        delta = float(np.sum(np.abs(new_weights - prev_weights)))
        delta_history.append(delta)
        stable_steps = stable_steps + 1 if delta < tol else 0

        if it > max(1, n_iter - avg_last):
            trailing_weights.append(new_weights.copy())

        prev_weights = seg_weights
        seg_weights = new_weights
        if stable_steps >= tol_window:
            break

    if trailing_weights:
        seg_weights = np.mean(np.stack(trailing_weights, axis=0), axis=0)
        seg_weights = np.clip(seg_weights, 0.0, np.inf)
        seg_weights /= seg_weights.sum()

    frame_weights = np.zeros(X.shape[0], dtype=np.float64)
    np.add.at(frame_weights, seg_start_idx, seg_weights)
    frame_weights /= frame_weights.sum()
    return frame_weights.astype(np.float32), np.asarray(delta_history, dtype=np.float32)


def build_segment_indices(frame_counts, lag):
    seg_start = []
    seg_end = []
    offset = 0

    for n_frames in frame_counts:
        if n_frames <= lag:
            offset += n_frames
            continue
        starts = np.arange(offset, offset + n_frames - lag, dtype=np.int64)
        ends = starts + lag
        seg_start.append(starts)
        seg_end.append(ends)
        offset += n_frames

    if not seg_start:
        raise RuntimeError(
            f"No valid RiteWeight segments were built. Reduce riteweight.lag (current lag={lag}) "
            "or provide longer trajectories."
        )

    return np.concatenate(seg_start), np.concatenate(seg_end)


# =========================================================
# === Helpers: elbow selection + labeling
# =========================================================

def choose_k_elbow(X, kmin=2, kmax=12, method="knee", random_state=0, n_init="auto", out_png=None):
    """
    Return best_k, ks, inertias.
    method:
      - "knee": distance-to-chord heuristic
      - "second_derivative": max second difference
    """
    ks = list(range(int(kmin), int(kmax) + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X)
        inertias.append(float(km.inertia_))
    inertias = np.array(inertias, dtype=float)

    if len(ks) == 1:
        best_k = ks[0]
    elif method == "second_derivative" and len(ks) >= 3:
        d2 = np.zeros_like(inertias)
        d2[1:-1] = inertias[:-2] - 2 * inertias[1:-1] + inertias[2:]
        idx = int(np.argmax(d2[1:-1]) + 1)
        best_k = ks[idx]
    else:
        # knee (max distance to line from first to last)
        x = np.array(ks, dtype=float)
        y = inertias
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        dist = np.abs(a * x + b * y + c) / (np.sqrt(a*a + b*b) + 1e-12)
        idx = int(np.argmax(dist))
        best_k = ks[idx]

    if out_png is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(ks, inertias, marker="o")
        plt.axvline(best_k, linestyle="--")
        plt.xlabel("k")
        plt.ylabel("KMeans inertia")
        plt.title(f"Elbow curve (best k = {best_k})")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    return int(best_k), ks, inertias


def kmeans_metastable_labeling(X, n_clusters, quantile=0.9, random_state=0, n_init="auto"):
    """
    KMeans + per-cluster distance quantile threshold => intermediate = -1.

    Returns:
      state_id: (n_frames,) int64 in {0..k-1, -1}
      dists: (n_frames,) float32
      thresholds: (k,) float32
      kmeans: fitted object
    """
    kmeans = KMeans(n_clusters=int(n_clusters), n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(X).astype(np.int64)

    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(X - centers[labels], axis=1).astype(np.float32)

    thresholds = np.zeros(int(n_clusters), dtype=np.float32)
    for c in range(int(n_clusters)):
        mask = labels == c
        thresholds[c] = np.quantile(dists[mask], float(quantile)) if np.any(mask) else np.inf

    state_id = labels.copy()
    too_far = dists > thresholds[labels]
    state_id[too_far] = -1
    return state_id, dists, thresholds, kmeans


def build_pairwise_labels(meta_state, n_states):
    """
    Build pairwise committor labels for all i<j.
    For each pair (i,j): label=0 if state==i, 1 if state==j, else -1.
    Returns:
      pair_labels: (n_frames, C(n,2)) int8
      pairs: list[(i,j)]
    """
    meta_state = meta_state.astype(np.int64)
    pairs = []
    cols = []
    for i in range(int(n_states)):
        for j in range(i + 1, int(n_states)):
            pairs.append((i, j))
            col = np.full(meta_state.shape[0], -1, dtype=np.int8)
            col[meta_state == i] = 0
            col[meta_state == j] = 1
            cols.append(col)
    if len(cols) == 0:
        pair_labels = np.zeros((meta_state.shape[0], 0), dtype=np.int8)
    else:
        pair_labels = np.stack(cols, axis=1)
    return pair_labels, pairs


def infer_dataset_path(output_dir, save_format, dataset_path=None):
    if dataset_path:
        return dataset_path
    ext = ".pt" if str(save_format).lower() == "pt" else ".npz"
    return os.path.join(output_dir, "dataset" + ext)


def load_saved_dataset(dataset_path):
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".pt":
        if not TORCH_OK:
            raise RuntimeError("torch is required to read a .pt dataset.")
        pack = torch.load(dataset_path, map_location="cpu")
        features = pack["features"].detach().cpu().numpy().astype(np.float32)
        weights = pack["weights"].detach().cpu().numpy().astype(np.float32)
        meta_state = pack.get("meta_state")
        if meta_state is not None:
            meta_state = meta_state.detach().cpu().numpy().astype(np.int64)
        dist_to_centroid = pack.get("dist_to_centroid")
        if dist_to_centroid is not None:
            dist_to_centroid = dist_to_centroid.detach().cpu().numpy().astype(np.float32)
        cv_data = pack.get("cv")
        if cv_data is not None:
            cv_data = cv_data.detach().cpu().numpy().astype(np.float32)
        pair_labels = pack.get("pair_labels")
        if pair_labels is not None:
            pair_labels = pair_labels.detach().cpu().numpy().astype(np.int8)
        meta = pack.get("meta", {})
        return {
            "format": "pt",
            "pack": pack,
            "features": features,
            "weights": weights,
            "meta_state": meta_state,
            "dist_to_centroid": dist_to_centroid,
            "cv": cv_data,
            "pair_labels": pair_labels,
            "meta": meta if isinstance(meta, dict) else {},
        }

    if ext == ".npz":
        data = np.load(dataset_path, allow_pickle=True)
        meta = {}
        if "meta_yaml" in data:
            meta_yaml = data["meta_yaml"]
            if len(meta_yaml) > 0:
                meta = yaml.safe_load(str(meta_yaml[0])) or {}
        return {
            "format": "npz",
            "pack": data,
            "features": data["features"].astype(np.float32),
            "weights": data["weights"].astype(np.float32),
            "meta_state": data["meta_state"].astype(np.int64) if "meta_state" in data else None,
            "dist_to_centroid": data["dist_to_centroid"].astype(np.float32) if "dist_to_centroid" in data else None,
            "cv": data["cv"].astype(np.float32) if "cv" in data else None,
            "pair_labels": data["pair_labels"].astype(np.int8) if "pair_labels" in data else None,
            "meta": meta if isinstance(meta, dict) else {},
        }

    raise ValueError(f"Unsupported dataset extension for relabel-only: {dataset_path}")


def save_dataset(dataset_path, save_format, features_all, weights, meta_state, dist_to_centroid,
                 thresholds, meta, cv_data=None, pair_labels=None):
    if save_format == "pt":
        if not TORCH_OK:
            raise RuntimeError("torch is required to save a .pt dataset.")
        pack = {
            "features": torch.from_numpy(features_all.astype(np.float32)),
            "weights": torch.from_numpy(weights.astype(np.float32)),
            "meta_state": torch.from_numpy(meta_state.astype(np.int64)),
            "dist_to_centroid": torch.from_numpy(dist_to_centroid.astype(np.float32)),
            "meta": meta,
        }
        if cv_data is not None:
            pack["cv"] = torch.from_numpy(cv_data.astype(np.float32))
        if pair_labels is not None:
            pack["pair_labels"] = torch.from_numpy(pair_labels.astype(np.int8))
        torch.save(pack, dataset_path)
    else:
        npz_dict = {
            "features": features_all.astype(np.float32),
            "weights": weights.astype(np.float32),
            "meta_state": meta_state.astype(np.int64),
            "dist_to_centroid": dist_to_centroid.astype(np.float32),
            "thresholds": thresholds.astype(np.float32),
        }
        if cv_data is not None:
            npz_dict["cv"] = cv_data.astype(np.float32)
        if pair_labels is not None:
            npz_dict["pair_labels"] = pair_labels.astype(np.int8)
        npz_dict["meta_yaml"] = np.array([yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)], dtype=object)
        np.savez_compressed(dataset_path, **npz_dict)


def select_k_for_clustering(X_cluster, config, write_diag=False, elbow_png=None):
    manual_k = config.get("n_clusters", None)
    if manual_k is not None:
        best_k = int(manual_k)
        if best_k < 1:
            raise ValueError("n_clusters must be >= 1 when choosing k by hand.")
        ks = [best_k]
        inertias = []
        print(f"[KMeans] Using user-specified k = {best_k}")
        return best_k, ks, inertias

    kmin = int(config.get("kmin", 2))
    kmax = int(config.get("kmax", 12))
    elbow_method = config.get("elbow_method", "knee")
    kmeans_random_state = int(config.get("kmeans_random_state", 0))
    kmeans_n_init = config.get("kmeans_n_init", "auto")
    best_k, ks, inertias = choose_k_elbow(
        X_cluster,
        kmin=kmin,
        kmax=kmax,
        method=elbow_method,
        random_state=kmeans_random_state,
        n_init=kmeans_n_init,
        out_png=(elbow_png if write_diag else None),
    )
    print(f"[KMeans] Elbow selected k = {best_k}")
    return best_k, ks, inertias.tolist()


# =========================================================
# === Main
# =========================================================

def run_pipeline(config, relabel_only=False):
    psf_path = config["topology_file"]
    dcd_folder = config["dcd_folder"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ---- IO names ----
    dataset_base = os.path.join(output_dir, "dataset")
    weights_csv = os.path.join(output_dir, "weights_and_labels.csv")  # optional debug
    riteweight_png = os.path.join(output_dir, "riteweight_convergence.png")
    elbow_png = os.path.join(output_dir, "elbow_kmeans.png")

    # ---- selections ----
    selection_weights = config.get("sel_weights", "protein and not name H*")
    selection_output = config.get("sel_output", "protein and not name H*")

    # ---- sampling / weighting ----
    every = int(config.get("every", 1))
    index_mismatch = bool(config.get("colvars_mismatch", True))
    periodic = bool(config.get("periodic", False))

    rw_cfg = config.get("riteweight", {})
    rw_n_clusters = rw_cfg.get("n_clusters", config.get("rw_n_clusters", 100))
    rw_n_iter = int(rw_cfg.get("n_iter", config.get("rw_n_iter", 200)))
    rw_tol = float(rw_cfg.get("tol", config.get("rw_tol", 1e-6)))
    rw_tol_window = int(rw_cfg.get("tol_window", config.get("rw_tol_window", 5)))
    rw_avg_last = int(rw_cfg.get("avg_last", config.get("rw_avg_last", 20)))
    rw_seed = int(rw_cfg.get("seed", config.get("rw_seed", 2026)))
    rw_lag = int(rw_cfg.get("lag", config.get("lag", 1)))
    if rw_lag < 1:
        raise ValueError("riteweight.lag must be >= 1.")

    # ---- elbow/kmeans ----
    inter_quantile = float(config.get("intermediate_quantile", 0.9))
    kmeans_random_state = int(config.get("kmeans_random_state", 0))
    kmeans_n_init = config.get("kmeans_n_init", "auto")

    standardize = bool(config.get("standardize_features", True))
    cluster_space = config.get("cluster_space", "features")  # "features" or "pca_highdim"
    pca_cluster_dim = int(config.get("pca_cluster_dim", 20))

    # ---- dataset saving ----
    save_format = config.get("save_format", "pt").lower()  # pt/npz
    save_cv = bool(config.get("save_cv", True))
    cvs_to_save = config.get("cvs_to_save", [])  # if empty -> all colvars
    write_diag = bool(config.get("write_diag_plots", True))
    write_concat_dcd = bool(config.get("write_concat_dcd", False))
    dataset_path = infer_dataset_path(output_dir, save_format, config.get("dataset_path"))

    make_pairwise = bool(config.get("make_pairwise_committor", True))

    if relabel_only:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"--relabel-only requires an existing dataset: {dataset_path}")

        loaded = load_saved_dataset(dataset_path)
        features_all = loaded["features"]
        weights = loaded["weights"]
        cv_data = loaded["cv"]
        meta = loaded["meta"]
        cv_headers = meta.get("cv_headers", None)
        n_frames, feat_dim = features_all.shape

        X_cluster = features_all
        if standardize:
            scaler = StandardScaler()
            X_cluster = scaler.fit_transform(X_cluster)

        if cluster_space == "pca_highdim":
            ncomp = min(pca_cluster_dim, X_cluster.shape[1])
            pca_cluster = PCA(n_components=ncomp)
            X_cluster = pca_cluster.fit_transform(X_cluster)

        best_k, ks, inertias = select_k_for_clustering(
            X_cluster,
            config,
            write_diag=write_diag,
            elbow_png=elbow_png,
        )
        meta_state, dist_to_centroid, thresholds, _ = kmeans_metastable_labeling(
            X_cluster,
            n_clusters=best_k,
            quantile=inter_quantile,
            random_state=kmeans_random_state,
            n_init=kmeans_n_init,
        )

        pair_labels = None
        pairs = None
        if make_pairwise:
            pair_labels, pairs = build_pairwise_labels(meta_state, best_k)

        meta.update({
            "n_frames": int(n_frames),
            "feature_dim": int(feat_dim),
            "k_selected": int(best_k),
            "pairs": pairs,
            "config": config,
            "cv_headers": cv_headers,
            "standardize_features": bool(standardize),
            "cluster_space": cluster_space,
            "pca_cluster_dim": int(pca_cluster_dim) if cluster_space == "pca_highdim" else None,
            "intermediate_quantile": float(inter_quantile),
            "elbow": {
                "ks": ks,
                "inertias": inertias,
                "method": config.get("elbow_method", "knee"),
            },
            "notes": "features are minimal Z-matrix coordinates in float32; relabel-only recomputes only meta-state labels.",
        })

        save_dataset(
            dataset_path=dataset_path,
            save_format=loaded["format"],
            features_all=features_all,
            weights=weights,
            meta_state=meta_state,
            dist_to_centroid=dist_to_centroid,
            thresholds=thresholds,
            meta=meta,
            cv_data=cv_data,
            pair_labels=pair_labels,
        )
        print(f"[RELABEL] Updated dataset labels in: {dataset_path}")

        if save_cv:
            if cv_data is None or not cv_headers:
                print("[WARN] Existing dataset has no saved CV block; skipping weights_and_labels.csv update.")
            else:
                weights_csv = os.path.join(output_dir, "weights_and_labels.csv")
                df = pd.DataFrame(cv_data, columns=cv_headers)
                df.insert(0, "frame", np.arange(n_frames, dtype=np.int64))
                df["weight"] = weights
                df["meta_state"] = meta_state
                df["is_intermediate"] = (meta_state == -1).astype(np.int8)
                df["dist_to_centroid"] = dist_to_centroid
                df["k_selected"] = best_k
                df.to_csv(weights_csv, index=False)
                print(f"[RELABEL] Updated labels CSV: {weights_csv}")
        return

    # ---- find DCD files ----
    match_prefix = config["match"]
    dcd_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(dcd_folder)
        for f in files
        if f.startswith(match_prefix) and f.endswith(".dcd")
    ])
    if len(dcd_files) == 0:
        raise RuntimeError(f"No DCD files found with prefix '{match_prefix}' in {dcd_folder}")

    all_features = []
    all_colvars = []
    all_universes = []
    frame_counts = []
    headers_ref = None

    # =====================================================
    # 1) Load trajectories and build internal-coordinate features
    # =====================================================
    for dcd_path in tqdm(dcd_files, desc="Processing trajectories"):
        base = os.path.splitext(dcd_path)[0]
        colvars_path = base + ".colvars.traj"
        if not os.path.exists(colvars_path):
            print(f"⚠️ Missing colvars for {dcd_path}, skipping.")
            continue

        u = Universe(psf_path, dcd_path)
        sel_w = u.select_atoms(selection_weights)
        sel_out = u.select_atoms(selection_output)

        n_atoms = sel_w.n_atoms
        bonds, angles, diheds = build_min_zmatrix_indices(n_atoms)

        feats = []
        for ts in u.trajectory[::every]:
            feats.append(internal_coords_min_zmatrix(sel_w.positions, bonds, angles, diheds))
        feats = np.asarray(feats, dtype=np.float32)  # (n_frames_traj, 3N-6)

        headers, colvars_data = read_colvars(colvars_path, index_mismatch=index_mismatch)
        colvars_data = colvars_data[::every]
        if len(colvars_data) != len(feats):
            raise ValueError(f"Frame mismatch: {dcd_path} (colvars {len(colvars_data)} vs traj {len(feats)})")

        if headers_ref is None:
            headers_ref = headers
        else:
            if headers != headers_ref:
                raise ValueError("Colvars headers differ across trajectories. Please unify colvars outputs.")

        all_features.append(feats)
        all_colvars.append(colvars_data.astype(np.float32))
        all_universes.append(sel_out.universe)
        frame_counts.append(len(feats))

    if len(all_features) == 0:
        raise RuntimeError("No valid trajectories after filtering colvars.")

    features_all = np.vstack(all_features).astype(np.float32)
    colvars_all = np.vstack(all_colvars).astype(np.float32)

    n_frames, feat_dim = features_all.shape
    print(f"[INFO] Total frames: {n_frames}, feature dim (3N-6): {feat_dim}, colvars dim: {colvars_all.shape[1]}")

    # =====================================================
    # 2) Reweighting weights from RiteWeight
    # =====================================================
    seg_start_idx, seg_end_idx = build_segment_indices(frame_counts, rw_lag)
    max_rw_clusters = max(1, min(int(seg_start_idx.size - 1), int(n_frames - 1)))
    if rw_n_clusters is None:
        rw_n_clusters = min(100, max_rw_clusters)
    rw_n_clusters = int(rw_n_clusters)
    if rw_n_clusters > max_rw_clusters:
        print(
            f"[WARN] riteweight.n_clusters={rw_n_clusters} is too large for {seg_start_idx.size} segments; "
            f"using {max_rw_clusters} instead."
        )
        rw_n_clusters = max_rw_clusters
    if rw_n_clusters < 1:
        raise RuntimeError("RiteWeight requires at least one cluster.")

    weights, delta_history = riteweight(
        features_all,
        seg_start_idx,
        seg_end_idx,
        n_clusters=rw_n_clusters,
        n_iter=rw_n_iter,
        tol=rw_tol,
        tol_window=rw_tol_window,
        avg_last=rw_avg_last,
        seed=rw_seed,
    )

    if write_diag and len(delta_history) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(delta_history) + 1), delta_history, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("L1 delta")
        plt.title("RiteWeight convergence")
        plt.tight_layout()
        plt.savefig(riteweight_png)
        plt.close()

    # =====================================================
    # 3) KMeans on 3N-6 features, automatic k via elbow
    # =====================================================
    X_cluster = features_all
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(X_cluster)

    pca_cluster = None
    if cluster_space == "pca_highdim":
        ncomp = min(pca_cluster_dim, X_cluster.shape[1])
        pca_cluster = PCA(n_components=ncomp)
        X_cluster = pca_cluster.fit_transform(X_cluster)

    best_k, ks, inertias = select_k_for_clustering(
        X_cluster,
        config,
        write_diag=write_diag,
        elbow_png=elbow_png,
    )

    meta_state, dist_to_centroid, thresholds, kmeans = kmeans_metastable_labeling(
        X_cluster,
        n_clusters=best_k,
        quantile=inter_quantile,
        random_state=kmeans_random_state,
        n_init=kmeans_n_init,
    )

    # Pairwise labels for committor-vector training
    pair_labels = None
    pairs = None
    if make_pairwise:
        pair_labels, pairs = build_pairwise_labels(meta_state, best_k)
        print(f"[PAIRWISE] Built C({best_k},2) = {pair_labels.shape[1]} pairwise label columns.")

    # =====================================================
    # 4) Optional: save a debug CSV (smallish: no 3N-6, only labels + CVs + weights)
    # =====================================================
    if save_cv:
        df = pd.DataFrame(colvars_all, columns=headers_ref)
        df.insert(0, "frame", np.arange(n_frames, dtype=np.int64))
        df["weight"] = weights
        df["riteweight_segment_start"] = 0
        df.loc[seg_start_idx, "riteweight_segment_start"] = 1
        df["meta_state"] = meta_state
        df["is_intermediate"] = (meta_state == -1).astype(np.int8)
        df["dist_to_centroid"] = dist_to_centroid
        df["k_selected"] = best_k
        df.to_csv(weights_csv, index=False)
        print(f"[CSV] Saved: {weights_csv}")

    # periodic encoding (if requested) for cvs_to_save only (saved into dataset CV block)
    cv_headers = headers_ref
    cv_data = colvars_all
    if save_cv and cvs_to_save:
        # subset
        idx = [cv_headers.index(c) for c in cvs_to_save]
        cv_data = colvars_all[:, idx].astype(np.float32)
        cv_headers = list(cvs_to_save)

    if periodic and save_cv and len(cv_headers) > 0:
        # Add sin/cos columns (degrees -> radians)
        sincos = []
        sincos_headers = []
        for j, name in enumerate(cv_headers):
            ang = cv_data[:, j] * np.pi / 180.0
            sincos.append(np.sin(ang).astype(np.float32))
            sincos.append(np.cos(ang).astype(np.float32))
            sincos_headers.extend([f"s{name}", f"c{name}"])
        sincos = np.stack(sincos, axis=1) if len(sincos) else np.zeros((n_frames, 0), np.float32)
        cv_data = np.concatenate([cv_data, sincos], axis=1).astype(np.float32)
        cv_headers = cv_headers + sincos_headers

    # =====================================================
    # 5) Save dataset as torch .pt (preferred) or npz
    # =====================================================
    meta = {
        "n_frames": int(n_frames),
        "feature_dim": int(feat_dim),
        "k_selected": int(best_k),
        "pairs": pairs,  # list[(i,j)] or None
        "cv_headers": cv_headers if save_cv else None,
        "config": config,
        "weighting_method": "riteweight",
        "riteweight": {
            "n_clusters": int(rw_n_clusters),
            "n_iter": int(rw_n_iter),
            "tol": float(rw_tol),
            "tol_window": int(rw_tol_window),
            "avg_last": int(rw_avg_last),
            "seed": int(rw_seed),
            "lag": int(rw_lag),
            "n_segments": int(seg_start_idx.size),
            "delta_history": delta_history.tolist(),
        },
        "standardize_features": bool(standardize),
        "cluster_space": cluster_space,
        "pca_cluster_dim": int(pca_cluster_dim) if cluster_space == "pca_highdim" else None,
        "intermediate_quantile": float(inter_quantile),
        "elbow": {"ks": ks, "inertias": inertias, "method": config.get("elbow_method", "knee")},
        "notes": "features are minimal Z-matrix coordinates in float32; weights come from RiteWeight and are normalized to sum=1.",
    }

    if save_format == "pt":
        if not TORCH_OK:
            print("[WARN] torch not available; falling back to NPZ.")
            save_format = "npz"

    dataset_path = infer_dataset_path(output_dir, save_format, config.get("dataset_path"))
    save_dataset(
        dataset_path=dataset_path,
        save_format=save_format,
        features_all=features_all,
        weights=weights,
        meta_state=meta_state,
        dist_to_centroid=dist_to_centroid,
        thresholds=thresholds,
        meta=meta,
        cv_data=(cv_data if save_cv else None),
        pair_labels=(pair_labels if make_pairwise else None),
    )
    print(f"[DATASET] Saved dataset: {dataset_path}")

    # optionally also save a lightweight npy for thresholds
    np.save(os.path.join(output_dir, "kmeans_thresholds.npy"), thresholds)

    # =====================================================
    # 6) Optional: write concatenated DCD
    # =====================================================
    if write_concat_dcd:
        out_dcd = os.path.join(output_dir, "concat.dcd")
        sel_out0 = all_universes[0].select_atoms(selection_output)
        with DCDWriter(out_dcd, sel_out0.n_atoms) as writer:
            for u in all_universes:
                for ts in u.trajectory[::every]:
                    writer.write(u.atoms)
        print(f"[DCD] Saved concatenated DCD: {out_dcd}")

        # copy topology for convenience
        out_psf = os.path.join(output_dir, "concat.psf")
        if not os.path.exists(out_psf):
            try:
                import shutil
                shutil.copy(psf_path, out_psf)
                print(f"[PSF] Copied topology: {out_psf}")
            except Exception:
                pass


def load_multistate_config(config_path):
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if isinstance(raw_config, dict) and "MultiState" in raw_config:
        config = raw_config["MultiState"]
        if not isinstance(config, dict):
            raise ValueError("Config block 'MultiState' must be a mapping.")
        return config

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a mapping at the top level.")

    return raw_config


def main():
    parser = argparse.ArgumentParser(description="Build 3N-6 dataset + weights + auto-k KMeans meta-state labels")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--relabel-only",
        action="store_true",
        help="Reload an existing dataset and recompute only the meta-state labels.",
    )
    args = parser.parse_args()
    config = load_multistate_config(args.config)
    run_pipeline(config, relabel_only=args.relabel_only)


if __name__ == "__main__":
    main()
