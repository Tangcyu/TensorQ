#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reweight unbiased-shooting trajectories via PCA(2D) free-energy reconstruction,
then cluster conformations into meta-stable states using KMeans with AUTOMATIC k
(from elbow), in the 3N-6 minimal internal-coordinate feature space.

Outputs a compact dataset (prefer torch .pt, fallback .npz) containing:
- features: (n_frames, 3N-6) float32   # for committor training
- weights:  (n_frames,) float32
- meta_state: (n_frames,) int64, in {0..k-1, -1}  # -1 = intermediate
- dist_to_centroid: (n_frames,) float32
- optional: CV matrix and headers
- optional: pairwise committor label matrix for all (i,j), i<j, with values {0,1,-1}
- optional: concatenated DCD + diagnostic figures

YAML keys (minimum):
  topology_file: "xxx.psf"
  dcd_folder: "path/to/dcds"
  output_dir: "out"
  match: "shoot_"          # filename prefix match for .dcd
  sel_weights: "protein and not name H*"
  sel_output:  "protein and not name H*"
  every: 1
  temperature: 300
  split: 0.1               # fraction for init/final histogram
  ndim: 2                  # PCA dim for reweighting projection (must be 2 here)
  colvars_mismatch: true

Elbow+KMeans:
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

from MDAnalysis import Universe
from MDAnalysis.coordinates.DCD import DCDWriter
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals

# Boltzmann constant in kcal/mol/K
kB = 0.0019872041

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


# =========================================================
# === Main
# =========================================================

def run_pipeline(config):
    psf_path = config["topology_file"]
    dcd_folder = config["dcd_folder"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ---- IO names ----
    dataset_base = os.path.join(output_dir, "dataset")
    weights_csv = os.path.join(output_dir, "weights_and_labels.csv")  # optional debug
    deltaF_png = os.path.join(output_dir, "deltaF_pca.png")
    elbow_png = os.path.join(output_dir, "elbow_kmeans.png")

    # ---- selections ----
    selection_weights = config.get("sel_weights", "protein and not name H*")
    selection_output = config.get("sel_output", "protein and not name H*")

    # ---- sampling / weighting ----
    temperature = float(config.get("temperature", 300))
    ndim = int(config.get("ndim", 2))
    if ndim != 2:
        raise ValueError("For the current reweighting implementation, ndim must be 2 (2D PCA grid).")
    split = float(config.get("split", 0.1))
    every = int(config.get("every", 1))
    index_mismatch = bool(config.get("colvars_mismatch", True))
    periodic = bool(config.get("periodic", False))
    beta = 1.0 / (kB * temperature)

    # ---- elbow/kmeans ----
    kmin = int(config.get("kmin", 2))
    kmax = int(config.get("kmax", 12))
    elbow_method = config.get("elbow_method", "knee")
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

    make_pairwise = bool(config.get("make_pairwise_committor", True))

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
    all_desc2 = []
    all_colvars = []
    all_universes = []
    headers_ref = None

    # =====================================================
    # 1) Load trajectories, build 3N-6 features, and 2D PCA descriptor for weights
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

        # 2D descriptor for reweighting: PCA on this traj's features
        pca2 = PCA(n_components=2)
        desc2 = pca2.fit_transform(feats).astype(np.float32)

        headers, colvars_data = read_colvars(colvars_path, index_mismatch=index_mismatch)
        colvars_data = colvars_data[::every]
        if len(colvars_data) != len(desc2):
            raise ValueError(f"Frame mismatch: {dcd_path} (colvars {len(colvars_data)} vs traj {len(desc2)})")

        if headers_ref is None:
            headers_ref = headers
        else:
            if headers != headers_ref:
                raise ValueError("Colvars headers differ across trajectories. Please unify colvars outputs.")

        all_features.append(feats)
        all_desc2.append(desc2)
        all_colvars.append(colvars_data.astype(np.float32))
        all_universes.append(sel_out.universe)

    if len(all_features) == 0:
        raise RuntimeError("No valid trajectories after filtering colvars.")

    features_all = np.vstack(all_features).astype(np.float32)
    desc2_all = np.vstack(all_desc2).astype(np.float32)
    colvars_all = np.vstack(all_colvars).astype(np.float32)

    n_frames, feat_dim = features_all.shape
    print(f"[INFO] Total frames: {n_frames}, feature dim (3N-6): {feat_dim}, colvars dim: {colvars_all.shape[1]}")

    # =====================================================
    # 2) Reweighting weights from 2D descriptor (same idea as your current script)
    # =====================================================
    # Init/final sets per trajectory
    desc_init = np.vstack([d[: max(1, int(split * len(d)))] for d in all_desc2]).astype(np.float32)
    desc_final = np.vstack([d[-max(1, int(split * len(d))):] for d in all_desc2]).astype(np.float32)

    # histogram bins
    xbins = np.linspace(np.min(desc2_all[:, 0]), np.max(desc2_all[:, 0]), 10)
    ybins = np.linspace(np.min(desc2_all[:, 1]), np.max(desc2_all[:, 1]), 10)
    H_init, _, _ = np.histogram2d(desc_init[:, 0], desc_init[:, 1], bins=(xbins, ybins), density=True)
    H_final, _, _ = np.histogram2d(desc_final[:, 0], desc_final[:, 1], bins=(xbins, ybins), density=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        deltaF = -kB * temperature * np.log(H_final / (H_init + 1e-10))
        finite = np.isfinite(deltaF)
        if not np.any(finite):
            raise RuntimeError("deltaF is entirely non-finite. Adjust bins/split.")
        deltaF -= np.nanmin(deltaF[finite])

    if write_diag:
        Xg, Yg = np.meshgrid(0.5 * (xbins[:-1] + xbins[1:]), 0.5 * (ybins[:-1] + ybins[1:]))
        plt.figure(figsize=(6, 5))
        plt.contourf(Xg, Yg, deltaF.T, levels=20, cmap="viridis")
        plt.colorbar(label="ΔF (kcal/mol)")
        plt.xlabel("PC1 (per-traj PCA on 3N-6)")
        plt.ylabel("PC2 (per-traj PCA on 3N-6)")
        plt.title("ΔF in 2D PCA projection")
        plt.tight_layout()
        plt.savefig(deltaF_png)
        plt.close()

    # compute weights by bin lookup
    weights = np.zeros(n_frames, dtype=np.float32)
    for i, (x, y) in enumerate(desc2_all):
        ix = np.digitize(x, xbins) - 1
        iy = np.digitize(y, ybins) - 1
        if 0 <= ix < deltaF.shape[0] and 0 <= iy < deltaF.shape[1]:
            df_ = deltaF[ix, iy]
            weights[i] = np.exp(-beta * df_) if np.isfinite(df_) else 0.0
        else:
            weights[i] = 0.0

    s = float(np.sum(weights))
    if s <= 0:
        raise RuntimeError("All weights are zero. Check ΔF binning / PCA projection / split.")
    weights /= s

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
        "standardize_features": bool(standardize),
        "cluster_space": cluster_space,
        "pca_cluster_dim": int(pca_cluster_dim) if cluster_space == "pca_highdim" else None,
        "intermediate_quantile": float(inter_quantile),
        "elbow": {"ks": ks, "inertias": inertias.tolist(), "method": elbow_method},
        "notes": "features are minimal Z-matrix (bonds, angles, dihedrals) in float32; weights normalized to sum=1.",
    }

    if save_format == "pt":
        if not TORCH_OK:
            print("[WARN] torch not available; falling back to NPZ.")
            save_format = "npz"

    if save_format == "pt":
        out_pt = dataset_base + ".pt"
        pack = {
            "features": torch.from_numpy(features_all),                 # float32
            "weights": torch.from_numpy(weights.astype(np.float32)),    # float32
            "meta_state": torch.from_numpy(meta_state.astype(np.int64)),# int64
            "dist_to_centroid": torch.from_numpy(dist_to_centroid.astype(np.float32)),
            "meta": meta,
        }
        if save_cv:
            pack["cv"] = torch.from_numpy(cv_data.astype(np.float32))
        if make_pairwise and pair_labels is not None:
            pack["pair_labels"] = torch.from_numpy(pair_labels.astype(np.int8))
        # save cluster transforms if you want reproducibility
        # (torch can store pickled sklearn objects but it’s brittle; better store params only)
        torch.save(pack, out_pt)
        print(f"[DATASET] Saved torch dataset: {out_pt}")

        # optionally also save a lightweight npy for thresholds
        np.save(os.path.join(output_dir, "kmeans_thresholds.npy"), thresholds)
    else:
        out_npz = dataset_base + ".npz"
        npz_dict = {
            "features": features_all.astype(np.float32),
            "weights": weights.astype(np.float32),
            "meta_state": meta_state.astype(np.int64),
            "dist_to_centroid": dist_to_centroid.astype(np.float32),
            "thresholds": thresholds.astype(np.float32),
        }
        if save_cv:
            npz_dict["cv"] = cv_data.astype(np.float32)
            # headers go in meta json-like
        if make_pairwise and pair_labels is not None:
            npz_dict["pair_labels"] = pair_labels.astype(np.int8)

        # store meta as yaml string (portable)
        meta_yaml = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
        npz_dict["meta_yaml"] = np.array([meta_yaml], dtype=object)
        np.savez_compressed(out_npz, **npz_dict)
        print(f"[DATASET] Saved NPZ dataset: {out_npz}")

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


def main():
    parser = argparse.ArgumentParser(description="Build 3N-6 dataset + weights + auto-k KMeans meta-state labels")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config)


if __name__ == "__main__":
    main()
