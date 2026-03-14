#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================================================
# === Utilities
# =========================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pair_list(n_states):
    return [(i, j) for i in range(n_states) for j in range(i + 1, n_states)]


def weighted_mean_2d(x, y, v, w, xedges, yedges):
    denom, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)
    numer, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w * v)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = numer / denom
    avg[denom <= 0] = np.nan
    return avg, denom


def compute_weighted_centroids(cv_all, weights_all, meta_state_all, cv_headers, planes, n_states):
    centroids_by_plane = {}
    for (cvx, cvy) in planes:
        ix = cv_headers.index(cvx)
        iy = cv_headers.index(cvy)
        out = []
        for s in range(n_states):
            mask = (meta_state_all == s)
            if not np.any(mask):
                out.append((s, np.nan, np.nan, 0.0))
                continue
            w = weights_all[mask]
            ws = float(np.sum(w))
            if ws <= 0:
                out.append((s, np.nan, np.nan, 0.0))
                continue
            cx = float(np.sum(w * cv_all[mask, ix]) / ws)
            cy = float(np.sum(w * cv_all[mask, iy]) / ws)
            out.append((s, cx, cy, ws))
        centroids_by_plane[f"{cvx}__{cvy}"] = out
    return centroids_by_plane


def plot_field_with_centroids(
    xedges, yedges, field, out_path,
    title, xlabel, ylabel,
    centroids=None, state_names=None,
    text_offset=(0.0, 0.0),
    marker="x", ms=40,
    vmin=None, vmax=None, cmap="RdBu_r",
    cbar_label="q",
):
    plt.figure(figsize=(4.0, 3.6), dpi=160)
    pcm = plt.pcolormesh(
        xedges, yedges, field.T,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cb = plt.colorbar(pcm)
    cb.set_label(cbar_label)

    if centroids is not None:
        dx, dy = float(text_offset[0]), float(text_offset[1])
        for (s, cx, cy, wsum) in centroids:
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            plt.scatter([cx], [cy], marker=marker, s=ms)

            if state_names is None or s >= len(state_names) or state_names[s] is None:
                label = f"S{s}"
            else:
                label = str(state_names[s])

            plt.text(cx + dx, cy + dy, label, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =========================================================
# === Self-consistency error E(x)
# =========================================================

def build_A_matrix(pairs, n_states, anchor_state=0):
    """
    Build A for least squares:
      l_ij ≈ s_j - s_i
    Fix gauge by removing anchor_state (set s_anchor=0), solve for remaining N-1 vars.
    A shape: (M, N-1)
    """
    var_index = {}
    col = 0
    for s in range(n_states):
        if s == anchor_state:
            continue
        var_index[s] = col
        col += 1

    M = len(pairs)
    A = np.zeros((M, n_states - 1), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        if j != anchor_state:
            A[k, var_index[j]] += 1.0
        if i != anchor_state:
            A[k, var_index[i]] -= 1.0
    return A, var_index


def compute_E_from_Q(Q, pairs, n_states, eps=1e-4, anchor_state=0, chunk=20000):
    """
    Q: (N, M) predicted pairwise committors in [0,1]
    Return:
      E: (N,) mean squared error between Q and Q_tilde derived from best-fit p(x)
    """
    N, M = Q.shape
    A, var_index = build_A_matrix(pairs, n_states, anchor_state=anchor_state)

    # Precompute pseudo-inverse of A (least squares): s_hat = pinv(A) * l
    # Use pinv for stability; sizes are small (M x (N-1))
    pinvA = np.linalg.pinv(A)  # shape ((N-1), M)

    E = np.empty((N,), dtype=np.float32)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = Q[start:end].astype(np.float64)

        # clip to avoid logit explosion
        q = np.clip(q, eps, 1.0 - eps)
        l = np.log(q) - np.log(1.0 - q)          # (B, M)

        # solve s (B, N-1)
        s_free = l @ pinvA.T                      # (B, N-1)

        # reconstruct full s with anchor=0
        B = s_free.shape[0]
        s_full = np.zeros((B, n_states), dtype=np.float64)
        for s, c in var_index.items():
            s_full[:, s] = s_free[:, c]
        s_full[:, anchor_state] = 0.0

        # softmax to get p
        s_shift = s_full - np.max(s_full, axis=1, keepdims=True)
        exp_s = np.exp(s_shift)
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)  # (B, N)

        # build q_tilde
        q_tilde = np.empty((B, M), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            denom = p[:, i] + p[:, j] + 1e-12
            q_tilde[:, k] = p[:, j] / denom

        # E(x) = mean over pairs
        err = (q - q_tilde) ** 2
        E[start:end] = np.mean(err, axis=1).astype(np.float32)

    return E


# =========================================================
# === Main inference + projection
# =========================================================

def run_infer_project(cfg):
    cfg = cfg["INFER"]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    pack = torch.load(cfg["dataset"], map_location="cpu")
    features = pack["features"].float()
    weights = pack["weights"].float().numpy()

    cv = pack.get("cv", None)
    meta = pack.get("meta", {}) or {}
    cv_headers = meta.get("cv_headers", None)
    if cv is None or cv_headers is None:
        raise RuntimeError("dataset.pt must contain 'cv' and meta['cv_headers'].")
    cv = cv.float().numpy()
    cv_headers = list(cv_headers)

    meta_state = pack.get("meta_state", None)
    if meta_state is None:
        raise RuntimeError("dataset.pt must contain 'meta_state' (needed for centroid overlay/filtering).")
    meta_state = meta_state.cpu().numpy().astype(np.int64)

    # -----------------------------
    # Number of states
    # -----------------------------
    n_states = cfg.get("n_states", None)
    if n_states is None:
        n_states = meta.get("k_selected", None)
    if n_states is None:
        raise ValueError("n_states not specified and not found in dataset meta['k_selected'].")
    n_states = int(n_states)

    pairs = pair_list(n_states)

    # -----------------------------
    # Planes
    # -----------------------------
    planes = cfg["planes"]
    for cvx, cvy in planes:
        if cvx not in cv_headers or cvy not in cv_headers:
            raise ValueError(f"CV plane ({cvx},{cvy}) not found in dataset meta['cv_headers'].")

    # -----------------------------
    # Device & model
    # -----------------------------
    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")

    model = torch.jit.load(cfg["model"], map_location=device)
    model.eval()

    # -----------------------------
    # Infer Q(x) for all frames
    # -----------------------------
    N = features.shape[0]
    batch_size = int(cfg.get("batch_size", 65536))

    with torch.no_grad():
        q_test = model(features[: min(N, 8)].to(device))
    M = int(q_test.shape[1])

    if M != len(pairs):
        raise RuntimeError(f"Model output dim {M} != C(n_states,2)={len(pairs)}")

    Q = np.empty((N, M), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            xb = features[start:end].to(device)
            Q[start:end] = model(xb).detach().float().cpu().numpy()

    if cfg.get("save_q", False):
        np.save(os.path.join(cfg["out_dir"], "Q.npy"), Q)

    # -----------------------------
    # Compute self-consistency error E(x) on FULL frames first
    # (then apply the same mask you use for projection)
    # -----------------------------
    ex_eps = float(cfg.get("E_eps", 1e-4))
    ex_chunk = int(cfg.get("E_chunk", 20000))
    anchor_state = int(cfg.get("E_anchor_state", 0))

    print("[E(x)] computing self-consistency error ...")
    E_all = compute_E_from_Q(Q, pairs, n_states, eps=ex_eps, anchor_state=anchor_state, chunk=ex_chunk)
    np.save(os.path.join(cfg["out_dir"], "E_x.npy"), E_all)
    print("[E(x)] saved raw E(x) as E_x.npy")

    # -----------------------------
    # Frame filtering for projection
    # -----------------------------
    mask = np.ones(N, dtype=bool)
    if cfg.get("exclude_intermediate", False):
        mask &= (meta_state != -1)
    if cfg.get("only_state", None) is not None:
        mask &= (meta_state == int(cfg["only_state"]))

    if not np.any(mask):
        raise RuntimeError("No frames left after filtering (exclude_intermediate/only_state).")

    # Apply mask for projection
    cv_m = cv[mask]
    w_m = weights[mask]
    Q_m = Q[mask]
    meta_state_m = meta_state[mask]
    E_m = E_all[mask]

    ws = float(np.sum(w_m))
    if ws <= 0:
        raise RuntimeError("Sum of weights after filtering is zero.")
    w_m = w_m / ws

    # -----------------------------
    # Centroids (weighted average in CV space)
    # -----------------------------
    plot_centroids = bool(cfg.get("plot_centroids", True))
    centroid_use_filtered = bool(cfg.get("centroid_use_filtered_frames", True))
    state_names = cfg.get("state_names", None)
    text_offset = cfg.get("centroid_text_offset", [0.0, 0.0])
    marker = cfg.get("centroid_marker", "x")
    ms = float(cfg.get("centroid_marker_size", 40))

    if plot_centroids:
        if centroid_use_filtered:
            centroids_by_plane = compute_weighted_centroids(
                cv_m, w_m, meta_state_m, cv_headers, planes, n_states
            )
        else:
            base_mask = np.ones(N, dtype=bool)
            if cfg.get("exclude_intermediate", False):
                base_mask &= (meta_state != -1)
            cv_c = cv[base_mask]
            w_c = weights[base_mask]
            w_c = w_c / (np.sum(w_c) + 1e-12)
            meta_c = meta_state[base_mask]
            centroids_by_plane = compute_weighted_centroids(
                cv_c, w_c, meta_c, cv_headers, planes, n_states
            )
    else:
        centroids_by_plane = {}

    # -----------------------------
    # Projection: all pairs × all planes (q_ij)
    # + extra: E(x) heatmap per plane
    # -----------------------------
    bins = int(cfg.get("bins", 60))
    fmt = cfg.get("format", "png")

    # E(x) color scaling (optional): robust by percentiles
    E_vmin = cfg.get("E_vmin", None)
    E_vmax = cfg.get("E_vmax", None)
    if E_vmin is None:
        E_vmin = float(np.nanpercentile(E_m, 5))
    if E_vmax is None:
        E_vmax = float(np.nanpercentile(E_m, 95))

    for cvx, cvy in planes:
        ix = cv_headers.index(cvx)
        iy = cv_headers.index(cvy)
        x = cv_m[:, ix]
        y = cv_m[:, iy]

        if cfg.get("xlim", None) is None:
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        else:
            xmin, xmax = map(float, cfg["xlim"])

        if cfg.get("ylim", None) is None:
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        else:
            ymin, ymax = map(float, cfg["ylim"])

        xedges = np.linspace(xmin, xmax, bins + 1)
        yedges = np.linspace(ymin, ymax, bins + 1)

        subdir = os.path.join(cfg["out_dir"], f"{cvx}__{cvy}")
        os.makedirs(subdir, exist_ok=True)

        centroid_list = centroids_by_plane.get(f"{cvx}__{cvy}", None)

        # ---- 1) E(x) projection heatmap ----
        E_field, _ = weighted_mean_2d(x, y, E_m, w_m, xedges, yedges)
        out_E = os.path.join(subdir, f"E__{cvx}__{cvy}.{fmt}")
        plot_field_with_centroids(
            xedges, yedges, E_field, out_E,
            title=f"E(x) on ({cvx},{cvy})",
            xlabel=cvx, ylabel=cvy,
            centroids=centroid_list if plot_centroids else None,
            state_names=state_names,
            text_offset=text_offset,
            marker=marker,
            ms=ms,
            vmin=E_vmin, vmax=E_vmax,
            cmap="viridis",
            cbar_label="E(x)",
        )

        # ---- 2) all q_ij projections ----
        for p_idx, (i, j) in enumerate(pairs):
            q_field, _ = weighted_mean_2d(x, y, Q_m[:, p_idx], w_m, xedges, yedges)
            out_fig = os.path.join(subdir, f"q_{i}_{j}__{cvx}__{cvy}.{fmt}")
            plot_field_with_centroids(
                xedges, yedges, q_field, out_fig,
                title=f"q_{i}_{j} on ({cvx},{cvy})",
                xlabel=cvx, ylabel=cvy,
                centroids=centroid_list if plot_centroids else None,
                state_names=state_names,
                text_offset=text_offset,
                marker=marker,
                ms=ms,
                vmin=0.0, vmax=1.0,
                cmap="RdBu_r",
                cbar_label="q",
            )

        print(f"[DONE] Plane ({cvx},{cvy}) → E(x) + {len(pairs)} q_ij figures")

    print("All projections finished.")


# =========================================================
# === CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Infer committor vector and project all pairs + self-consistency E(x) onto CV planes (config-driven)."
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if "INFER" not in cfg:
        raise KeyError("Config must contain a top-level key: INFER")

    run_infer_project(cfg)


if __name__ == "__main__":
    main()
