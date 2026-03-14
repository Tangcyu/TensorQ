#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import torch


# =========================================================
# === q_ij -> p(x) reconstruction (self-consistent simplex)
# =========================================================

def pair_list(K: int):
    return [(i, j) for i in range(K) for j in range(i + 1, K)]


def build_A_matrix(pairs, K, anchor_state=0):
    """
    logit(q_ij) ≈ s_j - s_i, fix gauge by s_anchor=0.
    A: (M, K-1)
    """
    var_index = {}
    col = 0
    for s in range(K):
        if s == anchor_state:
            continue
        var_index[s] = col
        col += 1

    M = len(pairs)
    A = np.zeros((M, K - 1), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        if j != anchor_state:
            A[k, var_index[j]] += 1.0
        if i != anchor_state:
            A[k, var_index[i]] -= 1.0
    return A, var_index


def reconstruct_p_from_Q(Q, K, eps=1e-4, anchor_state=0, chunk=20000):
    """
    Q: (N, M) predicted q_ij in [0,1], M=C(K,2)
    return P: (N, K) with rows on simplex
    """
    pairs = pair_list(K)
    N, M = Q.shape
    if M != len(pairs):
        raise ValueError(f"Q dim mismatch: got M={M}, expected C(K,2)={len(pairs)}")

    A, var_index = build_A_matrix(pairs, K, anchor_state=anchor_state)
    pinvA = np.linalg.pinv(A)  # (K-1, M)

    P = np.empty((N, K), dtype=np.float32)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = Q[start:end].astype(np.float64)
        q = np.clip(q, eps, 1.0 - eps)
        l = np.log(q) - np.log(1.0 - q)  # (B,M)

        s_free = l @ pinvA.T  # (B,K-1)
        B = s_free.shape[0]
        s_full = np.zeros((B, K), dtype=np.float64)
        for s, c in var_index.items():
            s_full[:, s] = s_free[:, c]
        s_full[:, anchor_state] = 0.0

        # softmax -> p
        s_shift = s_full - np.max(s_full, axis=1, keepdims=True)
        exp_s = np.exp(s_shift)
        p = exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-12)
        P[start:end] = p.astype(np.float32)

    return P


# =========================================================
# === MSM estimation with soft assignment and weights
# =========================================================

def build_pairs_from_traj(N, lag, traj_id=None, allow_cross=False):
    idx0 = np.arange(0, N - lag, dtype=np.int64)
    idx1 = idx0 + lag
    if traj_id is None or allow_cross:
        return idx0, idx1
    traj_id = np.asarray(traj_id)
    m = (traj_id[idx0] == traj_id[idx1])
    return idx0[m], idx1[m]


def compute_counts_soft(P, weights, idx0, idx1, normalize_weights=True):
    """
    C_ij = sum_t w_t p_i(t) p_j(t+lag)
    """
    w = weights.astype(np.float64)
    if normalize_weights:
        w = w / (np.sum(w) + 1e-12)

    P0 = P[idx0].astype(np.float64)  # (T,K)
    P1 = P[idx1].astype(np.float64)  # (T,K)
    wt = w[idx0][:, None]            # (T,1)

    # weighted outer product sum: (K,K)
    # C = (P0^T * wt^T) @ P1
    C = (P0 * wt).T @ P1
    return C.astype(np.float64)


def row_normalize(C):
    row_sum = np.sum(C, axis=1, keepdims=True) + 1e-18
    return (C / row_sum).astype(np.float64)


def stationary_from_T(T, max_iter=200000, tol=1e-12, teleport=1e-8):
    """
    Robust stationary distribution for (possibly imperfect) row-stochastic T.
    Uses power iteration on a slightly 'teleported' chain to ensure ergodicity.
    """
    K = T.shape[0]
    # make a valid stochastic matrix even if some rows are all zeros
    row_sum = T.sum(axis=1, keepdims=True)
    T_fix = T.copy()
    zero_rows = (row_sum[:, 0] <= 0)
    if np.any(zero_rows):
        T_fix[zero_rows] = 1.0 / K
        row_sum = T_fix.sum(axis=1, keepdims=True)

    T_fix = T_fix / row_sum

    # teleport (like PageRank) to avoid reducibility
    T_fix = (1 - teleport) * T_fix + teleport * (np.ones((K, K)) / K)

    pi = np.ones(K, dtype=np.float64) / K
    for _ in range(max_iter):
        pi_new = pi @ T_fix
        if np.linalg.norm(pi_new - pi, 1) < tol:
            pi = pi_new
            break
        pi = pi_new

    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Failed to compute a valid stationary distribution (pi is non-finite or zero).")
    return (pi / s).astype(np.float64)



def compute_frame_weights_from_pi(P, pi, weights_old=None, mode="replace"):
    """
    w_new(t) ∝ sum_i pi_i p_i(t)
    If mode == "replace": ignore weights_old, return normalized w_new
    If mode == "rescale": w_new ∝ weights_old * (sum_i pi_i p_i(t))
    """
    s = (P.astype(np.float64) @ pi.reshape(-1, 1)).reshape(-1)  # (N,)
    s = np.maximum(s, 0.0)

    if weights_old is None or mode == "replace":
        w_new = s
    elif mode == "rescale":
        w_new = weights_old.astype(np.float64) * s
    else:
        raise ValueError("mode must be 'replace' or 'rescale'")

    w_new = w_new / (np.sum(w_new) + 1e-18)
    return w_new.astype(np.float64)


# =========================================================
# === Main
# =========================================================

def run(cfg):
    cfg = cfg["MSM"]
    out_dir = cfg.get("out_dir", "msm_out")
    os.makedirs(out_dir, exist_ok=True)

    # ---- load dataset ----
    pack = torch.load(cfg["dataset"], map_location="cpu")

    weights = pack.get("weights", None)
    if weights is None:
        raise RuntimeError("dataset.pt must contain 'weights'")
    weights = weights.float().numpy()

    meta = pack.get("meta", {}) or {}
    traj_id = meta.get("traj_id", None)  # optional

    # ---- load Q ----
    # either provide Q.npy, or infer from model + features
    if "Q_npy" in cfg and cfg["Q_npy"] is not None:
        Q = np.load(cfg["Q_npy"]).astype(np.float32)
    else:
        # infer from model
        features = pack.get("features", None)
        if features is None:
            raise RuntimeError("dataset.pt must contain 'features' if you want to infer Q from model.")
        features = features.float()

        device_str = cfg.get("device", "cpu")
        device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")

        model = torch.jit.load(cfg["model"], map_location=device)
        model.eval()

        bs = int(cfg.get("batch_size", 65536))
        N = features.shape[0]
        Q_list = []
        with torch.no_grad():
            for s in range(0, N, bs):
                e = min(N, s + bs)
                xb = features[s:e].to(device)
                qb = model(xb).detach().float().cpu().numpy()
                Q_list.append(qb)
        Q = np.vstack(Q_list).astype(np.float32)
        np.save(os.path.join(out_dir, "Q.npy"), Q)

    N = Q.shape[0]

    # ---- determine K from Q dim or from meta ----
    K = cfg.get("n_states", None)
    if K is None:
        K = meta.get("k_selected", None)
    if K is None:
        # infer K from M=C(K,2)
        M = Q.shape[1]
        # solve K(K-1)/2=M
        K = int((1 + np.sqrt(1 + 8 * M)) / 2)
        if K * (K - 1) // 2 != M:
            raise RuntimeError("Cannot infer K from Q dim; please specify MSM.n_states")
    K = int(K)

    # ---- reconstruct P(x) from Q ----
    P = reconstruct_p_from_Q(
        Q, K,
        eps=float(cfg.get("E_eps", 1e-4)),
        anchor_state=int(cfg.get("E_anchor_state", 0)),
        chunk=int(cfg.get("E_chunk", 20000)),
    )
    np.save(os.path.join(out_dir, "P.npy"), P)

    # ---- build transition pairs ----
    lag = int(cfg.get("lag", 1))
    allow_cross = bool(cfg.get("allow_cross_traj_pairs", False))
    idx0, idx1 = build_pairs_from_traj(N, lag, traj_id=traj_id, allow_cross=allow_cross)
    print(f"[pairs] lag={lag}, pairs={len(idx0)}")

    # ---- counts and T ----
    C = compute_counts_soft(
        P, weights, idx0, idx1,
        normalize_weights=bool(cfg.get("normalize_input_weights", True)),
    )
    alpha = float(cfg.get("pseudocount", 1e-12))
    C = C + alpha
    T = row_normalize(C)

    np.save(os.path.join(out_dir, "C.npy"), C)
    np.save(os.path.join(out_dir, "T.npy"), T)

    # ---- stationary distribution ----
    pi = stationary_from_T(T)
    np.save(os.path.join(out_dir, "pi.npy"), pi)

    # ---- new frame weights ----
    mode = str(cfg.get("new_weight_mode", "replace"))  # replace / rescale
    w_new = compute_frame_weights_from_pi(P, pi, weights_old=weights, mode=mode)
    np.save(os.path.join(out_dir, "weights_new.npy"), w_new)

    # save a small csv
    import pandas as pd
    df = pd.DataFrame({
        "frame": np.arange(N, dtype=np.int64),
        "weight_old": weights.astype(np.float64),
        "weight_new": w_new.astype(np.float64),
    })
    df.to_csv(os.path.join(out_dir, "weights_new.csv"), index=False)

    print(f"[MSM] saved: C.npy, T.npy, pi.npy, weights_new.npy, weights_new.csv in {out_dir}")
    print(f"[MSM] pi (sum={pi.sum():.6f}) = {pi}")


def main():
    ap = argparse.ArgumentParser(description="Build MSM from existing q_ij, compute transition matrix and new weights.")
    ap.add_argument("--config", required=True, help="YAML config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    if "MSM" not in cfg:
        raise KeyError("Config must contain top-level key: MSM")
    run(cfg)


if __name__ == "__main__":
    main()
