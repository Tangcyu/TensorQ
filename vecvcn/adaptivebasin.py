#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import torch
import pandas as pd

import hdbscan


# =========================================================
# === Pair utilities
# =========================================================

def pair_list(n_states: int):
    return [(i, j) for i in range(n_states) for j in range(i + 1, n_states)]


def build_pair_labels_from_state(meta_state: np.ndarray, n_states: int) -> np.ndarray:
    """
    pair_labels: (N, M) with values in {-1,0,1}
    canonical pairs order: (0,1),(0,2)...(0,K-1),(1,2)...(K-2,K-1)
    if frame in state i:
      for pair (a,b):
        label=0 if i==a
        label=1 if i==b
        else -1
    """
    pairs = pair_list(n_states)
    N = meta_state.shape[0]
    M = len(pairs)
    pl = np.full((N, M), -1, dtype=np.int8)
    for k, (a, b) in enumerate(pairs):
        pl[meta_state == a, k] = 0
        pl[meta_state == b, k] = 1
    return pl


# =========================================================
# === Reconstruct p(x) and E(x) from Q (same as before)
# =========================================================

def build_A_matrix(pairs, n_states, anchor_state=0):
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


def reconstruct_p_from_Q(Q, pairs, n_states, eps=1e-4, anchor_state=0, chunk=20000):
    N, M = Q.shape
    A, var_index = build_A_matrix(pairs, n_states, anchor_state=anchor_state)
    pinvA = np.linalg.pinv(A)  # (N-1, M)

    P = np.empty((N, n_states), dtype=np.float32)
    E = np.empty((N,), dtype=np.float32)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = Q[start:end].astype(np.float64)
        q = np.clip(q, eps, 1.0 - eps)
        l = np.log(q) - np.log(1.0 - q)  # (B,M)

        s_free = l @ pinvA.T  # (B,N-1)
        B = s_free.shape[0]
        s_full = np.zeros((B, n_states), dtype=np.float64)
        for s, c in var_index.items():
            s_full[:, s] = s_free[:, c]
        s_full[:, anchor_state] = 0.0

        s_shift = s_full - np.max(s_full, axis=1, keepdims=True)
        exp_s = np.exp(s_shift)
        p = exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-12)
        P[start:end] = p.astype(np.float32)

        q_tilde = np.empty((B, M), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            denom = p[:, i] + p[:, j] + 1e-12
            q_tilde[:, k] = p[:, j] / denom
        E[start:end] = np.mean((q - q_tilde) ** 2, axis=1).astype(np.float32)

    return P, E


# =========================================================
# === Feature space helpers (internal coords)
# =========================================================

def standardize(X):
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd, mu, sd


def weighted_centroids(X, w, labels, n_states):
    N, D = X.shape
    C = np.full((n_states, D), np.nan, dtype=np.float64)
    for s in range(n_states):
        m = (labels == s)
        if not np.any(m):
            continue
        ws = float(np.sum(w[m]))
        if ws <= 0:
            continue
        C[s] = np.sum(X[m] * w[m, None], axis=0) / ws
    return C.astype(np.float32)


def tighten_by_distance(X, labels, centroids, q=0.9, standardize_features=True):
    """
    per-state keep closest q-quantile, others -> -1
    """
    new_labels = labels.copy()
    Xf = X.astype(np.float64)
    Cf = centroids.astype(np.float64)

    if standardize_features:
        Xs, mu, sd = standardize(Xf)
        Cs = (Cf - mu) / sd
        X_use, C_use = Xs, Cs
    else:
        X_use, C_use = Xf, Cf

    K = Cf.shape[0]
    for s in range(K):
        m = (new_labels == s)
        if not np.any(m):
            continue
        dist = np.linalg.norm(X_use[m] - C_use[s], axis=1)
        thr = np.quantile(dist, q)
        idx = np.where(m)[0]
        far = idx[dist > thr]
        new_labels[far] = -1
    return new_labels


# =========================================================
# === p_core selection (adaptive)
# =========================================================

def choose_p_core(p_max, E, w, mode, fixed, target_core_frac, E_core_quantile):
    if mode == "fixed":
        return float(fixed)

    if mode == "quantile":
        # pick threshold so that core fraction ~ target_core_frac
        return float(np.quantile(p_max, 1.0 - target_core_frac))

    if mode == "quantile_lowE":
        # restrict to low-E subset first
        E_thr = np.quantile(E, E_core_quantile)
        m = E <= E_thr
        if np.sum(m) < 100:
            # fallback
            return float(np.quantile(p_max, 1.0 - target_core_frac))
        return float(np.quantile(p_max[m], 1.0 - target_core_frac))

    raise ValueError(f"Unknown p_core_mode: {mode}")


# =========================================================
# === Candidate selection & HDBSCAN discovery
# =========================================================

def select_candidates(E, w, E_high_quantile, top_weight_frac, min_candidates):
    """
    先取 E 高分位，再在其中按 w 取 top_weight_frac
    """
    E_thr = np.quantile(E, E_high_quantile)
    idx = np.where(E >= E_thr)[0]
    if idx.size == 0:
        return np.array([], dtype=np.int64)

    # take top weights within idx
    ww = w[idx]
    order = np.argsort(-ww)
    take = max(int(np.ceil(top_weight_frac * len(order))), min_candidates)
    take = min(take, len(order))
    return idx[order[:take]].astype(np.int64)


def hdbscan_new_basins(Xcand, hcfg):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hcfg.get("min_cluster_size", 300)),
        min_samples=int(hcfg.get("min_samples", 50)),
        cluster_selection_epsilon=float(hcfg.get("cluster_selection_epsilon", 0.0)),
        cluster_selection_method=str(hcfg.get("cluster_selection_method", "eom")),
        prediction_data=False,
    )
    labels = clusterer.fit_predict(Xcand)
    probs = clusterer.probabilities_
    return labels.astype(np.int32), probs.astype(np.float32)


def summarize_clusters(cluster_labels, cluster_probs, cand_indices, w_full, min_weight_frac, prob_thr):
    """
    Return accepted clusters sorted by weight.
    Each cluster: dict with keys: cid, members (global indices), weight_sum
    """
    accepted = []
    total_w = float(np.sum(w_full))
    for cid in np.unique(cluster_labels):
        if cid < 0:
            continue
        m = cluster_labels == cid
        if not np.any(m):
            continue
        # membership probability filter
        m2 = m & (cluster_probs >= prob_thr)
        if not np.any(m2):
            continue
        members = cand_indices[m2]
        wsum = float(np.sum(w_full[members]))
        if wsum / total_w < min_weight_frac:
            continue
        accepted.append({"cid": int(cid), "members": members, "weight_sum": wsum})

    accepted.sort(key=lambda d: d["weight_sum"], reverse=True)
    return accepted


# =========================================================
# === Main driver
# =========================================================

def run(cfg):
    cfg = cfg["ADAPTIVE"]
    out_dir = cfg.get("out_dir", "adaptive_out")
    os.makedirs(out_dir, exist_ok=True)

    # ---- load dataset ----
    pack = torch.load(cfg["dataset"], map_location="cpu")
    features = pack["features"].float().numpy()  # internal feature space (N,D)
    weights = pack["weights"].float().numpy()
    meta = pack.get("meta", {}) or {}

    meta_state = pack.get("meta_state", None)
    if meta_state is None:
        raise RuntimeError("dataset.pt must contain meta_state.")
    meta_state = meta_state.cpu().numpy().astype(np.int64)

    K = meta.get("k_selected", None)
    if K is None:
        raise RuntimeError("dataset meta must contain k_selected.")
    K = int(K)

    N, D = features.shape

    # normalize weights
    w = weights.astype(np.float64)
    w = w / (np.sum(w) + 1e-12)
    w = w.astype(np.float32)

    # ---- model ----
    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")
    model = torch.jit.load(cfg["model"], map_location=device)
    model.eval()

    bs = int(cfg.get("batch_size", 65536))

    # ---- reconstruct config ----
    eps = float(cfg.get("E_eps", 1e-4))
    anchor = int(cfg.get("E_anchor_state", 0))
    chunk = int(cfg.get("E_chunk", 20000))

    # candidate params
    E_high_q = float(cfg.get("E_high_quantile", 0.95))
    top_weight_frac = float(cfg.get("top_weight_frac", 0.5))
    min_candidates = int(cfg.get("min_candidates", 2000))

    # HDBSCAN params
    hcfg = cfg.get("hdbscan", {})
    min_w_frac = float(cfg.get("min_cluster_weight_frac", 0.002))
    max_new = int(cfg.get("max_new_states_per_iter", 5))
    prob_thr = float(cfg.get("assign_threshold_prob", 0.90))

    # refinement params
    p_mode = str(cfg.get("p_core_mode", "quantile_lowE"))
    p_fixed = float(cfg.get("p_core_fixed", 0.90))
    target_core_frac = float(cfg.get("target_core_frac", 0.20))
    E_core_q = float(cfg.get("E_core_quantile", 0.30))
    use_dist = bool(cfg.get("use_distance_core", True))
    dist_q = float(cfg.get("distance_quantile", 0.90))
    std_feat = bool(cfg.get("standardize_features_for_distance", True))

    # convergence
    max_iters = int(cfg.get("max_iters", 10))
    tol_label_change = float(cfg.get("tol_label_change", 1e-3))

    save_debug = bool(cfg.get("save_debug_csv", True))

    labels = meta_state.copy()

    for it in range(1, max_iters + 1):
        print(f"\n========== ITER {it} (K={K}) ==========")
        pairs = pair_list(K)
        M = len(pairs)

        # ---- infer Q ----
        Q = np.empty((N, M), dtype=np.float32)
        with torch.no_grad():
            # sanity check dim
            t = model(torch.from_numpy(features[: min(N, 8)]).float().to(device))
            if int(t.shape[1]) != M:
                raise RuntimeError(f"Model output dim {int(t.shape[1])} != C(K,2)={M}; need retrain for current K.")
            for s in range(0, N, bs):
                e = min(N, s + bs)
                xb = torch.from_numpy(features[s:e]).float().to(device)
                Q[s:e] = model(xb).detach().float().cpu().numpy()

        # ---- reconstruct P and E ----
        P, E = reconstruct_p_from_Q(Q, pairs, K, eps=eps, anchor_state=anchor, chunk=chunk)
        p_max = np.max(P, axis=1)
        p_arg = np.argmax(P, axis=1)

        Ew = float(np.sum(w * E))
        print(f"[diag] weighted mean E = {Ew:.6e}")

        iter_dir = os.path.join(out_dir, f"iter_{it:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        np.save(os.path.join(iter_dir, "E_x.npy"), E)
        np.save(os.path.join(iter_dir, "P_x.npy"), P)

        # =====================================================
        # (A) Discover NEW basins from high E regions via HDBSCAN
        # =====================================================
        cand_idx = select_candidates(E, w, E_high_q, top_weight_frac, min_candidates)
        print(f"[discover] candidates: {cand_idx.size}")

        n_new_added = 0
        if cand_idx.size >= int(hcfg.get("min_cluster_size", 300)):
            Xcand = features[cand_idx]
            # standardize candidate features for clustering
            Xcand_std, _, _ = standardize(Xcand.astype(np.float64))
            Xcand_std = Xcand_std.astype(np.float32)

            clabels, cprobs = hdbscan_new_basins(Xcand_std, hcfg)
            accepted = summarize_clusters(clabels, cprobs, cand_idx, w, min_w_frac, prob_thr)

            if accepted:
                print(f"[discover] accepted clusters: {len(accepted)} (pre-cap)")
                accepted = accepted[:max_new]
                for c in accepted:
                    # add as new state id = K + n_new_added
                    new_state_id = K + n_new_added
                    labels[c["members"]] = new_state_id
                    print(f"  + new state {new_state_id} from cluster {c['cid']} "
                          f"(members={len(c['members'])}, wsum={c['weight_sum']:.3e})")
                    n_new_added += 1
            else:
                print("[discover] no acceptable clusters found.")
        else:
            print("[discover] not enough candidates for clustering.")

        if n_new_added > 0:
            K = K + n_new_added
            print(f"[discover] K increased to {K}")

        # =====================================================
        # (B) Refinement: align labels to q=0/1 semantics via p(x)
        #     (core from p_max + optional distance tighten in internal space)
        # =====================================================
        # recompute p_core (adaptive)
        p_core = choose_p_core(p_max, E, w, p_mode, p_fixed, target_core_frac, E_core_q)
        print(f"[refine] p_core = {p_core:.4f} (mode={p_mode})")

        new_labels = np.full((N,), -1, dtype=np.int64)
        core_mask = p_max >= p_core
        new_labels[core_mask] = p_arg[core_mask].astype(np.int64)

        # keep any newly added states' members as core candidates too
        # (they were assigned directly; ensure they stay labeled)
        newly_assigned = labels >= 0
        # but do not override confident p-based labels; just fill gaps
        fill_mask = (new_labels == -1) & newly_assigned
        new_labels[fill_mask] = labels[fill_mask]

        if use_dist:
            centroids = weighted_centroids(features, w, new_labels, K)
            new_labels = tighten_by_distance(features, new_labels, centroids, q=dist_q, standardize_features=std_feat)

        # measure label change
        label_change = float(np.mean(new_labels != labels))
        core_frac = float(np.mean(new_labels != -1))
        print(f"[refine] core fraction={core_frac:.3f}, label_change={label_change:.6e}")

        # update labels for next iter
        labels = new_labels

        # =====================================================
        # (C) Save updated dataset for retraining
        # =====================================================
        meta2 = dict(meta)
        meta2["k_selected"] = int(K)

        pl = build_pair_labels_from_state(labels, K)

        pack_out = dict(pack)
        pack_out["meta_state"] = torch.from_numpy(labels.astype(np.int64))
        pack_out["pair_labels"] = torch.from_numpy(pl.astype(np.int8))
        pack_out["meta"] = meta2
        pack_out["P_x"] = torch.from_numpy(P.astype(np.float32))
        pack_out["E_x"] = torch.from_numpy(E.astype(np.float32))

        out_pt = os.path.join(iter_dir, f"dataset_iter_{it:03d}.pt")
        torch.save(pack_out, out_pt)
        print(f"[save] {out_pt}")

        if save_debug:
            df = pd.DataFrame({
                "frame": np.arange(N, dtype=np.int64),
                "weight": w,
                "meta_state": labels,
                "p_max": p_max.astype(np.float32),
                "p_argmax": p_arg.astype(np.int32),
                "E": E.astype(np.float32),
            })
            df.to_csv(os.path.join(iter_dir, "debug.csv"), index=False)

        # =====================================================
        # (D) Convergence check
        # =====================================================
        if n_new_added == 0 and label_change < tol_label_change:
            print("Converged: no new states added and label change small.")
            final_pt = os.path.join(out_dir, "dataset_converged.pt")
            torch.save(pack_out, final_pt)
            print(f"[final] {final_pt}")
            break

        # NOTE:
        # After this iteration, you SHOULD retrain your model with the saved dataset_iter_XXX.pt
        # because K and/or labels changed. Then update cfg['model'] to the new best model and rerun.
        print("\n[NOTE] K/labels updated. Please retrain model on this dataset before next iteration.")
        print("       Then set ADAPTIVE.model to the new model path and rerun for the next iteration.")


def main():
    ap = argparse.ArgumentParser(description="Adaptive state discovery + label refinement using HDBSCAN (config-driven).")
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    if "ADAPTIVE" not in cfg:
        raise KeyError("Config must contain top-level key: ADAPTIVE")
    run(cfg)


if __name__ == "__main__":
    main()
