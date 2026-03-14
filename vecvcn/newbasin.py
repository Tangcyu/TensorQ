#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import torch


# =========================================================
# === Utilities
# =========================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pair_list(n_states):
    return [(i, j) for i in range(n_states) for j in range(i + 1, n_states)]

def infer_K_from_M(M: int) -> int:
    K = int((1 + int(np.sqrt(1 + 8 * M))) // 2)
    if K * (K - 1) // 2 != M:
        raise ValueError(f"Cannot infer integer K from M={M}")
    return K

def build_A_matrix(pairs, n_states, anchor_state=0):
    """
    Build A for least squares:
      logit(q_ij) ≈ s_j - s_i
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

def softmax_rows(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=1, keepdims=True) + 1e-18)

def infer_Q(model, features, device, batch_size=65536):
    N = features.shape[0]
    with torch.no_grad():
        q_test = model(features[: min(N, 8)].to(device))
    M = int(q_test.shape[1])

    Q = np.empty((N, M), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            xb = features[start:end].to(device)
            Q[start:end] = model(xb).detach().float().cpu().numpy()
    return Q

def fit_scores_and_w_from_Q(Q, pairs, n_states, eps=1e-6, anchor_state=0, chunk=20000):
    """
    Per-frame fit:
      logit(q_ij) ≈ s_j - s_i
    Solve via precomputed pinv(A); then w = softmax(s).
    """
    N, M = Q.shape
    A, var_index = build_A_matrix(pairs, n_states, anchor_state=anchor_state)
    pinvA = np.linalg.pinv(A)  # ((N-1), M)

    s_out = np.zeros((N, n_states), dtype=np.float32)
    w_out = np.zeros((N, n_states), dtype=np.float32)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = Q[start:end].astype(np.float64)
        q = np.clip(q, eps, 1.0 - eps)
        l = np.log(q) - np.log(1.0 - q)          # (B, M)

        s_free = l @ pinvA.T                      # (B, N-1)

        B = s_free.shape[0]
        s_full = np.zeros((B, n_states), dtype=np.float64)
        for s, c in var_index.items():
            s_full[:, s] = s_free[:, c]
        s_full[:, anchor_state] = 0.0

        w = softmax_rows(s_full).astype(np.float32)

        s_out[start:end] = s_full.astype(np.float32)
        w_out[start:end] = w

    return s_out, w_out

def compute_E_from_Q(Q, pairs, n_states, eps=1e-4, anchor_state=0, chunk=20000):
    """
    E(x) = mean_{pairs} (q_ij - qtilde_ij)^2
    where qtilde is derived from best-fit p (softmax scores) per-frame.
    """
    N, M = Q.shape
    A, var_index = build_A_matrix(pairs, n_states, anchor_state=anchor_state)
    pinvA = np.linalg.pinv(A)  # ((N-1), M)

    E = np.empty((N,), dtype=np.float32)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = Q[start:end].astype(np.float64)

        q = np.clip(q, eps, 1.0 - eps)
        l = np.log(q) - np.log(1.0 - q)          # (B, M)

        s_free = l @ pinvA.T                      # (B, N-1)

        B = s_free.shape[0]
        s_full = np.zeros((B, n_states), dtype=np.float64)
        for s, c in var_index.items():
            s_full[:, s] = s_free[:, c]
        s_full[:, anchor_state] = 0.0

        p = softmax_rows(s_full)                  # (B, N)

        q_tilde = np.empty((B, M), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            denom = p[:, i] + p[:, j] + 1e-12
            q_tilde[:, k] = p[:, j] / denom

        err = (q - q_tilde) ** 2
        E[start:end] = np.mean(err, axis=1).astype(np.float32)

    return E

def accumulate_soft_transitions(w, seglen=1000):
    """
    Many short shooting, contiguous packed:
    Use fixed segment length seglen (or you can store traj_id in dataset later).
    Soft transition:
      C_{kℓ} = sum w_k(t) w_ℓ(t+1) within each segment
      occ_k  = sum w_k(t)
      P_{kℓ} = C_{kℓ}/occ_k
    """
    N, K = w.shape
    C = np.zeros((K, K), dtype=np.float64)
    occ = np.zeros((K,), dtype=np.float64)

    for a in range(0, N, seglen):
        b = min(N, a + seglen)
        L = b - a
        if L < 2:
            continue
        wt = w[a:b-1]
        wt1 = w[a+1:b]
        occ += wt.sum(axis=0)
        C += wt.T @ wt1

    P = C / (occ[:, None] + 1e-18)
    return P.astype(np.float64), occ.astype(np.float64), C.astype(np.float64)

def stationary_left(P, eps=1e-12):
    Pt = P.T.astype(np.float64)
    w, v = np.linalg.eig(Pt)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, idx]).astype(np.float64)
    pi = np.abs(pi)
    s = float(pi.sum())
    if s < eps:
        pi = np.ones((P.shape[0],), dtype=np.float64) / float(P.shape[0])
    else:
        pi /= s
    return pi

def make_weights_from_pi(w_soft, pi, mode="soft"):
    """
    Return normalized per-frame weights (float32), to be stored as pack['weights'].
    mode="soft": weight(t) ∝ sum_k w_k(t) * pi_k
    mode="hard": weight(t) ∝ pi_{argmax w(t)}
    """
    if mode == "hard":
        k = np.argmax(w_soft, axis=1)
        w = pi[k].astype(np.float64)
    else:
        w = (w_soft.astype(np.float64) @ pi.astype(np.float64))
    w = np.clip(w, 0.0, None)
    w /= (w.sum() + 1e-18)
    return w.astype(np.float32)


# =========================================================
# === Auto-split ONLY inside existing metastates (meta_state>=0)
# =========================================================

def pca_first_component(X):
    X = X.astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2 or np.allclose(Xc.var(axis=0).sum(), 0.0):
        return np.zeros((X.shape[0],), dtype=np.float64)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    v1 = vt[0]
    return (Xc @ v1).astype(np.float64)

def kmeans_1d_two_clusters(x, n_init=8, max_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    n = x.size
    if n < 2:
        return np.zeros(n, dtype=np.int64), np.array([x.mean(), x.mean()]), 0.0

    best_sse = np.inf
    best_labels = None
    best_centers = None

    inits = []
    for _ in range(n_init):
        a, b = rng.choice(n, size=2, replace=False)
        inits.append((x[a], x[b]))
    inits.append((np.quantile(x, 0.25), np.quantile(x, 0.75)))

    for (c0, c1) in inits:
        centers = np.array([c0, c1], dtype=np.float64)
        labels = np.zeros(n, dtype=np.int64)

        for _ in range(max_iter):
            d0 = (x - centers[0]) ** 2
            d1 = (x - centers[1]) ** 2
            new_labels = (d1 < d0).astype(np.int64)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for kk in (0, 1):
                m = labels == kk
                if np.any(m):
                    centers[kk] = x[m].mean()
                else:
                    centers[kk] = x[rng.integers(0, n)]

        sse = ((x - centers[labels]) ** 2).sum()
        if sse < best_sse:
            best_sse = sse
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, float(best_sse)

def autosplit_from_existing_metastates(
    meta_state, w_soft, s_scores, E_x=None,
    high_wk=0.85,
    max_E=None,
    min_points=1000,
    min_frac=0.10,
    sse_improve=0.20,
    sep_sigma=1.0,
    seed=0,
):
    """
    Only split frames with meta_state==k (k>=0). Intermediate (-1) never used for splitting.
    New state ids are appended at the end (0..K_new-1).
    """
    meta_state = np.asarray(meta_state, dtype=np.int64)
    w = w_soft.astype(np.float64)
    s = s_scores.astype(np.float64)

    N, K = w.shape
    report = []
    w_current = w.copy()
    meta_new = meta_state.copy()

    for k in range(K):
        idx0 = np.where(meta_state == k)[0]
        if idx0.size < min_points:
            continue

        # purity / quality filters
        keep = np.ones(idx0.size, dtype=bool)
        if high_wk is not None:
            keep &= (w_current[idx0, k] >= float(high_wk))
        if E_x is not None and max_E is not None:
            keep &= (E_x[idx0] <= float(max_E))
        idx = idx0[keep]
        if idx.size < min_points:
            continue

        X = s[idx, :K]
        pc1 = pca_first_component(X)

        mu = pc1.mean()
        sse1 = float(((pc1 - mu) ** 2).sum())
        labels, centers, sse2 = kmeans_1d_two_clusters(pc1, seed=seed + k)
        improve = (sse1 - sse2) / (sse1 + 1e-18)

        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        frac_small = min(n0, n1) / float(idx.size)

        sigma = float(pc1.std(ddof=1) + 1e-18)
        sep = float(abs(centers[0] - centers[1]) / sigma)

        if improve < sse_improve or frac_small < min_frac or sep < sep_sigma:
            continue

        # append new state id
        new_id = int(w_current.shape[1])

        # expand w by 1 column
        w_ext = np.zeros((N, new_id + 1), dtype=np.float64)
        w_ext[:, :new_id] = w_current
        w_current = w_ext

        # smaller cluster becomes new state (more conservative)
        new_label = 0 if n0 < n1 else 1
        sel = idx[labels == new_label]

        # move membership from k -> new_id for selected ENDPOINT frames only
        w_k = w_current[sel, k].copy()
        w_current[sel, new_id] = w_k
        w_current[sel, k] = 0.0

        # renormalize rows
        w_current /= (w_current.sum(axis=1, keepdims=True) + 1e-18)

        # update meta_state for those endpoints
        meta_new[sel] = new_id

        report.append({
            "parent_state": int(k),
            "new_state": int(new_id),
            "n_used": int(idx.size),
            "frac_small": float(frac_small),
            "sep_sigma": float(sep),
            "sse_improve": float(improve),
            "centers": [float(centers[0]), float(centers[1])],
        })

    return meta_new, w_current.astype(np.float32), report

def expand_pair_labels(pair_labels_old, meta_state_new, K_old, K_new):
    """
    pair_labels_old: (N, C(K_old,2)) int8, pairs order == pair_list(K_old)
    meta_state_new: (N,), includes new ids in [0..K_new-1] and -1 for intermediate

    For new pairs (involving new states), fill endpoint frames:
      if meta_state==i -> 0, if meta_state==j -> 1, else -1
    Intermediate frames remain -1.
    """
    meta_state_new = np.asarray(meta_state_new, dtype=np.int64)

    if torch.is_tensor(pair_labels_old):
        pl_old = pair_labels_old.cpu().numpy().astype(np.int8)
    else:
        pl_old = np.asarray(pair_labels_old, dtype=np.int8)

    N = pl_old.shape[0]
    pairs_old = pair_list(K_old)
    pairs_new = pair_list(K_new)
    idx_old = {p: t for t, p in enumerate(pairs_old)}

    pl_new = -np.ones((N, len(pairs_new)), dtype=np.int8)

    # copy old columns
    for t, p in enumerate(pairs_new):
        if p in idx_old:
            pl_new[:, t] = pl_old[:, idx_old[p]]

    endpoints = np.where(meta_state_new >= 0)[0]
    ms = meta_state_new[endpoints]

    # fill only truly new pair columns
    for t, (i, j) in enumerate(pairs_new):
        if (i, j) in idx_old:
            continue
        mi = (ms == i)
        mj = (ms == j)
        if np.any(mi):
            pl_new[endpoints[mi], t] = np.int8(0)
        if np.any(mj):
            pl_new[endpoints[mj], t] = np.int8(1)

    return pl_new


# =========================================================
# === Main
# =========================================================

def run(cfg):
    cfg = cfg["MAKE_NEXT_DATASET"]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    pack = torch.load(cfg["dataset"], map_location="cpu")
    # required keys
    for k in ("features", "weights", "meta_state", "meta", "cv", "pair_labels"):
        if k not in pack:
            raise KeyError(f"dataset missing key: '{k}'")

    features = pack["features"].float()
    meta_state_old = pack["meta_state"].cpu().numpy().astype(np.int64)
    cv = pack["cv"]
    meta = pack["meta"]
    cv_headers = meta.get("cv_headers", None)

    N = int(features.shape[0])

    # model/device
    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")
    model = torch.jit.load(cfg["model"], map_location=device)
    model.eval()
    batch_size = int(cfg.get("batch_size", 65536))

    # infer Q
    print("[1] infer Q(x)")
    Q = infer_Q(model, features, device=device, batch_size=batch_size)
    M = int(Q.shape[1])

    # infer K_old from model output dim (robust)
    K_old = infer_K_from_M(M)
    pairs_old = pair_list(K_old)
    print(f"[info] inferred K_old={K_old} from M={M}")

    # fit scores and w
    print("[2] fit s_scores and w_soft")
    fit_eps = float(cfg.get("fit_eps", 1e-6))
    fit_chunk = int(cfg.get("fit_chunk", 20000))
    anchor_state = int(cfg.get("anchor_state", 0))
    s_scores, w_soft = fit_scores_and_w_from_Q(
        Q, pairs_old, K_old, eps=fit_eps, anchor_state=anchor_state, chunk=fit_chunk
    )

    # E(x) optional
    compute_E = bool(cfg.get("compute_E", True))
    E_x = None
    if compute_E:
        print("[3] compute E(x)")
        E_eps = float(cfg.get("E_eps", 1e-4))
        E_chunk = int(cfg.get("E_chunk", 20000))
        E_x = compute_E_from_Q(Q, pairs_old, K_old, eps=E_eps, anchor_state=anchor_state, chunk=E_chunk)

    # autosplit ONLY inside existing metastates
    print("[4] autosplit from existing metastates (meta_state>=0 only)")
    meta_new, w_soft2, report = autosplit_from_existing_metastates(
        meta_state=meta_state_old,
        w_soft=w_soft,
        s_scores=s_scores,
        E_x=E_x,
        high_wk=float(cfg.get("autosplit_high_wk", 0.85)),
        max_E=cfg.get("autosplit_max_E", None),
        min_points=int(cfg.get("autosplit_min_points", 1000)),
        min_frac=float(cfg.get("autosplit_min_frac", 0.10)),
        sse_improve=float(cfg.get("autosplit_sse_improve", 0.20)),
        sep_sigma=float(cfg.get("autosplit_sep_sigma", 1.0)),
        seed=int(cfg.get("autosplit_seed", 0)),
    )
    K_new = int(w_soft2.shape[1])
    print(f"[autosplit] created {len(report)} new state(s). K_new={K_new}")

    # expand pair_labels
    print("[5] expand pair_labels")
    pair_labels_new = expand_pair_labels(
        pair_labels_old=pack["pair_labels"],
        meta_state_new=meta_new,
        K_old=K_old,
        K_new=K_new
    )

    # MSM weights (store in pack['weights'])
    # We compute P/pi using w_soft2 inside short shooting segments (fixed length)
    seglen = int(cfg.get("segment_length", 1000))
    print(f"[6] compute P_soft/pi_soft (seglen={seglen}) and reweight frames -> pack['weights']")
    P_soft, occ_soft, C_soft = accumulate_soft_transitions(w_soft2, seglen=seglen)
    pi_soft = stationary_left(P_soft)

    reweight_mode = str(cfg.get("reweight_mode", "soft"))
    weights_next = make_weights_from_pi(w_soft2, pi_soft, mode=reweight_mode)  # float32, sums to 1

    # update pack (in-place style) and save next dataset
    out_pack = dict(pack)  # keep original keys
    out_pack["Q_pairwise"] = torch.from_numpy(Q)                    # (N, M)
    out_pack["s_scores"] = torch.from_numpy(s_scores)               # (N, K_old) (kept)
    out_pack["w_soft"] = torch.from_numpy(w_soft2)                  # (N, K_new)
    out_pack["meta_state"] = torch.from_numpy(meta_new.astype(np.int64))  # keep -1 intermediates
    out_pack["pair_labels"] = torch.from_numpy(pair_labels_new)     # (N, C(K_new,2))

    if E_x is not None:
        out_pack["E_x"] = torch.from_numpy(E_x)

    out_pack["P_soft"] = torch.from_numpy(P_soft.astype(np.float32))
    out_pack["pi_soft"] = torch.from_numpy(pi_soft.astype(np.float32))
    out_pack["occ_soft"] = torch.from_numpy(occ_soft.astype(np.float32))
    out_pack["C_soft"] = torch.from_numpy(C_soft.astype(np.float32))

    # IMPORTANT: store MSM weights directly in ['weights']
    out_pack["weights"] = torch.from_numpy(weights_next)

    # meta updates
    meta2 = dict(meta) if isinstance(meta, dict) else {}
    meta2["K_old"] = K_old
    meta2["K_new"] = K_new
    meta2["pairs_dim_old"] = int(K_old * (K_old - 1) // 2)
    meta2["pairs_dim_new"] = int(K_new * (K_new - 1) // 2)
    meta2["autosplit_report"] = report
    meta2["segment_length"] = seglen
    meta2["reweight_mode"] = reweight_mode
    out_pack["meta"] = meta2

    out_dataset = cfg.get("out_dataset", os.path.join(cfg["out_dir"], "dataset_next.pt"))
    torch.save(out_pack, out_dataset)
    print(f"[DONE] saved next dataset: {out_dataset}")

    # -----------------------------
    # Save CVs for analysis (your requested block)
    # -----------------------------
    save_cv = bool(cfg.get("save_cv", True))
    weights_csv = os.path.join(cfg["out_dir"], "weights_next.csv")
    if save_cv:
        if cv_headers is None:
            raise RuntimeError("meta['cv_headers'] missing; cannot write CSV with column names.")
        import pandas as pd
        cv_np = cv.cpu().numpy() if torch.is_tensor(cv) else np.asarray(cv)
        df = pd.DataFrame(cv_np, columns=cv_headers)
        df.insert(0, "frame", np.arange(N, dtype=np.int64))
        df["weight"] = weights_next
        df["meta_state"] = meta_new
        df["is_intermediate"] = (meta_new == -1).astype(np.int8)
        df.to_csv(weights_csv, index=False)
        print(f"[CSV] Saved: {weights_csv}")

    # save report json
    try:
        import json
        with open(os.path.join(cfg["out_dir"], "autosplit_report.json"), "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Auto-split existing metastates using w_soft and make next dataset (append new states).")
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if "MAKE_NEXT_DATASET" not in cfg:
        raise KeyError("Config must contain a top-level key: MAKE_NEXT_DATASET")
    run(cfg)


if __name__ == "__main__":
    main()
