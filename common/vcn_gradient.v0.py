import os
import torch
import numpy as np
import pandas as pd
import mdtraj as md
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from vcn.zmatrix import (
    get_internal_coordinates,
    get_minimal_internal_coordinates,
    get_pair_distances,
)


# =========================================================
# === Utility functions ===
# =========================================================

def load_yaml_config(file_path):
    """Load YAML configuration safely."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def _prepare_input_tensor(positions, periodic=False, device='cpu'):
    """Internal helper to build the model input tensor with gradients enabled."""
    if periodic:
        dim = positions.shape[1]
        # positions assumed in degrees as in your original code
        x = positions * np.pi / 180.0
        inp = np.concatenate(
            [np.sin(x), np.cos(x)],
            axis=1,
        )
    else:
        inp = positions

    input_tensor = torch.tensor(inp, dtype=torch.float, device=device, requires_grad=True)
    return input_tensor


def calc_committors_sig(model, positions, periodic=False, device='cpu', return_derivs=False):
    """
    Calculate committor values (and optionally gradients and second derivatives)
    for input coordinates.

    If return_derivs=False (default):
        returns q_values (numpy, shape [N])

    If return_derivs=True:
        returns (q_values, grad_q, grad2_q_diag) where
            q_values: shape [N]
            grad_q: shape [N, D]          (dq/dx_i)
            grad2_q_diag: shape [N, D]    (d^2 q / dx_i^2)  (diagonal of Hessian)
    """
    input_tensor = _prepare_input_tensor(positions, periodic=periodic, device=device)

    # Forward pass
    output_tensor = model(input_tensor).squeeze()  # shape [N]

    if not return_derivs:
        return output_tensor.detach().cpu().numpy().flatten()

    # First derivatives: ∇q wrt input features
    # grad_outputs=ones ensures sum over batch for autograd; create_graph=True for second derivs
    grad_q = torch.autograd.grad(
        outputs=output_tensor,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output_tensor),
        create_graph=True,
    )[0]  # shape [N, D]

    # Second derivatives: diagonal of the Hessian d^2 q / dx_i^2
    N, D = grad_q.shape
    grad2_diag_list = []
    for i in range(D):
        grad_i = grad_q[:, i]  # shape [N]
        # grad of grad_i wrt input_tensor gives shape [N, D], take the i-th component
        grad2_i_full = torch.autograd.grad(
            outputs=grad_i,
            inputs=input_tensor,
            grad_outputs=torch.ones_like(grad_i),
            retain_graph=True,
        )[0]  # shape [N, D]
        grad2_diag_list.append(grad2_i_full[:, i])  # keep only d^2 q / dx_i^2

    grad2_q_diag = torch.stack(grad2_diag_list, dim=1)  # shape [N, D]

    q_values = output_tensor.detach().cpu().numpy().flatten()
    grad_q_np = grad_q.detach().cpu().numpy()
    grad2_q_diag_np = grad2_q_diag.detach().cpu().numpy()

    return q_values, grad_q_np, grad2_q_diag_np


def calc_committors_id(model, positions, periodic=False, device='cpu', return_derivs=False):
    """
    Calculate committor values (and optionally gradients and second derivatives)
    for input coordinates.

    If return_derivs=False (default):
        returns q_values (numpy, shape [N])

    If return_derivs=True:
        returns (q_values, grad_q, grad2_q_diag) where
            q_values: shape [N]
            grad_q: shape [N, D]          (dq/dx_i)
            grad2_q_diag: shape [N, D]    (d^2 q / dx_i^2)  (diagonal of Hessian)
    """
    input_tensor = _prepare_input_tensor(positions, periodic=periodic, device=device)

    # Forward pass
    output_tensor = model.forward_id(input_tensor).squeeze()  # shape [N]

    if not return_derivs:
        return output_tensor.detach().cpu().numpy().flatten()

    # First derivatives: ∇q wrt input features
    # grad_outputs=ones ensures sum over batch for autograd; create_graph=True for second derivs
    grad_q = torch.autograd.grad(
        outputs=output_tensor,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output_tensor),
        create_graph=True,
    )[0]  # shape [N, D]

    # Second derivatives: diagonal of the Hessian d^2 q / dx_i^2
    N, D = grad_q.shape
    grad2_diag_list = []
    for i in range(D):
        grad_i = grad_q[:, i]  # shape [N]
        # grad of grad_i wrt input_tensor gives shape [N, D], take the i-th component
        grad2_i_full = torch.autograd.grad(
            outputs=grad_i,
            inputs=input_tensor,
            grad_outputs=torch.ones_like(grad_i),
            retain_graph=True,
        )[0]  # shape [N, D]
        grad2_diag_list.append(grad2_i_full[:, i])  # keep only d^2 q / dx_i^2

    grad2_q_diag = torch.stack(grad2_diag_list, dim=1)  # shape [N, D]

    q_values = output_tensor.detach().cpu().numpy().flatten()
    grad_q_np = grad_q.detach().cpu().numpy()
    grad2_q_diag_np = grad2_q_diag.detach().cpu().numpy()

    return q_values, grad_q_np, grad2_q_diag_np


def load_dcd_data(path0, dcdfile, topfile, atomselect, stride):
    """Load DCD trajectory and apply atom selection if given."""
    traj = md.load(os.path.join(path0, dcdfile), top=os.path.join(path0, topfile))[::int(stride) if stride else 1]
    if atomselect is not None:
        atomindex = traj.topology.select(atomselect) + 1
    else:
        atomindex = []
    return traj, atomindex


def convert_to_zmatrix(traj, atomindex, use_all=False, pair_distance=False):
    """Convert Cartesian trajectory to internal coordinates."""
    if use_all:
        labels, values = get_internal_coordinates(traj, atomindex)
    elif pair_distance:
        labels, values = get_pair_distances(traj, atomindex)
    else:
        labels, values = get_minimal_internal_coordinates(traj, atomindex)

    print(f"Converted trajectory to Z-matrix with {len(labels)} variables.")
    return pd.DataFrame(np.vstack(values), columns=labels), labels


# =========================================================
# === Plotting ===
# =========================================================

def plot_committor_2d(x, y, q_values, out_path, title_suffix=""):
    """2D scatter plot of committor values."""
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap('RdBu_r', 20)
    sc = plt.scatter(x, y, c=q_values, cmap=cmap, vmin=0, vmax=1)
    clb = plt.colorbar(sc)
    clb.ax.set_title(r'$q$', pad=12.0)
    clb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_committor_pairs(traj, q_values, cvs, out_dir, prefix):
    """Plot all 2D combinations of three CVs as 2D plots."""
    if len(cvs) == 3:
        combinations = [(cvs[0], cvs[1]), (cvs[1], cvs[2]), (cvs[0], cvs[2])]
        for x, y in combinations:
            out_path = os.path.join(out_dir, f"{prefix}_{x}_vs_{y}.png")
            plot_committor_2d(traj[x], traj[y], q_values, out_path)
    elif len(cvs) == 2:
        out_path = os.path.join(out_dir, f"{prefix}_{cvs[0]}_vs_{cvs[1]}.png")
        plot_committor_2d(traj[cvs[0]], traj[cvs[1]], q_values, out_path)
    else:
        print("Warning: plot_committor_pairs supports only 2 or 3 CVs.")


### NEW: plot gradient and “gradient of gradient” vs q
def plot_gradients_vs_q(q_values, grad_cvs, out_path, feature_names=None, grad2_q_diag=None):
    """
    Plot per-CV gradients dq/dCV_j as functions of q.

    - q_values: array shape [N]
    - grad_cvs: array shape [N, D] containing dq/d(CV_j)
    - out_path: path to save the combined plot
    - feature_names: list of length D with names for each CV (optional)
    - grad2_q_diag: optional second-derivative diag array (not required)
    """
    N, D = grad_cvs.shape

    # Sort by q for plotting
    order = np.argsort(q_values)
    q_sorted = q_values[order]
    grads_sorted = grad_cvs[order, :]
    grad2_q_diag_sorted = grad2_q_diag[order, :] if grad2_q_diag is not None else None

    # Prepare feature names
    if feature_names is None:
        feature_names = [f"cv_{i}" for i in range(D)]

    # Combined plot: all dq/dCV_j vs q
    plt.figure(figsize=(8, 5))
    for j in range(D):
        plt.plot(q_sorted, np.abs(grads_sorted[:, j]), label=feature_names[j], linewidth=1)

    # also plot L2 norm over CVs (thicker)
    grad_norm = np.abs(np.linalg.norm(grad_cvs, axis=1))
    grad_norm_sorted = grad_norm[order]
    plt.plot(q_sorted, grad_norm_sorted, label=r"$\|\nabla q\|$", color="k", linewidth=2)

    # plt.xlim(0.0, 1.0)
    plt.xlabel("q")
    plt.ylabel(r"dq/dCV")
    # plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path+".png", dpi=300)
    plt.close()
    
    
    # Combined plot: all d^2q/dCV_j^2 vs q
    if grad2_q_diag is not None:
        plt.figure(figsize=(8, 5))
        for j in range(D):
            plt.plot(q_sorted, np.abs(grad2_q_diag_sorted[:, j]), label=feature_names[j], linewidth=1)

        # also plot L2 norm over CVs (thicker)
        grad2_norm = np.abs(np.linalg.norm(grad2_q_diag, axis=1))
        grad2_norm_sorted = grad2_norm[order]
        plt.plot(q_sorted, grad2_norm_sorted, label=r"$\|\nabla q\|$", color="k", linewidth=2)

        # plt.xlim(0.0, 1.0)
        plt.xlabel("q")
        plt.ylabel(r"d^2q/dCV^2")
        # plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(out_path+"_second.png", dpi=300)
        plt.close()


    # Save separate per-CV plots for clarity
    base_dir = os.path.dirname(out_path)

def compute_sensitivity_metrics(grad_cvs, q_values, feature_names, out_dir, mask=None):
    """
    Compute simple sensitivity metrics per CV and save results + bar plots.

    Metrics computed per CV:
      - mean_abs_grad: mean(|dq/dCV|)
      - rms_grad: sqrt(mean((dq/dCV)^2))
      - std_grad: std(dq/dCV)
      - max_abs_grad: max(|dq/dCV|)
      - median_abs_grad: median(|dq/dCV|)
      - corr_absgrad_q: Pearson correlation between |dq/dCV| and q

    If mask is provided, metrics are computed on the masked subset as well and
    both overall and masked metrics are saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    D = grad_cvs.shape[1]
    rows = []

    # choose data according to mask
    if mask is None:
        idx = np.arange(len(q_values))
        suffix = "overall"
    else:
        idx = np.where(mask)[0]
        suffix = "transition"

    for j, name in enumerate(feature_names):
        g = grad_cvs[idx, j]
        abs_g = np.abs(g)
        mean_abs = np.mean(abs_g)
        rms = np.sqrt(np.mean(g ** 2))
        std = np.std(g)
        mx = np.max(abs_g)
        med = np.median(abs_g)

        # correlation with q (use absolute gradient vs q)
        if len(idx) > 1:
            try:
                corr = np.corrcoef(abs_g, q_values[idx])[0, 1]
            except Exception:
                corr = np.nan
        else:
            corr = np.nan

        rows.append(
            {
                "feature": name,
                "mean_abs_grad": mean_abs,
                "rms_grad": rms,
                "std_grad": std,
                "max_abs_grad": mx,
                "median_abs_grad": med,
                "corr_absgrad_q": corr,
            }
        )

    df = pd.DataFrame(rows).set_index("feature")
    csv_name = os.path.join(out_dir, f"sensitivity_metrics_{suffix}.csv")
    df.to_csv(csv_name)

    # bar plot ordered by mean_abs_grad
    order = df["mean_abs_grad"].sort_values(ascending=False).index
    plt.figure(figsize=(max(6, 0.6 * len(order)), 4))
    plt.bar(order, df.loc[order, "mean_abs_grad"], color="C0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean(|dq/dCV|)")
    plt.title(f"CV sensitivity ({suffix})")
    plt.tight_layout()
    barname = os.path.join(out_dir, f"sensitivity_bar_{suffix}.png")
    plt.savefig(barname, dpi=300)
    plt.close()

    return df


# =========================================================
# === Clustering ===
# =========================================================

def perform_kmeans_clustering(points, out_dir):
    """Run KMeans and save cluster centers."""
    inertia = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(points)
        inertia.append(km.inertia_)

    k_opt = KneeLocator(
        range(1, 10), inertia, curve="convex", direction="decreasing"
    ).elbow or 10

    print(f"Optimal number of clusters (k): {k_opt}")
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10).fit(points)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    distances = cdist(points, centroids)
    closest_frames = {
        i: np.where(cluster_labels == i)[0][np.argmin(distances[cluster_labels == i, i])]
        for i in range(k_opt)
    }
    farthest_frames = {
        i: np.where(cluster_labels == i)[0][np.argmax(distances[cluster_labels == i, i])]
        for i in range(k_opt)
    }

    selected_indices = list(closest_frames.values()) + list(farthest_frames.values())
    np.savetxt(os.path.join(out_dir, "selected_indices.txt"), selected_indices, fmt="%d")
    return selected_indices


def load_csv_trajectories(path0, label, traj_fns, stride):
    """Load CSV trajectory files."""
    if traj_fns is None:
        traj_fns = glob.glob(os.path.join(path0, f"{label}.csv.gz"))
    print(f"Found CSV files: {traj_fns}")

    if len(traj_fns) == 1:
        traj = pd.read_csv(traj_fns[0])
    else:
        traj = pd.concat([pd.read_csv(fn) for fn in traj_fns], ignore_index=True)

    if stride is not None:
        traj = traj[::int(stride)]

    return traj

def bin_gradients_by_q(q_values, grad_cvs, M):
    """
    Bin q into M bins in [0,1] and average gradients in each bin.

    Returns:
        q_bin_centers : shape [M]
        grad_binned   : shape [M, D]
        bin_edges     : shape [M+1]
        counts        : shape [M] (number of samples per bin)
    """
    q_values = np.asarray(q_values)
    grad_cvs = np.asarray(grad_cvs)
    N, D = grad_cvs.shape

    # Define uniform bins in [0, 1]
    bin_edges = np.linspace(0.0, 1.0, M + 1)

    # Assign each q to a bin index in [0, M-1]
    bin_indices = np.digitize(q_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, M - 1)

    grad_binned = np.full((M, D), np.nan, dtype=float)
    q_bin_centers = np.full(M, np.nan, dtype=float)
    counts = np.zeros(M, dtype=int)

    for i in range(M):
        mask = (bin_indices == i)
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            grad_binned[i, :] = grad_cvs[mask].mean(axis=0)
            q_bin_centers[i] = q_values[mask].mean()
        else:
            # Empty bin: place center at mid of the bin
            q_bin_centers[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

    return q_bin_centers, grad_binned, bin_edges, counts


# =========================================================
# === Main analysis ===
# =========================================================

def run_committor_gradient(config):

    label = config.get("label", "default_label")
    model_fn = config["model_fn"]
    path0 = config.get("Sampling_path", "./")
    out_dir = config.get("out_dir", "./output/")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )

    # Load settings
    activation = config.get("activation", "sig")
    
    use_z_matrix = config.get("z_matrix", False)
    pair_distance = config.get("pair_distance", False)
    use_all = config.get("use_all", False)
    dcdfile = config["dcdfile"]
    topfile = config["topfile"]
    stride = config.get("stride", 1)
    traj_fns = config.get("traj_fns", None)
    atomselect = config.get("atomselect", None)
    cvs_to_plot = config.get("cvs_to_plot", [])
    periodic = config.get("periodic", False)
    q_var = config.get("q_variance", 0.1)
    M_bins = config.get("M_bins", 15)

    # Load trajectory
    traj = load_csv_trajectories(path0, label, traj_fns, stride)
    
    if use_z_matrix:
        dcdtraj, atomindex = load_dcd_data(path0, dcdfile, topfile, atomselect, stride)
        z_data, labels = convert_to_zmatrix(dcdtraj, atomindex, use_all, pair_distance)
        traj = traj.reset_index(drop=True).join(z_data)

    cvs0 = labels if use_z_matrix else cvs_to_plot
    model = torch.jit.load(model_fn).to(device)
    model.eval()

    traj_values = traj[cvs0].to_numpy()

    # === MODIFIED: compute q, gradient and second derivative
    if activation == "sig":
        q_values, grad_q, grad2_q_diag = calc_committors_sig(
            model,
            traj_values,
            periodic=periodic,
            device=device,
            return_derivs=True,
        )
    elif activation == "id":
        q_values, grad_q, grad2_q_diag = calc_committors_id(
            model,
            traj_values,
            periodic=periodic,
            device=device,
            return_derivs=True,
        )
    q_column = pd.DataFrame(q_values, columns=["q_value"])
    traj = traj.reset_index(drop=True).join(q_column)
    print(q_values.min(), q_values.max())
    
    grad_norm = np.abs(np.linalg.norm(grad_q, axis=1))
    mask = (grad_norm < 0.08) & (q_values > 0.2) & (q_values < 1 - 0.2)
    sliced_points = traj[mask][cvs_to_plot+["q_value"]].copy()
    sliced_points.to_csv(os.path.join(out_dir, "sliced_grad.csv"), index=False)
    np.savetxt(os.path.join(out_dir, "sliced_grad.txt"), sliced_points.to_numpy(), fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

    mask_basin = (q_values < 0.2) | (q_values > 1 - 0.2)
    sliced_basins = traj[mask_basin][cvs_to_plot+["q_value"]].copy()
    np.savetxt(os.path.join(out_dir, "sliced_basin.txt"), sliced_basins.to_numpy(), fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
  
    # === NEW: build gradients w.r.t original CVs (cvs0)
    N = q_values.shape[0]
    D = len(cvs0)

    # grad_q_np is numpy array with shape [N, input_dim]
    grad_input = grad_q  # shape [N, input_dim]

    if periodic:
        # input layout: [sin(x_0), ..., sin(x_{D-1}), cos(x_0), ..., cos(x_{D-1})]
        grad_sin = grad_input[:, :D]
        grad_cos = grad_input[:, D: 2 * D]

        # positions are in degrees in traj_values; convert to radians
        radians = traj_values * np.pi / 180.0  # shape [N, D]
        sinx = np.sin(radians)
        cosx = np.cos(radians)
        factor = np.pi / 180.0
        grad_cvs = grad_sin * (cosx * factor) + grad_cos * ((-sinx) * factor)
    else:
        # inputs correspond directly to cvs0
        grad_cvs = grad_input.copy()

   # === NEW: bin q and average gradient within each bin
    q_bin_centers, grad_cvs_binned, bin_edges, bin_counts = bin_gradients_by_q(
        q_values, grad_cvs, M_bins
    )

    # === NEW: save binned gradients to CSV
    binned_data = {
        "q_min": bin_edges[:-1],
        "q_max": bin_edges[1:],
        "q_center": q_bin_centers,
        "count": bin_counts,
    }
    for j, cv in enumerate(cvs0):
        binned_data[f"mean_dq_d_{cv}"] = grad_cvs_binned[:, j]

    df_binned = pd.DataFrame(binned_data)
    df_binned.to_csv(
        os.path.join(out_dir, f"q_gradients_binned_M{M_bins}.csv"),
        index=False,
    )

    # # === ORIGINAL: save per-frame q, gradient, and second derivative
    # out_data = {"q": q_values}

    # for j, cv in enumerate(cvs0):
    #     out_data[f"dq_d_{cv}"] = grad_cvs[:, j]

    # if periodic:
    #     for i, fname in enumerate([f"sin({n})" for n in cvs0] + [f"cos({n})" for n in cvs0]):
    #         out_data[f"dq_d_{fname}"] = grad_input[:, i]
    #         out_data[f"d2q_d_{fname}2"] = grad2_q_diag[:, i]
    # else:
    #     for i, fname in enumerate(cvs0):
    #         out_data[f"dq_d_{fname}"] = grad_input[:, i]
    #         out_data[f"d2q_d_{fname}2"] = grad2_q_diag[:, i]

    # df_derivs = pd.DataFrame(out_data)
    # df_derivs.to_csv(os.path.join(out_dir, "q_grad_and_second_grad.csv"), index=False)

    # === MODIFIED: plot using BINNED gradients vs q
    plot_gradients_vs_q(
        q_bin_centers,              # binned q (length M_bins)
        grad_cvs_binned,            # binned gradients (M_bins x D)
        os.path.join(out_dir, "grad_vs_q"),
        feature_names=cvs0,
        grad2_q_diag=grad2_q_diag,         
    )
    
    print("Committor analysis completed successfully.")

# =========================================================
# === Entry point ===
# =========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VCN model with config file")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_committor_gradient(config["VCN_gradient"])
