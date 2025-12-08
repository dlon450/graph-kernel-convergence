from typing import Dict, Any, Iterable, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------
# 1) Training curves
# ---------------------------------------------------------------------

def plot_dynamics(
    hist: Dict[str, Iterable[float]],
    title: str = "Training / Validation",
    rmse: bool = True,
    savepath: Optional[str] = None,
) -> None:
    """
    Plot training + validation trajectories.

    hist is a dict with keys:
        'rmse_train', 'rmse_val', 'relerr_train', 'relerr_val'
    produced by your training loop.
    """
    if rmse:
        train_key, val_key = "rmse_train", "rmse_val"
        ylabel = "RMSE"
    else:
        train_key, val_key = "relerr_train", "relerr_val"
        ylabel = "Relative Error"

    train_vals = np.asarray(hist[train_key], dtype=float)
    val_vals = np.asarray(hist[val_key], dtype=float)
    xs = np.arange(1, len(train_vals) + 1)

    plt.figure()
    plt.plot(xs, train_vals, label="train")
    plt.plot(xs, val_vals, label="validation")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# 2) Small 3D helper
# ---------------------------------------------------------------------

def apply_data_aspect(ax: plt.Axes, X: np.ndarray) -> None:
    """
    Make 3D axes respect the data's x/y/z ranges, so ellipsoids look right.
    X: (N, 3) ambient coordinates.
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    # Match the axes box to the data's aspect so anisotropy is preserved
    ax.set_box_aspect(rng)


# ---------------------------------------------------------------------
# 3) f(x, ω) field on the manifold for a single start node
# ---------------------------------------------------------------------

def plot_fx_field_for_start(
    model: torch.nn.Module,
    X: np.ndarray,                 # (N, 3)
    start_indices: np.ndarray,     # (num_starts,)
    phi: np.ndarray,               # (num_starts, N), truth rows φ_t(x_j)[·]
    s_idx: int,                    # index *into* start_indices
    geod_vec: Optional[np.ndarray] = None,  # (N,) or None; geodesics from this start
    title_prefix: str = "",
    share_norm: bool = True,
    cmap_main: str = "OrRd",
    cmap_err: str = "coolwarm",
) -> None:
    """
    Visualize predicted vs ground-truth φ_t(x_j)[ω] for a single start node.

    Assumes your model takes a batch dict with keys:
        'start', 'omega', 'start_xyz', 'omega_xyz', 'geod'
    and returns f(x, ω) with shape (batch_size,) or (batch_size, 1).
    """
    model.eval()
    device = next(model.parameters()).device
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    # Convert "which start" from index into start_indices to the actual node id
    s = int(start_indices[s_idx])
    start_xyz = X[s]
    omega_xyz = X

    # If no precomputed geodesics are provided, assume S^2 and compute via arccos of dot products.
    if geod_vec is None:
        dots = np.clip((omega_xyz * start_xyz).sum(axis=1), -1.0, 1.0)
        geod = np.arccos(dots)
    else:
        geod = np.asarray(geod_vec, dtype=np.float32)

    # Build batch for all ω in the discretization
    batch = {
        "start": torch.full((N,), s, dtype=torch.long, device=device),
        "omega": torch.arange(N, dtype=torch.long, device=device),
        "start_xyz": torch.tensor(
            np.repeat(start_xyz[None, :], N, axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "omega_xyz": torch.tensor(omega_xyz, dtype=torch.float32, device=device),
        "geod": torch.tensor(geod[:, None], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        f_pred = model(batch).detach().cpu().numpy().reshape(-1)

    # Ground truth row for this start
    j = int(np.where(start_indices == s)[0][0])
    phi_row = np.asarray(phi[j], dtype=float).reshape(-1)

    # Shared or independent normalization for colormaps
    if share_norm:
        vmin = float(min(f_pred.min(), phi_row.min()))
        vmax = float(max(f_pred.max(), phi_row.max()))
        norm_main = Normalize(vmin=vmin, vmax=vmax)
        f_vis, g_vis = f_pred, phi_row
    else:
        f_vis = (f_pred - f_pred.min()) / (f_pred.max() - f_pred.min() + 1e-12)
        g_vis = (phi_row - phi_row.min()) / (phi_row.max() - phi_row.min() + 1e-12)
        norm_main = Normalize(vmin=0.0, vmax=1.0)

    err = f_pred - phi_row
    norm_err = TwoSlopeNorm(
        vmin=float(err.min()),
        vcenter=0.0,
        vmax=float(err.max()),
    )

    # ---- PRED ----
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=18, c=f_vis, cmap=cmap_main, norm=norm_main)
    ax1.set_title(f"{title_prefix} Prediction f(x, ·)")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    apply_data_aspect(ax1, X)
    cbar1 = fig1.colorbar(
        ScalarMappable(norm=norm_main, cmap=cmap_main),
        ax=ax1,
        shrink=0.8,
    )
    cbar1.set_label(
        "f(x, ω)" + (" (shared scale)" if share_norm else " (individually normalized)")
    )
    plt.show()

    # ---- TRUTH ----
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], s=18, c=g_vis, cmap=cmap_main, norm=norm_main)
    ax2.set_title(f"{title_prefix} Ground truth φ_t(x)[·]")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    apply_data_aspect(ax2, X)
    cbar2 = fig2.colorbar(
        ScalarMappable(norm=norm_main, cmap=cmap_main),
        ax=ax2,
        shrink=0.8,
    )
    cbar2.set_label(
        "φ_t(x, ω)" + (" (shared scale)" if share_norm else " (individually normalized)")
    )
    plt.show()

    # ---- ERROR ----
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2], s=18, c=err, cmap=cmap_err, norm=norm_err)
    ax3.set_title(f"{title_prefix} Error: f(x, ·) − φ_t(x)[·]")
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    apply_data_aspect(ax3, X)
    cbar3 = fig3.colorbar(
        ScalarMappable(norm=norm_err, cmap=cmap_err),
        ax=ax3,
        shrink=0.8,
    )
    cbar3.set_label("Prediction − Truth")
    plt.show()


def visualize_several_validation_starts(
    model: torch.nn.Module,
    X: np.ndarray,
    start_indices: np.ndarray,
    phi: np.ndarray,
    val_start_ids: Iterable[int],
    how_many: int = 3,
    geod_rows: Optional[Dict[int, np.ndarray]] = None,
    title_prefix: str = "Val start",
) -> None:
    """
    Convenience helper: loop over a few validation start nodes and call plot_fx_field_for_start.
    val_start_ids: iterable of node ids used as starts in validation.
    geod_rows: optional dict {start_node_id -> geodesic_row}, for non-sphere manifolds.
    """
    val_list = list(val_start_ids)[:how_many]
    for s in val_list:
        s = int(s)
        j = int(np.where(start_indices == s)[0][0])
        geod_vec = None if geod_rows is None else geod_rows.get(s, None)
        plot_fx_field_for_start(
            model,
            X,
            start_indices,
            phi,
            s_idx=j,
            geod_vec=geod_vec,
            title_prefix=f"{title_prefix} s={s} | j={j}",
        )


# ---------------------------------------------------------------------
# 4) 1D diagnostics: prediction vs actual, error vs truth
# ---------------------------------------------------------------------

def plot_pred_vs_actual(
    y_true,
    y_pred,
    title: str = "Validation: prediction vs actual",
    savepath: Optional[str] = None,
) -> None:
    """
    Scatter plot of predictions vs actuals, with y=x and line of best fit.
    Prints slope, intercept, and R^2 of the best-fit line.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.1, s=10, label="Validation points")

    # Perfect prediction line y = x
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle="--", linewidth=1.5, label="y = x (perfect)")

    # Best-fit line y = a x + b
    a, b = np.polyfit(y_true, y_pred, 1)
    ax.plot([min_val, max_val], [a * min_val + b, a * max_val + b],
            linestyle="-", linewidth=1.5, label=f"fit: y={a:.3f}x+{b:.3f}")

    # R^2
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    print(f"Best-fit slope:     {a:.6f}")
    print(f"Best-fit intercept: {b:.6f}")
    print(f"R^2:                {r2:.6f}")

    ax.set_xlabel("Ground truth (actual)")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_error_vs_truth(
    y_true,
    y_pred,
    eps: float = 1e-3,
    error_type: str = "relative",
    savepath: Optional[str] = None,
) -> None:
    """
    Scatter plot: x = ground truth (actual), y = relative error or squared error.

    If error_type == "relative":
        error = |pred - y| / max(|y|, eps)
    If error_type == "squared":
        error = (pred - y)^2
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if error_type == "relative":
        denom = np.maximum(np.abs(y_true), eps)
        error = np.abs(y_pred - y_true) / denom
    elif error_type == "squared":
        error = (y_pred - y_true) ** 2
    else:
        raise ValueError("error_type must be 'relative' or 'squared'")

    fig, ax = plt.subplots()
    ax.scatter(y_true, error, alpha=0.3, s=10)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Ground truth (actual)")
    ax.set_ylabel(f"{error_type} error")
    ax.set_title(f"Validation: {error_type.capitalize()} error vs ground truth")
    ax.grid(True)

    print(f"Mean {error_type} error:   {error.mean():.6f}")
    print(f"Median {error_type} error: {np.median(error):.6f}")
    print(f"95th pct {error_type} err: {np.percentile(error, 95):.6f}")

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# 5) Kernel visualization
# ---------------------------------------------------------------------

def plot_kernel_row_for_start(
    X: np.ndarray,          # (N, 3)
    K_true: np.ndarray,     # (N, N)
    K_nn_all: np.ndarray,   # (N, N)
    start_idx: int,
    title_prefix: str = "Kernel",
) -> None:
    """
    Visualize a single kernel row K(x, ·):
      - NN kernel row vs ground truth
      - error (NN − GT)

    start_idx is the index of x in the full grid X (0..N-1).
    """
    X = np.asarray(X, dtype=float)
    K_true = np.asarray(K_true, dtype=float)
    K_nn_all = np.asarray(K_nn_all, dtype=float)

    N = X.shape[0]
    i = int(start_idx)
    if not (0 <= i < N):
        raise ValueError(f"start_idx={i} out of range 0..{N-1}")

    gt_row = K_true[i]       # (N,)
    nn_row = K_nn_all[i]     # (N,)

    # Shared normalization for NN vs GT
    vmin = float(min(gt_row.min(), nn_row.min()))
    vmax = float(max(gt_row.max(), nn_row.max()))
    norm_main = Normalize(vmin=vmin, vmax=vmax)

    err = nn_row - gt_row
    norm_err = TwoSlopeNorm(
        vmin=float(err.min()),
        vcenter=0.0,
        vmax=float(err.max()),
    )

    # ---- NN kernel row ----
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=18, c=nn_row, cmap="OrRd", norm=norm_main)
    ax1.set_title(f"{title_prefix} NN K(x, ·), start={i}")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    apply_data_aspect(ax1, X)
    cbar1 = fig1.colorbar(
        ScalarMappable(norm=norm_main, cmap="OrRd"),
        ax=ax1,
        shrink=0.8,
    )
    cbar1.set_label("K_nn(x, ·)")
    plt.show()

    # ---- Ground-truth kernel row ----
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], s=18, c=gt_row, cmap="OrRd", norm=norm_main)
    ax2.set_title(f"{title_prefix} GT K(x, ·), start={i}")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    apply_data_aspect(ax2, X)
    cbar2 = fig2.colorbar(
        ScalarMappable(norm=norm_main, cmap="OrRd"),
        ax=ax2,
        shrink=0.8,
    )
    cbar2.set_label("K_true(x, ·)")
    plt.show()

    # ---- Error row ----
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2], s=18, c=err, cmap="coolwarm", norm=norm_err)
    ax3.set_title(f"{title_prefix} Error K_nn(x, ·) − K_true(x, ·)")
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    apply_data_aspect(ax3, X)
    cbar3 = fig3.colorbar(
        ScalarMappable(norm=norm_err, cmap="coolwarm"),
        ax=ax3,
        shrink=0.8,
    )
    cbar3.set_label("K_nn(x, ·) − K_true(x, ·)")
    plt.show()
