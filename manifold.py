# ============================================================
# End-to-end: Learn f(x, ω) ≈ φ_t(x)[ω] on S^2
#  - Build sphere graph + g-GRF signatures (diffusion modulation α_k = t^k/k!)
#  - Dataset: ([x_j || ω_{j,i}], target = φ_t(x_j)[ω_{j,i}])
#  - Train NN: continuous features (coords + geodesic). Optionally add ID embeddings.
# ============================================================

import math
import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from tqdm import tqdm

# =============== CONFIG ===============
CFG = dict(
    N=4000,                # sphere sample size
    k=8,                  # kNN
    num_starts=1000,       # number of x_j
    start_mode="farthest",# or "random"
    n_walks=5000,         # by default = N (per start)
    phalt=0.01,           # termination prob in Alg. 1
    t=20.,                # diffusion scale (τ = t)
    seed=7,

    subsample_per_start=None,  # None = use all ω; or int (e.g., 300)

    # model/training
    use_continuous_only=True,  # True: only coords+geodesic (continuous RF); False: + ID embeddings
    d_emb=64,
    hidden=1024,
    batch_size=1024,
    lr=1e-3,
    epochs=500,
    val_frac=0.2,
    torch_seed=0,
)

class RelativeErrorLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # avoid division by zero by clamping |target|
        denom = torch.clamp(torch.abs(target), min=self.eps)
        rel_err = torch.abs(pred - target) / denom
        return rel_err.mean()

class SignatureAtNodeRegressor(nn.Module):
    """
    f(x, ω): default uses ONLY coords + geodesic (continuous RF).
    If use_ids=True, adds ID embeddings for start and omega.
    """
    def __init__(self, num_nodes: int, x_coords: np.ndarray,
                    d_emb: int = 64, hidden: int = 128,
                    use_continuous_only: bool = True):
        super().__init__()
        self.register_buffer("Xcoords", torch.tensor(x_coords, dtype=torch.float32))  # (N,3)
        self.use_cont = use_continuous_only
        self.use_ids = not use_continuous_only
        if self.use_ids:
            self.start_emb = nn.Embedding(num_nodes, d_emb)
            self.omega_emb = nn.Embedding(num_nodes, d_emb)

        # inputs: geod (1), start_xyz (3), omega_xyz (3) -> 7 dims
        in_dim = 7
        if self.use_ids:
            in_dim += 2 * d_emb

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden*2), nn.ReLU(),
            nn.Linear(hidden*2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, batch):
        s = batch["start"]; o = batch["omega"]; geod = batch["geod"]
        pieces = [batch["start_xyz"], batch["omega_xyz"], geod]
        if self.use_ids:
            s_emb = self.start_emb(s); o_emb = self.omega_emb(o)
            pieces = [s_emb, o_emb] + pieces
        x = torch.cat(pieces, dim=-1)
        return self.mlp(x).squeeze(-1)

# ======================================

# ----------------------------
# Sphere + graph + g-GRF (Alg. 1) signatures
# ----------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * (i + 0.5)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return np.column_stack((x, y, z))

def build_knn_graph(X: np.ndarray, k: int = 16):
    """Symmetric kNN with Gaussian weights; return W_raw, Wf, neighbors, deg_unw, sigma^2."""
    N = X.shape[0]
    XXT = X @ X.T
    D2 = np.clip(2.0 - 2.0 * XXT, 0.0, None)  # chord^2 on S^2
    np.fill_diagonal(D2, np.inf)
    kth_sq = np.partition(D2, k, axis=1)[:, k]
    sigma2 = float(np.median(kth_sq))
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        nn = np.argpartition(D2[i], k)[:k]
        W[i, nn] = np.exp(-D2[i, nn] / sigma2)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    Wf = (D_inv_sqrt[:, None] * W) * D_inv_sqrt[None, :]
    A_unw = (W > 0).astype(np.int32)
    deg_unw = A_unw.sum(axis=1).astype(np.int32)
    neighbors = [np.flatnonzero(A_unw[i]).astype(np.int32) for i in range(N)]
    return W, Wf, neighbors, deg_unw, sigma2

def farthest_point_sample(coords: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = coords.shape[0]
    start = int(rng.integers(0, N))
    sel = [start]
    d2 = np.sum((coords - coords[start])**2, axis=1)
    for _ in range(1, m):
        idx = int(np.argmax(d2))
        sel.append(idx)
        d2 = np.minimum(d2, np.sum((coords - coords[idx])**2, axis=1))
    return np.array(sel, dtype=int)

def precompute_alpha(tau: float, Kmax: int = 10000, eps: float = 1e-300) -> np.ndarray:
    """Diffusion modulation α_k = τ^k / k! via stable recurrence; α_0=1."""
    vals = [1.0]
    k = 1
    while k < Kmax:
        nxt = vals[-1] * (tau / k)
        if nxt < eps: break
        vals.append(nxt); k += 1
    return np.array(vals, dtype=float)

def build_ggrf_signatures_on_sphere(
    N: int, k: int, num_starts: int, start_mode: str,
    n_walks: int, phalt: float, tau: float, seed: int
) -> Dict[str, Any]:
    """
    Build signatures φ_t(x_j) using g-GRF Algorithm 1 with uniform-by-count neighbor sampling.
    Returns dict with:
      X (N,3), start_indices (J,), starts_xyz (J,3),
      phi (J, N), params {...}
    """
    rng = np.random.default_rng(seed)
    X = fibonacci_sphere(N)
    W_raw, Wf, neighbors, deg_unw, sigma2 = build_knn_graph(X, k=k)

    if start_mode == "farthest":
        start_indices = farthest_point_sample(X, num_starts, seed=seed)
    elif start_mode == "random":
        start_indices = rng.choice(N, size=num_starts, replace=False)
    else:
        raise ValueError("start_mode must be 'farthest' or 'random'")

    # AFTER  — use the symmetric modulation f, i.e. α with τ/2
    f = precompute_alpha(0.5 * tau, Kmax=10000, eps=1e-300)
    def modulation(step: int) -> float:
        return float(f[step]) if step < f.shape[0] else 0.0

    # alpha = precompute_alpha(tau, Kmax=10000, eps=1e-300)
    # def modulation(step: int) -> float:
    #     return float(alpha[step]) if step < alpha.shape[0] else 0.0 

    if n_walks is None: n_walks = N

    phi = np.zeros((num_starts, N), dtype=float)

    def sample_feature(start_idx: int):
        vec = np.zeros(N, dtype=float)
        local_rng = np.random.default_rng(346511053)
        while vec.dtype == float:  # silly loop to scope variables (runs once)
            for _ in range(n_walks):
                u = int(start_idx)
                load = 1.0
                step = 0
                terminated = False
                while not terminated:
                    vec[u] += load * modulation(step)
                    step += 1
                    nbrs = neighbors[u]
                    if nbrs.size == 0: break
                    v = int(nbrs[local_rng.integers(0, nbrs.size)])
                    # Alg. 1 update (uniform-by-count neighbor sampling):
                    load *= (deg_unw[u] / (1.0 - phalt)) * float(Wf[u, v])
                    u = v
                    terminated = (local_rng.random() < phalt)
            break
        vec /= float(n_walks)
        return vec

    # for j, s in enumerate(start_indices):
    for j, s in tqdm(enumerate(start_indices), total=num_starts, desc="Building g-GRF signatures"):
        phi[j] = sample_feature(int(s))

    return {
        "X": X,
        "start_indices": start_indices,
        "starts_xyz": X[start_indices],
        "phi": phi,
        "params": {
            "N": int(N), "k": int(k), "num_starts": int(num_starts),
            "start_mode": start_mode, "n_walks": int(n_walks),
            "phalt": float(phalt), "tau": float(tau), "sigma2": float(sigma2)
        }
    }

# ----------------------------
# Supervision: (x_j, ω_{j,i}) -> y_{j,i} = φ_t(x_j)[ω_{j,i}]
# ----------------------------

def make_node_pair_supervision(out: Dict[str, Any], mode: str = "all", M_per_start=None, seed: int = 123, truncate=True):
    """
    Build arrays: start_ids (M,), omega_ids (M,), targets (M,) and geometry arrays.
    If M_per_start is set with mode="random", uniformly subsample that many ω per start x_j.
    """
    X = out["X"]
    start_ids = out["start_indices"]
    phi = out["phi"]
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    starts, omegas, targets = [], [], []
    for j, s in enumerate(start_ids):
        nodes = np.arange(N)
        if mode == "random" and (M_per_start is not None) and (M_per_start < N):
            nodes = rng.choice(N, size=M_per_start, replace=False)
        for i in nodes:
            if truncate and phi[j, i] < 0.1:
                if rng.random() < 0.975:
                    continue
            starts.append(int(s))
            omegas.append(int(i))
            targets.append(float(phi[j, i]))

    starts = np.array(starts, dtype=np.int64)
    omegas = np.array(omegas, dtype=np.int64)
    targets = np.array(targets, dtype=np.float32)
    return {"starts": starts, "omegas": omegas, "targets": targets, "X": X, "N": N}

# ----------------------------
# PyTorch model (continuous RF by default)
# ----------------------------

def ensure_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        print("PyTorch not found. Please install torch.")
        return False

if ensure_torch():

    class NodePairDataset(Dataset):
        def __init__(self, sup: Dict[str, Any], use_continuous_only: bool = True):
            self.starts = sup["starts"]      # (M,)
            self.omegas = sup["omegas"]      # (M,)
            self.targets = sup["targets"]    # (M,)
            self.X = sup["X"]                # (N,3)
            self.use_cont = use_continuous_only
        def __len__(self): return len(self.targets)
        def __getitem__(self, idx):
            s = int(self.starts[idx]); o = int(self.omegas[idx])
            start_xyz = self.X[s].astype(np.float32)
            omega_xyz = self.X[o].astype(np.float32)
            dot = float(np.clip(np.dot(start_xyz, omega_xyz), -1.0, 1.0))
            geod = float(np.arccos(dot))
            ex = {
                "start": s, "omega": o,
                "start_xyz": start_xyz, "omega_xyz": omega_xyz,
                "geod": geod, "target": float(self.targets[idx])
            }
            return ex

    def collate(batch):
        import torch
        return {
            "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
            "omega": torch.tensor([b["omega"] for b in batch], dtype=torch.long),
            "start_xyz": torch.tensor([b["start_xyz"] for b in batch], dtype=torch.float32),
            "omega_xyz": torch.tensor([b["omega_xyz"] for b in batch], dtype=torch.float32),
            "geod": torch.tensor([b["geod"] for b in batch], dtype=torch.float32).unsqueeze(-1),
            "target": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
        }
    
    # ----------------------------
    # Training utilities
    # ----------------------------

    def split_indices(M, val_frac=0.2, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.arange(M); rng.shuffle(idx)
        k = int(M * (1 - val_frac))
        return idx[:k], idx[k:]

    def split_by_start(starts_all: np.ndarray, val_frac: float = 0.2, seed: int = 0):
        rng = np.random.default_rng(seed)
        unique = np.unique(starts_all)
        rng.shuffle(unique)
        m_val = max(1, int(round(len(unique) * val_frac)))
        val_starts = np.sort(unique[:m_val])
        train_starts = np.sort(unique[m_val:])
        train_mask = np.isin(starts_all, train_starts)
        val_mask   = np.isin(starts_all, val_starts)
        train_idx = np.nonzero(train_mask)[0]
        val_idx   = np.nonzero(val_mask)[0]
        return train_idx, val_idx, train_starts, val_starts

    class NodePairDatasetWithGeod(NodePairDataset):
        def __init__(self, sup: Dict[str, Any], geod_matrix: np.ndarray, use_continuous_only: bool = True):
            super().__init__(sup, use_continuous_only=use_continuous_only)
            # geod_matrix can be full N x N, or rows only for the starts (pass an indexer to map)
            self.G = geod_matrix  # shape (N, N) OR (num_starts, N) if you index by start position
            self.map_from_start_to_row = None  # filled if G is subset

        def set_row_indexer(self, start_indices: np.ndarray):
            """Optional: if G has rows only for specific start nodes in 'start_indices' (validation/train starts),
               set this so we can pick correct geodesic rows quickly."""
            pos = {int(s): i for i, s in enumerate(start_indices)}
            self.map_from_start_to_row = pos

        def __getitem__(self, idx):
            ex = super().__getitem__(idx)
            s = int(ex["start"]); o = int(ex["omega"])
            if self.map_from_start_to_row is None:
                geod_val = float(self.G[s, o])
            else:
                geod_val = float(self.G[self.map_from_start_to_row[s], o])
            ex["geod"] = geod_val
            return ex

    def train_model_node_pairs_by_start(
        sup: Dict[str, Any], geod_matrix: np.ndarray,
        batch_size=1024, epochs=8, lr=1e-3,
        d_emb=64, hidden=128, use_continuous_only=True,
        val_frac=0.2, seed=0, val_start_ids=None
    ):
        import torch
        ds = NodePairDatasetWithGeod(sup, geod_matrix, use_continuous_only=use_continuous_only)

        if val_start_ids is None:
            tr_idx, va_idx, tr_starts, va_starts = split_by_start(sup["starts"], val_frac=val_frac, seed=seed)
        else:
            # user-specified validation starts
            all_starts = np.unique(sup["starts"])
            tr_starts = np.setdiff1d(all_starts, val_start_ids)
            va_starts = np.array(sorted(np.unique(val_start_ids)))
            tr_idx = np.nonzero(np.isin(sup["starts"], tr_starts))[0]
            va_idx = np.nonzero(np.isin(sup["starts"], va_starts))[0]

        class Subset(torch.utils.data.Dataset):
            def __init__(self, base, idxs): self.base, self.idxs = base, idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.base[self.idxs[i]]

        tr_ds, va_ds = Subset(ds, tr_idx), Subset(ds, va_idx)
        tr = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
        va = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

        model = SignatureAtNodeRegressor(
            num_nodes=sup["N"], x_coords=sup["X"],
            d_emb=d_emb, hidden=hidden, use_continuous_only=use_continuous_only
        )
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        # loss_fn = RelativeErrorLoss(eps=1e-1)

        def eval_loader(dl):
            model.eval()
            se, n = 0.0, 0
            with torch.no_grad():
                for batch in dl:
                    pred = model(batch); y = batch["target"]
                    se += torch.sum((pred - y)**2).item()
                    n += y.numel()
            return math.sqrt(se / n)

        def absolute_relative_error(dl, eps: float = 1e-3):
            model.eval()
            re_sum, n = 0.0, 0
            with torch.no_grad():
                for batch in dl:
                    pred = model(batch)
                    y = batch["target"]

                    # Avoid division by zero by clamping |y| from below
                    denom = torch.clamp(torch.abs(y), min=eps)
                    rel_err = torch.abs(pred - y) / denom

                    re_sum += rel_err.sum().item()
                    n += y.numel()

            return re_sum / n if n > 0 else float("nan")

        hist = {
            "rmse_train": [],
            "rmse_val": [],
            "relerr_train": [],
            "relerr_val": [],
        }

        for ep in range(1, epochs + 1):
            model.train()
            for batch in tr:
                pred = model(batch); y = batch["target"]
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # evaluation
            rmse_tr = eval_loader(tr)
            rmse_va = eval_loader(va)
            rel_tr  = absolute_relative_error(tr)
            rel_va  = absolute_relative_error(va)

            hist["rmse_train"].append(rmse_tr)
            hist["rmse_val"].append(rmse_va)
            hist["relerr_train"].append(rel_tr)
            hist["relerr_val"].append(rel_va)

            print(
                f"Epoch {ep:02d} | "
                f"RMSE train: {rmse_tr:.6f} | RMSE val: {rmse_va:.6f} | "
                f"RelErr train: {rel_tr:.6f} | RelErr val: {rel_va:.6f}"
            )

        return model, hist, {"train_starts": tr_starts, "val_starts": va_starts,
                             "train_idx": tr_idx, "val_idx": va_idx}

def collect_val_predictions(
    model,
    sup,
    geod_matrix,
    val_starts,
    batch_size=1024,
    use_continuous_only=True,
):
    """
    Run the trained model on the validation indices and return
    (y_true, y_pred) as 1D numpy arrays.
    """
    # Rebuild the same dataset used in training
    ds = NodePairDatasetWithGeod(
        sup, geod_matrix,
        use_continuous_only=use_continuous_only
    )

    class Subset(Dataset):
        def __init__(self, base, idxs): self.base, self.idxs = base, idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]

    val_idx = np.where(np.isin(sup["starts"], val_starts))[0]
    va_ds = Subset(ds, val_idx)
    va_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in va_loader:
            y_hat = model(batch).reshape(-1)
            y = batch["target"].reshape(-1)

            preds.append(y_hat.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return targets, preds


def plot_dynamics(hist, title="Training/Validation", rmse=True, savepath=None):
    if rmse: train_str = "rmse_train"; val_str = "rmse_val"
    else: train_str = "relerr_train"; val_str = "relerr_val"
    title_str = "RMSE" if rmse else "Relative Error"
    xs = np.arange(1, len(hist[train_str])+1)
    plt.figure()
    plt.plot(xs, hist[train_str], label="train")
    plt.plot(xs, hist[val_str],   label="validation")
    plt.xlabel("epoch"); plt.ylabel(title_str); plt.title(title + ": " + title_str)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

def apply_data_aspect(ax, X):
    """Make 3D axes respect the data's x/y/z ranges."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng  = maxs - mins
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    # Match the axes box to the data's aspect so anisotropy is preserved
    ax.set_box_aspect(rng)

def plot_fx_field_for_start_generic(model, X, start_indices, phi, s_idx, geod_vec=None,
                                    title_prefix="", share_norm=True,
                                    cmap_main="OrRd", cmap_err="coolwarm"):
    """Same as your plot_fx_field, but you can pass geod_vec explicitly (for non-sphere cases)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    model.eval()
    device = next(model.parameters()).device
    N = X.shape[0]
    s = int(start_indices[s_idx])
    start_xyz = X[s]
    omega_xyz = X

    if geod_vec is None:
        dots = np.clip((omega_xyz * start_xyz).sum(axis=1), -1.0, 1.0)
        geod = np.arccos(dots)
    else:
        geod = geod_vec.astype(np.float32)

    batch = {
        "start": torch.full((N,), s, dtype=torch.long, device=device),
        "omega": torch.arange(N, dtype=torch.long, device=device),
        "start_xyz": torch.tensor(np.repeat(start_xyz[None,:], N, axis=0), dtype=torch.float32, device=device),
        "omega_xyz": torch.tensor(omega_xyz, dtype=torch.float32, device=device),
        "geod": torch.tensor(geod[:,None], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        f_pred = model(batch).detach().cpu().numpy()

    # truth row for this start
    j = int(np.where(start_indices == s)[0][0])
    phi_row = phi[j].astype(float)

    # shared or independent normalization
    if share_norm:
        vmin = min(f_pred.min(), phi_row.min())
        vmax = max(f_pred.max(), phi_row.max())
        norm_main = Normalize(vmin=vmin, vmax=vmax)
        f_vis = f_pred; g_vis = phi_row
    else:
        f_vis = (f_pred - f_pred.min()) / (f_pred.max() - f_pred.min() + 1e-12)
        g_vis = (phi_row - phi_row.min()) / (phi_row.max() - phi_row.min() + 1e-12)
        norm_main = Normalize(vmin=0.0, vmax=1.0)

    err = f_pred - phi_row
    norm_err = TwoSlopeNorm(vmin=err.min(), vcenter=0.0, vmax=err.max())

    # ---- PRED ----
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2], s=18, c=f_vis, cmap=cmap_main, norm=norm_main)
    ax1.set_title(f"{title_prefix} NN prediction f(x, ·)")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    apply_data_aspect(ax1, X)    
    cbar1 = fig1.colorbar(ScalarMappable(norm=norm_main, cmap=cmap_main), ax=ax1, shrink=0.8)
    cbar1.set_label("f(x, ω)" + (" (shared scale)" if share_norm else " (individually normalized)"))
    plt.show()

    # ---- TRUTH ----
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X[:,0], X[:,1], X[:,2], s=18, c=g_vis, cmap=cmap_main, norm=norm_main)
    ax2.set_title(f"{title_prefix} Ground truth φ_t(x)[·]")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    apply_data_aspect(ax2, X)
    cbar2 = fig2.colorbar(ScalarMappable(norm=norm_main, cmap=cmap_main), ax=ax2, shrink=0.8)
    cbar2.set_label("φ_t(x, ω)" + (" (shared scale)" if share_norm else " (individually normalized)"))
    plt.show()

    # ---- ERROR ----
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(X[:,0], X[:,1], X[:,2], s=18, c=err, cmap=cmap_err, norm=norm_err)
    ax3.set_title(f"{title_prefix} Error: f(x, ·) − φ_t(x)[·]")
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    apply_data_aspect(ax3, X)
    cbar3 = fig3.colorbar(ScalarMappable(norm=norm_err, cmap=cmap_err), ax=ax3, shrink=0.8)
    cbar3.set_label("Prediction − Truth")
    plt.show()

def visualize_several_validation_starts(model, X, start_indices, phi, val_start_ids, how_many=3,
                                        geod_rows=None,
                                        title_prefix="Val start"):
    """Pick first `how_many` validation start nodes and plot."""
    val_list = list(val_start_ids)[:how_many]
    for s in val_list:
        j = int(np.where(start_indices == s)[0][0])
        geod_vec = None if geod_rows is None else geod_rows[int(s)]
        plot_fx_field_for_start_generic(model, X, start_indices, phi, j,
                                        geod_vec=geod_vec,
                                        title_prefix=f"{title_prefix} s={int(s)} | j={j} ")

def plot_pred_vs_actual(y_true, y_pred, title="Validation: prediction vs actual"):
    """
    Scatter plot of predictions vs actuals, with y=x and line of best fit.
    Prints slope, intercept, and R^2 of the best-fit line.
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots()

    # Scatter of predictions vs actual
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, label="Validation points")

    # Perfect prediction line y = x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle="--", linewidth=1.5, label="y = x (perfect)")

    # Best-fit line: y_pred ≈ m * y_true + b
    m, b = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(min_val, max_val, 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, linewidth=2,
            label=f"Best fit: y = {m:.3f}x + {b:.3f}")

    # R^2 for the fit
    ss_res = np.sum((y_pred - (m * y_true + b))**2)
    ss_tot = np.sum((y_pred - y_pred.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    print(f"Slope (m):     {m:.6f}")
    print(f"Intercept (b): {b:.6f}")
    print(f"R^2:           {r2:.6f}")

    plt.show()

def plot_error_vs_truth(y_true, y_pred, eps=1e-3, error_type="relative"):
    """
    Scatter plot: x = ground truth (actual), y = relative error or squared error.
    Relative error: |pred - y| / max(|y|, eps)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if error_type == "relative":
        denom = np.maximum(np.abs(y_true), eps)
        error = np.abs(y_pred - y_true) / denom
    else:
        error = (y_pred - y_true)**2

    fig, ax = plt.subplots()

    ax.scatter(y_true, error, alpha=0.3, s=10)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Ground truth (actual)")
    ax.set_ylabel(f"{error_type} error")
    ax.set_title(f"Validation: {error_type.capitalize()} error vs ground truth")
    ax.grid(True)

    # Optional: print some summary stats
    print(f"Mean {error_type} error:   {error.mean():.6f}")
    print(f"Median {error_type} error: {np.median(error):.6f}")
    print(f"95th pct {error_type} err: {np.percentile(error, 95):.6f}")

    plt.show()

def geodesic_matrix_sphere(X: np.ndarray) -> np.ndarray:
    """Exact spherical geodesic on S^2 via arccos of dot product."""
    dots = np.clip(X @ X.T, -1.0, 1.0)
    return np.arccos(dots)

if __name__ == "__main__":
    # Build signatures on S^2 (your code does this already)
    # out = build_ggrf_signatures_on_sphere(
    #     N=CFG["N"], k=CFG["k"], num_starts=CFG["num_starts"],
    #     start_mode=CFG["start_mode"], n_walks=CFG["n_walks"],
    #     phalt=CFG["phalt"], tau=CFG["t"], seed=CFG["seed"]
    # )
    # with open("out_0.1.pkl", "wb") as f:
    #     pickle.dump(out, f)

    with open("out_0.1.pkl", "rb") as f:
        out = pickle.load(f)
        
    X, phi, start_indices = out["X"], out["phi"], out["start_indices"]

    # Supervision (all ω per start or subsample)
    sup = make_node_pair_supervision(
        out,
        mode=("all" if CFG["subsample_per_start"] is None else "random"),
        M_per_start=CFG["subsample_per_start"],
        seed=123
    )
    sup["N"] = out["params"]["N"]

    # Geodesics on the sphere
    G_sphere = geodesic_matrix_sphere(X)  # N x N

    # Train with start-wise holdout (so any visualized row is validation-only)
    model_sphere, hist_sphere, splits_sphere = train_model_node_pairs_by_start(
        sup, G_sphere,
        batch_size=CFG["batch_size"], epochs=CFG["epochs"], lr=CFG["lr"],
        d_emb=CFG["d_emb"], hidden=CFG["hidden"], use_continuous_only=CFG["use_continuous_only"],
        val_frac=CFG["val_frac"], seed=CFG["torch_seed"]
    )

    with open("result.pkl", "wb") as f:
        pickle.dump({
            "model": model_sphere,
            "hist": hist_sphere,
            "indices": splits_sphere
        }, f)

    with open("result.pkl", "rb") as f:
        data = pickle.load(f)
        model_sphere, hist_sphere, splits_sphere = data["model"], data["hist"], data["indices"]

    sup_all = make_node_pair_supervision(
        out,
        mode=("all" if CFG["subsample_per_start"] is None else "random"),
        M_per_start=CFG["subsample_per_start"],
        seed=123,
        truncate=False
    )
    # Get predictions / actuals on validation set
    y_true_va, y_pred_va = collect_val_predictions(
        model_sphere,
        sup_all,
        G_sphere,
        val_starts=splits_sphere["val_starts"],
        batch_size=1024,
        use_continuous_only=True,
    )

    plot_dynamics(hist_sphere, title="Sphere: Relative Error", rmse=False, savepath="relerr.pdf")
    plot_dynamics(hist_sphere, title="Sphere: RMSE", savepath="rmse.pdf")
    plot_pred_vs_actual(y_true_va, y_pred_va)
    plot_error_vs_truth(y_true_va, y_pred_va)
    plot_error_vs_truth(y_true_va, y_pred_va, error_type="squared")
    visualize_several_validation_starts(
        model_sphere, X, start_indices, phi,
        val_start_ids=splits_sphere["val_starts"],
        how_many=3,
        title_prefix="Sphere Val start"
    )