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
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

# =============== CONFIG ===============
CFG = dict(
    N=600,                # sphere sample size
    k=8,                  # kNN
    num_starts=200,       # number of x_j
    start_mode="farthest",# or "random"
    n_walks=2000,         # by default = N (per start)
    phalt=0.01,           # termination prob in Alg. 1
    t=10.,                # diffusion scale (τ = t)
    seed=7,

    subsample_per_start=None,  # None = use all ω; or int (e.g., 300)

    # model/training
    use_continuous_only=True,  # True: only coords+geodesic (continuous RF); False: + ID embeddings
    d_emb=64,
    hidden=1024,
    batch_size=1024,
    lr=1e-3,
    epochs=300,
    val_frac=0.2,
    torch_seed=0,
)

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
    n_walks: int | None, phalt: float, tau: float, seed: int
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

    for j, s in enumerate(start_indices):
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

def make_node_pair_supervision(out: Dict[str, Any], mode: str = "all", M_per_start: int | None = None, seed: int = 123):
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
            if phi[j, i] < 0.01:
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

    def train_model_node_pairs(
        sup: Dict[str, Any],
        batch_size=1024, epochs=8, lr=1e-3,
        d_emb=64, hidden=128, use_continuous_only=True, val_frac=0.2, seed=0
    ):
        import torch
        ds = NodePairDataset(sup, use_continuous_only=use_continuous_only)
        # train_idx, val_idx = split_indices(len(ds), val_frac=val_frac, seed=seed)
        train_idx, val_idx, train_starts, val_starts = split_by_start(sup["starts"], val_frac=val_frac, seed=seed)

        class Subset(torch.utils.data.Dataset):
            def __init__(self, base, idxs): self.base, self.idxs = base, idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.base[self.idxs[i]]

        tr_ds, va_ds = Subset(ds, train_idx), Subset(ds, val_idx)
        tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
        va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

        model = SignatureAtNodeRegressor(
            num_nodes=sup["N"], x_coords=sup["X"],
            d_emb=d_emb, hidden=hidden, use_continuous_only=use_continuous_only
        )
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_fn = torch.nn.MSELoss()

        def eval_loader(dl):
            model.eval()
            se, n = 0.0, 0
            with torch.no_grad():
                for batch in dl:
                    pred = model(batch); y = batch["target"]
                    se += torch.sum((pred - y)**2).item()
                    n += y.numel()
            return math.sqrt(se / n)

        for ep in range(1, epochs+1):
            model.train()
            for batch in tr:
                pred = model(batch); y = batch["target"]
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
            rmse_tr = eval_loader(tr)
            rmse_va = eval_loader(va)
            print(f"Epoch {ep:02d} | RMSE train: {rmse_tr:.6f} | RMSE val: {rmse_va:.6f}")
        return model

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
        val_frac=0.2, seed=0, val_start_ids: np.ndarray | None = None
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

        def eval_loader(dl):
            model.eval()
            se, n = 0.0, 0
            with torch.no_grad():
                for batch in dl:
                    pred = model(batch); y = batch["target"]
                    se += torch.sum((pred - y)**2).item()
                    n += y.numel()
            return math.sqrt(se / n)

        hist = {"rmse_train": [], "rmse_val": []}
        for ep in range(1, epochs+1):
            model.train()
            for batch in tr:
                pred = model(batch); y = batch["target"]
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
            rmse_tr = eval_loader(tr)
            rmse_va = eval_loader(va)
            hist["rmse_train"].append(rmse_tr)
            hist["rmse_val"].append(rmse_va)
            print(f"Epoch {ep:02d} | RMSE train: {rmse_tr:.6f} | RMSE val: {rmse_va:.6f}")

        return model, hist, {"train_starts": tr_starts, "val_starts": va_starts,
                             "train_idx": tr_idx, "val_idx": va_idx}

def plot_dynamics(hist, title="Training/Validation RMSE"):
    import matplotlib.pyplot as plt
    xs = np.arange(1, len(hist["rmse_train"])+1)
    plt.figure()
    plt.plot(xs, hist["rmse_train"], label="train")
    plt.plot(xs, hist["rmse_val"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("RMSE"); plt.title(title)
    plt.legend(); plt.show()

def geodesic_matrix_sphere(X: np.ndarray) -> np.ndarray:
    """Exact spherical geodesic on S^2 via arccos of dot product."""
    dots = np.clip(X @ X.T, -1.0, 1.0)
    return np.arccos(dots)

def main(CFG):
    # Build signatures on S^2 (your code does this already)
    out = build_ggrf_signatures_on_sphere(
        N=CFG["N"], k=CFG["k"], num_starts=CFG["num_starts"],
        start_mode=CFG["start_mode"], n_walks=CFG["n_walks"],
        phalt=CFG["phalt"], tau=CFG["t"], seed=CFG["seed"]
    )
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

    plot_dynamics(hist_sphere, title="Sphere: RMSE")

    return out, model_sphere

if __name__ == "__main__":
    out, model = main(CFG)
    X  = out["X"]
    phi = out["phi"]
    start_indices = out["start_indices"]