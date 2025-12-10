from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.special import eval_legendre
from tqdm import tqdm

import pickle


# ============================================================
# Config
# ============================================================

@dataclass
class ManifoldConfig:
    # discretization / graph
    N: int                      # number of nodes in discretization
    k: int = 8                  # k-NN degree in graph
    num_starts: int = 1000      # number of source nodes x_j
    start_mode: str = "farthest"  # or "random"
    n_walks: Optional[int] = None # random walks per source (default: N)
    phalt: float = 0.01         # termination prob in Alg.1
    t: float = 20.0             # diffusion scale (tau)
    seed: int = 0

    # dataset
    subsample_per_start: Optional[int] = None  # None: all ω; otherwise random subset size

    # model / training
    use_continuous_only: bool = True   # True: coords+geodesic only; False: add ID embeddings
    d_emb: int = 64
    hidden: int = 128
    batch_size: int = 1024
    lr: float = 1e-3
    epochs: int = 1000
    val_frac: float = 0.2
    torch_seed: int = 0


# ============================================================
# Core utilities: discretization + graphs
# ============================================================


def fibonacci_sphere_scaled(n: int, a: float = 1.0, b: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Ellipsoid sample obtained by scaling Fibonacci sphere along axes."""
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * (i + 0.5)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    S = np.column_stack((x, y, z))
    return S * np.array([a, b, c], dtype=float)[None, :]


def torus_grid(N: int, R: float, r: float) -> np.ndarray:
    """Uniform grid sampling of a torus in R^3."""
    n_side = int(np.ceil(np.sqrt(N)))
    theta = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi)
    Theta = Theta.flatten()[:N]
    Phi = Phi.flatten()[:N]
    x = (R + r * np.cos(Phi)) * np.cos(Theta)
    y = (R + r * np.cos(Phi)) * np.sin(Theta)
    z = r * np.sin(Phi)
    return np.stack([x, y, z], axis=1)


def mobius_strip_grid(N: int, width: float) -> np.ndarray:
    """Uniform grid sampling of a Möbius strip in R^3."""
    n_side = int(np.ceil(np.sqrt(N)))
    u = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    v = np.linspace(-width / 2, width / 2, n_side)
    U, V = np.meshgrid(u, v)
    U = U.flatten()[:N]
    V = V.flatten()[:N]
    x = (1 + (V / 2) * np.cos(U / 2)) * np.cos(U)
    y = (1 + (V / 2) * np.cos(U / 2)) * np.sin(U)
    z = (V / 2) * np.sin(U / 2)
    return np.stack([x, y, z], axis=1)


def swiss_roll(N: int, noise: float = 0.0) -> np.ndarray:
    """Standard swiss roll sampling in R^3."""
    rng = np.random.default_rng(12345)
    t = (3 * np.pi / 2) * (1 + 2 * rng.random(N))  # angle
    x = t * np.cos(t)
    y = 21 * rng.random(N)                          # height
    z = t * np.sin(t)
    X = np.stack([x, y, z], axis=1)
    if noise > 0.0:
        X += noise * rng.normal(size=X.shape)
    return X


def build_knn_graph_euclidean(X: np.ndarray, k: int = 16):
    """
    Symmetric kNN with Gaussian weights on Euclidean distances.

    Returns
    -------
    W_raw : (N, N) float
        Symmetric adjacency with weights exp(-||x_i-x_j||^2 / sigma2).
    Wf : (N, N) float
        Symmetrically normalized adjacency D^{-1/2} W D^{-1/2}.
    neighbors : list[np.ndarray]
        neighbors[i] is an int array of neighbors of i.
    deg_unw : (N,) int
        Unweighted degree of each node.
    sigma2 : float
        Bandwidth used in the Gaussian.
    """
    N = X.shape[0]
    sq = np.sum(X * X, axis=1)
    XXT = X @ X.T
    D2 = sq[:, None] + sq[None, :] - 2.0 * XXT
    D2 = np.clip(D2, 0.0, None)
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
    """Farthest-point sampling in Euclidean metric."""
    rng = np.random.default_rng(seed)
    N = coords.shape[0]
    start = int(rng.integers(0, N))
    sel = [start]
    d2 = np.sum((coords - coords[start]) ** 2, axis=1)
    for _ in range(1, m):
        idx = int(np.argmax(d2))
        sel.append(idx)
        d2 = np.minimum(d2, np.sum((coords - coords[idx]) ** 2, axis=1))
    return np.array(sel, dtype=int)


def precompute_alpha(tau: float, Kmax: int = 10000, eps: float = 1e-300) -> np.ndarray:
    """
    Diffusion modulation coefficients α_k = τ^k / k! via stable recurrence.

    We use α with τ/2 in the symmetric g-GRF modulation.
    """
    vals = [1.0]
    k = 1
    while k < Kmax:
        nxt = vals[-1] * (tau / k)
        if nxt < eps:
            break
        vals.append(nxt)
        k += 1
    return np.array(vals, dtype=float)


# ============================================================
# g-GRF signature walks on a kNN graph (Algorithm 1)
# ============================================================

def build_signatures_from_coords(
    X: np.ndarray,
    k: int,
    num_starts: int,
    start_mode: str,
    n_walks: Optional[int],
    phalt: float,
    tau: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Build diffusion-modulated g-GRF signatures φ_t(x_j) from arbitrary coordinates.

    Parameters mirror the original `build_ggrf_signatures_from_coords` in manifold.py.
    """
    rng = np.random.default_rng(seed)
    W_raw, Wf, neighbors, deg_unw, sigma2 = build_knn_graph_euclidean(X, k=k)

    N = X.shape[0]
    if start_mode == "farthest":
        start_indices = farthest_point_sample(X, num_starts, seed=seed)
    elif start_mode == "random":
        start_indices = rng.choice(N, size=num_starts, replace=False)
    else:
        raise ValueError("start_mode must be 'farthest' or 'random'")

    alpha = precompute_alpha(0.5 * tau, Kmax=10000, eps=1e-300)

    def modulation(step: int) -> float:
        return float(alpha[step]) if step < alpha.shape[0] else 0.0

    if n_walks is None:
        n_walks = N

    phi = np.zeros((num_starts, N), dtype=float)

    def sample_feature(start_idx: int) -> np.ndarray:
        vec = np.zeros(N, dtype=float)
        local_rng = np.random.default_rng(346511053)
        for _ in range(n_walks):
            u = int(start_idx)
            load = 1.0
            step = 0
            while True:
                vec[u] += load * modulation(step)
                step += 1
                nbrs = neighbors[u]
                if nbrs.size == 0:
                    break
                v = int(nbrs[local_rng.integers(0, nbrs.size)])
                # uniform-by-count neighbor sampling correction + geometric continuation correction
                load *= (deg_unw[u] / (1.0 - phalt)) * float(Wf[u, v])
                u = v
                # terminate after the transition with prob phalt
                if local_rng.random() < phalt:
                    break
        vec /= float(n_walks)
        return vec

    for j, s in tqdm(enumerate(start_indices), total=num_starts, desc="Building g-GRF signatures"):
        phi[j] = sample_feature(int(s))

    return {
        "X": X,
        "start_indices": start_indices,
        "starts_xyz": X[start_indices],
        "phi": phi,
        "params": {
            "N": int(N),
            "k": int(k),
            "num_starts": int(num_starts),
            "start_mode": start_mode,
            "n_walks": int(n_walks),
            "phalt": float(phalt),
            "tau": float(tau),
            "sigma2": float(sigma2),
        },
        "Wf": Wf,
        "neighbors": neighbors,
    }


# ============================================================
# Supervision: (x_j, ω_{j,i}) -> y_{j,i} = φ_t(x_j)[ω_{j,i}]
# ============================================================

def make_node_pair_supervision(
    out: Dict[str, Any],
    mode: str = "all",
    M_per_start: Optional[int] = None,
    seed: int = 123,
    truncate: bool = True,
) -> Dict[str, Any]:
    """
    Build arrays: starts (M,), omegas (M,), targets (M,) and geometry arrays.

    If mode == "random" and M_per_start is set, uniformly subsample that many ω per start x_j.
    If truncate is True, downsample small φ-values the same way as in manifold.py.
    """
    X = out["X"]
    start_ids = out["start_indices"]
    phi = out["phi"]
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    starts: List[int] = []
    omegas: List[int] = []
    targets: List[float] = []

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

    starts_arr = np.array(starts, dtype=np.int64)
    omegas_arr = np.array(omegas, dtype=np.int64)
    targets_arr = np.array(targets, dtype=np.float32)
    return {
        "starts": starts_arr,
        "omegas": omegas_arr,
        "targets": targets_arr,
        "X": X,
        "N": N,
    }


# ============================================================
# PyTorch model and datasets
# ============================================================

class RelativeErrorLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(torch.abs(target), min=self.eps)
        rel_err = torch.abs(pred - target) / denom
        return rel_err.mean()


class SignatureAtNodeRegressor(nn.Module):
    """
    f(x, ω): default uses ONLY coords + geodesic (continuous RF).
    If use_ids=False, inputs = [start_xyz, omega_xyz, geod].
    If use_ids=True, inputs additionally include learned ID embeddings for start & omega.
    """
    def __init__(
        self,
        num_nodes: int,
        x_coords: np.ndarray,
        d_emb: int = 64,
        hidden: int = 128,
        use_continuous_only: bool = True,
    ):
        super().__init__()
        self.register_buffer("Xcoords", torch.tensor(x_coords, dtype=torch.float32))  # (N,3)
        self.use_cont = use_continuous_only
        self.use_ids = not use_continuous_only
        if self.use_ids:
            self.start_emb = nn.Embedding(num_nodes, d_emb)
            self.omega_emb = nn.Embedding(num_nodes, d_emb)

        # inputs: geod (1), start_xyz (3), omega_xyz (3) and optional 2*d_emb IDs
        in_dim = 1 + 3 + 3
        if self.use_ids:
            in_dim += 2 * d_emb

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = batch["start"]
        o = batch["omega"]
        geod = batch["geod"]  # (B,1)
        pieces = [
            batch["start_xyz"],
            batch["omega_xyz"],
            geod,
        ]
        if self.use_ids:
            s_emb = self.start_emb(s)
            o_emb = self.omega_emb(o)
            pieces = [s_emb, o_emb] + pieces
        x = torch.cat(pieces, dim=-1)
        return self.mlp(x).squeeze(-1)


class NodePairDataset(Dataset):
    """
    Basic dataset that recomputes geodesic from coords on the fly.

    Used mainly for sanity checks; for training we usually use NodePairDatasetWithGeod
    with a precomputed geodesic matrix.
    """
    def __init__(self, sup: Dict[str, Any], use_continuous_only: bool = True):
        self.starts = sup["starts"]      # (M,)
        self.omegas = sup["omegas"]      # (M,)
        self.targets = sup["targets"]    # (M,)
        self.X = sup["X"]                # (N,3)
        self.use_cont = use_continuous_only

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = int(self.starts[idx])
        o = int(self.omegas[idx])
        start_xyz = self.X[s].astype(np.float32)
        omega_xyz = self.X[o].astype(np.float32)
        dot = float(np.clip(np.dot(start_xyz, omega_xyz), -1.0, 1.0))
        geod = float(np.arccos(dot))
        return {
            "start": s,
            "omega": o,
            "start_xyz": start_xyz,
            "omega_xyz": omega_xyz,
            "geod": geod,
            "target": float(self.targets[idx]),
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
        "omega": torch.tensor([b["omega"] for b in batch], dtype=torch.long),
        "start_xyz": torch.tensor([b["start_xyz"] for b in batch], dtype=torch.float32),
        "omega_xyz": torch.tensor([b["omega_xyz"] for b in batch], dtype=torch.float32),
        "geod": torch.tensor([b["geod"] for b in batch], dtype=torch.float32).unsqueeze(-1),
        "target": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
    }


def split_by_start(starts_all: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    """
    Split indices so that validation contains entire start nodes, not just individual pairs.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(starts_all)
    rng.shuffle(unique)
    m_val = max(1, int(round(len(unique) * val_frac)))
    val_starts = np.sort(unique[:m_val])
    train_starts = np.sort(unique[m_val:])
    train_mask = np.isin(starts_all, train_starts)
    val_mask = np.isin(starts_all, val_starts)
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    return train_idx, val_idx, train_starts, val_starts


class NodePairDatasetWithGeod(NodePairDataset):
    """
    Dataset that uses a precomputed geodesic matrix.

    geod_matrix can be either full (N, N) or (num_starts, N) plus a row indexer.
    """
    def __init__(self, sup: Dict[str, Any], geod_matrix: np.ndarray, use_continuous_only: bool = True):
        super().__init__(sup, use_continuous_only=use_continuous_only)
        self.G = geod_matrix
        self.map_from_start_to_row: Optional[Dict[int, int]] = None

    def set_row_indexer(self, start_indices: np.ndarray) -> None:
        pos = {int(s): i for i, s in enumerate(start_indices)}
        self.map_from_start_to_row = pos

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = super().__getitem__(idx)
        s = int(ex["start"])
        o = int(ex["omega"])
        if self.map_from_start_to_row is None:
            geod_val = float(self.G[s, o])
        else:
            geod_val = float(self.G[self.map_from_start_to_row[s], o])
        ex["geod"] = geod_val
        return ex


def train_model_node_pairs_by_start(
    sup: Dict[str, Any],
    geod_matrix: np.ndarray,
    batch_size: int = 1024,
    epochs: int = 8,
    lr: float = 1e-3,
    d_emb: int = 64,
    hidden: int = 128,
    use_continuous_only: bool = True,
    val_frac: float = 0.2,
    seed: int = 0,
    val_start_ids: Optional[np.ndarray] = None,
):
    """
    Train SignatureAtNodeRegressor on node-pair supervision using a start-wise split.

    This is essentially the same routine as in manifold.py but packaged as a function.
    """
    ds = NodePairDatasetWithGeod(sup, geod_matrix, use_continuous_only=use_continuous_only)

    if val_start_ids is None:
        tr_idx, va_idx, tr_starts, va_starts = split_by_start(sup["starts"], val_frac=val_frac, seed=seed)
    else:
        all_starts = np.unique(sup["starts"])
        tr_starts = np.setdiff1d(all_starts, val_start_ids)
        va_starts = np.array(sorted(np.unique(val_start_ids)))
        tr_idx = np.nonzero(np.isin(sup["starts"], tr_starts))[0]
        va_idx = np.nonzero(np.isin(sup["starts"], va_starts))[0]

    class Subset(Dataset):
        def __init__(self, base: Dataset, idxs: np.ndarray):
            self.base = base
            self.idxs = idxs

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, i: int) -> Dict[str, Any]:
            return self.base[self.idxs[i]]

    tr_ds = Subset(ds, tr_idx)
    va_ds = Subset(ds, va_idx)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

    torch.manual_seed(seed)
    model = SignatureAtNodeRegressor(
        num_nodes=sup["N"],
        x_coords=sup["X"],
        d_emb=d_emb,
        hidden=hidden,
        use_continuous_only=use_continuous_only,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = RelativeErrorLoss(eps=1e-1)
    # loss_fn = nn.MSELoss()

    def eval_loader(dl: DataLoader) -> float:
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                pred = model(batch)
                y = batch["target"]
                se += torch.sum((pred - y) ** 2).item()
                n += y.numel()
        return math.sqrt(se / n)

    def absolute_relative_error(dl: DataLoader, eps: float = 1e-3) -> float:
        model.eval()
        re_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                pred = model(batch)
                y = batch["target"]
                denom = torch.clamp(torch.abs(y), min=eps)
                rel_err = torch.abs(pred - y) / denom
                re_sum += rel_err.sum().item()
                n += y.numel()
        return re_sum / n if n > 0 else float("nan")

    hist: Dict[str, List[float]] = {
        "rmse_train": [],
        "rmse_val": [],
        "relerr_train": [],
        "relerr_val": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        for batch in tr_loader:
            pred = model(batch)
            y = batch["target"]
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        rmse_tr = eval_loader(tr_loader)
        rmse_va = eval_loader(va_loader)
        rel_tr = absolute_relative_error(tr_loader)
        rel_va = absolute_relative_error(va_loader)

        hist["rmse_train"].append(rmse_tr)
        hist["rmse_val"].append(rmse_va)
        hist["relerr_train"].append(rel_tr)
        hist["relerr_val"].append(rel_va)

        print(
            f"Epoch {ep:02d} | "
            f"RMSE train: {rmse_tr:.6f} | RMSE val: {rmse_va:.6f} | "
            f"RelErr train: {rel_tr:.6f} | RelErr val: {rel_va:.6f}"
        )

    return model, hist, {
        "train_starts": tr_starts,
        "val_starts": va_starts,
        "train_idx": tr_idx,
        "val_idx": va_idx,
    }


def collect_val_predictions(
    model: nn.Module,
    sup: Dict[str, Any],
    geod_matrix: np.ndarray,
    val_starts: np.ndarray,
    batch_size: int = 1024,
    use_continuous_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on validation subset (specified by start IDs) and return (y_true, y_pred).
    """
    ds = NodePairDatasetWithGeod(sup, geod_matrix, use_continuous_only=use_continuous_only)

    class Subset(Dataset):
        def __init__(self, base: Dataset, idxs: np.ndarray):
            self.base = base
            self.idxs = idxs

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, i: int) -> Dict[str, Any]:
            return self.base[self.idxs[i]]

    val_idx = np.where(np.isin(sup["starts"], val_starts))[0]
    va_ds = Subset(ds, val_idx)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in va_loader:
            y_hat = model(batch).reshape(-1)
            y = batch["target"].reshape(-1)
            preds.append(y_hat.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    return targets_arr, preds_arr


# ============================================================
# Geodesics + ground-truth heat kernel on S^2
# ============================================================

def geodesic_matrix_sphere(X: np.ndarray) -> np.ndarray:
    """Exact spherical geodesic on S^2 via arccos of dot product."""
    dots = np.clip(X @ X.T, -1.0, 1.0)
    return np.arccos(dots)


import heapq


def dijkstra_single_source(X: np.ndarray, neighbors: List[np.ndarray], s: int) -> np.ndarray:
    """Geodesic (graph-shortest paths) from source s using Euclidean edge lengths."""
    N = X.shape[0]
    dist = np.full(N, np.inf, dtype=float)
    dist[s] = 0.0
    h: List[Tuple[float, int]] = [(0.0, s)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        Xu = X[u]
        for v in neighbors[u]:
            v = int(v)
            # Euclidean length of edge (u, v)
            w = float(np.linalg.norm(Xu - X[v]))
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist


def geodesic_matrix_graph(X: np.ndarray, neighbors: List[np.ndarray], sources: np.ndarray) -> np.ndarray:
    """Compute geodesic distances (rows only for given sources) via Dijkstra on kNN graph."""
    N = X.shape[0]
    G = np.zeros((len(sources), N), dtype=float)
    for i, s in tqdm(enumerate(sources), total=len(sources), desc="Computing graph geodesics"):
        G[i] = dijkstra_single_source(X, neighbors, int(s))
    return G


def ground_truth_heat_kernel(X: np.ndarray, t: float, L_max: int) -> np.ndarray:
    """
    Ground-truth heat kernel on S^2 via Legendre expansion:

        K_t(x, y) = sum_{l=0}^∞ (2l+1)/(4π) e^{-l(l+1)t} P_l(<x,y>)
    """
    N = X.shape[0]
    cosT = np.clip(X @ X.T, -1.0, 1.0)
    K = np.zeros((N, N), dtype=np.float64)
    for l in tqdm(range(L_max + 1), desc="Computing ground truth heat kernel"):
        P_l = eval_legendre(l, cosT)
        coeff = (2 * l + 1) / (4.0 * np.pi) * np.exp(-l * (l + 1) * t)
        K += coeff * P_l
    return K


def compute_nn_feature_matrix_on_sphere(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Compute g_nn(x_i, ω_j) for all x_i, ω_j on the discretized sphere grid X.
    Returns G of shape (N, N) with G[i, j] ≈ φ_t(x_i)[ω_j].
    """
    N = X.shape[0]
    device = next(model.parameters()).device

    omega_xyz = torch.tensor(X, dtype=torch.float32, device=device)    # (N,3)
    omega_ids = torch.arange(N, dtype=torch.long, device=device)       # (N,)

    G = np.zeros((N, N), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in tqdm(range(N), desc="NN prediction on all nodes"):
            start_xyz = X[s]
            dots = np.clip(X @ start_xyz, -1.0, 1.0)
            geod = np.arccos(dots).astype(np.float32)

            batch = {
                "start": torch.full((N,), s, dtype=torch.long, device=device),
                "omega": omega_ids,
                "start_xyz": torch.tensor(
                    np.repeat(start_xyz[None, :], N, axis=0),
                    dtype=torch.float32,
                    device=device,
                ),
                "omega_xyz": omega_xyz,
                "geod": torch.tensor(geod[:, None], dtype=torch.float32, device=device),
            }
            G[s] = model(batch).detach().cpu().numpy()

    return G


def compute_nn_kernel_and_errors(
    model: nn.Module,
    X: np.ndarray,
    t: float,
    L_max: int,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build NN-based kernel K_nn_all and compare to ground-truth heat kernel.

    Returns (K_nn_all, K_true, G_nn, mse, abs_rel_err).
    """
    G_nn = compute_nn_feature_matrix_on_sphere(model, X)     # (N, N)
    K_nn = G_nn @ G_nn.T                                     # (N, N)

    K_true = ground_truth_heat_kernel(X, t=t, L_max=L_max)   # (N, N)

    # --- Frobenius-norm scaling: match "energy" of K_nn to K_true ---
    fro_true = np.linalg.norm(K_true, ord="fro")
    fro_nn   = np.linalg.norm(K_nn,   ord="fro")

    if fro_nn > 0:
        scale = fro_true / fro_nn
        K_nn_scaled = K_nn * scale
    else:
        scale = 1.0
        K_nn_scaled = K_nn  # degenerate case; shouldn't really happen

    print(f"[Sphere] Frobenius norms: "
        f"||K_true||_F={fro_true:.6e}, ||K_nn||_F={fro_nn:.6e}, "
        f"scale={scale:.6e}")

    # use the *scaled* kernel from here on
    diff = K_nn_scaled - K_true
    mse = float(np.mean(diff**2))
    abs_rel = float(np.mean(np.abs(diff) / (np.abs(K_true) + eps)))

    print(f"[Sphere] raw MSE (Frob‑scaled)    = {mse:.6e}")
    print(f"[Sphere] raw absRel (Frob‑scaled) = {abs_rel:.6e}")

    return K_nn_scaled, K_true, G_nn, mse, abs_rel


# ============================================================
# Helpers for ellipsoid heat kernel & NN kernel
# ============================================================

def heat_kernel_from_graph_dense(
    W: np.ndarray,
    t: float,
    normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a graph Laplacian from a dense adjacency W and compute
    the heat kernel e^{-t L} via full eigen-decomposition.

    W: (N,N) dense adjacency
    t: diffusion time
    normalized:
      True  -> L_sym = I - D^{-1/2} W D^{-1/2}
      False -> L = D - W

    Returns:
      K_t : (N,N) dense heat kernel
      w   : eigenvalues
      V   : eigenvectors (columns)
    """
    N = W.shape[0]
    deg = W.sum(axis=1)
    if normalized:
        d_inv_sqrt = np.power(deg + 1e-12, -0.5)
        Wn = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
        L = np.eye(N) - Wn
    else:
        L = np.diag(deg) - W

    w, V = np.linalg.eigh(L)
    expw = np.exp(-t * w)
    K_t = (V * expw) @ V.T
    return K_t, w, V


def compute_nn_feature_matrix_on_manifold(
    model: nn.Module,
    X: np.ndarray,
    geod_matrix: np.ndarray,
) -> np.ndarray:
    """
    General version of compute_nn_feature_matrix_on_sphere:

    Uses a precomputed geodesic matrix geod_matrix[i,j] = d(x_i, x_j),
    matching the feature used in training on the manifold.

    Returns:
      G[i, j] ≈ φ_t(x_i)[ω_j].
    """
    N = X.shape[0]
    device = next(model.parameters()).device

    omega_xyz = torch.tensor(X, dtype=torch.float32, device=device)  # (N,3)
    omega_ids = torch.arange(N, dtype=torch.long, device=device)     # (N,)

    G = np.zeros((N, N), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s in tqdm(range(N), desc="NN prediction on manifold"):
            start_xyz = X[s]
            geod_row = geod_matrix[s].astype(np.float32)  # (N,)

            batch = {
                "start": torch.full((N,), s, dtype=torch.long, device=device),
                "omega": omega_ids,
                "start_xyz": torch.tensor(
                    np.repeat(start_xyz[None, :], N, axis=0),
                    dtype=torch.float32,
                    device=device,
                ),
                "omega_xyz": omega_xyz,
                "geod": torch.tensor(geod_row[:, None], dtype=torch.float32, device=device),
            }
            G[s] = model(batch).detach().cpu().numpy()

    return G


def frobenius_mse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean((A - B) ** 2))


def frobenius_rel_error(A: np.ndarray, B: np.ndarray) -> float:
    num = np.linalg.norm(A - B, ord="fro")
    den = np.linalg.norm(B, ord="fro") + 1e-16
    return float(num / den)


# ============================================================
# Manifold classes
# ============================================================

class Manifold:
    """
    Abstract base class tying together:
      - discretization (coordinates),
      - g-GRF signatures,
      - dataset creation,
      - NN training,
      - (optionally) a ground-truth kernel.
    """
    def __init__(self, cfg: ManifoldConfig):
        self.cfg = cfg
        self.X: Optional[np.ndarray] = None
        self.start_indices: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None
        self.Wf: Optional[np.ndarray] = None
        self.neighbors: Optional[List[np.ndarray]] = None

        self.sup: Optional[Dict[str, Any]] = None
        self.model: Optional[nn.Module] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self.splits: Optional[Dict[str, Any]] = None

    # ---- core steps -----------------------------------------------------

    def discretization(self) -> np.ndarray:
        """Return an (N, d) array of coordinates for the discretized manifold."""
        raise NotImplementedError

    def build_signature_walks(self) -> Dict[str, Any]:
        """
        Discretize the manifold and build g-GRF signatures φ_t(x_j).

        Stores X, start_indices, phi, Wf, neighbors on the instance.
        """      
        X = self.discretization()
        out = build_signatures_from_coords(
            X=X,
            k=self.cfg.k,
            num_starts=self.cfg.num_starts,
            start_mode=self.cfg.start_mode,
            n_walks=self.cfg.n_walks,
            phalt=self.cfg.phalt,
            tau=self.cfg.t,
            seed=self.cfg.seed,
        )
        self.X = out["X"]
        self.start_indices = out["start_indices"]
        self.phi = out["phi"]
        self.Wf = out["Wf"] # not needed for sphere
        self.neighbors = out["neighbors"]
        return out

    def make_dataset(
        self,
        mode: str = "random",
        truncate: bool = True,
        seed: int = 123,
    ) -> Dict[str, Any]:
        """
        Build supervised node-pair dataset (x_j, ω_{j,i}) -> φ_t(x_j)[ω_{j,i}].

        Uses the signatures stored on the instance (run build_signature_walks first).
        """
        if self.X is None or self.phi is None or self.start_indices is None:
            raise RuntimeError("Call build_signature_walks() before make_dataset().")

        out = {
            "X": self.X,
            "phi": self.phi,
            "start_indices": self.start_indices,
            "params": {"N": self.X.shape[0]},
            "Wf": self.Wf,
            "neighbors": self.neighbors,
        }

        sup = make_node_pair_supervision(
            out,
            mode=("all" if self.cfg.subsample_per_start is None else mode),
            M_per_start=self.cfg.subsample_per_start,
            seed=seed,
            truncate=truncate,
        )
        sup["N"] = self.X.shape[0]
        self.sup = sup
        return sup

    # ---- geodesics & training ------------------------------------------

    def _geodesic_matrix_for_training(self) -> np.ndarray:
        """
        Default geodesic: graph-shortest paths from start nodes (Dijkstra on kNN graph),
        stored in an N x N matrix with only rows for start nodes actually used.
        """
        if self.X is None or self.neighbors is None or self.start_indices is None:
            raise RuntimeError("Need signatures (and neighbors) before computing geodesics.")

        G_rows = geodesic_matrix_graph(self.X, self.neighbors, self.start_indices)
        N = self.X.shape[0]
        G_full = np.zeros((N, N), dtype=float)
        for row, s in enumerate(self.start_indices):
            G_full[int(s)] = G_rows[row]
        return G_full

    def train_model(self, val_start_ids: Optional[np.ndarray] = None):
        """
        Train SignatureAtNodeRegressor on the current dataset, using split_by_start.
        """
        if self.sup is None:
            self.make_dataset()

        geod_matrix = self._geodesic_matrix_for_training()

        model, hist, splits = train_model_node_pairs_by_start(
            sup=self.sup,
            geod_matrix=geod_matrix,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            lr=self.cfg.lr,
            d_emb=self.cfg.d_emb,
            hidden=self.cfg.hidden,
            use_continuous_only=self.cfg.use_continuous_only,
            val_frac=self.cfg.val_frac,
            seed=self.cfg.torch_seed,
            val_start_ids=val_start_ids,
        )

        self.model = model
        self.history = hist
        self.splits = splits
        return model, hist, splits

    # ---- evaluation & persistence ---------------------------------------
    def collect_val_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model on validation subset and return (y_true, y_pred).
        """
        if self.sup is None or self.model is None or self.splits is None:
            raise RuntimeError("Need dataset, trained model, and splits before collecting val predictions.")

        geod_matrix = self._geodesic_matrix_for_training()

        y_true, y_pred = collect_val_predictions(
            model=self.model,
            sup=self.sup,
            geod_matrix=geod_matrix,
            val_starts=self.splits["val_starts"],
            batch_size=self.cfg.batch_size,
            use_continuous_only=self.cfg.use_continuous_only,
        )
        return y_true, y_pred
    
    def compute_ground_truth(self, *args, **kwargs):
        """
        Optionally overridden in subclasses when a closed-form kernel is available.
        """
        raise NotImplementedError("Ground-truth kernel not implemented for this manifold.")

    def save_pickle(self, path: str) -> None:
        """Save the manifold instance (including trained model) to a file."""
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except FileNotFoundError as e:
            raise RuntimeError(f"Could not save to path: {path}") from e
    
    def load_pickle(self, path: str) -> None:
        """Load the manifold instance (including trained model) from a file."""
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
        except FileNotFoundError as e:
            raise RuntimeError(f"Could not load from path: {path}") from e


class Ellipsoid(Manifold):
    """
    Ellipsoid embedded in R^3: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1.

    Discretization is obtained by scaling a Fibonacci sphere.
    Geodesics default to graph geodesics on the kNN graph.
    """
    def __init__(self, cfg: ManifoldConfig, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        super().__init__(cfg)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def discretization(self) -> np.ndarray:
        return fibonacci_sphere_scaled(self.cfg.N, self.a, self.b, self.c)

    def compute_ground_truth(
        self,
        k_lap: Optional[int] = None,
        normalized_lap: bool = True,
        eps: float = 1e-3,
        t: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Approximate 'ground-truth' diffusion heat kernel on the ellipsoid via
        the graph Laplacian and compare it against the NN-induced kernel.

        Returns:
        K_nn    : (N,N) NN-based kernel (rescaled to best match K_true)
        K_true  : (N,N) Laplacian heat kernel
        G_nn    : (N,N) *rescaled* NN feature matrix
        mse     : MSE(K_nn, K_true)
        abs_rel : mean_{i,j} |K_nn - K_true| / (|K_true| + eps)
        """
        if self.model is None or self.X is None:
            raise RuntimeError("Need trained model and discretization before computing ground truth.")
        if t is None:
            t = self.cfg.t

        N = self.X.shape[0]

        # 1) Full graph geodesics for **all** nodes
        #    (base helper only fills rows for start_indices)
        all_sources = np.arange(N, dtype=int)
        geod_matrix_full = geodesic_matrix_graph(self.X, self.neighbors, all_sources)

        # 2) NN feature matrix G_nn[i, j] ≈ g(x_i, ω_j)
        G_nn = compute_nn_feature_matrix_on_manifold(self.model, self.X, geod_matrix_full)

        # 3) Build graph adjacency W on ellipsoid for Laplacian
        if k_lap is None:
            k_lap = self.cfg.k
        W_raw, _, _, _, _ = build_knn_graph_euclidean(self.X, k=k_lap)

        # 4) Spectral heat kernel from dense Laplacian
        K_true, w, V = heat_kernel_from_graph_dense(
            W_raw,
            t=t,
            normalized=normalized_lap,
        )

        # 5) Rescale NN feature matrix so that K_nn best matches K_true
        #    in Frobenius norm:  min_α || α G G^T - K_true ||_F^2
        K_nn_raw = G_nn @ G_nn.T

        num = float(np.sum(K_true * K_nn_raw))
        den = float(np.sum(K_nn_raw * K_nn_raw) + 1e-16)
        alpha = max(num / den, 0.0)          # optimal scalar on K_nn_raw
        beta = np.sqrt(alpha)                # scalar on G_nn

        G_nn_scaled = G_nn * beta
        K_nn = G_nn_scaled @ G_nn_scaled.T

        print(f"[Ellipsoid] kernel rescale alpha={alpha:.3e}, beta={beta:.3e}")
        print(f"[Ellipsoid] diag(K_true) in [{K_true.diagonal().min():.3e}, {K_true.diagonal().max():.3e}]")
        print(f"[Ellipsoid] diag(K_nn)   in [{K_nn.diagonal().min():.3e}, {K_nn.diagonal().max():.3e}]")

        # 6) Errors
        diff = K_nn - K_true
        mse = float(np.mean(diff ** 2))
        abs_rel = float(np.mean(np.abs(diff) / (np.abs(K_true) + eps)))

        print(f"[Ellipsoid] MSE(K_nn, K_true)    = {mse:.6e}")
        print(f"[Ellipsoid] mean abs rel. error = {abs_rel:.6e}")

        return K_nn, K_true, G_nn_scaled, mse, abs_rel


class Sphere(Ellipsoid):
    """
    Sphere S^2 as a special case of Ellipsoid with a=b=c=1.

    Uses Fibonacci sphere discretization and analytic spherical geodesic
    for the training feature.
    """
    def __init__(self, cfg: ManifoldConfig):
        super().__init__(cfg, a=1.0, b=1.0, c=1.0)

    def _geodesic_matrix_for_training(self) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("Need discretization (build_signature_walks) before geodesics.")
        return geodesic_matrix_sphere(self.X)

    def compute_ground_truth(self, L_max: int, t: Optional[float] = None):
        """
        Compute NN-based kernel from the trained model and compare against
        the ground-truth heat kernel on S^2.

        Returns (K_nn_all, K_true, G_nn, mse, abs_rel_err).
        """
        if self.model is None or self.X is None:
            raise RuntimeError("Need trained model and discretization before computing ground truth.")
        if not t:
            t = self.cfg.t
        return compute_nn_kernel_and_errors(
            self.model,
            self.X,
            t=t,
            L_max=L_max,
        )
    

class Torus(Manifold):
    """
    Standard torus embedded in R^3 with major radius R and minor radius r.

    Discretization is obtained by uniform grid sampling in (θ, φ).
    Geodesics default to graph geodesics on the kNN graph.
    """
    def __init__(self, cfg: ManifoldConfig, R: float = 2.0, r: float = 1.0):
        super().__init__(cfg)
        self.R = float(R)
        self.r = float(r)

    def discretization(self) -> np.ndarray:
        return torus_grid(self.cfg.N, self.R, self.r)
    # geodesics: use base implementation (graph geodesics)


class SwissRoll(Manifold):
    """
    Swiss roll manifold embedded in R^3.

    Discretization is obtained by standard swiss roll sampling.
    Geodesics default to graph geodesics on the kNN graph.
    """
    def __init__(self, cfg: ManifoldConfig, noise: float = 0.0):
        super().__init__(cfg)
        self.noise = float(noise)

    def discretization(self) -> np.ndarray:
        return swiss_roll(self.cfg.N, noise=self.noise)
    # geodesics: use base implementation (graph geodesics)


class MobiusStrip(Manifold):
    """
    Möbius strip embedded in R^3.

    Discretization is obtained by uniform grid sampling in (u, v).
    Geodesics default to graph geodesics on the kNN graph.
    """
    def __init__(self, cfg: ManifoldConfig, width: float = 1.0):
        super().__init__(cfg)
        self.width = float(width)

    def discretization(self) -> np.ndarray:
        return mobius_strip_grid(self.cfg.N, self.width)
    # geodesics: use base implementation (graph geodesics)