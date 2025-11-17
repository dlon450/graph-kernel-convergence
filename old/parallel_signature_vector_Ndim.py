import numpy as np
import matplotlib.pyplot as plt
import math
import os
from itertools import product, repeat
from concurrent.futures import ProcessPoolExecutor

# -----------------------------------
# Utilities for D-dimensional torus
# -----------------------------------

def neighbors(coord, n):
    """
    Return the 2D neighbors of a D-dimensional lattice point on an n-periodic torus.
    coord: tuple of length D, each in [0, n-1]
    """
    D = len(coord)
    nbrs = []
    for axis in range(D):
        c_minus = list(coord)
        c_plus  = list(coord)
        c_minus[axis] = (c_minus[axis] - 1) % n
        c_plus[axis]  = (c_plus[axis]  + 1) % n
        nbrs.append(tuple(c_minus))
        nbrs.append(tuple(c_plus))
    return nbrs  # length = 2D

def coord_to_index(coord, n):
    """Row-major base-n index for a D-tuple coord."""
    idx = 0
    for a in coord:
        idx = idx * n + a
    return idx

def index_to_coord(idx, n, D):
    """Inverse of coord_to_index."""
    out = [0]*D
    for k in range(D-1, -1, -1):
        out[k] = idx % n
        idx //= n
    return tuple(out)

def all_coords(n, D):
    """Iterator over all D-dim coordinates on {0,...,n-1}^D."""
    return product(range(n), repeat=D)

# -----------------------------------
# Core: random-walk signature (D-dim)
# -----------------------------------

def generate_signature_vector_diffusion_sym(n=10, D=2, source=None,
                                            num_walks=400, p_term=0.2,
                                            gamma=1.0, sigma=1.0, seed=0):
    """
    Generate the signature vector φ(source) on an n^D torus.
    Unbiased for sum_k (s^k/k!) * U^k e_source with s = sigma^2 / 4 and
    U[i,j] = gamma / sqrt(deg(i)deg(j)) on edges (deg = 2D here).
    """
    rng = np.random.default_rng(seed)
    N = n**D
    v = np.zeros(N, dtype=np.float64)

    if source is None:
        source = (0,)*D
    rr = tuple(source)

    # Poisson-like factorial weight: s^k/k! with s chosen so that
    # exp(-0.5*sigma^2) * (ΦΦ^T) matches exp(-0.5*sigma^2(I-U)).
    s = (sigma*sigma)/4.0

    # Degree on D-dim torus:
    deg = 2*D

    for _ in range(num_walks):
        r = rr
        load = 1.0        # carries U^k in expectation
        fk   = 1.0        # carries s^k/k!
        wsurv = 1.0       # carries (1-p_term)^{-k}
        k = 0

        while True:
            # contribute k-th term: (s^k/k!) * U^k with geometric de-bias
            v[coord_to_index(r, n)] += (load * fk * wsurv)

            # stop BEFORE attempting (k+1)-th term
            if rng.random() < p_term:
                break

            # survived: update survival weight
            wsurv *= 1.0 / (1.0 - p_term)

            # one unbiased random-walk step on the D-torus
            nbrs = neighbors(r, n); dv = len(nbrs)
            wr = nbrs[rng.integers(dv)]
            dw = len(neighbors(wr, n))  # = 2D, but compute for generality

            p  = 1.0 / dv
            u  = gamma / math.sqrt(dv * dw)  # ensures E[u/p] = gamma
            load *= (u / p)
            r = wr

            # factorial update: s^{k+1}/(k+1)!
            k  += 1
            fk *= s / k

    return v / num_walks  # shape (N,)

# -----------------------------------
# Exact diffusion kernel (D-dim)
# -----------------------------------

def exact_diffusion_kernel(n, D=2, sigma=1.0, gamma=1.0):
    """
    K = exp(-0.5 * sigma^2 * (I - U)), where U[i,j] = gamma/(2D) if neighbors else 0.
    """
    N = n**D
    U = np.zeros((N, N), dtype=np.float64)
    w = gamma / (2.0 * D)

    for coord in all_coords(n, D):
        i = coord_to_index(coord, n)
        for nbr in neighbors(coord, n):
            j = coord_to_index(nbr, n)
            U[i, j] = w

    L = np.eye(N) - U
    evals, Q = np.linalg.eigh(L)
    return Q @ np.diag(np.exp(-0.5*(sigma**2)*evals)) @ Q.T

# -----------------------------------
# Monte Carlo kernel from signatures (D-dim)
# -----------------------------------

def kernel_from_signatures(n, D=2, num_walks=400, p_term=0.2,
                           sigma=1.0, gamma=1.0, seed=0):
    """
    Build Φ whose s-th row is φ(source=s), then return exp(-0.5σ^2) * ΦΦ^T.
    """
    N = n**D
    Phi = np.zeros((N, N), dtype=np.float64)
    s_idx = 0
    for coord in all_coords(n, D):
        Phi[s_idx, :] = generate_signature_vector_diffusion_sym(
            n, D, coord, num_walks, p_term, gamma, sigma, seed + s_idx
        )
        s_idx += 1
    return np.exp(-0.5*sigma*sigma) * (Phi @ Phi.T)

# -----------------------------------
# RBF baselines on grids/torus (D-dim)
# -----------------------------------

def rbf_kernel_grid(n, D=2, length=1.0):
    """
    Euclidean-grid RBF (no wrap); nodes are integer D-tuples in [0,n-1]^D.
    """
    X = np.array(list(all_coords(n, D)), dtype=np.int32)  # (N, D)
    # Pairwise squared distances
    # (N,1,D) - (1,N,D) => (N,N,D) then sum over D
    diff = X[:, None, :] - X[None, :, :]
    D2 = np.sum(diff.astype(np.float64)**2, axis=2)
    return np.exp(-D2 / (2.0 * length * length))

def rbf_kernel_torus(n, D=2, length=1.0):
    """
    RBF on the D-torus (wrap-around per dimension).
    """
    X = np.array(list(all_coords(n, D)), dtype=np.int32)  # (N, D)
    N = X.shape[0]
    # For each dimension, wrap the coordinate differences
    D2 = np.zeros((N, N), dtype=np.float64)
    for d in range(D):
        diff = np.abs(X[:, None, d] - X[None, :, d])
        diff = np.minimum(diff, n - diff)
        D2 += diff * diff
    return np.exp(-D2 / (2.0 * length * length))

# -----------------------------------
# Metrics
# -----------------------------------

def mse(A, B):
    # MSE between normalized A and B ([0,1])
    return np.mean(((A / A.max()) - (B / B.max()))**2)

# -----------------------------------
# helpers for parallelizing replications
# -----------------------------------

def _khat_for_rep(n, D, num_walks, p_term, sigma, gamma, seed):
    # top-level so it's picklable by spawn
    return kernel_from_signatures(n, D, num_walks, p_term, sigma, gamma, seed)

# -----------------------------------
# Parallelized run_curve (D-dim)
# -----------------------------------

def run_curve_parallel(ns=(5,7,9,11,13,15), D=2,
                       num_walks=2000, p_term=0.1,
                       sigma=4.0, gamma=1.0,
                       seed=0, reps=100,
                       max_workers=None):
    """
    Parallelizes over 'reps' for each n and fixed dimension D.
    Computes K_true and K_rbf once per n in the parent.
    """
    max_workers = max_workers or (os.cpu_count() or 1)

    mses_rf  = [[] for _ in range(reps)]
    mses_rbf = [[] for _ in range(reps)]
    mses_true_vs_exact_kernel = [[] for _ in range(reps)]
    Ns = []

    for n in ns:
        print(f"n={n}, D={D}")
        # Precompute once per n
        K_true = exact_diffusion_kernel(n, D, sigma, gamma)
        # Use torus RBF baseline; match your 2D choice of length=sigma/2
        K_rbf  = rbf_kernel_torus(n, D=D, length=sigma/2)

        seeds = [seed + i for i in range(reps)]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for i, K_hat in enumerate(
                ex.map(_khat_for_rep,
                       repeat(n),
                       repeat(D),
                       repeat(num_walks),
                       repeat(p_term),
                       repeat(sigma),
                       repeat(gamma),
                       seeds,
                       chunksize=1)
            ):
                mses_rf[i].append(mse(K_hat, K_true))
                mses_rbf[i].append(mse(K_hat, K_rbf))
                # append once per n for this replication
                if len(mses_true_vs_exact_kernel[i]) < len(Ns) + 1:
                    mses_true_vs_exact_kernel[i].append(mse(K_true, K_rbf))

        Ns.append(n**D)

    return Ns, mses_rf, mses_rbf, mses_true_vs_exact_kernel

# -----------------------------------
# Example main (safe to run)
# -----------------------------------

if __name__ == "__main__":
    # read in .npz file if it exists
    # if os.path.exists("parallel_diffusion_kernel_approximation_Ddim.npz") and False:
    #     data = np.load("parallel_diffusion_kernel_approximation_Ddim.npz")
    #     Ns = data["Ns"]
    #     mses_rf = data["mses_rf"]
    #     mses_rbf = data["mses_rbf"]
    #     mses_true_vs_exact_kernel = data["mses_true_vs_exact_kernel"]
    # else:

    for dim in [3, 4, 5, 6]:

        Ns, mses_rf, mses_rbf, mses_true_vs_exact_kernel = run_curve_parallel(
            ns=(5,7,9), D=dim,  # try D=3, modest n to keep exact kernel feasible
            num_walks=1000, p_term=0.1,
            sigma=4.0, gamma=1.0,
            seed=0, reps=128, max_workers=16
        )

        # Save results
        np.savez(f"parallel_diffusion_kernel_approximation_{dim}dim.npz",
                Ns=Ns,
                mses_rf=mses_rf,
                mses_rbf=mses_rbf,
                mses_true_vs_exact_kernel=mses_true_vs_exact_kernel)

        # Plot shaded errorbar (g-GRF vs exact)
        plt.figure(figsize=(6,4))
        mean_rf  = np.mean(mses_rf, axis=0)
        std_rf   = np.std(mses_rf, axis=0)
        plt.errorbar(Ns, mean_rf, yerr=std_rf, marker="o", label=f"g-GRF ({dim}-dim) vs exact")
        plt.fill_between(Ns, mean_rf-std_rf, mean_rf+std_rf, alpha=0.2)
        plt.xlabel("number of nodes N"); plt.ylabel("MSE")
        plt.title(f"Diffusion kernel approximation vs exact ({dim}-dim)"); plt.legend(); plt.tight_layout()
        plt.savefig(f"diffusion_kernel_approx_vs_exact_{dim}dim.png", dpi=150)

        # RBF baseline plot
        plt.figure(figsize=(6,4))
        mean_rbf = np.mean(mses_rbf, axis=0)
        std_rbf  = np.std(mses_rbf, axis=0)
        plt.errorbar(Ns, mean_rbf, yerr=std_rbf, marker="o", label=f"RBF vs g-GRF ({dim}-dim)")
        plt.fill_between(Ns, mean_rbf-std_rbf, mean_rbf+std_rbf, alpha=0.2)
        plt.xlabel("number of nodes N"); plt.ylabel("MSE")
        plt.title(f"Gaussian kernel vs diffusion kernel approximation ({dim}-dim)"); plt.legend(); plt.tight_layout()
        plt.savefig(f"rbf_vs_diffusion_kernel_approx_{dim}dim.png", dpi=150)
