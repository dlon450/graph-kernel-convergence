import numpy as np
import matplotlib.pyplot as plt
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def neighbors(r, c, n):
    return [((r-1) % n, c % n), ((r+1) % n, c % n), (r % n, (c-1) % n), (r % n, (c+1) % n)]

# def generate_signature_vector_diffusion_sym(n=10, source=(0,0), num_walks=400, p_term=0.2, gamma=1.0, sigma=1.0, seed=0):
#     rng = np.random.default_rng(seed)
#     v = np.zeros((n,n), dtype=np.float64)
#     rr, cc = source
#     s = (sigma*sigma)/4.0
#     for _ in range(num_walks):
#         r, c = rr, cc
#         load = 1.0
#         k = 0
#         fk = 1.0
#         while True:
#             v[r,c] += load*fk
#             k += 1
#             nbrs = neighbors(r,c,n); dv = 4
#             wr, wc = nbrs[rng.integers(dv)]; dw = 4
#             p = 1.0/dv
#             u = gamma/np.sqrt(dv*dw)
#             load *= u/(p*(1.0-p_term))
#             r, c = wr, wc
#             fk *= s/k
#             if rng.random() < p_term: break
#     return (v/num_walks).reshape(-1)

def generate_signature_vector_diffusion_sym(n=10, source=(0,0), num_walks=400,
                                            p_term=0.2, gamma=1.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros((n,n), dtype=np.float64)
    rr, cc = source
    s = (sigma*sigma)/4.0  # Poisson mean for the k! weights

    for _ in range(num_walks):
        r, c = rr, cc
        load = 1.0                 # carries the Markov operator U^k in expectation
        fk   = 1.0                 # carries s^k/k!
        wsurv = 1.0                # carries (1-p_term)^{-k}
        k = 0

        while True:
            # contribute the k-th term:  (s^k/k!) * U^k  with geometric de-bias
            v[r, c] += (load * fk * wsurv)

            # decide whether we stop BEFORE attempting to produce the (k+1)-th term
            if rng.random() < p_term:
                break

            # we survived to do the (k+1)-th term, so update survival weight
            wsurv *= 1.0 / (1.0 - p_term)

            # take one random-walk step; keep the Markov expectation unbiased
            nbrs = neighbors(r, c, n); dv = len(nbrs)
            wr, wc = nbrs[rng.integers(dv)]; dw = len(neighbors(wr, wc, n))
            p  = 1.0 / dv
            u  = gamma / math.sqrt(dv * dw)  # u/p has expectation gamma
            load *= (u / p)                   # <-- remove /(1-p_term) here
            r, c = wr, wc

            # update factorial weight to get s^{k+1}/(k+1)!
            k  += 1
            fk *= s / k

    return (v / num_walks).reshape(-1)

def exact_diffusion_kernel(n, sigma=1.0, gamma=1.0):
    N = n*n
    U = np.zeros((N,N), dtype=np.float64)
    for r in range(n):
        for c in range(n):
            i = r*n + c
            for nr, nc in neighbors(r,c,n):
                j = nr*n + nc
                U[i,j] = gamma/4.0
    L = np.eye(N) - U
    w, Q = np.linalg.eigh(L)
    return Q @ np.diag(np.exp(-0.5*(sigma**2)*w)) @ Q.T

def kernel_from_signatures(n, num_walks=400, p_term=0.2, sigma=1.0, gamma=1.0, seed=0):
    N = n*n
    Phi = np.zeros((N,N), dtype=np.float64)
    s = 0
    for r in range(n):
        for c in range(n):
            Phi[s,:] = generate_signature_vector_diffusion_sym(n, (r,c), num_walks, p_term, gamma, sigma, seed+s)
            s += 1
    return np.exp(-0.5*sigma*sigma) * (Phi @ Phi.T)

def rbf_kernel_grid(n, length=1.0):
    X = np.array([(r,c) for r in range(n) for c in range(n)], dtype=np.float64)
    D2 = np.sum((X[:,None,:]-X[None,:,:])**2, axis=2)
    return np.exp(-D2/(2.0*length*length))

def rbf_kernel_torus(n, length):
    # coordinates (r,c)
    X = np.array([(r, c) for r in range(n) for c in range(n)], dtype=np.int32)
    dr = np.abs(X[:, None, 0] - X[None, :, 0])
    dc = np.abs(X[:, None, 1] - X[None, :, 1])
    dr = np.minimum(dr, n - dr)   # wrap-around on the torus
    dc = np.minimum(dc, n - dc)
    D2 = dr*dr + dc*dc
    return np.exp(-D2 / (2.0 * length * length))

def mse(A,B): 
    # return MSE between normalized A and B ([0, 1] range)
    return np.mean(((A / A.max()) - (B / B.max()))**2)

# -----------------------------------
# helpers for parallelizing replications
# -----------------------------------
from itertools import repeat

def _khat_for_rep(n, num_walks, p_term, sigma, gamma, seed):
    # top-level so it's picklable by spawn
    return kernel_from_signatures(n, num_walks, p_term, sigma, gamma, seed)

# ----------------------------
# parallelized run_curve
# ----------------------------
import os
from concurrent.futures import ProcessPoolExecutor

def run_curve_parallel(ns=(5,7,9,11,13,15,17,19,21,23,25),
                       num_walks=2000, p_term=0.1,
                       sigma=5.0, gamma=1.,
                       seed=0, reps=100,
                       max_workers=None):
    """
    Parallelizes over 'reps' for each n.
    Computes K_true and K_rbf once per n in the parent.
    """
    max_workers = max_workers or (os.cpu_count() or 1)

    # lists of length `reps`, each holding a list over |ns|
    mses_rf  = [[] for _ in range(reps)]
    mses_rbf = [[] for _ in range(reps)]
    mses_true_vs_exact_kernel = [[] for _ in range(reps)]
    Ns = []

    for n in ns:
        print(n)
        # Precompute once per n (deterministic)
        K_true = exact_diffusion_kernel(n, sigma, gamma)
        # K_rbf  = rbf_kernel_grid(n, length=sigma/2)
        K_rbf  = rbf_kernel_torus(n, length=sigma/2)

        seeds = [seed + i for i in range(reps)]

        # Dispatch replications in parallel; map preserves order
        # so i-th result corresponds to seed+i
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # chunksize=1 is fine for moderate reps; tune if reps is very large
            for i, K_hat in enumerate(
                ex.map(_khat_for_rep,
                       repeat(n),
                       repeat(num_walks),
                       repeat(p_term),
                       repeat(sigma),
                       repeat(gamma),
                       seeds,
                       chunksize=1)
            ):
                mses_rf[i].append(mse(K_hat, K_true))
                mses_rbf[i].append(mse(K_hat, K_rbf))
                # this one doesn't depend on the replication; append once
                if len(mses_true_vs_exact_kernel[i]) < len(Ns) + 1:
                    mses_true_vs_exact_kernel[i].append(mse(K_true, K_rbf))

        Ns.append(n*n)

    return Ns, mses_rf, mses_rbf, mses_true_vs_exact_kernel

if __name__ == "__main__":
    Ns, mses_rf, mses_rbf, mses_true_vs_exact_kernel = run_curve_parallel(max_workers=16, reps=128)
    # save mses
    np.savez("parallel_diffusion_kernel_approximation_results.npz",
             Ns=Ns,
             mses_rf=mses_rf,
             mses_rbf=mses_rbf,
             mses_true_vs_exact_kernel=mses_true_vs_exact_kernel)
    # plot shaded errorbar
    plt.figure(figsize=(6,4))
    plt.errorbar(Ns, np.mean(mses_rf, axis=0), yerr=np.std(mses_rf, axis=0), marker="o", label="g-GRF (diffusion) vs exact")
    plt.fill_between(Ns, np.mean(mses_rf, axis=0)-np.std(mses_rf, axis=0), np.mean(mses_rf, axis=0)+np.std(mses_rf, axis=0), alpha=0.2)
    # plt.plot(Ns, np.mean(mses_rf, axis=0), marker="o", label="g-GRF (diffusion) vs exact", errorbar=np.std(mses_rf, axis=0))
    # plt.plot(Ns, np.mean(mses_rbf, axis=0), marker="s", label="RBF vs g-GRF (diffusion)", errorbar=np.std(mses_rbf, axis=0))
    plt.xlabel("number of nodes N"); plt.ylabel("MSE")
    plt.title("Diffusion kernel approximation vs exact"); plt.legend(); plt.tight_layout()
    plt.savefig("parallel_diffusion_kernel_approximation.png", dpi=150)

    # plot shaded errorbar for RBF
    plt.figure(figsize=(6,4))
    plt.errorbar(Ns, np.mean(mses_rbf, axis=0), yerr=np.std(mses_rbf, axis=0), marker="o", label="RBF vs g-GRF (diffusion)")
    plt.fill_between(Ns, np.mean(mses_rbf, axis=0)-np.std(mses_rbf, axis=0), np.mean(mses_rbf, axis=0)+np.std(mses_rbf, axis=0), alpha=0.2)
    plt.xlabel("number of nodes N"); plt.ylabel("MSE")
    plt.title("Gaussian kernel vs diffusion kernel approximation"); plt.legend(); plt.tight_layout()
    plt.savefig("parallel_rbf_vs_diffusion_kernel_approximation.png", dpi=150)