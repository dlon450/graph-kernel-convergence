import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

def standard_gaussian_kernel(n, sigma=0.1, center=(0.5, 0.5)):
    """
    Standard Gaussian kernel at given center and sigma
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    K = np.exp(-((X - center[0])**2 + (Y - center[1])**2) / (2 * sigma**2))
    return K, x, y

def g_nd(n, d, sigma=0.1, center=None, images=1):
    if center is None: center = (0.5,)*d
    axes = [np.linspace(0.0, 1.0, n) for _ in range(d)]
    grids = np.meshgrid(*axes, indexing='ij')
    S = np.zeros((n,)*d)
    rng = range(-images, images+1)
    for sh in product(rng, repeat=d):
        r2 = 0
        for i, G in enumerate(grids): r2 += (G - center[i] - sh[i])**2
        S += np.exp(-r2/(sigma**2))
    return S / ((np.pi*sigma**2)**(d/2)), axes

def plot_values(values, save_fn=None):
    """
    Plot 2D array of values
    """
    plt.imshow(values, origin='lower')
    plt.clim(0, None)
    plt.colorbar() 
    if save_fn is not None:
        plt.savefig(save_fn)
    plt.show()

def plot_multiple_values(value_list, xticks, yticks, titles=None, save_fn=None):
    """
    Plot multiple 2D arrays side by side
    """
    n = len(value_list)
    plt.figure(figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, values in enumerate(value_list):
        plt.subplot(1, n, i + 1)
        plt.imshow(values, origin='lower')
        if titles is not None:
            plt.title(titles[i])
        plt.clim(0, None)
        plt.colorbar(shrink=0.8)
        plt.xticks(ticks=np.linspace(0, len(xticks)-1, min(5, len(xticks))), labels=np.round(np.linspace(xticks[0], xticks[-1], min(5, len(xticks))), 2))
        plt.yticks(ticks=np.linspace(0, len(yticks)-1, min(5, len(yticks))), labels=np.round(np.linspace(yticks[0], yticks[-1], min(5, len(yticks))), 2))
    if save_fn is not None:
        plt.savefig(save_fn)
    else:
        plt.show()
    plt.clf()

def neighbors(r, c, n):
    return [((r-1) % n, c % n), ((r+1) % n, c % n), (r % n, (c-1) % n), (r % n, (c+1) % n)]

def _simulate_walks_chunk(n, rr, cc, chunk_walks, p_term, sigma, seed):
    """
    Worker: simulate `chunk_walks` random walks starting at (rr, cc),
    and return a (n, n) array with accumulated contributions.
    """
    rng = np.random.default_rng(seed)
    v_local = np.zeros((n, n), dtype=np.float64)
    s = (sigma * sigma) / 2. # same as your code
    beta = 2. * n * n * s  # resolution-aware scale β(n)=n^2/2

    for _ in range(chunk_walks):
        r, c = rr, cc
        load = 1.0
        fk = np.exp(-beta)
        wsurv = 1.0
        k = 0

        while True:
            # contribute k-th term
            v_local[r, c] += (load * fk * wsurv)

            # geometric termination before attempting next step
            if rng.random() < p_term:
                break

            # survived: update survival debias
            # wsurv *= 1.0 / (1.0 - p_term)

            # one RW step (unbiased expectation for Markov operator)
            nbrs = neighbors(r, c, n)
            # dv = len(nbrs)
            # wr, wc = nbrs[rng.integers(dv)]
            # dw = len(neighbors(wr, wc, n))  # general graphs; 4 on the torus grid
            # p = 1.0 / dv
            # u = gamma / math.sqrt(dv * dw)
            # load *= (u / p)

            wr, wc = nbrs[rng.integers(len(nbrs))]
            load = load / (1.0 - p_term)

            r, c = wr, wc

            # update factorial weight to s^{k+1}/(k+1)!
            k += 1
            fk *= beta / k

    return v_local

def generate_signature_vector_diffusion_sym(
    n=10, source=(0, 0), num_walks=400,
    p_term=0.2, sigma=1.0, seed=0,
    workers=None, chunk_size=None, show_progress=True
):
    """
    Parallel version (drop-in compatible). Set `workers>1` to enable multiprocessing.

    Args:
        n, source, num_walks, p_term, gamma, sigma, seed: same semantics as your original.
        workers: number of processes (default: os.cpu_count()).
        chunk_size: walks per task; default balances overhead vs. latency.
        show_progress: show a tqdm progress bar over chunks.

    Returns:
        Flattened (n*n,) signature vector.
    """
    rr, cc = source
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    # Sequential fast path (identical to original behavior)
    if workers <= 1 or num_walks <= 1:
        v_local = _simulate_walks_chunk(n, rr, cc, num_walks, p_term, sigma, seed)
        return (v_local / num_walks).reshape(-1)

    # Create chunk sizes
    if chunk_size is None:
        # heuristic: ~4 chunks per worker
        chunk_size = max(1, num_walks // (workers * 4) or 1)

    chunk_counts = []
    remaining = num_walks
    while remaining > 0:
        take = min(chunk_size, remaining)
        chunk_counts.append(take)
        remaining -= take

    # Derive a reproducible seed for each chunk
    ss = np.random.SeedSequence(seed)
    child_ss = ss.spawn(len(chunk_counts))
    child_seeds = [int(cs.generate_state(1, dtype=np.uint64)[0]) for cs in child_ss]

    v_total = np.zeros((n, n), dtype=np.float64)

    # Submit tasks
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _simulate_walks_chunk, n, rr, cc, walks, p_term, sigma, sd
            )
            for walks, sd in zip(chunk_counts, child_seeds)
        ]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures),
                            desc=f"Generating signature vector N={n} (parallel)",
                            leave=False)

        for fut in iterator:
            v_total += fut.result()

    return (v_total / num_walks).reshape(-1)

def mse(a, b):
    return np.mean((a - b)**2)

def constant(d, sigma, n):
    return (2 * np.pi * sigma**2)**(d / 4.) *  (n**(d / 2.))

if __name__ == "__main__":
    Ns = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]
    p_terms = [0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001]
    # Ns = [5, 15]
    # p_terms = [0.005, 0.005]
    results_dict = {}
    sigma = 0.2
    center = (.5, .5)
    for n, p_term in zip(Ns, p_terms):

        K, xticks, yticks = standard_gaussian_kernel(n, sigma, center)
        # plot_values(K)

        sv = generate_signature_vector_diffusion_sym(
            n=n, source=(n//2, n//2), num_walks=1000000,
            p_term=p_term, sigma=sigma, seed=346511053,
            workers=16, show_progress=True
        ).reshape(n, n) * constant(2, sigma, n)
        # plot_values(sv)

        # sv_at = lambda r,c: np.roll(sv, (r - n//2, c - n//2), axis=(0,1))
        # inner = np.array([(sv_at(r,c) * sv).sum() for r in range(n) for c in range(n)]).reshape(n,n)
        inner = np.fft.ifft2(np.abs(np.fft.fft2(sv))**2).real
        inner = np.roll(inner, (-n//2, -n//2), axis=(0,1))
        # inner /= inner[n//2,n//2]
        # plot_values(inner)

        G, _ = g_nd(n, 2, sigma, center)

        plot_multiple_values(
            [K, sv, G * constant(2, sigma, n) / (n**2), inner],
            xticks=xticks,
            yticks=yticks,
            titles=["Gaussian Kernel", "Signature Vector", "Continuous g", "Inner Products"],
            save_fn=f"diffusion_sym_n{n}_p{p_term}.png"
        )
        results_dict[(n, p_term)] = (mse(K, inner), mse(G * constant(2, sigma, n) / (n**2), sv))
    print("MSE Results:")
    for k, v in results_dict.items():
        print(f"N={k[0]}, p_term={k[1]}: MSE_K_inner={v[0]}, MSE_g_sv={v[1]}")
    pd.DataFrame.from_dict(results_dict, orient='index', columns=['MSE']).to_csv("diffusion_sym_results.csv")
    # print("Plot MSEs:")
    # df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['MSE_K_inner', 'MSE_g_sv'])
    # print(df)
    # plt.figure(figsize=(8,6))
    # plt.plot(Ns, df['MSE_K_inner'], marker='o', label='MSE Kernel vs Signature Vector')
    # plt.plot(Ns, df['MSE_g_sv'], marker='o', label='MSE Continuous g vs Signature Vector')