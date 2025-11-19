import numpy as np
from itertools import product
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def constant(d, sigma, n):
    return (2*np.pi*sigma**2)**(d/4.) * (n**(d/2.))

def _simulate_walks_chunk_nd(n, d, start, chunk_walks, p_term, sigma, seed):
    rng = np.random.default_rng(seed)
    v_local = np.zeros((n,)*d, dtype=np.float64)
    s = sigma*sigma/2.0
    beta = d * n * n * s
    for _ in range(chunk_walks):
        pos = list(start)
        load = 1.0
        fk = np.exp(-beta)
        while True:
            v_local[tuple(pos)] += load * fk
            if rng.random() < p_term: break
            ax = rng.integers(d); sg = 1 if rng.random()<0.5 else -1
            pos[ax] = (pos[ax] + sg) % n
            load /= (1.0 - p_term)
            fk *= beta / (load * 0 + (1 if True else 1))  # fk *= beta/k via loop count
            # minimal k-tracking without extra vars:
            # rewrite: maintain fk as e^{-beta} * beta^k/k! by multiplying beta/k each step
            # here we keep a running integer-free update using a counterless trick:
        # the above inline is intentionally minimal; if you prefer explicit k:
    return v_local

def _simulate_walks_chunk_nd_k(n, d, start, chunk_walks, p_term, sigma, seed):
    rng = np.random.default_rng(seed)
    v_local = np.zeros((n,)*d, dtype=np.float64)
    s = sigma*sigma/2.0
    beta = d * n * n * s
    for _ in range(chunk_walks):
        pos = list(start)
        load = 1.0
        fk = np.exp(-beta)
        k = 0
        while True:
            v_local[tuple(pos)] += load * fk
            if rng.random() < p_term: break
            ax = rng.integers(d); sg = 1 if rng.random()<0.5 else -1
            pos[ax] = (pos[ax] + sg) % n
            load /= (1.0 - p_term)
            k += 1
            fk *= beta / k
    return v_local

def generate_signature_vector_diffusion_sym_nd(n=51, d=2, source=None, num_walks=1_000_000,
                                               p_term=0.01, sigma=0.2, seed=0,
                                               workers=None, chunk_size=None, show_progress=False):
    if source is None: source = tuple([n//2]*d)
    if workers is None: workers = max(1, os.cpu_count() or 1)
    if workers <= 1 or num_walks <= 1:
        v_local = _simulate_walks_chunk_nd_k(n, d, source, num_walks, p_term, sigma, seed)
        return (v_local / num_walks).reshape(-1)
    if chunk_size is None: chunk_size = max(1, num_walks // (workers * 4) or 1)
    counts, rem = [], num_walks
    while rem>0:
        take = min(chunk_size, rem); counts.append(take); rem -= take
    ss = np.random.SeedSequence(seed)
    seeds = [int(cs.generate_state(1, dtype=np.uint64)[0]) for cs in ss.spawn(len(counts))]
    v_total = np.zeros((n,)*d, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_simulate_walks_chunk_nd_k, n, d, source, w, p_term, sigma, s)
                for w, s in zip(counts, seeds)]
        iterator = as_completed(futs)
        iterator = tqdm(iterator, total=len(futs),
                desc=f"Generating signature vector N={n} (parallel)",
                leave=False)
        for f in iterator:
            v_total += f.result()
    return (v_total / num_walks).reshape(-1)

def inner_autocorr_nd(sv):
    f = np.fft.fftn(sv)
    inner = np.fft.ifftn(np.abs(f)**2).real
    n = sv.shape[0]; sh = tuple([-n//2]*sv.ndim)
    return np.roll(inner, sh, axis=tuple(range(sv.ndim)))

def standard_gaussian_kernel_nd(n, d, sigma=0.1, center=None):
    if center is None: center = (0.5,)*d
    axes = [np.linspace(0.0, 1.0, n) for _ in range(d)]
    grids = np.meshgrid(*axes, indexing='ij')
    r2 = 0
    for i, G in enumerate(grids): r2 += (G - center[i])**2
    return np.exp(-r2/(2*sigma**2)), axes

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

def run_experiment_nd(Ns, p_terms, d, sigma, sv_func, num_walks=1_000_000, pfix=None, seed=0, workers=1, center=None, csv_fn=None):
    if center is None: center = (0.5,)*d
    mse = lambda a,b: np.mean((a-b)**2)
    constant = lambda d,sigma,n: (2*np.pi*sigma**2)**(d/4.0) * (n**(d/2.0))
    res = {}
    for n, p_term in zip(Ns, p_terms):
        K, _ = standard_gaussian_kernel_nd(n, d, sigma, center)
        sv = sv_func(
            n=n, d=d, source=tuple([n//2]*d), num_walks=num_walks, p_term=p_term, sigma=sigma, seed=seed, workers=workers
            ).reshape((n,)*d) * constant(d, sigma, n)
        F = np.fft.fftn(sv); inner = np.fft.ifftn(np.abs(F)**2).real
        inner = np.roll(inner, tuple([-n//2]*d), axis=tuple(range(d)))
        G, _ = g_nd(n, d, sigma, center)
        res[(n, p_term)] = {'mse_K_inner': mse(K, inner), 'mse_g_sv': mse(G * constant(d, sigma, n) / (n**d), sv)}
    df = pd.DataFrame.from_dict(res, orient='index')
    print(df)
    if csv_fn: df.to_csv(csv_fn)
    return df

if __name__ == "__main__":
    Ns = [5, 15, 25]
    p_terms = [0.001, 0.0005, 0.0005]
    D = 6
    sigma = 0.2
    run_experiment_nd(Ns, p_terms, d=D, sigma=sigma,
                      sv_func=generate_signature_vector_diffusion_sym_nd,
                      num_walks=1_000_000, seed=346511053, workers=16)
